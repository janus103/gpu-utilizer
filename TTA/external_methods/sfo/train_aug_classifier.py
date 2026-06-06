#!/usr/bin/env python3
"""AugClassifier + Parallel Attention Training for Domain-Adaptive ViT

Architecture:
    Raw Image ──┬── Aug Classifier ──┬── Parallel Attention ──┐
    Aug Image ──┘       │ L_aug       │                       │
                        ▼             ▼                       ▼
                   Regression    Feature Map           ViT Embedding
                   Header        Expansion             ──⊗──⊕── Backbone → L_Task

Training:
    - Raw Image + Aug Image share the same geometric transforms (crop, flip)
    - Only Aug Image receives photometric augmentation (AugMix-style)
    - Aug Classifier predicts augmentation severity vector (regression)
    - Parallel Attention modulates Embedding features to compensate augmentation
    - L_Task trains Embedding + Parallel Attention so both raw and aug produce correct class
    - L_Augmentation trains Aug Classifier to predict applied augmentation intensities

Test Time:
    - Single image → Aug Classifier → Parallel Attention → Embedding correction → Backbone
"""
import argparse
import copy
import csv
import importlib
import logging
import math
import os
import random
import time
from collections import OrderedDict
from functools import partial
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from PIL import Image
from torchvision import transforms
from torchvision.datasets import ImageFolder

from timm import utils
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from timm.data.transforms import RandomResizedCropAndInterpolation, MaybeToTensor
from timm.data.auto_augment import (
    _AUGMIX_SL_TRANSFORMS_V2, AugMixSLOp, augmix_sl_ops_v2,
    _HPARAMS_DEFAULT, _LEVEL_DENOM,
)
from timm.layers import set_fast_norm
from timm.models import create_model, safe_model_name

_logger = logging.getLogger('train_aug_classifier')

NUM_AUG_TRANSFORMS = len(_AUGMIX_SL_TRANSFORMS_V2)  # 8 photometric transforms


# =============================================================================
# Augmentation: Paired transform with shared geometry, separate photometric
# =============================================================================

# Per-transform safe magnitude caps.
# Prevents degenerate outputs (all-black, full-inversion) while still
# allowing strong augmentation. Values are in normalized [0, 1] space.
#   NegativeIntensity : darkening factor in [0.1, 1.0]
#       → mag=1.0 → factor=0.1 → near-black; cap at 0.8 keeps factor ≥ 0.28
#   SolarizeIncreasing : threshold = 256 - int(mag/10 * 256)
#       → mag=1.0 → threshold=0 → full inversion → cap at 0.7
_SAFE_MAG_CAPS = {
    'NegativeIntensity': 0.8,
    'SolarizeIncreasing': 0.7,
}


class ScaledAugMixSL:
    """AugMix-style photometric augmentation with scale-controlled severity labels.

    Applies 1~max_depth photometric transforms from V2 policy.
    Severity shares are sampled from Dirichlet and scaled so that
    the total severity sum falls in [sum_min, sum_max].

    Returns (augmented_image, label_vector) where label_vector has
    num_transforms+1 entries: index 0 = source indicator, indices 1..N = per-transform severity.
    For augmented images: source=0, transform severities filled.
    For raw images (not called here): source=scale, transforms=0.
    """

    def __init__(
        self,
        scale: float = 1.0,
        max_depth: int = 3,
        sum_min: float = 0.5,
        sum_max: float = 2.5,
        hparams: Optional[Dict] = None,
    ):
        self.scale = scale
        self.max_depth = max_depth
        self.sum_min = sum_min
        self.sum_max = sum_max
        self.ops = augmix_sl_ops_v2(hparams=hparams or _HPARAMS_DEFAULT)
        self.num_ops = len(self.ops)
        self.safe_caps = {
            i: _SAFE_MAG_CAPS[op.name]
            for i, op in enumerate(self.ops)
            if op.name in _SAFE_MAG_CAPS
        }

    def __call__(self, img):
        num_transforms = np.random.randint(1, self.max_depth + 1)
        selected_indices = np.random.choice(self.num_ops, size=num_transforms, replace=False)

        if num_transforms == 1:
            shares = np.array([1.0])
        else:
            shares = np.random.dirichlet(np.ones(num_transforms))

        target_sum = np.random.uniform(self.sum_min, self.sum_max)
        severities = shares * target_sum

        # label: [source, transform_0, ..., transform_6]
        # For aug image: source=0, transforms filled with actual applied severity
        label = np.zeros(self.num_ops + 1, dtype=np.float32)

        for i, idx in enumerate(selected_indices):
            sv = float(severities[i])
            normalized_mag = min(sv / self.scale, 1.0) if self.scale > 0 else 0.0
            cap = self.safe_caps.get(idx, 1.0)
            normalized_mag = min(normalized_mag, cap)
            img = self.ops[idx](img, normalized_mag)
            label[idx + 1] = sv  # offset by 1 for source channel

        return img, label

    def raw_label(self):
        """Label vector for raw (source domain) image."""
        label = np.zeros(self.num_ops + 1, dtype=np.float32)
        label[0] = self.scale  # source indicator at max scale
        return label


class PairedAugDataset(torch.utils.data.Dataset):
    """Dataset that returns (raw_img, aug_img, raw_label, aug_label, class_label).

    Both raw and aug images share the same geometric transform (crop + flip).
    Only aug image receives additional photometric augmentation.
    """

    def __init__(
        self,
        root: str,
        img_size: int = 224,
        scale: float = 1.0,
        aug_max_depth: int = 3,
        aug_sum_min: float = 0.5,
        aug_sum_max: float = 2.5,
        mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        std: Tuple[float, ...] = (0.229, 0.224, 0.225),
        hparams: Optional[Dict] = None,
    ):
        self.dataset = ImageFolder(root)
        self.img_size = img_size

        self.geometric = transforms.Compose([
            RandomResizedCropAndInterpolation(img_size, scale=(0.08, 1.0), ratio=(3./4., 4./3.)),
            transforms.RandomHorizontalFlip(p=0.5),
        ])

        self.photometric = ScaledAugMixSL(
            scale=scale, max_depth=aug_max_depth,
            sum_min=aug_sum_min, sum_max=aug_sum_max, hparams=hparams,
        )

        self.to_tensor_norm = transforms.Compose([
            MaybeToTensor(),
            transforms.Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
        ])

        self.raw_label_cache = self.photometric.raw_label()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        pil_img, class_label = self.dataset[idx]
        pil_img = pil_img.convert('RGB')

        geo_img = self.geometric(pil_img)

        raw_tensor = self.to_tensor_norm(geo_img)

        aug_pil, aug_label = self.photometric(geo_img.copy())
        aug_tensor = self.to_tensor_norm(aug_pil)

        raw_label = self.raw_label_cache.copy()

        return raw_tensor, aug_tensor, raw_label, aug_label, class_label


# =============================================================================
# Aug Classifier: Inverted Depthwise Convolution + Regression Header
# =============================================================================

class InvertedDWBlock(nn.Module):
    """Inverted Depthwise Separable Convolution block.

    PW expand → DW (stride optional) → PW project
    Uses GroupNorm instead of BatchNorm for FedAvg compatibility.
    """

    def __init__(self, in_ch, out_ch, expand_ratio=4, stride=1):
        super().__init__()
        mid_ch = in_ch * expand_ratio
        self.pw_expand = nn.Conv2d(in_ch, mid_ch, 1, bias=False)
        self.gn1 = nn.GroupNorm(min(32, mid_ch), mid_ch)
        self.dw = nn.Conv2d(mid_ch, mid_ch, 3, stride=stride, padding=1, groups=mid_ch, bias=False)
        self.gn2 = nn.GroupNorm(min(32, mid_ch), mid_ch)
        self.pw_project = nn.Conv2d(mid_ch, out_ch, 1, bias=False)
        self.gn3 = nn.GroupNorm(min(32, out_ch), out_ch)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.act(self.gn1(self.pw_expand(x)))
        x = self.act(self.gn2(self.dw(x)))
        x = self.gn3(self.pw_project(x))
        return x


class AugClassifier(nn.Module):
    """Aug Classifier: predicts which augmentation type was applied.

    Architecture:
        Input (3, 224, 224)
        → InvertedDWBlock(3→32, stride=2)    → (32, 112, 112)
        → InvertedDWBlock(32→64, stride=2)   → (64, 56, 56)
        → InvertedDWBlock(64→128, stride=2)  → (128, 28, 28)
        → InvertedDWBlock(128→256, stride=2) → (256, 14, 14)
        → 1x1 Conv projection                → (embed_dim, 14, 14)
        ──── feature_map branch → Parallel Attention
        → Classification Header:
            → AdaptiveAvgPool → (embed_dim, 1, 1)
            → FC → num_classes (= NUM_AUG_TRANSFORMS = 8)

    Args:
        in_chans: Input image channels (default: 3).
        num_classes: Number of augmentation types to classify.
        embed_dim: ViT embedding dimension for projection alignment.
    """

    def __init__(self, in_chans=3, num_classes=NUM_AUG_TRANSFORMS, embed_dim=768):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim

        self.stage1 = InvertedDWBlock(in_chans, 32, expand_ratio=2, stride=2)
        self.stage2 = InvertedDWBlock(32, 64, expand_ratio=4, stride=2)
        self.stage3 = InvertedDWBlock(64, 128, expand_ratio=4, stride=2)
        self.stage4 = InvertedDWBlock(128, 256, expand_ratio=4, stride=2)

        self.act = nn.GELU()

        self.proj = nn.Sequential(
            nn.Conv2d(256, embed_dim, 1, bias=False),
            nn.GroupNorm(min(32, embed_dim), embed_dim),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, num_classes),
        )

        self._feature_map = None
        self._stage1_feature = None
        self._stage4_feature = None

    def forward(self, x):
        x = self.act(self.stage1(x))          # (B, 32, 112, 112)
        self._stage1_feature = x

        x = self.act(self.stage2(x))
        x = self.act(self.stage3(x))
        x = self.act(self.stage4(x))          # (B, 256, 14, 14)
        self._stage4_feature = x
        x = self.act(self.proj(x))            # (B, embed_dim, 14, 14)

        self._feature_map = x

        logits = self.head(self.pool(x))       # (B, num_classes)
        return logits

    def get_feature_map(self):
        """Returns (B, embed_dim, 14, 14) for Parallel Attention."""
        return self._feature_map

    def get_stage1_feature(self):
        """Returns (B, 32, 112, 112) shallow feature from stage1."""
        return self._stage1_feature

    def get_stage4_feature(self):
        """Returns (B, 256, 14, 14) feature from stage4."""
        return self._stage4_feature


# =============================================================================
# Parallel Attention: Inverted DWConv expansion to match Embedding shape
# =============================================================================

class ParallelAttentionConv(nn.Module):
    """Parallel Attention via Inverted DWConv on AugClassifier features.

    Input:  (B, embed_dim, 14, 14) from AugClassifier projected feature map
    Output: (B, embed_dim, 14, 14) attention mask in [0, 1]

    Architecture:
        InvertedDWBlock(embed_dim → embed_dim, stride=1)
        → InvertedDWBlock(embed_dim → embed_dim, stride=1)
        → Sigmoid
    """

    def __init__(self, in_ch=768, embed_dim=768):
        super().__init__()
        self.block1 = InvertedDWBlock(in_ch, embed_dim, expand_ratio=2, stride=1)
        self.block2 = InvertedDWBlock(embed_dim, embed_dim, expand_ratio=2, stride=1)
        self.act = nn.GELU()
        self.gate = nn.Sigmoid()

    def forward(self, x):
        """
        Args:
            x: (B, embed_dim, 14, 14) feature map from AugClassifier.
        Returns:
            (B, embed_dim, 14, 14) attention mask in [0, 1].
        """
        x = self.act(self.block1(x))
        x = self.gate(self.block2(x))
        return x


# =============================================================================
# Unified wrapper: single nn.Module for DataParallel compatibility
# =============================================================================

class AugAttentionWrapper(nn.Module):
    """Wraps ViT backbone + AugClassifier + ParallelAttention into a single Module.

    This allows nn.DataParallel to split the batch across GPUs correctly,
    since the entire forward (aug_classifier → parallel_attn → ViT backbone)
    runs inside one Module.forward().

    Forward returns (class_logits, aug_logits).
    """

    def __init__(self, backbone, aug_classifier, parallel_attn):
        super().__init__()
        self.backbone = backbone
        self.aug_classifier = aug_classifier
        self.parallel_attn = parallel_attn

    def forward(self, x):
        bb = self.backbone

        aug_logits = self.aug_classifier(x)
        feat_map = self.aug_classifier.get_feature_map()  # (B, embed_dim, 14, 14)

        attn_mask = self.parallel_attn(feat_map)           # (B, embed_dim, 14, 14)

        embedding_feat = bb.patch_embed.proj(x)            # (B, embed_dim, 14, 14)

        corrected = embedding_feat + embedding_feat * attn_mask

        x = corrected.flatten(2).transpose(1, 2)  # (B, N, embed_dim)
        x = bb.patch_embed.norm(x)

        x = bb._pos_embed(x)
        x = bb.patch_drop(x)
        x = bb.norm_pre(x)

        for blk in bb.blocks:
            x = blk(x)

        x = bb.norm(x)
        class_logits = bb.forward_head(x)

        return class_logits, aug_logits


# =============================================================================
# Helper utilities
# =============================================================================

def _get_base_model(model):
    """Unwrap DataParallel / compiled model to the raw module."""
    m = model
    if hasattr(m, 'module'):
        m = m.module
    if hasattr(m, '_orig_mod'):
        m = m._orig_mod
    return m


def _unwrap_wrapper(model):
    """Get AugAttentionWrapper from possible DataParallel wrapping."""
    if isinstance(model, nn.DataParallel):
        return model.module
    return model


def _is_embedding_param(param_name: str) -> bool:
    name = param_name.lower()
    return any(kw in name for kw in (
        'backbone.patch_embed', 'backbone.pos_embed',
        'backbone.cls_token', 'backbone.reg_token', 'backbone.norm_pre',
    ))


def _is_backbone_only_param(param_name: str) -> bool:
    """True for backbone params that are NOT embedding."""
    return param_name.startswith('backbone.') and not _is_embedding_param(param_name)


def _apply_train_mode(wrapper, train_mode):
    """Set requires_grad on the unified AugAttentionWrapper.

    Mode 0: all params trainable
    Mode 1: aug_classifier + parallel_attn + backbone embedding trainable
    Mode 2: aug_classifier + parallel_attn only
    """
    w = _unwrap_wrapper(wrapper)

    if train_mode == 0:
        for param in w.parameters():
            param.requires_grad = True
    elif train_mode == 1:
        for param in w.parameters():
            param.requires_grad = False
        for name, param in w.named_parameters():
            if not _is_backbone_only_param(name):
                param.requires_grad = True
    elif train_mode == 2:
        for param in w.parameters():
            param.requires_grad = False
        for name, param in w.named_parameters():
            if name.startswith(('aug_classifier.', 'parallel_attn.')):
                param.requires_grad = True
    else:
        raise ValueError(f'Unsupported train_mode: {train_mode}')

    total = sum(p.numel() for p in w.parameters() if p.requires_grad)
    return total


# =============================================================================
# Loss functions
# =============================================================================

def compute_aug_loss(pred, target, max_val):
    """Smooth L1 regression loss for augmentation prediction.

    Both pred and target have shape (B, num_aug_transforms + 1).
    Values are in [0, max_val].
    """
    return F.smooth_l1_loss(pred, target)


# =============================================================================
# Training loop
# =============================================================================

def train_one_epoch(
    wrapper, train_loader, optimizer_task, optimizer_aug,
    args, device, model_dtype, epoch, aug_warmup=False,
):
    """Train one epoch.

    When aug_warmup=True (epoch <= aug_warm_epochs):
        - Only L_aug is computed and backpropagated
        - Only optimizer_aug steps (AugClassifier learns to predict severity)
        - optimizer_task is not stepped (Embedding/PA stay frozen)
    """
    wrapper.train()

    losses_m = utils.AverageMeter()
    task_m = utils.AverageMeter()
    aug_m = utils.AverageMeter()
    top1_raw_m = utils.AverageMeter()
    top1_aug_m = utils.AverageMeter()

    max_val = args.scale * 1.5
    phase = 'AUG-WU' if aug_warmup else 'FULL'

    for batch_idx, (raw_img, aug_img, raw_label, aug_label, class_label) in enumerate(train_loader):
        raw_img = raw_img.to(device=device, dtype=model_dtype)
        aug_img = aug_img.to(device=device, dtype=model_dtype)
        raw_label = raw_label.to(device=device, dtype=model_dtype)
        aug_label = aug_label.to(device=device, dtype=model_dtype)
        class_label = class_label.to(device=device)

        # ── Forward: Raw Image (DataParallel splits batch across GPUs) ──
        raw_cls, raw_aug_pred = wrapper(raw_img)
        raw_aug_loss = compute_aug_loss(raw_aug_pred, raw_label, max_val)

        # ── Forward: Aug Image ──
        aug_cls, aug_aug_pred = wrapper(aug_img)
        aug_aug_loss = compute_aug_loss(aug_aug_pred, aug_label, max_val)

        aug_loss = (raw_aug_loss + aug_aug_loss) / 2.0

        if aug_warmup:
            total_loss = args.lambda_aug * aug_loss
            task_loss_val = 0.0

            optimizer_aug.zero_grad()
            total_loss.backward()
            if args.clip_grad is not None:
                w = _unwrap_wrapper(wrapper)
                nn.utils.clip_grad_norm_(w.aug_classifier.parameters(), args.clip_grad)
            optimizer_aug.step()
        else:
            raw_task_loss = F.cross_entropy(raw_cls, class_label)
            aug_task_loss = F.cross_entropy(aug_cls, class_label)
            task_loss = (raw_task_loss + aug_task_loss) / 2.0
            task_loss_val = task_loss.item()
            total_loss = task_loss + args.lambda_aug * aug_loss

            optimizer_task.zero_grad()
            optimizer_aug.zero_grad()
            total_loss.backward()
            if args.clip_grad is not None:
                w = _unwrap_wrapper(wrapper)
                all_params = [p for p in w.parameters() if p.requires_grad]
                nn.utils.clip_grad_norm_(all_params, args.clip_grad)
            optimizer_task.step()
            optimizer_aug.step()

        # ── Metrics ──
        bs = raw_img.shape[0]
        acc1_raw, = utils.accuracy(raw_cls.detach(), class_label, topk=(1,))
        acc1_aug, = utils.accuracy(aug_cls.detach(), class_label, topk=(1,))
        losses_m.update(total_loss.item(), bs * 2)
        task_m.update(task_loss_val, bs * 2)
        aug_m.update(aug_loss.item(), bs * 2)
        top1_raw_m.update(acc1_raw.item(), bs)
        top1_aug_m.update(acc1_aug.item(), bs)

        if batch_idx % args.log_interval == 0:
            _logger.info(
                f'  [{batch_idx:>4d}/{len(train_loader)}]({phase})  '
                f'loss={losses_m.val:.4f}({losses_m.avg:.4f})  '
                f'task={task_m.val:.4f}  aug={aug_m.val:.4f}  '
                f'raw@1={top1_raw_m.val:.1f}%  aug@1={top1_aug_m.val:.1f}%')

    return losses_m.avg, task_m.avg, aug_m.avg, top1_raw_m.avg, top1_aug_m.avg


# =============================================================================
# Evaluation
# =============================================================================

@torch.no_grad()
def evaluate(wrapper, loader, args, device, model_dtype):
    """Evaluate in test-time mode: single image through full pipeline."""
    wrapper.eval()

    losses_m = utils.AverageMeter()
    top1_m = utils.AverageMeter()
    top5_m = utils.AverageMeter()

    for images, target in loader:
        images = images.to(device=device, dtype=model_dtype)
        target = target.to(device=device)

        class_logits, _ = wrapper(images)

        loss = F.cross_entropy(class_logits, target)
        acc1, acc5 = utils.accuracy(class_logits, target, topk=(1, 5))
        losses_m.update(loss.item(), images.shape[0])
        top1_m.update(acc1.item(), images.shape[0])
        top5_m.update(acc5.item(), images.shape[0])

    return losses_m.avg, top1_m.avg, top5_m.avg


@torch.no_grad()
def evaluate_baseline(model, loader, device, model_dtype):
    """Evaluate vanilla model without AugClassifier (baseline)."""
    model.eval()
    top1_m = utils.AverageMeter()
    top5_m = utils.AverageMeter()

    for images, target in loader:
        images = images.to(device=device, dtype=model_dtype)
        target = target.to(device=device)
        output = model(images)
        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        top1_m.update(acc1.item(), images.shape[0])
        top5_m.update(acc5.item(), images.shape[0])

    return top1_m.avg, top5_m.avg


# =============================================================================
# Data loader helpers
# =============================================================================

def create_val_loader(args, data_config, model_dtype, device):
    from timm.data import create_dataset, create_loader
    val_dir = args.clean_data_dir
    input_img_mode = 'RGB' if data_config['input_size'][0] == 3 else 'L'

    dataset = create_dataset(
        '', root=val_dir, split=args.val_split, is_training=False,
        class_map=args.class_map, download=False,
        batch_size=args.batch_size, input_img_mode=input_img_mode,
    )
    loader = create_loader(
        dataset, input_size=data_config['input_size'],
        batch_size=args.batch_size, is_training=False,
        interpolation=data_config['interpolation'],
        num_workers=args.workers, crop_pct=data_config['crop_pct'],
        mean=data_config['mean'], std=data_config['std'],
        pin_memory=args.pin_mem,
        img_dtype=model_dtype or torch.float32,
        device=device, distributed=False,
        use_prefetcher=not args.no_prefetcher,
    )
    return loader


def cosine_lr(base_lr, min_lr, current_epoch, total_epochs):
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * current_epoch / total_epochs))


# =============================================================================
# Argument parsing
# =============================================================================

config_parser = parser = argparse.ArgumentParser(description='Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE')

parser = argparse.ArgumentParser(description='AugClassifier + Parallel Attention Training')

group = parser.add_argument_group('Data')
group.add_argument('--clean-data-dir', type=str, default='/data/imagenet/imagenet',
                   help='Clean ImageNet root (contains train/ and val/ splits)')
group.add_argument('--clean-split', type=str, default='train',
                   help='Split of clean ImageNet for training')
group.add_argument('--val-split', type=str, default='val',
                   help='Split of clean ImageNet for validation')
group.add_argument('--class-map', default='', type=str)

group = parser.add_argument_group('Model')
group.add_argument('--model', default='vit_base_patch16_224', type=str)
group.add_argument('--pretrained', action='store_true', default=False,
                   help='Use timm pretrained weights for backbone')
group.add_argument('--resume', type=str, default=None,
                   help='Resume from ViT checkpoint (backbone weights)')
group.add_argument('--resume-aug', type=str, default=None,
                   help='Resume AugClassifier + ParallelAttention from checkpoint')
group.add_argument('--num-classes', type=int, default=1000)
group.add_argument('--input-size', default=[3, 224, 224], nargs=3, type=int)
group.add_argument('--train-mode', type=int, default=1, choices=[0, 1, 2],
                   help='0=all, 1=aug_modules+embedding, 2=aug_modules only')

group = parser.add_argument_group('AugClassifier')
group.add_argument('--scale', type=float, default=1.0,
                   help='Maximum augmentation scale for regression targets')
group.add_argument('--aug-max-depth', type=int, default=3,
                   help='Max number of photometric transforms per image')
group.add_argument('--aug-sum-min', type=float, default=0.5,
                   help='Minimum total severity sum')
group.add_argument('--aug-sum-max', type=float, default=2.5,
                   help='Maximum total severity sum')
group.add_argument('--lambda-aug', type=float, default=1.0,
                   help='Weight for augmentation regression loss')
group.add_argument('--aug-warm-epochs', type=int, default=5,
                   help='Warm-up epochs: only AugClassifier trains (L_aug only), '
                        'Embedding/PA/backbone frozen')

group = parser.add_argument_group('Optimizer')
group.add_argument('--lr', type=float, default=1e-3)
group.add_argument('--lr-aug', type=float, default=1e-3,
                   help='Learning rate for AugClassifier')
group.add_argument('--min-lr', type=float, default=1e-5)
group.add_argument('--weight-decay', type=float, default=1e-4)
group.add_argument('--clip-grad', type=float, default=1.0)

group = parser.add_argument_group('Training')
group.add_argument('--epochs', type=int, default=30)
group.add_argument('-b', '--batch-size', type=int, default=128)
group.add_argument('--device', default='cuda', type=str)
group.add_argument('--seed', type=int, default=42)
group.add_argument('-j', '--workers', type=int, default=8)
group.add_argument('--pin-mem', action='store_true', default=False)
group.add_argument('--no-prefetcher', action='store_true', default=False)
group.add_argument('--amp', action='store_true', default=False)
group.add_argument('--amp-dtype', default='float16', type=str)
group.add_argument('--model-dtype', default=None, type=str)
group.add_argument('--log-interval', type=int, default=50)
group.add_argument("--local_rank", default=0, type=int)
group.add_argument('--device-modules', default=None, type=str, nargs='+')

group = parser.add_argument_group('Output')
group.add_argument('--output-dir', type=str, default=None)

group = parser.add_argument_group('Compatibility (unused)')
group.add_argument('--drop', type=float, default=0.0)
group.add_argument('--drop-path', type=float, default=None)
group.add_argument('--drop-block', type=float, default=None)
group.add_argument('--bn-momentum', type=float, default=None)
group.add_argument('--bn-eps', type=float, default=None)


def _parse_args():
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)
    args = parser.parse_args(remaining)
    return args


# =============================================================================
# Main
# =============================================================================

def main():
    utils.setup_default_logging()
    args = _parse_args()

    if args.output_dir is None:
        args.output_dir = (
            f'./output/aug_classifier_s{args.scale}'
            f'_tm{args.train_mode}_e{args.epochs}_b{args.batch_size}'
        )

    if args.device_modules:
        for module in args.device_modules:
            importlib.import_module(module)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    device = torch.device(args.device)
    utils.random_seed(args.seed, 0)

    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    effective_batch = args.batch_size * num_gpus

    model_dtype = None
    if args.model_dtype:
        model_dtype = getattr(torch, args.model_dtype)

    # ── Create ViT backbone ──
    _logger.info(f'Creating model: {args.model} (pretrained={args.pretrained})')
    backbone = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.num_classes,
        in_chans=args.input_size[0],
    )
    backbone.to(device=device, dtype=model_dtype)

    if args.resume:
        _logger.info(f'Loading backbone checkpoint: {args.resume}')
        ckpt = torch.load(args.resume, map_location='cpu')
        state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
        missing, unexpected = backbone.load_state_dict(state_dict, strict=False)
        if missing:
            _logger.info(f'  Missing keys ({len(missing)}): {missing[:10]}...')
        if unexpected:
            _logger.info(f'  Unexpected keys ({len(unexpected)}): {unexpected[:10]}...')

    embed_dim = backbone.embed_dim
    _logger.info(f'ViT embed_dim: {embed_dim}')

    # ── Create AugClassifier + ParallelAttention ──
    aug_classifier = AugClassifier(
        in_chans=args.input_size[0],
        num_classes=NUM_AUG_TRANSFORMS,
        embed_dim=embed_dim,
    ).to(device=device, dtype=model_dtype)

    parallel_attn = ParallelAttentionConv(
        in_ch=embed_dim,
        embed_dim=embed_dim,
    ).to(device=device, dtype=model_dtype)

    if args.resume_aug:
        _logger.info(f'Loading AugClassifier checkpoint: {args.resume_aug}')
        aug_ckpt = torch.load(args.resume_aug, map_location='cpu')
        aug_classifier.load_state_dict(aug_ckpt['aug_classifier'])
        parallel_attn.load_state_dict(aug_ckpt['parallel_attn'])

    # ── Wrap into single module ──
    wrapper = AugAttentionWrapper(backbone, aug_classifier, parallel_attn)
    wrapper.to(device=device, dtype=model_dtype)

    # ── Apply train mode (before DataParallel) ──
    trainable_count = _apply_train_mode(wrapper, args.train_mode)
    total_params = sum(p.numel() for p in wrapper.parameters())
    _logger.info(f'Trainable: {trainable_count:,}/{total_params:,} (train_mode={args.train_mode})')

    aug_cls_params = sum(p.numel() for p in aug_classifier.parameters())
    pa_params = sum(p.numel() for p in parallel_attn.parameters())
    _logger.info(f'AugClassifier params: {aug_cls_params:,}')
    _logger.info(f'ParallelAttention params: {pa_params:,}')

    # ── Multi-GPU: DataParallel ──
    if num_gpus > 1:
        wrapper = nn.DataParallel(wrapper)
        _logger.info(f'DataParallel enabled: {num_gpus} GPUs, '
                     f'per-GPU batch={args.batch_size}, '
                     f'effective batch={effective_batch}')
    else:
        _logger.info(f'Single GPU training, batch_size={args.batch_size}')

    data_config = resolve_data_config(vars(args) | {'pretrained': args.pretrained}, model=backbone, verbose=True)

    # ── Output dir ──
    os.makedirs(args.output_dir, exist_ok=True)
    summary_path = os.path.join(args.output_dir, 'summary.csv')

    # ── Create training dataset ──
    train_root = os.path.join(args.clean_data_dir, args.clean_split)
    _logger.info(f'Training data: {train_root}')
    train_dataset = PairedAugDataset(
        root=train_root,
        img_size=args.input_size[1],
        scale=args.scale,
        aug_max_depth=args.aug_max_depth,
        aug_sum_min=args.aug_sum_min,
        aug_sum_max=args.aug_sum_max,
        mean=data_config['mean'],
        std=data_config['std'],
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size * num_gpus,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    _logger.info(f'Training dataset: {len(train_dataset)} images, '
                 f'{len(train_loader)} batches '
                 f'(per-GPU bs={args.batch_size}, effective bs={effective_batch})')

    # ── Optimizers: separate for task (embedding+PA) and aug (AugClassifier) ──
    w = _unwrap_wrapper(wrapper)
    task_params = []
    aug_params = []
    for name, param in w.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith('aug_classifier.'):
            aug_params.append(param)
        else:
            task_params.append(param)

    optimizer_task = torch.optim.AdamW(task_params, lr=args.lr, weight_decay=args.weight_decay)
    optimizer_aug = torch.optim.AdamW(aug_params, lr=args.lr_aug, weight_decay=args.weight_decay)

    _logger.info(f'Optimizer task: AdamW lr={args.lr}, {sum(p.numel() for p in task_params):,} params')
    _logger.info(f'Optimizer aug:  AdamW lr={args.lr_aug}, {sum(p.numel() for p in aug_params):,} params')

    # ── Training loop ──
    total_epochs = args.aug_warm_epochs + args.epochs
    _logger.info(f'\nStarting training: {args.aug_warm_epochs} aug-warmup + {args.epochs} full '
                 f'= {total_epochs} total epochs')
    _logger.info(f'  scale={args.scale}, lambda_aug={args.lambda_aug}')
    _logger.info(f'  Aug transforms: {_AUGMIX_SL_TRANSFORMS_V2}')
    _logger.info(f'  Regression targets: {NUM_AUG_TRANSFORMS + 1} '
                 f'(source + {NUM_AUG_TRANSFORMS} transforms), '
                 f'range [0, {args.scale * 1.5}]')

    best_acc1 = 0.0

    for epoch in range(1, total_epochs + 1):
        epoch_start = time.time()
        aug_warmup = epoch <= args.aug_warm_epochs

        if aug_warmup:
            current_lr = 0.0
            current_lr_aug = cosine_lr(
                args.lr_aug, args.min_lr, epoch - 1, args.aug_warm_epochs)
        else:
            full_epoch = epoch - args.aug_warm_epochs
            current_lr = cosine_lr(args.lr, args.min_lr, full_epoch - 1, args.epochs)
            current_lr_aug = cosine_lr(args.lr_aug, args.min_lr, full_epoch - 1, args.epochs)

        for pg in optimizer_task.param_groups:
            pg['lr'] = current_lr
        for pg in optimizer_aug.param_groups:
            pg['lr'] = current_lr_aug

        phase_str = f'AUG-WARMUP {epoch}/{args.aug_warm_epochs}' if aug_warmup else \
            f'FULL {epoch - args.aug_warm_epochs}/{args.epochs}'
        _logger.info(f'\n{"="*60}')
        _logger.info(f'  Epoch {epoch}/{total_epochs} [{phase_str}]  '
                     f'(lr_task={current_lr:.6f}, lr_aug={current_lr_aug:.6f})')
        _logger.info(f'{"="*60}')

        train_loss, train_task, train_aug, train_raw_acc, train_aug_acc = train_one_epoch(
            wrapper, train_loader, optimizer_task, optimizer_aug,
            args, device, model_dtype, epoch, aug_warmup=aug_warmup,
        )

        epoch_time = time.time() - epoch_start
        _logger.info(f'  Epoch {epoch} summary: loss={train_loss:.4f} '
                     f'task={train_task:.4f} aug={train_aug:.4f} '
                     f'raw@1={train_raw_acc:.1f}% aug@1={train_aug_acc:.1f}% '
                     f'({epoch_time:.1f}s)')

        # ── Evaluate on clean validation set ──
        _logger.info(f'\n  Evaluating epoch {epoch} on clean val...')
        val_loader = create_val_loader(args, data_config, model_dtype, device)
        val_loss, val_acc1, val_acc5 = evaluate(
            wrapper, val_loader, args, device, model_dtype,
        )
        _logger.info(f'  Val Acc@1={val_acc1:.3f}%  Acc@5={val_acc5:.3f}%  loss={val_loss:.4f}')

        # ── Write CSV ──
        write_header = (epoch == 1)
        with open(summary_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if write_header:
                header = ['epoch', 'lr', 'train_loss', 'train_task', 'train_aug',
                          'train_raw_acc', 'train_aug_acc', 'val_acc1', 'val_acc5', 'val_loss']
                writer.writerow(header)
            row = [epoch, f'{current_lr:.6f}', f'{train_loss:.4f}', f'{train_task:.4f}',
                   f'{train_aug:.4f}', f'{train_raw_acc:.2f}', f'{train_aug_acc:.2f}',
                   f'{val_acc1:.3f}', f'{val_acc5:.3f}', f'{val_loss:.4f}']
            writer.writerow(row)

        # ── Save checkpoint (always save unwrapped state) ──
        w = _unwrap_wrapper(wrapper)
        ckpt_data = {
            'epoch': epoch,
            'state_dict': w.backbone.state_dict(),
            'aug_classifier': w.aug_classifier.state_dict(),
            'parallel_attn': w.parallel_attn.state_dict(),
            'val_acc1': val_acc1,
            'args': vars(args),
        }

        ckpt_path = os.path.join(args.output_dir, f'epoch_{epoch:03d}.pth')
        torch.save(ckpt_data, ckpt_path)

        if val_acc1 > best_acc1:
            best_acc1 = val_acc1
            best_path = os.path.join(args.output_dir, 'best.pth')
            torch.save(ckpt_data, best_path)
            _logger.info(f'  New best! Val Acc@1: {val_acc1:.3f}% → {best_path}')
        else:
            _logger.info(f'  Checkpoint saved: {ckpt_path}')

    _logger.info(f'\n{"="*60}')
    _logger.info(f'  Training complete. Summary: {summary_path}')
    _logger.info(f'  Best Val Acc@1: {best_acc1:.3f}%')
    _logger.info(f'{"="*60}')


if __name__ == '__main__':
    main()

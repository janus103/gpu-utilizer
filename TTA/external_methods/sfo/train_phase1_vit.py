#!/usr/bin/env python3
"""Phase 1 (ViT): Direct Aug Classifier — Reference-Free, InstanceNorm + Strided DW.

Architecture (no FSC, no external reference, no diff):
    patch_embed.proj → [B, 768, 14, 14]  (no pooling)
        → InstanceNorm2d(768, affine=True)        content removal
        → DW Conv(3×3, s=2) ×2 stages            14→7→4
        → (normed − clean_ref)                    learnable clean center
        → PW Conv(768→64, 1×1)                   channel reduction + mixing
        → [B, 64, 4, 4] → flatten → MLP          classifier heads

Key features:
    - No FSC centroids, no pseudo-labels, no diff computation.
    - InstanceNorm removes content (class identity); affine=True selectively restores
      augmentation-sensitive statistics (mean/var) per channel.
    - clean_ref is warm-started from clean samples, then refined during training.
    - Hyperspherical Manifold Loss (L_cls + L_radius + L_angular + L_dist).
    - Train/test input distribution is identical — no mismatch.
    - DW conv is channel-independent (768 channels at 196 vals/ch for stable IN).
    - PW conv at 4×4 performs channel reduction (768→64), 8× cheaper than at 14×14.

Usage:
    python train_phase1_vit.py --data-dir /path/to/imagenet \\
        --model vit_base_patch16_224 \\
        --initial-checkpoint ./ZOA_WEIGHT/ZOA_vit_base_timm_format.pth \\
        --train-split val --cache-images
"""
import argparse
import gc
import importlib
import logging
import math
import os
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from timm import utils
from timm.data import (
    resolve_data_config,
    augmix_sl_ops_v2,
    get_augmix_sl_num_transforms,
    get_augmix_sl_transform_names,
    AUGMIX_SL_V2_NUM_TRANSFORMS,
)
from timm.models import create_model

try:
    import wandb
    has_wandb = True
except ImportError:
    has_wandb = False

_logger = logging.getLogger('train_phase1_vit')


# =============================================================================
# Stem Feature Extractor — ViT patch_embed.proj only, NO pooling
# =============================================================================

class StemFeatureExtractor(nn.Module):
    """Extract raw patch_embed projection features without any pooling.

    Output: ``[B, 768, 14, 14]`` for ViT-B/16 with 224×224 input.
    """

    def __init__(self, model):
        super().__init__()
        if not hasattr(model, 'patch_embed'):
            raise ValueError("Model does not have patch_embed layer")
        pe = model.patch_embed
        if not hasattr(pe, 'proj'):
            raise ValueError("patch_embed does not have proj (Conv2d) layer")
        self.proj = pe.proj  # Conv2d(3, embed_dim, patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x)  # [B, 768, 14, 14]


# =============================================================================
# Direct Augmentation Classifier — InstanceNorm + Strided DW + PW
# =============================================================================

class DirectAugClassifier(nn.Module):
    """Reference-free augmentation classifier.

    No FSC centroids, no external diff, no self-reference needed.

    Architecture:
        stem features → InstanceNorm → DW stride stages → (normed − clean_ref) → PW → MLP

    For ViT: DW operates on all 768 channels (channel-independent, cheap).
    PW Conv performs channel reduction 768→64 at 4×4 spatial (12× cheaper than at 14×14).

    Args:
        stem_channels: Channels from stem (768 for ViT-B/16 patch_embed).
        output_channels: Channels after PW conv (64).
        num_dw_stages: Number of stride-2 DW stages (2 for 14→4).
        num_transforms: Number of augmentation types (7 for V2).
        hidden_dims: MLP hidden dimensions.
        dropout: Dropout rate.
    """

    def __init__(
        self,
        stem_channels: int = 768,
        output_channels: int = 64,
        num_dw_stages: int = 2,
        num_transforms: int = AUGMIX_SL_V2_NUM_TRANSFORMS,
        hidden_dims: list = None,
        dropout: float = 0.1,
        dw_init_mode: str = 'fan_in',
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]

        self.stem_channels = stem_channels
        self.output_channels = output_channels
        self.num_dw_stages = num_dw_stages
        self.num_transforms = num_transforms
        self._dw_init_mode = dw_init_mode
        self.num_classes = num_transforms + 1   # +1 for clean (class 0)
        self.feature_dim = output_channels * 4 * 4  # always 4×4 after DW

        # --- InstanceNorm: content (class identity) removal ---
        # 768ch × 196 vals/ch at 14×14 → very stable statistics
        self.inst_norm = nn.InstanceNorm2d(stem_channels, affine=True)

        # --- DW stride stages: learned spatial reduction ---
        # 768ch DW is channel-independent → params = 768×9 per stage (cheap)
        dw_layers = []
        for _ in range(num_dw_stages):
            dw_layers.extend([
                nn.Conv2d(
                    stem_channels, stem_channels, 3,
                    stride=2, padding=1, groups=stem_channels,
                ),
                nn.ReLU(inplace=True),
            ])
        self.dw_stages = nn.Sequential(*dw_layers)

        # --- Learnable clean prototype (warm-started before training) ---
        self.clean_ref = nn.Parameter(torch.zeros(1, stem_channels, 4, 4))

        # --- PW conv: channel reduction 768→64 + mixing ---
        # At 4×4 spatial: 768×64×16 = 786K FLOPs (vs 9.6M at 14×14)
        self.pw_conv = nn.Conv2d(stem_channels, output_channels, 1)

        # --- Per-augmentation learnable radius ---
        self.log_r = nn.Parameter(torch.full((num_transforms,), 3.0))

        # --- Shared MLP encoder: input = [z_flat ‖ ‖z‖] ---
        encoder_input_dim = self.feature_dim + 1
        layers = []
        in_dim = encoder_input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim
        self.shared_encoder = nn.Sequential(*layers)

        # --- Aug type head (8-class) ---
        self.aug_head = nn.Linear(hidden_dims[-1], self.num_classes)

        # --- Distance / corruption magnitude head ---
        self.dist_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], 1),
            nn.Softplus(),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.InstanceNorm2d):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                if m.groups == m.in_channels and m.groups > 1:
                    # Depthwise conv: configurable init mode
                    nn.init.kaiming_normal_(
                        m.weight, mode=self._dw_init_mode, nonlinearity='relu',
                    )
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @property
    def radius(self):
        """Per-augmentation radius R_i, always positive."""
        return torch.exp(self.log_r)

    def encode(self, features_spatial: torch.Tensor) -> torch.Tensor:
        """Raw stem features → augmentation-sensitive z vector.

        Pipeline: InstanceNorm → DW stride → (normed − clean_ref) → PW → flatten.

        Args:
            features_spatial: ``[B, 768, 14, 14]`` raw patch_embed features.

        Returns:
            z_flat: ``[B, feature_dim]`` projected vector.
        """
        normed = self.inst_norm(features_spatial)       # [B, 768, 14, 14]
        reduced = self.dw_stages(normed)                # [B, 768, 4, 4]
        diff = reduced - self.clean_ref                 # [B, 768, 4, 4]
        z = self.pw_conv(diff)                          # [B, 64, 4, 4]
        return z.flatten(1)                             # [B, 1024]

    def forward(self, z_flat: torch.Tensor):
        """Classifier heads.

        Args:
            z_flat: ``[B, feature_dim]`` from encode().

        Returns:
            aug_output: ``[B, num_classes]`` logits.
            dist_output: ``[B]`` corruption score.
        """
        dist = z_flat.norm(dim=1, keepdim=True)         # [B, 1]
        x = torch.cat([z_flat, dist], dim=1)            # [B, feat_dim+1]
        shared = self.shared_encoder(x)
        aug_output = self.aug_head(shared)
        dist_output = self.dist_head(shared).squeeze(-1)
        return aug_output, dist_output

    @torch.no_grad()
    def warmstart_clean_ref(
        self,
        stem_extractor: nn.Module,
        loader,
        device: torch.device,
        num_samples: int = 500,
    ):
        """Initialize clean_ref with mean of DW-processed InstanceNorm clean features.

        Must be called **before** ``calibrate_radius``.
        """
        self.eval()
        features_sum = torch.zeros_like(self.clean_ref.data)
        count = 0
        for clean_imgs, aug_imgs, labels in loader:
            if count >= num_samples:
                break
            f = stem_extractor(clean_imgs.to(device))   # [B, 768, 14, 14]
            normed = self.inst_norm(f)
            reduced = self.dw_stages(normed)            # [B, 768, 4, 4]
            features_sum += reduced.sum(dim=0, keepdim=True)
            count += clean_imgs.size(0)

        self.clean_ref.data.copy_(features_sum / count)
        _logger.info(
            f'Warm-started clean_ref from {count} clean samples '
            f'(norm={self.clean_ref.data.flatten().norm().item():.4f})'
        )

    @torch.no_grad()
    def calibrate_radius(
        self,
        stem_extractor: nn.Module,
        loader,
        device: torch.device,
        num_batches: int = 5,
    ):
        """Set initial ``log_r`` so ``R_i ≈ mean(‖z_i‖)`` for each aug type.

        Must be called **after** ``warmstart_clean_ref``.
        """
        self.eval()
        num_t = self.num_transforms
        sample_norms = [[] for _ in range(num_t)]

        for batch_idx, (clean_imgs, aug_imgs, labels) in enumerate(loader):
            if batch_idx >= num_batches:
                break
            for t in range(num_t):
                f_aug = stem_extractor(aug_imgs[:, t].to(device))
                z = self.encode(f_aug)
                sample_norms[t].append(z.norm(dim=1).cpu())

        for t in range(num_t):
            norms = torch.cat(sample_norms[t])
            mean_norm = norms.mean().item()
            self.log_r.data[t] = math.log(max(mean_norm, 1e-3))

        _logger.info(
            f'Calibrated R: [{", ".join(f"{r:.1f}" for r in self.radius.tolist())}]'
        )


# =============================================================================
# Hyperspherical Manifold Loss
# =============================================================================

class HypersphericalManifoldLoss(nn.Module):
    r"""Combined loss enforcing Hyperspherical structure.

    .. math::
        J = L_{cls} + \lambda_r \sum_i (||z_i|| - R_{t_i})^2
          + \lambda_a \sum_{i<j} \exp(\hat{z}_i \cdot \hat{z}_j)
          + \lambda_d \, \text{SmoothL1}(\hat{d}, ||z||)
    """

    def __init__(
        self,
        lambda_radius: float = 1.0,
        lambda_angular: float = 0.1,
        lambda_dist: float = 0.5,
    ):
        super().__init__()
        self.lambda_radius = lambda_radius
        self.lambda_angular = lambda_angular
        self.lambda_dist = lambda_dist
        self.ce = nn.CrossEntropyLoss()

    def forward(
        self,
        aug_outputs: torch.Tensor,
        class_labels: torch.Tensor,
        dist_preds: torch.Tensor,
        z_flat_all: torch.Tensor,
        z_flat_augs: torch.Tensor,
        aug_type_indices: torch.Tensor,
        radius: torch.Tensor,
    ) -> tuple:
        device = aug_outputs.device

        # (1) Classification loss
        L_cls = self.ce(aug_outputs, class_labels)

        # (2) Radius constraint: (||z_i|| − R_{type(i)})² for augmented only
        z_norms = z_flat_augs.norm(dim=1)
        target_radii = radius[aug_type_indices]
        L_radius = ((z_norms - target_radii) ** 2).mean()

        # (3) Angular repulsion: exp(cos_sim) between per-type mean directions
        L_angular = torch.tensor(0.0, device=device)
        num_t = radius.size(0)
        mean_dirs = []
        for t in range(num_t):
            mask = aug_type_indices == t
            if mask.any():
                vecs = z_flat_augs[mask]
                mean_dir = F.normalize(vecs.mean(dim=0, keepdim=True), dim=1, eps=1e-8)
                mean_dirs.append(mean_dir)
        if len(mean_dirs) > 1:
            dirs = torch.cat(mean_dirs, dim=0)
            sim = dirs @ dirs.T
            upper = torch.triu(torch.ones_like(sim, dtype=torch.bool), diagonal=1)
            L_angular = torch.exp(sim[upper]).mean()

        # (4) Distance prediction loss
        with torch.no_grad():
            dist_target = z_flat_all.norm(dim=1)
        L_dist = F.smooth_l1_loss(dist_preds, dist_target)

        L_total = (
            L_cls
            + self.lambda_radius * L_radius
            + self.lambda_angular * L_angular
            + self.lambda_dist * L_dist
        )
        info = {
            'L_total': L_total.item(),
            'L_cls': L_cls.item(),
            'L_radius': L_radius.item(),
            'L_angular': L_angular.item(),
            'L_dist': L_dist.item(),
            'R_mean': radius.detach().mean().item(),
            'R_std': radius.detach().std().item(),
        }
        return L_total, info


# =============================================================================
# Paired Augmentation Dataset
# =============================================================================

class PairedAugDataset(torch.utils.data.Dataset):
    """Returns ``(clean_img, aug_imgs[7], class_label)`` from the SAME image."""

    def __init__(
        self,
        dataset,
        augmix_ops,
        min_sl: float = 0.3,
        max_sl: float = 1.0,
        base_transform=None,
        final_transform=None,
        fixed_sl: float = None,
        cache_images: bool = False,
    ):
        self.dataset = dataset
        self.augmix_ops = augmix_ops
        self.num_ops = len(augmix_ops)
        self.min_sl = min_sl
        self.max_sl = max_sl
        self.base_transform = base_transform
        self.final_transform = final_transform
        self.fixed_sl = fixed_sl

        self._cache = None
        self._labels = None
        if cache_images:
            _logger.info(f'Caching {len(dataset)} images into RAM ...')
            from PIL import Image
            from tqdm import tqdm
            labels = []
            imgs = []
            for i in tqdm(range(len(dataset)), desc='Caching images', ncols=80):
                img_pil, label = dataset[i]
                if base_transform is not None:
                    img_pil = base_transform(img_pil)
                imgs.append(np.asarray(img_pil, dtype=np.uint8).copy())
                labels.append(label)
            self._cache = imgs
            self._labels = labels
            mem_mb = sum(a.nbytes for a in imgs) / 1e6
            _logger.info(f'Cached {len(imgs)} images ({mem_mb:.0f} MB)')

    def __len__(self):
        return len(self.dataset)

    def _get_pil_image(self, idx):
        if self._cache is not None:
            from PIL import Image
            return Image.fromarray(self._cache[idx]), self._labels[idx]
        img_pil, label = self.dataset[idx]
        if self.base_transform is not None:
            img_pil = self.base_transform(img_pil)
        return img_pil, label

    def __getitem__(self, idx):
        img_pil, label = self._get_pil_image(idx)
        clean_img = self.final_transform(img_pil)
        aug_imgs = torch.empty(self.num_ops, *clean_img.shape, dtype=clean_img.dtype)
        for i, op in enumerate(self.augmix_ops):
            sl = (
                self.fixed_sl
                if self.fixed_sl is not None
                else np.random.uniform(self.min_sl, self.max_sl)
            )
            aug_imgs[i] = self.final_transform(op(img_pil, sl))
        return clean_img, aug_imgs, label


# =============================================================================
# Training
# =============================================================================

def train_one_epoch(
    epoch: int,
    stem_extractor: nn.Module,
    aug_classifier: DirectAugClassifier,
    loader,
    optimizer,
    criterion: HypersphericalManifoldLoss,
    args,
    device: torch.device,
) -> OrderedDict:
    """One epoch of Phase-1 training (reference-free)."""
    aug_classifier.train()
    stem_extractor.eval()

    loss_m = utils.AverageMeter()
    cls_m = utils.AverageMeter()
    rad_m = utils.AverageMeter()
    ang_m = utils.AverageMeter()
    dist_m = utils.AverageMeter()
    batch_time_m = utils.AverageMeter()

    num_batches = len(loader)
    end = time.time()

    for batch_idx, (clean_imgs, aug_imgs, labels) in enumerate(loader):
        B = clean_imgs.size(0)
        T = aug_imgs.size(1)  # 7

        # ---- Stem features (frozen, no grad) ----
        with torch.no_grad():
            f_clean = stem_extractor(clean_imgs.to(device))       # [B, 768, 14, 14]
            f_augs_parts = []
            for t in range(T):
                f_augs_parts.append(
                    stem_extractor(aug_imgs[:, t].to(device))
                )
            f_augs_flat = torch.cat(f_augs_parts, dim=0)          # [T*B, 768, 14, 14]
            del f_augs_parts

        # ---- Encode directly (no diff, no reference!) ----
        f_all = torch.cat([f_clean, f_augs_flat], dim=0)          # [(1+T)*B, 768, 14, 14]
        del f_clean, f_augs_flat

        z_all = aug_classifier.encode(f_all)                      # [(1+T)*B, 1024]
        del f_all
        z_augs_cat = z_all[B:]

        aug_out, dist_out = aug_classifier(z_all)

        # ---- Labels ----
        lbl_clean = torch.zeros(B, dtype=torch.long, device=device)
        lbl_augs = torch.cat([
            torch.full((B,), t + 1, dtype=torch.long, device=device)
            for t in range(T)
        ])
        labels_all = torch.cat([lbl_clean, lbl_augs])
        aug_type_idx = torch.cat([
            torch.full((B,), t, dtype=torch.long, device=device)
            for t in range(T)
        ])

        # ---- Loss ----
        loss, info = criterion(
            aug_outputs=aug_out,
            class_labels=labels_all,
            dist_preds=dist_out,
            z_flat_all=z_all,
            z_flat_augs=z_augs_cat,
            aug_type_indices=aug_type_idx,
            radius=aug_classifier.radius,
        )

        optimizer.zero_grad()
        loss.backward()
        if args.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(aug_classifier.parameters(), args.clip_grad)
        optimizer.step()

        # ---- Logging ----
        n = B * (1 + T)
        loss_m.update(info['L_total'], n)
        cls_m.update(info['L_cls'], n)
        rad_m.update(info['L_radius'], n)
        ang_m.update(info['L_angular'], n)
        dist_m.update(info['L_dist'], n)
        batch_time_m.update(time.time() - end)
        end = time.time()

        if batch_idx % args.log_interval == 0 or batch_idx == num_batches - 1:
            lr = optimizer.param_groups[0]['lr']
            _logger.info(
                f'Train: {epoch} [{batch_idx:>4d}/{num_batches}]  '
                f'Loss: {loss_m.val:.4f}({loss_m.avg:.4f})  '
                f'CE: {cls_m.avg:.4f}  Rad: {rad_m.avg:.4f}  '
                f'Ang: {ang_m.avg:.4f}  Dist: {dist_m.avg:.4f}  '
                f'R={info["R_mean"]:.2f}+/-{info["R_std"]:.2f}  '
                f'LR: {lr:.2e}  T: {batch_time_m.val:.2f}s'
            )

    return OrderedDict([
        ('loss', loss_m.avg), ('L_cls', cls_m.avg), ('L_radius', rad_m.avg),
        ('L_angular', ang_m.avg), ('L_dist', dist_m.avg),
    ])


# =============================================================================
# Validation
# =============================================================================

@torch.no_grad()
def validate(
    stem_extractor: nn.Module,
    aug_classifier: DirectAugClassifier,
    val_dataset,
    augmix_ops,
    severity: float,
    base_transform,
    final_transform,
    args,
    device: torch.device,
    transform_names: list,
) -> tuple:
    """Validate 8-class accuracy: clean (class 0) + 7 aug types (classes 1–7)."""
    aug_classifier.eval()
    stem_extractor.eval()

    val_bs = args.validation_batch_size or args.batch_size
    per_cond = {}

    # ---- Condition 0: Clean ----
    clean_dataset = _SimpleTransformDataset(val_dataset, base_transform, final_transform)
    clean_loader = torch.utils.data.DataLoader(
        clean_dataset, batch_size=val_bs, shuffle=False,
        num_workers=min(args.workers, 4), pin_memory=args.pin_mem,
    )
    correct = total = 0
    for imgs, lbls in clean_loader:
        f = stem_extractor(imgs.to(device))
        z = aug_classifier.encode(f)
        aug_out, _ = aug_classifier(z)
        correct += (aug_out.argmax(1) == 0).sum().item()
        total += imgs.size(0)
    per_cond['clean'] = 100.0 * correct / total if total else 0.0
    _logger.info(f'  Val [clean]: {per_cond["clean"]:.2f}%')
    del clean_loader

    # ---- Conditions 1–7: Each augmentation type ----
    for t_idx, (op, t_name) in enumerate(zip(augmix_ops, transform_names)):
        aug_dataset = _SingleAugDataset(val_dataset, op, severity, base_transform, final_transform)
        aug_loader = torch.utils.data.DataLoader(
            aug_dataset, batch_size=val_bs, shuffle=False,
            num_workers=min(args.workers, 4), pin_memory=args.pin_mem,
        )
        correct = total = 0
        for aug_imgs, _ in aug_loader:
            f = stem_extractor(aug_imgs.to(device))
            z = aug_classifier.encode(f)
            aug_out, _ = aug_classifier(z)
            correct += (aug_out.argmax(1) == t_idx + 1).sum().item()
            total += aug_imgs.size(0)
        per_cond[t_name] = 100.0 * correct / total if total else 0.0
        _logger.info(f'  Val [{t_name}] (SL={severity}): {per_cond[t_name]:.2f}%')
        del aug_loader, aug_dataset
        gc.collect()

    mean_acc = sum(per_cond.values()) / len(per_cond)
    _logger.info(f'  Val Mean Acc (8-way): {mean_acc:.2f}%')
    return mean_acc, per_cond


class _SimpleTransformDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, base_transform, final_transform):
        self.dataset = dataset
        self.base_transform = base_transform
        self.final_transform = final_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        if self.base_transform is not None:
            img = self.base_transform(img)
        return self.final_transform(img), label


# =============================================================================
# Eval Analyze — Per-class diagnostics (logits, entropy, correct/incorrect)
# =============================================================================

@torch.no_grad()
def eval_analyze_per_class(
    backbone: nn.Module,
    stem_extractor: nn.Module,
    aug_classifier: DirectAugClassifier,
    val_loader,
    device: torch.device,
    num_imagenet_classes: int = 1000,
    num_aug_classes: int = 8,
) -> dict:
    """Run per-class analysis: logits, entropy, correct/incorrect for both classifiers.

    For clean ImageNet val images:
        - Aug Classifier: expects class 0 (clean); outputs 8-way logits.
        - Backbone: 1000-way ImageNet classification.

    Returns:
        Per-class dict: class_id -> {
            'aug_logits': [8], 'aug_entropy_mean', 'aug_correct', 'aug_incorrect',
            'backbone_logits_mean_max', 'backbone_logits_mean_target',
            'backbone_entropy_mean', 'backbone_correct', 'backbone_incorrect',
            'count',
        }
    """
    backbone.eval()
    stem_extractor.eval()
    aug_classifier.eval()

    # Accumulate per class: logits sum, entropy sum, correct, incorrect, count
    aug_logits_sum = torch.zeros(num_imagenet_classes, num_aug_classes)
    aug_entropy_sum = torch.zeros(num_imagenet_classes)
    aug_correct = torch.zeros(num_imagenet_classes, dtype=torch.long)
    aug_incorrect = torch.zeros(num_imagenet_classes, dtype=torch.long)

    backbone_logits_sum = torch.zeros(num_imagenet_classes, num_imagenet_classes)
    backbone_entropy_sum = torch.zeros(num_imagenet_classes)
    backbone_correct = torch.zeros(num_imagenet_classes, dtype=torch.long)
    backbone_incorrect = torch.zeros(num_imagenet_classes, dtype=torch.long)

    class_count = torch.zeros(num_imagenet_classes, dtype=torch.long)

    from tqdm import tqdm
    for imgs, labels in tqdm(val_loader, desc='Eval analyze', unit='batch'):
        imgs = imgs.to(device)
        labels = labels.to(device)

        # Stem → Aug classifier
        f = stem_extractor(imgs)
        z = aug_classifier.encode(f)
        aug_out, _ = aug_classifier(z)
        aug_probs = F.softmax(aug_out, dim=1)
        aug_pred = aug_out.argmax(1)
        aug_entropy = -(aug_probs * torch.log(aug_probs + 1e-10)).sum(1)

        # Backbone classifier
        backbone_out = backbone(imgs)
        backbone_probs = F.softmax(backbone_out, dim=1)
        backbone_pred = backbone_out.argmax(1)
        backbone_entropy = -(backbone_probs * torch.log(backbone_probs + 1e-10)).sum(1)

        for i in range(imgs.size(0)):
            c = labels[i].item()
            if c >= num_imagenet_classes:
                continue

            class_count[c] += 1
            aug_logits_sum[c] += aug_out[i].cpu()
            aug_entropy_sum[c] += aug_entropy[i].item()
            if aug_pred[i].item() == 0:
                aug_correct[c] += 1
            else:
                aug_incorrect[c] += 1

            backbone_logits_sum[c] += backbone_out[i].cpu()
            backbone_entropy_sum[c] += backbone_entropy[i].item()
            if backbone_pred[i].item() == labels[i].item():
                backbone_correct[c] += 1
            else:
                backbone_incorrect[c] += 1

    # Build per-class results
    results = {}
    for c in range(num_imagenet_classes):
        n = class_count[c].item()
        if n == 0:
            continue
        results[c] = {
            'count': n,
            'aug_logits': (aug_logits_sum[c] / n).tolist(),
            'aug_entropy_mean': (aug_entropy_sum[c] / n).item(),
            'aug_correct': aug_correct[c].item(),
            'aug_incorrect': aug_incorrect[c].item(),
            'backbone_logits_mean': (backbone_logits_sum[c] / n).tolist(),
            'backbone_logits_mean_max': (backbone_logits_sum[c] / n).max().item(),
            'backbone_logits_mean_target': (backbone_logits_sum[c][c] / n).item(),
            'backbone_entropy_mean': (backbone_entropy_sum[c] / n).item(),
            'backbone_correct': backbone_correct[c].item(),
            'backbone_incorrect': backbone_incorrect[c].item(),
        }
    return results


def _write_eval_analyze_results(
    results: dict,
    output_path: Path,
    transform_names: list,
) -> None:
    """Write per-class analysis to TSV."""
    aug_names = ['clean'] + list(transform_names)
    with open(output_path, 'w') as f:
        header = (
            'class_id\tcount\t'
            + '\t'.join(f'aug_logit_{i}' for i in range(len(aug_names)))
            + '\taug_entropy_mean\taug_correct\taug_incorrect\taug_acc\t'
            + 'backbone_logit_mean_max\tbackbone_logit_mean_target\t'
            'backbone_entropy_mean\tbackbone_correct\tbackbone_incorrect\tbackbone_acc'
        )
        f.write(header + '\n')
        for c in sorted(results.keys()):
            r = results[c]
            n = r['count']
            aug_acc = 100.0 * r['aug_correct'] / n if n else 0
            backbone_acc = 100.0 * r['backbone_correct'] / n if n else 0
            aug_logits_str = '\t'.join(f'{x:.4f}' for x in r['aug_logits'])
            row = (
                f"{c}\t{n}\t{aug_logits_str}\t"
                f"{r['aug_entropy_mean']:.4f}\t{r['aug_correct']}\t{r['aug_incorrect']}\t{aug_acc:.2f}\t"
                f"{r['backbone_logits_mean_max']:.4f}\t{r['backbone_logits_mean_target']:.4f}\t"
                f"{r['backbone_entropy_mean']:.4f}\t{r['backbone_correct']}\t"
                f"{r['backbone_incorrect']}\t{backbone_acc:.2f}"
            )
            f.write(row + '\n')
    _logger.info(f'Wrote per-class analysis to {output_path}')


class _SingleAugDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, aug_op, severity, base_transform, final_transform):
        self.dataset = dataset
        self.aug_op = aug_op
        self.severity = severity
        self.base_transform = base_transform
        self.final_transform = final_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        if self.base_transform is not None:
            img = self.base_transform(img)
        img = self.aug_op(img, self.severity)
        return self.final_transform(img), label


# =============================================================================
# Argument Parser
# =============================================================================

config_parser = argparse.ArgumentParser(description='Config', add_help=False)
config_parser.add_argument('-c', '--config', default='', type=str, help='YAML config')

parser = argparse.ArgumentParser(description='Phase 1 (ViT): Direct Aug Classifier')

group = parser.add_argument_group('Dataset')
group.add_argument('--data-dir', type=str, required=True)
group.add_argument('--train-split', default='val')
group.add_argument('--val-split', default='val')

group = parser.add_argument_group('Model')
group.add_argument('--model', default='vit_base_patch16_224', type=str)
group.add_argument('--pretrained', action='store_true')
group.add_argument('--initial-checkpoint', default='', type=str)
group.add_argument('--num-classes', type=int, default=1000)
group.add_argument('--img-size', type=int, default=224)

group = parser.add_argument_group('Classifier')
group.add_argument('--hidden-dims', type=int, nargs='+', default=[512, 256, 128])
group.add_argument('--dropout', type=float, default=0.1)
group.add_argument(
    '--dw-init-mode', type=str, default='fan_in', choices=['fan_in', 'fan_out'],
    help='Kaiming init mode for DW Conv. fan_in is correct per-group fan; '
         'fan_out uses PyTorch default (underestimates by groups×).',
)

group = parser.add_argument_group('Augmentation')
group.add_argument('--min-sl', type=float, default=0.3)
group.add_argument('--max-sl', type=float, default=1.0)
group.add_argument('--val-severity', type=float, default=0.7)

group = parser.add_argument_group('Loss weights')
group.add_argument('--lambda-radius', type=float, default=1.0)
group.add_argument('--lambda-angular', type=float, default=0.1)
group.add_argument('--lambda-dist', type=float, default=0.5)

group = parser.add_argument_group('Optimizer')
group.add_argument('--opt', default='adamw', type=str)
group.add_argument('--lr', type=float, default=1e-3)
group.add_argument('--weight-decay', type=float, default=1e-4)
group.add_argument('--momentum', type=float, default=0.9)
group.add_argument('--clip-grad', type=float, default=1.0)

group = parser.add_argument_group('Scheduler')
group.add_argument('--sched', type=str, default='cosine')
group.add_argument('--epochs', type=int, default=30)
group.add_argument('--warmup-epochs', type=int, default=3)
group.add_argument('--warmup-lr', type=float, default=1e-5)
group.add_argument('--min-lr', type=float, default=1e-6)

group = parser.add_argument_group('Misc')
group.add_argument(
    '-b', '--batch-size', type=int, default=128,
    help='Groups per batch (each = 1 clean + 7 aug). '
         'ViT 14×14 features are compact; 128 works well.',
)
group.add_argument('-vb', '--validation-batch-size', type=int, default=None)
group.add_argument(
    '-j', '--workers', type=int, default=None,
    help='DataLoader workers (default: auto)',
)
group.add_argument('--device', default='cuda', type=str)
group.add_argument('--seed', type=int, default=42)
group.add_argument('--pin-mem', action='store_true')
group.add_argument('--log-interval', type=int, default=50)
group.add_argument('--val-interval', type=int, default=1)
group.add_argument('--output', default='', type=str)
group.add_argument('--experiment', default='', type=str)
group.add_argument('--checkpoint-hist', type=int, default=3)
group.add_argument('--cache-images', action='store_true')
group.add_argument('--device-modules', default=None, type=str, nargs='+')
group.add_argument('--log-wandb', action='store_true')
group.add_argument('--wandb-project', default='phase1-direct-vit', type=str)
group.add_argument(
    '--warmstart-samples', type=int, default=500,
    help='Number of clean samples for clean_ref warm-start.',
)
group.add_argument(
    '--calibrate-batches', type=int, default=5,
    help='Number of batches for radius calibration.',
)

group = parser.add_argument_group('Resume')
group.add_argument(
    '--resume', default='', type=str,
    help='Path to a phase-1 checkpoint (.pth.tar) to resume from.',
)
group.add_argument(
    '--resume-weights-only', action='store_true',
    help='If set, only load aug_classifier weights (warm restart with fresh '
         'optimizer/scheduler). Otherwise load full state (optimizer, scheduler, epoch).',
)

group = parser.add_argument_group('Eval Analyze (per-class diagnostics)')
group.add_argument(
    '--eval-analyze', action='store_true',
    help='Run per-class analysis on ImageNet val: load backbone + aug classifier, '
         'output logits, entropy, correct/incorrect per class. No training.',
)
group.add_argument(
    '--backbone-checkpoint', default='', type=str,
    help='Path to backbone checkpoint for --eval-analyze. If empty, uses --initial-checkpoint.',
)
group.add_argument(
    '--aug-classifier-checkpoint', default='', type=str,
    help='Path to phase-1 aug classifier checkpoint for --eval-analyze.',
)
group.add_argument(
    '--eval-analyze-output', default='', type=str,
    help='Output path for per-class analysis (default: output_dir/eval_analyze_per_class.tsv).',
)


def _parse_args():
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            parser.set_defaults(**yaml.safe_load(f))
    args = parser.parse_args(remaining)
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


def _run_eval_analyze(args, device: torch.device) -> None:
    """Load backbone + aug classifier, run per-class analysis on ImageNet val."""
    aug_ckpt_path = args.aug_classifier_checkpoint
    backbone_ckpt_path = args.backbone_checkpoint or args.initial_checkpoint

    if not aug_ckpt_path:
        raise ValueError('--eval-analyze requires --aug-classifier-checkpoint')
    if not backbone_ckpt_path:
        raise ValueError(
            '--eval-analyze requires --backbone-checkpoint or --initial-checkpoint'
        )

    _logger.info(f'Loading backbone from {backbone_ckpt_path}')
    backbone = create_model(
        args.model,
        pretrained=False,
        num_classes=args.num_classes,
        checkpoint_path=backbone_ckpt_path,
    ).to(device).eval()
    for p in backbone.parameters():
        p.requires_grad = False

    stem_extractor = StemFeatureExtractor(backbone).to(device).eval()
    for p in stem_extractor.parameters():
        p.requires_grad = False

    with torch.no_grad():
        dummy = torch.randn(1, 3, args.img_size, args.img_size, device=device)
        dummy_out = stem_extractor(dummy)
        stem_channels = dummy_out.shape[1]
        stem_h = dummy_out.shape[2]

    num_dw_stages = 0
    h = stem_h
    while h > 4:
        h = (h - 3 + 2) // 2 + 1
        num_dw_stages += 1

    num_transforms = get_augmix_sl_num_transforms(version=2)
    transform_names = get_augmix_sl_transform_names(version=2)

    aug_classifier = DirectAugClassifier(
        stem_channels=stem_channels,
        output_channels=64,
        num_dw_stages=num_dw_stages,
        num_transforms=num_transforms,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
        dw_init_mode=args.dw_init_mode,
    ).to(device)

    _logger.info(f'Loading aug classifier from {aug_ckpt_path}')
    aug_ckpt = torch.load(aug_ckpt_path, map_location=device)
    aug_classifier.load_state_dict(aug_ckpt['aug_classifier'])
    aug_classifier.eval()

    data_config = resolve_data_config(vars(args), model=backbone, verbose=True)
    from torchvision import transforms
    from torchvision.datasets import ImageFolder
    base_transform = transforms.Compose([
        transforms.Resize(int(args.img_size / data_config['crop_pct'])),
        transforms.CenterCrop(args.img_size),
    ])
    final_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=data_config['mean'], std=data_config['std']),
    ])

    val_dir = os.path.join(args.data_dir, args.val_split)
    raw_val_dataset = ImageFolder(val_dir)
    clean_val_dataset = _SimpleTransformDataset(
        raw_val_dataset, base_transform, final_transform,
    )
    val_bs = args.validation_batch_size or args.batch_size
    val_loader = torch.utils.data.DataLoader(
        clean_val_dataset,
        batch_size=val_bs,
        shuffle=False,
        num_workers=min(args.workers, 4),
        pin_memory=args.pin_mem,
    )

    _logger.info(
        f'Eval analyze: {len(clean_val_dataset)} val samples, '
        f'{args.num_classes} ImageNet classes'
    )
    results = eval_analyze_per_class(
        backbone=backbone,
        stem_extractor=stem_extractor,
        aug_classifier=aug_classifier,
        val_loader=val_loader,
        device=device,
        num_imagenet_classes=args.num_classes,
        num_aug_classes=num_transforms + 1,
    )

    output_path = args.eval_analyze_output
    if not output_path:
        exp_name = args.experiment or f'phase1_{args.model}_direct'
        output_dir = Path(args.output) if args.output else Path(f'./output/{exp_name}')
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / 'eval_analyze_per_class.tsv'

    _write_eval_analyze_results(results, Path(output_path), transform_names)

    total_aug_correct = sum(r['aug_correct'] for r in results.values())
    total_backbone_correct = sum(r['backbone_correct'] for r in results.values())
    total_count = sum(r['count'] for r in results.values())
    _logger.info(
        f'Eval analyze done. Aug acc: {100*total_aug_correct/total_count:.2f}% '
        f'({total_aug_correct}/{total_count}), '
        f'Backbone acc: {100*total_backbone_correct/total_count:.2f}% '
        f'({total_backbone_correct}/{total_count})'
    )


# =============================================================================
# Main
# =============================================================================

def main():
    utils.setup_default_logging()
    args, args_text = _parse_args()

    if args.workers is None:
        args.workers = min(4, max(1, os.cpu_count() // 4))
        _logger.info(f'Auto workers: {args.workers}')

    if args.device_modules:
        for mod in args.device_modules:
            importlib.import_module(mod)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    device = torch.device(args.device)
    utils.random_seed(args.seed, 0)

    # =========================================================================
    # Eval Analyze mode: load backbone + aug classifier, run per-class analysis
    # =========================================================================
    if args.eval_analyze:
        _run_eval_analyze(args, device)
        return

    # =========================================================================
    # Backbone (frozen)
    # =========================================================================
    _logger.info(f'Creating backbone: {args.model}')
    backbone = create_model(
        args.model, pretrained=args.pretrained,
        num_classes=args.num_classes,
        checkpoint_path=args.initial_checkpoint,
    ).to(device).eval()
    for p in backbone.parameters():
        p.requires_grad = False

    # =========================================================================
    # Stem extractor (frozen, patch_embed only, NO pooling)
    # =========================================================================
    stem_extractor = StemFeatureExtractor(backbone).to(device).eval()
    for p in stem_extractor.parameters():
        p.requires_grad = False

    with torch.no_grad():
        dummy = torch.randn(1, 3, args.img_size, args.img_size, device=device)
        dummy_out = stem_extractor(dummy)
        stem_channels = dummy_out.shape[1]
        stem_h, stem_w = dummy_out.shape[2], dummy_out.shape[3]
    _logger.info(f'Stem (patch_embed): [{stem_channels}, {stem_h}, {stem_w}]')

    # Compute number of stride-2 stages needed: stem_h → 4
    num_dw_stages = 0
    h = stem_h
    while h > 4:
        h = (h - 3 + 2) // 2 + 1  # Conv2d(k=3, s=2, p=1) formula
        num_dw_stages += 1
    _logger.info(f'DW stages: {num_dw_stages} ({stem_h}→4)')
    assert h == 4, f'Cannot reduce {stem_h} to 4 with stride-2 3×3 conv (got {h})'

    # =========================================================================
    # Aug classifier (trainable)
    # =========================================================================
    num_transforms = get_augmix_sl_num_transforms(version=2)
    transform_names = get_augmix_sl_transform_names(version=2)

    aug_classifier = DirectAugClassifier(
        stem_channels=stem_channels,
        output_channels=64,
        num_dw_stages=num_dw_stages,
        num_transforms=num_transforms,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
        dw_init_mode=args.dw_init_mode,
    ).to(device)

    n_params = sum(p.numel() for p in aug_classifier.parameters())
    n_dw = sum(
        p.numel() for m in aug_classifier.dw_stages.modules()
        if isinstance(m, nn.Conv2d) for p in m.parameters()
    )
    n_pw = sum(p.numel() for p in aug_classifier.pw_conv.parameters())
    _logger.info(
        f'Aug classifier: {n_params:,} params '
        f'(IN={stem_channels * 2}, DW={n_dw:,}, clean_ref={aug_classifier.clean_ref.numel():,}, '
        f'PW={n_pw:,} [{stem_channels}→64], feat_dim={aug_classifier.feature_dim})'
    )

    # =========================================================================
    # Loss, optimizer, scheduler
    # =========================================================================
    criterion = HypersphericalManifoldLoss(
        lambda_radius=args.lambda_radius,
        lambda_angular=args.lambda_angular,
        lambda_dist=args.lambda_dist,
    ).to(device)

    from timm.optim import create_optimizer_v2
    optimizer = create_optimizer_v2(
        aug_classifier, opt=args.opt, lr=args.lr,
        weight_decay=args.weight_decay, momentum=args.momentum,
    )

    from timm.scheduler import create_scheduler_v2
    lr_scheduler, num_epochs = create_scheduler_v2(
        optimizer, sched=args.sched, num_epochs=args.epochs,
        warmup_epochs=args.warmup_epochs, warmup_lr=args.warmup_lr,
        min_lr=args.min_lr,
    )

    # =========================================================================
    # Datasets
    # =========================================================================
    data_config = resolve_data_config(vars(args), model=backbone, verbose=True)

    from torchvision import transforms
    base_transform = transforms.Compose([
        transforms.Resize(int(args.img_size / data_config['crop_pct'])),
        transforms.CenterCrop(args.img_size),
    ])
    final_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=data_config['mean'], std=data_config['std']),
    ])

    augmix_ops = augmix_sl_ops_v2()

    from torchvision.datasets import ImageFolder
    train_dir = os.path.join(args.data_dir, args.train_split)
    raw_dataset = ImageFolder(train_dir)
    _logger.info(f'Dataset: {len(raw_dataset)} images from {train_dir}')

    dataset_train = PairedAugDataset(
        raw_dataset, augmix_ops,
        min_sl=args.min_sl, max_sl=args.max_sl,
        base_transform=base_transform, final_transform=final_transform,
        cache_images=args.cache_images,
    )

    loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=args.pin_mem, drop_last=True,
        persistent_workers=False,
        prefetch_factor=2 if args.workers > 0 else None,
    )

    val_dir = os.path.join(args.data_dir, args.val_split)
    raw_val_dataset = ImageFolder(val_dir)

    # =========================================================================
    # Output
    # =========================================================================
    exp_name = args.experiment or f'phase1_{args.model}_direct'
    output_dir = Path(args.output) if args.output else Path(f'./output/{exp_name}')
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'args.yaml', 'w') as f:
        f.write(args_text)
    _logger.info(f'Output: {output_dir}')

    if args.log_wandb and has_wandb:
        wandb.init(project=args.wandb_project, name=exp_name, config=args)

    results_path = output_dir / 'phase1_results.txt'
    header = ['epoch', 'loss', 'L_cls', 'L_radius', 'L_angular', 'L_dist', 'lr']
    header += ['val_clean'] + [f'val_{n}' for n in transform_names] + ['val_mean', 'best']
    if args.resume:
        # Append to existing results file
        if not results_path.exists():
            with open(results_path, 'w') as f:
                f.write('\t'.join(header) + '\n')
    else:
        with open(results_path, 'w') as f:
            f.write('\t'.join(header) + '\n')

    # =========================================================================
    # Resume from checkpoint (if requested)
    # =========================================================================
    start_epoch = 0
    if args.resume:
        assert os.path.isfile(args.resume), f'Checkpoint not found: {args.resume}'
        ckpt = torch.load(args.resume, map_location=device)
        aug_classifier.load_state_dict(ckpt['aug_classifier'])
        _logger.info(
            f'Loaded aug_classifier from {args.resume} '
            f'(epoch {ckpt["epoch"]}, metric {ckpt.get("metric", "N/A")})'
        )
        if not args.resume_weights_only:
            optimizer.load_state_dict(ckpt['optimizer'])
            if lr_scheduler is not None and ckpt.get('lr_scheduler') is not None:
                lr_scheduler.load_state_dict(ckpt['lr_scheduler'])
            start_epoch = ckpt['epoch'] + 1
            _logger.info(
                f'Full resume: optimizer + scheduler restored, '
                f'starting from epoch {start_epoch}'
            )
        else:
            _logger.info(
                'Warm restart: fresh optimizer/scheduler, starting from epoch 0'
            )
        del ckpt
        torch.cuda.empty_cache()

    # =========================================================================
    # Warm-start clean_ref → Calibrate radius
    # =========================================================================
    if args.resume and not args.resume_weights_only:
        _logger.info('Skipping warm-start/calibration (restored from checkpoint)')
    else:
        _logger.info(f'Warm-starting clean_ref ({args.warmstart_samples} samples)...')
        aug_classifier.warmstart_clean_ref(
            stem_extractor, loader_train, device,
            num_samples=args.warmstart_samples,
        )

        _logger.info(f'Calibrating radius ({args.calibrate_batches} batches)...')
        aug_classifier.calibrate_radius(
            stem_extractor, loader_train, device,
            num_batches=args.calibrate_batches,
        )

    # =========================================================================
    # Training loop
    # =========================================================================
    best_metric = None
    best_epoch = None
    top_checkpoints = []

    _logger.info(
        f'Phase 1 (Direct): epochs {start_epoch}..{num_epochs-1}, '
        f'DW stages={num_dw_stages}, stem={stem_channels}ch, '
        f'SL=[{args.min_sl},{args.max_sl}], val_SL={args.val_severity}, '
        f'workers={args.workers}, bs={args.batch_size}'
    )

    try:
        for epoch in range(start_epoch, num_epochs):
            train_metrics = train_one_epoch(
                epoch, stem_extractor, aug_classifier, loader_train,
                optimizer, criterion, args, device,
            )

            if lr_scheduler is not None:
                lr_scheduler.step(epoch + 1)

            # ---- Validation ----
            mean_acc = 0.0
            per_cond = {}
            if (epoch + 1) % args.val_interval == 0 or epoch == num_epochs - 1:
                mean_acc, per_cond = validate(
                    stem_extractor, aug_classifier, raw_val_dataset, augmix_ops,
                    severity=args.val_severity,
                    base_transform=base_transform, final_transform=final_transform,
                    args=args, device=device, transform_names=transform_names,
                )

            # ---- Checkpoint ----
            ckpt_data = {
                'epoch': epoch,
                'aug_classifier': aug_classifier.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': (
                    lr_scheduler.state_dict() if lr_scheduler is not None else None
                ),
                'metric': mean_acc,
                'args': args.__dict__,
            }
            torch.save(ckpt_data, output_dir / 'last.pth.tar')

            is_best = ''
            if per_cond:
                current_metric = mean_acc
                if best_metric is None or current_metric > best_metric:
                    best_metric = current_metric
                    best_epoch = epoch
                    torch.save(ckpt_data, output_dir / 'best.pth.tar')
                    is_best = '*'

                ckpt_path = str(output_dir / f'checkpoint-{epoch}.pth.tar')
                max_keep = args.checkpoint_hist
                if len(top_checkpoints) < max_keep:
                    torch.save(ckpt_data, ckpt_path)
                    top_checkpoints.append((current_metric, epoch, ckpt_path))
                    top_checkpoints.sort(key=lambda x: x[0], reverse=True)
                else:
                    worst_m, worst_e, worst_p = top_checkpoints[-1]
                    if current_metric > worst_m:
                        if os.path.exists(worst_p):
                            os.remove(worst_p)
                        torch.save(ckpt_data, ckpt_path)
                        top_checkpoints[-1] = (current_metric, epoch, ckpt_path)
                        top_checkpoints.sort(key=lambda x: x[0], reverse=True)

            # ---- Log ----
            lr = optimizer.param_groups[0]['lr']
            row = [
                str(epoch), f'{train_metrics["loss"]:.4f}',
                f'{train_metrics["L_cls"]:.4f}', f'{train_metrics["L_radius"]:.4f}',
                f'{train_metrics["L_angular"]:.4f}', f'{train_metrics["L_dist"]:.4f}',
                f'{lr:.2e}',
            ]
            if per_cond:
                row.append(f'{per_cond.get("clean", 0):.2f}')
                for tn in transform_names:
                    row.append(f'{per_cond.get(tn, 0):.2f}')
                row.append(f'{mean_acc:.2f}')
            else:
                row += [''] * (2 + len(transform_names))
            row.append(is_best)
            with open(results_path, 'a') as f:
                f.write('\t'.join(row) + '\n')

            if args.log_wandb and has_wandb:
                log_dict = {'epoch': epoch, 'lr': lr, **train_metrics}
                if per_cond:
                    log_dict['val_mean_acc'] = mean_acc
                    for k, v in per_cond.items():
                        log_dict[f'val_{k}'] = v
                wandb.log(log_dict)

    except KeyboardInterrupt:
        _logger.info('Interrupted.')

    if best_metric is not None:
        _logger.info(f'*** Best val mean acc: {best_metric:.2f}% (epoch {best_epoch})')
        for m, e, p in top_checkpoints:
            _logger.info(f'  epoch {e}: {m:.2f}% — {p}')


if __name__ == '__main__':
    main()

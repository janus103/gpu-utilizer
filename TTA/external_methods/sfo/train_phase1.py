#!/usr/bin/env python3
"""Phase 1: Distance-Aware Augmentation Classifier Pre-Training (Hyperspherical Manifold).

RAM Usage free -h | awk '/^Mem:/ {print "Total: " $2, " Used: " $3, " Free: " $4, " Available: " $7}'

Trains the augmentation classifier with self-reference and Hyperspherical structure:
  - Self-reference: z = f(x_aug) - f(x_clean) — pure augmentation displacement.
  - Hyperspherical structure: per-aug radius + angular repulsion.
  - Distance head: predicts corruption magnitude.
  - 8-class classification: clean (class 0) + 7 V2 augmentation types (classes 1-7).

Data: Mini-batch groups of (1 clean + 7 augmented) from the SAME original image.

Usage (ResNet):
    python train_phase1.py --data-dir /path/to/imagenet --model resnet50 --pretrained \\
        --train-split validation --proj-type C --stem-mode conv1

Usage (ViT):
    python train_phase1.py --data-dir /path/to/imagenet --model vit_base_patch16_224 --pretrained \\
        --train-split validation --proj-type C --stem-mode patch_embed \\
        --hidden-dims 2048 512

Multi-job safe (12+ concurrent on single GPU):
    python train_phase1.py ... --batch-size 16 -j 2
"""
import argparse
import gc
import importlib
import logging
import math
import os
import time
from collections import OrderedDict
from contextlib import suppress
from functools import partial
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

_logger = logging.getLogger('train_phase1')


# =============================================================================
# Stem Feature Extractor (configurable stem mode)
# =============================================================================

class StemFeatureExtractor(nn.Module):
    """Extract spatial features from the first layer with configurable depth.

    CNN modes (ResNet, etc.):
        ``conv1``: conv1 only (raw, unnormalized).
        ``conv1_bn1``: conv1 + bn1 (batch-normalized).
        ``conv1_bn1_act1``: conv1 + bn1 + act1 (normalized + activated).

    ViT modes:
        ``patch_embed``: patch_embed.proj only (Conv2d projection, raw).
        ``patch_embed_norm``: patch_embed.proj + patch_embed.norm.

    Output: ``[B, C, 4, 4]`` spatial tensor (NOT flattened).

    Note:
        For CNN (e.g. ResNet50): C=64, feature_dim=1,024.
        For ViT-B/16: C=768, feature_dim=12,288.
    """

    # Modes grouped by architecture family
    CNN_MODES = ('conv1', 'conv1_bn1', 'conv1_bn1_act1')
    VIT_MODES = ('patch_embed', 'patch_embed_norm')

    def __init__(self, model, stem_mode='conv1'):
        super().__init__()
        self.stem_mode = stem_mode

        if stem_mode in self.CNN_MODES:
            self._init_cnn(model, stem_mode)
        elif stem_mode in self.VIT_MODES:
            self._init_vit(model, stem_mode)
        else:
            raise ValueError(
                f"Unknown stem_mode='{stem_mode}'. "
                f"Choose from {self.CNN_MODES + self.VIT_MODES}"
            )

        self.pool = nn.AdaptiveAvgPool2d((4, 4))

    def _init_cnn(self, model, stem_mode):
        """Initialise for CNN-style models (ResNet, etc.)."""
        if not hasattr(model, 'conv1'):
            raise ValueError("Model does not have conv1 layer")
        self.conv1 = model.conv1

        if stem_mode in ('conv1_bn1', 'conv1_bn1_act1'):
            if not hasattr(model, 'bn1'):
                raise ValueError(f"stem_mode='{stem_mode}' requires bn1")
            self.bn1 = model.bn1

        if stem_mode == 'conv1_bn1_act1':
            if not hasattr(model, 'act1'):
                raise ValueError(f"stem_mode='{stem_mode}' requires act1")
            self.act1 = model.act1

    def _init_vit(self, model, stem_mode):
        """Initialise for ViT-style models (patch_embed.proj)."""
        if not hasattr(model, 'patch_embed'):
            raise ValueError("Model does not have patch_embed layer")
        pe = model.patch_embed
        if not hasattr(pe, 'proj'):
            raise ValueError("patch_embed does not have proj (Conv2d) layer")
        self.proj = pe.proj  # Conv2d(3, embed_dim, patch_size, stride=patch_size)

        if stem_mode == 'patch_embed_norm':
            if hasattr(pe, 'norm') and not isinstance(pe.norm, nn.Identity):
                self.patch_norm = pe.norm
            else:
                _logger.warning(
                    "patch_embed_norm requested but patch_embed.norm is Identity — "
                    "falling back to patch_embed (no norm)."
                )
                self.patch_norm = None
        else:
            self.patch_norm = None

    def forward(self, x):
        if self.stem_mode in self.CNN_MODES:
            x = self.conv1(x)
            if self.stem_mode in ('conv1_bn1', 'conv1_bn1_act1'):
                x = self.bn1(x)
            if self.stem_mode == 'conv1_bn1_act1':
                x = self.act1(x)
        else:
            # ViT: patch_embed.proj → [B, embed_dim, H', W'] (NCHW)
            x = self.proj(x)
            if self.patch_norm is not None:
                # norm expects NLC format: [B, N, C]
                B, C, H, W = x.shape
                x = x.flatten(2).transpose(1, 2)   # [B, N, C]
                x = self.patch_norm(x)
                x = x.transpose(1, 2).view(B, C, H, W)  # back to NCHW

        x = self.pool(x)
        return x  # [B, C, 4, 4]


# =============================================================================
# Distance-Aware Augmentation Classifier
# =============================================================================

class DistanceAwareAugClassifier(nn.Module):
    """Augmentation classifier with learnable projection + Hyperspherical structure.

    Optional channel reduction (``reduce_channels``):
        When the stem output has many channels (e.g. ViT-B/16 = 768), a
        trainable 1x1 Conv2d reduces them before the projection layer.
        Since 1x1 conv is linear, ``reduce(f_aug - f_clean) ==
        reduce(f_aug) - reduce(f_clean)``, so applying it on the diff is
        mathematically equivalent to applying it on individual features.

    Projection types (applied on ``[B, C', 4, 4]`` spatial diff, where
    C' = ``reduce_channels`` if set, else ``channels``):
        ``A``: ``Linear(feat_dim, feat_dim)`` after flatten.
        ``B``: ``Conv2d(C', C', 1)`` — 1x1 pointwise.
        ``C``: Depthwise-separable conv — spatial + channel mixing.

    Two output heads:
        ``aug_head``:  ``[B, num_classes]`` (8-class: clean + 7 aug types).
        ``dist_head``: ``[B]`` corruption magnitude score (Softplus > 0).

    Per-augmentation learnable radius ``R_i = exp(log_r_i)``.
    """

    def __init__(
        self,
        channels: int = 64,
        spatial_size: int = 4,
        num_transforms: int = AUGMIX_SL_V2_NUM_TRANSFORMS,
        proj_type: str = 'B',
        hidden_dims: list = None,
        dropout: float = 0.1,
        reduce_channels: int = None,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256]

        # --- Channel reduction (e.g. ViT 768 → 64) ---
        if reduce_channels is not None and reduce_channels != channels:
            self.channel_reduce = nn.Conv2d(channels, reduce_channels, 1)
            proj_ch = reduce_channels
        else:
            self.channel_reduce = None
            proj_ch = channels

        self.stem_channels = channels       # original stem output channels
        self.channels = proj_ch             # effective channels after reduction
        self.spatial_size = spatial_size
        self.feature_dim = proj_ch * spatial_size * spatial_size
        self.num_transforms = num_transforms
        self.num_classes = num_transforms + 1  # +1 for clean (class 0)
        self.proj_type = proj_type

        # --- Projection layer (operates on reduced channels) ---
        if proj_type == 'A':
            self.proj = nn.Linear(self.feature_dim, self.feature_dim)
        elif proj_type == 'B':
            self.proj = nn.Conv2d(proj_ch, proj_ch, 1)
        elif proj_type == 'C':
            self.proj = nn.Sequential(
                nn.Conv2d(proj_ch, proj_ch, 3, 1, 1, groups=proj_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(proj_ch, proj_ch, 1),
            )
        else:
            raise ValueError(f"Unknown proj_type: {proj_type}")

        # --- Per-augmentation learnable radius (calibrated before training) ---
        self.log_r = nn.Parameter(torch.full((num_transforms,), 3.0))  # R≈20 fallback

        # --- Shared encoder: input = [z_flat || ||z||] ---
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

        # --- Distance / corruption head ---
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
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @property
    def radius(self):
        """Per-augmentation radius R_i, always positive."""
        return torch.exp(self.log_r)

    def project_diff(self, diff_spatial: torch.Tensor) -> torch.Tensor:
        """Project spatial difference through the learnable projection.

        If ``channel_reduce`` is set, applies 1x1 conv to reduce channels
        first (e.g. 768 → 64 for ViT).

        Args:
            diff_spatial: ``[B, C_stem, 4, 4]`` spatial difference tensor.

        Returns:
            z_flat: ``[B, feature_dim]`` projected vector.
        """
        if self.channel_reduce is not None:
            diff_spatial = self.channel_reduce(diff_spatial)
        if self.proj_type == 'A':
            return self.proj(diff_spatial.flatten(1))
        return self.proj(diff_spatial).flatten(1)

    @torch.no_grad()
    def calibrate_radius(self, sample_diffs: list):
        """Set initial log_r so R_i ≈ mean(||z_i||) for each aug type.

        Call once before training to avoid huge initial L_radius.

        Args:
            sample_diffs: List of T tensors, each ``[N, C, 4, 4]``.
        """
        device = self.log_r.device
        for t, diff_spatial in enumerate(sample_diffs):
            z = self.project_diff(diff_spatial.to(device))
            mean_norm = z.norm(dim=1).mean().item()
            self.log_r.data[t] = math.log(max(mean_norm, 1e-3))
        _logger.info(
            f'Calibrated R: [{", ".join(f"{r:.1f}" for r in self.radius.tolist())}]'
        )

    def forward(self, z_flat: torch.Tensor):
        """Forward pass.

        Args:
            z_flat: ``[B, feature_dim]`` projected difference.

        Returns:
            aug_output: ``[B, num_classes]`` logits.
            dist_output: ``[B]`` corruption score.
        """
        dist = z_flat.norm(dim=1, keepdim=True)  # [B, 1]
        x = torch.cat([z_flat, dist], dim=1)      # [B, feat_dim+1]
        shared = self.shared_encoder(x)
        aug_output = self.aug_head(shared)
        dist_output = self.dist_head(shared).squeeze(-1)
        return aug_output, dist_output


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

        # (2) Radius constraint: (||z_i|| - R_{type(i)})^2 for augmented only
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
    """Returns ``(clean_img, aug_imgs[7], class_label)`` from the SAME image.

    Each ``__getitem__`` applies all 7 V2 augmentation types to one image.
    If ``fixed_sl`` is set, all augmentations use that severity (for validation).

    Performance optimisations:
      - ``cache_images=True``: pre-loads all base-transformed images into RAM
        as uint8 numpy arrays, eliminating disk I/O after the first pass.
        Cost: ~7.2 GB for 50 K images at 224x224 (shared via COW with fork workers).
      - Pre-allocated output tensor avoids list+stack overhead.
    """

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

        # ---- Optional image cache (eliminates disk I/O per epoch) ----
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
        """Get base-transformed PIL image (from cache or disk)."""
        if self._cache is not None:
            from PIL import Image
            arr = self._cache[idx]
            return Image.fromarray(arr), self._labels[idx]
        img_pil, label = self.dataset[idx]
        if self.base_transform is not None:
            img_pil = self.base_transform(img_pil)
        return img_pil, label

    def __getitem__(self, idx):
        img_pil, label = self._get_pil_image(idx)

        clean_img = self.final_transform(img_pil)

        # Pre-allocate output tensor (avoids list + torch.stack overhead)
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
    aug_classifier: DistanceAwareAugClassifier,
    loader,
    optimizer,
    criterion: HypersphericalManifoldLoss,
    args,
    device: torch.device,
    fsc_centroids_spatial: torch.Tensor = None,
) -> OrderedDict:
    """One epoch of Phase-1 training.

    Args:
        fsc_centroids_spatial: ``[num_classes, C, 4, 4]`` FSC centroids in spatial
            format.  Required when ``args.ref_mode == 'fsc'``, ignored otherwise.
    """
    aug_classifier.train()
    stem_extractor.eval()

    loss_m = utils.AverageMeter()
    cls_m = utils.AverageMeter()
    rad_m = utils.AverageMeter()
    ang_m = utils.AverageMeter()
    dist_m = utils.AverageMeter()
    batch_time_m = utils.AverageMeter()

    ref_mode = args.ref_mode
    num_batches = len(loader)
    end = time.time()

    for batch_idx, (clean_imgs, aug_imgs, labels) in enumerate(loader):
        B = clean_imgs.size(0)
        T = aug_imgs.size(1)  # 7

        # ---- Stem features: memory-efficient (one view at a time) ----
        with torch.no_grad():
            f_clean = stem_extractor(clean_imgs.to(device))       # [B, C, 4, 4]
            f_augs_parts = []
            for t in range(T):
                f_augs_parts.append(
                    stem_extractor(aug_imgs[:, t].to(device))     # [B, C, 4, 4]
                )
            f_augs_flat = torch.cat(f_augs_parts, dim=0)          # [T*B, C, 4, 4]
            del f_augs_parts

        # ---- Compute differences based on ref_mode ----
        if ref_mode == 'fsc':
            # FSC-based: diff = f(img) - global_mean_centroid
            ref_spatial = fsc_centroids_spatial.expand(B, -1, -1, -1)   # [B, C, 4, 4]
            diff_clean = f_clean - ref_spatial
            ref_rep = fsc_centroids_spatial.expand(T * B, -1, -1, -1)   # [T*B, C, 4, 4]
            diff_augs = f_augs_flat - ref_rep
        else:
            # Self-reference: diff = f(aug) - f(clean)
            f_clean_rep = f_clean.repeat(T, 1, 1, 1)
            diff_augs = f_augs_flat - f_clean_rep
            diff_clean = torch.zeros_like(f_clean)
            del f_clean_rep
        del f_augs_flat

        # ---- Project + classify in single batched call ----
        diff_all = torch.cat([diff_clean, diff_augs], dim=0)
        z_all = aug_classifier.project_diff(diff_all)
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
# Validation: Per-Augmentation Accuracy (8 conditions)
# =============================================================================

@torch.no_grad()
def validate(
    stem_extractor: nn.Module,
    aug_classifier: DistanceAwareAugClassifier,
    val_dataset,
    augmix_ops,
    severity: float,
    base_transform,
    final_transform,
    args,
    device: torch.device,
    transform_names: list,
    fsc_centroids_spatial: torch.Tensor = None,
) -> tuple:
    """Validate 8-class accuracy: clean (class 0) + 7 aug types (classes 1-7).

    Supports both ``self`` and ``fsc`` reference modes.
    """
    aug_classifier.eval()
    stem_extractor.eval()
    ref_mode = args.ref_mode

    val_bs = args.validation_batch_size or args.batch_size

    # ---- Pre-compute clean stem features + labels ----
    clean_dataset = _SimpleTransformDataset(val_dataset, base_transform, final_transform)
    clean_loader = torch.utils.data.DataLoader(
        clean_dataset, batch_size=val_bs, shuffle=False,
        num_workers=min(args.workers, 4), pin_memory=args.pin_mem,
    )

    all_f_clean = []
    all_labels = []
    for imgs, lbls in clean_loader:
        f = stem_extractor(imgs.to(device))
        all_f_clean.append(f.cpu())
        all_labels.append(lbls)
    all_f_clean = torch.cat(all_f_clean, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    per_cond = {}

    # ---- Helper: compute ref for a batch slice ----
    def _get_ref(start, end_idx):
        """Return spatial reference [bs, C, 4, 4] for the given slice."""
        bs = end_idx - start
        if ref_mode == 'fsc':
            return fsc_centroids_spatial.expand(bs, -1, -1, -1)  # global mean
        else:
            return all_f_clean[start:end_idx].to(device)  # self-ref = clean features

    # ---- Condition 0: Clean ----
    correct = total = 0
    for start in range(0, len(all_f_clean), val_bs):
        end_idx = min(start + val_bs, len(all_f_clean))
        f_clean = all_f_clean[start:end_idx].to(device)
        ref = _get_ref(start, end_idx)
        diff = f_clean - ref if ref_mode == 'fsc' else torch.zeros_like(f_clean)
        z = aug_classifier.project_diff(diff)
        aug_out, _ = aug_classifier(z)
        correct += (aug_out.argmax(1) == 0).sum().item()
        total += f_clean.size(0)
    per_cond['clean'] = 100.0 * correct / total if total else 0.0
    _logger.info(f'  Val [clean]: {per_cond["clean"]:.2f}%')

    # ---- Conditions 1-7: Each augmentation type ----
    for t_idx, (op, t_name) in enumerate(zip(augmix_ops, transform_names)):
        aug_dataset = _SingleAugDataset(val_dataset, op, severity, base_transform, final_transform)
        aug_loader = torch.utils.data.DataLoader(
            aug_dataset, batch_size=val_bs, shuffle=False,
            num_workers=min(args.workers, 4), pin_memory=args.pin_mem,
        )

        correct = total = ptr = 0
        for aug_imgs, _ in aug_loader:
            bs = aug_imgs.size(0)
            f_aug = stem_extractor(aug_imgs.to(device))
            ref = _get_ref(ptr, ptr + bs)
            diff = f_aug - ref
            z = aug_classifier.project_diff(diff)
            aug_out, _ = aug_classifier(z)
            correct += (aug_out.argmax(1) == t_idx + 1).sum().item()
            total += bs
            ptr += bs

        acc = 100.0 * correct / total if total else 0.0
        per_cond[t_name] = acc
        _logger.info(f'  Val [{t_name}] (SL={severity}): {acc:.2f}%')
        del aug_loader, aug_dataset
        gc.collect()

    mean_acc = sum(per_cond.values()) / len(per_cond)
    _logger.info(f'  Val Mean Acc (8-way): {mean_acc:.2f}%')
    return mean_acc, per_cond


class _SimpleTransformDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, base_transform, final_transform):
        self.dataset, self.base_transform, self.final_transform = dataset, base_transform, final_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        if self.base_transform is not None:
            img = self.base_transform(img)
        return self.final_transform(img), label


class _SingleAugDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, aug_op, severity, base_transform, final_transform):
        self.dataset, self.aug_op, self.severity = dataset, aug_op, severity
        self.base_transform, self.final_transform = base_transform, final_transform

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

parser = argparse.ArgumentParser(description='Phase 1: Distance-Aware Aug Classifier')

group = parser.add_argument_group('Dataset')
group.add_argument('--data-dir', type=str, required=True)
group.add_argument('--train-split', default='val')
group.add_argument('--val-split', default='val')

group = parser.add_argument_group('Model')
group.add_argument('--model', default='resnet50', type=str)
group.add_argument('--pretrained', action='store_true')
group.add_argument('--initial-checkpoint', default='', type=str)
group.add_argument('--num-classes', type=int, default=1000)
group.add_argument('--img-size', type=int, default=224)
group.add_argument('--stem-mode', type=str, default='conv1',
                   choices=[
                       'conv1', 'conv1_bn1', 'conv1_bn1_act1',    # CNN (ResNet, etc.)
                       'patch_embed', 'patch_embed_norm',          # ViT
                   ],
                   help='Stem depth: conv1* for CNN, patch_embed* for ViT')

group = parser.add_argument_group('Projection & Classifier')
group.add_argument('--proj-type', type=str, default='B', choices=['A', 'B', 'C'])
group.add_argument('--reduce-channels', type=int, default=None,
                   help='Reduce stem channels via trainable 1x1 conv before projection '
                        '(e.g. 64 for ViT 768→64). None = no reduction.')
group.add_argument('--hidden-dims', type=int, nargs='+', default=[512, 256])
group.add_argument('--dropout', type=float, default=0.1)

group = parser.add_argument_group('Reference mode')
group.add_argument('--ref-mode', type=str, default='self', choices=['self', 'fsc'],
                   help='Reference for z computation: '
                        '"self" = f(aug)-f(clean) (paired, class-free), '
                        '"fsc" = f(img)-centroid[label] (FSC-based, Phase-2 compatible)')
group.add_argument('--fsc-path', type=str, default='',
                   help='Path to FSC file (required when --ref-mode fsc)')

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
group.add_argument('-b', '--batch-size', type=int, default=32,
                   help='Groups per batch (each = 1 clean + 7 aug)')
group.add_argument('-vb', '--validation-batch-size', type=int, default=None)
group.add_argument('-j', '--workers', type=int, default=None,
                   help='DataLoader workers (default: auto, recommend 2-4 for multi-job)')
group.add_argument('--device', default='cuda', type=str)
group.add_argument('--seed', type=int, default=42)
group.add_argument('--pin-mem', action='store_true')
group.add_argument('--log-interval', type=int, default=50)
group.add_argument('--val-interval', type=int, default=1)
group.add_argument('--output', default='', type=str)
group.add_argument('--experiment', default='', type=str)
group.add_argument('--checkpoint-hist', type=int, default=3)
group.add_argument('--cache-images', action='store_true',
                   help='Pre-cache base-transformed images in RAM (~7GB for 50K). '
                        'Eliminates disk I/O bottleneck. Shared via COW with fork workers.')
group.add_argument('--device-modules', default=None, type=str, nargs='+')
group.add_argument('--log-wandb', action='store_true')
group.add_argument('--wandb-project', default='phase1-aug-classifier', type=str)


def _parse_args():
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            parser.set_defaults(**yaml.safe_load(f))
    args = parser.parse_args(remaining)
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


# =============================================================================
# Main
# =============================================================================

def main():
    utils.setup_default_logging()
    args, args_text = _parse_args()

    # ---- Workers: default 4 (safe for multi-job shared memory) ----
    if args.workers is None:
        args.workers = min(4, max(1, os.cpu_count() // 4))
        _logger.info(f'Auto workers: {args.workers} (multi-job safe)')

    if args.device_modules:
        for mod in args.device_modules:
            importlib.import_module(mod)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    device = torch.device(args.device)
    utils.random_seed(args.seed, 0)

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
    # Stem extractor (frozen)
    # =========================================================================
    stem_extractor = StemFeatureExtractor(backbone, stem_mode=args.stem_mode).to(device).eval()
    for p in stem_extractor.parameters():
        p.requires_grad = False

    with torch.no_grad():
        dummy = torch.randn(1, 3, args.img_size, args.img_size, device=device)
        dummy_out = stem_extractor(dummy)
        channels, spatial = dummy_out.shape[1], dummy_out.shape[2]
    _logger.info(f'Stem ({args.stem_mode}): [{channels}, {spatial}, {spatial}]')

    # =========================================================================
    # FSC centroids (required for --ref-mode fsc)
    # =========================================================================
    fsc_centroids_spatial = None
    if args.ref_mode == 'fsc':
        if not args.fsc_path:
            raise ValueError('--fsc-path is required when --ref-mode fsc')
        _logger.info(f'Loading FSC: {args.fsc_path}')
        fsc_data = torch.load(args.fsc_path, map_location='cpu')
        fsc_centroids = fsc_data['centroids'].to(device)  # [num_classes, D]
        # Use global mean centroid (average over all classes) — class-agnostic
        fsc_mean = fsc_centroids.mean(dim=0)              # [D]
        fsc_centroids_spatial = fsc_mean.view(1, channels, spatial, spatial)  # [1, C, 4, 4]
        _logger.info(f'FSC loaded: {fsc_centroids.shape} → global mean centroid {fsc_centroids_spatial.shape}')
        fsc_stem = fsc_data.get('stem_mode', fsc_data.get('feature_source', '?'))
        _logger.info(f'FSC stem_mode: {fsc_stem}, current stem_mode: {args.stem_mode}')
    else:
        _logger.info('Reference mode: self (no FSC needed)')

    # =========================================================================
    # Aug classifier (trainable)
    # =========================================================================
    num_transforms = get_augmix_sl_num_transforms(version=2)
    transform_names = get_augmix_sl_transform_names(version=2)

    aug_classifier = DistanceAwareAugClassifier(
        channels=channels, spatial_size=spatial,
        num_transforms=num_transforms, proj_type=args.proj_type,
        hidden_dims=args.hidden_dims, dropout=args.dropout,
        reduce_channels=args.reduce_channels,
    ).to(device)
    n_params = sum(p.numel() for p in aug_classifier.parameters())
    n_proj = sum(p.numel() for p in aug_classifier.proj.parameters())
    n_reduce = (
        sum(p.numel() for p in aug_classifier.channel_reduce.parameters())
        if aug_classifier.channel_reduce is not None else 0
    )
    _logger.info(
        f'Aug classifier: {n_params:,} params '
        f'(reduce={n_reduce:,}, proj[{args.proj_type}]={n_proj:,}, '
        f'feat_dim={aug_classifier.feature_dim})'
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
        persistent_workers=False,   # False: safer for shared memory with multi-job
        prefetch_factor=2 if args.workers > 0 else None,
    )

    val_dir = os.path.join(args.data_dir, args.val_split)
    raw_val_dataset = ImageFolder(val_dir)

    # =========================================================================
    # Output
    # =========================================================================
    rc_tag = f'_rc{args.reduce_channels}' if args.reduce_channels else ''
    exp_name = args.experiment or (
        f'phase1_{args.model}_{args.stem_mode}_proj{args.proj_type}{rc_tag}_{args.ref_mode}'
    )
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
    with open(results_path, 'w') as f:
        f.write('\t'.join(header) + '\n')

    # =========================================================================
    # Calibrate radius from sample batches
    # =========================================================================
    _logger.info(f'Calibrating initial radius (ref_mode={args.ref_mode})...')
    sample_diffs = [[] for _ in range(num_transforms)]
    cal_batches = min(5, len(loader_train))
    for cal_idx, (clean_imgs, aug_imgs, labels_cal) in enumerate(loader_train):
        if cal_idx >= cal_batches:
            break
        with torch.no_grad():
            f_clean = stem_extractor(clean_imgs.to(device))
            if args.ref_mode == 'fsc':
                ref = fsc_centroids_spatial.expand(f_clean.size(0), -1, -1, -1)
            else:
                ref = f_clean
            for t in range(num_transforms):
                f_aug = stem_extractor(aug_imgs[:, t].to(device))
                sample_diffs[t].append((f_aug - ref).cpu())
    sample_diffs = [torch.cat(sd, dim=0) for sd in sample_diffs]
    aug_classifier.calibrate_radius(sample_diffs)
    del sample_diffs

    # =========================================================================
    # Training loop
    # =========================================================================
    best_metric = None
    best_epoch = None
    top_checkpoints = []

    rc_str = f', reduce={args.reduce_channels}' if args.reduce_channels else ''
    _logger.info(
        f'Phase 1: {num_epochs} epochs, proj={args.proj_type}{rc_str}, stem={args.stem_mode}, '
        f'ref={args.ref_mode}, SL=[{args.min_sl},{args.max_sl}], val_SL={args.val_severity}, '
        f'workers={args.workers}, bs={args.batch_size}'
    )

    try:
        for epoch in range(num_epochs):
            train_metrics = train_one_epoch(
                epoch, stem_extractor, aug_classifier, loader_train,
                optimizer, criterion, args, device,
                fsc_centroids_spatial=fsc_centroids_spatial,
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
                    fsc_centroids_spatial=fsc_centroids_spatial,
                )

            # ---- Checkpoint ----
            ckpt_data = {
                'epoch': epoch,
                'aug_classifier': aug_classifier.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict() if lr_scheduler is not None else None,
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

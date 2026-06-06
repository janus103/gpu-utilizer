#!/usr/bin/env python3
"""Phase 2: Meta-Learning for TTA with Clean Confidence Objective.

Loads a Phase-1 checkpoint (DistanceAwareAugClassifier) and meta-learns it
so that its inner-loop signal improves classification after TTA adaptation.

Inner loss (unsupervised — no labels at test time):
    L_in = α·CE(aug_head(z_θ), target=clean)   # push toward clean
         + δ·dist_head(z_θ)                     # minimize corruption magnitude
         + β·cos(g_θ, g_θ₀)                     # repel from corrupted
         - γ·cos(g_θ, c_ŷ)                      # attract to centroid
         + λ·‖θ - θ₀‖²                          # trust region

Outer loss (supervised — meta-update aug_classifier):
    L_out = CE(backbone(x; θ_K), y)

Usage:
    python train_phase2.py \\
        --data-dir /path/to/imagenet \\
        --model resnet50 --pretrained \\
        --phase1-checkpoint output2/phase1_conv1_bn_projC/best.pth.tar \\
        --fsc-path ZOA_FSC/resnet50_FSC_conv1_only.pth
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

# Re-use Phase-1 classes (identical architecture)
from train_phase1 import (
    StemFeatureExtractor,
    DistanceAwareAugClassifier,
    PairedAugDataset,
)

try:
    import wandb
    has_wandb = True
except ImportError:
    has_wandb = False

_logger = logging.getLogger('train_phase2')


# =============================================================================
# Curriculum Helpers
# =============================================================================

def get_curriculum_depth(epoch: int, total_epochs: int, max_depth: int = 3) -> int:
    """Step curriculum for augmentation depth.

    Splits total training into ``max_depth`` equal phases, starting at depth=1
    and increasing by 1 each phase.

    Examples (max_depth=3):
        0-33%  -> depth 1  (single augmentation)
        33-66% -> depth 2  (up to 2 combined)
        66-100% -> depth 3 (up to 3 combined)
    """
    if total_epochs <= 0 or max_depth <= 1:
        return max_depth
    progress = epoch / total_epochs
    for d in range(1, max_depth + 1):
        if progress < d / max_depth:
            return d
    return max_depth


def get_curriculum_sl_range(
    epoch: int,
    total_epochs: int,
    min_sl_start: float = 0.1,
    min_sl_end: float = 0.3,
    max_sl_start: float = 0.5,
    max_sl_end: float = 1.0,
) -> tuple:
    """Linear ramp for severity level range.

    Gradually widens the SL window from an easy range to the full range:
        epoch 0:   [min_sl_start, max_sl_start]  (e.g. [0.1, 0.5])
        last epoch: [min_sl_end,   max_sl_end]    (e.g. [0.3, 1.0])

    Returns:
        (current_min_sl, current_max_sl)
    """
    if total_epochs <= 1:
        return min_sl_end, max_sl_end
    progress = epoch / (total_epochs - 1)
    cur_min = min_sl_start + progress * (min_sl_end - min_sl_start)
    cur_max = max_sl_start + progress * (max_sl_end - max_sl_start)
    return cur_min, cur_max


# =============================================================================
# Functional Stem Forward (differentiable through adapted weights)
# =============================================================================

def functional_stem_forward(
    images: torch.Tensor,
    conv1_weight: torch.Tensor,
    conv1_bias,
    conv1_module: nn.Module,
    model: nn.Module = None,
    stem_mode: str = 'conv1',
) -> torch.Tensor:
    """Functional stem forward for differentiable inner loop.

    Uses ``F.conv2d`` so that ``create_graph=True`` gradients flow through
    ``conv1_weight``.  bn1 / act1 are called as regular modules (frozen but
    gradient passes through their ops).

    Returns:
        Spatial features ``[B, C, 4, 4]`` (NOT flattened).
    """
    x = images.contiguous()
    x = F.conv2d(
        x, conv1_weight, bias=conv1_bias,
        stride=conv1_module.stride, padding=conv1_module.padding,
        dilation=conv1_module.dilation, groups=conv1_module.groups,
    )
    if stem_mode in ('conv1_bn1', 'conv1_bn1_act1'):
        x = model.bn1(x)
    if stem_mode == 'conv1_bn1_act1':
        x = model.act1(x)
    x = F.adaptive_avg_pool2d(x, (4, 4))
    return x  # [B, C, 4, 4]


def functional_conv1_output(
    images: torch.Tensor,
    conv1_weight: torch.Tensor,
    conv1_bias,
    conv1_module: nn.Module,
) -> torch.Tensor:
    """Raw conv1 output [B, 64, 112, 112] for backbone forward."""
    return F.conv2d(
        images.contiguous(), conv1_weight, bias=conv1_bias,
        stride=conv1_module.stride, padding=conv1_module.padding,
        dilation=conv1_module.dilation, groups=conv1_module.groups,
    )


def backbone_forward_after_conv1(model: nn.Module, conv1_output: torch.Tensor) -> torch.Tensor:
    """bn1 → act1 → maxpool → layer1-4 → head → logits."""
    x = model.bn1(conv1_output)
    x = model.act1(x)
    x = model.maxpool(x)
    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)
    x = model.forward_head(x)
    return x


# =============================================================================
# Functional Projection (explicit weight/bias for create_graph)
# =============================================================================

def functional_project_diff(
    diff_spatial: torch.Tensor,
    aug_classifier: DistanceAwareAugClassifier,
) -> torch.Tensor:
    """Project diff through aug_classifier's projection (in graph)."""
    return aug_classifier.project_diff(diff_spatial)


# =============================================================================
# Differentiable Inner Loop: Clean Confidence Objective
# =============================================================================

def _inner_step_lr(step: int, inner_steps: int, inner_lr: float, inner_mode: str) -> float:
    """Return the effective learning rate for a given inner step.

    For ``truncated_scaled`` mode, the lr ramps linearly from
    ``inner_lr / inner_steps`` at step 0 to ``inner_lr`` at the last step.
    All other modes use a constant ``inner_lr``.
    """
    if inner_mode == 'truncated_scaled' and inner_steps > 1:
        return inner_lr * (step + 1) / inner_steps
    return inner_lr


def meta_inner_loop(
    images: torch.Tensor,
    model: nn.Module,
    aug_classifier: DistanceAwareAugClassifier,
    fsc_centroids: torch.Tensor,
    stem_mode: str,
    inner_lr: float,
    inner_steps: int,
    inner_mode: str = 'fomaml',
    alpha: float = 1.0,
    delta: float = 0.3,
    beta: float = 0.05,
    gamma: float = 0.3,
    lam: float = 1e-4,
) -> torch.Tensor:
    """Differentiable inner loop with Clean Confidence objective.

    Adapts conv1 weight virtually (in-graph) using:
        L_in = α·CE(aug_head(z), 0)  + δ·dist_head(z)
             + β·cos(g, g₀)          - γ·cos(g, c_ŷ)
             + λ·‖w - w₀‖²

    Three inner-loop modes control meta-gradient flow:

    ``fomaml`` (B):
        Only 1 inner step with ``create_graph=True``.
        ``inner_steps`` is forced to 1. Fastest and most stable.

    ``truncated`` (C):
        K inner steps for adaptation, but only the **last step** preserves
        the computation graph.  Earlier steps are detached so the
        meta-gradient flows through 1 step only.

    ``truncated_scaled`` (D):
        Same as ``truncated``, plus a linear lr ramp per step:
        ``lr_k = inner_lr * (k+1) / K``.  Early steps are gentle
        (exploratory), the last step is most aggressive and carries
        the meta-gradient.

    Args:
        images: ``[B, 3, H, W]``.
        model: Pretrained backbone (frozen).
        aug_classifier: Phase-1 trained classifier (meta-updated in outer loop).
        fsc_centroids: ``[num_classes, C*4*4]``.
        stem_mode: ``'conv1'`` or ``'conv1_bn1'`` or ``'conv1_bn1_act1'``.
        inner_lr: Inner loop learning rate.
        inner_steps: Number of adaptation steps K.
        inner_mode: ``'fomaml'``, ``'truncated'``, or ``'truncated_scaled'``.
        alpha..lam: Loss term weights.

    Returns:
        adapted_weight: Conv1 weight **in the autograd graph**.
    """
    device = images.device
    adapted_weight = model.conv1.weight.detach().clone().requires_grad_(True)
    w0 = adapted_weight.detach()  # for trust region
    conv1_bias = model.conv1.bias

    # ---- Original (corrupted) stem features (fixed reference) ----
    with torch.no_grad():
        g0_spatial = functional_stem_forward(
            images, model.conv1.weight, conv1_bias, model.conv1,
            model=model, stem_mode=stem_mode,
        )
        g0_flat = g0_spatial.flatten(1).detach()  # [B, D]

    # ---- Pseudo-label → FSC centroid selection ----
    with torch.no_grad():
        logits_init = model(images)
        pred_labels = logits_init.argmax(dim=1)  # [B]
    fsc_ref = fsc_centroids[pred_labels]  # [B, D]
    # Reshape to spatial for projection subtraction
    C, S = aug_classifier.channels, aug_classifier.spatial_size
    fsc_ref_spatial = fsc_ref.view(-1, C, S, S)  # [B, C, 4, 4]

    # ---- Clean target for CE ----
    clean_target = torch.zeros(images.size(0), dtype=torch.long, device=device)

    # ---- Determine effective K ----
    K = 1 if inner_mode == 'fomaml' else inner_steps

    for k in range(K):
        # Decide whether to keep the computation graph for this step.
        # fomaml: K=1, always create_graph=True (single step).
        # truncated / truncated_scaled: only the last step keeps the graph.
        is_last_step = (k == K - 1)
        create_graph = is_last_step

        # Effective lr for this step
        step_lr = _inner_step_lr(k, K, inner_lr, inner_mode)

        # Functional stem forward (gradient flows through adapted_weight)
        g_spatial = functional_stem_forward(
            images, adapted_weight, conv1_bias, model.conv1,
            model=model, stem_mode=stem_mode,
        )
        g_flat = g_spatial.flatten(1)  # [B, D]

        # Diff from centroid → projection → classifier
        diff_spatial = g_spatial - fsc_ref_spatial  # [B, C, 4, 4]
        z = aug_classifier.project_diff(diff_spatial)  # [B, D]
        aug_out, dist_out = aug_classifier(z)  # [B, 8], [B]

        # (A) Clean Confidence: CE toward class 0
        L_clean = F.cross_entropy(aug_out, clean_target)

        # (B) Distance minimization
        L_dist = dist_out.mean()

        # (C) Repel from corrupted
        cos_orig = F.cosine_similarity(g_flat, g0_flat, dim=1).mean()

        # (D) Attract to centroid
        cos_cent = F.cosine_similarity(g_flat, fsc_ref, dim=1).mean()

        # (E) Trust region
        L_trust = (adapted_weight - w0).pow(2).sum()

        # Combined inner loss
        loss_inner = (
            alpha * L_clean
            + delta * L_dist
            + beta * cos_orig
            - gamma * cos_cent
            + lam * L_trust
        )

        # Gradient step
        (grad,) = torch.autograd.grad(
            loss_inner, adapted_weight, create_graph=create_graph,
        )
        adapted_weight = adapted_weight - step_lr * grad

        # Detach intermediate steps (truncated / truncated_scaled)
        if not is_last_step:
            adapted_weight = adapted_weight.detach().requires_grad_(True)

    return adapted_weight


# =============================================================================
# Full Meta-Step: Inner Loop + Outer Classification
# =============================================================================

def meta_step(
    images: torch.Tensor,
    labels: torch.Tensor,
    model: nn.Module,
    aug_classifier: DistanceAwareAugClassifier,
    fsc_centroids: torch.Tensor,
    stem_mode: str,
    inner_lr: float,
    inner_steps: int,
    inner_mode: str,
    alpha: float, delta: float, beta: float, gamma: float, lam: float,
) -> tuple:
    """One meta-step: inner adaptation + outer classification loss.

    Returns:
        L_outer: Classification loss (in graph — call ``.backward()``).
        info: Dict with diagnostics.
    """
    model.eval()

    adapted_weight = meta_inner_loop(
        images, model, aug_classifier, fsc_centroids, stem_mode,
        inner_lr, inner_steps, inner_mode=inner_mode,
        alpha=alpha, delta=delta, beta=beta, gamma=gamma, lam=lam,
    )

    # Outer forward: adapted conv1 → full backbone → logits
    conv1_out = functional_conv1_output(
        images, adapted_weight, model.conv1.bias, model.conv1,
    )
    logits = backbone_forward_after_conv1(model, conv1_out)
    L_outer = F.cross_entropy(logits, labels)

    with torch.no_grad():
        maxk = min(5, logits.size(1))
        _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
        correct = pred.eq(labels.unsqueeze(1).expand_as(pred))
        top1 = correct[:, :1].float().sum().item() / images.size(0) * 100.0
        top5 = correct[:, :maxk].float().sum().item() / images.size(0) * 100.0

    # ---- Diagnostics: weight change magnitude ----
    with torch.no_grad():
        w_orig = model.conv1.weight.detach()
        w_delta = (adapted_weight.detach() - w_orig).norm().item()
        w_orig_norm = w_orig.norm().item()
        w_delta_ratio = w_delta / (w_orig_norm + 1e-10)

    return L_outer, {
        'loss': L_outer.item(),
        'top1': top1,
        'top5': top5,
        'w_delta': w_delta,
        'w_delta_ratio': w_delta_ratio,
    }


# =============================================================================
# Training
# =============================================================================

def train_one_epoch(
    epoch, model, aug_classifier, fsc_centroids, loader,
    meta_optimizer, args, device, stem_mode,
) -> OrderedDict:
    aug_classifier.train()
    model.eval()

    loss_m = utils.AverageMeter()
    top1_m = utils.AverageMeter()
    top5_m = utils.AverageMeter()
    grad_m = utils.AverageMeter()
    w_delta_m = utils.AverageMeter()
    w_delta_ratio_m = utils.AverageMeter()
    batch_time_m = utils.AverageMeter()

    num_batches = len(loader)
    end = time.time()

    for batch_idx, (images, labels, _aug_labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        meta_optimizer.zero_grad()

        L_outer, info = meta_step(
            images, labels, model, aug_classifier, fsc_centroids, stem_mode,
            inner_lr=args.inner_lr, inner_steps=args.inner_steps,
            inner_mode=args.inner_mode,
            alpha=args.alpha, delta=args.delta,
            beta=args.beta, gamma=args.gamma, lam=args.lam,
        )

        L_outer.backward()

        # Gradient norm for monitoring (before clipping)
        gnorm = 0.0
        for p in aug_classifier.parameters():
            if p.grad is not None:
                gnorm += p.grad.data.norm(2).item() ** 2
        gnorm = gnorm ** 0.5
        grad_m.update(gnorm)

        if args.clip_grad is not None:
            nn.utils.clip_grad_norm_(aug_classifier.parameters(), args.clip_grad)
        meta_optimizer.step()

        bs = images.size(0)
        loss_m.update(info['loss'], bs)
        top1_m.update(info['top1'], bs)
        top5_m.update(info['top5'], bs)
        w_delta_m.update(info['w_delta'], bs)
        w_delta_ratio_m.update(info['w_delta_ratio'], bs)
        batch_time_m.update(time.time() - end)
        end = time.time()

        if batch_idx % args.log_interval == 0 or batch_idx == num_batches - 1:
            lr = meta_optimizer.param_groups[0]['lr']
            _logger.info(
                f'Meta: {epoch} [{batch_idx:>4d}/{num_batches}]  '
                f'L: {loss_m.val:.4f}({loss_m.avg:.4f})  '
                f'T1: {top1_m.val:.1f}%({top1_m.avg:.1f}%)  '
                f'GN: {grad_m.val:.4f}  '
                f'Δw: {w_delta_m.val:.4f}({w_delta_ratio_m.val:.4f})  '
                f'T: {batch_time_m.val:.2f}s  LR: {lr:.2e}'
            )

    return OrderedDict([
        ('loss', loss_m.avg), ('top1', top1_m.avg), ('top5', top5_m.avg),
        ('grad_norm', grad_m.avg),
        ('w_delta', w_delta_m.avg), ('w_delta_ratio', w_delta_ratio_m.avg),
    ])


# =============================================================================
# Validation: Per-Transform TTA
# =============================================================================

def validate_tta(
    model, aug_classifier, fsc_centroids, raw_val_dataset,
    augmix_ops, transform_names, num_transforms, severity,
    base_transform, final_transform, args, device, stem_mode,
) -> dict:
    """Validate by running actual TTA (simple inner loop) per augmentation type.

    NOTE: no ``@torch.no_grad()`` decorator here because the inner loop
    needs gradients for conv1 adaptation.  Evaluation sections are wrapped
    with ``torch.no_grad()`` explicitly.
    """
    from timm.data.auto_augment import AugMixSLAugmentFixed

    model.eval()
    aug_classifier.eval()
    val_bs = args.validation_batch_size or args.batch_size

    per_t_top1 = {}

    for t_idx in range(num_transforms):
        t_name = transform_names[t_idx]
        fixed_aug = AugMixSLAugmentFixed(
            ops=augmix_ops, fixed_transforms=[(t_idx, severity)], max_sl=args.max_sl,
        )
        val_dataset = PairedAugDataset(
            raw_val_dataset, augmix_ops,
            base_transform=base_transform, final_transform=final_transform,
            fixed_sl=severity,
        )
        # Use _SingleAugDataset for simpler loading
        from train_phase1 import _SingleAugDataset
        aug_dataset = _SingleAugDataset(
            raw_val_dataset, augmix_ops[t_idx], severity,
            base_transform, final_transform,
        )
        loader = torch.utils.data.DataLoader(
            aug_dataset, batch_size=val_bs, shuffle=False,
            num_workers=min(args.workers, 4), pin_memory=args.pin_mem,
        )

        top1_m = utils.AverageMeter()
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            original_w = model.conv1.weight.data.clone()

            # Simple (non-differentiable) TTA inner loop
            # — needs gradients for conv1, so NOT inside no_grad
            _tta_inner_loop_simple(
                imgs, model, aug_classifier, fsc_centroids, stem_mode,
                inner_lr=args.inner_lr, inner_steps=args.inner_steps,
                inner_mode=args.inner_mode,
                alpha=args.alpha, delta=args.delta,
                beta=args.beta, gamma=args.gamma, lam=args.lam,
            )

            # Evaluate (no gradient needed)
            with torch.no_grad():
                logits = model(imgs)
                _, pred = logits.topk(1, dim=1)
                top1_m.update(
                    pred.eq(labels.unsqueeze(1)).float().sum().item() / imgs.size(0) * 100.0,
                    imgs.size(0),
                )

            # Reset conv1
            model.conv1.weight.data.copy_(original_w)

        per_t_top1[t_name] = top1_m.avg
        _logger.info(f'  Val TTA [{t_name}] (SL={severity}): Top1={top1_m.avg:.2f}%')
        del loader, aug_dataset
        gc.collect()

    mean_top1 = sum(per_t_top1.values()) / num_transforms
    _logger.info(f'  Val TTA Mean: Top1={mean_top1:.2f}%')
    aug_classifier.train()
    return {'mean_top1': mean_top1, 'per_transform_top1': per_t_top1}


def _tta_inner_loop_simple(
    images, model, aug_classifier, fsc_centroids, stem_mode,
    inner_lr, inner_steps, inner_mode,
    alpha, delta, beta, gamma, lam,
):
    """Non-differentiable TTA inner loop (in-place conv1 update).

    Mirrors ``meta_inner_loop`` but without computation graph preservation.
    Supports the same three modes for step count and lr scaling:
      - ``fomaml``: single step at constant lr.
      - ``truncated``: K steps at constant lr.
      - ``truncated_scaled``: K steps with linearly ramping lr.
    """
    device = images.device
    original_w = model.conv1.weight.data.clone()

    for p in model.parameters():
        p.requires_grad = False
    model.conv1.weight.requires_grad = True

    # Pseudo-label → centroid
    with torch.no_grad():
        logits_init = model(images)
        pred_labels = logits_init.argmax(dim=1)
    fsc_ref = fsc_centroids[pred_labels]
    C, S = aug_classifier.channels, aug_classifier.spatial_size
    fsc_ref_spatial = fsc_ref.view(-1, C, S, S)

    with torch.no_grad():
        x0 = model.conv1(images.contiguous())
        if stem_mode in ('conv1_bn1', 'conv1_bn1_act1'):
            x0 = model.bn1(x0)
        if stem_mode == 'conv1_bn1_act1':
            x0 = model.act1(x0)
        x0 = F.adaptive_avg_pool2d(x0, (4, 4))
        g0_flat = x0.flatten(1).detach()

    clean_target = torch.zeros(images.size(0), dtype=torch.long, device=device)

    # Effective step count (fomaml forces K=1)
    K = 1 if inner_mode == 'fomaml' else inner_steps

    for k in range(K):
        step_lr = _inner_step_lr(k, K, inner_lr, inner_mode)

        # Manual zero_grad + forward
        if model.conv1.weight.grad is not None:
            model.conv1.weight.grad.zero_()

        x = model.conv1(images.contiguous())
        if stem_mode in ('conv1_bn1', 'conv1_bn1_act1'):
            x = model.bn1(x)
        if stem_mode == 'conv1_bn1_act1':
            x = model.act1(x)
        g_spatial = F.adaptive_avg_pool2d(x, (4, 4))
        g_flat = g_spatial.flatten(1)

        diff_spatial = g_spatial - fsc_ref_spatial
        z = aug_classifier.project_diff(diff_spatial)
        aug_out, dist_out = aug_classifier(z)

        L_clean = F.cross_entropy(aug_out, clean_target)
        L_dist = dist_out.mean()
        cos_orig = F.cosine_similarity(g_flat, g0_flat, dim=1).mean()
        cos_cent = F.cosine_similarity(g_flat, fsc_ref, dim=1).mean()
        L_trust = (model.conv1.weight - original_w).pow(2).sum()

        loss = (
            alpha * L_clean + delta * L_dist + beta * cos_orig
            - gamma * cos_cent + lam * L_trust
        )
        loss.backward()

        # Manual SGD step with per-step lr (no momentum, matching training)
        with torch.no_grad():
            model.conv1.weight.sub_(step_lr * model.conv1.weight.grad)

    model.conv1.weight.requires_grad = False


# =============================================================================
# Argument Parser
# =============================================================================

config_parser = argparse.ArgumentParser(description='Config', add_help=False)
config_parser.add_argument('-c', '--config', default='', type=str)

parser = argparse.ArgumentParser(description='Phase 2: Meta-Learning TTA with Clean Confidence')

group = parser.add_argument_group('Dataset')
group.add_argument('--data-dir', type=str, required=True)
group.add_argument('--train-split', default='train')
group.add_argument('--val-split', default='val')

group = parser.add_argument_group('Model')
group.add_argument('--model', default='resnet50', type=str)
group.add_argument('--pretrained', action='store_true')
group.add_argument('--initial-checkpoint', default='', type=str)
group.add_argument('--num-classes', type=int, default=1000)
group.add_argument('--img-size', type=int, default=224)

group = parser.add_argument_group('Phase-1 Checkpoint (required)')
group.add_argument('--phase1-checkpoint', type=str, required=True,
                   help='Path to Phase-1 best.pth.tar (auto-detects stem-mode, proj-type, etc.)')
group.add_argument('--fsc-path', type=str, required=True,
                   help='Path to FSC centroid file (e.g. ZOA_FSC/resnet50_FSC_conv1_only.pth)')

group = parser.add_argument_group('Inner loop (TTA simulation)')
group.add_argument('--inner-lr', type=float, default=0.01)
group.add_argument('--inner-steps', type=int, default=5,
                   help='Number of inner adaptation steps K (default: 5). '
                        'With fomaml mode, forced to 1 regardless of this value.')
group.add_argument('--inner-mode', type=str, default='truncated',
                   choices=['fomaml', 'truncated', 'truncated_scaled'],
                   help='Inner loop meta-gradient mode (default: truncated). '
                        'fomaml: single step, create_graph on that step. '
                        'truncated: K steps, create_graph only on last step. '
                        'truncated_scaled: same as truncated + linear lr ramp per step.')
group.add_argument('--alpha', type=float, default=1.0, help='Clean CE weight')
group.add_argument('--delta', type=float, default=0.3, help='Dist minimization weight')
group.add_argument('--beta', type=float, default=0.05, help='Repel corrupted weight')
group.add_argument('--gamma', type=float, default=0.3, help='Attract centroid weight')
group.add_argument('--lam', type=float, default=1e-4, help='Trust region weight')

group = parser.add_argument_group('Augmentation')
group.add_argument('--new-depth', type=int, default=3,
                   help='Maximum augmentation depth (default: 3). '
                        'With curriculum this is the final depth; without it is fixed.')
group.add_argument('--min-sl', type=float, default=0.3,
                   help='Minimum severity level (default: 0.3). '
                        'With SL curriculum this is the final min-SL.')
group.add_argument('--max-sl', type=float, default=1.0,
                   help='Maximum severity level (default: 1.0). '
                        'With SL curriculum this is the final max-SL.')
group.add_argument('--val-severity', type=float, default=1.0)

group = parser.add_argument_group('Curriculum')
group.add_argument('--curriculum', action='store_true', default=False,
                   help='Enable curriculum learning: gradually increase augmentation difficulty. '
                        'Depth ramps 1->2->...->new-depth over equal epoch phases; '
                        'SL range widens from easy to full.')
group.add_argument('--cur-min-sl-start', type=float, default=0.1,
                   help='Curriculum: initial min-SL at epoch 0 (default: 0.1)')
group.add_argument('--cur-max-sl-start', type=float, default=0.5,
                   help='Curriculum: initial max-SL at epoch 0 (default: 0.5)')

group = parser.add_argument_group('Meta-optimizer')
group.add_argument('--opt', default='adamw', type=str)
group.add_argument('--lr', type=float, default=1e-4, help='Meta learning rate')
group.add_argument('--weight-decay', type=float, default=1e-4)
group.add_argument('--momentum', type=float, default=0.9)
group.add_argument('--clip-grad', type=float, default=1.0)

group = parser.add_argument_group('Scheduler')
group.add_argument('--sched', type=str, default='cosine')
group.add_argument('--epochs', type=int, default=50)
group.add_argument('--warmup-epochs', type=int, default=3)
group.add_argument('--warmup-lr', type=float, default=1e-6)
group.add_argument('--min-lr', type=float, default=1e-6)

group = parser.add_argument_group('Misc')
group.add_argument('-b', '--batch-size', type=int, default=32)
group.add_argument('-vb', '--validation-batch-size', type=int, default=None)
group.add_argument('-j', '--workers', type=int, default=None)
group.add_argument('--device', default='cuda', type=str)
group.add_argument('--seed', type=int, default=42)
group.add_argument('--pin-mem', action='store_true')
group.add_argument('--log-interval', type=int, default=50)
group.add_argument('--val-interval', type=int, default=1)
group.add_argument('--output', default='', type=str)
group.add_argument('--experiment', default='', type=str)
group.add_argument('--checkpoint-hist', type=int, default=3)
group.add_argument('--device-modules', default=None, type=str, nargs='+')
group.add_argument('--log-wandb', action='store_true')
group.add_argument('--wandb-project', default='phase2-meta-tta', type=str)


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

    if args.workers is None:
        args.workers = min(4, max(1, os.cpu_count() // 4))

    if args.device_modules:
        for mod in args.device_modules:
            importlib.import_module(mod)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    device = torch.device(args.device)
    utils.random_seed(args.seed, 0)

    # =========================================================================
    # Load Phase-1 checkpoint → auto-detect settings
    # =========================================================================
    _logger.info(f'Loading Phase-1 checkpoint: {args.phase1_checkpoint}')
    p1_ckpt = torch.load(args.phase1_checkpoint, map_location='cpu', weights_only=False)
    p1_args = p1_ckpt['args']
    stem_mode = p1_args['stem_mode']
    proj_type = p1_args['proj_type']
    hidden_dims = p1_args['hidden_dims']
    dropout = p1_args.get('dropout', 0.1)
    _logger.info(f'Phase-1 config: stem_mode={stem_mode}, proj_type={proj_type}, '
                 f'hidden_dims={hidden_dims}, metric={p1_ckpt.get("metric", "?"):.2f}%')

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
    # Load FSC centroids
    # =========================================================================
    _logger.info(f'Loading FSC: {args.fsc_path}')
    fsc_data = torch.load(args.fsc_path, map_location='cpu', weights_only=False)
    fsc_centroids = fsc_data['centroids'].to(device)
    _logger.info(f'FSC: {fsc_centroids.shape}')

    # =========================================================================
    # Aug classifier (from Phase-1, meta-updated)
    # =========================================================================
    with torch.no_grad():
        dummy = torch.randn(1, 3, args.img_size, args.img_size, device=device)
        x = backbone.conv1(dummy)
        if stem_mode in ('conv1_bn1', 'conv1_bn1_act1'):
            x = backbone.bn1(x)
        if stem_mode == 'conv1_bn1_act1':
            x = backbone.act1(x)
        x = F.adaptive_avg_pool2d(x, (4, 4))
        channels, spatial = x.shape[1], x.shape[2]

    num_transforms = get_augmix_sl_num_transforms(version=2)
    transform_names = get_augmix_sl_transform_names(version=2)

    aug_classifier = DistanceAwareAugClassifier(
        channels=channels, spatial_size=spatial,
        num_transforms=num_transforms, proj_type=proj_type,
        hidden_dims=hidden_dims, dropout=dropout,
    ).to(device)
    aug_classifier.load_state_dict(p1_ckpt['aug_classifier'])
    _logger.info(f'Loaded Phase-1 weights into aug_classifier '
                 f'({sum(p.numel() for p in aug_classifier.parameters()):,} params)')

    # =========================================================================
    # Dataset (augmented training data)
    # =========================================================================
    data_config = resolve_data_config(vars(args), model=backbone, verbose=True)
    from torchvision import transforms
    from torchvision.datasets import ImageFolder
    from timm.data import create_augmix_sl_transform

    base_transform = transforms.Compose([
        transforms.Resize(int(args.img_size / data_config['crop_pct'])),
        transforms.CenterCrop(args.img_size),
    ])
    final_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=data_config['mean'], std=data_config['std']),
    ])

    augmix_ops = augmix_sl_ops_v2()
    augmix_sl_transform = create_augmix_sl_transform(
        max_depth=args.new_depth, min_sl=args.min_sl, max_sl=args.max_sl,
        version=2, normalize_labels=True,
    )

    train_dir = os.path.join(args.data_dir, args.train_split)
    raw_train = ImageFolder(train_dir)
    _logger.info(f'Train: {len(raw_train)} images from {train_dir}')

    from train_phase1 import PairedAugDataset as _unused
    # Use standard AugMixSL dataset (not paired) for meta-training
    from train_meta import AugMixSLDataset, collate_aug_labels
    dataset_train = AugMixSLDataset(
        raw_train, augmix_sl_transform,
        base_transform=base_transform, final_transform=final_transform,
    )
    loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=args.pin_mem,
        collate_fn=collate_aug_labels, drop_last=True,
        persistent_workers=False, prefetch_factor=2 if args.workers > 0 else None,
    )

    val_dir = os.path.join(args.data_dir, args.val_split)
    raw_val = ImageFolder(val_dir)

    # =========================================================================
    # Meta-optimizer + scheduler
    # =========================================================================
    from timm.optim import create_optimizer_v2
    meta_optimizer = create_optimizer_v2(
        aug_classifier, opt=args.opt, lr=args.lr,
        weight_decay=args.weight_decay, momentum=args.momentum,
    )
    from timm.scheduler import create_scheduler_v2
    lr_scheduler, num_epochs = create_scheduler_v2(
        meta_optimizer, sched=args.sched, num_epochs=args.epochs,
        warmup_epochs=args.warmup_epochs, warmup_lr=args.warmup_lr,
        min_lr=args.min_lr,
    )

    # =========================================================================
    # Output: --output is base dir, --experiment is subfolder
    # =========================================================================
    exp_name = args.experiment or f'phase2_{args.model}_{stem_mode}_proj{proj_type}'
    base_dir = Path(args.output) if args.output else Path('./output')
    output_dir = base_dir / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'args.yaml', 'w') as f:
        f.write(args_text)

    results_path = output_dir / 'phase2_results.txt'
    header = [
        'epoch', 'L_outer', 'train_top1', 'train_top5',
        'grad_norm', 'w_delta', 'w_delta_ratio', 'lr',
    ]
    if args.curriculum:
        header += ['cur_depth', 'cur_min_sl', 'cur_max_sl']
    header += [f'val_{n}' for n in transform_names] + ['val_mean_top1', 'best']
    with open(results_path, 'w') as f:
        f.write('\t'.join(header) + '\n')

    if args.log_wandb and has_wandb:
        wandb.init(project=args.wandb_project, name=exp_name, config=args)

    # =========================================================================
    # Training loop
    # =========================================================================
    best_metric = None
    best_epoch = None
    top_checkpoints = []

    effective_K = 1 if args.inner_mode == 'fomaml' else args.inner_steps
    cur_tag = 'curriculum' if args.curriculum else 'fixed'
    _logger.info(
        f'Phase 2 ({cur_tag}): {num_epochs} epochs, stem={stem_mode}, proj={proj_type}, '
        f'inner_mode={args.inner_mode}, inner_lr={args.inner_lr}, K={effective_K}, '
        f'α={args.alpha}, δ={args.delta}, β={args.beta}, γ={args.gamma}, λ={args.lam}'
    )
    if args.curriculum:
        _logger.info(
            f'  Curriculum: depth=1->{args.new_depth}, '
            f'SL=[{args.cur_min_sl_start},{args.cur_max_sl_start}]'
            f'->[{args.min_sl},{args.max_sl}]'
        )
    else:
        _logger.info(
            f'  Augmentation: depth={args.new_depth}, '
            f'SL=[{args.min_sl},{args.max_sl}]'
        )
    if args.inner_mode == 'truncated_scaled':
        _logger.info(
            f'  LR scaling: step 0 lr={args.inner_lr / effective_K:.4f} '
            f'-> step {effective_K - 1} lr={args.inner_lr:.4f}'
        )

    try:
        for epoch in range(num_epochs):
            # ---- Curriculum: update augmentation difficulty ----
            if args.curriculum:
                cur_depth = get_curriculum_depth(
                    epoch, num_epochs, max_depth=args.new_depth,
                )
                cur_min_sl, cur_max_sl = get_curriculum_sl_range(
                    epoch, num_epochs,
                    min_sl_start=args.cur_min_sl_start,
                    min_sl_end=args.min_sl,
                    max_sl_start=args.cur_max_sl_start,
                    max_sl_end=args.max_sl,
                )
                # Update transform object in-place (no loader rebuild needed)
                dataset_train.augmix_sl_transform.max_depth = cur_depth
                dataset_train.augmix_sl_transform.min_sl = cur_min_sl
                dataset_train.augmix_sl_transform.max_sl = cur_max_sl
                _logger.info(
                    f'Curriculum epoch {epoch}: '
                    f'depth={cur_depth}, SL=[{cur_min_sl:.2f}, {cur_max_sl:.2f}]'
                )
            else:
                cur_depth = args.new_depth
                cur_min_sl = args.min_sl
                cur_max_sl = args.max_sl

            train_metrics = train_one_epoch(
                epoch, backbone, aug_classifier, fsc_centroids, loader_train,
                meta_optimizer, args, device, stem_mode,
            )
            if lr_scheduler is not None:
                lr_scheduler.step(epoch + 1)

            # ---- Validation TTA ----
            val_metrics = None
            if (epoch + 1) % args.val_interval == 0 or epoch == num_epochs - 1:
                val_metrics = validate_tta(
                    backbone, aug_classifier, fsc_centroids, raw_val,
                    augmix_ops, transform_names, num_transforms,
                    severity=args.val_severity,
                    base_transform=base_transform, final_transform=final_transform,
                    args=args, device=device, stem_mode=stem_mode,
                )

            # ---- Checkpoint ----
            ckpt_data = {
                'epoch': epoch,
                'aug_classifier': aug_classifier.state_dict(),
                'optimizer': meta_optimizer.state_dict(),
                'metric': val_metrics['mean_top1'] if val_metrics else 0.0,
                'phase1_args': p1_args,
                'args': args.__dict__,
            }
            torch.save(ckpt_data, output_dir / 'last.pth.tar')

            is_best = ''
            if val_metrics:
                current = val_metrics['mean_top1']
                if best_metric is None or current > best_metric:
                    best_metric = current
                    best_epoch = epoch
                    torch.save(ckpt_data, output_dir / 'best.pth.tar')
                    is_best = '*'

                ckpt_path = str(output_dir / f'checkpoint-{epoch}.pth.tar')
                if len(top_checkpoints) < args.checkpoint_hist:
                    torch.save(ckpt_data, ckpt_path)
                    top_checkpoints.append((current, epoch, ckpt_path))
                    top_checkpoints.sort(key=lambda x: x[0], reverse=True)
                else:
                    wm, we, wp = top_checkpoints[-1]
                    if current > wm:
                        if os.path.exists(wp):
                            os.remove(wp)
                        torch.save(ckpt_data, ckpt_path)
                        top_checkpoints[-1] = (current, epoch, ckpt_path)
                        top_checkpoints.sort(key=lambda x: x[0], reverse=True)

            # ---- Log ----
            lr = meta_optimizer.param_groups[0]['lr']
            row = [
                str(epoch), f'{train_metrics["loss"]:.4f}',
                f'{train_metrics["top1"]:.2f}', f'{train_metrics["top5"]:.2f}',
                f'{train_metrics["grad_norm"]:.4f}',
                f'{train_metrics["w_delta"]:.6f}',
                f'{train_metrics["w_delta_ratio"]:.6f}',
                f'{lr:.2e}',
            ]
            if args.curriculum:
                row += [str(cur_depth), f'{cur_min_sl:.2f}', f'{cur_max_sl:.2f}']
            if val_metrics:
                for tn in transform_names:
                    row.append(f'{val_metrics["per_transform_top1"].get(tn, 0):.2f}')
                row.append(f'{val_metrics["mean_top1"]:.2f}')
            else:
                row += [''] * (num_transforms + 1)
            row.append(is_best)
            with open(results_path, 'a') as f:
                f.write('\t'.join(row) + '\n')

            if args.log_wandb and has_wandb:
                log_dict = {'epoch': epoch, 'lr': lr, **train_metrics}
                if args.curriculum:
                    log_dict.update({
                        'cur_depth': cur_depth,
                        'cur_min_sl': cur_min_sl,
                        'cur_max_sl': cur_max_sl,
                    })
                if val_metrics:
                    log_dict['val_mean_top1'] = val_metrics['mean_top1']
                    for tn in transform_names:
                        log_dict[f'val_{tn}'] = val_metrics['per_transform_top1'].get(tn, 0)
                wandb.log(log_dict)

    except KeyboardInterrupt:
        _logger.info('Interrupted.')

    if best_metric is not None:
        _logger.info(f'*** Best val TTA mean top1: {best_metric:.2f}% (epoch {best_epoch})')
        for m, e, p in top_checkpoints:
            _logger.info(f'  epoch {e}: {m:.2f}% — {p}')


if __name__ == '__main__':
    main()

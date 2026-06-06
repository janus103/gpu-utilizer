#!/usr/bin/env python3
"""Meta-Learning Training Script with Curriculum (train_meta_cur.py).

Variant of train_meta.py that adds:
  - Auxiliary augmentation prediction loss (L_aug) in the outer loop for direct
    aug_classifier supervision.
  - Curriculum: aux_weight cosine decay (high early -> low later) and
    new_depth step schedule (1 -> 2 -> 3 over 25%/50% epoch boundaries).

Usage:
    python train_meta_cur.py --data-dir /path/to/imagenet --model resnet50 \\
        --fsc-path ./FSC/resnet50_FSC_stem.pth --initial-checkpoint ...
"""
import argparse
import copy
import gc
import importlib
import json
import logging
import math
import os
import time
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime
from typing import Dict, List, Tuple
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils
import yaml

from timm import utils
from timm.data import (
    create_dataset,
    create_loader,
    resolve_data_config,
    create_augmix_sl_transform,
    create_augmix_sl_validation_transforms,
    get_augmix_sl_num_transforms,
    get_augmix_sl_transform_names,
    AUGMIX_SL_V2_NUM_TRANSFORMS,
)
from timm.layers import set_fast_norm
from timm.models import create_model, safe_model_name, resume_checkpoint, load_checkpoint
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler_v2, scheduler_kwargs
from timm.utils import NativeScaler


try:
    import wandb
    has_wandb = True
except ImportError:
    has_wandb = False

has_compile = hasattr(torch, 'compile')


_logger = logging.getLogger('train_augmix_stem')


# =============================================================================
# Curriculum Helpers
# =============================================================================

def get_curriculum_aux_weight(
    epoch: int,
    total_epochs: int,
    aux_max: float = 5.0,
    aux_min: float = 0.1,
) -> float:
    """Cosine decay for aux_weight: high early (aug supervision) -> low later (meta-learning)."""
    progress = epoch / max(total_epochs - 1, 1)
    return aux_min + 0.5 * (aux_max - aux_min) * (1 + math.cos(math.pi * progress))


def get_curriculum_depth(epoch: int, total_epochs: int) -> int:
    """Step curriculum for new_depth: 1 (0-25%%) -> 2 (25-50%%) -> 3 (50-100%%)."""
    if total_epochs <= 0:
        return 3
    progress = epoch / total_epochs
    if progress < 0.25:
        return 1
    elif progress < 0.5:
        return 2
    else:
        return 3


# =============================================================================
# Stem Feature Extractor (from compute_fsc_stem.py)
# =============================================================================

class StemFeatureExtractor(nn.Module):
    """Extract features from conv1 ONLY (no bn1, no act1).

    Conv1-only stem produces raw, unnormalized features that are maximally
    sensitive to input perturbations — ideal for augmentation detection.

    For ResNet50 with 224x224 input:
    - conv1 (7x7, stride=2, padding=3): 3 -> 64 channels, 112x112 spatial
    - AdaptiveAvgPool2d(4, 4): 64 channels, 4x4 spatial
    - Flatten: 64 * 4 * 4 = 1024-dim vector
    """

    def __init__(self, model):
        super().__init__()
        if hasattr(model, 'conv1'):
            self.conv1 = model.conv1
        else:
            raise ValueError("Model does not have conv1 layer")

        # Pooling only — no bn1, no act1
        self.pool = nn.AdaptiveAvgPool2d((4, 4))

    def forward(self, x):
        x = x.contiguous()
        x = self.conv1(x)
        # No bn1, no act1 — raw conv1 output
        x = self.pool(x)
        x = x.flatten(1)
        return x


def meta_stem_forward(
    images: torch.Tensor,
    conv1_weight: torch.Tensor,
    conv1_bias,
    conv1_module: nn.Module,
) -> torch.Tensor:
    """Functional conv1-only forward for differentiable meta-learning inner loop.

    Uses ``F.conv2d`` with explicit weight/bias so that ``create_graph=True``
    gradients flow through the adapted parameters.

    Args:
        images: Input images ``[B, 3, H, W]``.
        conv1_weight: Conv1 weight tensor (possibly adapted).
        conv1_bias: Conv1 bias tensor or None.
        conv1_module: Original conv1 module (to read stride, padding, etc.).

    Returns:
        stem_features: ``[B, 1024]`` (64 * 4 * 4 for ResNet50).
    """
    x = images.contiguous()
    x = F.conv2d(
        x,
        conv1_weight,
        bias=conv1_bias,
        stride=conv1_module.stride,
        padding=conv1_module.padding,
        dilation=conv1_module.dilation,
        groups=conv1_module.groups,
    )
    # No bn1, no act1 — raw conv1 output
    x = F.adaptive_avg_pool2d(x, (4, 4))
    return x.flatten(1)


# =============================================================================
# Meta-Learning Inner Loop (Step 3: simple version, no meta-gradient)
# =============================================================================

def meta_inner_loop_simple(
    images: torch.Tensor,
    model: nn.Module,
    aug_classifier: nn.Module,
    fsc_centroids: torch.Tensor,
    fsc_diff_mode: str,
    inner_lr: float,
    inner_steps: int,
    inner_alpha: float = 1.0,
    inner_beta: float = 0.1,
    inner_gamma: float = 0.5,
    inner_momentum: float = 0.0,
) -> dict:
    """Simulate TTA: adapt conv1 using combined contrastive inner loss.

    This is the **simple** (non-differentiable) version for Step 3 verification
    and validation TTA. It updates conv1.weight in-place using standard backward
    — no computation graph is preserved for meta-gradients.

    Inner loss combines three objectives (matching the differentiable version):
      (1) ``-alpha * entropy``: erase augmentation signal.
      (2) ``+beta * cos_sim(adapted, original)``: repel from corrupted features.
      (3) ``-gamma * cos_sim(adapted, centroid)``: attract toward clean centroid.

    Args:
        images: Input batch ``[B, 3, H, W]`` (already on device).
        model: Full pretrained model (conv1 will be modified in-place).
        aug_classifier: Frozen aug classifier (provides entropy signal).
        fsc_centroids: FSC centroids ``[num_classes, 1024]``.
        fsc_diff_mode: ``'subtract'`` or ``'orthogonal'``.
        inner_lr: Learning rate for conv1 adaptation.
        inner_steps: Number of gradient steps (K).
        inner_alpha: Weight for entropy maximization term (default: 1.0).
        inner_beta: Weight for repel-corrupted term (default: 0.1).
        inner_gamma: Weight for attract-centroid term (default: 0.5).
        inner_momentum: SGD momentum for conv1 adaptation (default: 0.0).

    Returns:
        Dictionary with diagnostics:
            ``entropy_before``, ``entropy_after``, ``num_updates``,
            ``param_delta_norm``, ``fsc_diff_norm_before``, ``fsc_diff_norm_after``.
    """
    device = images.device

    # ---- Save original conv1 state for diagnostics & reset ----
    original_weight = model.conv1.weight.data.clone()

    # ---- Freeze everything, unfreeze conv1 only ----
    for p in model.parameters():
        p.requires_grad = False
    model.conv1.weight.requires_grad = True

    # ---- FSC mean (fsc_mode=mean, fixed) ----
    fsc_mean = fsc_centroids.mean(dim=0, keepdim=True).expand(images.size(0), -1)

    # ---- Original (corrupted) stem features for contrastive repel term ----
    with torch.no_grad():
        stem_feat_orig = model.conv1(images.contiguous())
        stem_feat_orig = F.adaptive_avg_pool2d(stem_feat_orig, (4, 4)).flatten(1)
        stem_feat_orig = stem_feat_orig.detach()

    # ---- Optimizer for conv1 only ----
    optimizer = torch.optim.SGD([model.conv1.weight], lr=inner_lr, momentum=inner_momentum)

    # ---- Diagnostics: before TTA ----
    with torch.no_grad():
        fsc_diff_0 = compute_fsc_diff(stem_feat_orig, fsc_mean, mode=fsc_diff_mode)
        output_0 = aug_classifier(fsc_diff_0)
        probs_0 = output_0 if aug_classifier.use_softmax else F.softmax(output_0, dim=1)
        entropy_before = -(probs_0 * torch.log(probs_0 + 1e-10)).sum(1).mean().item()
        fsc_diff_norm_before = fsc_diff_0.norm(dim=1).mean().item()

    # ---- Inner loop: K gradient steps ----
    num_updates = 0
    for _k in range(inner_steps):
        optimizer.zero_grad()

        # Forward: conv1 → pool → flatten → FSC_diff → aug_classifier
        stem_feat = model.conv1(images.contiguous())
        stem_feat = F.adaptive_avg_pool2d(stem_feat, (4, 4)).flatten(1)
        fsc_diff = compute_fsc_diff(stem_feat, fsc_mean, mode=fsc_diff_mode)
        aug_output = aug_classifier(fsc_diff)

        # (1) Entropy maximization: erase augmentation signal
        if aug_classifier.use_softmax:
            probs = aug_output
            log_probs = torch.log(probs + 1e-10)
        else:
            probs = F.softmax(aug_output, dim=1)
            log_probs = F.log_softmax(aug_output, dim=1)
        entropy = -(probs * log_probs).sum(dim=1).mean()
        loss_entropy = -entropy

        # (2) Repel corrupted: push away from original corrupted features
        cos_sim_orig = F.cosine_similarity(stem_feat, stem_feat_orig, dim=1).mean()

        # (3) Attract centroid: pull toward clean centroid direction
        cos_sim_centroid = F.cosine_similarity(stem_feat, fsc_mean, dim=1).mean()

        # Combined inner loss
        loss = (
            inner_alpha * loss_entropy
            + inner_beta * cos_sim_orig
            - inner_gamma * cos_sim_centroid
        )

        loss.backward()
        optimizer.step()
        num_updates += 1

    # ---- Diagnostics: after TTA ----
    with torch.no_grad():
        stem_feat_k = model.conv1(images.contiguous())
        stem_feat_k = F.adaptive_avg_pool2d(stem_feat_k, (4, 4)).flatten(1)
        fsc_diff_k = compute_fsc_diff(stem_feat_k, fsc_mean, mode=fsc_diff_mode)
        output_k = aug_classifier(fsc_diff_k)
        probs_k = output_k if aug_classifier.use_softmax else F.softmax(output_k, dim=1)
        entropy_after = -(probs_k * torch.log(probs_k + 1e-10)).sum(1).mean().item()
        fsc_diff_norm_after = fsc_diff_k.norm(dim=1).mean().item()

    # ---- Parameter change ----
    param_delta_norm = (model.conv1.weight.data - original_weight).norm().item()

    # ---- Freeze conv1 again ----
    model.conv1.weight.requires_grad = False

    return {
        'entropy_before': entropy_before,
        'entropy_after': entropy_after,
        'num_updates': num_updates,
        'param_delta_norm': param_delta_norm,
        'fsc_diff_norm_before': fsc_diff_norm_before,
        'fsc_diff_norm_after': fsc_diff_norm_after,
        'original_weight': original_weight,  # for reset
    }


def reset_conv1(model: nn.Module, original_weight: torch.Tensor) -> None:
    """Restore conv1 to its original (pretrained) weights."""
    model.conv1.weight.data.copy_(original_weight)


# =============================================================================
# Meta-Learning Outer Loop Evaluation (Step 4)
# =============================================================================

def meta_outer_eval(
    images: torch.Tensor,
    class_labels: torch.Tensor,
    model: nn.Module,
) -> dict:
    """Evaluate classification with the current (adapted) conv1.

    Runs the **full model** forward pass: conv1 → bn1 → act1 → backbone → fc.
    Conv1 should already be adapted by the inner loop before calling this.

    Args:
        images: Input batch ``[B, 3, H, W]`` (already on device).
        class_labels: Ground-truth class labels ``[B]``.
        model: Full pretrained model (conv1 may be adapted).

    Returns:
        Dictionary with:
            ``loss`` (CE), ``top1`` (%), ``top5`` (%).
    """
    model.eval()
    with torch.no_grad():
        output = model(images)
        loss = F.cross_entropy(output, class_labels)

        # Top-1 and Top-5 accuracy
        maxk = min(5, output.size(1))
        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        correct = pred.eq(class_labels.unsqueeze(1).expand_as(pred))
        top1 = correct[:, :1].float().sum().item() / images.size(0) * 100.0
        top5 = correct[:, :maxk].float().sum().item() / images.size(0) * 100.0

    return {'loss': loss.item(), 'top1': top1, 'top5': top5}


def meta_step_simple(
    images: torch.Tensor,
    class_labels: torch.Tensor,
    model: nn.Module,
    aug_classifier: nn.Module,
    fsc_centroids: torch.Tensor,
    fsc_diff_mode: str,
    inner_lr: float,
    inner_steps: int,
) -> dict:
    """Run one full meta-step: inner loop TTA + outer loop evaluation.

    Combines Step 3 (inner loop) and Step 4 (outer eval) into one call.
    This is the non-differentiable version for pipeline verification.

    Returns:
        Dictionary with inner loop diagnostics + outer eval metrics:
            - Inner: ``entropy_before``, ``entropy_after``, ``param_delta_norm``, ...
            - Before TTA: ``pretrained_loss``, ``pretrained_top1``, ``pretrained_top5``
            - After TTA:  ``adapted_loss``, ``adapted_top1``, ``adapted_top5``
    """
    # ---- Outer eval BEFORE inner loop (pretrained baseline) ----
    before = meta_outer_eval(images, class_labels, model)

    # ---- Inner loop: adapt conv1 ----
    inner_diag = meta_inner_loop_simple(
        images, model, aug_classifier, fsc_centroids,
        fsc_diff_mode, inner_lr, inner_steps,
    )

    # ---- Outer eval AFTER inner loop (adapted) ----
    after = meta_outer_eval(images, class_labels, model)

    # ---- Reset conv1 to pretrained ----
    reset_conv1(model, inner_diag['original_weight'])

    return {
        # Inner loop diagnostics
        'entropy_before': inner_diag['entropy_before'],
        'entropy_after': inner_diag['entropy_after'],
        'num_updates': inner_diag['num_updates'],
        'param_delta_norm': inner_diag['param_delta_norm'],
        'fsc_diff_norm_before': inner_diag['fsc_diff_norm_before'],
        'fsc_diff_norm_after': inner_diag['fsc_diff_norm_after'],
        # Classification before TTA
        'pretrained_loss': before['loss'],
        'pretrained_top1': before['top1'],
        'pretrained_top5': before['top5'],
        # Classification after TTA
        'adapted_loss': after['loss'],
        'adapted_top1': after['top1'],
        'adapted_top5': after['top5'],
    }


# =============================================================================
# Step 5: Differentiable Meta-Learning (Higher-Order Gradient)
# =============================================================================

def backbone_forward_after_conv1(
    model: nn.Module,
    conv1_output: torch.Tensor,
) -> torch.Tensor:
    """Run the full model forward from after conv1 output to logits.

    All layers (bn1, act1, layer1-4, fc) are frozen but gradient flows
    through their operations because ``conv1_output`` requires grad.

    Args:
        model: Pretrained model (backbone frozen).
        conv1_output: Output of conv1 ``[B, 64, 112, 112]``.

    Returns:
        logits: Classification logits ``[B, num_classes]``.
    """
    x = model.bn1(conv1_output)
    x = model.act1(x)
    x = model.maxpool(x)
    x = model.layer1(x)
    x = model.layer2(x)
    x = model.layer3(x)
    x = model.layer4(x)
    x = model.forward_head(x)
    return x


def meta_inner_loop_differentiable(
    images: torch.Tensor,
    model: nn.Module,
    aug_classifier: nn.Module,
    fsc_centroids: torch.Tensor,
    fsc_diff_mode: str,
    inner_lr: float,
    inner_steps: int,
    inner_alpha: float = 1.0,
    inner_beta: float = 0.1,
    inner_gamma: float = 0.5,
) -> torch.Tensor:
    """Differentiable inner loop: adapt conv1 with computation graph preserved.

    Uses ``torch.autograd.grad(create_graph=True)`` so that the meta-gradient
    from the outer loop can flow through the inner loop to aug_classifier.

    Inner loss combines three complementary objectives:
      (1) Entropy maximization: erase augmentation signal from aug_classifier.
      (2) Repel corrupted: push adapted features away from original (corrupted) features.
      (3) Attract centroid: pull adapted features toward clean centroid direction.

    ``loss_inner = -alpha * entropy + beta * cos_sim(adapted, original)
                   - gamma * cos_sim(adapted, centroid)``

    Key differences from ``meta_inner_loop_simple``:
      - conv1_weight is a **virtual** tensor (not in-place on model).
      - All operations use ``F.conv2d`` (functional) instead of ``model.conv1()``.
      - The returned adapted_weight is still in the autograd graph.

    Args:
        images: Input batch ``[B, 3, H, W]``.
        model: Pretrained model (conv1 weight is cloned, not modified in-place).
        aug_classifier: Aug classifier (requires_grad=True, NOT updated here).
        fsc_centroids: FSC centroids ``[num_classes, 1024]``.
        fsc_diff_mode: ``'subtract'`` or ``'orthogonal'``.
        inner_lr: Inner loop learning rate.
        inner_steps: Number of inner gradient steps (K).
        inner_alpha: Weight for entropy maximization term (default: 1.0).
        inner_beta: Weight for repel-corrupted term (default: 0.1).
        inner_gamma: Weight for attract-centroid term (default: 0.5).

    Returns:
        adapted_weight: Adapted conv1 weight ``[64, 3, 7, 7]`` **in the graph**.
    """
    # Start from pretrained conv1 (detached — becomes a new leaf)
    adapted_weight = model.conv1.weight.detach().clone().requires_grad_(True)
    conv1_bias = model.conv1.bias  # None for ResNet50

    # FSC mean (fixed)
    fsc_mean = fsc_centroids.mean(dim=0, keepdim=True).expand(images.size(0), -1)

    # Compute original (corrupted) stem features for contrastive repel term
    # Detached — fixed reference, no gradient through this path.
    with torch.no_grad():
        stem_feat_orig = meta_stem_forward(
            images, model.conv1.weight, conv1_bias, model.conv1,
        ).detach()

    for _k in range(inner_steps):
        # Functional forward: conv1 → pool → flatten
        stem_feat = meta_stem_forward(
            images, adapted_weight, conv1_bias, model.conv1,
        )

        # FSC_diff → aug_classifier (in graph, aug_classifier params tracked)
        fsc_diff = compute_fsc_diff(stem_feat, fsc_mean, mode=fsc_diff_mode)
        aug_output = aug_classifier(fsc_diff)

        # (1) Entropy maximization: erase augmentation signal
        if aug_classifier.use_softmax:
            probs = aug_output
            log_probs = torch.log(probs + 1e-10)
        else:
            probs = F.softmax(aug_output, dim=1)
            log_probs = F.log_softmax(aug_output, dim=1)
        entropy = -(probs * log_probs).sum(dim=1).mean()
        loss_entropy = -entropy

        # (2) Repel corrupted: push away from original corrupted features
        cos_sim_orig = F.cosine_similarity(stem_feat, stem_feat_orig, dim=1).mean()

        # (3) Attract centroid: pull toward clean centroid direction
        cos_sim_centroid = F.cosine_similarity(stem_feat, fsc_mean, dim=1).mean()

        # Combined inner loss
        loss_inner = (
            inner_alpha * loss_entropy
            + inner_beta * cos_sim_orig
            - inner_gamma * cos_sim_centroid
        )

        # Differentiable gradient w.r.t. adapted_weight
        (grad,) = torch.autograd.grad(
            loss_inner, adapted_weight, create_graph=True,
        )

        # Virtual update (not in-place — stays in graph)
        adapted_weight = adapted_weight - inner_lr * grad

    return adapted_weight


def meta_step_differentiable(
    images: torch.Tensor,
    class_labels: torch.Tensor,
    model: nn.Module,
    aug_classifier: nn.Module,
    fsc_centroids: torch.Tensor,
    fsc_diff_mode: str,
    inner_lr: float,
    inner_steps: int,
    inner_alpha: float = 1.0,
    inner_beta: float = 0.1,
    inner_gamma: float = 0.5,
    aug_labels: torch.Tensor = None,
    aux_weight: float = 0.0,
    criterion: nn.Module = None,
) -> Tuple[torch.Tensor, dict]:
    """One full differentiable meta-step: inner loop + outer loss (+ optional L_aug).

    When aux_weight > 0 and criterion/aug_labels are provided, adds auxiliary
    augmentation prediction loss on original (pre-adaptation) features.

    Returns:
        L_outer: Combined loss (scalar, in graph — call ``.backward()``).
        info: Dict with ``top1``, ``top5``, ``loss``, ``loss_aug`` (if computed).
    """
    model.eval()

    conv1_bias = model.conv1.bias
    fsc_mean = fsc_centroids.mean(dim=0, keepdim=True).expand(images.size(0), -1)

    # ---- Auxiliary aug prediction loss (original features, direct supervision) ----
    L_aug = None
    if aux_weight > 0 and criterion is not None and aug_labels is not None:
        with torch.no_grad():
            stem_feat_orig = meta_stem_forward(
                images, model.conv1.weight, conv1_bias, model.conv1,
            )
            fsc_diff_orig = compute_fsc_diff(stem_feat_orig, fsc_mean, mode=fsc_diff_mode)
        aug_pred = aug_classifier(fsc_diff_orig)
        if aug_classifier.use_softmax:
            # A-mode: output is probability → NLL-style loss: -target * log(pred)
            L_aug = -(aug_labels * torch.log(aug_pred + 1e-10)).sum(dim=1).mean()
        else:
            # B-mode: output is raw logits → criterion handles softmax/sigmoid internally
            L_aug = criterion(aug_pred, aug_labels)

    # ---- Inner loop (differentiable) ----
    adapted_weight = meta_inner_loop_differentiable(
        images, model, aug_classifier, fsc_centroids,
        fsc_diff_mode, inner_lr, inner_steps,
        inner_alpha=inner_alpha,
        inner_beta=inner_beta,
        inner_gamma=inner_gamma,
    )

    # ---- Outer forward: adapted conv1 → full backbone → logits ----
    conv1_output = F.conv2d(
        images.contiguous(),
        adapted_weight,
        bias=model.conv1.bias,
        stride=model.conv1.stride,
        padding=model.conv1.padding,
        dilation=model.conv1.dilation,
        groups=model.conv1.groups,
    )
    logits = backbone_forward_after_conv1(model, conv1_output)

    # ---- Outer loss ----
    L_cls = F.cross_entropy(logits, class_labels)
    if L_aug is not None:
        L_outer = L_cls + aux_weight * L_aug
    else:
        L_outer = L_cls

    # ---- Diagnostics (detached) ----
    with torch.no_grad():
        maxk = min(5, logits.size(1))
        _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
        correct = pred.eq(class_labels.unsqueeze(1).expand_as(pred))
        top1 = correct[:, :1].float().sum().item() / images.size(0) * 100.0
        top5 = correct[:, :maxk].float().sum().item() / images.size(0) * 100.0

    info = {
        'loss': L_outer.item(),
        'top1': top1,
        'top5': top5,
    }
    if L_aug is not None:
        info['loss_aug'] = L_aug.item()

    return L_outer, info


# =============================================================================
# FSC Difference Computation Functions
# =============================================================================

def compute_fsc_diff(features: torch.Tensor, fsc: torch.Tensor, mode: str = 'subtract') -> torch.Tensor:
    """Compute the difference between features and FSC centroids.
    
    Args:
        features: Current image features [batch_size, feature_dim].
        fsc: FSC centroids for the batch [batch_size, feature_dim].
        mode: Difference computation mode:
            - 'subtract': Simple subtraction (features - fsc)
            - 'orthogonal': Orthogonal component of (features - fsc) w.r.t. fsc direction
              This captures the deviation perpendicular to the class prototype.
    
    Returns:
        fsc_diff: Difference tensor [batch_size, feature_dim].
    """
    if mode == 'subtract':
        # Simple subtraction: captures total deviation from centroid
        return features - fsc
    
    elif mode == 'orthogonal':
        # Orthogonal projection: captures deviation perpendicular to FSC direction
        # This isolates the component that is NOT explained by the class prototype
        #
        # diff = features - fsc
        # diff_parallel = proj_{fsc}(diff) = (diff · fsc / ||fsc||^2) * fsc
        # diff_orthogonal = diff - diff_parallel
        #
        # This captures augmentation-induced changes that are orthogonal to
        # the class-specific direction, potentially more informative for
        # augmentation detection.
        
        diff = features - fsc
        
        # Compute ||fsc||^2 for each sample
        fsc_norm_sq = (fsc * fsc).sum(dim=1, keepdim=True) + 1e-8  # [batch, 1]
        
        # Compute projection coefficient: (diff · fsc) / ||fsc||^2
        proj_coeff = (diff * fsc).sum(dim=1, keepdim=True) / fsc_norm_sq  # [batch, 1]
        
        # Compute parallel component
        diff_parallel = proj_coeff * fsc  # [batch, feature_dim]
        
        # Compute orthogonal component
        diff_orthogonal = diff - diff_parallel  # [batch, feature_dim]
        
        return diff_orthogonal
    
    else:
        raise ValueError(f"Unknown fsc_diff_mode: {mode}. Choose 'subtract' or 'orthogonal'.")


# =============================================================================
# Soft Cross Entropy Loss for Probability Distributions
# =============================================================================

class SoftCrossEntropyLoss(nn.Module):
    """Cross Entropy Loss for soft labels (probability distributions).
    
    Computes: -sum(target * log_softmax(pred)) averaged over batch.
    
    This is equivalent to CE = H(target) + KL(target || softmax(pred)),
    where H(target) is the entropy of the target distribution.
    
    Since H(target) is constant w.r.t. model parameters, this produces
    the same gradients as KLDivLoss, but with different loss values.
    
    Args:
        reduction: 'mean' (default) or 'sum'.
    """
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Raw logits [batch_size, num_classes].
            target: Soft labels (probability distribution) [batch_size, num_classes].
                    Should sum to 1 along dim=1.
        
        Returns:
            Cross entropy loss.
        """
        log_probs = F.log_softmax(pred, dim=1)
        loss = -(target * log_probs).sum(dim=1)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


# =============================================================================
# Augmentation Classifier Module
# =============================================================================

class AugClassifier(nn.Module):
    """Classifier that predicts augmentation type probabilities from FSC_diff.

    Input: FSC_diff tensor of shape [batch_size, feature_dim]
    Output: Predictions of shape [batch_size, num_transforms]

    Output mode (mutually exclusive):
      - ``use_softmax=True``  (A): output is a probability distribution (sums to 1).
      - ``use_softmax=False`` (B, default): output is raw logits.
      - ``use_sigmoid=True``  (legacy): output has independent sigmoid per dim.

    Args:
        feature_dim: Dimension of input features (1024 for stem features).
        num_transforms: Number of transform types to predict (7 for V2 policy).
        hidden_dims: List of hidden layer dimensions.
        dropout: Dropout probability.
        use_sigmoid: If True, apply sigmoid to output (legacy, for BCE loss).
        use_softmax: If True, apply softmax to output (probability distribution).
    """

    def __init__(
        self,
        feature_dim: int,
        num_transforms: int = AUGMIX_SL_V2_NUM_TRANSFORMS,
        hidden_dims: list = None,
        dropout: float = 0.1,
        use_sigmoid: bool = False,
        use_softmax: bool = False,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [512, 256]

        self.feature_dim = feature_dim
        self.num_transforms = num_transforms
        self.use_sigmoid = use_sigmoid
        self.use_softmax = use_softmax

        # Build MLP
        layers = []
        in_dim = feature_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(in_dim, num_transforms))

        self.mlp = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: FSC_diff tensor of shape [batch_size, feature_dim].

        Returns:
            Predictions of shape [batch_size, num_transforms].
            If ``use_softmax``, output sums to 1 along dim=1.
        """
        out = self.mlp(x)
        if self.use_softmax:
            out = F.softmax(out, dim=1)
        elif self.use_sigmoid:
            out = torch.sigmoid(out)
        return out


# =============================================================================
# Dataset Wrapper for Augmentation Labels
# =============================================================================

class AugMixSLDataset(torch.utils.data.Dataset):
    """Dataset wrapper that applies AugMixSL augmentation and returns labels.
    
    This wrapper:
    1. Loads the original image and label
    2. Applies the base transforms (resize, crop, etc.)
    3. Applies AugMixSL augmentation
    4. Returns (image, original_label, aug_labels)
    
    Args:
        dataset: Original dataset (e.g., ImageFolder).
        augmix_sl_transform: AugMixSLAugment instance.
        base_transform: Base transforms to apply before augmentation (resize, crop).
        final_transform: Final transforms (ToTensor, Normalize).
    """
    
    def __init__(
        self,
        dataset,
        augmix_sl_transform,
        base_transform=None,
        final_transform=None,
    ):
        self.dataset = dataset
        self.augmix_sl_transform = augmix_sl_transform
        self.base_transform = base_transform
        self.final_transform = final_transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # Get original image and label
        img, label = self.dataset[idx]
        
        # Apply base transform (resize, crop) if provided
        if self.base_transform is not None:
            img = self.base_transform(img)
        
        # Apply AugMixSL augmentation - returns (augmented_img, aug_labels)
        img, aug_labels = self.augmix_sl_transform(img)
        
        # Apply final transform (ToTensor, Normalize) if provided
        if self.final_transform is not None:
            img = self.final_transform(img)
        
        # Convert aug_labels to tensor
        aug_labels = torch.from_numpy(aug_labels).float()
        
        return img, label, aug_labels


class AugMixSLValidationDataset(torch.utils.data.Dataset):
    """Validation dataset with fixed augmentation groups.
    
    Divides the validation set into groups, each receiving a fixed augmentation.
    This ensures consistent evaluation across epochs.
    
    Args:
        dataset: Original validation dataset.
        validation_transforms: List of AugMixSLAugmentFixed instances.
        base_transform: Base transforms to apply before augmentation.
        final_transform: Final transforms (ToTensor, Normalize).
    """
    
    def __init__(
        self,
        dataset,
        validation_transforms,
        base_transform=None,
        final_transform=None,
    ):
        self.dataset = dataset
        self.validation_transforms = validation_transforms
        self.num_groups = len(validation_transforms)
        self.base_transform = base_transform
        self.final_transform = final_transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        
        # Determine which group this sample belongs to
        group_idx = idx % self.num_groups
        transform = self.validation_transforms[group_idx]
        
        if self.base_transform is not None:
            img = self.base_transform(img)
        
        # Apply fixed augmentation
        img, aug_labels = transform(img)
        
        if self.final_transform is not None:
            img = self.final_transform(img)
        
        aug_labels = torch.from_numpy(aug_labels).float()
        
        return img, label, aug_labels


class SingleTransformValidationDataset(torch.utils.data.Dataset):
    """Validation dataset that applies a SINGLE transform with fixed SL to all images.
    
    This is used for per-transform accuracy evaluation.
    Each transform is tested individually across all images.
    
    Args:
        dataset: Original validation dataset.
        transform_op: Single AugMixSLOp instance to apply.
        transform_idx: Index of this transform in the transform list.
        num_transforms: Total number of transforms (for label tensor size).
        severity_level: Fixed severity level to apply (default: 0.5 = average).
        max_sl: Maximum severity level for scaling (SL is scaled by max_sl so max_sl maps to 1.0).
        base_transform: Base transforms to apply before augmentation.
        final_transform: Final transforms (ToTensor, Normalize).
    """
    
    def __init__(
        self,
        dataset,
        transform_op,
        transform_idx: int,
        num_transforms: int,
        severity_level: float = 0.5,
        max_sl: float = 1.0,
        base_transform=None,
        final_transform=None,
    ):
        self.dataset = dataset
        self.transform_op = transform_op
        self.transform_idx = transform_idx
        self.num_transforms = num_transforms
        self.severity_level = severity_level
        self.max_sl = max_sl
        self.base_transform = base_transform
        self.final_transform = final_transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        
        # Apply base transform (resize, crop)
        if self.base_transform is not None:
            img = self.base_transform(img)
        
        # Apply single transform with fixed SL
        img = self.transform_op(img, self.severity_level)
        
        # Create ground truth label tensor
        # Scale SL by max_sl so that max_sl maps to 1.0 (consistent with training)
        scaled_sl = self.severity_level / self.max_sl
        aug_labels = np.zeros(self.num_transforms, dtype=np.float32)
        aug_labels[self.transform_idx] = scaled_sl
        
        # Apply final transform (ToTensor, Normalize)
        if self.final_transform is not None:
            img = self.final_transform(img)
        
        aug_labels = torch.from_numpy(aug_labels).float()
        
        return img, label, aug_labels


# =============================================================================
# Custom Collate Function
# =============================================================================

def collate_aug_labels(batch):
    """Custom collate function for augmentation label dataset."""
    images = []
    labels = []
    aug_labels = []
    
    for img, label, aug_label in batch:
        images.append(img)
        labels.append(label)
        aug_labels.append(aug_label)
    
    images = torch.stack(images, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)
    aug_labels = torch.stack(aug_labels, dim=0)
    
    return images, labels, aug_labels


# =============================================================================
# Argument Parser
# =============================================================================

config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')


parser = argparse.ArgumentParser(description='Augmentation Prediction Training (Stem Features + V2 Policy)')

# Dataset parameters
group = parser.add_argument_group('Dataset parameters')
parser.add_argument('data', nargs='?', metavar='DIR', const=None,
                    help='path to dataset (positional is *deprecated*, use --data-dir)')
group.add_argument('--data-dir', metavar='DIR',
                    help='path to dataset (root dir)')
group.add_argument('--dataset', metavar='NAME', default='',
                    help='dataset type + name ("<type>/<name>") (default: ImageFolder)')
group.add_argument('--train-split', metavar='NAME', default='train',
                   help='dataset train split (default: train)')
group.add_argument('--val-split', metavar='NAME', default='validation',
                   help='dataset validation split (default: validation)')
group.add_argument('--class-map', default='', type=str, metavar='FILENAME',
                   help='path to class to idx mapping file (default: "")')

# Model parameters
group = parser.add_argument_group('Model parameters')
group.add_argument('--model', default='resnet50', type=str, metavar='MODEL',
                   help='Name of backbone model (default: "resnet50")')
group.add_argument('--pretrained', action='store_true', default=False,
                   help='Start with pretrained version of specified network')
group.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
                   help='Load this checkpoint into model after initialization')
group.add_argument('--resume', default='', type=str, metavar='PATH',
                   help='Resume full model and optimizer state from checkpoint')
group.add_argument('--num-classes', type=int, default=1000, metavar='N',
                   help='number of label classes (default: 1000 for ImageNet)')
group.add_argument('--img-size', type=int, default=224, metavar='N',
                   help='Image size (default: 224)')
group.add_argument('-b', '--batch-size', type=int, default=128, metavar='N',
                   help='Input batch size for training (default: 128)')
group.add_argument('-vb', '--validation-batch-size', type=int, default=None, metavar='N',
                   help='Validation batch size override')

# Augmentation Prediction Task parameters
group = parser.add_argument_group('Augmentation Prediction parameters')
group.add_argument('--fsc-path', type=str, required=True,
                   help='Path to Stem FSC file (e.g., ./FSC/resnet50_FSC_stem.pth)')
group.add_argument('--new-depth', type=int, default=3,
                   help='Maximum number of transforms to apply per image (k value, default: 3)')
group.add_argument('--sl-loss-type', type=str, default='mse', choices=['mse', 'bce', 'kldiv', 'ce'],
                   help='Loss type for SL prediction: "mse" for regression, "bce" for multi-label probability, '
                        '"kldiv" for KL divergence, "ce" for cross entropy '
                        '(kldiv/ce recommended with --normalize-aug-labels) (default: mse)')
group.add_argument('--aug-classifier-hidden', type=int, nargs='+', default=[512, 256],
                   help='Hidden layer dimensions for aug classifier (default: 512 256)')
group.add_argument('--aug-classifier-dropout', type=float, default=0.1,
                   help='Dropout rate for aug classifier (default: 0.1)')
group.add_argument('--aug-classifier-softmax', action='store_true', default=False,
                   help='Apply softmax to aug classifier output (A-mode: probability distribution). '
                        'Default is B-mode (raw logits).')
group.add_argument('--min-sl', type=float, default=0.1,
                   help='Minimum severity level for augmentation (default: 0.1)')
group.add_argument('--max-sl', type=float, default=1.0,
                   help='Maximum severity level for augmentation (default: 1.0)')
group.add_argument('--normalize-aug-labels', action='store_true', default=False,
                   help='Normalize aug_labels to sum to 1 (probability distribution). '
                        'Useful for cross-entropy style learning where labels represent softmax probabilities.')
group.add_argument('--val-groups', type=int, default=6,
                   help='Number of validation groups with different fixed augmentations (default: 6)')
group.add_argument('--fsc-diff-mode', type=str, default='subtract', choices=['subtract', 'orthogonal'],
                   help='FSC difference computation mode: "subtract" for simple subtraction, '
                        '"orthogonal" for orthogonal projection (default: subtract)')

# Meta-learning parameters
group = parser.add_argument_group('Meta-learning parameters')
group.add_argument('--inner-lr', type=float, default=0.01,
                   help='Inner loop (TTA simulation) learning rate for conv1 (default: 0.01)')
group.add_argument('--inner-steps', type=int, default=1,
                   help='Inner loop gradient steps K (default: 1). '
                        'K=1 recommended to start; FOMAML=full MAML when K=1.')
group.add_argument('--inner-alpha', type=float, default=1.0,
                   help='Inner loss weight for entropy maximization (erase aug signal) (default: 1.0)')
group.add_argument('--inner-beta', type=float, default=0.1,
                   help='Inner loss weight for repel-corrupted (push away from original features) (default: 0.1)')
group.add_argument('--inner-gamma', type=float, default=0.5,
                   help='Inner loss weight for attract-centroid (pull toward clean centroid) (default: 0.5)')
group.add_argument('--inner-momentum', type=float, default=0.0,
                   help='SGD momentum for inner loop conv1 adaptation (default: 0.0)')
group.add_argument('--aux-weight-max', type=float, default=5.0,
                   help='Curriculum: initial aux loss weight (cosine decay to aux-weight-min) (default: 5.0)')
group.add_argument('--aux-weight-min', type=float, default=0.1,
                   help='Curriculum: final aux loss weight (default: 0.1). Set both to 0 to disable L_aug.')
group.add_argument('--no-depth-curriculum', action='store_true',
                   help='Disable depth curriculum; use fixed --new-depth instead of 1->2->3 schedule.')

# Device & distributed
group = parser.add_argument_group('Device parameters')
group.add_argument('--device', default='cuda', type=str,
                    help="Device (accelerator) to use.")
group.add_argument('--amp', action='store_true', default=False,
                   help='use AMP for mixed precision training')
group.add_argument('--amp-dtype', default='float16', type=str,
                   help='lower precision AMP dtype (default: float16)')
group.add_argument("--local_rank", default=0, type=int)
group.add_argument('--device-modules', default=None, type=str, nargs='+',
                    help="Python imports for device backend modules.")

# Optimizer parameters
group = parser.add_argument_group('Optimizer parameters')
group.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                   help='Optimizer (default: "adamw")')
group.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                   help='learning rate (default: 1e-3)')
group.add_argument('--weight-decay', type=float, default=1e-4,
                   help='weight decay (default: 1e-4)')
group.add_argument('--momentum', type=float, default=0.9, metavar='M',
                   help='Optimizer momentum (default: 0.9)')
group.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                   help='Clip gradient norm (default: None)')

# Learning rate schedule parameters
group = parser.add_argument_group('Learning rate schedule parameters')
group.add_argument('--sched', type=str, default='cosine', metavar='SCHEDULER',
                   help='LR scheduler (default: "cosine"')
group.add_argument('--epochs', type=int, default=100, metavar='N',
                   help='number of epochs to train (default: 100)')
group.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                   help='epochs to warmup LR')
group.add_argument('--warmup-lr', type=float, default=1e-5, metavar='LR',
                   help='warmup learning rate (default: 1e-5)')
group.add_argument('--min-lr', type=float, default=1e-6, metavar='LR',
                   help='lower lr bound for cyclic schedulers (default: 1e-6)')
group.add_argument('--decay-rate', type=float, default=0.1, metavar='RATE',
                   help='LR decay rate (default: 0.1)')

# Misc
group = parser.add_argument_group('Miscellaneous parameters')
group.add_argument('--seed', type=int, default=42, metavar='S',
                   help='random seed (default: 42)')
group.add_argument('--log-interval', type=int, default=50, metavar='N',
                   help='how many batches to wait before logging training status')
group.add_argument('--val-interval', type=int, default=1, metavar='N',
                   help='how many epochs between validation')
group.add_argument('--checkpoint-hist', type=int, default=10, metavar='N',
                   help='number of checkpoints to keep (default: 10)')
group.add_argument('-j', '--workers', type=int, default=None, metavar='N',
                   help='how many training processes to use (default: half of CPU cores)')
group.add_argument('--pin-mem', action='store_true', default=False,
                   help='Pin CPU memory in DataLoader')
group.add_argument('--output', default='', type=str, metavar='PATH',
                   help='path to output folder (default: none, current dir)')
group.add_argument('--experiment', default='', type=str, metavar='NAME',
                   help='name of train experiment')
group.add_argument('--log-wandb', action='store_true', default=False,
                   help='log training and validation metrics to wandb')
group.add_argument('--wandb-project', default='aug-prediction-stem', type=str,
                   help='wandb project name')


def _parse_args():
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)
    
    args = parser.parse_args(remaining)
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


# =============================================================================
# Main Training Functions
# =============================================================================

def main():
    utils.setup_default_logging()
    args, args_text = _parse_args()
    
    # Auto-set workers to half of CPU cores if not specified (capped at 16 to avoid deadlock)
    if args.workers is None:
        max_workers = 16  # Upper limit to prevent shared memory issues / deadlock
        args.workers = min(max(1, os.cpu_count() // 2), max_workers)
        _logger.info(
            f'Auto-setting workers to {args.workers} '
            f'(half of {os.cpu_count()} CPU cores, capped at {max_workers})'
        )
    
    if args.device_modules:
        for module in args.device_modules:
            importlib.import_module(module)
    
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    
    device = utils.init_distributed_device(args)
    if args.distributed:
        _logger.info(
            f'Training in distributed mode with multiple processes, 1 device per process. '
            f'Process {args.rank}, total {args.world_size}, device {args.device}.')
    else:
        _logger.info(f'Training with a single process on 1 device ({args.device}).')
    
    # Setup AMP
    amp_dtype = torch.float16
    amp_autocast = suppress
    loss_scaler = None
    if args.amp:
        if args.amp_dtype == 'bfloat16':
            amp_dtype = torch.bfloat16
        amp_autocast = partial(torch.autocast, device_type=device.type, dtype=amp_dtype)
        if device.type == 'cuda' and amp_dtype == torch.float16:
            loss_scaler = NativeScaler(device=device.type)
        _logger.info('Using native Torch AMP.')
    
    utils.random_seed(args.seed, args.rank)
    
    # ==========================================================================
    # Load FSC (Feature Space Centroid) from Stem
    # ==========================================================================
    _logger.info(f'Loading Stem FSC from: {args.fsc_path}')
    fsc_data = torch.load(args.fsc_path, map_location='cpu')
    fsc_centroids = fsc_data['centroids'].to(device)  # [num_classes, stem_feature_dim]
    feature_dim = fsc_data['feature_dim']  # Should be 1024 for stem
    _logger.info(f'Stem FSC loaded: {fsc_centroids.shape}, feature_dim={feature_dim}')
    
    # ==========================================================================
    # Create Backbone Model (Frozen)
    # ==========================================================================
    _logger.info(f'Creating backbone model: {args.model}')
    backbone = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.num_classes,
        checkpoint_path=args.initial_checkpoint,
    )
    backbone = backbone.to(device)
    backbone.eval()
    
    # Freeze all backbone parameters
    for param in backbone.parameters():
        param.requires_grad = False
    
    param_count = sum(p.numel() for p in backbone.parameters())
    _logger.info(f'Backbone {args.model} created, param count: {param_count / 1e6:.2f}M (all frozen)')
    
    # ==========================================================================
    # Create Stem Feature Extractor (Frozen)
    # ==========================================================================
    _logger.info('Creating stem feature extractor...')
    stem_extractor = StemFeatureExtractor(backbone)
    stem_extractor = stem_extractor.to(device)
    stem_extractor.eval()
    
    # Freeze stem extractor (it uses backbone's conv1, bn1, act1 which are already frozen)
    for param in stem_extractor.parameters():
        param.requires_grad = False
    
    _logger.info(f'Stem feature extractor created (output dim: {feature_dim})')
    
    # ==========================================================================
    # Create Aug Classifier (Trainable) - Using V2 Policy
    # ==========================================================================
    num_transforms = get_augmix_sl_num_transforms(version=2)
    transform_names = get_augmix_sl_transform_names(version=2)
    _logger.info(f'Using V2 policy with {num_transforms} transforms: {transform_names}')
    
    # Note: use_sigmoid=False because we use BCEWithLogitsLoss (handles sigmoid internally)
    aug_classifier = AugClassifier(
        feature_dim=feature_dim,  # 1024 for stem
        num_transforms=num_transforms,
        hidden_dims=args.aug_classifier_hidden,
        dropout=args.aug_classifier_dropout,
        use_sigmoid=False,
        use_softmax=args.aug_classifier_softmax,
    )
    aug_classifier = aug_classifier.to(device)
    
    aug_param_count = sum(p.numel() for p in aug_classifier.parameters())
    _logger.info(f'Aug classifier created, param count: {aug_param_count / 1e6:.2f}M (trainable)')
    
    # ==========================================================================
    # Setup Loss Function
    # ==========================================================================
    if args.sl_loss_type == 'bce':
        # BCEWithLogitsLoss is AMP-safe and numerically stable
        # It applies sigmoid internally, so model outputs raw logits
        criterion = nn.BCEWithLogitsLoss()
        if args.normalize_aug_labels:
            _logger.warning(
                'Using BCE loss with --normalize-aug-labels may not be ideal. '
                'Consider using --sl-loss-type kldiv or ce for probability distributions.'
            )
    elif args.sl_loss_type == 'kldiv':
        # KL Divergence for comparing probability distributions
        # Model should output log_softmax, target should be normalized (sum to 1)
        criterion = nn.KLDivLoss(reduction='batchmean')
        if not args.normalize_aug_labels:
            _logger.warning(
                'Using KL divergence loss without --normalize-aug-labels. '
                'Labels may not sum to 1, which is required for proper KL divergence.'
            )
    elif args.sl_loss_type == 'ce':
        # Soft Cross Entropy for probability distributions
        # CE = H(target) + KL(target || pred), same gradients as KL but different loss values
        criterion = SoftCrossEntropyLoss(reduction='mean')
        if not args.normalize_aug_labels:
            _logger.warning(
                'Using Cross Entropy loss without --normalize-aug-labels. '
                'Labels may not sum to 1, which is required for proper cross entropy.'
            )
    else:
        # MSE loss for regression
        criterion = nn.MSELoss()
    criterion = criterion.to(device)
    _logger.info(f'Using {args.sl_loss_type.upper()} loss')
    
    # ==========================================================================
    # Create Optimizer (only for aug_classifier)
    # ==========================================================================
    optimizer = create_optimizer_v2(
        aug_classifier,
        opt=args.opt,
        lr=args.lr,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
    )
    _logger.info(
        f'Created meta-optimizer: {args.opt}, lr={args.lr} '
        f'(inner_lr={args.inner_lr}, inner_steps={args.inner_steps})'
    )
    
    # ==========================================================================
    # Resume from checkpoint if specified
    # ==========================================================================
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        aug_classifier.load_state_dict(checkpoint['aug_classifier'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        _logger.info(f'Resumed from epoch {start_epoch}')
    
    # ==========================================================================
    # Create Datasets and DataLoaders
    # ==========================================================================
    data_config = resolve_data_config(vars(args), model=backbone, verbose=utils.is_primary(args))
    
    if args.data and not args.data_dir:
        args.data_dir = args.data
    
    # Create base transforms
    from torchvision import transforms
    from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
    
    base_transform = transforms.Compose([
        transforms.Resize(int(args.img_size / data_config['crop_pct'])),
        transforms.CenterCrop(args.img_size),
    ])
    
    final_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=data_config['mean'], std=data_config['std']),
    ])
    
    # Create AugMixSL transforms with V2 policy
    augmix_sl_transform = create_augmix_sl_transform(
        max_depth=args.new_depth,
        min_sl=args.min_sl,
        max_sl=args.max_sl,
        version=2,  # Use V2 policy
        normalize_labels=args.normalize_aug_labels,
    )
    
    # Get V2 transform ops for per-transform validation
    from timm.data import augmix_sl_ops_v2
    val_transform_ops = augmix_sl_ops_v2()
    val_severity_level = args.max_sl  # Use max SL for validation (stronger signal)
    
    # Create raw datasets (without transforms)
    _logger.info(f'Loading dataset from: {args.data_dir}')
    
    # Use ImageFolder directly for more control over transforms
    from torchvision.datasets import ImageFolder
    
    train_dir = os.path.join(args.data_dir, args.train_split)
    val_dir = os.path.join(args.data_dir, args.val_split)
    
    raw_train_dataset = ImageFolder(train_dir)
    raw_val_dataset = ImageFolder(val_dir)
    
    # Wrap training dataset with AugMixSL
    dataset_train = AugMixSLDataset(
        raw_train_dataset,
        augmix_sl_transform=augmix_sl_transform,
        base_transform=base_transform,
        final_transform=final_transform,
    )
    
    _logger.info(f'Train dataset: {len(dataset_train)} samples')
    _logger.info(f'Val dataset: {len(raw_val_dataset)} samples (per-transform evaluation)')
    _logger.info(f'Validation transforms ({num_transforms}): {transform_names}')
    _logger.info(f'Validation SL: {val_severity_level}')
    
    # Create training data loader
    sampler_train = None
    if args.distributed:
        sampler_train = torch.utils.data.distributed.DistributedSampler(dataset_train)
    
    loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=(sampler_train is None),
        sampler=sampler_train,
        num_workers=args.workers,
        pin_memory=args.pin_mem,
        collate_fn=collate_aug_labels,
        drop_last=True,
        persistent_workers=True if args.workers > 0 else False,  # Reuse workers (faster startup)
        prefetch_factor=2 if args.workers > 0 else None,  # Reduced to prevent memory pressure
    )
    
    # Note: Validation loaders are created per-transform inside validate_per_transform()
    
    # ==========================================================================
    # Setup Learning Rate Scheduler
    # ==========================================================================
    lr_scheduler = None
    if args.sched:
        lr_scheduler, num_epochs = create_scheduler_v2(
            optimizer,
            sched=args.sched,
            num_epochs=args.epochs,
            warmup_epochs=args.warmup_epochs,
            warmup_lr=args.warmup_lr,
            min_lr=args.min_lr,
            decay_rate=args.decay_rate,
        )
    else:
        num_epochs = args.epochs
    
    # ==========================================================================
    # Setup Output Directory
    # ==========================================================================
    output_dir = None
    if args.output:
        output_dir = Path(args.output)
    else:
        exp_name = args.experiment or f'augmix_stem_{args.model}'
        output_dir = Path(f'./output/{exp_name}')
    
    if utils.is_primary(args):
        output_dir.mkdir(parents=True, exist_ok=True)
        # Save args
        with open(output_dir / 'args.yaml', 'w') as f:
            f.write(args_text)
        _logger.info(f'Output directory: {output_dir}')
    
    # ==========================================================================
    # Setup Wandb
    # ==========================================================================
    if args.log_wandb and utils.is_primary(args) and has_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.experiment or f'augmix_stem_{args.model}',
            config=args,
        )
    
    # ==========================================================================
    # Meta-Learning Training Loop (Phase 2)
    # ==========================================================================
    _logger.info(
        f'Starting meta-learning (curriculum) for {num_epochs} epochs  '
        f'(inner_lr={args.inner_lr}, inner_steps={args.inner_steps}, '
        f'inner_momentum={args.inner_momentum}, '
        f'meta_lr={args.lr}, fsc_diff_mode={args.fsc_diff_mode}, '
        f'inner_alpha={args.inner_alpha}, inner_beta={args.inner_beta}, '
        f'inner_gamma={args.inner_gamma}, '
        f'aux_weight={args.aux_weight_max}->{args.aux_weight_min}, '
        f'depth_curriculum={not args.no_depth_curriculum})'
    )

    best_metric = None
    best_epoch = None
    # Top-K checkpoint tracker: list of (metric, epoch, filepath), sorted descending
    top_checkpoints = []  # keeps up to checkpoint_hist best checkpoints

    # ==========================================================================
    # Initialize results file
    # ==========================================================================
    results_path = None
    if utils.is_primary(args) and output_dir:
        results_path = os.path.join(output_dir, 'meta_train_results.txt')
        if start_epoch == 0 or not os.path.exists(results_path):
            header = [
                'epoch', 'L_outer', 'train_top1', 'train_top5', 'lr',
                'aux_weight', 'depth',  # Curriculum
                # Diagnostics (health monitoring)
                'aug_grad_norm',       # Step 5: must be > 0
                'entropy_bef',         # Step 3: should increase →
                'entropy_aft',         # Step 3: → to this
                'param_delta',         # Step 3: conv1 change magnitude
                'fsc_norm_bef',        # Step 3: corruption signal →
                'fsc_norm_aft',        # Step 3: → should decrease
                'pretrained_top1',     # Step 4: baseline (no TTA)
                # Validation
            ]
            header += [f'val_{n}' for n in transform_names]
            header += ['val_mean_top1', 'best']
            with open(results_path, 'w') as f:
                f.write('\t'.join(header) + '\n')
            _logger.info(f'Results will be saved to: {results_path}')

    try:
        for epoch in range(start_epoch, num_epochs):
            if args.distributed:
                loader_train.sampler.set_epoch(epoch)

            # ---- Curriculum: aux_weight (cosine decay) and depth (step) ----
            current_aux_weight = get_curriculum_aux_weight(
                epoch, num_epochs,
                aux_max=args.aux_weight_max,
                aux_min=args.aux_weight_min,
            )
            if args.no_depth_curriculum:
                current_depth = args.new_depth
            else:
                current_depth = get_curriculum_depth(epoch, num_epochs)
                dataset_train.augmix_sl_transform.max_depth = current_depth
            if utils.is_primary(args):
                _logger.info(
                    f'Curriculum epoch {epoch}: aux_weight={current_aux_weight:.4f}, '
                    f'depth={current_depth}'
                )

            # ---- Train one epoch (meta-learning) ----
            train_metrics = train_one_epoch_meta(
                epoch=epoch,
                model=backbone,
                aug_classifier=aug_classifier,
                fsc_centroids=fsc_centroids,
                loader=loader_train,
                meta_optimizer=optimizer,
                args=args,
                device=device,
                lr_scheduler=lr_scheduler,
                criterion=criterion,
                aux_weight=current_aux_weight,
            )

            if lr_scheduler is not None:
                lr_scheduler.step(epoch + 1)

            # ---- Validation: fixed per-transform TTA ----
            val_metrics = None
            if (epoch + 1) % args.val_interval == 0 or epoch == num_epochs - 1:
                _logger.info(f'Running validation TTA (epoch {epoch}) ...')
                val_metrics = validate_meta_tta(
                    model=backbone,
                    aug_classifier=aug_classifier,
                    fsc_centroids=fsc_centroids,
                    raw_val_dataset=raw_val_dataset,
                    val_transform_ops=val_transform_ops,
                    transform_names=transform_names,
                    num_transforms=num_transforms,
                    severity_level=val_severity_level,
                    base_transform=base_transform,
                    final_transform=final_transform,
                    args=args,
                    device=device,
                )

            # ---- Save checkpoint ----
            if utils.is_primary(args) and output_dir:
                checkpoint = {
                    'epoch': epoch,
                    'aug_classifier': aug_classifier.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'args': args,
                }
                torch.save(checkpoint, os.path.join(output_dir, 'last.pth.tar'))

                if val_metrics is not None:
                    current_metric = val_metrics['mean_top1']
                    if best_metric is None or current_metric > best_metric:
                        best_metric = current_metric
                        best_epoch = epoch
                        torch.save(checkpoint, os.path.join(
                            output_dir, 'best.pth.tar',
                        ))
                        _logger.info(
                            f'New best val_mean_top1: {best_metric:.2f}% '
                            f'at epoch {epoch}'
                        )

                    # ---- Top-K checkpoint management ----
                    ckpt_path = os.path.join(
                        output_dir, f'checkpoint-{epoch}.pth.tar',
                    )
                    max_keep = args.checkpoint_hist
                    if len(top_checkpoints) < max_keep:
                        # Still have room — save unconditionally
                        torch.save(checkpoint, ckpt_path)
                        top_checkpoints.append(
                            (current_metric, epoch, ckpt_path),
                        )
                        top_checkpoints.sort(key=lambda x: x[0], reverse=True)
                        _logger.info(
                            f'Saved checkpoint-{epoch} '
                            f'(val_mean_top1={current_metric:.2f}%, '
                            f'rank {len(top_checkpoints)}/{max_keep})'
                        )
                    else:
                        # Full — replace worst if current is better
                        worst_metric, worst_epoch, worst_path = top_checkpoints[-1]
                        if current_metric > worst_metric:
                            # Remove worst checkpoint file
                            if os.path.exists(worst_path):
                                os.remove(worst_path)
                                _logger.info(
                                    f'Removed checkpoint-{worst_epoch} '
                                    f'(val_mean_top1={worst_metric:.2f}%)'
                                )
                            # Save new checkpoint
                            torch.save(checkpoint, ckpt_path)
                            top_checkpoints[-1] = (
                                current_metric, epoch, ckpt_path,
                            )
                            top_checkpoints.sort(
                                key=lambda x: x[0], reverse=True,
                            )
                            rank = next(
                                i + 1 for i, t in enumerate(top_checkpoints)
                                if t[1] == epoch
                            )
                            _logger.info(
                                f'Saved checkpoint-{epoch} '
                                f'(val_mean_top1={current_metric:.2f}%, '
                                f'rank {rank}/{max_keep})'
                            )

                if args.log_wandb and has_wandb:
                    log_dict = {
                        'epoch': epoch,
                        'L_outer': train_metrics['L_outer'],
                        'train_top1': train_metrics['top1'],
                        'lr': optimizer.param_groups[0]['lr'],
                    }
                    if val_metrics is not None:
                        log_dict['val_mean_top1'] = val_metrics['mean_top1']
                        for t_name in transform_names:
                            log_dict[f'val_{t_name}'] = val_metrics[
                                'per_transform_top1'
                            ].get(t_name, 0)
                    wandb.log(log_dict)

                if results_path is not None:
                    lr = optimizer.param_groups[0]['lr']
                    is_best = '*' if (best_epoch == epoch) else ''
                    row = [
                        str(epoch),
                        f'{train_metrics["L_outer"]:.4f}',
                        f'{train_metrics["top1"]:.2f}',
                        f'{train_metrics["top5"]:.2f}',
                        f'{lr:.2e}',
                        f'{current_aux_weight:.4f}',
                        str(current_depth),
                        # Diagnostics
                        f'{train_metrics.get("aug_grad_norm", 0):.4f}',
                        f'{train_metrics.get("entropy_before", 0):.4f}',
                        f'{train_metrics.get("entropy_after", 0):.4f}',
                        f'{train_metrics.get("param_delta", 0):.6f}',
                        f'{train_metrics.get("fsc_norm_before", 0):.4f}',
                        f'{train_metrics.get("fsc_norm_after", 0):.4f}',
                        f'{train_metrics.get("pretrained_top1", 0):.2f}',
                    ]
                    if val_metrics is not None:
                        for t_name in transform_names:
                            row.append(f'{val_metrics["per_transform_top1"].get(t_name, 0):.2f}')
                        row.append(f'{val_metrics["mean_top1"]:.2f}')
                    else:
                        row += [''] * (num_transforms + 1)
                    row.append(is_best)
                    with open(results_path, 'a') as f:
                        f.write('\t'.join(row) + '\n')

    except KeyboardInterrupt:
        pass

    if args.distributed:
        torch.distributed.destroy_process_group()

    if best_metric is not None:
        _logger.info(
            f'*** Best val_mean_top1: {best_metric:.2f}% (epoch {best_epoch})'
        )


def train_one_epoch_meta(
    epoch: int,
    model: nn.Module,
    aug_classifier: nn.Module,
    fsc_centroids: torch.Tensor,
    loader,
    meta_optimizer,
    args,
    device: torch.device,
    lr_scheduler=None,
    criterion: nn.Module = None,
    aux_weight: float = 0.0,
) -> dict:
    """One epoch of MAML-style meta-learning.

    When criterion and aux_weight > 0, adds auxiliary aug prediction loss (L_aug)
    on original features; L_outer = L_cls + aux_weight * L_aug.

    Returns:
        Dict with training metrics + diagnostics for health monitoring.
    """
    loss_m = utils.AverageMeter()
    loss_aug_m = utils.AverageMeter()
    top1_m = utils.AverageMeter()
    top5_m = utils.AverageMeter()
    batch_time_m = utils.AverageMeter()
    grad_norm_m = utils.AverageMeter()  # Step 5: aug_classifier meta-gradient norm

    # Diagnostic accumulators (sampled periodically to avoid overhead)
    diag_entropy_before = utils.AverageMeter()
    diag_entropy_after = utils.AverageMeter()
    diag_param_delta = utils.AverageMeter()
    diag_fsc_norm_before = utils.AverageMeter()
    diag_fsc_norm_after = utils.AverageMeter()
    diag_pretrained_top1 = utils.AverageMeter()

    aug_classifier.train()
    model.eval()

    end = time.time()
    num_batches = len(loader)
    diag_interval = max(1, num_batches // 10)  # sample ~10 times per epoch

    for batch_idx, (images, labels, aug_labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)
        aug_labels = aug_labels.to(device)

        # ---- Differentiable meta-step ----
        meta_optimizer.zero_grad()

        L_outer, info = meta_step_differentiable(
            images, labels, model, aug_classifier, fsc_centroids,
            fsc_diff_mode=args.fsc_diff_mode,
            inner_lr=args.inner_lr,
            inner_steps=args.inner_steps,
            inner_alpha=args.inner_alpha,
            inner_beta=args.inner_beta,
            inner_gamma=args.inner_gamma,
            aug_labels=aug_labels,
            aux_weight=aux_weight,
            criterion=criterion,
        )

        # ---- Meta-update: backprop L_outer → update aug_classifier ----
        L_outer.backward()

        # Step 5 diagnostic: aug_classifier gradient norm (must be > 0)
        total_grad_norm = 0.0
        for p in aug_classifier.parameters():
            if p.grad is not None:
                total_grad_norm += p.grad.data.norm(2).item() ** 2
        total_grad_norm = total_grad_norm ** 0.5
        grad_norm_m.update(total_grad_norm)

        if args.clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(
                aug_classifier.parameters(), args.clip_grad,
            )
        meta_optimizer.step()

        # ---- Step 3 & 4 diagnostics (sampled periodically) ----
        if batch_idx % diag_interval == 0:
            with torch.no_grad():
                # Pretrained baseline (no TTA)
                before = meta_outer_eval(images, labels, model)
                diag_pretrained_top1.update(before['top1'], images.size(0))

            # Run simple inner loop for diagnostics (entropy, param delta, etc.)
            simple_diag = meta_inner_loop_simple(
                images, model, aug_classifier, fsc_centroids,
                fsc_diff_mode=args.fsc_diff_mode,
                inner_lr=args.inner_lr,
                inner_steps=args.inner_steps,
                inner_alpha=args.inner_alpha,
                inner_beta=args.inner_beta,
                inner_gamma=args.inner_gamma,
                inner_momentum=args.inner_momentum,
            )
            diag_entropy_before.update(simple_diag['entropy_before'])
            diag_entropy_after.update(simple_diag['entropy_after'])
            diag_param_delta.update(simple_diag['param_delta_norm'])
            diag_fsc_norm_before.update(simple_diag['fsc_diff_norm_before'])
            diag_fsc_norm_after.update(simple_diag['fsc_diff_norm_after'])

            # Reset conv1 after diagnostic run
            reset_conv1(model, simple_diag['original_weight'])

        # ---- Logging ----
        batch_size = images.size(0)
        loss_m.update(info['loss'], batch_size)
        if 'loss_aug' in info:
            loss_aug_m.update(info['loss_aug'], batch_size)
        top1_m.update(info['top1'], batch_size)
        top5_m.update(info['top5'], batch_size)
        batch_time_m.update(time.time() - end)
        end = time.time()

        if batch_idx % args.log_interval == 0 or batch_idx == num_batches - 1:
            lr = meta_optimizer.param_groups[0]['lr']
            if utils.is_primary(args):
                log_msg = (
                    f'Meta-Train: {epoch} [{batch_idx:>4d}/{num_batches}]  '
                    f'L_outer: {loss_m.val:.4f} ({loss_m.avg:.4f})  '
                    f'Top1: {top1_m.val:.2f}% ({top1_m.avg:.2f}%)  '
                    f'GradNorm: {grad_norm_m.val:.4f}  '
                    f'Time: {batch_time_m.val:.3f}s  LR: {lr:.2e}'
                )
                if aux_weight > 0 and loss_aug_m.count > 0:
                    log_msg += f'  L_aug: {loss_aug_m.val:.4f} ({loss_aug_m.avg:.4f})'
                _logger.info(log_msg)

    # ---- Epoch summary with diagnostics ----
    if utils.is_primary(args):
        diag_msg = (
            f'  [Diagnostics] epoch={epoch}  '
            f'entropy: {diag_entropy_before.avg:.4f}→{diag_entropy_after.avg:.4f}  '
            f'param_delta: {diag_param_delta.avg:.6f}  '
            f'fsc_norm: {diag_fsc_norm_before.avg:.4f}→{diag_fsc_norm_after.avg:.4f}  '
            f'pretrained_top1: {diag_pretrained_top1.avg:.2f}%  '
            f'adapted_top1: {top1_m.avg:.2f}%  '
            f'aug_grad_norm: {grad_norm_m.avg:.4f}'
        )
        if aux_weight > 0 and loss_aug_m.count > 0:
            diag_msg += f'  L_aug: {loss_aug_m.avg:.4f}'
        _logger.info(diag_msg)

    result = OrderedDict([
        ('L_outer', loss_m.avg),
        ('top1', top1_m.avg),
        ('top5', top5_m.avg),
        ('loss_aug', loss_aug_m.avg),
        # Diagnostics
        ('aug_grad_norm', grad_norm_m.avg),
        ('entropy_before', diag_entropy_before.avg),
        ('entropy_after', diag_entropy_after.avg),
        ('param_delta', diag_param_delta.avg),
        ('fsc_norm_before', diag_fsc_norm_before.avg),
        ('fsc_norm_after', diag_fsc_norm_after.avg),
        ('pretrained_top1', diag_pretrained_top1.avg),
    ])
    return result


# =============================================================================
# Meta-Learning Validation: Fixed Per-Transform TTA Evaluation
# =============================================================================

def validate_meta_tta(
    model: nn.Module,
    aug_classifier: nn.Module,
    fsc_centroids: torch.Tensor,
    raw_val_dataset,
    val_transform_ops,
    transform_names: list,
    num_transforms: int,
    severity_level: float,
    base_transform,
    final_transform,
    args,
    device: torch.device,
) -> dict:
    """Validate meta-learned aug_classifier via actual TTA on fixed augmentations.

    For each of the 7 augmentation types (at max SL):
      1. Apply the single fixed augmentation to val images.
      2. Run TTA inner loop (meta_inner_loop_simple) on each batch.
      3. Evaluate classification accuracy after TTA.

    Returns:
        Dict with ``mean_top1``, ``mean_top5``, ``per_transform`` detail.
    """
    from timm.data.auto_augment import AugMixSLAugmentFixed

    model.eval()
    aug_classifier.eval()

    per_transform_top1 = {}
    per_transform_top5 = {}

    val_batch_size = args.validation_batch_size or args.batch_size

    for t_idx in range(num_transforms):
        t_name = transform_names[t_idx]

        # Fixed single-transform augmentation at max SL
        fixed_aug = AugMixSLAugmentFixed(
            ops=val_transform_ops,
            fixed_transforms=[(t_idx, severity_level)],
            max_sl=args.max_sl,
        )

        # Wrap val dataset
        val_dataset = AugMixSLDataset(
            raw_val_dataset,
            augmix_sl_transform=fixed_aug,
            base_transform=base_transform,
            final_transform=final_transform,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=args.pin_mem,
            collate_fn=collate_aug_labels,
            drop_last=False,
        )

        top1_m = utils.AverageMeter()
        top5_m = utils.AverageMeter()

        for images, labels, _aug_labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Save pretrained conv1
            original_weight = model.conv1.weight.data.clone()

            # Inner loop TTA (simple, no meta-gradient)
            _diag = meta_inner_loop_simple(
                images, model, aug_classifier, fsc_centroids,
                fsc_diff_mode=args.fsc_diff_mode,
                inner_lr=args.inner_lr,
                inner_steps=args.inner_steps,
                inner_alpha=args.inner_alpha,
                inner_beta=args.inner_beta,
                inner_gamma=args.inner_gamma,
                inner_momentum=args.inner_momentum,
            )

            # Evaluate after TTA
            after = meta_outer_eval(images, labels, model)
            top1_m.update(after['top1'], images.size(0))
            top5_m.update(after['top5'], images.size(0))

            # Reset conv1
            reset_conv1(model, original_weight)

        per_transform_top1[t_name] = top1_m.avg
        per_transform_top5[t_name] = top5_m.avg
        _logger.info(
            f'  Val TTA [{t_name}] (SL={severity_level}): '
            f'Top1={top1_m.avg:.2f}%, Top5={top5_m.avg:.2f}%'
        )

    mean_top1 = sum(per_transform_top1.values()) / num_transforms
    mean_top5 = sum(per_transform_top5.values()) / num_transforms
    _logger.info(
        f'  Val TTA Mean: Top1={mean_top1:.2f}%, Top5={mean_top5:.2f}%'
    )

    aug_classifier.train()

    return {
        'mean_top1': mean_top1,
        'mean_top5': mean_top5,
        'per_transform_top1': per_transform_top1,
        'per_transform_top5': per_transform_top5,
    }


# =============================================================================
# Legacy Phase 1 functions (kept for reference, not used in meta-training)
# =============================================================================

def train_one_epoch(
    epoch,
    stem_extractor,
    aug_classifier,
    fsc_centroids,
    loader,
    optimizer,
    criterion,
    args,
    device,
    lr_scheduler=None,
    amp_autocast=suppress,
    loss_scaler=None,
):
    """Train for one epoch using stem features.
    
    Note: Uses ground truth labels for FSC selection (no backbone forward needed).
    """
    
    losses_m = utils.AverageMeter()
    batch_time_m = utils.AverageMeter()
    data_time_m = utils.AverageMeter()
    
    stem_extractor.eval()  # Keep stem extractor in eval mode
    aug_classifier.train()
    
    end = time.time()
    num_batches = len(loader)
    
    for batch_idx, (images, labels, aug_labels) in enumerate(loader):
        data_time_m.update(time.time() - end)
        
        images = images.to(device)
        labels = labels.to(device)
        aug_labels = aug_labels.to(device)
        
        with amp_autocast():
            # Extract stem features from frozen stem extractor
            # Note: During training, we use ground truth labels for FSC selection
            # This avoids running the full backbone network for classification
            with torch.no_grad():
                # Get stem features (1024-dim)
                stem_features = stem_extractor(images)
            
            # Select FSC centroids using GROUND TRUTH labels
            fsc_for_batch = fsc_centroids[labels]  # [batch_size, stem_feature_dim]
            
            # Compute FSC_diff based on specified mode (subtract or orthogonal)
            fsc_diff = compute_fsc_diff(stem_features, fsc_for_batch, mode=args.fsc_diff_mode)
            
            # Predict augmentation labels
            aug_pred = aug_classifier(fsc_diff)
            
            # Compute loss
            if args.sl_loss_type == 'kldiv':
                # KL divergence expects log probabilities vs probability targets
                aug_pred_log = F.log_softmax(aug_pred, dim=1)
                loss = criterion(aug_pred_log, aug_labels)
            else:
                loss = criterion(aug_pred, aug_labels)
        
        # Backward pass
        optimizer.zero_grad()
        if loss_scaler is not None:
            loss_scaler(
                loss,
                optimizer,
                clip_grad=args.clip_grad,
                parameters=aug_classifier.parameters(),
            )
        else:
            loss.backward()
            if args.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(aug_classifier.parameters(), args.clip_grad)
            optimizer.step()
        
        losses_m.update(loss.item(), images.size(0))
        batch_time_m.update(time.time() - end)
        end = time.time()
        
        if batch_idx % args.log_interval == 0 or batch_idx == num_batches - 1:
            lr = optimizer.param_groups[0]['lr']
            if utils.is_primary(args):
                _logger.info(
                    f'Train: {epoch} [{batch_idx:>4d}/{num_batches}]  '
                    f'Loss: {losses_m.val:.4f} ({losses_m.avg:.4f})  '
                    f'Time: {batch_time_m.val:.3f}s  '
                    f'LR: {lr:.2e}'
                )
    
    return OrderedDict([('loss', losses_m.avg)])


def validate_per_transform(
    stem_extractor,
    aug_classifier,
    fsc_centroids,
    raw_val_dataset,
    val_transform_ops,
    transform_names,
    num_transforms,
    severity_level,
    base_transform,
    final_transform,
    criterion,
    args,
    device,
    amp_autocast=suppress,
):
    """Validate the model by testing each transform individually.
    
    For each transform:
    1. Apply that single transform (with fixed SL) to all validation images
    2. Measure whether the model correctly identifies which transform was applied
    
    Returns mean accuracy across all transforms.
    
    Args:
        stem_extractor: Stem feature extractor.
        aug_classifier: Augmentation classifier.
        fsc_centroids: FSC centroids tensor.
        raw_val_dataset: Raw validation ImageFolder dataset.
        val_transform_ops: List of AugMixSLOp instances for V2 transforms.
        transform_names: List of transform names.
        num_transforms: Number of transforms.
        severity_level: Fixed severity level for validation.
        base_transform: Base image transforms.
        final_transform: Final image transforms (ToTensor, Normalize).
        criterion: Loss function.
        args: Training arguments.
        device: Device to use.
        amp_autocast: AMP autocast context.
    
    Returns:
        OrderedDict with 'mean_acc' and 'per_transform_acc'.
    """
    
    stem_extractor.eval()
    aug_classifier.eval()
    
    per_transform_acc = {}
    per_transform_correct = {}
    per_transform_total = {}
    
    if utils.is_primary(args):
        _logger.info(f'Per-transform validation (SL={severity_level:.2f}):')
    
    with torch.no_grad():
        for t_idx, (t_op, t_name) in enumerate(zip(val_transform_ops, transform_names)):
            # Create dataset for this single transform
            single_transform_dataset = SingleTransformValidationDataset(
                dataset=raw_val_dataset,
                transform_op=t_op,
                transform_idx=t_idx,
                num_transforms=num_transforms,
                severity_level=severity_level,
                max_sl=args.max_sl,
                base_transform=base_transform,
                final_transform=final_transform,
            )
            
            # Create data loader
            sampler = None
            if args.distributed:
                sampler = torch.utils.data.distributed.DistributedSampler(
                    single_transform_dataset, shuffle=False
                )
            
            loader = torch.utils.data.DataLoader(
                single_transform_dataset,
                batch_size=args.validation_batch_size or args.batch_size,
                shuffle=False,
                sampler=sampler,
                num_workers=args.workers,
                pin_memory=args.pin_mem,
                collate_fn=collate_aug_labels,
                persistent_workers=False,  # Must be False: new DataLoader created each iteration
                prefetch_factor=2 if args.workers > 0 else None,
            )
            
            # Evaluate on this transform
            correct_count = 0
            total_count = 0
            
            for batch_idx, (images, labels, aug_labels) in enumerate(loader):
                images = images.to(device)
                labels = labels.to(device)
                aug_labels = aug_labels.to(device)
                
                with amp_autocast():
                    # Extract stem features
                    stem_features = stem_extractor(images)
                    
                    # Select FSC centroids using GROUND TRUTH labels
                    fsc_for_batch = fsc_centroids[labels]
                    
                    # Compute FSC_diff
                    fsc_diff = compute_fsc_diff(stem_features, fsc_for_batch, mode=args.fsc_diff_mode)
                    
                    # Predict augmentation labels
                    aug_pred = aug_classifier(fsc_diff)
                
                # Get predictions for the current transform only
                # For this single-transform validation, we check if the model
                # correctly identifies that THIS transform was applied
                if args.sl_loss_type == 'bce':
                    aug_pred_prob = torch.sigmoid(aug_pred)
                elif args.sl_loss_type in ('kldiv', 'ce'):
                    # Both KL divergence and Cross Entropy use softmax for probability output
                    aug_pred_prob = F.softmax(aug_pred, dim=1)
                else:
                    aug_pred_prob = aug_pred
                
                # Check if the correct transform has highest prediction
                # (since only one transform is applied, it should be predicted highest)
                pred_transform_idx = aug_pred_prob.argmax(dim=1)
                correct = (pred_transform_idx == t_idx).sum().item()
                
                correct_count += correct
                total_count += images.size(0)
            
            # Calculate accuracy for this transform
            transform_acc = 100.0 * correct_count / total_count if total_count > 0 else 0.0
            per_transform_acc[t_name] = transform_acc
            per_transform_correct[t_name] = correct_count
            per_transform_total[t_name] = total_count
            
            if utils.is_primary(args):
                _logger.info(f'  {t_name}: {transform_acc:.2f}% ({correct_count}/{total_count})')
            
            # Explicit memory cleanup to prevent OOM across transform iterations
            del loader
            del single_transform_dataset
            del sampler
            gc.collect()
            torch.cuda.empty_cache()
    
    # Calculate mean accuracy
    mean_acc = sum(per_transform_acc.values()) / len(per_transform_acc) if per_transform_acc else 0.0
    
    if utils.is_primary(args):
        _logger.info(f'  Mean Accuracy: {mean_acc:.2f}%')
    
    return OrderedDict([
        ('mean_acc', mean_acc),
        ('per_transform_acc', per_transform_acc),
    ])


if __name__ == '__main__':
    main()

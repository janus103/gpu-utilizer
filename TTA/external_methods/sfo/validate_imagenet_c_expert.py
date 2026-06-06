#!/usr/bin/env python3
"""ImageNet-C Expert Validation Script.

Evaluates ImageNet-C using augmentation-specific expert weights selected by
a trained Aug Classifier.  The Aug Classifier determines which augmentation
each corruption is most similar to, then loads the corresponding expert
weight(s) for evaluation.

Pipeline:
1. Load pretrained ResNet50, Aug Classifier, and FSC centroids.
2. Discover best expert checkpoints per augmentation type.
3. For each corruption (severity=5 by default):
   a. Classify augmentation type via stem features -> FSC diff -> Aug Classifier.
   b. Select and apply expert weight(s).
   c. Evaluate classification accuracy.
4. Save results to .txt file.

Usage examples:
    # Top-1, per-corruption, mean FSC
    python validate_imagenet_c_expert.py \
        --imagenet-c-dir /path/to/ImageNet-C \
        --imagenet-val-dir /path/to/imagenet/val \
        --aug-classifier-ckpt ./output/STEM_KLDIV_ORTHOGONAL/best.pth.tar \
        --fsc-path ./FSC/resnet50_FSC_stem.pth \
        --expert-weight-dir /home/oem/jin/SOA/new_weight \
        --expert-type stem \
        --weight-mode top1 \
        --classify-mode per-corruption \
        --fsc-mode mean \
        --results-dir ./results

    # Top-2 with threshold, per-batch, predicted FSC, geometric BN stats
    python validate_imagenet_c_expert.py \
        --imagenet-c-dir /path/to/ImageNet-C \
        --imagenet-val-dir /path/to/imagenet/val \
        --aug-classifier-ckpt ./output/STEM_KLDIV_ORTHOGONAL/best.pth.tar \
        --fsc-path ./FSC/resnet50_FSC_stem.pth \
        --expert-weight-dir /home/oem/jin/SOA/new_weight \
        --expert-type stem-all-bn \
        --weight-mode top2-threshold \
        --threshold 0.3 \
        --bn-mode geometric_stats \
        --classify-mode per-batch \
        --fsc-mode predicted \
        --results-dir ./results
"""

import argparse
import copy
import logging
import os
import re
from collections import OrderedDict
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from timm.data import create_transform, resolve_data_config
from timm.data import get_augmix_sl_num_transforms, get_augmix_sl_transform_names
from timm.models import create_model
from timm.utils import AverageMeter, accuracy, setup_default_logging

from train_augmix_stem import (
    AugClassifier,
    StemFeatureExtractor,
    compute_fsc_diff,
)

_logger = logging.getLogger("validate_imagenet_c_expert")

DEFAULT_CORRUPTIONS = [
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "defocus_blur",
    "glass_blur",
    "motion_blur",
    "zoom_blur",
    "snow",
    "frost",
    "fog",
    "brightness",
    "contrast",
    "elastic_transform",
    "pixelate",
    "jpeg_compression",
]


# =============================================================================
# Dataset
# =============================================================================

class ImageNetCDataset(Dataset):
    """ImageNet-C folder that may contain class subdirs or flat files."""

    def __init__(
        self,
        root: Path,
        filename_label_map: Dict[str, int],
        class_to_idx: Dict[str, int],
        transform,
    ) -> None:
        self.root = root
        self.transform = transform

        entries = list(root.iterdir())
        has_class_dirs = any(p.is_dir() for p in entries)

        kept_paths: List[Path] = []
        targets: List[int] = []

        patterns = ("*.JPEG", "*.jpeg", "*.jpg", "*.png")

        if has_class_dirs:
            for class_dir in sorted(p for p in entries if p.is_dir()):
                target = class_to_idx.get(class_dir.name)
                if target is None:
                    continue
                for pattern in patterns:
                    for path in sorted(class_dir.glob(pattern)):
                        kept_paths.append(path)
                        targets.append(target)
        else:
            paths: List[Path] = []
            for pattern in patterns:
                paths.extend(sorted(root.glob(pattern)))

            for path in paths:
                target = filename_label_map.get(path.name)
                if target is None:
                    continue
                kept_paths.append(path)
                targets.append(target)

            missing = len(paths) - len(kept_paths)
            if missing > 0:
                _logger.warning("Skipped %d files without labels in %s", missing, root)

        if not kept_paths:
            raise RuntimeError(f"No usable images found in {root}")

        self.paths = kept_paths
        self.targets = targets

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        path = self.paths[index]
        target = self.targets[index]
        img = Image.open(path).convert("RGB")
        return self.transform(img), target


def _build_label_map(val_dir: Path) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Build mappings for filenames and class dirs from ImageNet val directory."""
    from torchvision import datasets

    val_dataset = datasets.ImageFolder(val_dir)
    filename_label_map = {Path(p).name: target for p, target in val_dataset.samples}
    class_to_idx = val_dataset.class_to_idx
    _logger.info("Loaded label map for %d validation images", len(filename_label_map))
    return filename_label_map, class_to_idx


# =============================================================================
# Expert Checkpoint Discovery
# =============================================================================

def _find_best_expert_checkpoint(expert_dir: Path) -> Tuple[Path, float]:
    """Find the best checkpoint in an expert directory.

    Looks for files matching ``best_ep{epoch}_{accuracy}.pth.tar`` and selects
    the one with the highest accuracy.  Falls back to ``best.pth.tar`` only if
    no epoch-named files exist.
    """
    pattern = re.compile(r"best_ep(\d+)_(\d+\.\d+)\.pth\.tar")

    best_path: Optional[Path] = None
    best_acc = -1.0

    for f in expert_dir.iterdir():
        match = pattern.match(f.name)
        if match:
            acc = float(match.group(2))
            if acc > best_acc:
                best_acc = acc
                best_path = f

    if best_path is not None:
        return best_path, best_acc

    # Fallback: best.pth.tar
    fallback = expert_dir / "best.pth.tar"
    if fallback.exists():
        ckpt = torch.load(fallback, map_location="cpu")
        acc = ckpt.get("top1", 0.0)
        return fallback, acc

    raise FileNotFoundError(f"No best checkpoint found in {expert_dir}")


def discover_experts(
    weight_dir: Path,
    expert_type: str,
    transform_names: List[str],
) -> Dict[int, Dict]:
    """Discover best expert checkpoints for each augmentation type.

    Returns a dict mapping ``transform_index`` -> metadata dict containing
    ``name``, ``path`` (absolute), ``accuracy``, and ``state_dict``.
    """
    experts: Dict[int, Dict] = {}
    suffix = expert_type  # 'stem' or 'stem-all-bn'

    for t_idx, t_name in enumerate(transform_names):
        dir_name = f"expert_{t_name}_{suffix}"
        expert_dir = weight_dir / dir_name

        if not expert_dir.exists():
            _logger.warning("Expert directory not found: %s", expert_dir)
            continue

        ckpt_path, acc = _find_best_expert_checkpoint(expert_dir)
        abs_path = str(ckpt_path.resolve())
        ckpt = torch.load(ckpt_path, map_location="cpu")

        experts[t_idx] = {
            "name": t_name,
            "path": abs_path,
            "accuracy": acc,
            "state_dict": ckpt["model"],
        }
        _logger.info(
            "Expert [%d] %s: %s (acc=%.2f%%)", t_idx, t_name, abs_path, acc,
        )

    return experts


# =============================================================================
# Aug Classification
# =============================================================================

def _classify_batch(
    images: torch.Tensor,
    targets: torch.Tensor,
    classification_stem: nn.Module,
    classification_backbone: Optional[nn.Module],
    aug_classifier: nn.Module,
    fsc_centroids: torch.Tensor,
    fsc_mode: str,
    fsc_diff_mode: str,
    device: torch.device,
    amp_autocast,
) -> torch.Tensor:
    """Run the aug classifier on one batch and return avg softmax probs.

    Returns:
        avg_probs: ``[num_transforms]`` averaged softmax probabilities.
    """
    with torch.no_grad():
        with amp_autocast():
            stem_features = classification_stem(images)

            if fsc_mode == "mean":
                # 전체 1000개 클래스 FSC의 평균 사용
                # Use the mean of all 1000-class FSC centroids
                fsc_mean = fsc_centroids.mean(dim=0, keepdim=True)
                fsc_for_batch = fsc_mean.expand(images.size(0), -1)
            elif fsc_mode == "predicted":
                # 한번의 Forward를 통해 predicted label로 FSC 선택
                # One forward pass to get predicted labels, then select FSC
                assert classification_backbone is not None, (
                    "classification_backbone is required for fsc_mode='predicted'"
                )
                logits = classification_backbone(images)
                pred_labels = logits.argmax(dim=1)
                fsc_for_batch = fsc_centroids[pred_labels]
            else:
                raise ValueError(f"Unknown fsc_mode: {fsc_mode}")

            fsc_diff = compute_fsc_diff(
                stem_features, fsc_for_batch, mode=fsc_diff_mode,
            )
            aug_pred = aug_classifier(fsc_diff)
            aug_probs = F.softmax(aug_pred, dim=1)
            avg_probs = aug_probs.mean(dim=0)

    return avg_probs


def _classify_corruption(
    loader: DataLoader,
    classification_stem: nn.Module,
    classification_backbone: Optional[nn.Module],
    aug_classifier: nn.Module,
    fsc_centroids: torch.Tensor,
    fsc_mode: str,
    fsc_diff_mode: str,
    device: torch.device,
    amp_autocast,
) -> torch.Tensor:
    """Classify all batches in a corruption and return aggregated probs.

    Returns:
        avg_probs: ``[num_transforms]`` averaged softmax probabilities.
    """
    all_probs: List[torch.Tensor] = []
    for images, target in loader:
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        batch_probs = _classify_batch(
            images, target,
            classification_stem, classification_backbone,
            aug_classifier, fsc_centroids,
            fsc_mode, fsc_diff_mode, device, amp_autocast,
        )
        all_probs.append(batch_probs)

    avg_probs = torch.stack(all_probs).mean(dim=0)
    return avg_probs


# =============================================================================
# Expert Weight Interpolation  (5 Methods for BN Running Statistics)
# =============================================================================

def _interpolate_expert_weights(
    expert1_state: Dict[str, torch.Tensor],
    expert2_state: Dict[str, torch.Tensor],
    w1: float,
    w2: float,
    pretrained_state: Dict[str, torch.Tensor],
    bn_mode: str,
) -> Dict[str, torch.Tensor]:
    """Interpolate two expert state dicts with the specified BN statistics handling.

    Args:
        expert1_state: Top-1 expert's state dict (partial, only modified layers).
        expert2_state: Top-2 expert's state dict (partial, only modified layers).
        w1: Weight for expert1 (from re-normalized softmax probability).
        w2: Weight for expert2 (from re-normalized softmax probability).
        pretrained_state: Full pretrained model state dict.
        bn_mode: One of five BN statistics interpolation methods.

    Returns:
        Interpolated partial state dict (only expert-modified keys).
    """
    interpolated: Dict[str, torch.Tensor] = {}
    all_keys = set(expert1_state.keys()) | set(expert2_state.keys())

    for key in all_keys:
        # 누락된 키는 pretrained 값으로 대체
        # Use pretrained values as fallback for missing keys
        v1 = expert1_state.get(key, pretrained_state.get(key))
        v2 = expert2_state.get(key, pretrained_state.get(key))

        if v1 is None or v2 is None:
            continue

        is_running_mean = "running_mean" in key
        is_running_var = "running_var" in key
        is_running_stat = is_running_mean or is_running_var
        is_num_batches = "num_batches_tracked" in key

        # num_batches_tracked는 카운터이므로 보간하지 않고 큰 값을 사용
        # num_batches_tracked is a counter, not a learned parameter; use the max
        if is_num_batches:
            interpolated[key] = torch.max(v1, v2)
            continue

        if bn_mode == "linear_all":
            # -----------------------------------------------------------
            # Method 1: 모든 파라미터 선형 보간 (Linear interpolation of ALL parameters)
            # 학습된 파라미터(weight, bias)와 러닝 통계(running_mean, running_var)를
            # 모두 동일한 가중치(w1, w2)로 선형 보간합니다.
            # 장점: 단순하고 일관성 있음
            # 단점: running statistics는 확률 분포의 모멘트이므로 선형 보간이
            #       통계적으로 정확하지 않을 수 있음
            #
            # Linearly interpolates ALL parameters including running statistics
            # with the same weights (w1, w2).
            # Pros: Simple and consistent
            # Cons: Running statistics are moments of probability distributions,
            #       so linear interpolation may not be statistically precise
            # -----------------------------------------------------------
            interpolated[key] = w1 * v1.float() + w2 * v2.float()

        elif bn_mode == "top1_stats":
            # -----------------------------------------------------------
            # Method 2: Top-1 expert의 러닝 통계 사용 (Top-1 expert's running stats)
            # 학습된 파라미터(weight, bias)는 선형 보간하되,
            # running_mean과 running_var는 확률이 가장 높은 Top-1 expert의 값을
            # 그대로 사용합니다.
            # 장점: 지배적인 expert의 통계를 그대로 보존
            # 단점: Top-2 expert의 통계 정보가 완전히 무시됨
            #
            # Interpolates learnable parameters (weight, bias) linearly,
            # but uses Top-1 expert's running_mean and running_var as-is.
            # Pros: Preserves the dominant expert's distribution statistics
            # Cons: Completely ignores Top-2 expert's statistical properties
            # -----------------------------------------------------------
            if is_running_stat:
                interpolated[key] = v1.clone()
            else:
                interpolated[key] = w1 * v1.float() + w2 * v2.float()

        elif bn_mode == "pretrained_stats":
            # -----------------------------------------------------------
            # Method 3: 사전학습 모델의 러닝 통계 사용 (Pretrained model's running stats)
            # 학습된 파라미터는 선형 보간하되, running statistics는
            # 원래 사전학습된 모델의 값을 사용합니다.
            # 이는 "중립적인" 기준 통계를 제공합니다.
            # 장점: expert 학습에 의한 통계 편향을 피할 수 있음
            # 단점: augmentation 적응과 관련된 통계 정보가 손실됨
            #
            # Interpolates learnable parameters but uses the original pretrained
            # model's running statistics as a neutral baseline.
            # Pros: Avoids potential statistical bias from expert fine-tuning
            # Cons: Loses augmentation-adapted statistical information
            # -----------------------------------------------------------
            if is_running_stat:
                interpolated[key] = pretrained_state[key].clone()
            else:
                interpolated[key] = w1 * v1.float() + w2 * v2.float()

        elif bn_mode == "batch_stats":
            # -----------------------------------------------------------
            # Method 4: 배치 통계 사용 (Use batch statistics at inference time)
            # 학습된 파라미터는 선형 보간하고, 추론 시 running statistics 대신
            # 현재 배치의 실시간 통계(batch mean, batch variance)를 사용합니다.
            # BN 레이어를 train 모드로 설정하고 track_running_stats=False로 합니다.
            # 장점: 현재 입력 데이터 분포에 가장 적응적
            # 단점: 배치 크기가 작으면 통계가 불안정할 수 있음;
            #       추론 결과가 배치 구성에 따라 달라짐 (비결정적)
            #
            # Interpolates learnable parameters and uses live batch statistics
            # (batch mean/variance) instead of stored running statistics.
            # BN layers are set to train mode with track_running_stats=False.
            # Pros: Most adaptive to the current input data distribution
            # Cons: Unstable with small batch sizes; results depend on batch
            #       composition (non-deterministic)
            # -----------------------------------------------------------
            # running stats는 placeholder로 보간 (실제 추론 시 무시됨)
            # Running stats are interpolated as placeholder (ignored at inference)
            interpolated[key] = w1 * v1.float() + w2 * v2.float()

        elif bn_mode == "geometric_stats":
            # -----------------------------------------------------------
            # Method 5: 기하 보간 (Geometric interpolation for variance)
            # running_var에 대해 기하 평균(geometric mean)을 사용하고,
            # running_mean에 대해서는 선형 보간을 사용합니다.
            # 분산(variance)은 본질적으로 곱셈적(multiplicative) 성질을 가지므로
            # 기하 평균이 더 적합할 수 있습니다.
            #   var_interp = var1^w1 * var2^w2
            #   mean_interp = w1 * mean1 + w2 * mean2
            # 장점: 분산의 곱셈적 특성을 반영하여 더 통계적으로 타당
            # 단점: 구현이 복잡하고, 선형 보간 대비 큰 차이가 없을 수 있음
            #
            # Uses geometric mean for running_var (since variance has
            # multiplicative properties) and linear interpolation for
            # running_mean.
            #   var_interp = var1^w1 * var2^w2
            #   mean_interp = w1 * mean1 + w2 * mean2
            # Pros: Respects the multiplicative nature of variance
            # Cons: More complex; may not differ significantly from linear
            # -----------------------------------------------------------
            if is_running_var:
                interpolated[key] = (
                    v1.float().clamp(min=1e-10).pow(w1)
                    * v2.float().clamp(min=1e-10).pow(w2)
                )
            elif is_running_mean:
                interpolated[key] = w1 * v1.float() + w2 * v2.float()
            else:
                interpolated[key] = w1 * v1.float() + w2 * v2.float()

        else:
            raise ValueError(f"Unknown bn_mode: {bn_mode}")

    return interpolated


# =============================================================================
# Expert Weight Application
# =============================================================================

def _apply_expert_weights(
    model: nn.Module,
    aug_probs: torch.Tensor,
    experts: Dict[int, Dict],
    pretrained_state: Dict[str, torch.Tensor],
    transform_names: List[str],
    weight_mode: str,
    threshold: float,
    bn_mode: str,
) -> Tuple[str, Dict]:
    """Apply expert weights to *model* based on aug classifier probabilities.

    For ``top1`` mode the expert state dict is loaded directly (full replacement
    of the expert-modified layers).  For ``top2`` / ``top2-threshold`` modes the
    two expert state dicts are interpolated according to their softmax
    probabilities.

    Returns:
        description: Human-readable string describing the applied expert(s).
        info: Dict with detailed information for logging / analysis.
    """
    sorted_probs, sorted_indices = aug_probs.sort(descending=True)

    top1_idx = sorted_indices[0].item()
    top1_prob = sorted_probs[0].item()
    top1_name = transform_names[top1_idx]

    info: Dict = {
        "top1_idx": top1_idx,
        "top1_name": top1_name,
        "top1_prob": top1_prob,
    }

    # ---- Reset to pretrained first ----
    model.load_state_dict(pretrained_state)

    # ---- Top-1: unconditional single expert replacement ----
    if weight_mode == "top1":
        if top1_idx in experts:
            expert = experts[top1_idx]
            model.load_state_dict(expert["state_dict"], strict=False)
            description = (
                f"Top-1: {top1_name} (prob={top1_prob:.4f}) "
                f"| weight={expert['path']}"
            )
            _logger.info("Applied expert: %s", description)
        else:
            description = (
                f"Top-1: {top1_name} (prob={top1_prob:.4f}) "
                f"- NOT FOUND, using pretrained"
            )
            _logger.warning(description)
        info["mode"] = "top1"
        return description, info

    # ---- Top-2 preparation ----
    top2_idx = sorted_indices[1].item()
    top2_prob = sorted_probs[1].item()
    top2_name = transform_names[top2_idx]

    info["top2_idx"] = top2_idx
    info["top2_name"] = top2_name
    info["top2_prob"] = top2_prob

    # Decide whether to interpolate
    use_interpolation = False
    if weight_mode == "top2":
        # 항상 Top-2로 보간
        # Always interpolate Top-2 regardless of threshold
        use_interpolation = True
    elif weight_mode == "top2-threshold":
        # Top-2 확률이 threshold 이상인 경우에만 보간
        # Interpolate only when Top-2 probability >= threshold
        use_interpolation = top2_prob >= threshold

    if use_interpolation and top1_idx in experts and top2_idx in experts:
        # Re-normalize probabilities so that w1 + w2 = 1
        total = top1_prob + top2_prob
        w1 = top1_prob / total
        w2 = top2_prob / total

        interpolated = _interpolate_expert_weights(
            experts[top1_idx]["state_dict"],
            experts[top2_idx]["state_dict"],
            w1, w2, pretrained_state, bn_mode,
        )
        model.load_state_dict(interpolated, strict=False)

        # batch_stats 모드: BN을 train 모드로 전환하여 배치 통계 사용
        # batch_stats mode: switch BN to train mode to use batch statistics
        if bn_mode == "batch_stats":
            for m in model.modules():
                if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    m.training = True
                    m.track_running_stats = False

        description = (
            f"Interpolated: {top1_name}({w1:.4f}) + {top2_name}({w2:.4f}), "
            f"bn_mode={bn_mode} | "
            f"w1={experts[top1_idx]['path']} | "
            f"w2={experts[top2_idx]['path']}"
        )
        info["mode"] = "interpolated"
        info["w1"] = w1
        info["w2"] = w2
        _logger.info("Applied experts: %s", description)

    else:
        # Fallback to top-1 only
        if top1_idx in experts:
            expert = experts[top1_idx]
            model.load_state_dict(expert["state_dict"], strict=False)
            description = (
                f"Top-1 (fallback, top2_prob={top2_prob:.4f} < thr={threshold}): "
                f"{top1_name} (prob={top1_prob:.4f}) "
                f"| weight={expert['path']}"
            )
        else:
            description = (
                f"Top-1: {top1_name} (prob={top1_prob:.4f}) "
                f"- NOT FOUND, using pretrained"
            )
        info["mode"] = "top1_fallback"
        _logger.info("Applied expert: %s", description)

    return description, info


def _reset_bn_to_eval(model: nn.Module) -> None:
    """Reset all BN layers back to eval mode after batch_stats inference."""
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            m.training = False
            m.track_running_stats = True


# =============================================================================
# Evaluation Helpers
# =============================================================================

def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    use_batch_stats: bool = False,
) -> Tuple[float, float]:
    """Run standard top-1 / top-5 evaluation on *loader*.

    Args:
        use_batch_stats: If True, keep BN in train mode (batch statistics).
    """
    top1_m = AverageMeter()
    top5_m = AverageMeter()

    model.eval()
    # batch_stats 모드일 때 BN만 train 모드로 전환
    # For batch_stats mode, override BN layers to train mode
    if use_batch_stats:
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.train()

    if amp_enabled:
        autocast_ctx = torch.autocast
        autocast_kwargs = dict(device_type=device.type)
    else:
        autocast_ctx = nullcontext
        autocast_kwargs = {}

    with torch.no_grad():
        for images, target in loader:
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            with autocast_ctx(**autocast_kwargs):
                output = model(images)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            batch_size = images.size(0)
            top1_m.update(acc1.item(), batch_size)
            top5_m.update(acc5.item(), batch_size)

    return top1_m.avg, top5_m.avg


def _evaluate_per_corruption(
    model: nn.Module,
    loader: DataLoader,
    classification_stem: nn.Module,
    classification_backbone: Optional[nn.Module],
    aug_classifier: nn.Module,
    fsc_centroids: torch.Tensor,
    experts: Dict[int, Dict],
    pretrained_state: Dict[str, torch.Tensor],
    transform_names: List[str],
    args: argparse.Namespace,
    fsc_diff_mode: str,
    device: torch.device,
    amp_autocast,
) -> Tuple[float, float, str]:
    """Per-corruption evaluation: classify the entire corruption, then evaluate.

    1. Run all batches through aug classifier -> aggregate probabilities.
    2. Select and apply expert weight(s).
    3. Run evaluation on all batches.

    Returns:
        (top1, top5, expert_description)
    """
    # --- Classification phase ---
    aug_probs = _classify_corruption(
        loader, classification_stem, classification_backbone,
        aug_classifier, fsc_centroids,
        args.fsc_mode, fsc_diff_mode, device, amp_autocast,
    )
    _logger.info(
        "Aug classifier probs: %s",
        {n: f"{p:.4f}" for n, p in zip(transform_names, aug_probs.tolist())},
    )

    # --- Apply expert weights ---
    description, info = _apply_expert_weights(
        model, aug_probs, experts, pretrained_state,
        transform_names, args.weight_mode, args.threshold, args.bn_mode,
    )

    # --- Evaluation phase ---
    use_batch_stats = (
        args.bn_mode == "batch_stats" and args.weight_mode != "top1"
    )
    top1, top5 = _evaluate(model, loader, device, args.amp, use_batch_stats)

    # Reset BN after batch_stats usage
    if use_batch_stats:
        _reset_bn_to_eval(model)

    return top1, top5, description


def _evaluate_per_batch(
    model: nn.Module,
    loader: DataLoader,
    classification_stem: nn.Module,
    classification_backbone: Optional[nn.Module],
    aug_classifier: nn.Module,
    fsc_centroids: torch.Tensor,
    experts: Dict[int, Dict],
    pretrained_state: Dict[str, torch.Tensor],
    transform_names: List[str],
    args: argparse.Namespace,
    fsc_diff_mode: str,
    device: torch.device,
    amp_autocast,
) -> Tuple[float, float, str]:
    """Per-batch evaluation: classify and evaluate each batch independently.

    For each batch:
    1. Run through aug classifier -> determine expert.
    2. Apply expert weight(s).
    3. Evaluate this batch.

    Returns:
        (top1, top5, summary_description)
    """
    top1_m = AverageMeter()
    top5_m = AverageMeter()

    # Expert 선택 빈도 추적
    # Track expert selection frequency
    expert_counts: Dict[str, int] = {}
    last_applied_idx: Optional[int] = None
    use_batch_stats = (
        args.bn_mode == "batch_stats" and args.weight_mode != "top1"
    )

    if amp_autocast is nullcontext:
        eval_autocast_ctx = nullcontext
        eval_autocast_kwargs: Dict = {}
    else:
        eval_autocast_ctx = torch.autocast
        eval_autocast_kwargs = dict(device_type=device.type)

    for batch_idx, (images, target) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # --- Classification ---
        aug_probs = _classify_batch(
            images, target,
            classification_stem, classification_backbone,
            aug_classifier, fsc_centroids,
            args.fsc_mode, fsc_diff_mode, device, amp_autocast,
        )

        # --- Decide expert ---
        sorted_probs, sorted_indices = aug_probs.sort(descending=True)
        top1_idx = sorted_indices[0].item()
        top1_name = transform_names[top1_idx]
        expert_counts[top1_name] = expert_counts.get(top1_name, 0) + 1

        # Optimization: skip weight reload if same top-1 expert for top1 mode
        need_reload = True
        if args.weight_mode == "top1" and top1_idx == last_applied_idx:
            need_reload = False

        if need_reload:
            _, info = _apply_expert_weights(
                model, aug_probs, experts, pretrained_state,
                transform_names, args.weight_mode, args.threshold, args.bn_mode,
            )
            last_applied_idx = top1_idx if args.weight_mode == "top1" else None

        # --- Evaluate this batch ---
        model.eval()
        if use_batch_stats:
            for m in model.modules():
                if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    m.train()

        with torch.no_grad():
            with eval_autocast_ctx(**eval_autocast_kwargs):
                output = model(images)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        top1_m.update(acc1.item(), images.size(0))
        top5_m.update(acc5.item(), images.size(0))

    # Reset BN
    if use_batch_stats:
        _reset_bn_to_eval(model)

    # 가장 많이 선택된 expert를 요약으로 출력
    # Summary: most frequently selected expert
    dominant_expert = max(expert_counts, key=expert_counts.get) if expert_counts else "N/A"
    description = (
        f"Per-batch dominant expert: {dominant_expert} "
        f"(counts: {expert_counts})"
    )
    _logger.info(description)

    return top1_m.avg, top5_m.avg, description


# =============================================================================
# Results I/O
# =============================================================================

def _generate_results_filename(args: argparse.Namespace) -> str:
    """Generate a unique results filename based on run options."""
    parts = ["expert_val"]
    parts.append(args.expert_type)
    parts.append(args.weight_mode)
    parts.append(args.classify_mode)
    parts.append(f"fsc-{args.fsc_mode}")

    if args.weight_mode != "top1":
        parts.append(f"bn-{args.bn_mode}")
    if args.weight_mode == "top2-threshold":
        parts.append(f"thr-{args.threshold}")

    parts.append(f"sev{args.severity}")
    return "_".join(parts) + ".txt"


def _save_results(
    results: List[Dict],
    args: argparse.Namespace,
    expert_descriptions: Dict[str, str],
) -> None:
    """Save results to a tab-separated .txt file in --results-dir."""
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    filename = _generate_results_filename(args)
    filepath = results_dir / filename

    with filepath.open("w") as f:
        # Header
        f.write("Corruption\tTop1\tTop5\n")

        for r in results:
            f.write(f"{r['corruption']}\t{r['top1']:.3f}\t{r['top5']:.3f}\n")

        # Mean row
        if results:
            mean_top1 = sum(r["top1"] for r in results) / len(results)
            mean_top5 = sum(r["top5"] for r in results) / len(results)
            f.write(f"mean\t{mean_top1:.3f}\t{mean_top5:.3f}\n")

        # Expert selection summary (appended as comments)
        f.write("\n# --- Expert Selection Summary ---\n")
        for corr, desc in expert_descriptions.items():
            f.write(f"# {corr}: {desc}\n")

        # Run configuration summary
        f.write("\n# --- Run Configuration ---\n")
        f.write(f"# model: {args.model}\n")
        f.write(f"# expert_type: {args.expert_type}\n")
        f.write(f"# weight_mode: {args.weight_mode}\n")
        f.write(f"# classify_mode: {args.classify_mode}\n")
        f.write(f"# fsc_mode: {args.fsc_mode}\n")
        f.write(f"# fsc_diff_mode: {args.fsc_diff_mode}\n")
        f.write(f"# bn_mode: {args.bn_mode}\n")
        f.write(f"# threshold: {args.threshold}\n")
        f.write(f"# severity: {args.severity}\n")
        f.write(f"# batch_size: {args.batch_size}\n")

    _logger.info("Results saved to: %s", filepath.resolve())


# =============================================================================
# Argument Parser
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ImageNet-C Expert Validation using Aug Classifier",
    )

    # --- Data ---
    g = parser.add_argument_group("Data")
    g.add_argument("--imagenet-c-dir", required=True, type=str,
                    help="Root directory of ImageNet-C dataset.")
    g.add_argument("--imagenet-val-dir", required=True, type=str,
                    help="ImageNet validation directory (for label mapping).")
    g.add_argument("--corruptions", nargs="+", default=None, type=str,
                    help="Corruption names to evaluate (default: all 15).")
    g.add_argument("--severity", default=5, type=int,
                    help="Severity level to evaluate (default: 5).")

    # --- Model ---
    g = parser.add_argument_group("Model")
    g.add_argument("--model", default="resnet50", type=str)
    g.add_argument("--pretrained", action="store_true", default=True)
    g.add_argument("--num-classes", default=1000, type=int)
    g.add_argument("--input-size", nargs=3, default=None, type=int,
                    help="Override model input size (C H W).")
    g.add_argument("--mean", nargs="+", default=None, type=float)
    g.add_argument("--std", nargs="+", default=None, type=float)
    g.add_argument("--crop-pct", default=None, type=float)
    g.add_argument("--interpolation", default="", type=str)

    # --- Aug Classifier ---
    g = parser.add_argument_group("Aug Classifier")
    g.add_argument(
        "--aug-classifier-ckpt",
        default="./output/STEM_KLDIV_ORTHOGONAL/best.pth.tar",
        type=str,
        help="Path to trained aug classifier checkpoint.",
    )
    g.add_argument("--fsc-path", required=True, type=str,
                    help="Path to Stem FSC file (e.g., ./FSC/resnet50_FSC_stem.pth).")
    g.add_argument(
        "--fsc-diff-mode", default="orthogonal", type=str,
        choices=["subtract", "orthogonal"],
        help="FSC difference computation mode (default: orthogonal).",
    )
    g.add_argument(
        "--aug-classifier-hidden", nargs="+", default=[512, 256, 128], type=int,
        help="Hidden dims for aug classifier MLP (default: 512 256 128). "
             "Overridden by checkpoint args if available.",
    )

    # --- Expert Weights ---
    g = parser.add_argument_group("Expert Weights")
    g.add_argument(
        "--expert-weight-dir",
        default="/home/oem/jin/SOA/new_weight",
        type=str,
        help="Directory containing expert weight subdirectories.",
    )
    g.add_argument(
        "--expert-type", default="stem", type=str,
        choices=["stem", "stem-all-bn"],
        help="Expert weight type: 'stem' (conv1+bn1 only) or "
             "'stem-all-bn' (conv1 + all BN layers) (default: stem).",
    )

    # --- Weight Application Mode ---
    g = parser.add_argument_group("Weight Application")
    g.add_argument(
        "--weight-mode", default="top1", type=str,
        choices=["top1", "top2", "top2-threshold"],
        help="Expert weight selection mode (default: top1).",
    )
    g.add_argument(
        "--threshold", default=0.3, type=float,
        help="Probability threshold for top2-threshold mode (default: 0.3).",
    )
    g.add_argument(
        "--bn-mode", default="linear_all", type=str,
        choices=[
            "linear_all",
            "top1_stats",
            "pretrained_stats",
            "batch_stats",
            "geometric_stats",
        ],
        help="BN running statistics interpolation method for top2 modes "
             "(default: linear_all). Ignored for top1 mode.",
    )

    # --- Classification Mode ---
    g = parser.add_argument_group("Classification Mode")
    g.add_argument(
        "--classify-mode", default="per-corruption", type=str,
        choices=["per-corruption", "per-batch"],
        help="Aug classification granularity (default: per-corruption).",
    )
    g.add_argument(
        "--fsc-mode", default="mean", type=str,
        choices=["mean", "predicted"],
        help="FSC centroid selection: 'mean' (average of all 1000 classes) or "
             "'predicted' (forward pass to get predicted label) (default: mean).",
    )

    # --- Runtime ---
    g = parser.add_argument_group("Runtime")
    g.add_argument("--batch-size", default=64, type=int)
    g.add_argument("--workers", default=8, type=int)
    g.add_argument("--device", default="cuda", type=str)
    g.add_argument("--amp", action="store_true",
                    help="Use AMP (mixed precision) for inference.")

    # --- Output ---
    g = parser.add_argument_group("Output")
    g.add_argument(
        "--results-dir", default="./results", type=str,
        help="Directory to save results .txt file.",
    )

    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    setup_default_logging()
    args = parse_args()

    device = torch.device(args.device)
    if torch.cuda.is_available() and args.device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    # AMP setup
    amp_autocast = nullcontext
    if args.amp:
        from functools import partial
        amp_autocast = partial(
            torch.autocast, device_type=device.type, dtype=torch.float16,
        )

    # ---- V2 transform names ----
    transform_names = list(get_augmix_sl_transform_names(version=2))
    num_transforms = get_augmix_sl_num_transforms(version=2)
    _logger.info("V2 transforms (%d): %s", num_transforms, transform_names)

    # ================================================================
    # Create pretrained model
    # ================================================================
    _logger.info("Creating pretrained %s ...", args.model)
    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.num_classes,
    )
    model.to(device)
    model.eval()
    pretrained_state = copy.deepcopy(model.state_dict())

    # ================================================================
    # Classification components (frozen, independent copies)
    # ================================================================
    # Classification stem: pretrained 가중치를 항상 유지하는 독립 사본
    # Classification stem: independent frozen copy that always uses pretrained weights
    classification_stem = copy.deepcopy(StemFeatureExtractor(model))
    classification_stem.to(device)
    classification_stem.eval()
    for p in classification_stem.parameters():
        p.requires_grad = False

    classification_backbone: Optional[nn.Module] = None
    if args.fsc_mode == "predicted":
        # predicted FSC 모드: 전체 backbone 사본으로 예측 label 산출
        # predicted FSC mode: full backbone copy for getting predicted labels
        classification_backbone = copy.deepcopy(model)
        classification_backbone.to(device)
        classification_backbone.eval()
        for p in classification_backbone.parameters():
            p.requires_grad = False
        _logger.info("Created classification backbone copy for predicted FSC mode.")

    # ================================================================
    # FSC centroids
    # ================================================================
    _logger.info("Loading FSC centroids from: %s", args.fsc_path)
    fsc_data = torch.load(args.fsc_path, map_location="cpu")
    fsc_centroids = fsc_data["centroids"].to(device)
    feature_dim = fsc_data["feature_dim"]
    _logger.info("FSC centroids: %s, feature_dim=%d", fsc_centroids.shape, feature_dim)

    # ================================================================
    # Aug Classifier
    # ================================================================
    aug_ckpt_path = Path(args.aug_classifier_ckpt)
    if not aug_ckpt_path.exists():
        raise FileNotFoundError(
            f"Aug classifier checkpoint not found: {aug_ckpt_path.resolve()}"
        )
    _logger.info("Loading aug classifier from: %s", aug_ckpt_path.resolve())
    aug_ckpt = torch.load(aug_ckpt_path, map_location="cpu")

    # checkpoint에 저장된 학습 args에서 설정 추출
    # Extract config from checkpoint's saved training args
    aug_train_args = aug_ckpt.get("args", None)
    hidden_dims = list(args.aug_classifier_hidden)
    fsc_diff_mode = args.fsc_diff_mode

    if aug_train_args is not None:
        if hasattr(aug_train_args, "aug_classifier_hidden"):
            hidden_dims = list(aug_train_args.aug_classifier_hidden)
            _logger.info(
                "Overriding aug_classifier_hidden from checkpoint: %s",
                hidden_dims,
            )
        if hasattr(aug_train_args, "fsc_diff_mode"):
            fsc_diff_mode = aug_train_args.fsc_diff_mode
            _logger.info(
                "Overriding fsc_diff_mode from checkpoint: %s", fsc_diff_mode,
            )

    aug_classifier = AugClassifier(
        feature_dim=feature_dim,
        num_transforms=num_transforms,
        hidden_dims=hidden_dims,
        dropout=0.1,
        use_sigmoid=False,
    )
    aug_classifier.load_state_dict(aug_ckpt["aug_classifier"])
    aug_classifier.to(device)
    aug_classifier.eval()
    for p in aug_classifier.parameters():
        p.requires_grad = False
    _logger.info(
        "Aug classifier loaded (hidden=%s, fsc_diff_mode=%s)",
        hidden_dims, fsc_diff_mode,
    )

    # ================================================================
    # Discover expert checkpoints
    # ================================================================
    _logger.info(
        "Discovering experts in %s (type=%s) ...",
        args.expert_weight_dir, args.expert_type,
    )
    experts = discover_experts(
        Path(args.expert_weight_dir), args.expert_type, transform_names,
    )
    if not experts:
        raise RuntimeError(
            f"No expert checkpoints found in {args.expert_weight_dir} "
            f"for type '{args.expert_type}'"
        )

    # ================================================================
    # Label map & data transform
    # ================================================================
    filename_label_map, class_to_idx = _build_label_map(
        Path(args.imagenet_val_dir),
    )
    data_config = resolve_data_config(vars(args), model=model)
    transform = create_transform(**data_config, is_training=False)

    # ================================================================
    # Evaluate each corruption
    # ================================================================
    corruptions = args.corruptions or DEFAULT_CORRUPTIONS
    results: List[Dict] = []
    expert_descriptions: Dict[str, str] = {}

    _logger.info(
        "\n%s\nStarting evaluation: %d corruptions, severity=%d\n"
        "  expert_type=%s, weight_mode=%s, classify_mode=%s, fsc_mode=%s, "
        "bn_mode=%s, threshold=%.2f\n%s",
        "=" * 70, len(corruptions), args.severity,
        args.expert_type, args.weight_mode, args.classify_mode, args.fsc_mode,
        args.bn_mode, args.threshold, "=" * 70,
    )

    for corr_idx, corruption in enumerate(corruptions):
        # Find corruption directory with severity sub-folder
        corr_dir = Path(args.imagenet_c_dir) / corruption / str(args.severity)
        if not corr_dir.exists():
            # Fallback: no severity sub-folder
            corr_dir = Path(args.imagenet_c_dir) / corruption
        if not corr_dir.exists():
            _logger.warning("Skip missing corruption: %s", corruption)
            continue

        _logger.info(
            "\n[%d/%d] Evaluating: %s (severity=%d, dir=%s)",
            corr_idx + 1, len(corruptions), corruption, args.severity, corr_dir,
        )

        dataset = ImageNetCDataset(
            corr_dir, filename_label_map, class_to_idx, transform,
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
        )

        if args.classify_mode == "per-corruption":
            top1, top5, desc = _evaluate_per_corruption(
                model, loader,
                classification_stem, classification_backbone,
                aug_classifier, fsc_centroids,
                experts, pretrained_state, transform_names,
                args, fsc_diff_mode, device, amp_autocast,
            )
        else:  # per-batch
            top1, top5, desc = _evaluate_per_batch(
                model, loader,
                classification_stem, classification_backbone,
                aug_classifier, fsc_centroids,
                experts, pretrained_state, transform_names,
                args, fsc_diff_mode, device, amp_autocast,
            )

        _logger.info(
            ">>> %s: Top1=%.3f%%, Top5=%.3f%% (%d images)",
            corruption, top1, top5, len(dataset),
        )
        results.append(
            {"corruption": corruption, "top1": top1, "top5": top5},
        )
        expert_descriptions[corruption] = desc

    # ================================================================
    # Save results
    # ================================================================
    _save_results(results, args, expert_descriptions)

    # ================================================================
    # Print summary
    # ================================================================
    if results:
        mean_top1 = sum(r["top1"] for r in results) / len(results)
        mean_top5 = sum(r["top5"] for r in results) / len(results)
        _logger.info(
            "\n%s\nFinal Mean: Top1=%.3f%%, Top5=%.3f%%\n%s",
            "=" * 70, mean_top1, mean_top5, "=" * 70,
        )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""ImageNet-C Expert TTA (Test-Time Adaptation) Validation Script.

Instead of directly loading full expert weights for the stem, this script
*trains* (adapts) the stem layer at test time using the aug classifier
prediction as hard-label supervision.

Pipeline:
1. Load pretrained ResNet50, Aug Classifier, and FSC centroids.
2. Discover best expert checkpoints per augmentation type.
3. For each corruption (severity=5 by default):
   For each mini-batch:
   a. Classify augmentation via frozen classification_stem -> FSC diff -> Aug Classifier.
   b. Determine Top-1 augmentation as hard-label target.
   c. Prepare model: load expert weights (conv1 for stem, backbone BN for stem-all-bn).
   d. TTA-train the stem until target_threshold or max iterations.
   e. Evaluate classification accuracy on the same mini-batch.
4. Save results to .txt file.

Expert type behaviour:
    stem        : conv1 loaded from expert (frozen), only bn1 TTA-trained.
    stem-all-bn : backbone BN loaded from expert (frozen), conv1+bn1 TTA-trained.

Usage examples:
    # stem type (BN-only TTA, conv1 from expert)
    python validate_imagenet_c_expert_tta.py \\
        --imagenet-c-dir /path/to/ImageNet-C \\
        --imagenet-val-dir /path/to/imagenet/val \\
        --aug-classifier-ckpt ./output/STEM_KLDIV_ORTHOGONAL/best.pth.tar \\
        --fsc-path ./FSC/resnet50_FSC_stem.pth \\
        --expert-weight-dir /home/oem/jin/SOA/new_weight \\
        --expert-type stem \\
        --tta-mode static \\
        --target-threshold 0.95 \\
        --tta-lr 0.05 \\
        --results-dir ./results

    # stem-all-bn type (conv1+BN TTA, backbone BN from expert)
    python validate_imagenet_c_expert_tta.py \\
        --imagenet-c-dir /path/to/ImageNet-C \\
        --imagenet-val-dir /path/to/imagenet/val \\
        --aug-classifier-ckpt ./output/STEM_KLDIV_ORTHOGONAL/best.pth.tar \\
        --fsc-path ./FSC/resnet50_FSC_stem.pth \\
        --expert-weight-dir /home/oem/jin/SOA/new_weight \\
        --expert-type stem-all-bn \\
        --tta-mode continue \\
        --bn-train-mode affine-and-running \\
        --target-threshold 0.95 \\
        --tta-lr 0.05 \\
        --results-dir ./results
"""

import argparse
import copy
import logging
import re
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

_logger = logging.getLogger("validate_imagenet_c_expert_tta")

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
                _logger.warning(
                    "Skipped %d files without labels in %s", missing, root,
                )

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


def _build_label_map(
    val_dir: Path,
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Build mappings for filenames and class dirs from ImageNet val."""
    from torchvision import datasets

    val_dataset = datasets.ImageFolder(val_dir)
    filename_label_map = {
        Path(p).name: target for p, target in val_dataset.samples
    }
    class_to_idx = val_dataset.class_to_idx
    _logger.info(
        "Loaded label map for %d validation images", len(filename_label_map),
    )
    return filename_label_map, class_to_idx


# =============================================================================
# Expert Checkpoint Discovery
# =============================================================================

def _find_best_expert_checkpoint(
    expert_dir: Path,
) -> Tuple[Path, float]:
    """Find the best checkpoint in an expert directory.

    Looks for ``best_ep{epoch}_{accuracy}.pth.tar`` and picks the highest
    accuracy.  Falls back to ``best.pth.tar`` when no epoch-named files exist.
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

    Returns ``{transform_index: {name, path, accuracy, state_dict}}``.
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
# Aug Classification  (frozen, per-batch, fsc_mode=mean only)
# =============================================================================

def _classify_batch_frozen(
    images: torch.Tensor,
    classification_stem: nn.Module,
    aug_classifier: nn.Module,
    fsc_centroids: torch.Tensor,
    fsc_diff_mode: str,
) -> torch.Tensor:
    """Classify augmentation type using the frozen classification_stem.

    FSC mode is fixed to ``mean`` (average of all 1000-class centroids).

    Returns:
        avg_probs: ``[num_transforms]`` averaged softmax probabilities.
    """
    with torch.no_grad():
        stem_features = classification_stem(images)
        fsc_mean = fsc_centroids.mean(dim=0, keepdim=True)
        fsc_for_batch = fsc_mean.expand(images.size(0), -1)
        fsc_diff = compute_fsc_diff(
            stem_features, fsc_for_batch, mode=fsc_diff_mode,
        )
        aug_logits = aug_classifier(fsc_diff)
        aug_probs = F.softmax(aug_logits, dim=1)
        avg_probs = aug_probs.mean(dim=0)

    return avg_probs


# =============================================================================
# Expert Weight Loading for TTA
# =============================================================================

def _load_expert_for_tta(
    model: nn.Module,
    top1_idx: int,
    experts: Dict[int, Dict],
    expert_type: str,
    transform_names: List[str],
) -> str:
    """Load expert weights appropriate for TTA.

    * ``stem``        : load only **conv1** from expert (bn1 is TTA-trained).
    * ``stem-all-bn`` : load only **backbone BN** (layer1-4 BN) from expert
                        (conv1 + bn1 are TTA-trained).

    Returns:
        Human-readable description of what was loaded.
    """
    if top1_idx not in experts:
        top1_name = (
            transform_names[top1_idx] if top1_idx < len(transform_names) else "?"
        )
        desc = f"Expert {top1_idx} ({top1_name}) NOT FOUND, using pretrained"
        _logger.warning(desc)
        return desc

    expert = experts[top1_idx]
    expert_state = expert["state_dict"]
    top1_name = expert["name"]

    if expert_type == "stem":
        # conv1만 expert에서 로드 (bn1은 TTA 학습 대상)
        conv1_state = {
            k: v for k, v in expert_state.items()
            if k.startswith("conv1")
        }
        model.load_state_dict(conv1_state, strict=False)
        desc = (
            f"Loaded conv1 from expert [{top1_idx}] {top1_name} "
            f"({len(conv1_state)} keys) | {expert['path']}"
        )

    elif expert_type == "stem-all-bn":
        # Backbone BN만 expert에서 로드 (conv1, bn1은 TTA 학습 대상)
        backbone_state = {
            k: v for k, v in expert_state.items()
            if not k.startswith("conv1") and not k.startswith("bn1")
        }
        model.load_state_dict(backbone_state, strict=False)
        desc = (
            f"Loaded backbone BN from expert [{top1_idx}] {top1_name} "
            f"({len(backbone_state)} keys) | {expert['path']}"
        )

    else:
        desc = f"Unknown expert_type: {expert_type}"
        _logger.error(desc)

    _logger.info(desc)
    return desc


# =============================================================================
# TTA Training Helpers
# =============================================================================

def _setup_tta_trainable_params(
    model: nn.Module,
    expert_type: str,
    bn_train_mode: str,
) -> List[nn.Parameter]:
    """Configure which parameters are trainable for TTA and set BN mode.

    For ``stem``        : only ``bn1.weight``, ``bn1.bias`` (conv1 frozen/expert).
    For ``stem-all-bn`` : ``conv1.*`` + ``bn1.weight``, ``bn1.bias``.

    Args:
        bn_train_mode:
            ``affine-only``        – bn1 stays in eval mode; only affine params trained.
            ``affine-and-running`` – bn1 set to train mode; running stats also updated.

    Returns:
        List of trainable parameters (for the optimizer).
    """
    # 전체 파라미터 freeze
    for p in model.parameters():
        p.requires_grad = False

    trainable_params: List[nn.Parameter] = []

    if expert_type == "stem":
        # bn1 affine 파라미터만 학습 (conv1은 expert → frozen)
        for name, p in model.named_parameters():
            if name in ("bn1.weight", "bn1.bias"):
                p.requires_grad = True
                trainable_params.append(p)

    elif expert_type == "stem-all-bn":
        # conv1 전체 + bn1 affine 파라미터 학습
        for name, p in model.named_parameters():
            if name.startswith("conv1"):
                p.requires_grad = True
                trainable_params.append(p)
            elif name in ("bn1.weight", "bn1.bias"):
                p.requires_grad = True
                trainable_params.append(p)

    # BN 모드 설정
    if bn_train_mode == "affine-and-running":
        model.bn1.train()   # 배치 통계 사용 + running stats 갱신
    else:
        model.bn1.eval()    # 저장된 running stats 사용

    return trainable_params


def _tta_stem_forward(
    model: nn.Module,
    images: torch.Tensor,
) -> torch.Tensor:
    """Forward through the model's own stem layers to get stem features.

    Replicates :class:`StemFeatureExtractor` but uses the model's layers
    directly so that gradients flow to trainable stem parameters.

    Returns:
        stem_features: ``[batch_size, 1024]``  (64 * 4 * 4 for ResNet50)
    """
    x = images.contiguous()
    x = model.conv1(x)
    x = model.bn1(x)
    x = model.act1(x)
    x = F.adaptive_avg_pool2d(x, (4, 4))
    return x.flatten(1)


def _tta_adapt_batch(
    model: nn.Module,
    images: torch.Tensor,
    hard_label: int,
    aug_classifier: nn.Module,
    fsc_centroids: torch.Tensor,
    fsc_diff_mode: str,
    trainable_params: List[nn.Parameter],
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[float, int, float]:
    """TTA training loop for one mini-batch.

    Trains the stem so that the frozen aug classifier predicts *hard_label*
    with avg softmax probability >= ``target_threshold``.

    Returns:
        final_prob  – final avg probability for the hard label.
        num_updates – number of gradient updates actually performed.
        final_loss  – loss value at the last computed forward (0.0 if no update).
    """
    # ---- Optimizer (새 배치마다 생성) ----
    if args.tta_optimizer == "sgd":
        optimizer = torch.optim.SGD(
            trainable_params, lr=args.tta_lr, momentum=0.9,
        )
    elif args.tta_optimizer == "adam":
        optimizer = torch.optim.Adam(trainable_params, lr=args.tta_lr)
    elif args.tta_optimizer == "adamw":
        optimizer = torch.optim.AdamW(trainable_params, lr=args.tta_lr)
    else:
        raise ValueError(f"Unknown tta_optimizer: {args.tta_optimizer}")

    hard_label_tensor = torch.full(
        (images.size(0),), hard_label, dtype=torch.long, device=device,
    )
    loss_fn = nn.CrossEntropyLoss()

    # FSC mean (고정, fsc_mode=mean)
    fsc_mean = fsc_centroids.mean(dim=0, keepdim=True).expand(
        images.size(0), -1,
    )

    final_prob = 0.0
    final_loss = 0.0
    num_updates = 0

    for _iter in range(args.tta_max_iter):
        optimizer.zero_grad()

        # ---- Forward: stem → FSC diff → aug_classifier ----
        stem_features = _tta_stem_forward(model, images)
        fsc_diff = compute_fsc_diff(stem_features, fsc_mean, mode=fsc_diff_mode)
        aug_logits = aug_classifier(fsc_diff)       # frozen, graph preserved

        # ---- 확률 확인 (before update) ----
        with torch.no_grad():
            aug_probs = F.softmax(aug_logits.detach(), dim=1)
            final_prob = aug_probs[:, hard_label].mean().item()

        # threshold 달성 시 학습 종료
        if final_prob >= args.target_threshold:
            break

        # ---- Backward & step ----
        loss = loss_fn(aug_logits, hard_label_tensor)
        loss.backward()
        optimizer.step()

        num_updates += 1
        final_loss = loss.item()

    return final_prob, num_updates, final_loss


# =============================================================================
# Per-Corruption TTA Evaluation
# =============================================================================

def _evaluate_corruption_tta(
    model: nn.Module,
    loader: DataLoader,
    classification_stem: nn.Module,
    aug_classifier: nn.Module,
    fsc_centroids: torch.Tensor,
    experts: Dict[int, Dict],
    pretrained_state: Dict[str, torch.Tensor],
    transform_names: List[str],
    args: argparse.Namespace,
    fsc_diff_mode: str,
    device: torch.device,
) -> Tuple[float, float, str]:
    """Per-batch TTA evaluation for one corruption.

    For each mini-batch:
    1. Classify augmentation → Top-1 hard label  (frozen classification_stem)
    2. Prepare model (reset if static / first batch, load expert)
    3. TTA-train stem until threshold or max iterations
    4. Evaluate on the **same** mini-batch

    Returns:
        (mean_top1, mean_top5, summary_description)
    """
    top1_m = AverageMeter()
    top5_m = AverageMeter()

    # 통계 추적
    expert_counts: Dict[str, int] = {}
    total_tta_updates = 0
    total_batches = 0

    for batch_idx, (images, target) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # ==============================================================
        # Step 1: Aug 분류 (frozen classification_stem)
        # ==============================================================
        avg_probs = _classify_batch_frozen(
            images, classification_stem, aug_classifier,
            fsc_centroids, fsc_diff_mode,
        )
        top1_idx = avg_probs.argmax().item()
        top1_name = transform_names[top1_idx]
        top1_prob = avg_probs[top1_idx].item()
        expert_counts[top1_name] = expert_counts.get(top1_name, 0) + 1

        # ==============================================================
        # Step 2: 모델 준비 (static: 매 배치 리셋 / continue: 첫 배치만 리셋)
        # ==============================================================
        if args.tta_mode == "static" or batch_idx == 0:
            model.load_state_dict(pretrained_state)

        # Expert 가중치 로드 (단순 교체, 확률 가중 없음)
        expert_desc = _load_expert_for_tta(
            model, top1_idx, experts, args.expert_type, transform_names,
        )

        # ==============================================================
        # Step 3: TTA stem 학습
        # ==============================================================
        model.eval()  # 전체 eval → _setup_tta_trainable_params 에서 bn1만 전환
        trainable_params = _setup_tta_trainable_params(
            model, args.expert_type, args.bn_train_mode,
        )

        if not trainable_params:
            _logger.warning("  Batch %d: no trainable params, skipping TTA", batch_idx)
            final_prob, num_updates, final_loss = 0.0, 0, 0.0
        else:
            final_prob, num_updates, final_loss = _tta_adapt_batch(
                model, images, top1_idx,
                aug_classifier, fsc_centroids, fsc_diff_mode,
                trainable_params, args, device,
            )

        total_tta_updates += num_updates
        total_batches += 1

        _logger.info(
            "  Batch %d: aug=%s (cls_prob=%.4f), TTA updates=%d, "
            "final_prob=%.4f, loss=%.4f",
            batch_idx, top1_name, top1_prob, num_updates,
            final_prob, final_loss,
        )

        # ==============================================================
        # Step 4: 같은 배치로 평가
        # ==============================================================
        model.eval()
        with torch.no_grad():
            output = model(images)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        top1_m.update(acc1.item(), images.size(0))
        top5_m.update(acc5.item(), images.size(0))

    # ---- 요약 ----
    dominant_expert = (
        max(expert_counts, key=expert_counts.get) if expert_counts else "N/A"
    )
    avg_updates = total_tta_updates / total_batches if total_batches > 0 else 0
    description = (
        f"TTA mode={args.tta_mode}, dominant_aug={dominant_expert}, "
        f"avg_updates={avg_updates:.1f}, counts={expert_counts}"
    )
    _logger.info(description)

    return top1_m.avg, top5_m.avg, description


# =============================================================================
# Results I/O
# =============================================================================

def _generate_results_filename(args: argparse.Namespace) -> str:
    """Generate a unique results filename from run options."""
    parts = ["tta_val"]
    parts.append(args.expert_type)
    parts.append(args.tta_mode)
    parts.append(f"bn-{args.bn_train_mode}")
    parts.append(f"lr{args.tta_lr}")
    parts.append(f"thr{args.target_threshold}")
    parts.append(f"opt-{args.tta_optimizer}")
    parts.append(f"maxiter{args.tta_max_iter}")
    parts.append(f"sev{args.severity}")
    return "_".join(parts) + ".txt"


def _save_results(
    results: List[Dict],
    args: argparse.Namespace,
    expert_descriptions: Dict[str, str],
) -> None:
    """Save results to a tab-separated .txt file in ``--results-dir``."""
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    filename = _generate_results_filename(args)
    filepath = results_dir / filename

    with filepath.open("w") as f:
        f.write("Corruption\tTop1\tTop5\n")

        for r in results:
            f.write(f"{r['corruption']}\t{r['top1']:.3f}\t{r['top5']:.3f}\n")

        if results:
            mean_top1 = sum(r["top1"] for r in results) / len(results)
            mean_top5 = sum(r["top5"] for r in results) / len(results)
            f.write(f"mean\t{mean_top1:.3f}\t{mean_top5:.3f}\n")

        # TTA 요약
        f.write("\n# --- TTA Summary per Corruption ---\n")
        for corr, desc in expert_descriptions.items():
            f.write(f"# {corr}: {desc}\n")

        # 실행 설정
        f.write("\n# --- Run Configuration ---\n")
        f.write(f"# model: {args.model}\n")
        f.write(f"# expert_type: {args.expert_type}\n")
        f.write(f"# tta_mode: {args.tta_mode}\n")
        f.write(f"# bn_train_mode: {args.bn_train_mode}\n")
        f.write(f"# tta_optimizer: {args.tta_optimizer}\n")
        f.write(f"# tta_lr: {args.tta_lr}\n")
        f.write(f"# target_threshold: {args.target_threshold}\n")
        f.write(f"# tta_max_iter: {args.tta_max_iter}\n")
        f.write(f"# fsc_diff_mode: {args.fsc_diff_mode}\n")
        f.write(f"# severity: {args.severity}\n")
        f.write(f"# batch_size: {args.batch_size}\n")

    _logger.info("Results saved to: %s", filepath.resolve())


# =============================================================================
# Argument Parser
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ImageNet-C Expert TTA Validation",
    )

    # --- Data ---
    g = parser.add_argument_group("Data")
    g.add_argument(
        "--imagenet-c-dir", required=True, type=str,
        help="Root directory of ImageNet-C dataset.",
    )
    g.add_argument(
        "--imagenet-val-dir", required=True, type=str,
        help="ImageNet validation directory (for label mapping).",
    )
    g.add_argument(
        "--corruptions", nargs="+", default=None, type=str,
        help="Corruption names to evaluate (default: all 15).",
    )
    g.add_argument(
        "--severity", default=5, type=int,
        help="Severity level to evaluate (default: 5).",
    )

    # --- Model ---
    g = parser.add_argument_group("Model")
    g.add_argument("--model", default="resnet50", type=str)
    g.add_argument("--pretrained", action="store_true", default=True)
    g.add_argument("--num-classes", default=1000, type=int)
    g.add_argument(
        "--input-size", nargs=3, default=None, type=int,
        help="Override model input size (C H W).",
    )
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
    g.add_argument(
        "--fsc-path", required=True, type=str,
        help="Path to Stem FSC file (e.g., ./FSC/resnet50_FSC_stem.pth).",
    )
    g.add_argument(
        "--fsc-diff-mode", default="orthogonal", type=str,
        choices=["subtract", "orthogonal"],
        help="FSC difference computation mode (default: orthogonal).",
    )
    g.add_argument(
        "--aug-classifier-hidden", nargs="+", default=[512, 256, 128], type=int,
        help="Hidden dims for aug classifier MLP. "
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
        help="Expert weight type (default: stem). "
             "'stem': conv1 from expert (frozen), bn1 TTA-trained. "
             "'stem-all-bn': backbone BN from expert (frozen), "
             "conv1+bn1 TTA-trained.",
    )

    # --- TTA Configuration ---
    g = parser.add_argument_group("TTA Configuration")
    g.add_argument(
        "--tta-mode", default="static", type=str,
        choices=["static", "continue"],
        help="Stem reset policy (default: static). "
             "'static': reset to pretrained each mini-batch. "
             "'continue': carry TTA-adapted stem across batches.",
    )
    g.add_argument(
        "--target-threshold", default=0.95, type=float,
        help="Stop TTA when avg softmax prob for hard label >= threshold "
             "(default: 0.95).",
    )
    g.add_argument(
        "--tta-lr", default=0.05, type=float,
        help="Learning rate for TTA (default: 0.05).",
    )
    g.add_argument(
        "--tta-optimizer", default="sgd", type=str,
        choices=["sgd", "adam", "adamw"],
        help="Optimizer for TTA (default: sgd).",
    )
    g.add_argument(
        "--tta-max-iter", default=50, type=int,
        help="Max gradient updates per mini-batch (default: 50).",
    )
    g.add_argument(
        "--bn-train-mode", default="affine-only", type=str,
        choices=["affine-only", "affine-and-running"],
        help="BN training mode for stem bn1 (default: affine-only). "
             "'affine-only': only weight/bias trained, BN eval mode. "
             "'affine-and-running': weight/bias trained + running stats "
             "updated, BN train mode.",
    )

    # --- Runtime ---
    g = parser.add_argument_group("Runtime")
    g.add_argument("--batch-size", default=64, type=int)
    g.add_argument("--workers", default=8, type=int)
    g.add_argument("--device", default="cuda", type=str)
    g.add_argument(
        "--amp", action="store_true",
        help="Use AMP (mixed precision) for evaluation inference.",
    )

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
    # Classification components  (frozen, independent copies)
    # ================================================================
    # classification_stem: pretrained 가중치를 항상 유지하는 독립 사본
    classification_stem = copy.deepcopy(StemFeatureExtractor(model))
    classification_stem.to(device)
    classification_stem.eval()
    for p in classification_stem.parameters():
        p.requires_grad = False

    # ================================================================
    # FSC centroids
    # ================================================================
    _logger.info("Loading FSC centroids from: %s", args.fsc_path)
    fsc_data = torch.load(args.fsc_path, map_location="cpu")
    fsc_centroids = fsc_data["centroids"].to(device)
    feature_dim = fsc_data["feature_dim"]
    _logger.info(
        "FSC centroids: %s, feature_dim=%d", fsc_centroids.shape, feature_dim,
    )

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
        "\n%s\n"
        "Starting TTA evaluation: %d corruptions, severity=%d\n"
        "  expert_type=%s, tta_mode=%s, bn_train_mode=%s\n"
        "  optimizer=%s, lr=%.4f, threshold=%.2f, max_iter=%d\n"
        "%s",
        "=" * 70, len(corruptions), args.severity,
        args.expert_type, args.tta_mode, args.bn_train_mode,
        args.tta_optimizer, args.tta_lr, args.target_threshold,
        args.tta_max_iter, "=" * 70,
    )

    for corr_idx, corruption in enumerate(corruptions):
        corr_dir = Path(args.imagenet_c_dir) / corruption / str(args.severity)
        if not corr_dir.exists():
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

        top1, top5, desc = _evaluate_corruption_tta(
            model, loader,
            classification_stem, aug_classifier, fsc_centroids,
            experts, pretrained_state, transform_names,
            args, fsc_diff_mode, device,
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

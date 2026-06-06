#!/usr/bin/env python3
"""ImageNet-C Validation with Meta-TTA (train_meta_cur.py approach).

Combines the ImageNet-C data loading from validate_imagenet_c_expert_tta_maximize.py
with the meta-learning TTA inner loop from train_meta_cur.py.

Meta-TTA adapts only ``conv1.weight`` using a combined contrastive inner loss:
  (1) ``-alpha * entropy``: erase augmentation signal from aug classifier output.
  (2) ``+beta * cos_sim(adapted, original)``: repel from corrupted features.
  (3) ``-gamma * cos_sim(adapted, centroid)``: attract toward clean FSC centroid.

The stem feature path is conv1-only (no bn1, no act1), consistent with the
meta-learning training pipeline in train_meta_cur.py.

Pipeline:
1. Load pretrained model, trained Aug Classifier, and FSC centroids.
2. For each corruption (severity=5 by default):
   For each mini-batch:
   a. Evaluate pretrained baseline (before TTA).
   b. Run meta_inner_loop_simple to adapt conv1.
   c. Evaluate classification accuracy with adapted conv1.
   d. Reset conv1 (static mode) or carry over (continue mode).
3. Save results to .txt file.

TTA mode controls conv1 reset policy:
    static   : reset to pretrained each mini-batch (independent adaptation).
    continue : carry adapted conv1 across batches (cumulative adaptation).

Usage examples:
    python validate_imagenet_c_meta_tta.py \\
        --imagenet-c-dir /path/to/ImageNet-C \\
        --imagenet-val-dir /path/to/imagenet/val \\
        --aug-classifier-ckpt ./output/META_CUR/best.pth.tar \\
        --fsc-path ./FSC/resnet50_FSC_stem.pth \\
        --tta-mode static \\
        --inner-lr 0.01 \\
        --inner-steps 1 \\
        --results-dir ./results

    python validate_imagenet_c_meta_tta.py \\
        --imagenet-c-dir /path/to/ImageNet-C \\
        --imagenet-val-dir /path/to/imagenet/val \\
        --aug-classifier-ckpt ./output/META_CUR/best.pth.tar \\
        --fsc-path ./FSC/resnet50_FSC_stem.pth \\
        --tta-mode continue \\
        --inner-lr 0.01 \\
        --inner-steps 3 \\
        --inner-alpha 1.0 --inner-beta 0.1 --inner-gamma 0.5 \\
        --results-dir ./results
"""

import argparse
import copy
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from timm.data import create_transform, resolve_data_config
from timm.data import get_augmix_sl_num_transforms
from timm.models import create_model
from timm.utils import AverageMeter, setup_default_logging

from train_meta_cur import (
    AugClassifier,
    compute_fsc_diff,
    meta_inner_loop_simple,
    meta_outer_eval,
    reset_conv1,
)

_logger = logging.getLogger("validate_meta_tta")

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
# Dataset (from validate_imagenet_c_expert_tta_maximize.py)
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
# Per-Corruption Evaluation with Meta-TTA
# =============================================================================

def _evaluate_corruption_meta_tta(
    model: nn.Module,
    loader: DataLoader,
    aug_classifier: nn.Module,
    fsc_centroids: torch.Tensor,
    pretrained_state: Dict[str, torch.Tensor],
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[float, float, float, float, str]:
    """Per-batch meta-TTA evaluation for one corruption.

    For each mini-batch:
    1. Reset model if static / first batch.
    2. Evaluate pretrained baseline (before TTA).
    3. Run meta_inner_loop_simple to adapt conv1.
    4. Evaluate classification accuracy with adapted conv1.

    Returns:
        (pretrained_top1, pretrained_top5, adapted_top1, adapted_top5,
         summary_description)
    """
    pre_top1_m = AverageMeter()
    pre_top5_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()

    total_tta_updates = 0
    total_entropy_before = 0.0
    total_entropy_after = 0.0
    total_param_delta = 0.0
    total_fsc_norm_before = 0.0
    total_fsc_norm_after = 0.0
    total_batches = 0

    for batch_idx, (images, target) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # ---- Step 1: Model preparation ----
        if args.tta_mode == "static" or batch_idx == 0:
            model.load_state_dict(pretrained_state)
        model.eval()

        # ---- Step 2: Pretrained baseline (before TTA for this batch) ----
        baseline = meta_outer_eval(images, target, model)
        pre_top1_m.update(baseline['top1'], images.size(0))
        pre_top5_m.update(baseline['top5'], images.size(0))

        # ---- Step 3: Meta-TTA inner loop (adapt conv1) ----
        inner_diag = meta_inner_loop_simple(
            images, model, aug_classifier, fsc_centroids,
            fsc_diff_mode=args.fsc_diff_mode,
            inner_lr=args.inner_lr,
            inner_steps=args.inner_steps,
            inner_alpha=args.inner_alpha,
            inner_beta=args.inner_beta,
            inner_gamma=args.inner_gamma,
            inner_momentum=args.inner_momentum,
        )

        total_tta_updates += inner_diag['num_updates']
        total_entropy_before += inner_diag['entropy_before']
        total_entropy_after += inner_diag['entropy_after']
        total_param_delta += inner_diag['param_delta_norm']
        total_fsc_norm_before += inner_diag['fsc_diff_norm_before']
        total_fsc_norm_after += inner_diag['fsc_diff_norm_after']
        total_batches += 1

        # ---- Step 4: Evaluate with adapted conv1 ----
        after = meta_outer_eval(images, target, model)
        top1_m.update(after['top1'], images.size(0))
        top5_m.update(after['top5'], images.size(0))

        _logger.info(
            "  Batch %d: pre_top1=%.2f%% → adapted_top1=%.2f%% | "
            "entropy=%.4f→%.4f, param_delta=%.6f, "
            "fsc_norm=%.4f→%.4f",
            batch_idx,
            baseline['top1'], after['top1'],
            inner_diag['entropy_before'], inner_diag['entropy_after'],
            inner_diag['param_delta_norm'],
            inner_diag['fsc_diff_norm_before'],
            inner_diag['fsc_diff_norm_after'],
        )

        # ---- Step 5: Reset conv1 if static mode ----
        if args.tta_mode == "static":
            reset_conv1(model, inner_diag['original_weight'])
        # For 'continue' mode, keep the adapted conv1 for next batch

    # ---- Summary ----
    avg_updates = total_tta_updates / total_batches if total_batches > 0 else 0
    avg_entropy_bef = (
        total_entropy_before / total_batches if total_batches > 0 else 0
    )
    avg_entropy_aft = (
        total_entropy_after / total_batches if total_batches > 0 else 0
    )
    avg_param_delta = (
        total_param_delta / total_batches if total_batches > 0 else 0
    )
    avg_fsc_bef = (
        total_fsc_norm_before / total_batches if total_batches > 0 else 0
    )
    avg_fsc_aft = (
        total_fsc_norm_after / total_batches if total_batches > 0 else 0
    )
    description = (
        f"mode={args.tta_mode}, updates={avg_updates:.1f}, "
        f"entropy={avg_entropy_bef:.4f}→{avg_entropy_aft:.4f}, "
        f"param_delta={avg_param_delta:.6f}, "
        f"fsc_norm={avg_fsc_bef:.4f}→{avg_fsc_aft:.4f}"
    )
    _logger.info(description)

    return (
        pre_top1_m.avg, pre_top5_m.avg,
        top1_m.avg, top5_m.avg,
        description,
    )


# =============================================================================
# Results I/O
# =============================================================================

def _generate_results_filename(args: argparse.Namespace) -> str:
    """Generate a unique results filename from run options."""
    parts = ["meta_tta"]
    parts.append(args.tta_mode)
    parts.append(f"lr{args.inner_lr}")
    parts.append(f"steps{args.inner_steps}")
    parts.append(f"a{args.inner_alpha}_b{args.inner_beta}_g{args.inner_gamma}")
    parts.append(f"mom{args.inner_momentum}")
    parts.append(f"fsc-{args.fsc_diff_mode}")
    parts.append(f"sev{args.severity}")
    return "_".join(parts) + ".txt"


def _save_results(
    results: List[Dict],
    args: argparse.Namespace,
    tta_summaries: Dict[str, str],
) -> None:
    """Save results to a tab-separated .txt file."""
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    filename = _generate_results_filename(args)
    filepath = results_dir / filename

    with filepath.open("w") as f:
        f.write("Corruption\tPre_Top1\tPre_Top5\tAdapted_Top1\tAdapted_Top5\tDelta_Top1\n")

        pre_top1_sum = 0.0
        pre_top5_sum = 0.0
        ada_top1_sum = 0.0
        ada_top5_sum = 0.0

        for r in results:
            delta = r["adapted_top1"] - r["pre_top1"]
            f.write(
                f"{r['corruption']}\t"
                f"{r['pre_top1']:.3f}\t{r['pre_top5']:.3f}\t"
                f"{r['adapted_top1']:.3f}\t{r['adapted_top5']:.3f}\t"
                f"{delta:+.3f}\n"
            )
            pre_top1_sum += r["pre_top1"]
            pre_top5_sum += r["pre_top5"]
            ada_top1_sum += r["adapted_top1"]
            ada_top5_sum += r["adapted_top5"]

        if results:
            n = len(results)
            mean_pre1 = pre_top1_sum / n
            mean_pre5 = pre_top5_sum / n
            mean_ada1 = ada_top1_sum / n
            mean_ada5 = ada_top5_sum / n
            mean_delta = mean_ada1 - mean_pre1
            f.write(
                f"mean\t{mean_pre1:.3f}\t{mean_pre5:.3f}\t"
                f"{mean_ada1:.3f}\t{mean_ada5:.3f}\t{mean_delta:+.3f}\n"
            )

        f.write("\n# --- Meta-TTA Summary per Corruption ---\n")
        for corr, desc in tta_summaries.items():
            f.write(f"# {corr}: {desc}\n")

        f.write("\n# --- Run Configuration ---\n")
        f.write(f"# model: {args.model}\n")
        f.write(f"# tta_mode: {args.tta_mode}\n")
        f.write(f"# inner_lr: {args.inner_lr}\n")
        f.write(f"# inner_steps: {args.inner_steps}\n")
        f.write(f"# inner_alpha: {args.inner_alpha}\n")
        f.write(f"# inner_beta: {args.inner_beta}\n")
        f.write(f"# inner_gamma: {args.inner_gamma}\n")
        f.write(f"# inner_momentum: {args.inner_momentum}\n")
        f.write(f"# fsc_diff_mode: {args.fsc_diff_mode}\n")
        f.write(f"# severity: {args.severity}\n")
        f.write(f"# batch_size: {args.batch_size}\n")

    _logger.info("Results saved to: %s", filepath.resolve())


# =============================================================================
# Argument Parser
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ImageNet-C Meta-TTA Validation (train_meta_cur.py approach)",
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
        default="./output/META_CUR/best.pth.tar",
        type=str,
        help="Path to trained aug classifier checkpoint (from train_meta_cur.py).",
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

    # --- Meta-TTA Configuration ---
    g = parser.add_argument_group("Meta-TTA Configuration")
    g.add_argument(
        "--tta-mode", default="static", type=str,
        choices=["static", "continue"],
        help="Conv1 reset policy (default: static). "
             "'static': reset to pretrained each mini-batch. "
             "'continue': carry adapted conv1 across batches.",
    )
    g.add_argument(
        "--inner-lr", default=0.01, type=float,
        help="Inner loop learning rate for conv1 adaptation (default: 0.01).",
    )
    g.add_argument(
        "--inner-steps", default=1, type=int,
        help="Inner loop gradient steps K (default: 1).",
    )
    g.add_argument(
        "--inner-alpha", default=1.0, type=float,
        help="Weight for entropy maximization (erase aug signal) (default: 1.0).",
    )
    g.add_argument(
        "--inner-beta", default=0.1, type=float,
        help="Weight for repel-corrupted (push from original features) (default: 0.1).",
    )
    g.add_argument(
        "--inner-gamma", default=0.5, type=float,
        help="Weight for attract-centroid (pull toward clean centroid) (default: 0.5).",
    )
    g.add_argument(
        "--inner-momentum", default=0.0, type=float,
        help="SGD momentum for inner loop conv1 adaptation (default: 0.0).",
    )

    # --- Runtime ---
    g = parser.add_argument_group("Runtime")
    g.add_argument("--batch-size", default=64, type=int)
    g.add_argument("--workers", default=8, type=int)
    g.add_argument("--device", default="cuda", type=str)

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

    # ---- Aug classifier transform count ----
    num_transforms = get_augmix_sl_num_transforms(version=2)

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
    # Aug Classifier (frozen — used as entropy signal source for TTA)
    # ================================================================
    aug_ckpt_path = Path(args.aug_classifier_ckpt)
    if not aug_ckpt_path.exists():
        raise FileNotFoundError(
            f"Aug classifier checkpoint not found: {aug_ckpt_path.resolve()}"
        )
    _logger.info("Loading aug classifier from: %s", aug_ckpt_path.resolve())
    aug_ckpt = torch.load(aug_ckpt_path, map_location="cpu")

    aug_train_args = aug_ckpt.get("args", None)
    hidden_dims = list(args.aug_classifier_hidden)
    fsc_diff_mode = args.fsc_diff_mode
    use_softmax = False

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
        if hasattr(aug_train_args, "aug_classifier_softmax"):
            use_softmax = aug_train_args.aug_classifier_softmax
            _logger.info(
                "Overriding use_softmax from checkpoint: %s", use_softmax,
            )

    # Apply overridden fsc_diff_mode
    args.fsc_diff_mode = fsc_diff_mode

    aug_classifier = AugClassifier(
        feature_dim=feature_dim,
        num_transforms=num_transforms,
        hidden_dims=hidden_dims,
        dropout=0.1,
        use_sigmoid=False,
        use_softmax=use_softmax,
    )
    aug_classifier.load_state_dict(aug_ckpt["aug_classifier"])
    aug_classifier.to(device)
    aug_classifier.eval()
    for p in aug_classifier.parameters():
        p.requires_grad = False
    _logger.info(
        "Aug classifier loaded (hidden=%s, fsc_diff_mode=%s, "
        "use_softmax=%s, num_transforms=%d)",
        hidden_dims, fsc_diff_mode, use_softmax, num_transforms,
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
    tta_summaries: Dict[str, str] = {}

    _logger.info(
        "\n%s\n"
        "Starting Meta-TTA: %d corruptions, severity=%d\n"
        "  tta_mode=%s, inner_lr=%.4f, inner_steps=%d, inner_momentum=%.2f\n"
        "  inner_alpha=%.2f, inner_beta=%.2f, inner_gamma=%.2f\n"
        "  fsc_diff_mode=%s\n"
        "%s",
        "=" * 70, len(corruptions), args.severity,
        args.tta_mode, args.inner_lr, args.inner_steps, args.inner_momentum,
        args.inner_alpha, args.inner_beta, args.inner_gamma,
        args.fsc_diff_mode, "=" * 70,
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

        pre_top1, pre_top5, ada_top1, ada_top5, desc = (
            _evaluate_corruption_meta_tta(
                model, loader,
                aug_classifier, fsc_centroids, pretrained_state,
                args, device,
            )
        )

        delta = ada_top1 - pre_top1
        _logger.info(
            ">>> %s: Pretrained=%.3f%% → Adapted=%.3f%% (delta=%+.3f%%) "
            "(%d images)",
            corruption, pre_top1, ada_top1, delta, len(dataset),
        )
        results.append({
            "corruption": corruption,
            "pre_top1": pre_top1,
            "pre_top5": pre_top5,
            "adapted_top1": ada_top1,
            "adapted_top5": ada_top5,
        })
        tta_summaries[corruption] = desc

    # ================================================================
    # Save results
    # ================================================================
    _save_results(results, args, tta_summaries)

    # ================================================================
    # Print summary
    # ================================================================
    if results:
        n = len(results)
        mean_pre1 = sum(r["pre_top1"] for r in results) / n
        mean_pre5 = sum(r["pre_top5"] for r in results) / n
        mean_ada1 = sum(r["adapted_top1"] for r in results) / n
        mean_ada5 = sum(r["adapted_top5"] for r in results) / n
        mean_delta = mean_ada1 - mean_pre1
        _logger.info(
            "\n%s\n"
            "Final Mean:\n"
            "  Pretrained : Top1=%.3f%%, Top5=%.3f%%\n"
            "  Adapted    : Top1=%.3f%%, Top5=%.3f%%\n"
            "  Delta Top1 : %+.3f%%\n"
            "%s",
            "=" * 70,
            mean_pre1, mean_pre5,
            mean_ada1, mean_ada5,
            mean_delta,
            "=" * 70,
        )


if __name__ == "__main__":
    main()

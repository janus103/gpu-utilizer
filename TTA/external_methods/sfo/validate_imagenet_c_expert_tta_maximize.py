#!/usr/bin/env python3
"""ImageNet-C Entropy Maximization TTA Validation Script.

Adversarial TTA: trains the stem to MAXIMIZE the entropy of the aug
classifier output, effectively removing corruption-specific signals from
stem features so that the pretrained backbone sees clean-like features.

No expert weights are used — the entire backbone stays pretrained.

Pipeline:
1. Load pretrained model, frozen Aug Classifier, and FSC centroids.
2. For each corruption (severity=5 by default):
   For each mini-batch:
   a. Entropy-max TTA: train stem so aug classifier cannot distinguish
      the augmentation (output → uniform distribution).
   b. Stop when max avg softmax prob <= max_prob_threshold, or max iter.
   c. Evaluate classification accuracy on the same mini-batch.
3. Save results to .txt file.

TTA target controls which stem parameters are trained:
    bn-only : only bn1.weight and bn1.bias  (conv1 frozen/pretrained).
    conv-bn : conv1.* + bn1.weight and bn1.bias.

Usage examples:
    python validate_imagenet_c_expert_tta_maximize.py \\
        --imagenet-c-dir /path/to/ImageNet-C \\
        --imagenet-val-dir /path/to/imagenet/val \\
        --aug-classifier-ckpt ./output/STEM_KLDIV_ORTHOGONAL/best.pth.tar \\
        --fsc-path ./FSC/resnet50_FSC_stem.pth \\
        --tta-target bn-only \\
        --tta-mode static \\
        --max-prob-threshold 0.20 \\
        --tta-lr 0.05 \\
        --results-dir ./results

    python validate_imagenet_c_expert_tta_maximize.py \\
        --imagenet-c-dir /path/to/ImageNet-C \\
        --imagenet-val-dir /path/to/imagenet/val \\
        --aug-classifier-ckpt ./output/STEM_KLDIV_ORTHOGONAL/best.pth.tar \\
        --fsc-path ./FSC/resnet50_FSC_stem.pth \\
        --tta-target conv-bn \\
        --tta-mode continue \\
        --bn-train-mode affine-and-running \\
        --max-prob-threshold 0.20 \\
        --tta-lr 0.01 \\
        --results-dir ./results
"""

import argparse
import copy
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from timm.data import create_transform, resolve_data_config
from timm.data import get_augmix_sl_num_transforms
from timm.models import create_model
from timm.utils import AverageMeter, accuracy, setup_default_logging

from train_augmix_stem import (
    AugClassifier,
    compute_fsc_diff,
)

_logger = logging.getLogger("validate_tta_entropy")

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
# TTA Training Helpers
# =============================================================================

def _setup_tta_trainable_params(
    model: nn.Module,
    tta_target: str,
    bn_train_mode: str,
) -> List[nn.Parameter]:
    """Configure which stem parameters are trainable for TTA.

    Args:
        tta_target:
            ``bn-only`` – only bn1.weight and bn1.bias (conv1 frozen).
            ``conv-bn`` – conv1.* + bn1.weight and bn1.bias.
        bn_train_mode:
            ``affine-only``        – bn1 eval mode, only affine params trained.
            ``affine-and-running`` – bn1 train mode, running stats also updated.

    Returns:
        List of trainable parameters for the optimizer.
    """
    for p in model.parameters():
        p.requires_grad = False

    trainable_params: List[nn.Parameter] = []

    if tta_target == "bn-only":
        for name, p in model.named_parameters():
            if name in ("bn1.weight", "bn1.bias"):
                p.requires_grad = True
                trainable_params.append(p)

    elif tta_target == "conv-bn":
        for name, p in model.named_parameters():
            if name.startswith("conv1"):
                p.requires_grad = True
                trainable_params.append(p)
            elif name in ("bn1.weight", "bn1.bias"):
                p.requires_grad = True
                trainable_params.append(p)

    if bn_train_mode == "affine-and-running":
        model.bn1.train()
    else:
        model.bn1.eval()

    return trainable_params


def _tta_stem_forward(
    model: nn.Module,
    images: torch.Tensor,
) -> torch.Tensor:
    """Forward through model's stem to get features for FSC diff.

    Matches ``StemFeatureExtractor`` layout:
    conv1 → bn1 → act1 → AdaptiveAvgPool2d(4,4) → flatten → [B, 1024].
    Uses model's own layers so gradients flow to trainable parameters.
    """
    x = images.contiguous()
    x = model.conv1(x)
    x = model.bn1(x)
    x = model.act1(x)
    x = F.adaptive_avg_pool2d(x, (4, 4))
    return x.flatten(1)


def _tta_adapt_batch_entropy(
    model: nn.Module,
    images: torch.Tensor,
    aug_classifier: nn.Module,
    fsc_centroids: torch.Tensor,
    fsc_diff_mode: str,
    trainable_params: List[nn.Parameter],
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[float, float, int, float]:
    """Entropy-maximization TTA for one mini-batch.

    Maximises the entropy of the frozen aug classifier's softmax output,
    pushing it toward a uniform distribution.  This forces the stem to
    *remove* corruption-specific signals from features.

    Stops when ``avg_probs.max() <= max_prob_threshold`` or after
    ``tta_max_iter`` gradient updates.

    Returns:
        final_max_prob – max avg softmax probability after adaptation.
        final_entropy  – entropy of avg softmax distribution after adaptation.
        num_updates    – number of gradient updates performed.
        final_loss     – last loss value (0.0 if no update was needed).
    """
    # ---- Optimizer ----
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

    # FSC mean (고정, fsc_mode=mean)
    fsc_mean = fsc_centroids.mean(dim=0, keepdim=True).expand(
        images.size(0), -1,
    )

    final_max_prob = 1.0
    final_entropy = 0.0
    final_loss = 0.0
    num_updates = 0

    for _iter in range(args.tta_max_iter):
        optimizer.zero_grad()

        # ---- Forward: stem → FSC diff → aug_classifier ----
        stem_features = _tta_stem_forward(model, images)
        fsc_diff = compute_fsc_diff(stem_features, fsc_mean, mode=fsc_diff_mode)
        aug_logits = aug_classifier(fsc_diff)           # frozen, graph preserved

        # ---- Check stopping criterion (before update) ----
        with torch.no_grad():
            avg_probs = F.softmax(aug_logits.detach(), dim=1).mean(dim=0)
            final_max_prob = avg_probs.max().item()
            final_entropy = -(
                avg_probs * torch.log(avg_probs + 1e-10)
            ).sum().item()

        if final_max_prob <= args.max_prob_threshold:
            break

        # ---- Entropy maximization loss ----
        # H = -Σ p_i log(p_i);  loss = -H  → minimise to maximise entropy
        probs = F.softmax(aug_logits, dim=1)
        log_probs = F.log_softmax(aug_logits, dim=1)
        entropy_per_sample = -(probs * log_probs).sum(dim=1)       # [B]
        loss = -entropy_per_sample.mean()                           # scalar

        loss.backward()
        optimizer.step()

        num_updates += 1
        final_loss = loss.item()

    return final_max_prob, final_entropy, num_updates, final_loss


# =============================================================================
# Per-Corruption Evaluation
# =============================================================================

def _evaluate_corruption_tta(
    model: nn.Module,
    loader: DataLoader,
    aug_classifier: nn.Module,
    fsc_centroids: torch.Tensor,
    pretrained_state: Dict[str, torch.Tensor],
    args: argparse.Namespace,
    fsc_diff_mode: str,
    device: torch.device,
) -> Tuple[float, float, str]:
    """Per-batch entropy-max TTA evaluation for one corruption.

    For each mini-batch:
    1. Reset model if static / first batch.
    2. Entropy-max TTA on stem.
    3. Evaluate on the same batch.

    Returns:
        (mean_top1, mean_top5, summary_description)
    """
    top1_m = AverageMeter()
    top5_m = AverageMeter()

    total_tta_updates = 0
    total_entropy = 0.0
    total_max_prob = 0.0
    total_batches = 0

    for batch_idx, (images, target) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # ---- Step 1: Model 준비 ----
        if args.tta_mode == "static" or batch_idx == 0:
            model.load_state_dict(pretrained_state)

        model.eval()
        trainable_params = _setup_tta_trainable_params(
            model, args.tta_target, args.bn_train_mode,
        )

        # ---- Step 2: Entropy-max TTA ----
        if not trainable_params:
            _logger.warning(
                "  Batch %d: no trainable params, skipping TTA", batch_idx,
            )
            max_prob, entropy, num_updates, loss_val = 1.0, 0.0, 0, 0.0
        else:
            max_prob, entropy, num_updates, loss_val = _tta_adapt_batch_entropy(
                model, images,
                aug_classifier, fsc_centroids, fsc_diff_mode,
                trainable_params, args, device,
            )

        total_tta_updates += num_updates
        total_entropy += entropy
        total_max_prob += max_prob
        total_batches += 1

        _logger.info(
            "  Batch %d: updates=%d, max_prob=%.4f, entropy=%.4f, loss=%.4f",
            batch_idx, num_updates, max_prob, entropy, loss_val,
        )

        # ---- Step 3: 같은 배치로 평가 ----
        model.eval()
        with torch.no_grad():
            output = model(images)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        top1_m.update(acc1.item(), images.size(0))
        top5_m.update(acc5.item(), images.size(0))

    # ---- 요약 ----
    avg_updates = total_tta_updates / total_batches if total_batches > 0 else 0
    avg_entropy = total_entropy / total_batches if total_batches > 0 else 0
    avg_max_prob = total_max_prob / total_batches if total_batches > 0 else 0
    description = (
        f"mode={args.tta_mode}, avg_updates={avg_updates:.1f}, "
        f"avg_max_prob={avg_max_prob:.4f}, avg_entropy={avg_entropy:.4f}"
    )
    _logger.info(description)

    return top1_m.avg, top5_m.avg, description


# =============================================================================
# Results I/O
# =============================================================================

def _generate_results_filename(args: argparse.Namespace) -> str:
    """Generate a unique results filename from run options."""
    parts = ["tta_entropy"]
    parts.append(args.tta_target)
    parts.append(args.tta_mode)
    parts.append(f"bn-{args.bn_train_mode}")
    parts.append(f"lr{args.tta_lr}")
    parts.append(f"maxprob{args.max_prob_threshold}")
    parts.append(f"opt-{args.tta_optimizer}")
    parts.append(f"maxiter{args.tta_max_iter}")
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
        f.write("Corruption\tTop1\tTop5\n")

        for r in results:
            f.write(f"{r['corruption']}\t{r['top1']:.3f}\t{r['top5']:.3f}\n")

        if results:
            mean_top1 = sum(r["top1"] for r in results) / len(results)
            mean_top5 = sum(r["top5"] for r in results) / len(results)
            f.write(f"mean\t{mean_top1:.3f}\t{mean_top5:.3f}\n")

        f.write("\n# --- TTA Entropy Summary per Corruption ---\n")
        for corr, desc in tta_summaries.items():
            f.write(f"# {corr}: {desc}\n")

        f.write("\n# --- Run Configuration ---\n")
        f.write(f"# model: {args.model}\n")
        f.write(f"# tta_target: {args.tta_target}\n")
        f.write(f"# tta_mode: {args.tta_mode}\n")
        f.write(f"# bn_train_mode: {args.bn_train_mode}\n")
        f.write(f"# tta_optimizer: {args.tta_optimizer}\n")
        f.write(f"# tta_lr: {args.tta_lr}\n")
        f.write(f"# max_prob_threshold: {args.max_prob_threshold}\n")
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
        description="ImageNet-C Entropy-Max TTA Validation (no expert weights)",
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

    # --- Aug Classifier (frozen, for entropy signal only) ---
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

    # --- TTA Configuration ---
    g = parser.add_argument_group("TTA Configuration")
    g.add_argument(
        "--tta-target", default="bn-only", type=str,
        choices=["bn-only", "conv-bn"],
        help="Which stem parameters to train (default: bn-only). "
             "'bn-only': bn1 affine only (conv1 frozen). "
             "'conv-bn': conv1 + bn1 affine.",
    )
    g.add_argument(
        "--tta-mode", default="static", type=str,
        choices=["static", "continue"],
        help="Stem reset policy (default: static). "
             "'static': reset to pretrained each mini-batch. "
             "'continue': carry adapted stem across batches.",
    )
    g.add_argument(
        "--max-prob-threshold", default=0.20, type=float,
        help="Stop TTA when max avg softmax prob <= this value "
             "(default: 0.20; uniform for K=7 is ~0.143).",
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
             "'affine-only': weight/bias only, BN eval mode. "
             "'affine-and-running': + running stats updated, BN train mode.",
    )

    # --- Runtime ---
    g = parser.add_argument_group("Runtime")
    g.add_argument("--batch-size", default=64, type=int)
    g.add_argument("--workers", default=8, type=int)
    g.add_argument("--device", default="cuda", type=str)
    g.add_argument(
        "--amp", action="store_true",
        help="Use AMP for evaluation inference.",
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
    # Aug Classifier (frozen — used only as entropy signal source)
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
        "Aug classifier loaded (hidden=%s, fsc_diff_mode=%s, num_transforms=%d)",
        hidden_dims, fsc_diff_mode, num_transforms,
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
        "Starting Entropy-Max TTA: %d corruptions, severity=%d\n"
        "  tta_target=%s, tta_mode=%s, bn_train_mode=%s\n"
        "  optimizer=%s, lr=%.4f, max_prob_thr=%.3f, max_iter=%d\n"
        "%s",
        "=" * 70, len(corruptions), args.severity,
        args.tta_target, args.tta_mode, args.bn_train_mode,
        args.tta_optimizer, args.tta_lr, args.max_prob_threshold,
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
            aug_classifier, fsc_centroids, pretrained_state,
            args, fsc_diff_mode, device,
        )

        _logger.info(
            ">>> %s: Top1=%.3f%%, Top5=%.3f%% (%d images)",
            corruption, top1, top5, len(dataset),
        )
        results.append(
            {"corruption": corruption, "top1": top1, "top5": top5},
        )
        tta_summaries[corruption] = desc

    # ================================================================
    # Save results
    # ================================================================
    _save_results(results, args, tta_summaries)

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

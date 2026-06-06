#!/usr/bin/env python3
"""ImageNet-C evaluation helper.

This script evaluates a classification checkpoint (or timm pretrained weights)
on the ImageNet-C dataset and writes per-corruption / per-severity accuracy to
a CSV file. It is single-process and runs on one GPU/CPU.
"""

import argparse
import csv
import logging
import os
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from contextlib import nullcontext

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from timm.data import create_transform, resolve_data_config
from timm.models import create_model, load_checkpoint
from timm.utils import AverageMeter, ParseKwargs, accuracy, setup_default_logging

# Reuse the AugAware components from train2 so that checkpoints trained with the
# custom stem AE + LALP wrapper can be evaluated correctly.
from train2 import AugAwareWrapper, AugmentationManager, StemAutoEncoder

_logger = logging.getLogger("validate_imagenet_c")

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
    from torchvision import datasets  # imported lazily to keep deps minimal

    val_dataset = datasets.ImageFolder(val_dir)
    filename_label_map = {Path(p).name: target for p, target in val_dataset.samples}
    class_to_idx = val_dataset.class_to_idx
    _logger.info("Loaded label map for %d validation images", len(filename_label_map))
    return filename_label_map, class_to_idx


def _maybe_wrap_with_augaware(
    model: nn.Module,
    args: argparse.Namespace,
    device: torch.device,
) -> nn.Module:
    """Optionally wrap model with AugAwareWrapper to match train2 checkpoints."""
    if not args.train2_augaware:
        return model

    aug_manager = AugmentationManager(sensitivity=args.aug_sensitivity)
    n_aug = len(aug_manager)
    latent_size = args.latent_size or 128

    stem_ae = getattr(model, "stem_ae", None)
    if stem_ae is None:
        if args.latent_size is None and not args.ae_checkpoint:
            raise ValueError(
                "train2-augaware mode requires --latent-size or --ae-checkpoint"
            )
        stem_ae = StemAutoEncoder(
            input_dim=2 * model.conv1.out_channels * model.conv1.in_channels,
            latent_dim=latent_size,
        )
        if args.ae_checkpoint:
            ckpt = torch.load(args.ae_checkpoint, map_location="cpu")
            state_dict = ckpt.get("state_dict", ckpt)
            stem_ae.load_state_dict(state_dict)
        model.stem_ae = stem_ae

    for param in stem_ae.parameters():
        param.requires_grad = False
    stem_ae.eval()

    lalp_init = torch.full(
        (n_aug, latent_size),
        torch.finfo(torch.float32).eps,
        device=device,
    )
    model.lalp = nn.Parameter(lalp_init)
    model.register_parameter("lalp", model.lalp)

    model = AugAwareWrapper(
        model=model,
        stem_ae=stem_ae,
        lalp=model.lalp,
        n_aug=n_aug,
        use_aug_main_loss=args.use_aug_main_loss,
    )

    if args.freeze_non_selfsup:
        for name, param in model.named_parameters():
            if name.endswith("lalp") or "ssl_header" in name:
                continue
            param.requires_grad = False

    return model


def _create_model(args: argparse.Namespace, device: torch.device) -> nn.Module:
    model_kwargs = dict(args.model_kwargs)
    factory_kwargs = {}

    if args.ae_checkpoint:
        model_kwargs.setdefault("stem_ae_checkpoint", args.ae_checkpoint)
    if args.latent_size is not None:
        model_kwargs.setdefault("stem_ae_latent_size", args.latent_size)
    if args.ae_checkpoint or args.latent_size is not None or args.model.endswith("_ae"):
        model_kwargs.setdefault("pretrained_strict", False)

    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.num_classes,
        in_chans=args.in_chans,
        global_pool=args.gp,
        **factory_kwargs,
        **model_kwargs,
    )

    stem_ae_in_model = getattr(model, "stem_ae", None)
    if stem_ae_in_model is not None:
        for param in stem_ae_in_model.parameters():
            param.requires_grad = False
        stem_ae_in_model.eval()

    model = _maybe_wrap_with_augaware(model, args, device)
    model.to(device)

    if args.checkpoint:
        load_checkpoint(model, args.checkpoint, use_ema=args.use_ema)

    return model


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
) -> Tuple[float, float]:
    top1_m = AverageMeter()
    top5_m = AverageMeter()

    model.eval()
    if amp_enabled:
        autocast = torch.autocast
        autocast_kwargs = dict(device_type=device.type)
    else:
        autocast = nullcontext
        autocast_kwargs = {}

    with torch.no_grad():
        for images, target in loader:
            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            with autocast(**autocast_kwargs):
                output = model(images)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            batch_size = images.size(0)
            top1_m.update(acc1.item(), batch_size)
            top5_m.update(acc5.item(), batch_size)

    return top1_m.avg, top5_m.avg


def _severity_root(corruption_dir: Path, severities: Sequence[int]) -> Iterable[Tuple[int, Path]]:
    has_severity_dirs = any((corruption_dir / str(s)).exists() for s in severities)
    if has_severity_dirs:
        for severity in severities:
            root = corruption_dir / str(severity)
            if root.exists():
                yield severity, root
    else:
        if corruption_dir.exists():
            yield 0, corruption_dir


def evaluate_imagenet_c(args: argparse.Namespace) -> List[Dict[str, object]]:
    device = torch.device(args.device)

    model = _create_model(args, device)

    data_config = resolve_data_config(vars(args), model=model)
    transform = create_transform(**data_config, is_training=False)

    filename_label_map, class_to_idx = _build_label_map(Path(args.imagenet_val_dir))

    corruptions = args.corruptions or DEFAULT_CORRUPTIONS
    severities = args.severities

    results: List[Dict[str, object]] = []
    for corruption in corruptions:
        corr_dir = Path(args.imagenet_c_dir) / corruption
        if not corr_dir.exists():
            _logger.warning("Skip missing corruption directory: %s", corr_dir)
            continue

        for severity, severity_root in _severity_root(corr_dir, severities):
            dataset = ImageNetCDataset(
                severity_root, filename_label_map, class_to_idx, transform
            )
            loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.workers,
                pin_memory=True,
            )
            top1, top5 = _evaluate(model, loader, device, args.amp)
            _logger.info(
                "%s (severity %d): top1=%.3f top5=%.3f over %d images",
                corruption,
                severity,
                top1,
                top5,
                len(dataset),
            )
            results.append(
                dict(
                    corruption=corruption,
                    severity=severity,
                    num_samples=len(dataset),
                    top1=top1,
                    top5=top5,
                )
            )

    return results


def _write_csv(results: List[Dict[str, object]], path: Path) -> None:
    if not results:
        _logger.warning("No results to write.")
        return

    fields = ["corruption", "severity", "num_samples", "top1", "top5"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

        mean_top1 = sum(r["top1"] for r in results) / len(results)
        mean_top5 = sum(r["top5"] for r in results) / len(results)
        writer.writerow(
            dict(
                corruption="mean",
                severity="all",
                num_samples=sum(r["num_samples"] for r in results),
                top1=mean_top1,
                top5=mean_top5,
            )
        )
    _logger.info("Saved results to %s", path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate on ImageNet-C")
    parser.add_argument("--imagenet-c-dir", required=True, type=str)
    parser.add_argument("--imagenet-val-dir", required=True, type=str)
    parser.add_argument("--corruptions", nargs="+", default=None, type=str)
    parser.add_argument("--severities", nargs="+", default=[1, 2, 3, 4, 5], type=int)
    parser.add_argument("--results-csv", default="imagenet-c-results.csv", type=str)

    parser.add_argument("--model", default="resnet50", type=str)
    parser.add_argument("--checkpoint", default="", type=str)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--use-ema", action="store_true")
    parser.add_argument("--num-classes", default=1000, type=int)
    parser.add_argument("--in-chans", default=None, type=int)
    parser.add_argument("--input-size", nargs=3, default=None, type=int)
    parser.add_argument("--mean", nargs="+", default=None, type=float)
    parser.add_argument("--std", nargs="+", default=None, type=float)
    parser.add_argument("--crop-pct", default=None, type=float)
    parser.add_argument("--interpolation", default="", type=str)
    parser.add_argument("--gp", default=None, type=str)
    parser.add_argument("--model-kwargs", nargs="*", default={}, action=ParseKwargs)

    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--workers", default=8, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--amp", action="store_true")

    parser.add_argument(
        "--train2-augaware",
        action="store_true",
        help="Wrap the model with the train2 AugAware stem AE (for custom checkpoints).",
    )
    parser.add_argument("--latent-size", default=None, type=int)
    parser.add_argument("--ae-checkpoint", default="", type=str)
    parser.add_argument("--aug-sensitivity", default=1.0, type=float)
    parser.add_argument("--use-aug-main-loss", action="store_true")
    parser.add_argument("--freeze-non-selfsup", action="store_true")

    return parser.parse_args()


def main() -> None:
    setup_default_logging()
    args = parse_args()

    if torch.cuda.is_available() and args.device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    results = evaluate_imagenet_c(args)
    _write_csv(results, Path(args.results_csv))


if __name__ == "__main__":
    main()

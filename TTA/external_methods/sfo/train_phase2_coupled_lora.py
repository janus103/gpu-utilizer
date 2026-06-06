#!/usr/bin/env python3
"""Phase-2 coupled training (Phase-1 aligned, aug-classifier only).

Key alignment with current Phase-1 setup:
- Uses Phase-1 DirectAugClassifier checkpoint as the only adaptation signal source.
- No FSC centroid dependency.
- Stem-only LoRA adapter:
    W_eff = W0 + sum_m c_m * (B_m @ A_m)
- Task-coupled training:
    Inner (unlabeled): clean/dist/znorm/trust on support image.
    Outer (labeled): CE on clean query image.
- Robust resume support:
    adapter/optimizer/epoch/global_step/carry/RNG state restore.
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder

from timm.data import (
    create_augmix_sl_transform,
    get_augmix_sl_num_transforms,
    get_augmix_sl_transform_names,
    resolve_data_config,
)
from timm.data.auto_augment import (
    augment_and_mix_transform,
    auto_augment_transform,
    rand_augment_transform,
)
from timm.models import create_model
from timm.scheduler import create_scheduler_v2
from timm.utils import AverageMeter, setup_default_logging

from phase2_episode_scheduler import (
    CarryController,
    CarryPolicy,
    EpisodeModeScheduler,
    EpisodeSchedulerConfig,
)
from phase2_losses import InnerLossConfig, compute_inner_loss
from phase2_stem_adapter import build_stem_adapter

_logger = logging.getLogger("train_phase2_coupled_lora")


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_checkpoint_state(model: nn.Module, checkpoint_path: str) -> None:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state = ckpt
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            state = ckpt["state_dict"]
        elif "model" in ckpt:
            state = ckpt["model"]
    cleaned = {}
    for k, v in state.items():
        nk = k[7:] if k.startswith("module.") else k
        cleaned[nk] = v
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    _logger.info(
        "Loaded checkpoint %s (missing=%d, unexpected=%d)",
        checkpoint_path, len(missing), len(unexpected),
    )


def _capture_rng_state() -> Dict[str, object]:
    state: Dict[str, object] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Optional[Dict[str, object]]) -> None:
    if not state:
        return
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state:
        torch.set_rng_state(state["torch"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _optimizer_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device=device, non_blocking=True)


def _resolve_resume_path(resume_arg: str, out_dir: Path) -> Optional[Path]:
    if not resume_arg:
        return None
    if resume_arg.strip().lower() == "auto":
        return out_dir / "last.pth.tar"
    return Path(resume_arg)


def _ensure_results_header(results_path: Path, reset: bool) -> None:
    header = (
        "epoch\tloss\ttop1\tinner_pclean\tinner_dist\tinner_znorm\tinner_entropy\t"
        "mode_static\tmode_continual\tval_clean_top1\tval_clean_top5\tbest\n"
    )
    if reset:
        with open(results_path, "w") as f:
            f.write(header)
        return
    if not results_path.exists() or results_path.stat().st_size == 0:
        with open(results_path, "w") as f:
            f.write(header)


def get_curriculum_depth(epoch: int, total_epochs: int, max_depth: int) -> int:
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
    min_sl_start: float,
    min_sl_end: float,
    max_sl_start: float,
    max_sl_end: float,
) -> Tuple[float, float]:
    if total_epochs <= 1:
        return min_sl_end, max_sl_end
    progress = epoch / float(total_epochs - 1)
    cur_min = min_sl_start + progress * (min_sl_end - min_sl_start)
    cur_max = max_sl_start + progress * (max_sl_end - max_sl_start)
    return cur_min, cur_max


def _build_aa_transform(config_str: str, img_size: int, mean: Tuple[float, ...]) -> Optional[object]:
    """Build AutoAugment/RandAugment transform from config string."""
    if not config_str or not config_str.strip():
        return None
    aa_params = dict(
        translate_const=int(img_size * 0.45),
        img_mean=tuple(min(255, round(255 * x)) for x in mean),
    )
    if config_str.strip().startswith("rand"):
        return rand_augment_transform(config_str, aa_params)
    if config_str.strip().startswith("augmix"):
        aa_params["translate_pct"] = 0.3
        return augment_and_mix_transform(config_str, aa_params)
    return auto_augment_transform(config_str, aa_params)


class Phase2PairDataset(Dataset):
    """Returns clean/query and corrupted/support tensors from one source image."""

    def __init__(
        self,
        dataset: Dataset,
        base_transform,
        final_transform,
        augmix_transform,
        aa_transform=None,
    ):
        self.dataset = dataset
        self.base_transform = base_transform
        self.final_transform = final_transform
        self.augmix_transform = augmix_transform
        self.aa_transform = aa_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        if self.base_transform is not None:
            img = self.base_transform(img)

        clean_img = img
        support_img = img
        if self.aa_transform is not None:
            support_img = self.aa_transform(support_img)
        support_img, _aug_labels = self.augmix_transform(support_img)

        clean_tensor = self.final_transform(clean_img)
        support_tensor = self.final_transform(support_img)
        return clean_tensor, support_tensor, int(label)


class FixedTransformValidationDataset(Dataset):
    """Validation dataset that applies one fixed AugMixSL transform."""

    def __init__(self, dataset: Dataset, fixed_aug_transform, base_transform, final_transform):
        self.dataset = dataset
        self.fixed_aug_transform = fixed_aug_transform
        self.base_transform = base_transform
        self.final_transform = final_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        if self.base_transform is not None:
            img = self.base_transform(img)

        out = self.fixed_aug_transform(img)
        if isinstance(out, tuple):
            img = out[0]
        else:
            img = out

        if self.final_transform is not None:
            img = self.final_transform(img)
        return img, int(label)


def _ensure_tsv_header(path: Path, header: List[str], reset: bool) -> None:
    line = "\t".join(header) + "\n"
    if reset:
        with open(path, "w") as f:
            f.write(line)
        return
    if not path.exists() or path.stat().st_size == 0:
        with open(path, "w") as f:
            f.write(line)


@torch.no_grad()
def accuracy_topk(logits: torch.Tensor, labels: torch.Tensor, topk: Tuple[int, ...] = (1, 5)) -> Dict[str, float]:
    maxk = min(max(topk), logits.size(1))
    _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(labels.view(1, -1).expand_as(pred))
    out = {}
    for k in topk:
        kk = min(k, logits.size(1))
        out[f"top{k}"] = correct[:kk].reshape(-1).float().sum(0).item() * 100.0 / labels.size(0)
    return out


def _load_phase1_aug_classifier(
    model_name: str,
    aug_ckpt_path: str,
    stem_channels: int,
    device: torch.device,
):
    ckpt = torch.load(aug_ckpt_path, map_location="cpu")
    train_args = ckpt.get("args", {})

    if isinstance(train_args, argparse.Namespace):
        train_args = vars(train_args)
    if train_args is None:
        train_args = {}

    hidden_dims = train_args.get("hidden_dims", [512, 256, 128])
    dropout = train_args.get("dropout", 0.1)
    dw_init_mode = train_args.get("dw_init_mode", "fan_in")

    num_transforms = get_augmix_sl_num_transforms(version=2)

    if "vit" in model_name.lower():
        from train_phase1_vit import DirectAugClassifier
        num_dw_stages = 2
    else:
        from train_phase1_resnet import DirectAugClassifier
        num_dw_stages = 5

    aug_classifier = DirectAugClassifier(
        stem_channels=stem_channels,
        output_channels=64,
        num_dw_stages=num_dw_stages,
        num_transforms=num_transforms,
        hidden_dims=hidden_dims,
        dropout=dropout,
        dw_init_mode=dw_init_mode,
    )
    aug_classifier.load_state_dict(ckpt["aug_classifier"], strict=True)
    aug_classifier.to(device).eval()
    for p in aug_classifier.parameters():
        p.requires_grad = False

    _logger.info(
        "Loaded phase1 aug classifier from %s (hidden=%s, stem_channels=%d)",
        aug_ckpt_path, hidden_dims, stem_channels,
    )
    return aug_classifier


def adapt_coeff_inner(
    model: nn.Module,
    adapter,
    aug_classifier: nn.Module,
    support_images: torch.Tensor,
    c_init: torch.Tensor,
    inner_lr: float,
    inner_steps: int,
    inner_cfg: InnerLossConfig,
    clean_dist_ref: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Inner adaptation on coefficient vector using Phase-1 aug signals."""
    c = c_init.detach().clone().requires_grad_(True)
    last_diag: Dict[str, float] = {}

    for _ in range(inner_steps):
        stem_spatial = adapter.forward_stem_spatial(support_images, c)
        z = aug_classifier.encode(stem_spatial)
        aug_logits, dist_out = aug_classifier(z)

        loss, diag_t = compute_inner_loss(
            aug_logits=aug_logits,
            dist_out=dist_out,
            z_flat=z,
            coeff=c,
            cfg=inner_cfg,
            clean_dist_ref=clean_dist_ref,
        )
        grad_c = torch.autograd.grad(loss, c, create_graph=False)[0]
        c = (c - inner_lr * grad_c).detach().requires_grad_(True)
        last_diag = {k: float(v.item()) for k, v in diag_t.items()}

    return c.detach(), last_diag


@torch.no_grad()
def validate_clean(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = 20,
) -> Dict[str, float]:
    top1_m = AverageMeter()
    top5_m = AverageMeter()
    for bidx, (images, labels) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        acc = accuracy_topk(logits, labels, topk=(1, 5))
        top1_m.update(acc["top1"], images.size(0))
        top5_m.update(acc["top5"], images.size(0))
        if max_batches > 0 and (bidx + 1) >= max_batches:
            break
    return {"top1": top1_m.avg, "top5": top5_m.avg}


def validate_7aug_max_tta(
    model: nn.Module,
    adapter,
    aug_classifier: nn.Module,
    raw_val_dataset: Dataset,
    transform_names: List[str],
    base_transform,
    final_transform,
    args: argparse.Namespace,
    device: torch.device,
    inner_cfg: InnerLossConfig,
    clean_dist_ref: Optional[torch.Tensor] = None,
) -> Dict[str, object]:
    """Meta-cur style validation: per-transform fixed augmentation at max SL."""
    from timm.data import augmix_sl_ops_v2
    from timm.data.auto_augment import AugMixSLAugmentFixed

    model.eval()
    adapter.eval()
    aug_classifier.eval()

    val_transform_ops = augmix_sl_ops_v2()
    num_transforms = min(len(transform_names), len(val_transform_ops))
    severity_level = args.max_sl if args.val_7aug_severity < 0 else args.val_7aug_severity
    val_batch_size = args.val_batch_size or args.batch_size
    val_workers = args.workers if args.val_7aug_workers < 0 else args.val_7aug_workers
    max_batches = args.val_7aug_batches

    inner_lr = args.inner_lr_train if args.val_7aug_inner_lr <= 0 else args.val_7aug_inner_lr
    inner_steps = args.inner_steps_train if args.val_7aug_inner_steps <= 0 else args.val_7aug_inner_steps

    carry_ctrl = CarryController(
        CarryPolicy(
            decay=args.carry_decay,
            reset_every=args.carry_reset_every,
            scope=args.carry_scope,
        )
    )

    per_transform_top1: Dict[str, float] = {}
    per_transform_top5: Dict[str, float] = {}

    for t_idx in range(num_transforms):
        t_name = transform_names[t_idx]

        fixed_aug = AugMixSLAugmentFixed(
            ops=val_transform_ops,
            fixed_transforms=[(t_idx, severity_level)],
            max_sl=args.max_sl,
        )
        val_dataset = FixedTransformValidationDataset(
            dataset=raw_val_dataset,
            fixed_aug_transform=fixed_aug,
            base_transform=base_transform,
            final_transform=final_transform,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=val_workers,
            pin_memory=True,
            drop_last=False,
            persistent_workers=False,
        )

        top1_m = AverageMeter()
        top5_m = AverageMeter()
        carry_state = torch.zeros(args.num_bases, device=device)

        for bidx, (images, labels) in enumerate(val_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if args.val_7aug_mode == "static":
                c0 = torch.zeros_like(carry_state)
            else:
                c0 = carry_state.clone()

            c_final, _inner_diag = adapt_coeff_inner(
                model=model,
                adapter=adapter,
                aug_classifier=aug_classifier,
                support_images=images,
                c_init=c0,
                inner_lr=inner_lr,
                inner_steps=inner_steps,
                inner_cfg=inner_cfg,
                clean_dist_ref=clean_dist_ref,
            )

            with torch.no_grad():
                logits = adapter.forward_logits(model, images, c_final)
                acc = accuracy_topk(logits, labels, topk=(1, 5))
                top1_m.update(acc["top1"], images.size(0))
                top5_m.update(acc["top5"], images.size(0))

            if args.val_7aug_mode == "continual":
                carry_state = carry_ctrl.next_state(
                    mode="continual",
                    global_step=bidx,
                    prev_state=carry_state,
                    current_state=c_final,
                )

            if max_batches > 0 and (bidx + 1) >= max_batches:
                break

        per_transform_top1[t_name] = top1_m.avg
        per_transform_top5[t_name] = top5_m.avg
        _logger.info(
            "  Val7Aug [%s] (SL=%.2f, mode=%s): Top1=%.2f Top5=%.2f",
            t_name, severity_level, args.val_7aug_mode, top1_m.avg, top5_m.avg,
        )

    mean_top1 = sum(per_transform_top1.values()) / max(1, len(per_transform_top1))
    mean_top5 = sum(per_transform_top5.values()) / max(1, len(per_transform_top5))
    _logger.info("  Val7Aug mean: Top1=%.2f Top5=%.2f", mean_top1, mean_top5)
    return {
        "severity": severity_level,
        "mean_top1": mean_top1,
        "mean_top5": mean_top5,
        "per_transform_top1": per_transform_top1,
        "per_transform_top5": per_transform_top5,
    }


@torch.no_grad()
def _build_clean_dist_ref(value: Optional[list], num_classes: int, device: torch.device) -> Optional[torch.Tensor]:
    if value is None:
        return None
    t = torch.tensor(value, device=device, dtype=torch.float32)
    if t.numel() != num_classes:
        raise ValueError(f"clean_dist_ref length must be {num_classes}, got {t.numel()}")
    t = t.clamp(min=1e-8)
    t = t / t.sum()
    return t


def run_smoke_test(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    model = create_model(args.model, pretrained=False, num_classes=args.num_classes).to(device).eval()
    for p in model.parameters():
        p.requires_grad = False

    adapter = build_stem_adapter(
        model=model,
        target=args.stem_target,
        num_bases=args.num_bases,
        rank=args.adapter_rank,
        scale=args.adapter_scale,
        init_std=args.adapter_init_std,
        train_affine=args.train_stem_affine,
    ).to(device)

    # Fake phase1 aug classifier compatible with current model stem channels.
    stem_channels = adapter.spec.module.out_channels
    if "vit" in args.model.lower():
        from train_phase1_vit import DirectAugClassifier
        num_dw_stages = 2
    else:
        from train_phase1_resnet import DirectAugClassifier
        num_dw_stages = 5
    aug_classifier = DirectAugClassifier(stem_channels=stem_channels, output_channels=64, num_dw_stages=num_dw_stages)
    aug_classifier.to(device).eval()
    for p in aug_classifier.parameters():
        p.requires_grad = False

    opt = torch.optim.AdamW(adapter.parameters(), lr=1e-3)

    xq = torch.randn(args.batch_size, 3, args.img_size, args.img_size, device=device)
    xs = torch.randn(args.batch_size, 3, args.img_size, args.img_size, device=device)
    y = torch.randint(0, args.num_classes, (args.batch_size,), device=device)

    inner_cfg = InnerLossConfig(
        w_clean_rel=1.0,
        w_dist_rel=1.0,
        w_znorm_rel=1.0,
        w_trust=1e-3,
        p_clean_ref=0.55,
        dist_ref=2.0,
        znorm_ref=2.0,
        clean_index=0,
    )
    c0 = torch.zeros(args.num_bases, device=device)
    c_final, diag = adapt_coeff_inner(
        model=model,
        adapter=adapter,
        aug_classifier=aug_classifier,
        support_images=xs,
        c_init=c0,
        inner_lr=0.05,
        inner_steps=2,
        inner_cfg=inner_cfg,
    )
    logits = adapter.forward_logits(model, xq, c_final)
    loss = F.cross_entropy(logits, y)
    opt.zero_grad()
    loss.backward()
    opt.step()

    print(
        f"[SMOKE] ok | logits={tuple(logits.shape)} | outer_loss={loss.item():.4f} "
        f"| p_clean={diag.get('p_clean', 0):.4f}"
    )


def main():
    args = parse_args()
    setup_default_logging()
    seed_all(args.seed)

    if args.smoke_test:
        run_smoke_test(args)
        return

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    out_dir = Path(args.output) / args.experiment
    resume_path = _resolve_resume_path(args.resume, out_dir)
    resume_ckpt: Optional[Dict[str, object]] = None
    if resume_path is not None:
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        resume_ckpt = torch.load(resume_path, map_location="cpu")
        if not isinstance(resume_ckpt, dict):
            raise ValueError(f"Resume checkpoint must be a dict: {resume_path}")
        if not args.phase1_aug_ckpt:
            resume_phase1 = str(resume_ckpt.get("phase1_aug_ckpt", "")).strip()
            if resume_phase1:
                args.phase1_aug_ckpt = resume_phase1
                _logger.info("Using --phase1-aug-ckpt from resume checkpoint: %s", resume_phase1)

    _logger.info("Creating model: %s", args.model)
    model = create_model(args.model, pretrained=args.pretrained, num_classes=args.num_classes)
    if args.initial_checkpoint:
        load_checkpoint_state(model, args.initial_checkpoint)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False

    adapter = build_stem_adapter(
        model=model,
        target=args.stem_target,
        num_bases=args.num_bases,
        rank=args.adapter_rank,
        scale=args.adapter_scale,
        init_std=args.adapter_init_std,
        train_affine=args.train_stem_affine,
    ).to(device)

    stem_channels = adapter.spec.module.out_channels
    if not args.phase1_aug_ckpt:
        raise ValueError("Could not determine phase1 aug ckpt. Provide --phase1-aug-ckpt or --resume.")
    aug_classifier = _load_phase1_aug_classifier(
        model_name=args.model,
        aug_ckpt_path=args.phase1_aug_ckpt,
        stem_channels=stem_channels,
        device=device,
    )

    data_cfg = resolve_data_config(vars(args), model=model)
    base_transform = transforms.Compose([
        transforms.Resize(int(args.img_size / data_cfg["crop_pct"])),
        transforms.CenterCrop(args.img_size),
    ])
    final_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=data_cfg["mean"], std=data_cfg["std"]),
    ])

    augmix_transform = create_augmix_sl_transform(
        max_depth=args.new_depth,
        min_sl=args.min_sl,
        max_sl=args.max_sl,
        version=2,
        normalize_labels=True,
    )

    aa_transform = _build_aa_transform(
        args.aa or "", args.img_size, tuple(data_cfg["mean"]),
    )
    if aa_transform is not None:
        _logger.info("Using AA on support (before AugMixSL): %s", args.aa)

    train_dir = os.path.join(args.data_dir, args.train_split)
    val_dir = os.path.join(args.data_dir, args.val_split)
    raw_train = ImageFolder(train_dir)
    raw_val = ImageFolder(val_dir, transform=transforms.Compose([base_transform, final_transform]))
    raw_val_pil = ImageFolder(val_dir)  # No transform: for Val7Aug which applies base→aug→final

    train_ds = Phase2PairDataset(
        dataset=raw_train,
        base_transform=base_transform,
        final_transform=final_transform,
        augmix_transform=augmix_transform,
        aa_transform=aa_transform,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True if args.workers > 0 else False,
    )
    val_loader = DataLoader(
        raw_val,
        batch_size=args.val_batch_size or args.batch_size,
        shuffle=False,
        num_workers=min(args.workers, 4),
        pin_memory=True,
    )

    optimizer = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler, _ = create_scheduler_v2(
        optimizer, sched=args.sched, num_epochs=args.epochs,
        warmup_epochs=args.warmup_epochs, warmup_lr=args.warmup_lr,
        min_lr=args.min_lr,
    )

    inner_cfg = InnerLossConfig(
        w_clean_rel=args.w_clean_rel,
        w_dist_rel=args.w_dist_rel,
        w_znorm_rel=args.w_znorm_rel,
        w_trust=args.w_trust,
        w_kl=args.w_kl,
        p_clean_ref=args.p_clean_ref,
        dist_ref=args.dist_ref,
        znorm_ref=args.znorm_ref,
        rel_mode=args.rel_mode,
        normalize_rel=not args.disable_rel_norm,
        clean_index=0,
    )
    num_aug_classes = int(getattr(aug_classifier, "num_classes", aug_classifier.aug_head.out_features))
    clean_dist_ref = _build_clean_dist_ref(args.clean_dist_ref, num_classes=num_aug_classes, device=device)
    transform_names = get_augmix_sl_transform_names(version=2)

    scheduler_cfg = EpisodeSchedulerConfig(
        mode=args.episode_mode,
        ratio_start=args.continual_ratio_start,
        ratio_end=args.continual_ratio_end,
        total_epochs=args.epochs,
        pattern=args.episode_pattern,
        seed=args.seed,
    )
    episode_scheduler = EpisodeModeScheduler(scheduler_cfg)
    carry_ctrl = CarryController(
        CarryPolicy(
            decay=args.carry_decay,
            reset_every=args.carry_reset_every,
            scope=args.carry_scope,
        )
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "phase2_coupled_results.tsv"
    val7aug_path = out_dir / "phase2_coupled_val7aug.tsv"
    start_epoch = 0
    best_top1 = -1.0
    global_step = 0
    carry_state = torch.zeros(args.num_bases, device=device)

    if resume_ckpt is not None:
        if "adapter" not in resume_ckpt:
            raise ValueError(f"Resume checkpoint missing 'adapter': {resume_path}")
        adapter.load_state_dict(resume_ckpt["adapter"], strict=True)

        if args.resume_no_optimizer:
            _logger.info("Resume requested with --resume-no-optimizer: optimizer state will not be restored.")
        else:
            if "optimizer" not in resume_ckpt:
                raise ValueError(f"Resume checkpoint missing 'optimizer': {resume_path}")
            optimizer.load_state_dict(resume_ckpt["optimizer"])
            _optimizer_to_device(optimizer, device)

        start_epoch = int(resume_ckpt.get("epoch", -1)) + 1
        best_top1 = float(resume_ckpt.get("best_top1", best_top1))
        if "global_step" in resume_ckpt:
            global_step = int(resume_ckpt["global_step"])
        else:
            global_step = max(0, start_epoch * len(train_loader))

        resume_carry = resume_ckpt.get("carry_state", None)
        if resume_carry is not None:
            if not torch.is_tensor(resume_carry):
                resume_carry = torch.tensor(resume_carry, dtype=torch.float32)
            carry_state = resume_carry.to(device=device, dtype=torch.float32)
            if carry_state.numel() != args.num_bases:
                raise ValueError(
                    f"Resume carry_state size mismatch: expected {args.num_bases}, got {carry_state.numel()}"
                )
            carry_state = carry_state.view(args.num_bases)

        sched_rng = resume_ckpt.get("episode_scheduler_rng_state", None)
        if sched_rng is not None:
            episode_scheduler.rng.setstate(sched_rng)

        if not args.resume_no_optimizer and lr_scheduler is not None and resume_ckpt.get("lr_scheduler") is not None:
            lr_scheduler.load_state_dict(resume_ckpt["lr_scheduler"])

        _restore_rng_state(resume_ckpt.get("rng_state"))

        resume_phase1 = str(resume_ckpt.get("phase1_aug_ckpt", "")).strip()
        if resume_phase1 and args.phase1_aug_ckpt != resume_phase1:
            _logger.warning(
                "phase1_aug_ckpt mismatch on resume: arg=%s, ckpt=%s",
                args.phase1_aug_ckpt, resume_phase1,
            )

        _logger.info(
            "Resumed from %s | start_epoch=%d global_step=%d best_top1=%.3f carry_norm=%.4f",
            str(resume_path), start_epoch, global_step, best_top1, float(carry_state.norm().item()),
        )

    _ensure_results_header(results_path, reset=(resume_ckpt is None))
    if args.val_7aug_max:
        val7_header = ["epoch"] + [f"top1_{n}" for n in transform_names] + ["mean_top1", "mean_top5", "severity"]
        _ensure_tsv_header(val7aug_path, val7_header, reset=(resume_ckpt is None))
        _logger.info(
            "Val7Aug enabled: interval=%d mode=%s severity=%s",
            args.val_7aug_interval,
            args.val_7aug_mode,
            "max_sl" if args.val_7aug_severity < 0 else f"{args.val_7aug_severity:.2f}",
        )

    _logger.info(
        "Start Phase-2 (phase1-aligned): model=%s, stem=%s, bases=%d, rank=%d, "
        "start_epoch=%d/%d",
        args.model, adapter.spec.weight_name, args.num_bases, args.adapter_rank, start_epoch, args.epochs,
    )

    if start_epoch >= args.epochs:
        _logger.info(
            "Nothing to run: start_epoch(%d) >= epochs(%d).",
            start_epoch, args.epochs,
        )
        return

    for epoch in range(start_epoch, args.epochs):
        model.eval()
        adapter.train()

        if args.curriculum:
            cur_depth = get_curriculum_depth(epoch, args.epochs, args.new_depth)
            cur_min_sl, cur_max_sl = get_curriculum_sl_range(
                epoch=epoch,
                total_epochs=args.epochs,
                min_sl_start=args.cur_min_sl_start,
                min_sl_end=args.min_sl,
                max_sl_start=args.cur_max_sl_start,
                max_sl_end=args.max_sl,
            )
            train_ds.augmix_transform.max_depth = cur_depth
            train_ds.augmix_transform.min_sl = cur_min_sl
            train_ds.augmix_transform.max_sl = cur_max_sl
            _logger.info(
                "Epoch %d curriculum: depth=%d, sl=[%.2f, %.2f]",
                epoch, cur_depth, cur_min_sl, cur_max_sl,
            )

        loss_m = AverageMeter()
        top1_m = AverageMeter()
        pclean_m = AverageMeter()
        dist_m = AverageMeter()
        znorm_m = AverageMeter()
        entropy_m = AverageMeter()
        static_count = 0
        continual_count = 0
        time_m = AverageMeter()
        end = time.time()

        for bidx, (query_images, support_images, labels) in enumerate(train_loader):
            query_images = query_images.to(device, non_blocking=True)
            support_images = support_images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            mode = episode_scheduler.mode_for_step(epoch, global_step)
            if mode == "static":
                c0 = torch.zeros_like(carry_state)
                static_count += 1
            else:
                c0 = carry_state.clone()
                continual_count += 1

            c_final, inner_diag = adapt_coeff_inner(
                model=model,
                adapter=adapter,
                aug_classifier=aug_classifier,
                support_images=support_images,
                c_init=c0,
                inner_lr=args.inner_lr_train,
                inner_steps=args.inner_steps_train,
                inner_cfg=inner_cfg,
                clean_dist_ref=clean_dist_ref,
            )

            logits_q = adapter.forward_logits(model, query_images, c_final)
            outer_loss = F.cross_entropy(logits_q, labels)

            optimizer.zero_grad()
            outer_loss.backward()
            if args.clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), args.clip_grad)
            optimizer.step()

            with torch.no_grad():
                acc = accuracy_topk(logits_q, labels, topk=(1, 5))
                bs = labels.size(0)
                loss_m.update(outer_loss.item(), bs)
                top1_m.update(acc["top1"], bs)
                pclean_m.update(inner_diag.get("p_clean", 0.0), bs)
                dist_m.update(inner_diag.get("dist", 0.0), bs)
                znorm_m.update(inner_diag.get("znorm", 0.0), bs)
                entropy_m.update(inner_diag.get("entropy", 0.0), bs)

                carry_state = carry_ctrl.next_state(
                    mode=mode,
                    global_step=global_step,
                    prev_state=carry_state,
                    current_state=c_final,
                )

            global_step += 1
            time_m.update(time.time() - end)
            end = time.time()

            if bidx % args.log_interval == 0 or bidx == len(train_loader) - 1:
                _logger.info(
                    "Epoch %d [%4d/%4d] L=%.4f (%.4f) T1=%.2f mode=%s "
                    "p=%.3f d=%.3f z=%.3f H=%.3f dt=%.2fs",
                    epoch, bidx, len(train_loader),
                    loss_m.val, loss_m.avg, top1_m.avg, mode,
                    pclean_m.avg, dist_m.avg, znorm_m.avg, entropy_m.avg, time_m.val,
                )

        val_clean = validate_clean(model, val_loader, device, max_batches=args.val_batches)
        val7_metrics = None
        if args.val_7aug_max and (((epoch + 1) % args.val_7aug_interval == 0) or epoch == args.epochs - 1):
            _logger.info("Running meta-style Val7Aug TTA (epoch %d) ...", epoch)
            val7_metrics = validate_7aug_max_tta(
                model=model,
                adapter=adapter,
                aug_classifier=aug_classifier,
                raw_val_dataset=raw_val_pil,
                transform_names=transform_names,
                base_transform=base_transform,
                final_transform=final_transform,
                args=args,
                device=device,
                inner_cfg=inner_cfg,
                clean_dist_ref=clean_dist_ref,
            )
        is_best = ""
        if val_clean["top1"] > best_top1:
            best_top1 = val_clean["top1"]
            is_best = "*"

        if lr_scheduler is not None:
            lr_scheduler.step(epoch + 1)

        ckpt = {
            "epoch": epoch,
            "adapter": adapter.state_dict(),
            "optimizer": optimizer.state_dict(),
            "lr_scheduler": lr_scheduler.state_dict() if lr_scheduler is not None else None,
            "args": vars(args),
            "best_top1": best_top1,
            "global_step": global_step,
            "carry_state": carry_state.detach().cpu(),
            "episode_scheduler_rng_state": episode_scheduler.rng.getstate(),
            "rng_state": _capture_rng_state(),
            "adapter_config": {
                "target": args.stem_target,
                "num_bases": args.num_bases,
                "rank": args.adapter_rank,
                "scale": args.adapter_scale,
                "init_std": args.adapter_init_std,
                "train_affine": args.train_stem_affine,
            },
            "phase1_aug_ckpt": args.phase1_aug_ckpt,
        }
        torch.save(ckpt, out_dir / "last.pth.tar")
        if is_best:
            torch.save(ckpt, out_dir / "best.pth.tar")

        with open(results_path, "a") as f:
            f.write(
                f"{epoch}\t{loss_m.avg:.6f}\t{top1_m.avg:.3f}\t"
                f"{pclean_m.avg:.4f}\t{dist_m.avg:.4f}\t{znorm_m.avg:.4f}\t{entropy_m.avg:.4f}\t"
                f"{static_count}\t{continual_count}\t{val_clean['top1']:.3f}\t{val_clean['top5']:.3f}\t{is_best}\n"
            )
        if val7_metrics is not None:
            row = [str(epoch)]
            for name in transform_names:
                row.append(f"{val7_metrics['per_transform_top1'].get(name, 0.0):.3f}")
            row += [
                f"{val7_metrics['mean_top1']:.3f}",
                f"{val7_metrics['mean_top5']:.3f}",
                f"{val7_metrics['severity']:.3f}",
            ]
            with open(val7aug_path, "a") as f:
                f.write("\t".join(row) + "\n")

        _logger.info(
            "Epoch %d done: train_top1=%.2f val_clean_top1=%.2f (best=%.2f) mode_count S/C=%d/%d",
            epoch, top1_m.avg, val_clean["top1"], best_top1, static_count, continual_count,
        )
        if val7_metrics is not None:
            _logger.info(
                "Epoch %d Val7Aug mean_top1=%.2f mean_top5=%.2f",
                epoch, val7_metrics["mean_top1"], val7_metrics["mean_top5"],
            )

    _logger.info("Training finished. best_clean_top1=%.3f", best_top1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Phase-2 coupled LoRA training (phase1 aligned)")

    g = parser.add_argument_group("Data")
    g.add_argument("--data-dir", type=str, default="", help="ImageNet root containing train/val splits.")
    g.add_argument("--train-split", type=str, default="train")
    g.add_argument("--val-split", type=str, default="val")
    g.add_argument("--img-size", type=int, default=224)
    g.add_argument("--batch-size", type=int, default=64)
    g.add_argument("--val-batch-size", type=int, default=0)
    g.add_argument("--workers", type=int, default=4)

    g = parser.add_argument_group("Model")
    g.add_argument("--model", type=str, default="resnet50")
    g.add_argument("--num-classes", type=int, default=1000)
    g.add_argument("--pretrained", action="store_true", default=False)
    g.add_argument("--initial-checkpoint", type=str, default="")
    g.add_argument("--phase1-aug-ckpt", type=str, default="", help="Phase-1 aug classifier checkpoint.")
    g.add_argument("--stem-target", type=str, default="auto", choices=["auto", "conv1", "patch_embed.proj"])

    g = parser.add_argument_group("Adapter")
    g.add_argument("--num-bases", type=int, default=4)
    g.add_argument("--adapter-rank", type=int, default=4)
    g.add_argument("--adapter-scale", type=float, default=1.0)
    g.add_argument("--adapter-init-std", type=float, default=1e-3)
    g.add_argument("--train-stem-affine", action="store_true", default=False)

    g = parser.add_argument_group("Train Augmentation")
    g.add_argument("--new-depth", type=int, default=3)
    g.add_argument("--min-sl", type=float, default=0.3)
    g.add_argument("--max-sl", type=float, default=1.0)
    g.add_argument("--aa", type=str, default=None, metavar="NAME",
                   help="AutoAugment/RandAugment policy (e.g. rand-m9-mstd0.5). Applied before AugMixSL on support.")
    g.add_argument("--curriculum", action="store_true", default=False)
    g.add_argument("--cur-min-sl-start", type=float, default=0.1)
    g.add_argument("--cur-max-sl-start", type=float, default=0.5)

    g = parser.add_argument_group("Inner Loop (support)")
    g.add_argument("--inner-lr-train", type=float, default=0.05)
    g.add_argument("--inner-steps-train", type=int, default=1)
    g.add_argument("--w-clean-rel", type=float, default=1.0)
    g.add_argument("--w-dist-rel", type=float, default=1.0)
    g.add_argument("--w-znorm-rel", type=float, default=1.0)
    g.add_argument("--w-trust", type=float, default=1e-3)
    g.add_argument("--w-kl", type=float, default=0.0)
    g.add_argument("--clean-dist-ref", type=float, nargs="+", default=None,
                   help="Optional clean distribution ref for KL (length 8 for phase1 classifier).")
    g.add_argument("--p-clean-ref", type=float, default=0.55)
    g.add_argument("--dist-ref", type=float, default=2.0)
    g.add_argument("--znorm-ref", type=float, default=2.0)
    g.add_argument("--rel-mode", type=str, default="relu", choices=["relu", "mse"])
    g.add_argument("--disable-rel-norm", action="store_true", default=False)

    g = parser.add_argument_group("Episode Schedule")
    g.add_argument("--episode-mode", type=str, default="mixed", choices=["static", "continual", "mixed"])
    g.add_argument("--continual-ratio-start", type=float, default=0.2)
    g.add_argument("--continual-ratio-end", type=float, default=0.5)
    g.add_argument("--episode-pattern", type=str, default="", help="Manual pattern like 'S,S,C,C'.")
    g.add_argument("--carry-decay", type=float, default=1.0)
    g.add_argument("--carry-reset-every", type=int, default=0)
    g.add_argument("--carry-scope", type=str, default="within_domain", choices=["none", "within_episode", "within_domain"])

    g = parser.add_argument_group("Optimization")
    g.add_argument("--epochs", type=int, default=20)
    g.add_argument("--lr", type=float, default=1e-3)
    g.add_argument("--sched", type=str, default="cosine")
    g.add_argument("--warmup-epochs", type=int, default=2)
    g.add_argument("--warmup-lr", type=float, default=1e-5)
    g.add_argument("--min-lr", type=float, default=1e-6)
    g.add_argument("--weight-decay", type=float, default=1e-4)
    g.add_argument("--clip-grad", type=float, default=1.0)

    g = parser.add_argument_group("Validation (Meta-Style 7Aug)")
    g.add_argument("--val-7aug-max", action="store_true", default=False,
                   help="Run per-transform TTA validation on 7 AugMixSL-V2 transforms.")
    g.add_argument("--val-7aug-interval", type=int, default=1,
                   help="Run Val7Aug every N epochs.")
    g.add_argument("--val-7aug-severity", type=float, default=-1.0,
                   help="Severity SL for Val7Aug. <0 uses --max-sl.")
    g.add_argument("--val-7aug-batches", type=int, default=0,
                   help="Max batches per transform for Val7Aug (0 = full).")
    g.add_argument("--val-7aug-workers", type=int, default=-1,
                   help="DataLoader workers for Val7Aug (-1 = use --workers).")
    g.add_argument("--val-7aug-mode", type=str, default="static", choices=["static", "continual"],
                   help="Inner-loop carry mode during Val7Aug.")
    g.add_argument("--val-7aug-inner-lr", type=float, default=-1.0,
                   help="Inner LR for Val7Aug (<=0 uses --inner-lr-train).")
    g.add_argument("--val-7aug-inner-steps", type=int, default=-1,
                   help="Inner steps for Val7Aug (<=0 uses --inner-steps-train).")

    g = parser.add_argument_group("Misc")
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--no-cuda", action="store_true", default=False)
    g.add_argument("--log-interval", type=int, default=20)
    g.add_argument("--val-batches", type=int, default=20)
    g.add_argument("--output", type=str, default="./output")
    g.add_argument("--experiment", type=str, default="phase2_coupled_lora")
    g.add_argument("--resume", type=str, default="",
                   help="Resume from checkpoint path. Use 'auto' for <output>/<experiment>/last.pth.tar.")
    g.add_argument("--resume-no-optimizer", action="store_true", default=False,
                   help="Resume model/state but do not restore optimizer state.")
    g.add_argument("--smoke-test", action="store_true", default=False)

    args = parser.parse_args()

    if not args.smoke_test:
        if not args.data_dir:
            parser.error("--data-dir is required unless --smoke-test is set.")
        if not args.phase1_aug_ckpt and not args.resume:
            parser.error("--phase1-aug-ckpt is required unless --smoke-test or --resume is set.")
    if args.val_7aug_interval <= 0:
        parser.error("--val-7aug-interval must be > 0.")
    if args.val_7aug_batches < 0:
        parser.error("--val-7aug-batches must be >= 0.")
    return args


if __name__ == "__main__":
    main()

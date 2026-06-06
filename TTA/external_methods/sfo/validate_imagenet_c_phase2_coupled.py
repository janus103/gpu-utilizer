#!/usr/bin/env python3
"""ImageNet-C evaluation for Phase-2 coupled TTA (Phase-1 aligned).

Phase-1 aligned behavior:
- Uses Phase-1 DirectAugClassifier checkpoint as adaptation signal source.
- No FSC dependency.
- Raw-input TTA only (no synthetic image generation during test).

Modes:
1) adapter mode (recommended): provide --adapter-ckpt from phase2 training.
2) direct mode (bootstrap): no adapter ckpt, directly updates stem weight in-place
   using Phase-1 signals (baseline fallback).
"""

from __future__ import annotations

import argparse
import copy
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from timm.data import create_transform, get_augmix_sl_num_transforms, resolve_data_config
from timm.models import create_model
from timm.utils import AverageMeter, setup_default_logging

from phase2_episode_scheduler import CarryController, CarryPolicy
from phase2_losses import (
    InnerLossConfig,
    StepGuardConfig,
    compute_inner_loss,
    is_clean_like,
    stability_score,
    violates_caps,
)
from phase2_stem_adapter import build_stem_adapter, resolve_stem_target

_logger = logging.getLogger("validate_phase2_coupled")


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
    """ImageNet-C folder that may contain class-subdirs or flat files."""

    def __init__(self, root: Path, filename_label_map: Dict[str, int], class_to_idx: Dict[str, int], transform):
        self.root = root
        self.transform = transform

        entries = list(root.iterdir())
        has_class_dirs = any(p.is_dir() for p in entries)
        paths: List[Path] = []
        labels: List[int] = []
        patterns = ("*.JPEG", "*.jpeg", "*.jpg", "*.png")

        if has_class_dirs:
            for class_dir in sorted(p for p in entries if p.is_dir()):
                target = class_to_idx.get(class_dir.name)
                if target is None:
                    continue
                for pattern in patterns:
                    for path in sorted(class_dir.glob(pattern)):
                        paths.append(path)
                        labels.append(target)
        else:
            all_paths: List[Path] = []
            for pattern in patterns:
                all_paths.extend(sorted(root.glob(pattern)))
            for path in all_paths:
                target = filename_label_map.get(path.name)
                if target is None:
                    continue
                paths.append(path)
                labels.append(target)

        if not paths:
            raise RuntimeError(f"No usable images found in {root}")
        self.paths = paths
        self.targets = labels

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), self.targets[idx]


def _build_label_map(val_dir: Path) -> Tuple[Dict[str, int], Dict[str, int]]:
    from torchvision import datasets

    val_dataset = datasets.ImageFolder(val_dir)
    filename_label_map = {Path(p).name: y for p, y in val_dataset.samples}
    class_to_idx = val_dataset.class_to_idx
    return filename_label_map, class_to_idx


@torch.no_grad()
def eval_logits(logits: torch.Tensor, target: torch.Tensor) -> Tuple[float, float]:
    maxk = min(5, logits.size(1))
    _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    n = target.size(0)
    top1 = correct[:1].reshape(-1).float().sum(0).item() * 100.0 / n
    top5 = correct[:maxk].reshape(-1).float().sum(0).item() * 100.0 / n
    return top1, top5


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
    else:
        from train_phase1_resnet import DirectAugClassifier

    aug_classifier = DirectAugClassifier(
        stem_channels=stem_channels,
        output_channels=64,
        num_dw_stages=5,
        num_transforms=num_transforms,
        hidden_dims=hidden_dims,
        dropout=dropout,
        dw_init_mode=dw_init_mode,
    )
    aug_classifier.load_state_dict(ckpt["aug_classifier"], strict=True)
    aug_classifier.to(device).eval()
    for p in aug_classifier.parameters():
        p.requires_grad = False

    _logger.info("Loaded phase1 aug classifier: %s", aug_ckpt_path)
    return aug_classifier


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


def _forward_stem_spatial_direct(model: nn.Module, target_name: str, images: torch.Tensor) -> torch.Tensor:
    if target_name == "conv1.weight":
        return model.conv1(images.contiguous())
    if target_name == "patch_embed.proj.weight":
        return model.patch_embed.proj(images.contiguous())
    raise ValueError(f"Unsupported direct target: {target_name}")


def _inner_adapt_coeff_with_guard(
    model: nn.Module,
    adapter,
    aug_classifier: nn.Module,
    images: torch.Tensor,
    c_init: torch.Tensor,
    inner_lr: float,
    inner_steps: int,
    inner_cfg: InnerLossConfig,
    guard_cfg: StepGuardConfig,
    clean_dist_ref: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """Adapter mode: adapt coefficient vector only."""

    def _collect(coeff: torch.Tensor) -> Dict[str, float]:
        with torch.no_grad():
            stem_spatial = adapter.forward_stem_spatial(images, coeff)
            z = aug_classifier.encode(stem_spatial)
            aug_logits, dist_out = aug_classifier(z)
            _loss, diag_t = compute_inner_loss(
                aug_logits=aug_logits,
                dist_out=dist_out,
                z_flat=z,
                coeff=coeff,
                cfg=inner_cfg,
                clean_dist_ref=clean_dist_ref,
            )
        return {k: float(v.item()) for k, v in diag_t.items()}

    c = c_init.detach().clone()
    before = _collect(c)
    stats = {
        "accepted": 0.0,
        "rejected": 0.0,
        "skipped": 0.0,
        "before_p_clean": before["p_clean"],
        "before_dist": before["dist"],
        "before_znorm": before["znorm"],
    }

    if guard_cfg.enabled and guard_cfg.skip_clean_like and is_clean_like(before, inner_cfg, guard_cfg):
        stats["skipped"] = 1.0
        stats.update({
            "after_p_clean": before["p_clean"],
            "after_dist": before["dist"],
            "after_znorm": before["znorm"],
            "coeff_norm": float(c.norm().item()),
        })
        return c, stats

    best_c = c.clone()
    best_score = stability_score(before, inner_cfg, guard_cfg)

    for _ in range(inner_steps):
        c_var = c.detach().clone().requires_grad_(True)
        stem_spatial = adapter.forward_stem_spatial(images, c_var)
        z = aug_classifier.encode(stem_spatial)
        aug_logits, dist_out = aug_classifier(z)
        loss, _diag = compute_inner_loss(
            aug_logits=aug_logits,
            dist_out=dist_out,
            z_flat=z,
            coeff=c_var,
            cfg=inner_cfg,
            clean_dist_ref=clean_dist_ref,
        )
        grad_c = torch.autograd.grad(loss, c_var, create_graph=False)[0]
        cand = (c_var - inner_lr * grad_c).detach()

        if not guard_cfg.enabled:
            c = cand
            best_c = cand
            stats["accepted"] += 1.0
            continue

        cand_diag = _collect(cand)
        coeff_norm = float(cand.norm().item())
        if violates_caps(cand_diag, coeff_norm, inner_cfg, guard_cfg):
            stats["rejected"] += 1.0
            continue

        score = stability_score(cand_diag, inner_cfg, guard_cfg)
        if score >= best_score + guard_cfg.min_score_improve:
            c = cand
            best_c = cand
            best_score = score
            stats["accepted"] += 1.0
        else:
            stats["rejected"] += 1.0

    if guard_cfg.enabled:
        c = best_c

    after = _collect(c)
    stats.update({
        "after_p_clean": after["p_clean"],
        "after_dist": after["dist"],
        "after_znorm": after["znorm"],
        "coeff_norm": float(c.norm().item()),
    })
    return c, stats


def _inner_adapt_direct_with_guard(
    model: nn.Module,
    target_name: str,
    aug_classifier: nn.Module,
    images: torch.Tensor,
    inner_lr: float,
    inner_steps: int,
    inner_momentum: float,
    lam: float,
    clip_grad: float,
    inner_cfg: InnerLossConfig,
    guard_cfg: StepGuardConfig,
    clean_dist_ref: Optional[torch.Tensor] = None,
    max_param_delta: float = 1.0,
) -> Dict[str, float]:
    """Direct mode: adapt stem weight in-place (phase1 baseline fallback)."""
    stem_param = dict(model.named_parameters())[target_name]
    original = stem_param.data.clone()

    for p in model.parameters():
        p.requires_grad = False
    stem_param.requires_grad = True

    optimizer = torch.optim.SGD([stem_param], lr=inner_lr, momentum=inner_momentum)

    dummy_coeff = torch.zeros(1, device=images.device)

    def _collect() -> Dict[str, float]:
        with torch.no_grad():
            stem_spatial = _forward_stem_spatial_direct(model, target_name, images)
            z = aug_classifier.encode(stem_spatial)
            aug_logits, dist_out = aug_classifier(z)
            _loss, diag_t = compute_inner_loss(
                aug_logits=aug_logits,
                dist_out=dist_out,
                z_flat=z,
                coeff=dummy_coeff,
                cfg=inner_cfg,
                clean_dist_ref=clean_dist_ref,
            )
        return {k: float(v.item()) for k, v in diag_t.items()}

    before = _collect()
    stats = {
        "accepted": 0.0,
        "rejected": 0.0,
        "skipped": 0.0,
        "before_p_clean": before["p_clean"],
        "before_dist": before["dist"],
        "before_znorm": before["znorm"],
    }

    if guard_cfg.enabled and guard_cfg.skip_clean_like and is_clean_like(before, inner_cfg, guard_cfg):
        stats["skipped"] = 1.0
        stats.update({
            "after_p_clean": before["p_clean"],
            "after_dist": before["dist"],
            "after_znorm": before["znorm"],
            "param_delta": 0.0,
        })
        return {"original_weight": original, **stats}

    best_weight = original.clone()
    best_score = stability_score(before, inner_cfg, guard_cfg)

    for _ in range(inner_steps):
        prev = stem_param.data.clone()
        optimizer.zero_grad()

        stem_spatial = _forward_stem_spatial_direct(model, target_name, images)
        z = aug_classifier.encode(stem_spatial)
        aug_logits, dist_out = aug_classifier(z)
        loss, _diag = compute_inner_loss(
            aug_logits=aug_logits,
            dist_out=dist_out,
            z_flat=z,
            coeff=dummy_coeff,
            cfg=inner_cfg,
            clean_dist_ref=clean_dist_ref,
        )
        if lam > 0:
            loss = loss + lam * (stem_param - original).pow(2).sum()

        loss.backward()
        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_([stem_param], clip_grad)
        optimizer.step()

        if not guard_cfg.enabled:
            best_weight = stem_param.data.clone()
            stats["accepted"] += 1.0
            continue

        cand = _collect()
        param_delta = (stem_param.data - original).norm().item()
        dist_limit = inner_cfg.dist_ref * (1.0 + guard_cfg.max_dist_rise)
        znorm_limit = inner_cfg.znorm_ref * (1.0 + guard_cfg.max_znorm_rise)
        violates = (
            float(cand["dist"]) > dist_limit
            or float(cand["znorm"]) > znorm_limit
            or (max_param_delta > 0 and param_delta > max_param_delta)
        )
        if violates:
            stem_param.data.copy_(prev)
            stats["rejected"] += 1.0
            continue

        score = stability_score(cand, inner_cfg, guard_cfg)
        if score >= best_score + guard_cfg.min_score_improve:
            best_score = score
            best_weight = stem_param.data.clone()
            stats["accepted"] += 1.0
        else:
            stem_param.data.copy_(prev)
            stats["rejected"] += 1.0

    if guard_cfg.enabled:
        stem_param.data.copy_(best_weight)

    after = _collect()
    stats.update({
        "after_p_clean": after["p_clean"],
        "after_dist": after["dist"],
        "after_znorm": after["znorm"],
        "param_delta": float((stem_param.data - original).norm().item()),
    })
    stem_param.requires_grad = False
    return {"original_weight": original, **stats}


def _evaluate_corruption(
    model: nn.Module,
    adapter,
    use_adapter_mode: bool,
    target_name: str,
    loader: DataLoader,
    aug_classifier: nn.Module,
    args: argparse.Namespace,
    clean_dist_ref: Optional[torch.Tensor],
) -> Tuple[float, float, float, float, Dict[str, float]]:
    pre1_m, pre5_m = AverageMeter(), AverageMeter()
    ada1_m, ada5_m = AverageMeter(), AverageMeter()

    agg = {
        "accepted": 0.0,
        "rejected": 0.0,
        "skipped": 0.0,
        "before_p_clean": 0.0,
        "after_p_clean": 0.0,
        "before_dist": 0.0,
        "after_dist": 0.0,
        "before_znorm": 0.0,
        "after_znorm": 0.0,
        "state_delta": 0.0,
        "inner_ms": 0.0,
        "forward_ms": 0.0,
    }

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
    guard_cfg = StepGuardConfig(
        enabled=not args.disable_tta_guard,
        skip_clean_like=not args.disable_clean_skip,
        skip_tol=args.skip_tol,
        max_dist_rise=args.max_dist_rise,
        max_znorm_rise=args.max_znorm_rise,
        max_coeff_norm=args.max_coeff_norm,
        score_dist_penalty=args.score_dist_penalty,
        score_znorm_penalty=args.score_znorm_penalty,
        min_score_improve=args.min_score_improve,
    )

    carry_ctrl = CarryController(
        CarryPolicy(
            decay=args.carry_decay,
            reset_every=args.carry_reset_every,
            scope=args.carry_scope,
        )
    )
    static_mode = args.tta_mode == "static"

    if use_adapter_mode:
        carry_state = torch.zeros(args.num_bases, device=args.device)
    else:
        carry_state = None
        pretrained_state = copy.deepcopy(model.state_dict())

    total_batches = 0

    for bidx, (images, target) in enumerate(loader):
        images = images.to(args.device, non_blocking=True)
        target = target.to(args.device, non_blocking=True)

        if use_adapter_mode:
            c0 = torch.zeros(args.num_bases, device=args.device) if static_mode else carry_state.clone()
            t0 = time.perf_counter()
            with torch.no_grad():
                pre_logits = adapter.forward_logits(model, images, c0)
            t1 = time.perf_counter()
            pre1, pre5 = eval_logits(pre_logits, target)

            c_final, stats = _inner_adapt_coeff_with_guard(
                model=model,
                adapter=adapter,
                aug_classifier=aug_classifier,
                images=images,
                c_init=c0,
                inner_lr=args.inner_lr,
                inner_steps=args.inner_steps,
                inner_cfg=inner_cfg,
                guard_cfg=guard_cfg,
                clean_dist_ref=clean_dist_ref,
            )
            t2 = time.perf_counter()
            with torch.no_grad():
                ada_logits = adapter.forward_logits(model, images, c_final)
            t3 = time.perf_counter()

            if static_mode:
                pass
            else:
                carry_state = carry_ctrl.next_state(
                    mode="continual",
                    global_step=bidx,
                    prev_state=carry_state,
                    current_state=c_final,
                )
            delta_state = float(c_final.norm().item())
        else:
            if static_mode or bidx == 0:
                model.load_state_dict(pretrained_state)

            t0 = time.perf_counter()
            with torch.no_grad():
                pre_logits = model(images)
            t1 = time.perf_counter()
            pre1, pre5 = eval_logits(pre_logits, target)

            direct = _inner_adapt_direct_with_guard(
                model=model,
                target_name=target_name,
                aug_classifier=aug_classifier,
                images=images,
                inner_lr=args.inner_lr,
                inner_steps=args.inner_steps,
                inner_momentum=args.inner_momentum,
                lam=args.lam,
                clip_grad=args.clip_grad,
                inner_cfg=inner_cfg,
                guard_cfg=guard_cfg,
                clean_dist_ref=clean_dist_ref,
                max_param_delta=args.max_param_delta,
            )
            t2 = time.perf_counter()
            with torch.no_grad():
                ada_logits = model(images)
            t3 = time.perf_counter()

            if static_mode:
                model.load_state_dict(pretrained_state)

            stats = direct
            delta_state = float(direct.get("param_delta", 0.0))

        ada1, ada5 = eval_logits(ada_logits, target)
        pre1_m.update(pre1, images.size(0))
        pre5_m.update(pre5, images.size(0))
        ada1_m.update(ada1, images.size(0))
        ada5_m.update(ada5, images.size(0))

        agg["accepted"] += stats["accepted"]
        agg["rejected"] += stats["rejected"]
        agg["skipped"] += stats["skipped"]
        agg["before_p_clean"] += stats["before_p_clean"]
        agg["after_p_clean"] += stats["after_p_clean"]
        agg["before_dist"] += stats["before_dist"]
        agg["after_dist"] += stats["after_dist"]
        agg["before_znorm"] += stats["before_znorm"]
        agg["after_znorm"] += stats["after_znorm"]
        agg["state_delta"] += delta_state
        agg["inner_ms"] += (t2 - t1) * 1000.0
        agg["forward_ms"] += ((t1 - t0) + (t3 - t2)) * 1000.0
        total_batches += 1

        if bidx % args.log_interval == 0 or bidx == len(loader) - 1:
            _logger.info(
                "  Batch %d: pre=%.2f -> ada=%.2f | p=%.3f->%.3f d=%.3f->%.3f z=%.3f->%.3f "
                "acc/rej=%.1f/%.1f skip=%.1f",
                bidx, pre1, ada1,
                stats["before_p_clean"], stats["after_p_clean"],
                stats["before_dist"], stats["after_dist"],
                stats["before_znorm"], stats["after_znorm"],
                stats["accepted"], stats["rejected"], stats["skipped"],
            )

    if total_batches > 0:
        for k in agg:
            agg[k] /= total_batches

    return pre1_m.avg, pre5_m.avg, ada1_m.avg, ada5_m.avg, agg


def _result_filename(args: argparse.Namespace, use_adapter_mode: bool) -> str:
    kind = "coupled" if use_adapter_mode else "direct"
    name = [
        "p2tta",
        kind,
        args.tta_mode,
        f"lr{args.inner_lr}",
        f"s{args.inner_steps}",
        f"sev{args.severity}",
    ]
    if use_adapter_mode:
        name += [f"b{args.num_bases}", f"r{args.adapter_rank}"]
    if not args.disable_tta_guard:
        name.append("guard")
    return "_".join(name) + ".txt"


def _save_results(
    results: List[Dict],
    summaries: Dict[str, Dict[str, float]],
    args: argparse.Namespace,
    use_adapter_mode: bool,
) -> Path:
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out = results_dir / _result_filename(args, use_adapter_mode)

    with open(out, "w") as f:
        f.write("Corruption\tPre_Top1\tPre_Top5\tAdapted_Top1\tAdapted_Top5\tDelta_Top1\n")
        for r in results:
            delta = r["adapted_top1"] - r["pre_top1"]
            f.write(
                f"{r['corruption']}\t{r['pre_top1']:.3f}\t{r['pre_top5']:.3f}\t"
                f"{r['adapted_top1']:.3f}\t{r['adapted_top5']:.3f}\t{delta:+.3f}\n"
            )

        if results:
            n = len(results)
            mp1 = sum(r["pre_top1"] for r in results) / n
            mp5 = sum(r["pre_top5"] for r in results) / n
            ma1 = sum(r["adapted_top1"] for r in results) / n
            ma5 = sum(r["adapted_top5"] for r in results) / n
            f.write(f"mean\t{mp1:.3f}\t{mp5:.3f}\t{ma1:.3f}\t{ma5:.3f}\t{(ma1-mp1):+.3f}\n")

        f.write("\n# --- TTA Summary ---\n")
        for corr, s in summaries.items():
            f.write(
                f"# {corr}: p={s['before_p_clean']:.4f}->{s['after_p_clean']:.4f}, "
                f"d={s['before_dist']:.3f}->{s['after_dist']:.3f}, "
                f"z={s['before_znorm']:.3f}->{s['after_znorm']:.3f}, "
                f"acc/rej={s['accepted']:.2f}/{s['rejected']:.2f}, skip={s['skipped']:.2f}, "
                f"delta={s['state_delta']:.5f}, inner_ms={s['inner_ms']:.2f}, fwd_ms={s['forward_ms']:.2f}\n"
            )

        f.write("\n# --- Run Configuration ---\n")
        f.write(f"# mode_kind: {'adapter' if use_adapter_mode else 'direct'}\n")
        f.write(f"# tta_input_policy: {args.tta_input_policy}\n")
        f.write(f"# tta_mode: {args.tta_mode}\n")
        f.write(f"# inner_lr: {args.inner_lr}\n")
        f.write(f"# inner_steps: {args.inner_steps}\n")
        f.write(f"# inner_momentum: {args.inner_momentum}\n")
        f.write(f"# lam: {args.lam}\n")
        f.write(f"# clip_grad: {args.clip_grad}\n")
        f.write(f"# w_clean_rel: {args.w_clean_rel}\n")
        f.write(f"# w_dist_rel: {args.w_dist_rel}\n")
        f.write(f"# w_znorm_rel: {args.w_znorm_rel}\n")
        f.write(f"# w_trust: {args.w_trust}\n")
        f.write(f"# w_kl: {args.w_kl}\n")
        f.write(f"# disable_tta_guard: {args.disable_tta_guard}\n")
        f.write(f"# disable_clean_skip: {args.disable_clean_skip}\n")
        f.write(f"# carry_scope: {args.carry_scope}\n")
        f.write(f"# carry_decay: {args.carry_decay}\n")
        f.write(f"# carry_reset_every: {args.carry_reset_every}\n")
    return out


def run_smoke_test(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    model = create_model(args.model, pretrained=False, num_classes=args.num_classes).to(device).eval()
    for p in model.parameters():
        p.requires_grad = False

    spec = resolve_stem_target(model, target=args.stem_target)
    stem_channels = spec.module.out_channels

    # Fake phase1 classifier for smoke only.
    if "vit" in args.model.lower():
        from train_phase1_vit import DirectAugClassifier
    else:
        from train_phase1_resnet import DirectAugClassifier
    aug_classifier = DirectAugClassifier(stem_channels=stem_channels, output_channels=64, num_dw_stages=5).to(device).eval()

    x = torch.randn(args.batch_size, 3, args.img_size, args.img_size, device=device)
    y = torch.randint(0, args.num_classes, (args.batch_size,), device=device)

    inner_cfg = InnerLossConfig()
    guard_cfg = StepGuardConfig()

    if args.adapter_ckpt:
        adapter = build_stem_adapter(
            model=model,
            target=args.stem_target,
            num_bases=args.num_bases,
            rank=args.adapter_rank,
            scale=args.adapter_scale,
            init_std=args.adapter_init_std,
            train_affine=args.train_stem_affine,
        ).to(device)
        c0 = torch.zeros(args.num_bases, device=device)
        c1, diag = _inner_adapt_coeff_with_guard(
            model, adapter, aug_classifier, x, c0,
            inner_lr=args.inner_lr,
            inner_steps=2,
            inner_cfg=inner_cfg,
            guard_cfg=guard_cfg,
        )
        with torch.no_grad():
            logits = adapter.forward_logits(model, x, c1)
    else:
        _diag = _inner_adapt_direct_with_guard(
            model=model,
            target_name=spec.weight_name,
            aug_classifier=aug_classifier,
            images=x,
            inner_lr=args.inner_lr,
            inner_steps=2,
            inner_momentum=0.0,
            lam=0.0,
            clip_grad=0.0,
            inner_cfg=inner_cfg,
            guard_cfg=guard_cfg,
            max_param_delta=1.0,
        )
        with torch.no_grad():
            logits = model(x)
        diag = {
            "after_p_clean": _diag["after_p_clean"],
            "after_dist": _diag["after_dist"],
            "after_znorm": _diag["after_znorm"],
        }

    t1, t5 = eval_logits(logits, y)
    print(f"[SMOKE] ok | logits={tuple(logits.shape)} top1={t1:.2f} top5={t5:.2f} p={diag['after_p_clean']:.4f}")


def main():
    args = parse_args()
    setup_default_logging()

    if args.smoke_test:
        run_smoke_test(args)
        return

    if args.tta_input_policy != "raw":
        raise ValueError("This validator supports only --tta-input-policy raw.")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.device = device

    model = create_model(args.model, pretrained=args.pretrained, num_classes=args.num_classes).to(device).eval()
    if args.initial_checkpoint:
        ckpt = torch.load(args.initial_checkpoint, map_location="cpu")
        state = ckpt.get("state_dict", ckpt.get("model", ckpt)) if isinstance(ckpt, dict) else ckpt
        state = { (k[7:] if k.startswith("module.") else k): v for k,v in state.items() }
        model.load_state_dict(state, strict=False)
    for p in model.parameters():
        p.requires_grad = False

    use_adapter_mode = bool(args.adapter_ckpt)

    if use_adapter_mode:
        adapter_ckpt = torch.load(args.adapter_ckpt, map_location="cpu")
        a_cfg = adapter_ckpt.get("adapter_config", {})
        target = a_cfg.get("target", args.stem_target)
        num_bases = int(a_cfg.get("num_bases", args.num_bases))
        rank = int(a_cfg.get("rank", args.adapter_rank))
        scale = float(a_cfg.get("scale", args.adapter_scale))
        init_std = float(a_cfg.get("init_std", args.adapter_init_std))
        train_affine = bool(a_cfg.get("train_affine", args.train_stem_affine))

        args.num_bases = num_bases
        args.adapter_rank = rank

        adapter = build_stem_adapter(
            model=model,
            target=target,
            num_bases=num_bases,
            rank=rank,
            scale=scale,
            init_std=init_std,
            train_affine=train_affine,
        ).to(device)
        adapter.load_state_dict(adapter_ckpt["adapter"], strict=True)
        adapter.eval()
        stem_channels = adapter.spec.module.out_channels
        target_name = adapter.spec.weight_name
    else:
        adapter = None
        spec = resolve_stem_target(model, target=args.stem_target)
        stem_channels = spec.module.out_channels
        target_name = spec.weight_name

    aug_classifier = _load_phase1_aug_classifier(
        model_name=args.model,
        aug_ckpt_path=args.aug_ckpt,
        stem_channels=stem_channels,
        device=device,
    )

    num_aug_classes = int(getattr(aug_classifier, "num_classes", aug_classifier.aug_head.out_features))
    clean_dist_ref = _build_clean_dist_ref(args.clean_dist_ref, num_classes=num_aug_classes, device=device)

    filename_label_map, class_to_idx = _build_label_map(Path(args.imagenet_val_dir))
    data_cfg = resolve_data_config(vars(args), model=model)
    transform = create_transform(**data_cfg, is_training=False)

    corruptions = args.corruptions or DEFAULT_CORRUPTIONS
    results: List[Dict] = []
    summaries: Dict[str, Dict[str, float]] = {}

    _logger.info(
        "Phase2 TTA (%s): mode=%s, inner_lr=%.4f, steps=%d",
        "adapter" if use_adapter_mode else "direct",
        args.tta_mode, args.inner_lr, args.inner_steps,
    )

    for i, corr in enumerate(corruptions):
        corr_dir = Path(args.imagenet_c_dir) / corr / str(args.severity)
        if not corr_dir.exists():
            corr_dir = Path(args.imagenet_c_dir) / corr
        if not corr_dir.exists():
            _logger.warning("Skip missing corruption: %s", corr)
            continue

        _logger.info("[%d/%d] %s", i + 1, len(corruptions), corr)

        ds = ImageNetCDataset(corr_dir, filename_label_map, class_to_idx, transform)
        loader = DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=True,
        )

        pre1, pre5, ada1, ada5, summary = _evaluate_corruption(
            model=model,
            adapter=adapter,
            use_adapter_mode=use_adapter_mode,
            target_name=target_name,
            loader=loader,
            aug_classifier=aug_classifier,
            args=args,
            clean_dist_ref=clean_dist_ref,
        )

        results.append({
            "corruption": corr,
            "pre_top1": pre1,
            "pre_top5": pre5,
            "adapted_top1": ada1,
            "adapted_top5": ada5,
        })
        summaries[corr] = summary
        _logger.info("  %s: pre=%.3f -> ada=%.3f (delta=%+.3f)", corr, pre1, ada1, ada1 - pre1)

    out = _save_results(results, summaries, args, use_adapter_mode)
    _logger.info("Results saved to: %s", out.resolve())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Validate ImageNet-C with Phase-2 coupled TTA (phase1 aligned)")

    g = parser.add_argument_group("Data")
    g.add_argument("--imagenet-c-dir", type=str, default="")
    g.add_argument("--imagenet-val-dir", type=str, default="")
    g.add_argument("--corruptions", nargs="+", default=None, type=str)
    g.add_argument("--severity", type=int, default=5)
    g.add_argument("--batch-size", type=int, default=64)
    g.add_argument("--workers", type=int, default=4)
    g.add_argument("--img-size", type=int, default=224)

    g = parser.add_argument_group("Model/Checkpoint")
    g.add_argument("--model", type=str, default="resnet50")
    g.add_argument("--num-classes", type=int, default=1000)
    g.add_argument("--pretrained", action="store_true", default=False)
    g.add_argument("--initial-checkpoint", type=str, default="")
    g.add_argument("--aug-ckpt", type=str, default="", help="Phase-1 aug classifier checkpoint (required).")
    g.add_argument("--adapter-ckpt", type=str, default="", help="Optional phase-2 adapter checkpoint.")
    g.add_argument("--stem-target", type=str, default="auto", choices=["auto", "conv1", "patch_embed.proj"])
    g.add_argument("--num-bases", type=int, default=4)
    g.add_argument("--adapter-rank", type=int, default=4)
    g.add_argument("--adapter-scale", type=float, default=1.0)
    g.add_argument("--adapter-init-std", type=float, default=1e-3)
    g.add_argument("--train-stem-affine", action="store_true", default=False)

    g = parser.add_argument_group("TTA")
    g.add_argument("--tta-input-policy", type=str, default="raw", choices=["raw"])
    g.add_argument("--tta-mode", type=str, default="static", choices=["static", "continue", "continual"])
    g.add_argument("--inner-lr", type=float, default=0.01)
    g.add_argument("--inner-steps", type=int, default=1)
    g.add_argument("--inner-momentum", type=float, default=0.0)
    g.add_argument("--lam", type=float, default=0.0, help="Trust region for direct mode.")
    g.add_argument("--clip-grad", type=float, default=0.0)

    g.add_argument("--w-clean-rel", type=float, default=1.0)
    g.add_argument("--w-dist-rel", type=float, default=1.0)
    g.add_argument("--w-znorm-rel", type=float, default=1.0)
    g.add_argument("--w-trust", type=float, default=1e-3)
    g.add_argument("--w-kl", type=float, default=0.0)
    g.add_argument("--clean-dist-ref", type=float, nargs="+", default=None)

    g.add_argument("--p-clean-ref", type=float, default=0.55)
    g.add_argument("--dist-ref", type=float, default=2.0)
    g.add_argument("--znorm-ref", type=float, default=2.0)
    g.add_argument("--rel-mode", type=str, default="relu", choices=["relu", "mse"])
    g.add_argument("--disable-rel-norm", action="store_true", default=False)

    g.add_argument("--disable-tta-guard", action="store_true", default=False)
    g.add_argument("--disable-clean-skip", action="store_true", default=False)
    g.add_argument("--skip-tol", type=float, default=0.05)
    g.add_argument("--max-dist-rise", type=float, default=0.10)
    g.add_argument("--max-znorm-rise", type=float, default=0.10)
    g.add_argument("--max-coeff-norm", type=float, default=1.0)
    g.add_argument("--max-param-delta", type=float, default=1.0)
    g.add_argument("--score-dist-penalty", type=float, default=0.5)
    g.add_argument("--score-znorm-penalty", type=float, default=0.5)
    g.add_argument("--min-score-improve", type=float, default=1e-4)

    g.add_argument("--carry-scope", type=str, default="within_domain", choices=["none", "within_episode", "within_domain"])
    g.add_argument("--carry-decay", type=float, default=1.0)
    g.add_argument("--carry-reset-every", type=int, default=0)

    g = parser.add_argument_group("Misc")
    g.add_argument("--results-dir", type=str, default="./results_phase2_coupled")
    g.add_argument("--log-interval", type=int, default=20)
    g.add_argument("--no-cuda", action="store_true", default=False)
    g.add_argument("--smoke-test", action="store_true", default=False)

    args = parser.parse_args()
    if args.tta_mode == "continue":
        args.tta_mode = "continual"

    if not args.smoke_test:
        if not args.imagenet_c_dir:
            parser.error("--imagenet-c-dir is required unless --smoke-test is set.")
        if not args.imagenet_val_dir:
            parser.error("--imagenet-val-dir is required unless --smoke-test is set.")
        if not args.aug_ckpt:
            parser.error("--aug-ckpt is required unless --smoke-test is set.")
    return args


if __name__ == "__main__":
    main()

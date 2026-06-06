#!/usr/bin/env python3
"""Mask Refiner MLP Training for Target Domain Adaptation.

Loads a frozen ViT (with parallel attention) and trains a lightweight MLP
that predicts per-class alpha/beta vectors to refine the spatial attention
mask.  Each mini-batch goes through two ViT forwards:

  1st forward (no grad) → extract mask, channel attention, logits
  MLP(features)         → alpha, beta
  Build new_mask        = alpha * original_scale + beta  (per spatial class)
  2nd forward (grad through mask) → new logits
  Loss                  = entropy(new logits)  or  cross_entropy(new logits, targets)

Training objective (--loss-mode):
  entropy : minimise prediction entropy (unsupervised, no labels needed)
  ce      : minimise cross-entropy with ground-truth labels (supervised upper bound)

Target domain data (e.g. ImageNet-C) is loaded directly via --data-dir.
"""
import argparse
import glob
import logging
import math
import os
import time
from collections import OrderedDict
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from timm import utils
from timm.data import create_dataset, create_loader, resolve_data_config
from timm.models import create_model, load_checkpoint

_logger = logging.getLogger("train_mask_refiner")

K = 10   # num_class_of_prop in SpatialAttention
TOP_K = 30
NUM_CLASSES = 1000


# ──────────────────────────────────────────────────────────────────────
# Mask Refiner MLP
# ──────────────────────────────────────────────────────────────────────

class MaskRefinerMLP(nn.Module):
    """Predicts alpha (K) and beta (K) vectors to refine spatial-attention mask.

    Input  (1024-d): mask_flat(196) | channel_attn(768) | top30_logits(30) | rank_embed(30)
    Output (2K=20) : alpha(10) centred at 1, beta(10) centred at 0
    """

    def __init__(self, input_dim: int = 196 + 768 + TOP_K + TOP_K,
                 hidden_dim: int = 256, output_dim: int = 2 * K, top_k: int = TOP_K):
        super().__init__()
        self.top_k = top_k
        self.rank_embed = nn.Embedding(top_k, 1)

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, output_dim),
            nn.Tanh(),
        )
        self._init_near_identity()

    def _init_near_identity(self):
        """Small (non-zero) last layer → initial output ≈ 0 → alpha≈1, beta≈0."""
        with torch.no_grad():
            last = self.net[-2]
            nn.init.normal_(last.weight, std=1e-4)
            nn.init.zeros_(last.bias)
            nn.init.zeros_(self.rank_embed.weight)

    def forward(self, mask_flat: torch.Tensor, channel_attn: torch.Tensor,
                top_logits: torch.Tensor, top_indices: torch.Tensor,
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        rank_emb = self.rank_embed(top_indices).squeeze(-1)      # (B, TOP_K)
        x = torch.cat([mask_flat, channel_attn, top_logits, rank_emb], dim=1)
        out = self.net(x)                                         # (B, 2K)
        alpha_raw, beta_raw = out.chunk(2, dim=1)
        return 1.0 + alpha_raw, beta_raw * 0.1


# ──────────────────────────────────────────────────────────────────────
# Mask helpers
# ──────────────────────────────────────────────────────────────────────

def build_original_scales(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.arange(K, 0, -1, device=device, dtype=dtype) / K   # [1.0, 0.9, …, 0.1]


def apply_alpha_beta_to_mask(
    mask: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor, spatial_attn,
) -> torch.Tensor:
    """new_scale_k = alpha_k * original_scale_k + beta_k  →  (B,1,H,W) mask."""
    logits = spatial_attn._logits                                 # (B, K, H, W)
    class_idx = F.softmax(logits, dim=1).argmax(dim=1)            # (B, H, W)

    new_scales = alpha * build_original_scales(mask.device, mask.dtype).unsqueeze(0) + beta
    B, H, W = class_idx.shape
    flat_idx = class_idx.reshape(B, -1)
    return torch.gather(new_scales, 1, flat_idx).reshape(B, 1, H, W)


# ──────────────────────────────────────────────────────────────────────
# SpatialAttention monkey-patch (mask injection for 2nd forward)
# ──────────────────────────────────────────────────────────────────────

_original_spatial_forward = None


def _patched_spatial_forward(self, x):
    """Return injected mask instead of computing one, when available."""
    injected = getattr(self, "_injected_mask", None)
    if injected is not None:
        x_attn = self.relu_reduction(self.norm_reduction(self.reduction_conv(x)))
        self._logits = self.norm(self.conv(x_attn))
        self._injected_mask = None
        return x, injected
    return _original_spatial_forward(self, x)


def patch_spatial_attn_forward(spatial_attn):
    global _original_spatial_forward
    if _original_spatial_forward is None:
        _original_spatial_forward = type(spatial_attn).forward
    import types
    spatial_attn.forward = types.MethodType(_patched_spatial_forward, spatial_attn)


def inject_mask(spatial_attn, new_mask: torch.Tensor, detach: bool = True):
    """Stage *new_mask* so the next SpatialAttention.forward returns it."""
    spatial_attn._last_mask = new_mask.detach()
    spatial_attn._injected_mask = new_mask if not detach else new_mask.detach()


# ──────────────────────────────────────────────────────────────────────
# Feature extraction (1st forward, no grad)
# ──────────────────────────────────────────────────────────────────────

def _unwrap(model):
    m = model
    if hasattr(m, "module"):
        m = m.module
    if hasattr(m, "_orig_mod"):
        m = m._orig_mod
    return m


@torch.no_grad()
def extract_features(model, images: torch.Tensor, device: torch.device) -> Dict[str, torch.Tensor]:
    """1st forward: run frozen ViT and collect mask / channel-attn / logits."""
    base = _unwrap(model)

    logits = model(images)
    if isinstance(logits, (tuple, list)):
        logits = logits[0]

    masks = base.get_last_masks()
    mask = masks[0] if masks else torch.ones(images.shape[0], 1, 14, 14, device=device)

    x_4d = base.patch_embed.proj(images)
    ch_attn = base.channel_attn(x_4d).squeeze(-1).squeeze(-1)    # (B, C)

    return {"mask": mask, "channel_attn": ch_attn, "logits": logits}


def prepare_mlp_inputs(feats: Dict[str, torch.Tensor], top_k: int = TOP_K):
    mask_flat = feats["mask"].flatten(1)
    top_vals, _ = feats["logits"].topk(top_k, dim=1)
    rank_pos = torch.arange(top_k, device=mask_flat.device).unsqueeze(0).expand(mask_flat.shape[0], -1)
    return mask_flat, feats["channel_attn"], top_vals, rank_pos


# ──────────────────────────────────────────────────────────────────────
# Loss
# ──────────────────────────────────────────────────────────────────────

def entropy_loss(logits: torch.Tensor) -> torch.Tensor:
    p = F.softmax(logits, dim=1)
    return -(p * F.log_softmax(logits, dim=1)).sum(dim=1).mean()


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train Mask Refiner MLP (two-pass, frozen ViT)")

    g = p.add_argument_group("Data")
    g.add_argument("--data-dir", required=True,
                   help="Root of target-domain dataset (e.g. ImageNet-C corruption dir)")
    g.add_argument("--val-split", default="",
                   help="Subfolder split name (default: empty = use data-dir directly)")
    g.add_argument("--workers", type=int, default=4)
    g.add_argument("--batch-size", type=int, default=32)

    g = p.add_argument_group("Frozen ViT")
    g.add_argument("--checkpoint", required=True)
    g.add_argument("--model", default="vit_base_patch16_224")
    g.add_argument("--num-classes", type=int, default=1000)
    g.add_argument("--vit-kernel-size", type=int, default=2)
    g.add_argument("--spatial-group-size", type=int, default=1)

    g = p.add_argument_group("MLP")
    g.add_argument("--hidden-dim", type=int, default=256)
    g.add_argument("--finetune-attn", action="store_true", default=False,
                   help="Also finetune SpatialAttention last conv/norm and "
                        "ChannelAttention together with the MLP")

    g = p.add_argument_group("Loss")
    g.add_argument("--loss-mode", default="entropy", choices=["entropy", "ce"],
                   help="entropy: unsupervised entropy minimization; "
                        "ce: supervised cross-entropy with labels (default: entropy)")

    g = p.add_argument_group("Training")
    g.add_argument("--epochs", type=int, default=10)
    g.add_argument("--lr", type=float, default=1e-3)
    g.add_argument("--weight-decay", type=float, default=1e-4)
    g.add_argument("--warmup-epochs", type=int, default=1)
    g.add_argument("--min-lr", type=float, default=1e-6)
    g.add_argument("--clip-grad", type=float, default=1.0)
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--log-interval", type=int, default=50)

    g = p.add_argument_group("Output")
    g.add_argument("--output", default="./VIT_IMG_PAR")
    g.add_argument("--experiment", default="mask_refiner")

    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────
# Training & Validation
# ──────────────────────────────────────────────────────────────────────

def compute_loss(new_logits, targets, mode):
    if mode == "ce":
        return F.cross_entropy(new_logits, targets)
    return entropy_loss(new_logits)


def _set_attn_train(base_model, training: bool):
    """Toggle train/eval for the finetuned attention sub-layers only."""
    for layer in [base_model.spatial_attn.conv, base_model.spatial_attn.norm,
                  base_model.channel_attn]:
        layer.train(training)


def train_one_epoch(epoch, model, mlp, loader, optimizer, args, device, base_model):
    mlp.train()
    if args.finetune_attn:
        _set_attn_train(base_model, True)

    all_params = list(mlp.parameters())
    if args.finetune_attn:
        all_params += [p for p in base_model.spatial_attn.conv.parameters()]
        all_params += [p for p in base_model.spatial_attn.norm.parameters()]
        all_params += [p for p in base_model.channel_attn.parameters()]

    loss_m   = utils.AverageMeter()
    ent_m    = utils.AverageMeter()
    ent2_m   = utils.AverageMeter()
    batch_m  = utils.AverageMeter()
    data_m   = utils.AverageMeter()

    end = time.time()
    for batch_idx, (images, targets) in enumerate(loader):
        images  = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        data_m.update(time.time() - end)

        # ── 1st forward (no grad): extract features ──────────────
        feats = extract_features(model, images, device)
        mask_flat, ch_attn, top_logits, rank_pos = prepare_mlp_inputs(feats)

        # ── MLP → alpha, beta → new mask ─────────────────────────
        alpha, beta = mlp(mask_flat, ch_attn, top_logits, rank_pos)
        new_mask = apply_alpha_beta_to_mask(feats["mask"], alpha, beta, base_model.spatial_attn)

        # ── 2nd forward ──────────────────────────────────────────
        if args.finetune_attn:
            new_logits = model(images)
        else:
            inject_mask(base_model.spatial_attn, new_mask, detach=False)
            new_logits = model(images)
        if isinstance(new_logits, (tuple, list)):
            new_logits = new_logits[0]

        loss = compute_loss(new_logits, targets, args.loss_mode)

        if args.finetune_attn:
            mask_mean = new_mask.mean(dim=[1, 2, 3]).clamp(min=1e-6)
            orig_mean = feats["mask"].mean(dim=[1, 2, 3]).clamp(min=1e-6)
            ratio = (mask_mean / orig_mean).unsqueeze(1)
            mlp_logits = feats["logits"] * ratio
            loss = loss + 0.1 * compute_loss(mlp_logits, targets, args.loss_mode)

        # ── Backward ─────────────────────────────────────────────
        optimizer.zero_grad()
        loss.backward()
        if args.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(all_params, args.clip_grad)
        optimizer.step()

        # ── Logging ──────────────────────────────────────────────
        bs = images.shape[0]
        loss_m.update(loss.item(), bs)
        with torch.no_grad():
            ent_m.update(entropy_loss(feats["logits"]).item(), bs)
            ent2_m.update(entropy_loss(new_logits).item(), bs)
        batch_m.update(time.time() - end)
        end = time.time()

        if batch_idx % args.log_interval == 0:
            _logger.info(
                f"Train [{epoch}][{batch_idx}/{len(loader)}]  "
                f"Loss: {loss_m.val:.4f}({loss_m.avg:.4f})  "
                f"H1: {ent_m.val:.4f}({ent_m.avg:.4f})  "
                f"H2: {ent2_m.val:.4f}({ent2_m.avg:.4f})  "
                f"LR: {optimizer.param_groups[0]['lr']:.2e}  "
                f"Time: {batch_m.val:.3f}s  Data: {data_m.val:.3f}s"
            )

    return {"loss": loss_m.avg, "ent_before": ent_m.avg, "ent_after": ent2_m.avg}


@torch.no_grad()
def validate(model, mlp, loader, args, device, base_model):
    mlp.eval()
    if args.finetune_attn:
        _set_attn_train(base_model, False)
    loss_m = utils.AverageMeter()
    ent1_m = utils.AverageMeter()
    ent2_m = utils.AverageMeter()
    top1_m = utils.AverageMeter()
    top5_m = utils.AverageMeter()
    top1_before_m = utils.AverageMeter()

    for images, targets in loader:
        images  = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        feats = extract_features(model, images, device)
        mask_flat, ch_attn, top_logits, rank_pos = prepare_mlp_inputs(feats)
        alpha, beta = mlp(mask_flat, ch_attn, top_logits, rank_pos)
        new_mask = apply_alpha_beta_to_mask(feats["mask"], alpha, beta, base_model.spatial_attn)

        if not args.finetune_attn:
            inject_mask(base_model.spatial_attn, new_mask, detach=True)
        logits = model(images)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]

        bs = images.shape[0]
        loss_m.update(compute_loss(logits, targets, args.loss_mode).item(), bs)
        ent1_m.update(entropy_loss(feats["logits"]).item(), bs)
        ent2_m.update(entropy_loss(logits).item(), bs)
        acc1_before, _ = utils.accuracy(feats["logits"], targets, topk=(1, 5))
        top1_before_m.update(acc1_before.item(), bs)
        acc1, acc5 = utils.accuracy(logits, targets, topk=(1, 5))
        top1_m.update(acc1.item(), bs)
        top5_m.update(acc5.item(), bs)

    return OrderedDict(loss=loss_m.avg, ent_before=ent1_m.avg, entropy=ent2_m.avg,
                       top1_before=top1_before_m.avg, top1=top1_m.avg, top5=top5_m.avg)


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    utils.setup_default_logging()
    args = parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _logger.info(f"Device: {device}")

    # ── Frozen ViT ───────────────────────────────────────────────
    model = create_model(
        args.model, pretrained=False, num_classes=args.num_classes,
        parallel_attention=True,
        sam_kernel_size=args.vit_kernel_size,
        spatial_group_size=args.spatial_group_size,
    )
    load_checkpoint(model, args.checkpoint)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False

    base_model = _unwrap(model)
    patch_spatial_attn_forward(base_model.spatial_attn)
    embed_dim = base_model.embed_dim

    input_dim = 14 * 14 + embed_dim + TOP_K + TOP_K
    _logger.info(f"MLP input_dim={input_dim}  (mask=196, ch_attn={embed_dim}, "
                 f"topk_logits={TOP_K}, rank_emb={TOP_K})")

    # ── MLP ──────────────────────────────────────────────────────
    mlp = MaskRefinerMLP(input_dim=input_dim, hidden_dim=args.hidden_dim,
                         output_dim=2 * K, top_k=TOP_K).to(device)
    _logger.info(f"MLP params: {sum(p.numel() for p in mlp.parameters()):,}")
    _logger.info(f"Loss mode: {args.loss_mode}")

    # ── Unfreeze attention layers if --finetune-attn ─────────────
    finetune_params = []
    if args.finetune_attn:
        sa = base_model.spatial_attn
        for layer in [sa.conv, sa.norm]:
            for p in layer.parameters():
                p.requires_grad = True
                finetune_params.append(p)
        ca = base_model.channel_attn
        for p in ca.parameters():
            p.requires_grad = True
            finetune_params.append(p)
        ca.relu.inplace = False
        ca.relu2.inplace = False
        sa.relu_reduction.inplace = False

        ft_count = sum(p.numel() for p in finetune_params)
        _logger.info(f"--finetune-attn: unfroze SpatialAttn(conv,norm) + ChannelAttn "
                     f"({ft_count:,} params)")

    # ── Optimizer / Scheduler ────────────────────────────────────
    param_groups = [{"params": mlp.parameters()}]
    if finetune_params:
        param_groups.append({"params": finetune_params, "lr": args.lr * 0.1})
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay)
    warmup = args.warmup_epochs

    def lr_lambda(epoch):
        if epoch < warmup:
            return (epoch + 1) / max(warmup, 1)
        progress = (epoch - warmup) / max(args.epochs - warmup, 1)
        return max(args.min_lr / args.lr, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Data ─────────────────────────────────────────────────────
    data_config = resolve_data_config(vars(args), model=model)

    ds_kwargs = dict(root=args.data_dir, is_training=False, batch_size=args.batch_size)
    if args.val_split:
        ds_kwargs["split"] = args.val_split

    dataset = create_dataset("", **ds_kwargs)
    loader = create_loader(
        dataset, input_size=data_config["input_size"],
        batch_size=args.batch_size, is_training=False,
        interpolation=data_config["interpolation"],
        crop_pct=data_config["crop_pct"],
        num_workers=args.workers,
        mean=data_config["mean"], std=data_config["std"],
        device=device, use_prefetcher=False,
    )
    _logger.info(f"Dataset: {len(dataset)} images, {len(loader)} batches (bs={args.batch_size})")

    # ── Output ───────────────────────────────────────────────────
    exp_dir = os.path.join(args.output, args.experiment)
    os.makedirs(exp_dir, exist_ok=True)
    with open(os.path.join(exp_dir, "args.yaml"), "w") as f:
        yaml.safe_dump(vars(args), f, default_flow_style=False)

    # ── Loop ─────────────────────────────────────────────────────
    max_ckpt = 3
    best_top1 = -1.0

    for epoch in range(args.epochs):
        train_metrics = train_one_epoch(
            epoch, model, mlp, loader, optimizer, args, device, base_model)
        scheduler.step()

        val = validate(model, mlp, loader, args, device, base_model)
        _logger.info(
            f"Epoch {epoch}  "
            f"train_loss={train_metrics['loss']:.4f}  "
            f"val_loss={val['loss']:.4f}  "
            f"H1={val['ent_before']:.4f} → H2={val['entropy']:.4f}  "
            f"Top1: {val['top1_before']:.2f} → {val['top1']:.2f}  "
            f"Top5={val['top5']:.2f}"
        )

        cur_top1 = val["top1"]
        is_best = cur_top1 > best_top1
        if is_best:
            best_top1 = cur_top1

        state = {
            "epoch": epoch,
            "mlp_state_dict": mlp.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_top1": best_top1,
            "val_metrics": dict(val),
            "args": vars(args),
        }
        if args.finetune_attn:
            state["spatial_attn_conv"] = base_model.spatial_attn.conv.state_dict()
            state["spatial_attn_norm"] = base_model.spatial_attn.norm.state_dict()
            state["channel_attn"] = base_model.channel_attn.state_dict()

        ckpt_path = os.path.join(exp_dir, f"checkpoint-{epoch}.pth.tar")
        torch.save(state, ckpt_path)
        if is_best:
            torch.save(state, os.path.join(exp_dir, "model_best.pth.tar"))
            _logger.info(f"  * New best Top1: {best_top1:.2f}")

        old_ckpts = sorted(glob.glob(os.path.join(exp_dir, "checkpoint-*.pth.tar")))
        while len(old_ckpts) > max_ckpt:
            os.remove(old_ckpts.pop(0))

    # ── Final evaluation with best checkpoint ──────────────────
    best_path = os.path.join(exp_dir, "model_best.pth.tar")
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device)
        mlp.load_state_dict(ckpt["mlp_state_dict"])
        if args.finetune_attn and "spatial_attn_conv" in ckpt:
            base_model.spatial_attn.conv.load_state_dict(ckpt["spatial_attn_conv"])
            base_model.spatial_attn.norm.load_state_dict(ckpt["spatial_attn_norm"])
            base_model.channel_attn.load_state_dict(ckpt["channel_attn"])
        _logger.info(f"Loaded best checkpoint (epoch {ckpt['epoch']}, top1={ckpt['best_top1']:.2f})")

    final = validate(model, mlp, loader, args, device, base_model)
    _logger.info(
        f"FINAL  val_loss={final['loss']:.4f}  "
        f"H1={final['ent_before']:.4f} → H2={final['entropy']:.4f}  "
        f"Top1: {final['top1_before']:.2f} → {final['top1']:.2f}  "
        f"Top5={final['top5']:.2f}"
    )


if __name__ == "__main__":
    main()

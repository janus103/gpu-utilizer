#!/usr/bin/env python3
"""Standalone validation for trained Mask Refiner checkpoints.

Loads a frozen ViT + trained MLP (and optionally finetuned attention layers),
then runs proper two-pass validation:
  1st forward (no grad) → extract mask, channel_attn, logits  (= "before")
  MLP(features)         → alpha, beta → new_mask
  inject new_mask into SpatialAttention
  2nd forward           → new logits                          (= "after")

This correctly injects the MLP-refined mask in ALL modes (including
--finetune-attn), fixing the validation bug in train_mask_refiner.py
where inject_mask was skipped under --finetune-attn.
"""
import argparse
import logging
import os
from collections import OrderedDict

import torch
import torch.nn.functional as F

from timm import utils
from timm.data import create_dataset, create_loader, resolve_data_config
from timm.models import create_model, load_checkpoint

from train_mask_refiner import (
    K, TOP_K,
    MaskRefinerMLP,
    apply_alpha_beta_to_mask,
    extract_features,
    prepare_mlp_inputs,
    inject_mask,
    patch_spatial_attn_forward,
    entropy_loss,
    _unwrap,
)

_logger = logging.getLogger("validate_mask_refiner")


def parse_args():
    p = argparse.ArgumentParser(description="Validate trained Mask Refiner checkpoint")

    g = p.add_argument_group("Data")
    g.add_argument("--data-dir", required=True)
    g.add_argument("--val-split", default="")
    g.add_argument("--workers", type=int, default=4)
    g.add_argument("--batch-size", type=int, default=32)

    g = p.add_argument_group("Refiner checkpoint")
    g.add_argument("--refiner-checkpoint", required=True,
                   help="Trained mask refiner checkpoint (model_best.pth.tar)")

    g = p.add_argument_group("Output")
    g.add_argument("--log-interval", type=int, default=50)

    return p.parse_args()


@torch.no_grad()
def validate(model, mlp, loader, device, base_model, has_finetuned_attn):
    mlp.eval()
    model.eval()

    loss_ce_before = utils.AverageMeter()
    loss_ce_after = utils.AverageMeter()
    ent_before_m = utils.AverageMeter()
    ent_after_m = utils.AverageMeter()
    top1_before_m = utils.AverageMeter()
    top5_before_m = utils.AverageMeter()
    top1_after_m = utils.AverageMeter()
    top5_after_m = utils.AverageMeter()

    n_batches = len(loader)
    for batch_idx, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        bs = images.shape[0]

        # 1st forward: original model → "before" metrics
        feats = extract_features(model, images, device)
        logits_before = feats["logits"]

        acc1_b, acc5_b = utils.accuracy(logits_before, targets, topk=(1, 5))
        top1_before_m.update(acc1_b.item(), bs)
        top5_before_m.update(acc5_b.item(), bs)
        ent_before_m.update(entropy_loss(logits_before).item(), bs)
        loss_ce_before.update(F.cross_entropy(logits_before, targets).item(), bs)

        # MLP → alpha, beta → new_mask
        mask_flat, ch_attn, top_logits, rank_pos = prepare_mlp_inputs(feats)
        alpha, beta = mlp(mask_flat, ch_attn, top_logits, rank_pos)
        new_mask = apply_alpha_beta_to_mask(
            feats["mask"], alpha, beta, base_model.spatial_attn)

        # 2nd forward: inject refined mask → "after" metrics
        inject_mask(base_model.spatial_attn, new_mask, detach=True)
        logits_after = model(images)
        if isinstance(logits_after, (tuple, list)):
            logits_after = logits_after[0]

        acc1_a, acc5_a = utils.accuracy(logits_after, targets, topk=(1, 5))
        top1_after_m.update(acc1_a.item(), bs)
        top5_after_m.update(acc5_a.item(), bs)
        ent_after_m.update(entropy_loss(logits_after).item(), bs)
        loss_ce_after.update(F.cross_entropy(logits_after, targets).item(), bs)

        if batch_idx % 50 == 0:
            _logger.info(
                f"[{batch_idx}/{n_batches}]  "
                f"Top1: {top1_before_m.avg:.2f} → {top1_after_m.avg:.2f}  "
                f"Ent: {ent_before_m.avg:.4f} → {ent_after_m.avg:.4f}")

    return OrderedDict(
        top1_before=top1_before_m.avg, top1_after=top1_after_m.avg,
        top5_before=top5_before_m.avg, top5_after=top5_after_m.avg,
        ce_before=loss_ce_before.avg, ce_after=loss_ce_after.avg,
        ent_before=ent_before_m.avg, ent_after=ent_after_m.avg,
    )


def main():
    utils.setup_default_logging()
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load refiner checkpoint to read training args ─────────
    ckpt = torch.load(args.refiner_checkpoint, map_location="cpu")
    ta = ckpt.get("args", {})
    has_finetuned_attn = ta.get("finetune_attn", False)
    vit_checkpoint = ta["checkpoint"]
    _logger.info(f"Refiner checkpoint: epoch {ckpt.get('epoch')}, "
                 f"best_top1={ckpt.get('best_top1')}, "
                 f"finetune_attn={has_finetuned_attn}")
    _logger.info(f"ViT checkpoint (from training args): {vit_checkpoint}")

    # ── Build frozen ViT ──────────────────────────────────────
    model = create_model(
        ta.get("model", "vit_base_patch16_224"),
        pretrained=False,
        num_classes=ta.get("num_classes", 1000),
        parallel_attention=True,
        sam_kernel_size=ta.get("vit_kernel_size", 2),
        spatial_group_size=ta.get("spatial_group_size", 1),
    )
    load_checkpoint(model, vit_checkpoint)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad = False

    base_model = _unwrap(model)
    patch_spatial_attn_forward(base_model.spatial_attn)
    embed_dim = base_model.embed_dim

    # ── Restore finetuned attention layers ────────────────────
    if has_finetuned_attn:
        if "spatial_attn_conv" in ckpt:
            base_model.spatial_attn.conv.load_state_dict(ckpt["spatial_attn_conv"])
            _logger.info("  Loaded finetuned spatial_attn.conv")
        if "spatial_attn_norm" in ckpt:
            base_model.spatial_attn.norm.load_state_dict(ckpt["spatial_attn_norm"])
            _logger.info("  Loaded finetuned spatial_attn.norm")
        if "channel_attn" in ckpt:
            base_model.channel_attn.load_state_dict(ckpt["channel_attn"])
            _logger.info("  Loaded finetuned channel_attn")

    # ── Build & load MLP ──────────────────────────────────────
    hidden_dim = ta.get("hidden_dim", 256)
    input_dim = 14 * 14 + embed_dim + TOP_K + TOP_K
    mlp = MaskRefinerMLP(input_dim=input_dim, hidden_dim=hidden_dim,
                         output_dim=2 * K, top_k=TOP_K).to(device)
    mlp.load_state_dict(ckpt["mlp_state_dict"])
    _logger.info(f"MLP loaded ({sum(p.numel() for p in mlp.parameters()):,} params)")

    # ── Data ──────────────────────────────────────────────────
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
    _logger.info(f"Dataset: {len(dataset)} images, {len(loader)} batches")

    # ── Validate ──────────────────────────────────────────────
    _logger.info("=" * 60)
    _logger.info("Validation with MLP mask injection (correct 2-pass)")
    _logger.info("=" * 60)

    results = validate(model, mlp, loader, device, base_model, has_finetuned_attn)

    _logger.info("=" * 60)
    _logger.info("RESULTS")
    _logger.info(f"  Top1 : {results['top1_before']:.2f} → {results['top1_after']:.2f}  "
                 f"(delta={results['top1_after'] - results['top1_before']:+.2f})")
    _logger.info(f"  Top5 : {results['top5_before']:.2f} → {results['top5_after']:.2f}  "
                 f"(delta={results['top5_after'] - results['top5_before']:+.2f})")
    _logger.info(f"  CE   : {results['ce_before']:.4f} → {results['ce_after']:.4f}")
    _logger.info(f"  Ent  : {results['ent_before']:.4f} → {results['ent_after']:.4f}")
    _logger.info("=" * 60)

    # also print with RESULT tag for easy grep
    print(f"RESULT  Top1: {results['top1_before']:.2f} → {results['top1_after']:.2f}  "
          f"Top5: {results['top5_before']:.2f} → {results['top5_after']:.2f}  "
          f"CE: {results['ce_before']:.4f} → {results['ce_after']:.4f}  "
          f"Ent: {results['ent_before']:.4f} → {results['ent_after']:.4f}")


if __name__ == "__main__":
    main()

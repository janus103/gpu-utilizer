#!/usr/bin/env python3
"""Evaluate a ViT checkpoint on a single dataset using EMA weights.

Uses timm's load_checkpoint (which prefers state_dict_ema over state_dict),
unlike resume_checkpoint which always loads the non-EMA state_dict.
"""
import argparse
import logging
from collections import OrderedDict

import torch
import torch.nn as nn

from timm import utils
from timm.data import create_dataset, create_loader, resolve_data_config
from timm.models import create_model, load_checkpoint

_logger = logging.getLogger("eval_vit_ema")


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate ViT checkpoint (EMA-aware)")

    p.add_argument("--data-dir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--model", default="vit_base_patch16_224")
    p.add_argument("--num-classes", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--log-interval", type=int, default=50)

    p.add_argument("--parallel-attention", action="store_true", default=False)
    p.add_argument("--vit-kernel-size", type=int, default=7)
    p.add_argument("--spatial-group-size", type=int, default=1)
    p.add_argument("--use-se-module", action="store_true", default=False)
    p.add_argument("--use-sam-module", type=int, default=-1)
    p.add_argument("--reverse-se", action="store_true", default=False)
    p.add_argument("--vit-last", action="store_true", default=False)
    p.add_argument("--vit-closed", type=str, default=None, choices=["same", "diff"])
    p.add_argument("--train-mode", type=int, default=0)
    p.add_argument("--no-ema", action="store_true", default=False,
                   help="Force loading non-EMA state_dict for comparison")

    return p.parse_args()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()

    top1_m = utils.AverageMeter()
    top5_m = utils.AverageMeter()
    loss_m = utils.AverageMeter()

    n_batches = len(loader)
    for batch_idx, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        output = model(images)
        if isinstance(output, (tuple, list)):
            output = output[0]

        loss = loss_fn(output, targets)
        acc1, acc5 = utils.accuracy(output, targets, topk=(1, 5))

        bs = images.shape[0]
        loss_m.update(loss.item(), bs)
        top1_m.update(acc1.item(), bs)
        top5_m.update(acc5.item(), bs)

        if batch_idx % 50 == 0:
            _logger.info(
                f"[{batch_idx}/{n_batches}]  "
                f"Top1: {top1_m.avg:.2f}  Top5: {top5_m.avg:.2f}  "
                f"Loss: {loss_m.avg:.4f}")

    return OrderedDict(top1=top1_m.avg, top5=top5_m.avg, loss=loss_m.avg)


def main():
    utils.setup_default_logging()
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vit_kwargs = {}
    if args.parallel_attention or args.use_sam_module != -1:
        vit_kwargs["sam_kernel_size"] = args.vit_kernel_size
        vit_kwargs["spatial_group_size"] = args.spatial_group_size
    if args.vit_last:
        vit_kwargs["vit_last"] = True
    if args.vit_closed is not None:
        vit_kwargs["vit_closed"] = args.vit_closed

    model = create_model(
        args.model,
        pretrained=False,
        num_classes=args.num_classes,
        use_se_module=args.use_se_module,
        use_sam_module=args.use_sam_module,
        reverse_se_sam=args.reverse_se,
        parallel_attention=args.parallel_attention,
        **vit_kwargs,
    )

    use_ema = not args.no_ema
    load_checkpoint(model, args.checkpoint, use_ema=use_ema)
    model.to(device).eval()

    n_params = sum(p.numel() for p in model.parameters())
    _logger.info(f"Model: {args.model}, params: {n_params:,}, EMA: {use_ema}")

    data_config = resolve_data_config(vars(args), model=model)
    dataset = create_dataset("", root=args.data_dir, is_training=False, batch_size=args.batch_size)
    loader = create_loader(
        dataset,
        input_size=data_config["input_size"],
        batch_size=args.batch_size,
        is_training=False,
        interpolation=data_config["interpolation"],
        crop_pct=data_config["crop_pct"],
        num_workers=args.workers,
        mean=data_config["mean"],
        std=data_config["std"],
        device=device,
        use_prefetcher=False,
    )
    _logger.info(f"Dataset: {len(dataset)} images, {len(loader)} batches")

    results = evaluate(model, loader, device)

    _logger.info(f"Top1={results['top1']:.2f}  Top5={results['top5']:.2f}  Loss={results['loss']:.4f}")
    print(f"RESULT  Top1={results['top1']:.2f}  Top5={results['top5']:.2f}  Loss={results['loss']:.4f}")


if __name__ == "__main__":
    main()

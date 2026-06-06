#!/usr/bin/env python3
"""Validate AugAttentionWrapper on ImageNet-C (15 corruptions × 5 severities).

Loads checkpoints produced by train_aug_classifier.py and evaluates
on all 15 ImageNet-C corruptions.

Usage:
    python validate_aug_classifier.py \
        --checkpoint ./output/aug_cls_vitb_pretrained/best.pth \
        --batch-size 256 --workers 8
"""
import argparse
import csv
import importlib
import logging
import os
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from timm import utils
from timm.data import create_dataset, create_loader, resolve_data_config
from timm.models import create_model, safe_model_name

from train_aug_classifier import (
    AugClassifier,
    ParallelAttentionConv,
    AugAttentionWrapper,
    NUM_AUG_TRANSFORMS,
)
from timm.data.auto_augment import _AUGMIX_SL_TRANSFORMS_V2

AUG_HEAD_NAMES = ['source'] + list(_AUGMIX_SL_TRANSFORMS_V2)

_logger = logging.getLogger('validate_aug_classifier')

CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise',
    'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
    'snow', 'frost', 'fog', 'brightness',
    'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
]


def create_corruption_loader(corruption, severity, args, data_config, model_dtype, device):
    data_dir = os.path.join(args.data_root, corruption, str(severity))
    input_img_mode = 'RGB' if data_config['input_size'][0] == 3 else 'L'

    dataset = create_dataset(
        '', root=data_dir, split=args.val_split, is_training=False,
        class_map='', download=False,
        batch_size=args.batch_size, input_img_mode=input_img_mode,
    )
    loader = create_loader(
        dataset, input_size=data_config['input_size'],
        batch_size=args.batch_size, is_training=False,
        interpolation=data_config['interpolation'],
        num_workers=args.workers, crop_pct=data_config['crop_pct'],
        mean=data_config['mean'], std=data_config['std'],
        pin_memory=args.pin_mem,
        img_dtype=model_dtype or torch.float32,
        device=device, distributed=False,
        use_prefetcher=not args.no_prefetcher,
    )
    return loader


@torch.no_grad()
def evaluate(wrapper, loader, device, model_dtype, prefetcher=True):
    wrapper.eval()
    top1_m = utils.AverageMeter()
    top5_m = utils.AverageMeter()
    all_aug_logits = []

    for images, target in loader:
        if not prefetcher:
            images = images.to(device=device, dtype=model_dtype)
            target = target.to(device=device)

        class_logits, aug_logits = wrapper(images)

        acc1, acc5 = utils.accuracy(class_logits, target, topk=(1, 5))
        top1_m.update(acc1.item(), images.shape[0])
        top5_m.update(acc5.item(), images.shape[0])
        all_aug_logits.append(aug_logits.cpu())

    all_aug_logits = torch.cat(all_aug_logits, dim=0)  # (N, 8)
    return top1_m.avg, top5_m.avg, all_aug_logits


def compute_aug_stats(aug_logits):
    """Compute per-channel statistics from aug classifier head output.

    Returns dict with mean, std, min, max, median for each of the 8 channels.
    """
    arr = aug_logits.numpy()  # (N, 8)
    stats = {}
    for i, name in enumerate(AUG_HEAD_NAMES):
        col = arr[:, i]
        stats[name] = {
            'mean': float(np.mean(col)),
            'std': float(np.std(col)),
            'min': float(np.min(col)),
            'max': float(np.max(col)),
            'median': float(np.median(col)),
        }
    return stats


def log_aug_stats(corruption, stats):
    """Pretty-print aug classifier head statistics for one corruption."""
    _logger.info(f'    Aug Head output distribution for [{corruption}]:')
    _logger.info(f'    {"channel":<26s} {"mean":>7s} {"std":>7s} '
                 f'{"min":>7s} {"median":>7s} {"max":>7s}')
    _logger.info(f'    {"-"*26} {"-"*7} {"-"*7} {"-"*7} {"-"*7} {"-"*7}')
    for name in AUG_HEAD_NAMES:
        s = stats[name]
        _logger.info(f'    {name:<26s} {s["mean"]:7.4f} {s["std"]:7.4f} '
                     f'{s["min"]:7.4f} {s["median"]:7.4f} {s["max"]:7.4f}')


def _parse_args():
    config_parser = argparse.ArgumentParser(description='Config', add_help=False)
    config_parser.add_argument('-c', '--config', default='', type=str, metavar='FILE')

    parser = argparse.ArgumentParser(
        description='Validate AugAttentionWrapper on ImageNet-C')

    group = parser.add_argument_group('Data')
    group.add_argument('--data-root', type=str, default='/home/oem/servers/imagenet-c')
    group.add_argument('--severity', type=int, nargs='+', default=[5],
                       help='Severity level(s) to evaluate (default: [5])')
    group.add_argument('--val-split', type=str, default='validation')
    group.add_argument('--corruption', type=str, nargs='+', default=None,
                       help='Corruption(s) to evaluate (default: all 15)')

    group = parser.add_argument_group('Model')
    group.add_argument('--model', default='vit_base_patch16_224', type=str)
    group.add_argument('--checkpoint', type=str, required=True,
                       help='Path to best.pth from train_aug_classifier.py')
    group.add_argument('--num-classes', type=int, default=1000)
    group.add_argument('--input-size', default=[3, 224, 224], nargs=3, type=int)
    group.add_argument('--scale', type=float, default=1.0,
                       help='Must match the scale used during training')

    group = parser.add_argument_group('Runtime')
    group.add_argument('-b', '--batch-size', type=int, default=256)
    group.add_argument('-j', '--workers', type=int, default=4)
    group.add_argument('--device', default='cuda', type=str)
    group.add_argument('--model-dtype', default=None, type=str)
    group.add_argument('--pin-mem', action='store_true', default=False)
    group.add_argument('--no-prefetcher', action='store_true', default=False)
    group.add_argument('--seed', type=int, default=42)
    group.add_argument('--device-modules', default=None, type=str, nargs='+')

    group = parser.add_argument_group('Output')
    group.add_argument('--output-dir', type=str, default=None,
                       help='Save CSV results here (default: same dir as checkpoint)')

    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    args = parser.parse_args(remaining)
    return args


def main():
    utils.setup_default_logging()
    args = _parse_args()

    corruptions = args.corruption if args.corruption else CORRUPTIONS
    for c in corruptions:
        if c not in CORRUPTIONS:
            raise ValueError(f'Unknown corruption: {c}. Choose from {CORRUPTIONS}')

    if args.device_modules:
        for module in args.device_modules:
            importlib.import_module(module)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    device = torch.device(args.device)
    utils.random_seed(args.seed, 0)

    model_dtype = None
    if args.model_dtype:
        model_dtype = getattr(torch, args.model_dtype)

    # ── Load checkpoint ──
    _logger.info(f'Loading checkpoint: {args.checkpoint}')
    ckpt = torch.load(args.checkpoint, map_location='cpu')

    saved_args = ckpt.get('args', {})
    scale = saved_args.get('scale', args.scale)
    model_name = saved_args.get('model', args.model)
    num_classes = saved_args.get('num_classes', args.num_classes)
    _logger.info(f'  Checkpoint info: model={model_name}, scale={scale}')
    if 'val_acc1' in ckpt:
        _logger.info(f'  Saved val_acc1: {ckpt["val_acc1"]:.3f}%')
    if 'epoch' in ckpt:
        _logger.info(f'  Saved epoch: {ckpt["epoch"]}')

    # ── Create backbone ──
    backbone = create_model(
        model_name,
        pretrained=False,
        num_classes=num_classes,
        in_chans=args.input_size[0],
    )
    backbone.to(device=device, dtype=model_dtype)

    _logger.info(f'Loading backbone state_dict...')
    backbone_sd = ckpt['state_dict']
    missing, unexpected = backbone.load_state_dict(backbone_sd, strict=False)
    if missing:
        _logger.info(f'  Missing keys ({len(missing)}): {missing[:5]}...')
    if unexpected:
        _logger.info(f'  Unexpected keys ({len(unexpected)}): {unexpected[:5]}...')

    embed_dim = backbone.embed_dim

    # ── Create AugClassifier + ParallelAttention ──
    aug_classifier = AugClassifier(
        in_chans=args.input_size[0],
        num_classes=NUM_AUG_TRANSFORMS + 1,
        scale=scale,
    ).to(device=device, dtype=model_dtype)

    parallel_attn = ParallelAttentionConv(
        in_ch=256,
        embed_dim=embed_dim,
    ).to(device=device, dtype=model_dtype)

    _logger.info(f'Loading aug_classifier state_dict...')
    aug_classifier.load_state_dict(ckpt['aug_classifier'])
    _logger.info(f'Loading parallel_attn state_dict...')
    parallel_attn.load_state_dict(ckpt['parallel_attn'])

    # ── Assemble wrapper ──
    wrapper = AugAttentionWrapper(backbone, aug_classifier, parallel_attn)
    wrapper.to(device=device, dtype=model_dtype)
    wrapper.eval()

    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    if num_gpus > 1:
        wrapper = nn.DataParallel(wrapper)
        _logger.info(f'DataParallel enabled: {num_gpus} GPUs')

    total_params = sum(p.numel() for p in wrapper.parameters())
    _logger.info(f'Total parameters: {total_params:,}')

    data_config = resolve_data_config(
        vars(args) | {'pretrained': False}, model=backbone, verbose=True,
    )

    # ── Output setup ──
    output_dir = args.output_dir or os.path.dirname(args.checkpoint)
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, 'imagenetc_results.csv')

    # ── Evaluate ──
    _logger.info(f'\nEvaluating {len(corruptions)} corruption(s), '
                 f'severity={args.severity}')
    _logger.info('=' * 70)

    all_results = []
    all_aug_stats = []

    for severity in args.severity:
        _logger.info(f'\n--- Severity {severity} ---')
        acc1_sum = 0.0

        for corruption in corruptions:
            t0 = time.time()
            loader = create_corruption_loader(
                corruption, severity, args, data_config, model_dtype, device)
            acc1, acc5, aug_logits = evaluate(
                wrapper, loader, device, model_dtype,
                prefetcher=not args.no_prefetcher,
            )
            elapsed = time.time() - t0
            acc1_sum += acc1
            _logger.info(f'  {corruption:<22s}  Acc@1={acc1:.3f}%  '
                         f'Acc@5={acc5:.3f}%  ({elapsed:.1f}s)')

            stats = compute_aug_stats(aug_logits)
            log_aug_stats(corruption, stats)

            result_row = {
                'severity': severity,
                'corruption': corruption,
                'acc1': acc1,
                'acc5': acc5,
            }
            for name in AUG_HEAD_NAMES:
                result_row[f'{name}_mean'] = stats[name]['mean']
                result_row[f'{name}_std'] = stats[name]['std']
            all_results.append(result_row)

        if len(corruptions) > 1:
            mean_acc1 = acc1_sum / len(corruptions)
            _logger.info(f'\n  Mean Acc@1 (severity={severity}): {mean_acc1:.3f}%')
            mean_row = {
                'severity': severity,
                'corruption': 'MEAN',
                'acc1': mean_acc1,
                'acc5': 0.0,
            }
            for name in AUG_HEAD_NAMES:
                vals = [r[f'{name}_mean'] for r in all_results
                        if r['severity'] == severity and r['corruption'] != 'MEAN']
                mean_row[f'{name}_mean'] = sum(vals) / len(vals)
                std_vals = [r[f'{name}_std'] for r in all_results
                            if r['severity'] == severity and r['corruption'] != 'MEAN']
                mean_row[f'{name}_std'] = sum(std_vals) / len(std_vals)
            all_results.append(mean_row)

    # ── Write CSV ──
    aug_fields = []
    for name in AUG_HEAD_NAMES:
        aug_fields.extend([f'{name}_mean', f'{name}_std'])
    fieldnames = ['severity', 'corruption', 'acc1', 'acc5'] + aug_fields

    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_results:
            out = {
                'severity': row['severity'],
                'corruption': row['corruption'],
                'acc1': f'{row["acc1"]:.3f}',
                'acc5': f'{row["acc5"]:.3f}',
            }
            for name in AUG_HEAD_NAMES:
                out[f'{name}_mean'] = f'{row[f"{name}_mean"]:.4f}'
                out[f'{name}_std'] = f'{row[f"{name}_std"]:.4f}'
            writer.writerow(out)
    _logger.info(f'\nResults saved to: {csv_path}')
    _logger.info('=' * 70)


if __name__ == '__main__':
    main()

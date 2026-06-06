#!/usr/bin/env python3
"""TTA experiment: entropy minimization on SpatialAttention2 logits.

Loads a FedAvg checkpoint (e.g. round 20), performs N adaptation steps
using SA2 entropy minimization loss on each corruption's data, then evaluates.

Compares: no-TTA baseline, entropy-min TTA, domain-CE-to-0 TTA, IM-loss TTA.
"""
import argparse
import copy
import csv
import importlib
import logging
import os
import time
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from timm import utils
from timm.data import create_dataset, create_loader, resolve_data_config
from timm.models import create_model, safe_model_name
from timm.models.vision_transformer import SpatialAttention2

_logger = logging.getLogger('tta_vit_ent')

CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise',
    'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
    'snow', 'frost', 'fog', 'brightness',
    'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
]


def _get_base_model(model):
    m = model
    if hasattr(m, 'module'):
        m = m.module
    if hasattr(m, '_orig_mod'):
        m = m._orig_mod
    return m


def _is_auxiliary_param(name):
    n = name.lower()
    return any(kw in n for kw in (
        'se_module', 'sam_module', 'channel_attn', 'spatial_attn',
        'se_module_last', 'sam_module_last', 'channel_attn_last', 'spatial_attn_last',
    ))


def _is_embedding_param(name):
    return name.lower().startswith((
        'patch_embed', 'pos_embed', 'cls_token', 'reg_token',
        'norm_pre', 'conv1', 'bn1', 'act1',
    ))


def _apply_train_mode(model, train_mode):
    if train_mode == 0:
        for p in model.parameters():
            p.requires_grad = True
    elif train_mode == 1:
        for p in model.parameters():
            p.requires_grad = False
        for n, p in model.named_parameters():
            if _is_auxiliary_param(n) or _is_embedding_param(n):
                p.requires_grad = True
    elif train_mode == 2:
        for p in model.parameters():
            p.requires_grad = False
        for n, p in model.named_parameters():
            if _is_auxiliary_param(n):
                p.requires_grad = True
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_trainable_state(model):
    return OrderedDict(
        (n, p.data.clone()) for n, p in model.named_parameters() if p.requires_grad
    )


def set_trainable_state(model, state):
    with torch.no_grad():
        for n, p in model.named_parameters():
            if n in state:
                p.data.copy_(state[n])


def _make_loader(data_dir, split, args, data_config, model_dtype, device, is_training=False):
    img_mode = args.input_img_mode or 'RGB'
    ds = create_dataset(
        '', root=data_dir, split=split, is_training=is_training,
        class_map='', download=False,
        batch_size=args.batch_size, input_img_mode=img_mode,
    )
    if is_training:
        return create_loader(
            ds, input_size=data_config['input_size'],
            batch_size=args.batch_size, is_training=True, no_aug=True,
            num_workers=args.workers,
            mean=data_config['mean'], std=data_config['std'],
            pin_memory=args.pin_mem,
            img_dtype=model_dtype or torch.float32,
            device=device, distributed=False,
            use_prefetcher=not args.no_prefetcher,
        )
    return create_loader(
        ds, input_size=data_config['input_size'],
        batch_size=args.batch_size, is_training=False,
        interpolation=data_config['interpolation'],
        num_workers=args.workers, crop_pct=data_config['crop_pct'],
        mean=data_config['mean'], std=data_config['std'],
        pin_memory=args.pin_mem,
        img_dtype=model_dtype or torch.float32,
        device=device, distributed=False,
        use_prefetcher=not args.no_prefetcher,
    )


def compute_entropy_loss(logits):
    """Per-position entropy minimization: make each position more confident."""
    probs = F.softmax(logits, dim=1)
    log_probs = (probs + 1e-8).log()
    return -(probs * log_probs).sum(dim=1).mean()


def compute_domain_loss(logits, domain_label):
    B, N, H, W = logits.shape
    logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, N)
    labels = torch.full((logits_flat.shape[0],), domain_label,
                        device=logits.device, dtype=torch.long)
    return F.cross_entropy(logits_flat, labels)


def compute_im_loss(logits):
    """IM loss: minimize conditional entropy, maximize marginal entropy."""
    probs = F.softmax(logits, dim=1)
    log_probs = (probs + 1e-8).log()
    h_cond = -(probs * log_probs).sum(dim=1).mean()
    p_marg = probs.mean(dim=(0, 2, 3))
    h_marg = -(p_marg * (p_marg + 1e-8).log()).sum()
    return h_cond - h_marg


@torch.no_grad()
def evaluate(model, loader, device, model_dtype, prefetcher=True):
    model.eval()
    top1_m = utils.AverageMeter()
    for images, target in loader:
        if not prefetcher:
            images = images.to(device=device, dtype=model_dtype)
            target = target.to(device=device)
        output = model(images)
        if isinstance(output, (tuple, list)):
            output = output[0]
        acc1, = utils.accuracy(output, target, topk=(1,))
        top1_m.update(acc1.item(), images.shape[0])
    return top1_m.avg


def run_tta(model, base_model, train_loader, tta_loss_fn, args, device, model_dtype, lbatch):
    """Perform TTA: lbatch steps of adaptation using tta_loss_fn on SA2 logits."""
    model.train()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.tta_lr, weight_decay=0)
    prefetcher = not args.no_prefetcher

    for batch_idx, (images, _target) in enumerate(train_loader):
        if batch_idx >= lbatch:
            break
        if not prefetcher:
            images = images.to(device=device, dtype=model_dtype)

        tta_bs = args.tta_batch_size
        if tta_bs > 0 and tta_bs < images.shape[0]:
            images = images[:tta_bs]

        optimizer.zero_grad()
        _ = model(images)
        logits = base_model.spatial_attn._logits
        loss = tta_loss_fn(logits)
        loss.backward()
        optimizer.step()


def _parse_args():
    config_parser = argparse.ArgumentParser(description='Config', add_help=False)
    config_parser.add_argument('-c', '--config', default='', type=str)

    parser = argparse.ArgumentParser(description='TTA via SA2 entropy minimization')

    g = parser.add_argument_group('Data')
    g.add_argument('--data-root', type=str, default='/home/oem/servers/imagenet-c')
    g.add_argument('--severity', type=int, default=5)
    g.add_argument('--val-split', type=str, default='validation')
    g.add_argument('--input-img-mode', default=None, type=str)
    g.add_argument('--corruption', type=str, nargs='+', default=None,
                   help='Corruption(s) to evaluate (default: all 15)')

    g = parser.add_argument_group('Model')
    g.add_argument('--model', default='vit_base_patch16_224', type=str)
    g.add_argument('--resume', type=str,
                   default='./VIT_IMG_PAR/Normal_parallel_train_1_kernel_size_2/model_best.pth.tar')
    g.add_argument('--fedavg-ckpt', type=str, default=None)
    g.add_argument('--num-classes', type=int, default=1000)
    g.add_argument('--input-size', default=[3, 224, 224], nargs=3, type=int)
    g.add_argument('--parallel-attention', action='store_true', default=True)
    g.add_argument('--vit-kernel-size', type=int, default=2)
    g.add_argument('--spatial-group-size', type=int, default=1)
    g.add_argument('--vit-last', action='store_true', default=False)
    g.add_argument('--vit-closed', type=str, default=None, choices=['same', 'diff'])
    g.add_argument('--use-se-module', action='store_true', default=False)
    g.add_argument('--use-sam-module', type=int, default=-1)
    g.add_argument('--reverse-se', action='store_true', default=False)
    g.add_argument('--train-mode', type=int, default=1, choices=[0, 1, 2])
    g.add_argument('--sam-norm-type', type=int, default=0, choices=[0, 1, 2, 3, 4])
    g.add_argument('--vit-early-norm-types', type=int, nargs=4, default=None,
                   choices=[0, 1, 2, 3, 4])
    g.add_argument('--prop-size', type=int, default=5)

    g = parser.add_argument_group('TTA')
    g.add_argument('--lbatch', type=int, nargs='+', default=[0, 1, 5, 10, 20],
                   help='Number of adaptation batches to try (default: 0 1 5 10 20)')
    g.add_argument('--tta-lr', type=float, default=1e-3,
                   help='Learning rate for TTA adaptation (default: 1e-3)')
    g.add_argument('--tta-batch-size', type=int, default=64,
                   help='Batch size for TTA forward/backward (0 = same as --batch-size)')
    g.add_argument('--tta-loss', type=str, default='entropy',
                   choices=['entropy', 'domain0', 'im'],
                   help='TTA loss: entropy=entropy-min, domain0=CE-to-class0, im=IM-loss')

    g = parser.add_argument_group('Runtime')
    g.add_argument('-b', '--batch-size', type=int, default=256)
    g.add_argument('-j', '--workers', type=int, default=4)
    g.add_argument('--device', default='cuda', type=str)
    g.add_argument('--model-dtype', default=None, type=str)
    g.add_argument('--pin-mem', action='store_true', default=False)
    g.add_argument('--no-prefetcher', action='store_true', default=False)
    g.add_argument('--seed', type=int, default=42)
    g.add_argument('--device-modules', default=None, type=str, nargs='+')
    g.add_argument('--output-dir', type=str, default='./output/tta_entropy')

    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            parser.set_defaults(**yaml.safe_load(f))
    return parser.parse_args(remaining)


def main():
    utils.setup_default_logging()
    args = _parse_args()

    corruptions = args.corruption if args.corruption else CORRUPTIONS
    for c in corruptions:
        if c not in CORRUPTIONS:
            raise ValueError(f'Unknown corruption: {c}')

    if args.device_modules:
        for m in args.device_modules:
            importlib.import_module(m)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    device = torch.device(args.device)
    utils.random_seed(args.seed, 0)

    model_dtype = None
    if args.model_dtype:
        model_dtype = getattr(torch, args.model_dtype)

    vit_norm_kwargs = {}
    if args.use_sam_module != -1 or args.parallel_attention:
        vit_norm_kwargs['sam_kernel_size'] = args.vit_kernel_size
        vit_norm_kwargs['spatial_group_size'] = args.spatial_group_size
    if args.vit_last:
        vit_norm_kwargs['vit_last'] = True
    if args.vit_closed is not None:
        vit_norm_kwargs['vit_closed'] = args.vit_closed
    if args.vit_early_norm_types is not None:
        vit_norm_kwargs['vit_early_norm_types'] = args.vit_early_norm_types

    in_chans = args.input_size[0] if args.input_size else 3
    model = create_model(
        args.model, num_classes=args.num_classes, in_chans=in_chans,
        parallel_attention=args.parallel_attention,
        use_se_module=args.use_se_module, use_sam_module=args.use_sam_module,
        reverse_se_sam=args.reverse_se, sam_norm_type=args.sam_norm_type,
        **vit_norm_kwargs,
    )
    model.to(device=device, dtype=model_dtype)

    if args.resume:
        _logger.info(f'Loading base checkpoint: {args.resume}')
        ckpt = torch.load(args.resume, map_location='cpu')
        sd = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
        model.load_state_dict(sd, strict=False)

    if args.prop_size > 0:
        base = _get_base_model(model)
        ch = base.embed_dim
        base.spatial_attn = SpatialAttention2(
            kernel_size=args.vit_kernel_size, channels=ch, prop_size=args.prop_size,
        ).to(device=device, dtype=model_dtype)
        _logger.info(f'SpatialAttention2: prop_size={args.prop_size}, kernel={args.vit_kernel_size}')

    _apply_train_mode(model, args.train_mode)

    if args.fedavg_ckpt:
        _logger.info(f'Loading FedAvg checkpoint: {args.fedavg_ckpt}')
        ckpt = torch.load(args.fedavg_ckpt, map_location='cpu')
        fedavg_state = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
        set_trainable_state(model, fedavg_state)
        if 'round' in ckpt:
            _logger.info(f'  Round: {ckpt["round"]}, mean_acc1: {ckpt.get("mean_acc1", "N/A")}')

    base_model = _get_base_model(model)
    initial_state = get_trainable_state(model)

    data_config = resolve_data_config(
        vars(args) | {'pretrained': False}, model=model, verbose=True,
    )

    if args.tta_loss == 'entropy':
        tta_loss_fn = compute_entropy_loss
    elif args.tta_loss == 'domain0':
        tta_loss_fn = lambda logits: compute_domain_loss(logits, 0)
    elif args.tta_loss == 'im':
        tta_loss_fn = compute_im_loss

    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, 'tta_results.csv')
    prefetcher = not args.no_prefetcher

    _logger.info(f'\nTTA loss: {args.tta_loss}, lr: {args.tta_lr}')
    _logger.info(f'lbatch sweep: {args.lbatch}')
    _logger.info(f'Corruptions: {len(corruptions)}')
    _logger.info('=' * 80)

    all_results = []

    for corruption in corruptions:
        c_dir = os.path.join(args.data_root, corruption, str(args.severity))
        train_loader = _make_loader(
            c_dir, args.val_split, args, data_config, model_dtype, device, is_training=True,
        )
        eval_loader = _make_loader(
            c_dir, args.val_split, args, data_config, model_dtype, device, is_training=False,
        )

        row = {'corruption': corruption}

        for lb in args.lbatch:
            set_trainable_state(model, initial_state)

            if lb > 0:
                run_tta(model, base_model, train_loader, tta_loss_fn,
                        args, device, model_dtype, lb)

            acc1 = evaluate(model, eval_loader, device, model_dtype, prefetcher=prefetcher)
            row[f'lb{lb}'] = acc1
            _logger.info(f'  {corruption:<22s}  lbatch={lb:>3d}  Acc@1={acc1:.3f}%')

        all_results.append(row)
        _logger.info('')

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['corruption'] + [f'lb{lb}' for lb in args.lbatch]
        writer.writerow(header)
        for row in all_results:
            writer.writerow([row['corruption']] + [f'{row[f"lb{lb}"]:.3f}' for lb in args.lbatch])
        mean_row = ['MEAN']
        for lb in args.lbatch:
            m = sum(r[f'lb{lb}'] for r in all_results) / len(all_results)
            mean_row.append(f'{m:.3f}')
        writer.writerow(mean_row)

    _logger.info('=' * 80)
    _logger.info('SUMMARY')
    _logger.info('=' * 80)
    header = f'{"corruption":<22s}'
    for lb in args.lbatch:
        header += f'  {"lb"+str(lb):>8s}'
    _logger.info(header)
    _logger.info('-' * (22 + 10 * len(args.lbatch)))

    for row in all_results:
        line = f'{row["corruption"]:<22s}'
        for lb in args.lbatch:
            line += f'  {row[f"lb{lb}"]:8.3f}'
        _logger.info(line)

    line = f'{"MEAN":<22s}'
    for lb in args.lbatch:
        m = sum(r[f'lb{lb}'] for r in all_results) / len(all_results)
        line += f'  {m:8.3f}'
    _logger.info(line)
    _logger.info(f'\nCSV saved: {csv_path}')


if __name__ == '__main__':
    main()

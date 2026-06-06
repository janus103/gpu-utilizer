#!/usr/bin/env python3
"""Test-Time Adaptation for FedAvg-trained ResNet with Parallel Attention.

Loads a checkpoint from train_fedavg_resnet_policy.py, runs BN statistics
adaptation (lbatch forward passes in train mode to update BN running_mean/var),
then evaluates on ImageNet-C corruptions.

No SE/SAM Gaussian loss is used -- only BN tracking during the lbatch phase.
"""
import argparse
import csv
import importlib
import logging
import os
import time
from collections import OrderedDict
from contextlib import suppress
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from timm import utils
from timm.data import create_dataset, create_loader, resolve_data_config
from timm.layers import set_fast_norm
from timm.models import create_model, safe_model_name

from timm.models.vision_transformer import SpatialAttention2

_logger = logging.getLogger('tta_resnet')

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


def _is_auxiliary_param(param_name: str) -> bool:
    name = param_name.lower()
    aux_keywords = (
        'se_module', 'sam_module',
        'channel_attn', 'spatial_attn',
        'se_module_last', 'sam_module_last',
        'channel_attn_last', 'spatial_attn_last',
    )
    return any(kw in name for kw in aux_keywords)


def _is_embedding_param(param_name: str) -> bool:
    name = param_name.lower()
    return name.startswith(('conv1',))


def _apply_train_mode(model: nn.Module, train_mode: int) -> int:
    if train_mode == 0:
        for param in model.parameters():
            param.requires_grad = True
    elif train_mode == 1:
        for param in model.parameters():
            param.requires_grad = False
        for name, param in model.named_parameters():
            if _is_auxiliary_param(name) or _is_embedding_param(name):
                param.requires_grad = True
    elif train_mode == 2:
        for param in model.parameters():
            param.requires_grad = False
        for name, param in model.named_parameters():
            if _is_auxiliary_param(name):
                param.requires_grad = True
    else:
        raise ValueError(f'Unsupported train_mode: {train_mode}')

    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable_count


def set_trainable_state(model, state):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in state:
                param.data.copy_(state[name])


def create_client_loader(corruption, args, data_config, model_dtype, device):
    data_dir = os.path.join(args.data_root, corruption, str(args.severity))
    input_img_mode = args.input_img_mode or ('RGB' if data_config['input_size'][0] == 3 else 'L')

    dataset = create_dataset(
        '', root=data_dir, split=args.val_split, is_training=True,
        class_map=args.class_map, download=False,
        batch_size=args.batch_size, seed=args.seed,
        input_img_mode=input_img_mode,
    )
    loader = create_loader(
        dataset, input_size=data_config['input_size'],
        batch_size=args.batch_size, is_training=True, no_aug=True,
        num_workers=args.workers,
        mean=data_config['mean'], std=data_config['std'],
        pin_memory=args.pin_mem,
        img_dtype=model_dtype or torch.float32,
        device=device, distributed=False,
        use_prefetcher=not args.no_prefetcher,
    )
    return loader


def create_eval_loader(corruption, args, data_config, model_dtype, device):
    data_dir = os.path.join(args.data_root, corruption, str(args.severity))
    input_img_mode = args.input_img_mode or ('RGB' if data_config['input_size'][0] == 3 else 'L')

    dataset = create_dataset(
        '', root=data_dir, split=args.val_split, is_training=False,
        class_map=args.class_map, download=False,
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


def bn_adapt(model, loader, args, device, model_dtype, lbatch):
    """Run lbatch forward passes in train mode to update BN running statistics."""
    model.train()
    _set_bn_train_partial(model, args.bn_adapt_num)
    prefetcher = not args.no_prefetcher

    adapt_time = utils.AverageMeter()
    top1_m = utils.AverageMeter()
    end = time.time()

    with torch.no_grad():
        for batch_idx, (images, target) in enumerate(loader):
            if not prefetcher:
                images = images.to(device=device, dtype=model_dtype)
                target = target.to(device=device)

            output = model(images)
            if isinstance(output, (tuple, list)):
                output = output[0]

            acc1, = utils.accuracy(output.detach(), target, topk=(1,))
            top1_m.update(acc1.item(), images.shape[0])

            adapt_time.update(time.time() - end)
            end = time.time()

            if batch_idx % args.log_interval == 0:
                _logger.info(
                    f'  BN-Adapt [{batch_idx:>4d}/{lbatch}]  '
                    f'Acc@1={top1_m.val:.2f}%({top1_m.avg:.2f}%)  '
                    f'Time={adapt_time.val:.3f}s')

            if batch_idx + 1 >= lbatch:
                break

    _logger.info(f'  BN-Adapt done: {min(lbatch, batch_idx + 1)} batches, '
                 f'running Acc@1={top1_m.avg:.2f}%')
    return top1_m.avg


@torch.no_grad()
def evaluate(model, loader, args, device, model_dtype):
    model.eval()

    losses_m = utils.AverageMeter()
    top1_m = utils.AverageMeter()
    top5_m = utils.AverageMeter()
    prefetcher = not args.no_prefetcher

    for images, target in loader:
        if not prefetcher:
            images = images.to(device=device, dtype=model_dtype)
            target = target.to(device=device)

        output = model(images)
        if isinstance(output, (tuple, list)):
            output = output[0]
        loss = F.cross_entropy(output, target)

        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        losses_m.update(loss.item(), images.shape[0])
        top1_m.update(acc1.item(), images.shape[0])
        top5_m.update(acc5.item(), images.shape[0])

    return losses_m.avg, top1_m.avg, top5_m.avg


# ── Argument parsing ──

config_parser = parser = argparse.ArgumentParser(description='Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE')

parser = argparse.ArgumentParser(description='TTA — BN Adapt for FedAvg ResNet')

group = parser.add_argument_group('Data')
group.add_argument('--data-root', type=str, default='/home/oem/servers/imagenet-c')
group.add_argument('--severity', type=int, default=5)
group.add_argument('--val-split', type=str, default='validation')
group.add_argument('--class-map', default='', type=str)
group.add_argument('--input-img-mode', default=None, type=str)

group = parser.add_argument_group('Model')
group.add_argument('--model', default='resnet50', type=str)
group.add_argument('--resume-base', type=str, default=None,
                   help='Full pretrained backbone checkpoint to load first '
                        '(the base model that FedAvg was trained on top of)')
group.add_argument('--checkpoint', type=str, required=True,
                   help='Path to FedAvg checkpoint (.pth) with trainable params only')
group.add_argument('--num-classes', type=int, default=1000)
group.add_argument('--input-size', default=[3, 224, 224], nargs=3, type=int)
group.add_argument('--parallel-attention', action='store_true', default=True)
group.add_argument('--vit-kernel-size', type=int, default=3)
group.add_argument('--spatial-group-size', type=int, default=1)
group.add_argument('--sam-norm-type', type=int, default=0, choices=[0, 1, 2, 3, 4])
group.add_argument('--use-se-module', action='store_true', default=False)
group.add_argument('--use-sam-module', type=int, default=-1)
group.add_argument('--reverse-se', action='store_true', default=False)
group.add_argument('--train-mode', type=int, default=1, choices=[0, 1, 2])
group.add_argument('--prop-size', type=int, default=5)
group.add_argument('--detach', action='store_true', default=False)
group.add_argument('--residual', action='store_true', default=False)

group = parser.add_argument_group('TTA / BN Adapt')
group.add_argument('--lbatch', type=int, default=10, metavar='M',
                   help='Number of forward-pass batches in train mode for BN adaptation (default: 10)')
group.add_argument('--bn-adapt-num', type=int, default=-1, metavar='N',
                   help='Number of BN layers (from the front) to keep in train mode for tracking. '
                        '-1 = all BN layers adapt (default). '
                        '1 = only the stem BN (bn1). '
                        '0 = no BN adaptation (all eval).')
group.add_argument('--bn-momentum', type=float, default=None,
                   help='Override BN momentum for adaptation (default: use model default)')

group = parser.add_argument_group('Runtime')
group.add_argument('-b', '--batch-size', type=int, default=64)
group.add_argument('--device', default='cuda', type=str)
group.add_argument('--seed', type=int, default=42)
group.add_argument('-j', '--workers', type=int, default=4)
group.add_argument('--pin-mem', action='store_true', default=False)
group.add_argument('--no-prefetcher', action='store_true', default=False)
group.add_argument('--amp', action='store_true', default=False)
group.add_argument('--amp-dtype', default='float16', type=str)
group.add_argument('--model-dtype', default=None, type=str)
group.add_argument('--log-interval', type=int, default=50)
group.add_argument("--local_rank", default=0, type=int)
group.add_argument('--device-modules', default=None, type=str, nargs='+')

group = parser.add_argument_group('Output')
group.add_argument('--output-dir', type=str, default=None,
                   help='Output directory for summary CSV (default: same dir as checkpoint)')

group = parser.add_argument_group('Compatibility (unused)')
group.add_argument('--drop', type=float, default=0.0)
group.add_argument('--drop-path', type=float, default=None)
group.add_argument('--drop-block', type=float, default=None)
group.add_argument('--bn-eps', type=float, default=None)


def _parse_args():
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)
    args = parser.parse_args(remaining)
    return args


def _save_bn_state(model):
    """Snapshot BN running_mean / running_var / num_batches_tracked."""
    state = {}
    for name, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)):
            state[name] = {
                'running_mean': m.running_mean.clone(),
                'running_var': m.running_var.clone(),
                'num_batches_tracked': m.num_batches_tracked.clone(),
            }
    return state


def _restore_bn_state(model, state):
    """Restore BN running statistics from a previous snapshot."""
    for name, m in model.named_modules():
        if name in state:
            m.running_mean.copy_(state[name]['running_mean'])
            m.running_var.copy_(state[name]['running_var'])
            m.num_batches_tracked.copy_(state[name]['num_batches_tracked'])


def _set_bn_momentum(model, momentum):
    """Override BN momentum for all BatchNorm layers."""
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)):
            m.momentum = momentum


def _get_bn_layers(model):
    """Return ordered list of (name, module) for all BN layers."""
    return [
        (name, m) for name, m in model.named_modules()
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm))
    ]


def _set_bn_train_partial(model, bn_adapt_num):
    """Set the first bn_adapt_num BN layers to train mode, the rest to eval.

    Call after model.train() to selectively freeze later BN layers.
    -1 means all BN layers stay in train mode (no-op after model.train()).
     0 means all BN layers are set to eval mode.
    """
    bn_layers = _get_bn_layers(model)
    if bn_adapt_num < 0:
        return len(bn_layers)

    for idx, (name, m) in enumerate(bn_layers):
        if idx >= bn_adapt_num:
            m.eval()
    active = min(bn_adapt_num, len(bn_layers))
    return active


def main():
    utils.setup_default_logging()
    args = _parse_args()

    use_sa2 = args.prop_size > 0

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

    # ── Create model ──
    model_kwargs = {}
    if args.use_sam_module != -1 or args.parallel_attention:
        model_kwargs['sam_kernel_size'] = args.vit_kernel_size
        model_kwargs['spatial_group_size'] = args.spatial_group_size

    in_chans = args.input_size[0] if args.input_size else 3

    model = create_model(
        args.model,
        num_classes=args.num_classes,
        in_chans=in_chans,
        parallel_attention=args.parallel_attention,
        use_se_module=args.use_se_module,
        use_sam_module=args.use_sam_module,
        reverse_se_sam=args.reverse_se,
        sam_norm_type=args.sam_norm_type,
        **model_kwargs,
    )
    model.to(device=device, dtype=model_dtype)

    # ── Replace SpatialAttention with SpatialAttention2 if prop_size > 0 ──
    if use_sa2:
        base = _get_base_model(model)
        if isinstance(base.conv1, nn.Sequential):
            ch = base.conv1[-1].out_channels
        else:
            ch = base.conv1.out_channels
        base.spatial_attn = SpatialAttention2(
            kernel_size=args.vit_kernel_size, channels=ch,
            prop_size=args.prop_size,
        ).to(device=device, dtype=model_dtype)
        _logger.info(f'SpatialAttention2 installed: prop_size={args.prop_size}, '
                     f'channels={ch}, kernel={args.vit_kernel_size}')

    # ── Attention fusion options ──
    if args.parallel_attention:
        base = _get_base_model(model)
        base.attn_detach = args.detach
        base.attn_residual = args.residual
        _logger.info(f'Attention fusion: detach={args.detach}, residual={args.residual}')

    # ── Auto-detect base checkpoint from FedAvg checkpoint args ──
    if args.resume_base is None:
        probe_ckpt = torch.load(args.checkpoint, map_location='cpu')
        saved_args = probe_ckpt.get('args', {})
        if 'resume' in saved_args and saved_args['resume']:
            args.resume_base = saved_args['resume']
            _logger.info(f'Auto-detected base checkpoint from FedAvg args: {args.resume_base}')
        del probe_ckpt

    # ── Load base pretrained backbone first ──
    if args.resume_base:
        _logger.info(f'Loading base backbone: {args.resume_base}')
        base_ckpt = torch.load(args.resume_base, map_location='cpu')
        base_sd = base_ckpt['state_dict'] if 'state_dict' in base_ckpt else base_ckpt
        model_sd = model.state_dict()
        base_skipped = [k for k in base_sd
                        if k in model_sd and base_sd[k].shape != model_sd[k].shape]
        for k in base_skipped:
            del base_sd[k]
        if base_skipped:
            _logger.info(f'  Skipped (shape mismatch): {base_skipped}')
        missing, unexpected = model.load_state_dict(base_sd, strict=False)
        _logger.info(f'  Base loaded: {len(model_sd) - len(missing)} keys, '
                     f'{len(missing)} missing, {len(unexpected)} unexpected')

    # ── Load FedAvg trainable-params checkpoint on top ──
    _logger.info(f'Loading FedAvg checkpoint: {args.checkpoint}')
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt

    model_sd = model.state_dict()
    skipped = [k for k in state_dict
               if k in model_sd and state_dict[k].shape != model_sd[k].shape]
    for k in skipped:
        del state_dict[k]
    if skipped:
        _logger.info(f'  Skipped (shape mismatch): {skipped}')
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    _logger.info(f'  FedAvg overlay: {len(state_dict) - len(unexpected)} keys loaded')
    if unexpected:
        _logger.info(f'  Unexpected keys ({len(unexpected)}): {unexpected}')

    _apply_train_mode(model, args.train_mode)
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _logger.info(f'Model: {safe_model_name(args.model)}, '
                 f'trainable: {trainable:,}/{total_params:,} (train_mode={args.train_mode})')

    data_config = resolve_data_config(vars(args) | {'pretrained': False}, model=model, verbose=True)

    if args.bn_momentum is not None:
        _set_bn_momentum(model, args.bn_momentum)
        _logger.info(f'BN momentum overridden to {args.bn_momentum}')

    # ── Output ──
    output_dir = args.output_dir or os.path.dirname(args.checkpoint)
    os.makedirs(output_dir, exist_ok=True)
    bn_suffix = f'_bn{args.bn_adapt_num}' if args.bn_adapt_num >= 0 else ''
    summary_path = os.path.join(output_dir, f'tta_lbatch{args.lbatch}{bn_suffix}.csv')

    # ── Save original BN state for per-corruption reset ──
    original_bn_state = _save_bn_state(model)

    # ── Log BN adapt scope ──
    bn_layers = _get_bn_layers(model)
    if args.bn_adapt_num < 0:
        active_count = len(bn_layers)
    else:
        active_count = min(args.bn_adapt_num, len(bn_layers))
    _logger.info(f'BN adapt scope: {active_count}/{len(bn_layers)} layers will track '
                 f'(bn_adapt_num={args.bn_adapt_num})')
    if active_count > 0 and active_count < len(bn_layers):
        for idx, (name, _) in enumerate(bn_layers[:active_count]):
            _logger.info(f'  [train] {idx}: {name}')
        _logger.info(f'  ... remaining {len(bn_layers) - active_count} BN layers frozen (eval)')

    # ── Per-corruption: BN adapt + evaluate ──
    _logger.info(f'\nTTA BN-Adapt: lbatch={args.lbatch}, severity={args.severity}')
    _logger.info(f'{"="*60}')

    eval_results = OrderedDict()
    acc1_sum = 0.0

    for corruption in CORRUPTIONS:
        _logger.info(f'\n  [{corruption}]')

        _restore_bn_state(model, original_bn_state)

        if args.lbatch > 0:
            adapt_loader = create_client_loader(
                corruption, args, data_config, model_dtype, device)
            bn_adapt(model, adapt_loader, args, device, model_dtype, args.lbatch)

        eval_loader = create_eval_loader(corruption, args, data_config, model_dtype, device)
        eval_loss, eval_acc1, eval_acc5 = evaluate(
            model, eval_loader, args, device, model_dtype)

        eval_results[corruption] = {'loss': eval_loss, 'top1': eval_acc1, 'top5': eval_acc5}
        acc1_sum += eval_acc1
        _logger.info(f'    Acc@1={eval_acc1:.3f}%  Acc@5={eval_acc5:.3f}%  Loss={eval_loss:.4f}')

    mean_acc1 = acc1_sum / len(CORRUPTIONS)
    _logger.info(f'\n{"="*60}')
    _logger.info(f'  Mean Acc@1: {mean_acc1:.3f}%  (lbatch={args.lbatch})')
    _logger.info(f'{"="*60}')

    # ── Write summary CSV ──
    with open(summary_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['lbatch', 'severity', 'mean_acc1'] + list(CORRUPTIONS)
        writer.writerow(header)
        row = [args.lbatch, args.severity, f'{mean_acc1:.3f}']
        for c in CORRUPTIONS:
            row.append(f'{eval_results[c]["top1"]:.3f}')
        writer.writerow(row)

    _logger.info(f'  Summary written: {summary_path}')


if __name__ == '__main__':
    main()

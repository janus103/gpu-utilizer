#!/usr/bin/env python3
"""ViT validation on ImageNet-C with parallel-attention + SpatialAttention2.

Loads checkpoints produced by train_fedavg_adain_policy.py (partial state_dict
containing only trainable parameters) and evaluates on a chosen corruption.

Usage:
    python tta_vit.py --corruption gaussian_noise \
        --fedavg-ckpt ./output/fedavg_direct_k2_tm1_le1_p5_wu10_aa5/best.pth \
        --resume ./VIT_IMG_PAR/Normal_parallel_train_1_kernel_size_2/model_best.pth.tar \
        --parallel-attention --vit-kernel-size 2 --spatial-group-size 1 \
        --train-mode 1 --prop-size 5 --batch-size 512
"""
import argparse
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
from timm.models import create_model, safe_model_name, resume_checkpoint
from timm.models.vision_transformer import SpatialAttention2

_logger = logging.getLogger('tta_vit')


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
    embedding_prefixes = (
        'patch_embed', 'pos_embed', 'cls_token', 'reg_token',
        'norm_pre', 'conv1', 'bn1', 'act1',
    )
    return name.startswith(embedding_prefixes)


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
    if trainable_count == 0:
        raise ValueError(
            f'No trainable parameters for train_mode={train_mode}. '
            'Check --parallel-attention / --use-se-module / --use-sam-module.'
        )
    return trainable_count


def set_trainable_state(model, state):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in state:
                param.data.copy_(state[name])


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


@torch.no_grad()
def evaluate(model, loader, device, model_dtype, prefetcher=True):
    model.eval()
    top1_m = utils.AverageMeter()
    top5_m = utils.AverageMeter()

    for images, target in loader:
        if not prefetcher:
            images = images.to(device=device, dtype=model_dtype)
            target = target.to(device=device)

        output = model(images)
        if isinstance(output, (tuple, list)):
            output = output[0]

        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        top1_m.update(acc1.item(), images.shape[0])
        top5_m.update(acc5.item(), images.shape[0])

    return top1_m.avg, top5_m.avg


def _parse_args():
    config_parser = argparse.ArgumentParser(description='Config', add_help=False)
    config_parser.add_argument('-c', '--config', default='', type=str, metavar='FILE')

    parser = argparse.ArgumentParser(description='ViT validation on ImageNet-C')

    group = parser.add_argument_group('Data')
    group.add_argument('--data-root', type=str, default='/home/oem/servers/imagenet-c')
    group.add_argument('--severity', type=int, default=5)
    group.add_argument('--val-split', type=str, default='validation')
    group.add_argument('--class-map', default='', type=str)
    group.add_argument('--input-img-mode', default=None, type=str)
    group.add_argument('--corruption', type=str, nargs='+', default=None,
                       help='Corruption(s) to evaluate (default: all 15)')

    group = parser.add_argument_group('Model')
    group.add_argument('--model', default='vit_base_patch16_224', type=str)
    group.add_argument('--resume', type=str,
                       default='./VIT_IMG_PAR/Normal_parallel_train_1_kernel_size_2/model_best.pth.tar',
                       help='Base model checkpoint (full state_dict)')
    group.add_argument('--fedavg-ckpt', type=str, default=None,
                       help='FedAvg checkpoint (partial state_dict from train_fedavg_adain_policy.py)')
    group.add_argument('--num-classes', type=int, default=1000)
    group.add_argument('--input-size', default=[3, 224, 224], nargs=3, type=int)
    group.add_argument('--parallel-attention', action='store_true', default=True)
    group.add_argument('--vit-kernel-size', type=int, default=2)
    group.add_argument('--spatial-group-size', type=int, default=1)
    group.add_argument('--vit-last', action='store_true', default=False)
    group.add_argument('--vit-closed', type=str, default=None, choices=['same', 'diff'])
    group.add_argument('--use-se-module', action='store_true', default=False)
    group.add_argument('--use-sam-module', type=int, default=-1)
    group.add_argument('--reverse-se', action='store_true', default=False)
    group.add_argument('--train-mode', type=int, default=1, choices=[0, 1, 2],
                       help='0=all, 1=aux+embedding, 2=aux only')
    group.add_argument('--sam-norm-type', type=int, default=0, choices=[0, 1, 2, 3, 4])
    group.add_argument('--vit-early-norm-types', type=int, nargs=4, default=None,
                       choices=[0, 1, 2, 3, 4])

    group = parser.add_argument_group('SpatialAttention2')
    group.add_argument('--prop-size', type=int, default=5,
                       help='N-class spatial attention (0=use original SpatialAttention, >0=SpatialAttention2)')

    group = parser.add_argument_group('Runtime')
    group.add_argument('-b', '--batch-size', type=int, default=512)
    group.add_argument('-j', '--workers', type=int, default=4)
    group.add_argument('--device', default='cuda', type=str)
    group.add_argument('--model-dtype', default=None, type=str)
    group.add_argument('--pin-mem', action='store_true', default=False)
    group.add_argument('--no-prefetcher', action='store_true', default=False)
    group.add_argument('--seed', type=int, default=42)
    group.add_argument('--device-modules', default=None, type=str, nargs='+')
    group.add_argument('--log-interval', type=int, default=50)

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

    vit_model = 'vit' in args.model.lower()
    vit_norm_kwargs = {}
    if args.vit_early_norm_types is not None:
        if not vit_model:
            raise ValueError('--vit-early-norm-types is only for ViT models.')
        vit_norm_kwargs['vit_early_norm_types'] = args.vit_early_norm_types
    if args.use_sam_module != -1 or args.parallel_attention:
        vit_norm_kwargs['sam_kernel_size'] = args.vit_kernel_size
        vit_norm_kwargs['spatial_group_size'] = args.spatial_group_size
    if vit_model and args.vit_last:
        vit_norm_kwargs['vit_last'] = True
    if vit_model and args.vit_closed is not None:
        vit_norm_kwargs['vit_closed'] = args.vit_closed

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
        **vit_norm_kwargs,
    )
    model.to(device=device, dtype=model_dtype)

    # Step 1: Load base model (full state_dict)
    if args.resume:
        _logger.info(f'Loading base checkpoint: {args.resume}')
        ckpt = torch.load(args.resume, map_location='cpu')
        state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            _logger.info(f'  Missing keys ({len(missing)}): {missing}')
        if unexpected:
            _logger.info(f'  Unexpected keys ({len(unexpected)}): {unexpected}')

    # Step 2: Replace SpatialAttention with SpatialAttention2 if prop_size > 0
    use_sa2 = args.prop_size > 0
    if use_sa2:
        base = _get_base_model(model)
        ch = base.embed_dim
        base.spatial_attn = SpatialAttention2(
            kernel_size=args.vit_kernel_size, channels=ch,
            prop_size=args.prop_size,
        ).to(device=device, dtype=model_dtype)
        _logger.info(f'SpatialAttention2 installed: prop_size={args.prop_size}, '
                     f'channels={ch}, kernel={args.vit_kernel_size}')

    # Step 3: Apply train_mode to mark trainable params (needed for set_trainable_state)
    trainable_count = _apply_train_mode(model, args.train_mode)
    total_params = sum(p.numel() for p in model.parameters())
    _logger.info(f'Model: {safe_model_name(args.model)}, '
                 f'trainable: {trainable_count:,}/{total_params:,} '
                 f'(train_mode={args.train_mode})')

    # Step 4: Load FedAvg checkpoint (partial state_dict of trainable params only)
    if args.fedavg_ckpt:
        _logger.info(f'Loading FedAvg checkpoint: {args.fedavg_ckpt}')
        ckpt = torch.load(args.fedavg_ckpt, map_location='cpu')
        fedavg_state = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
        set_trainable_state(model, fedavg_state)
        loaded_keys = [k for k in fedavg_state.keys()
                       if k in dict(model.named_parameters())]
        _logger.info(f'  Loaded {len(loaded_keys)} trainable parameter tensors')
        if 'round' in ckpt:
            _logger.info(f'  Round: {ckpt["round"]}')
        if 'mean_acc1' in ckpt:
            _logger.info(f'  Saved mean_acc1: {ckpt["mean_acc1"]:.3f}%')

    data_config = resolve_data_config(
        vars(args) | {'pretrained': False}, model=model, verbose=True,
    )

    # Evaluate
    _logger.info(f'\nEvaluating {len(corruptions)} corruption(s), severity={args.severity}')
    _logger.info('=' * 60)

    acc1_sum = 0.0
    for corruption in corruptions:
        eval_loader = create_eval_loader(corruption, args, data_config, model_dtype, device)
        acc1, acc5 = evaluate(
            model, eval_loader, device, model_dtype,
            prefetcher=not args.no_prefetcher,
        )
        acc1_sum += acc1
        _logger.info(f'  {corruption:<22s}  Acc@1={acc1:.3f}%  Acc@5={acc5:.3f}%')

    if len(corruptions) > 1:
        mean_acc1 = acc1_sum / len(corruptions)
        _logger.info(f'\n  Mean Acc@1: {mean_acc1:.3f}%')

    _logger.info('=' * 60)


if __name__ == '__main__':
    main()

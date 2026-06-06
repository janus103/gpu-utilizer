#!/usr/bin/env python3
"""BN Adaptation Evaluation for FedAvg Round Checkpoints.

Loads a FedAvg round checkpoint, adapts BatchNorm running stats using
a few batches from each corruption, then evaluates.

Usage:
    python eval_bn_adapt.py \
        --model resnet50 \
        --resume ./R50_IMG_PAR/Normal_parallel_K7_GRP_1/model_best.pth.tar \
        --resume-round ./output/.../round_007.pth \
        --resume-aug ./output/aug_cls_joint/best.pth \
        --parallel-attention --vit-kernel-size 7 --spatial-group-size 1 \
        --var-feature --moe-channel \
        --lbatch 10 --batch-size 256
"""
import argparse
import logging
import os
import time
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm import utils
from timm.data import create_dataset, create_loader, resolve_data_config
from timm.models import create_model, safe_model_name

from timm.models.vision_transformer import SpatialAttention3, MoEChannelAttention
from train_aug_classifier import AugClassifier, NUM_AUG_TRANSFORMS

_logger = logging.getLogger('eval_bn_adapt')

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


def set_trainable_state(model, state):
    """Load trainable state, handling 'module.' prefix from DataParallel."""
    cleaned = OrderedDict()
    for k, v in state.items():
        cleaned[k.replace('module.', '', 1) if k.startswith('module.') else k] = v

    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in cleaned:
                param.data.copy_(cleaned[name])


def create_loader_for_corruption(corruption, args, data_config, model_dtype, device, is_training=False):
    data_dir = os.path.join(args.data_root, corruption, str(args.severity))
    input_img_mode = args.input_img_mode or ('RGB' if data_config['input_size'][0] == 3 else 'L')
    dataset = create_dataset(
        '', root=data_dir, split=args.val_split, is_training=is_training,
        class_map='', download=False,
        batch_size=args.batch_size, input_img_mode=input_img_mode,
    )
    loader = create_loader(
        dataset, input_size=data_config['input_size'],
        batch_size=args.batch_size, is_training=is_training,
        no_aug=True,
        interpolation=data_config['interpolation'],
        num_workers=args.workers, crop_pct=data_config['crop_pct'],
        mean=data_config['mean'], std=data_config['std'],
        pin_memory=args.pin_mem,
        img_dtype=model_dtype or torch.float32,
        device=device, distributed=False,
        use_prefetcher=not args.no_prefetcher,
    )
    return loader


def adapt_bn(model, loader, lbatch, device, model_dtype):
    """Run lbatch forward passes in train mode to adapt BN running stats.

    Uses the default BN momentum (exponential moving average) on top of
    the existing running stats — does NOT reset them to zero.
    Only BN layers are set to train mode; everything else stays in eval.
    """
    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.train()

    with torch.no_grad():
        for i, (images, _) in enumerate(loader):
            if i >= lbatch:
                break
            images = images.to(device=device, dtype=model_dtype or torch.float32)
            with torch.cuda.amp.autocast(enabled=model_dtype == torch.float16, dtype=torch.float16):
                model(images)


@torch.no_grad()
def evaluate(model, loader, device, model_dtype):
    model.eval()
    top1_m = utils.AverageMeter()
    top5_m = utils.AverageMeter()

    for images, target in loader:
        images = images.to(device=device, dtype=model_dtype or torch.float32)
        target = target.to(device=device)
        with torch.cuda.amp.autocast(enabled=model_dtype == torch.float16, dtype=torch.float16):
            output = model(images)
        if isinstance(output, (tuple, list)):
            output = output[0]
        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        top1_m.update(acc1.item(), images.shape[0])
        top5_m.update(acc5.item(), images.shape[0])

    return top1_m.avg, top5_m.avg


def main():
    utils.setup_default_logging()

    parser = argparse.ArgumentParser(description='BN Adaptation Eval for FedAvg checkpoints')
    parser.add_argument('--model', default='resnet50', type=str)
    parser.add_argument('--resume', type=str, default='', help='Optional base ResNet checkpoint')
    parser.add_argument('--resume-round', type=str, default='', help='Optional FedAvg round checkpoint')
    parser.add_argument('--resume-aug', type=str, default=None, help='AugClassifier checkpoint')
    parser.add_argument('--allow-random-init', action='store_true',
                        help='Allow source-only evaluation without SOA checkpoints.')
    parser.add_argument('--num-classes', type=int, default=1000)
    parser.add_argument('--input-size', default=[3, 224, 224], nargs=3, type=int)

    parser.add_argument('--parallel-attention', action='store_true', default=True)
    parser.add_argument('--vit-kernel-size', type=int, default=7)
    parser.add_argument('--spatial-group-size', type=int, default=1)
    parser.add_argument('--sam-norm-type', type=int, default=0)
    parser.add_argument('--var-feature', action='store_true', default=False)
    parser.add_argument('--moe-channel', action='store_true', default=False)
    parser.add_argument('--use-se-module', action='store_true', default=False)
    parser.add_argument('--use-sam-module', type=int, default=-1)
    parser.add_argument('--reverse-se', action='store_true', default=False)

    parser.add_argument('--data-root', type=str, default='/home/oem/servers/imagenet-c')
    parser.add_argument('--severity', type=int, default=5)
    parser.add_argument('--val-split', type=str, default='validation')
    parser.add_argument('--input-img-mode', default=None, type=str)
    parser.add_argument('--lbatch', type=int, required=True,
                        help='Number of batches for BN adaptation per corruption (0=no adaptation)')
    parser.add_argument('-b', '--batch-size', type=int, default=256)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--pin-mem', action='store_true', default=False)
    parser.add_argument('--no-prefetcher', action='store_true', default=False)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--amp', action='store_true', help='Use CUDA AMP FP16 for BN adaptation/evaluation forwards.')

    parser.add_argument('--drop', type=float, default=0.0)
    parser.add_argument('--drop-path', type=float, default=None)
    parser.add_argument('--drop-block', type=float, default=None)
    parser.add_argument('--bn-momentum', type=float, default=None)
    parser.add_argument('--bn-eps', type=float, default=None)

    args = parser.parse_args()
    device = torch.device(args.device)

    model_dtype = torch.float16 if args.amp else None
    model_kwargs = {}
    if args.use_sam_module != -1 or args.parallel_attention:
        model_kwargs['sam_kernel_size'] = args.vit_kernel_size
        model_kwargs['spatial_group_size'] = args.spatial_group_size

    in_chans = args.input_size[0]

    model = create_model(
        args.model,
        num_classes=args.num_classes,
        in_chans=in_chans,
        parallel_attention=args.parallel_attention,
        use_se_module=args.use_se_module,
        use_sam_module=args.use_sam_module,
        reverse_se_sam=args.reverse_se,
        sam_norm_type=args.sam_norm_type,
        var_feature=args.var_feature,
        **model_kwargs,
    )
    model.to(device=device)

    if args.resume:
        _logger.info(f'Loading base checkpoint: {args.resume}')
        ckpt = torch.load(args.resume, map_location='cpu')
        state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
        model_sd = model.state_dict()
        skipped = [k for k in state_dict if k in model_sd and state_dict[k].shape != model_sd[k].shape]
        for k in skipped:
            del state_dict[k]
        model.load_state_dict(state_dict, strict=False)
    elif args.allow_random_init:
        _logger.warning('No --resume checkpoint provided; using random-initialized base model.')
    else:
        raise ValueError('--resume is required unless --allow-random-init is set.')

    # Replace SpatialAttention → SpatialAttention3
    base = _get_base_model(model)
    ch = base.conv1[-1].out_channels if isinstance(base.conv1, nn.Sequential) else base.conv1.out_channels
    base.spatial_attn = SpatialAttention3(
        kernel_size=args.vit_kernel_size, channels=ch, var_feature=args.var_feature,
    ).to(device=device)
    _logger.info(f'SpatialAttention3 installed: channels={ch}')

    # Load AugClassifier
    aug_num_transforms = NUM_AUG_TRANSFORMS
    if args.resume_aug:
        _logger.info(f'Loading AugClassifier: {args.resume_aug}')
        aug_ckpt = torch.load(args.resume_aug, map_location='cpu')
        aug_embed_dim = aug_ckpt.get('embed_dim', 768)
        aug_num_transforms = aug_ckpt.get('num_transforms', NUM_AUG_TRANSFORMS)
        aug_cls = AugClassifier(in_chans=in_chans, num_classes=aug_num_transforms, embed_dim=aug_embed_dim)
        aug_sd = aug_ckpt['state_dict'] if 'state_dict' in aug_ckpt else aug_ckpt
        aug_cls.load_state_dict(aug_sd, strict=True)
        aug_cls.to(device=device)
        aug_cls.eval()
        for p in aug_cls.parameters():
            p.requires_grad = False
        base.aug_classifier = aug_cls
        _logger.info(f'  AugClassifier frozen, stage1 mean residual active')

    # MoE Channel Attention
    if args.moe_channel:
        old_ca = base.channel_attn
        moe_ca = MoEChannelAttention(
            channels=ch, num_experts=aug_num_transforms,
        ).to(device=device)
        moe_ca.init_from_channel_attn(old_ca)
        base.channel_attn = moe_ca
        _logger.info(f'MoEChannelAttention installed: {aug_num_transforms} experts '
                     f'(initialized from pretrained ChannelAttention)')

    if args.resume_round:
        _logger.info(f'Loading round checkpoint: {args.resume_round}')
        rnd_ckpt = torch.load(args.resume_round, map_location='cpu')
        set_trainable_state(model, rnd_ckpt['state_dict'])
        _logger.info(f'  Round {rnd_ckpt["round"]}, mean_acc1={rnd_ckpt.get("mean_acc1", "N/A")}')
    elif args.allow_random_init:
        _logger.warning('No --resume-round checkpoint provided; using randomly initialized trainable state.')
    else:
        raise ValueError('--resume-round is required unless --allow-random-init is set.')

    # Attention fusion
    base.attn_detach = True
    base.attn_residual = False

    data_config = resolve_data_config(vars(args) | {'pretrained': False}, model=model, verbose=True)

    _logger.info(f'\nEvaluating with BN adaptation (lbatch={args.lbatch})')
    _logger.info(f'{"="*60}')

    # Save original BN state for resetting per-corruption
    original_bn_state = {}
    for name, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            original_bn_state[name] = {
                'running_mean': m.running_mean.clone(),
                'running_var': m.running_var.clone(),
                'num_batches_tracked': m.num_batches_tracked.clone(),
            }

    acc1_sum = 0.0
    results = OrderedDict()

    for corruption in CORRUPTIONS:
        # Restore original BN state
        for name, m in model.named_modules():
            if name in original_bn_state:
                m.running_mean.copy_(original_bn_state[name]['running_mean'])
                m.running_var.copy_(original_bn_state[name]['running_var'])
                m.num_batches_tracked.copy_(original_bn_state[name]['num_batches_tracked'])

        loader = create_loader_for_corruption(
            corruption, args, data_config, model_dtype, device, is_training=True)

        if args.lbatch > 0:
            adapt_bn(model, loader, args.lbatch, device, model_dtype)

        eval_loader = create_loader_for_corruption(
            corruption, args, data_config, model_dtype, device, is_training=False)
        acc1, acc5 = evaluate(model, eval_loader, device, model_dtype)

        results[corruption] = {'top1': acc1, 'top5': acc5}
        acc1_sum += acc1
        _logger.info(f'  {corruption:<22s}  Acc@1={acc1:.3f}%  Acc@5={acc5:.3f}%')

    mean_acc1 = acc1_sum / len(CORRUPTIONS)
    _logger.info(f'\n{"="*60}')
    _logger.info(f'  Mean Acc@1: {mean_acc1:.3f}%  (lbatch={args.lbatch})')
    _logger.info(f'{"="*60}')


if __name__ == '__main__':
    main()

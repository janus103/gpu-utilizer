#!/usr/bin/env python3
"""Evaluate a FedAvg-trained ViT model on all 15 ImageNet-C corruptions.

Mirrors the model construction logic of train_fedavg_vit_policy2.py:
  1. create_model (ViT, no var_feature kwarg)
  2. Load base backbone checkpoint (--resume)
  3. Replace SpatialAttention → SpatialAttention3 (sigmoid, var_feature)
  4. Set attn_detach / attn_residual
  5. Load frozen AugClassifier (--resume-aug)
  6. Replace ChannelAttention → MoEChannelAttention (--moe-channel)
  7. Overlay FedAvg trainable state (--resume-round)
  8. Evaluate on 15 ImageNet-C corruptions

Usage:
    python eval_vit.py \
        --resume ./VIT_IMG_PAR_resume/Normal_parallel_train_1/model_best.pth.tar \
        --resume-round ./output/fedavg_vit_sa3_k2_tm3_le1_aa10_det_vf_aug_moe/best.pth \
        --resume-aug ./output/aug_cls_joint/best.pth \
        --vit-kernel-size 2 --var-feature --moe-channel --detach \
        --batch-size 512
"""
import argparse
import csv
import logging
import os
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm import utils
from timm.data import create_dataset, create_loader, resolve_data_config
from timm.models import create_model, safe_model_name

from timm.models.vision_transformer import SpatialAttention3, MoEChannelAttention
from train_aug_classifier import AugClassifier, NUM_AUG_TRANSFORMS

_logger = logging.getLogger('eval_vit')

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


def _is_mobilevit_model(model_name: str) -> bool:
    return model_name.lower().startswith('mobilevit')


def _get_attention_channels(model):
    base = _get_base_model(model)
    if hasattr(base, 'embed_dim'):
        return base.embed_dim
    channel_attn = getattr(base, 'channel_attn', None)
    if channel_attn is not None and hasattr(channel_attn, 'fc2'):
        return channel_attn.fc2.out_channels
    raise AttributeError(
        'Unable to infer SOA attention channels. Expected a ViT embed_dim '
        'or an initialized channel_attn.fc2 module.'
    )


def set_trainable_state(model, state):
    cleaned = OrderedDict()
    for k, v in state.items():
        cleaned[k.replace('module.', '', 1) if k.startswith('module.') else k] = v

    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in cleaned:
                param.data.copy_(cleaned[name])


def create_eval_loader(corruption, args, data_config, model_dtype, device):
    data_dir = os.path.join(args.data_root, corruption, str(args.severity))
    input_img_mode = args.input_img_mode or ('RGB' if data_config['input_size'][0] == 3 else 'L')
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
def evaluate(model, loader, device, model_dtype):
    model.eval()
    top1_m = utils.AverageMeter()
    top5_m = utils.AverageMeter()
    loss_m = utils.AverageMeter()

    for images, target in loader:
        images = images.to(device=device, dtype=model_dtype or torch.float32)
        target = target.to(device=device)
        with torch.cuda.amp.autocast(enabled=model_dtype == torch.float16, dtype=torch.float16):
            output = model(images)
        if isinstance(output, (tuple, list)):
            output = output[0]
        loss = F.cross_entropy(output.float(), target)
        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        loss_m.update(loss.item(), images.shape[0])
        top1_m.update(acc1.item(), images.shape[0])
        top5_m.update(acc5.item(), images.shape[0])

    return loss_m.avg, top1_m.avg, top5_m.avg


def main():
    utils.setup_default_logging()

    parser = argparse.ArgumentParser(
        description='Evaluate FedAvg ViT checkpoint on ImageNet-C')

    parser.add_argument('--model', default='vit_base_patch16_224', type=str)
    parser.add_argument('--resume', type=str, default=None,
                        help='Base ViT backbone checkpoint')
    parser.add_argument('--resume-round', type=str, default=None,
                        help='FedAvg round/best checkpoint (best.pth or round_*.pth). '
                             'Optional with --allow-random-init.')
    parser.add_argument('--allow-random-init', action='store_true', default=False,
                        help='Allow evaluation without --resume/--resume-round. '
                             'Useful for plumbing checks where weights do not matter.')
    parser.add_argument('--resume-aug', type=str, default=None,
                        help='AugClassifier checkpoint')
    parser.add_argument('--num-classes', type=int, default=1000)
    parser.add_argument('--input-size', default=[3, 224, 224], nargs=3, type=int)

    parser.add_argument('--parallel-attention', action='store_true', default=True)
    parser.add_argument('--vit-kernel-size', type=int, default=2)
    parser.add_argument('--spatial-group-size', type=int, default=1)
    parser.add_argument('--sam-norm-type', type=int, default=0)
    parser.add_argument('--var-feature', action='store_true', default=False)
    parser.add_argument('--moe-channel', action='store_true', default=False)
    parser.add_argument('--detach', action='store_true', default=False)
    parser.add_argument('--residual', action='store_true', default=False)
    parser.add_argument('--use-se-module', action='store_true', default=False)
    parser.add_argument('--use-sam-module', type=int, default=-1)
    parser.add_argument('--reverse-se', action='store_true', default=False)

    parser.add_argument('--data-root', type=str, default='/home/oem/servers/imagenet-c')
    parser.add_argument('--severity', type=int, default=5)
    parser.add_argument('--val-split', type=str, default='validation')
    parser.add_argument('--input-img-mode', default=None, type=str)
    parser.add_argument('-b', '--batch-size', type=int, default=256)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--pin-mem', action='store_true', default=False)
    parser.add_argument('--no-prefetcher', action='store_true', default=False)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--amp', action='store_true', help='Use CUDA AMP FP16 for evaluation forwards.')
    parser.add_argument('--output-csv', type=str, default=None,
                        help='Output CSV path (default: <ckpt_dir>/eval_corruptions.csv)')

    args = parser.parse_args()
    if args.resume_round is None and not args.allow_random_init:
        raise ValueError('--resume-round is required unless --allow-random-init is set.')

    device = torch.device(args.device)
    model_dtype = torch.float16 if args.amp else None
    is_mobilevit = _is_mobilevit_model(args.model)

    # ── Create model (no var_feature in create_model) ──
    vit_kwargs = {
        'sam_kernel_size': args.vit_kernel_size,
        'spatial_group_size': args.spatial_group_size,
    }
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
        **vit_kwargs,
    )
    model.to(device=device)

    # ── Load base backbone checkpoint ──
    if args.resume:
        _logger.info(f'Loading base checkpoint: {args.resume}')
        ckpt = torch.load(args.resume, map_location='cpu')
        state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
        model_sd = model.state_dict()
        skipped = [k for k in state_dict
                   if k in model_sd and state_dict[k].shape != model_sd[k].shape]
        for k in skipped:
            del state_dict[k]
        if skipped:
            _logger.info(f'  Skipped (shape mismatch): {skipped}')
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            _logger.info(f'  Missing keys ({len(missing)}): {missing}')
        if unexpected:
            _logger.info(f'  Unexpected keys ({len(unexpected)}): {unexpected}')

    # ── Replace SpatialAttention → SpatialAttention3 ──
    base = _get_base_model(model)
    ch = _get_attention_channels(model)
    base.spatial_attn = SpatialAttention3(
        kernel_size=args.vit_kernel_size, channels=ch,
        var_feature=args.var_feature, activation='sigmoid',
    ).to(device=device)
    attn_site = 'stem' if is_mobilevit else 'patch_embed'
    _logger.info(f'SpatialAttention3 installed at {attn_site}: channels={ch}, '
                 f'kernel={args.vit_kernel_size}, var_feature={args.var_feature}, '
                 f'activation=sigmoid')

    # ── Attention fusion options ──
    base.attn_detach = args.detach
    base.attn_residual = args.residual
    _logger.info(f'Attention fusion: detach={base.attn_detach}, residual={base.attn_residual}')

    # ── Load frozen AugClassifier ──
    aug_num_transforms = NUM_AUG_TRANSFORMS
    if args.resume_aug:
        _logger.info(f'Loading AugClassifier: {args.resume_aug}')
        aug_ckpt = torch.load(args.resume_aug, map_location='cpu')
        aug_embed_dim = aug_ckpt.get('embed_dim', 768)
        aug_num_transforms = aug_ckpt.get('num_transforms', NUM_AUG_TRANSFORMS)
        _logger.info(f'  embed_dim={aug_embed_dim}, num_transforms={aug_num_transforms}')

        aug_cls = AugClassifier(
            in_chans=in_chans,
            num_classes=aug_num_transforms,
            embed_dim=aug_embed_dim,
        )
        aug_sd = aug_ckpt['state_dict'] if 'state_dict' in aug_ckpt else aug_ckpt
        aug_cls.load_state_dict(aug_sd, strict=True)
        aug_cls.to(device=device)
        aug_cls.eval()
        for p in aug_cls.parameters():
            p.requires_grad = False
        base.aug_classifier = aug_cls
        _logger.info(f'  AugClassifier frozen: {sum(p.numel() for p in aug_cls.parameters()):,} params')

    # ── MoE Channel Attention ──
    if args.moe_channel:
        if not args.resume_aug and not args.allow_random_init:
            raise ValueError('--moe-channel requires --resume-aug')
        if not args.resume_aug:
            _logger.warning('No --resume-aug provided; initializing MoEChannelAttention with default augmentation classes.')
        old_ca = base.channel_attn
        moe_ca = MoEChannelAttention(
            channels=ch, num_experts=aug_num_transforms,
        ).to(device=device)
        moe_ca.init_from_channel_attn(old_ca)
        base.channel_attn = moe_ca
        _logger.info(f'MoEChannelAttention installed: {aug_num_transforms} experts')

    # ── Load FedAvg round checkpoint (trainable state overlay) ──
    if args.resume_round:
        _logger.info(f'Loading FedAvg checkpoint: {args.resume_round}')
        rnd_ckpt = torch.load(args.resume_round, map_location='cpu')
        set_trainable_state(model, rnd_ckpt['state_dict'])
        _logger.info(f'  Round {rnd_ckpt["round"]}, '
                     f'mean_acc1={rnd_ckpt.get("mean_acc1", "N/A")}')
    else:
        _logger.warning('No --resume-round provided; evaluating randomly initialized SOA/model weights.')

    model.eval()

    data_config = resolve_data_config(
        vars(args) | {'pretrained': False}, model=model, verbose=True)

    # ── Output CSV ──
    if args.output_csv is None:
        if args.resume_round:
            ckpt_dir = os.path.dirname(os.path.abspath(args.resume_round))
        else:
            ckpt_dir = os.path.join('./output', f'{safe_model_name(args.model)}_random_init_eval')
        os.makedirs(ckpt_dir, exist_ok=True)
        args.output_csv = os.path.join(ckpt_dir, 'eval_corruptions.csv')

    # ── Evaluate ──
    _logger.info(f'\nEvaluating {len(CORRUPTIONS)} corruptions (severity={args.severity})')
    _logger.info('=' * 60)

    results = OrderedDict()
    acc1_sum = 0.0

    for corruption in CORRUPTIONS:
        data_dir = os.path.join(args.data_root, corruption, str(args.severity))
        if not os.path.isdir(data_dir):
            _logger.warning(f'  {data_dir} not found, skipping.')
            continue

        loader = create_eval_loader(corruption, args, data_config, model_dtype, device)
        loss, acc1, acc5 = evaluate(model, loader, device, model_dtype)
        results[corruption] = {'loss': loss, 'top1': acc1, 'top5': acc5}
        acc1_sum += acc1
        _logger.info(f'  {corruption:<22s}  Acc@1={acc1:.3f}%  Acc@5={acc5:.3f}%')

    mean_acc1 = acc1_sum / len(results) if results else 0.0
    _logger.info(f'\n{"="*60}')
    _logger.info(f'  Mean Acc@1: {mean_acc1:.3f}%')
    _logger.info(f'{"="*60}')

    with open(args.output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['corruption', 'top1', 'top5', 'loss'])
        for c, r in results.items():
            writer.writerow([c, f'{r["top1"]:.3f}', f'{r["top5"]:.3f}', f'{r["loss"]:.4f}'])
        writer.writerow(['mean', f'{mean_acc1:.3f}', '', ''])

    _logger.info(f'Results saved to {args.output_csv}')


if __name__ == '__main__':
    main()

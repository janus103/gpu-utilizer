#!/usr/bin/env python3
"""Evaluate a FedAvg-trained model on all 15 ImageNet-C corruptions.

Loads a FedAvg checkpoint (best.pth), reconstructs the model from the
stored args, loads the backbone checkpoint, overlays the FedAvg-trained
parameters, and evaluates on every corruption at the given severity.
"""
import argparse
import csv
import importlib
import logging
import os
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm import utils
from timm.data import create_dataset, create_loader, resolve_data_config
from timm.models import create_model, safe_model_name

_logger = logging.getLogger('eval_fedavg')

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


def create_model_from_ckpt_args(ckpt_args, device, model_dtype=None):
    """Recreate the exact model architecture from saved fedavg checkpoint args."""
    model_name = ckpt_args['model']
    is_vit = 'vit' in model_name.lower()

    model_kwargs = {}
    if ckpt_args.get('use_sam_module', -1) != -1 or ckpt_args.get('parallel_attention', False):
        model_kwargs['sam_kernel_size'] = ckpt_args.get('vit_kernel_size', 7)
        model_kwargs['spatial_group_size'] = ckpt_args.get('spatial_group_size', 1)

    if is_vit:
        if ckpt_args.get('vit_last', False):
            model_kwargs['vit_last'] = True
        if ckpt_args.get('vit_closed') is not None:
            model_kwargs['vit_closed'] = ckpt_args['vit_closed']
        if ckpt_args.get('vit_early_norm_types') is not None:
            model_kwargs['vit_early_norm_types'] = ckpt_args['vit_early_norm_types']

    in_chans = ckpt_args.get('input_size', [3, 224, 224])[0]

    model = create_model(
        model_name,
        num_classes=ckpt_args.get('num_classes', 1000),
        in_chans=in_chans,
        parallel_attention=ckpt_args.get('parallel_attention', True),
        use_se_module=ckpt_args.get('use_se_module', False),
        use_sam_module=ckpt_args.get('use_sam_module', -1),
        reverse_se_sam=ckpt_args.get('reverse_se', False),
        sam_norm_type=ckpt_args.get('sam_norm_type', 0),
        **model_kwargs,
    )
    model.to(device=device, dtype=model_dtype)
    return model


def load_model_with_fedavg(model, backbone_path, fedavg_state):
    """Load backbone checkpoint, then overlay FedAvg-trained parameters."""
    if backbone_path:
        _logger.info(f'Loading backbone: {backbone_path}')
        ckpt = torch.load(backbone_path, map_location='cpu')
        sd = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            _logger.info(f'  Missing keys: {len(missing)}')
        if unexpected:
            _logger.info(f'  Unexpected keys: {len(unexpected)}')

    n_loaded = 0
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in fedavg_state:
                param.data.copy_(fedavg_state[name])
                n_loaded += 1
    _logger.info(f'Overlaid {n_loaded} FedAvg-trained parameters')


@torch.no_grad()
def evaluate(model, loader, device, model_dtype, no_prefetcher):
    model.eval()
    top1_m = utils.AverageMeter()
    top5_m = utils.AverageMeter()
    loss_m = utils.AverageMeter()

    for images, target in loader:
        if no_prefetcher:
            images = images.to(device=device, dtype=model_dtype)
            target = target.to(device=device)

        output = model(images)
        if isinstance(output, (tuple, list)):
            output = output[0]
        loss = F.cross_entropy(output, target)
        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))

        loss_m.update(loss.item(), images.shape[0])
        top1_m.update(acc1.item(), images.shape[0])
        top5_m.update(acc5.item(), images.shape[0])

    return loss_m.avg, top1_m.avg, top5_m.avg


def main():
    utils.setup_default_logging()

    parser = argparse.ArgumentParser(description='Evaluate FedAvg checkpoint on ImageNet-C')
    parser.add_argument('--fedavg-ckpt', type=str, required=True,
                        help='Path to best.pth from FedAvg training')
    parser.add_argument('--resume', type=str, default=None,
                        help='Override backbone checkpoint path (default: from ckpt args)')
    parser.add_argument('--data-root', type=str, default='/home/oem/servers/imagenet-c')
    parser.add_argument('--severity', type=int, default=5)
    parser.add_argument('--val-split', type=str, default='validation')
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--no-prefetcher', action='store_true', default=False)
    parser.add_argument('--pin-mem', action='store_true', default=False)
    parser.add_argument('--output-csv', type=str, default=None,
                        help='Output CSV path (default: <ckpt_dir>/eval_corruptions.csv)')
    parser.add_argument('--device-modules', default=None, type=str, nargs='+')
    args = parser.parse_args()

    if args.device_modules:
        for mod in args.device_modules:
            importlib.import_module(mod)

    device = torch.device(args.device)

    _logger.info(f'Loading FedAvg checkpoint: {args.fedavg_ckpt}')
    fedavg_ckpt = torch.load(args.fedavg_ckpt, map_location='cpu')
    ckpt_args = fedavg_ckpt['args']
    _logger.info(f'  Round {fedavg_ckpt["round"]}, Mean Acc@1: {fedavg_ckpt["mean_acc1"]:.3f}%')
    _logger.info(f'  Model: {ckpt_args["model"]}, train_mode: {ckpt_args.get("train_mode", "?")}')

    model = create_model_from_ckpt_args(ckpt_args, device)
    resume_path = args.resume or ckpt_args.get('resume')
    load_model_with_fedavg(model, resume_path, fedavg_ckpt['state_dict'])
    model.eval()

    input_size = ckpt_args.get('input_size', [3, 224, 224])
    data_config = resolve_data_config(
        {'input_size': input_size, 'pretrained': False}, model=model, verbose=True)

    if args.output_csv is None:
        ckpt_dir = os.path.dirname(os.path.abspath(args.fedavg_ckpt))
        args.output_csv = os.path.join(ckpt_dir, 'eval_corruptions.csv')

    _logger.info(f'\nEvaluating {len(CORRUPTIONS)} corruptions (severity={args.severity})')
    results = OrderedDict()
    acc1_sum = 0.0

    for corruption in CORRUPTIONS:
        data_dir = os.path.join(args.data_root, corruption, str(args.severity))
        if not os.path.isdir(data_dir):
            _logger.warning(f'  {data_dir} not found, skipping.')
            continue

        img_mode = 'RGB' if data_config['input_size'][0] == 3 else 'L'
        dataset = create_dataset(
            '', root=data_dir, split=args.val_split, is_training=False,
            batch_size=args.batch_size, input_img_mode=img_mode,
        )
        loader = create_loader(
            dataset, input_size=data_config['input_size'],
            batch_size=args.batch_size, is_training=False,
            interpolation=data_config['interpolation'],
            num_workers=args.workers, crop_pct=data_config['crop_pct'],
            mean=data_config['mean'], std=data_config['std'],
            pin_memory=args.pin_mem,
            device=device, distributed=False,
            use_prefetcher=not args.no_prefetcher,
        )

        loss, acc1, acc5 = evaluate(model, loader, device, None, args.no_prefetcher)
        results[corruption] = {'loss': loss, 'top1': acc1, 'top5': acc5}
        acc1_sum += acc1
        _logger.info(f'  {corruption:<22s}  Acc@1={acc1:.3f}%  Acc@5={acc5:.3f}%')

    mean_acc1 = acc1_sum / len(results) if results else 0.0
    _logger.info(f'\nMean Acc@1: {mean_acc1:.3f}%')

    with open(args.output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['corruption', 'top1', 'top5', 'loss'])
        for c, r in results.items():
            writer.writerow([c, f'{r["top1"]:.3f}', f'{r["top5"]:.3f}', f'{r["loss"]:.4f}'])
        writer.writerow(['mean', f'{mean_acc1:.3f}', '', ''])

    _logger.info(f'Results saved to {args.output_csv}')


if __name__ == '__main__':
    main()

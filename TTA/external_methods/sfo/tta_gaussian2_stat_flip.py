#!/usr/bin/env python3
"""Prediction Flip Analysis: With vs Without Parallel Attention

For the same checkpoint, compares predictions with attention enabled (mask*attn*x)
vs disabled (identity, x passed through unchanged). Reports:

  - Overall accuracy for both modes
  - Samples where predictions agree (X==Y) vs disagree (X!=Y)
  - Per-group accuracy breakdown (correct/incorrect for each mode)
  - Flip direction analysis (attention helps vs hurts)

Usage:
  CUDA_VISIBLE_DEVICES=0 python tta_gaussian2_stat_flip.py \
      --data-dir /home/oem/servers/imagenet-c/gaussian_noise/5 \
      --model vit_base_patch16_224 --num-classes 1000 \
      --input-size 3 224 224 --batch-size 64 \
      --resume ./VIT_IMG_PAR/.../model_best.pth.tar \
      --parallel-attention --train-mode 1 --vit-kernel-size 1 --vit-last
"""
import argparse
import importlib
import logging
import os
from contextlib import suppress
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import yaml

from timm import utils
from timm.data import create_dataset, create_loader, resolve_data_config
from timm.layers import set_fast_norm
from timm.models import create_model, safe_model_name, resume_checkpoint

_logger = logging.getLogger('tta_flip')


config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')

parser = argparse.ArgumentParser(description='Prediction Flip Analysis: Attention ON vs OFF')

group = parser.add_argument_group('Dataset parameters')
parser.add_argument('data', nargs='?', metavar='DIR', const=None)
group.add_argument('--data-dir', metavar='DIR', help='path to dataset')
group.add_argument('--dataset', metavar='NAME', default='')
group.add_argument('--train-split', metavar='NAME', default='train')
group.add_argument('--val-split', metavar='NAME', default='validation')
group.add_argument('--dataset-download', action='store_true', default=False)
group.add_argument('--class-map', default='', type=str, metavar='FILENAME')
group.add_argument('--input-img-mode', default=None, type=str)
group.add_argument('--input-key', default=None, type=str)
group.add_argument('--target-key', default=None, type=str)

group = parser.add_argument_group('Model parameters')
group.add_argument('--model', default='resnet26', type=str, metavar='MODEL')
group.add_argument('--pretrained', action='store_true', default=False)
group.add_argument('--pretrained-path', default=None, type=str)
group.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH')
group.add_argument('--resume', default='', type=str, metavar='PATH')
group.add_argument('--no-resume-opt', action='store_true', default=False)
group.add_argument('--num-classes', type=int, default=None, metavar='N')
group.add_argument('--gp', default=None, type=str, metavar='POOL')
group.add_argument('--img-size', type=int, default=None, metavar='N')
group.add_argument('--in-chans', type=int, default=None, metavar='N')
group.add_argument('--input-size', default=None, nargs=3, type=int, metavar='N')
group.add_argument('--crop-pct', default=None, type=float, metavar='N')
group.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN')
group.add_argument('--std', type=float, nargs='+', default=None, metavar='STD')
group.add_argument('--interpolation', default='', type=str, metavar='NAME')
group.add_argument('-b', '--batch-size', type=int, default=128, metavar='N')
group.add_argument('-vb', '--validation-batch-size', type=int, default=None, metavar='N')
group.add_argument('--channels-last', action='store_true', default=False)
group.add_argument('--fuser', default='', type=str)
group.add_argument('--fast-norm', default=False, action='store_true')
group.add_argument('--model-kwargs', nargs='*', default={}, action=utils.ParseKwargs)

scripting_group = group.add_mutually_exclusive_group()
scripting_group.add_argument('--torchscript', dest='torchscript', action='store_true')
scripting_group.add_argument('--torchcompile', nargs='?', type=str, default=None, const='inductor')

group = parser.add_argument_group('Device parameters')
group.add_argument('--device', default='cuda', type=str)
group.add_argument('--amp', action='store_true', default=False)
group.add_argument('--amp-dtype', default='float16', type=str)
group.add_argument('--model-dtype', default=None, type=str)
group.add_argument("--local_rank", default=0, type=int)
group.add_argument('--device-modules', default=None, type=str, nargs='+')

group = parser.add_argument_group('Miscellaneous parameters')
group.add_argument('--seed', type=int, default=42, metavar='S')
group.add_argument('-j', '--workers', type=int, default=4, metavar='N')
group.add_argument('--pin-mem', action='store_true', default=False)
group.add_argument('--no-prefetcher', action='store_true', default=False)
group.add_argument('--log-interval', type=int, default=50, metavar='N')

group = parser.add_argument_group('SEModule / Attention parameters')
group.add_argument('--use-se-module', action='store_true', default=False)
group.add_argument('--use-sam-module', type=int, default=-1)
group.add_argument('--reverse-se', action='store_true', default=False)
group.add_argument('--vit-early-norm-types', type=int, nargs=4, default=None, choices=[0, 1, 2, 3, 4])
group.add_argument('--vit-kernel-size', type=int, default=7)
group.add_argument('--spatial-group-size', type=int, default=1)
group.add_argument('--vit-last', action='store_true', default=False)
group.add_argument('--parallel-attention', action='store_true', default=False)
group.add_argument('--vit-closed', type=str, default=None, choices=['same', 'diff'])
group.add_argument('--train-mode', type=int, default=0, choices=[0, 1, 2])

group = parser.add_argument_group('Analysis parameters')
group.add_argument('--max-batches', type=int, default=0,
                   help='Max batches to process (0 = all)')

# Compatibility (unused but accepted)
group = parser.add_argument_group('Compatibility (unused)')
group.add_argument('--drop', type=float, default=0.0)
group.add_argument('--drop-path', type=float, default=None)
group.add_argument('--drop-block', type=float, default=None)
group.add_argument('--bn-momentum', type=float, default=None)
group.add_argument('--bn-eps', type=float, default=None)
group.add_argument('--lbatch', type=int, default=0)
group.add_argument('--epochs', type=int, default=300)
group.add_argument('--opt', default='sgd', type=str)
group.add_argument('--lr', type=float, default=None)
group.add_argument('--min-lr', type=float, default=0)
group.add_argument('--weight-decay', type=float, default=2e-5)
group.add_argument('--clip-grad', type=float, default=None)


def _parse_args():
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)
    args = parser.parse_args(remaining)
    return args


def _get_base_model(model):
    m = model
    if hasattr(m, 'module'):
        m = m.module
    if hasattr(m, '_orig_mod'):
        m = m._orig_mod
    return m


def disable_parallel_attention(model):
    """Monkey-patch _apply_parallel_attention to identity (return x unchanged).
    Returns the original methods for restoration."""
    base = _get_base_model(model)
    saved = {}

    if hasattr(base, '_apply_parallel_attention'):
        saved['_apply_parallel_attention'] = base._apply_parallel_attention
        base._apply_parallel_attention = lambda x: x

    if hasattr(base, '_apply_parallel_attention_last'):
        saved['_apply_parallel_attention_last'] = base._apply_parallel_attention_last
        base._apply_parallel_attention_last = lambda x: x

    return saved


def restore_parallel_attention(model, saved):
    """Restore original attention methods."""
    base = _get_base_model(model)
    for name, method in saved.items():
        setattr(base, name, method)


def main():
    utils.setup_default_logging()
    args = _parse_args()

    if args.device_modules:
        for module in args.device_modules:
            importlib.import_module(module)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    args.prefetcher = not args.no_prefetcher
    device = utils.init_distributed_device(args)
    _logger.info(f'Device: {device}')

    model_dtype = None
    if args.model_dtype:
        model_dtype = getattr(torch, args.model_dtype)

    utils.random_seed(args.seed, args.rank)
    if args.fast_norm:
        set_fast_norm()

    in_chans = 3
    if args.in_chans is not None:
        in_chans = args.in_chans
    elif args.input_size is not None:
        in_chans = args.input_size[0]

    factory_kwargs = {}
    if args.pretrained_path:
        factory_kwargs['pretrained_cfg_overlay'] = dict(
            file=args.pretrained_path, num_classes=-1,
        )

    vit_model = 'vit' in args.model.lower()
    vit_norm_kwargs = {}
    if args.vit_early_norm_types is not None:
        vit_norm_kwargs['vit_early_norm_types'] = args.vit_early_norm_types
    if args.use_sam_module != -1 or args.parallel_attention:
        vit_norm_kwargs['sam_kernel_size'] = args.vit_kernel_size
        vit_norm_kwargs['spatial_group_size'] = args.spatial_group_size
    if vit_model and args.vit_last:
        vit_norm_kwargs['vit_last'] = True
    if vit_model and args.vit_closed is not None:
        vit_norm_kwargs['vit_closed'] = args.vit_closed

    model = create_model(
        args.model,
        pretrained=args.pretrained,
        in_chans=in_chans,
        num_classes=args.num_classes,
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=args.drop_block,
        global_pool=args.gp,
        bn_momentum=args.bn_momentum,
        bn_eps=args.bn_eps,
        scriptable=args.torchscript,
        checkpoint_path=args.initial_checkpoint,
        use_se_module=args.use_se_module,
        use_sam_module=args.use_sam_module,
        reverse_se_sam=args.reverse_se,
        parallel_attention=args.parallel_attention,
        **factory_kwargs,
        **vit_norm_kwargs,
        **args.model_kwargs,
    )
    model.to(device=device, dtype=model_dtype)

    if args.resume:
        resume_checkpoint(model, args.resume, optimizer=None, loss_scaler=None, log_info=True)

    _logger.info(f'Model {safe_model_name(args.model)} loaded, '
                 f'params: {sum(p.numel() for p in model.parameters()):,}')

    data_config = resolve_data_config(vars(args), model=model, verbose=True)

    if args.input_img_mode is None:
        input_img_mode = 'RGB' if data_config['input_size'][0] == 3 else 'L'
    else:
        input_img_mode = args.input_img_mode

    dataset = create_dataset(
        args.dataset,
        root=args.data_dir,
        split=args.val_split,
        is_training=False,
        class_map=args.class_map,
        download=args.dataset_download,
        batch_size=args.batch_size,
        input_img_mode=input_img_mode,
        input_key=args.input_key,
        target_key=args.target_key,
    )
    loader = create_loader(
        dataset,
        input_size=data_config['input_size'],
        batch_size=args.validation_batch_size or args.batch_size,
        is_training=False,
        interpolation=data_config['interpolation'],
        num_workers=args.workers,
        crop_pct=data_config['crop_pct'],
        mean=data_config['mean'],
        std=data_config['std'],
        pin_memory=args.pin_mem,
        img_dtype=model_dtype or torch.float32,
        device=device,
        distributed=args.distributed,
        use_prefetcher=not args.no_prefetcher,
    )

    # ── Run both modes ──
    model.eval()
    amp_autocast = suppress
    if args.amp:
        amp_dtype = torch.bfloat16 if args.amp_dtype == 'bfloat16' else torch.float16
        amp_autocast = partial(torch.autocast, device_type=device.type, dtype=amp_dtype)

    prefetcher = not args.no_prefetcher

    all_preds_on = []     # attention ON predictions
    all_preds_off = []    # attention OFF predictions
    all_targets = []
    all_conf_on = []      # attention ON confidence (max softmax prob)
    all_conf_off = []

    _logger.info(f'Running dual inference (attention ON / OFF) on: {args.data_dir}')

    with torch.no_grad():
        for batch_idx, (input_data, target) in enumerate(loader):
            if args.max_batches > 0 and batch_idx >= args.max_batches:
                break

            if not prefetcher:
                input_data = input_data.to(device=device, dtype=model_dtype)
                target = target.to(device=device)
            if args.channels_last:
                input_data = input_data.contiguous(memory_format=torch.channels_last)

            # --- Attention ON ---
            with amp_autocast():
                logits_on = model(input_data)
                if isinstance(logits_on, (tuple, list)):
                    logits_on = logits_on[0]

            probs_on = torch.softmax(logits_on, dim=1)
            conf_on, preds_on = probs_on.max(dim=1)

            # --- Attention OFF ---
            saved = disable_parallel_attention(model)
            with amp_autocast():
                logits_off = model(input_data)
                if isinstance(logits_off, (tuple, list)):
                    logits_off = logits_off[0]
            restore_parallel_attention(model, saved)

            probs_off = torch.softmax(logits_off, dim=1)
            conf_off, preds_off = probs_off.max(dim=1)

            all_preds_on.append(preds_on.cpu())
            all_preds_off.append(preds_off.cpu())
            all_targets.append(target.cpu())
            all_conf_on.append(conf_on.cpu())
            all_conf_off.append(conf_off.cpu())

            if batch_idx % args.log_interval == 0:
                n_done = (batch_idx + 1) * (args.validation_batch_size or args.batch_size)
                _logger.info(f'  Batch [{batch_idx:>4d}/{len(loader)}]  samples: ~{n_done}')

    preds_on = torch.cat(all_preds_on)      # (N,)
    preds_off = torch.cat(all_preds_off)     # (N,)
    targets = torch.cat(all_targets)         # (N,)
    conf_on = torch.cat(all_conf_on)         # (N,)
    conf_off = torch.cat(all_conf_off)       # (N,)

    N = len(targets)

    correct_on = (preds_on == targets)       # (N,) bool
    correct_off = (preds_off == targets)     # (N,) bool
    agree = (preds_on == preds_off)          # (N,) bool — same prediction
    disagree = ~agree

    n_agree = agree.sum().item()
    n_disagree = disagree.sum().item()

    acc_on = correct_on.float().mean().item() * 100
    acc_off = correct_off.float().mean().item() * 100

    # ── Print results ──
    W = 90
    print(f"\n{'='*W}")
    print(f"  Prediction Flip Analysis: Attention ON vs OFF")
    print(f"{'='*W}")
    print(f"  Dataset:   {args.data_dir}")
    print(f"  Samples:   {N}")
    print()
    print(f"  {'':30s} {'Attn ON':>12} {'Attn OFF':>12} {'Diff':>12}")
    print(f"  {'-'*30} {'-'*12} {'-'*12} {'-'*12}")
    print(f"  {'Overall Accuracy':30s} {acc_on:>11.3f}% {acc_off:>11.3f}% {acc_on-acc_off:>+11.3f}%")
    print(f"  {'Mean Confidence':30s} {conf_on.mean().item():>12.4f} {conf_off.mean().item():>12.4f} "
          f"{conf_on.mean().item()-conf_off.mean().item():>+12.4f}")

    # ── Agree vs Disagree split ──
    print(f"\n{'='*W}")
    print(f"  Prediction Agreement Analysis")
    print(f"{'='*W}")
    print(f"  Predictions agree   (ON == OFF): {n_agree:>7d} / {N}  ({n_agree/N*100:.2f}%)")
    print(f"  Predictions disagree (ON != OFF): {n_disagree:>7d} / {N}  ({n_disagree/N*100:.2f}%)")

    # Accuracy within agree group
    if n_agree > 0:
        acc_on_agree = correct_on[agree].float().mean().item() * 100
        acc_off_agree = correct_off[agree].float().mean().item() * 100
        conf_on_agree = conf_on[agree].mean().item()
        conf_off_agree = conf_off[agree].mean().item()
    else:
        acc_on_agree = acc_off_agree = conf_on_agree = conf_off_agree = 0

    if n_disagree > 0:
        acc_on_disagree = correct_on[disagree].float().mean().item() * 100
        acc_off_disagree = correct_off[disagree].float().mean().item() * 100
        conf_on_disagree = conf_on[disagree].mean().item()
        conf_off_disagree = conf_off[disagree].mean().item()
    else:
        acc_on_disagree = acc_off_disagree = conf_on_disagree = conf_off_disagree = 0

    print(f"\n  {'Group':<30s} {'Count':>7} {'Acc ON':>10} {'Acc OFF':>10} {'Δ Acc':>10} "
          f"{'Conf ON':>10} {'Conf OFF':>10}")
    print(f"  {'-'*30} {'-'*7} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    print(f"  {'Agree (ON==OFF)':30s} {n_agree:>7d} {acc_on_agree:>9.3f}% {acc_off_agree:>9.3f}% "
          f"{acc_on_agree-acc_off_agree:>+9.3f}% {conf_on_agree:>10.4f} {conf_off_agree:>10.4f}")
    print(f"  {'Disagree (ON!=OFF)':30s} {n_disagree:>7d} {acc_on_disagree:>9.3f}% {acc_off_disagree:>9.3f}% "
          f"{acc_on_disagree-acc_off_disagree:>+9.3f}% {conf_on_disagree:>10.4f} {conf_off_disagree:>10.4f}")

    # ── Detailed flip direction analysis ──
    print(f"\n{'='*W}")
    print(f"  Flip Direction Analysis (within disagreeing samples)")
    print(f"{'='*W}")

    if n_disagree > 0:
        # Among disagreeing samples, 4 categories:
        on_right_off_wrong = (correct_on & ~correct_off & disagree).sum().item()
        on_wrong_off_right = (~correct_on & correct_off & disagree).sum().item()
        both_wrong_diff = (~correct_on & ~correct_off & disagree).sum().item()
        both_right_diff = (correct_on & correct_off & disagree).sum().item()  # rare: different pred but both in top-1? impossible for single label

        print(f"  {'Category':<45s} {'Count':>7} {'% of Disagree':>14} {'% of Total':>12}")
        print(f"  {'-'*45} {'-'*7} {'-'*14} {'-'*12}")
        print(f"  {'Attn ON correct, OFF wrong (attn helps)':45s} {on_right_off_wrong:>7d} "
              f"{on_right_off_wrong/n_disagree*100:>13.2f}% {on_right_off_wrong/N*100:>11.3f}%")
        print(f"  {'Attn ON wrong, OFF correct (attn hurts)':45s} {on_wrong_off_right:>7d} "
              f"{on_wrong_off_right/n_disagree*100:>13.2f}% {on_wrong_off_right/N*100:>11.3f}%")
        print(f"  {'Both wrong, different prediction':45s} {both_wrong_diff:>7d} "
              f"{both_wrong_diff/n_disagree*100:>13.2f}% {both_wrong_diff/N*100:>11.3f}%")

        net_benefit = on_right_off_wrong - on_wrong_off_right
        print(f"\n  Net benefit of attention: {net_benefit:+d} samples "
              f"({net_benefit/N*100:+.3f}% of total)")
        if net_benefit > 0:
            print(f"  → Attention provides net POSITIVE effect")
        elif net_benefit < 0:
            print(f"  → Attention provides net NEGATIVE effect")
        else:
            print(f"  → Attention provides NEUTRAL effect")

        # Confidence analysis for flipped samples
        if on_right_off_wrong > 0:
            mask_helps = correct_on & ~correct_off & disagree
            c_on_helps = conf_on[mask_helps].mean().item()
            c_off_helps = conf_off[mask_helps].mean().item()
        else:
            c_on_helps = c_off_helps = 0
        if on_wrong_off_right > 0:
            mask_hurts = ~correct_on & correct_off & disagree
            c_on_hurts = conf_on[mask_hurts].mean().item()
            c_off_hurts = conf_off[mask_hurts].mean().item()
        else:
            c_on_hurts = c_off_hurts = 0

        print(f"\n  Confidence when attention flips prediction:")
        print(f"  {'':40s} {'Conf ON':>10} {'Conf OFF':>10}")
        print(f"  {'-'*40} {'-'*10} {'-'*10}")
        print(f"  {'Attn helps (ON correct)':40s} {c_on_helps:>10.4f} {c_off_helps:>10.4f}")
        print(f"  {'Attn hurts (OFF correct)':40s} {c_on_hurts:>10.4f} {c_off_hurts:>10.4f}")
    else:
        print("  No disagreeing predictions found.")

    # ── Agree group: both correct vs both wrong ──
    print(f"\n{'='*W}")
    print(f"  Agreement Group Breakdown")
    print(f"{'='*W}")

    if n_agree > 0:
        both_correct = (correct_on & correct_off & agree).sum().item()
        both_wrong = (~correct_on & ~correct_off & agree).sum().item()

        print(f"  {'Category':<40s} {'Count':>7} {'% of Agree':>12} {'% of Total':>12}")
        print(f"  {'-'*40} {'-'*7} {'-'*12} {'-'*12}")
        print(f"  {'Both correct (same right answer)':40s} {both_correct:>7d} "
              f"{both_correct/n_agree*100:>11.2f}% {both_correct/N*100:>11.3f}%")
        print(f"  {'Both wrong (same wrong answer)':40s} {both_wrong:>7d} "
              f"{both_wrong/n_agree*100:>11.2f}% {both_wrong/N*100:>11.3f}%")

        if both_wrong > 0:
            mask_bw = ~correct_on & ~correct_off & agree
            c_on_bw = conf_on[mask_bw].mean().item()
            c_off_bw = conf_off[mask_bw].mean().item()
            print(f"\n  When both wrong with same prediction:")
            print(f"    Mean confidence ON:  {c_on_bw:.4f}")
            print(f"    Mean confidence OFF: {c_off_bw:.4f}")

    # ── Summary ──
    print(f"\n{'='*W}")
    print(f"  Summary")
    print(f"{'='*W}")
    print(f"  Attention ON accuracy:   {acc_on:.3f}%")
    print(f"  Attention OFF accuracy:  {acc_off:.3f}%")
    print(f"  Accuracy difference:     {acc_on - acc_off:+.3f}%")
    print(f"  Prediction stability:    {n_agree/N*100:.2f}% agree")
    if n_disagree > 0:
        print(f"  Among flipped samples:   attn helps {on_right_off_wrong}, "
              f"attn hurts {on_wrong_off_right}, both wrong {both_wrong_diff}")

    print(f"\n{'='*W}")
    print(f"  Done.")
    print(f"{'='*W}")


if __name__ == '__main__':
    main()

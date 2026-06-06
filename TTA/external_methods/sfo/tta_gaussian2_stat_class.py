#!/usr/bin/env python3
"""Per-Class Channel Attention Statistics for Source Domain

Analyzes whether channel attention distributions differ significantly across classes
in the source (clean) domain. Runs inference on the source dataset, groups channel
attention outputs by predicted (or ground-truth) class, and reports:
  - Per-class channel attention mean/std (768-dim vector per class)
  - Between-class variance vs within-class variance (ANOVA-like)
  - Top channels with largest inter-class variation
  - One-way ANOVA F-test per channel across classes

Usage (same flags as tta_gaussian2_stat.py, but only needs --data-dir for source):
  CUDA_VISIBLE_DEVICES=0 python tta_gaussian2_stat_class.py \
      --data-dir /data/imagenet/imagenet/val \
      --model vit_base_patch16_224 --num-classes 1000 \
      --input-size 3 224 224 --batch-size 64 \
      --resume ./VIT_IMG_PAR/.../model_best.pth.tar \
      --parallel-attention --train-mode 1 --vit-kernel-size 1 --vit-last
"""
import argparse
import importlib
import json
import logging
import os
from collections import OrderedDict, defaultdict
from contextlib import suppress
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import yaml

try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except (ImportError, ValueError):
    HAS_SCIPY = False

from timm import utils
from timm.data import create_dataset, create_loader, resolve_data_config
from timm.layers import set_fast_norm
from timm.models import create_model, safe_model_name, resume_checkpoint

_logger = logging.getLogger('tta_stat_class')


config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')

parser = argparse.ArgumentParser(description='Per-Class Channel Attention Statistics')

group = parser.add_argument_group('Dataset parameters')
parser.add_argument('data', nargs='?', metavar='DIR', const=None)
group.add_argument('--data-dir', metavar='DIR', help='path to source domain dataset')
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
group.add_argument('--use-gt-label', action='store_true', default=False,
                   help='Group by ground-truth label instead of predicted class')
group.add_argument('--top-classes', type=int, default=20,
                   help='Number of classes to show in detailed reports')
group.add_argument('--top-channels', type=int, default=30,
                   help='Number of channels to show in per-channel reports')
group.add_argument('--min-samples', type=int, default=10,
                   help='Minimum samples per class to include in analysis')
group.add_argument('--save-stats', type=str, default=None,
                   help='Save per-class stats to this .pt file')

# Unused but kept for CLI compatibility with tta_gaussian2_stat.py
group = parser.add_argument_group('Compatibility (unused)')
group.add_argument('--drop', type=float, default=0.0)
group.add_argument('--drop-path', type=float, default=None)
group.add_argument('--drop-block', type=float, default=None)
group.add_argument('--bn-momentum', type=float, default=None)
group.add_argument('--bn-eps', type=float, default=None)


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


class AttentionCollectorBatch:
    """Collects channel attention per batch, returns and clears each time."""

    def __init__(self):
        self._ca_batch = None
        self._hooks = []

    def register(self, model):
        base = _get_base_model(model)
        if hasattr(base, 'channel_attn'):
            self._hooks.append(base.channel_attn.register_forward_hook(self._hook))
        if hasattr(base, 'channel_attn_last'):
            self._hooks.append(base.channel_attn_last.register_forward_hook(self._hook))

    def _hook(self, module, input, output):
        self._ca_batch = output.detach().cpu()

    def pop(self):
        """Return the last batch's channel attention and reset."""
        ca = self._ca_batch
        self._ca_batch = None
        return ca

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


def _oneway_anova_f(groups):
    """One-way ANOVA F-statistic. groups: list of 1-D numpy arrays."""
    all_vals = np.concatenate(groups)
    grand_mean = np.mean(all_vals)
    N = len(all_vals)
    k = len(groups)

    ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
    ss_within = sum(np.sum((g - np.mean(g)) ** 2) for g in groups)

    if k <= 1 or ss_within < 1e-15:
        return 0.0, 1.0

    df_between = k - 1
    df_within = N - k

    ms_between = ss_between / df_between
    ms_within = ss_within / df_within

    f_stat = ms_between / ms_within if ms_within > 1e-15 else 0.0

    if HAS_SCIPY:
        p_val = 1.0 - scipy_stats.f.cdf(f_stat, df_between, df_within)
    else:
        p_val = float('nan')

    return f_stat, p_val


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

    # ── Collect per-sample channel attention grouped by class ──
    collector = AttentionCollectorBatch()
    collector.register(model)
    model.eval()

    amp_autocast = suppress
    if args.amp:
        amp_dtype = torch.bfloat16 if args.amp_dtype == 'bfloat16' else torch.float16
        amp_autocast = partial(torch.autocast, device_type=device.type, dtype=amp_dtype)

    prefetcher = not args.no_prefetcher
    # class_id → list of (C,) tensors (one per sample)
    class_ca = defaultdict(list)
    top1_m = utils.AverageMeter()
    top5_m = utils.AverageMeter()

    _logger.info(f'Collecting channel attention per class from: {args.data_dir}')
    _logger.info(f'Grouping by: {"ground-truth label" if args.use_gt_label else "predicted class"}')

    with torch.no_grad():
        for batch_idx, (input_data, target) in enumerate(loader):
            if args.max_batches > 0 and batch_idx >= args.max_batches:
                break

            if not prefetcher:
                input_data = input_data.to(device=device, dtype=model_dtype)
                target = target.to(device=device)
            if args.channels_last:
                input_data = input_data.contiguous(memory_format=torch.channels_last)

            with amp_autocast():
                output = model(input_data)
                if isinstance(output, (tuple, list)):
                    output = output[0]

            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            top1_m.update(acc1.item(), output.shape[0])
            top5_m.update(acc5.item(), output.shape[0])

            ca_batch = collector.pop()  # (B, C, 1, 1)
            if ca_batch is None:
                continue
            ca_batch = ca_batch.squeeze(-1).squeeze(-1)  # (B, C)

            if args.use_gt_label:
                labels = target.cpu()
            else:
                labels = output.argmax(dim=1).cpu()

            for i in range(ca_batch.shape[0]):
                cls = labels[i].item()
                class_ca[cls].append(ca_batch[i])  # (C,)

            if batch_idx % args.log_interval == 0:
                _logger.info(f'  Batch [{batch_idx:>4d}/{len(loader)}]  '
                             f'Acc@1: {top1_m.avg:.3f}  classes so far: {len(class_ca)}')

    collector.remove()

    _logger.info(f'  Final — Acc@1: {top1_m.avg:.3f}  Acc@5: {top5_m.avg:.3f}')
    _logger.info(f'  Classes with data: {len(class_ca)}')

    # ── Filter classes with enough samples ──
    valid_classes = {c: torch.stack(vs) for c, vs in class_ca.items()
                     if len(vs) >= args.min_samples}
    _logger.info(f'  Classes with >= {args.min_samples} samples: {len(valid_classes)}')

    if not valid_classes:
        print("No classes with sufficient samples. Try lowering --min-samples.")
        return

    num_channels = next(iter(valid_classes.values())).shape[1]

    # ── Compute per-class mean and std ──
    # class_means: {class_id: (C,) numpy}, class_stds: {class_id: (C,) numpy}
    class_means = {}
    class_stds = {}
    class_counts = {}
    for cls, tensor in valid_classes.items():
        t = tensor.float().numpy()  # (N_cls, C)
        class_means[cls] = np.mean(t, axis=0)
        class_stds[cls] = np.std(t, axis=0)
        class_counts[cls] = t.shape[0]

    # ── Global mean (across all samples) ──
    all_means = np.stack(list(class_means.values()))  # (num_classes, C)
    global_mean = np.mean(all_means, axis=0)  # (C,)

    # ── Between-class variance vs within-class variance per channel ──
    n_classes = len(valid_classes)
    between_var = np.zeros(num_channels)
    within_var = np.zeros(num_channels)

    for cls in valid_classes:
        n_c = class_counts[cls]
        between_var += n_c * (class_means[cls] - global_mean) ** 2
        within_var += (n_c - 1) * class_stds[cls] ** 2

    total_n = sum(class_counts.values())
    between_var /= (total_n - 1)
    within_var /= (total_n - n_classes) if total_n > n_classes else 1.0
    variance_ratio = between_var / (within_var + 1e-12)

    # ── Print overall summary ──
    print(f"\n{'='*90}")
    print(f"  Per-Class Channel Attention Analysis — Source Domain")
    print(f"{'='*90}")
    print(f"  Dataset:    {args.data_dir}")
    print(f"  Acc@1:      {top1_m.avg:.3f}")
    print(f"  Classes:    {n_classes} (with >= {args.min_samples} samples)")
    print(f"  Total samples: {total_n}")
    print(f"  Channels:   {num_channels}")
    print(f"  Group by:   {'ground-truth' if args.use_gt_label else 'predicted'}")

    # ── Variance ratio analysis ──
    print(f"\n{'='*90}")
    print(f"  Between-class vs Within-class Variance Ratio (top {args.top_channels} channels)")
    print(f"{'='*90}")
    print(f"  {'Ch':>5} {'Between Var':>14} {'Within Var':>14} {'Ratio':>12} {'Global Mean':>14}")
    print(f"  {'-'*5} {'-'*14} {'-'*14} {'-'*12} {'-'*14}")

    sorted_by_ratio = np.argsort(-variance_ratio)
    for rank, c in enumerate(sorted_by_ratio[:args.top_channels]):
        print(f"  {c:>5d} {between_var[c]:>14.6f} {within_var[c]:>14.6f} "
              f"{variance_ratio[c]:>12.4f} {global_mean[c]:>14.6f}")

    low_ratio = variance_ratio[sorted_by_ratio[-1]]
    high_ratio = variance_ratio[sorted_by_ratio[0]]
    median_ratio = np.median(variance_ratio)
    print(f"\n  Variance ratio summary:")
    print(f"    Max:    {high_ratio:.4f}  (Ch {sorted_by_ratio[0]})")
    print(f"    Median: {median_ratio:.4f}")
    print(f"    Min:    {low_ratio:.4f}  (Ch {sorted_by_ratio[-1]})")
    print(f"    Channels with ratio > 1.0: {np.sum(variance_ratio > 1.0)}/{num_channels}")
    print(f"    Channels with ratio > 0.5: {np.sum(variance_ratio > 0.5)}/{num_channels}")
    print(f"    Channels with ratio > 0.1: {np.sum(variance_ratio > 0.1)}/{num_channels}")

    # ── Per-channel ANOVA F-test ──
    print(f"\n{'='*90}")
    print(f"  Per-Channel One-Way ANOVA (top {args.top_channels} by F-statistic)")
    print(f"{'='*90}")
    print(f"  {'Ch':>5} {'F-stat':>14} {'p-value':>14} {'Var Ratio':>12} {'Global Mean':>14}")
    print(f"  {'-'*5} {'-'*14} {'-'*14} {'-'*12} {'-'*14}")

    anova_results = []
    for c in range(num_channels):
        groups = [valid_classes[cls][:, c].float().numpy() for cls in valid_classes]
        f_stat, p_val = _oneway_anova_f(groups)
        anova_results.append((c, f_stat, p_val))

    anova_results.sort(key=lambda x: x[1], reverse=True)
    for c, f_stat, p_val in anova_results[:args.top_channels]:
        sig = '***' if p_val < 0.001 else ('**' if p_val < 0.01 else ('*' if p_val < 0.05 else ''))
        p_str = f'{p_val:.2e}' if not np.isnan(p_val) else 'N/A (no scipy)'
        print(f"  {c:>5d} {f_stat:>14.2f} {p_str:>14} {variance_ratio[c]:>12.4f} "
              f"{global_mean[c]:>14.6f} {sig}")

    n_sig = sum(1 for _, _, p in anova_results if p < 0.05)
    n_highly_sig = sum(1 for _, _, p in anova_results if p < 0.001)
    n_not_sig = sum(1 for _, _, p in anova_results if p >= 0.05)
    print(f"\n  ANOVA summary:")
    print(f"    Significant (p<0.05):        {n_sig}/{num_channels}")
    print(f"    Highly significant (p<0.001): {n_highly_sig}/{num_channels}")
    print(f"    Not significant (p>=0.05):    {n_not_sig}/{num_channels}")

    # ── Classes with most distinct attention profiles ──
    # Distance of each class mean from global mean (L2 over channels)
    print(f"\n{'='*90}")
    print(f"  Classes with Most Distinct Channel Attention (top {args.top_classes})")
    print(f"{'='*90}")

    class_distances = []
    for cls in valid_classes:
        dist = np.linalg.norm(class_means[cls] - global_mean)
        class_distances.append((cls, dist, class_counts[cls]))
    class_distances.sort(key=lambda x: x[1], reverse=True)

    print(f"  {'Class':>7} {'L2 Dist':>12} {'Samples':>10} {'Mean Attn':>12} {'Std Attn':>12}")
    print(f"  {'-'*7} {'-'*12} {'-'*10} {'-'*12} {'-'*12}")
    for cls, dist, cnt in class_distances[:args.top_classes]:
        m = np.mean(class_means[cls])
        s = np.mean(class_stds[cls])
        print(f"  {cls:>7d} {dist:>12.4f} {cnt:>10d} {m:>12.6f} {s:>12.6f}")

    # ── Classes with most similar attention (smallest distance) ──
    print(f"\n{'='*90}")
    print(f"  Classes with Most Similar Channel Attention (bottom {args.top_classes})")
    print(f"{'='*90}")
    print(f"  {'Class':>7} {'L2 Dist':>12} {'Samples':>10} {'Mean Attn':>12} {'Std Attn':>12}")
    print(f"  {'-'*7} {'-'*12} {'-'*10} {'-'*12} {'-'*12}")
    for cls, dist, cnt in class_distances[-args.top_classes:]:
        m = np.mean(class_means[cls])
        s = np.mean(class_stds[cls])
        print(f"  {cls:>7d} {dist:>12.4f} {cnt:>10d} {m:>12.6f} {s:>12.6f}")

    # ── Overall conclusion ──
    pct_sig = n_sig / num_channels * 100 if num_channels > 0 else 0
    mean_ratio = np.mean(variance_ratio)

    print(f"\n{'='*90}")
    print(f"  Conclusion")
    print(f"{'='*90}")
    if pct_sig > 80 and mean_ratio > 0.3:
        print(f"  Channel attention shows STRONG class-dependent variation.")
        print(f"  {pct_sig:.1f}% of channels are significant, mean var ratio = {mean_ratio:.4f}.")
        print(f"  → Per-class adaptation or class-conditional correction is justified.")
    elif pct_sig > 50 or mean_ratio > 0.1:
        print(f"  Channel attention shows MODERATE class-dependent variation.")
        print(f"  {pct_sig:.1f}% of channels are significant, mean var ratio = {mean_ratio:.4f}.")
        print(f"  → Class-agnostic global correction may still be effective,")
        print(f"    but per-class refinement could help for outlier classes.")
    else:
        print(f"  Channel attention shows WEAK class-dependent variation.")
        print(f"  {pct_sig:.1f}% of channels are significant, mean var ratio = {mean_ratio:.4f}.")
        print(f"  → Class-agnostic global correction should be sufficient.")
        print(f"    Per-class adaptation is unlikely to provide significant benefit.")

    # ── Save stats ──
    if args.save_stats:
        save_data = {
            'global_mean': torch.from_numpy(global_mean),
            'variance_ratio': torch.from_numpy(variance_ratio),
            'between_var': torch.from_numpy(between_var),
            'within_var': torch.from_numpy(within_var),
            'class_means': {cls: torch.from_numpy(m) for cls, m in class_means.items()},
            'class_stds': {cls: torch.from_numpy(s) for cls, s in class_stds.items()},
            'class_counts': class_counts,
            'anova_f': {c: f for c, f, _ in anova_results},
            'anova_p': {c: p for c, _, p in anova_results},
        }
        torch.save(save_data, args.save_stats)
        _logger.info(f'Per-class stats saved to {args.save_stats}')

    print(f"\n{'='*90}")
    print(f"  Done.")
    print(f"{'='*90}")


if __name__ == '__main__':
    main()

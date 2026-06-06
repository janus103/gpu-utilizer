#!/usr/bin/env python3
"""Channel Attention Statistics: Target Domain vs Source Domain

Compares the distribution of ChannelAttention (SE-like) and SpatialAttention outputs
between a target domain (--data-dir, e.g. ImageNet-C) and a source domain
(--data-dir_source, e.g. clean ImageNet val).

For each domain, the script runs inference through the model and collects:
  - ChannelAttention output: (B, C, 1, 1)  per-channel attention weights
  - SpatialAttention mask:   (B, C, H, W)  spatial mask values

Then prints per-channel and global statistics (mean, std, median, Q1-Q4, min, max)
for both domains, their differences, and runs a Welch t-test to assess significance.
"""
import argparse
import importlib
import json
import logging
import math
import os
from collections import OrderedDict
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
from timm.models import create_model, safe_model_name, resume_checkpoint, load_checkpoint

_logger = logging.getLogger('tta_stat')


config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')

parser = argparse.ArgumentParser(description='Channel Attention Statistics: Target vs Source Domain')

group = parser.add_argument_group('Dataset parameters')
parser.add_argument('data', nargs='?', metavar='DIR', const=None,
                    help='path to dataset (positional is *deprecated*, use --data-dir)')
group.add_argument('--data-dir', metavar='DIR',
                    help='path to target domain dataset')
group.add_argument('--data-dir_source', metavar='DIR', default=None,
                    help='path to source domain dataset')
group.add_argument('--dataset', metavar='NAME', default='',
                    help='dataset type + name')
group.add_argument('--train-split', metavar='NAME', default='train',
                   help='dataset train split (default: train)')
group.add_argument('--val-split', metavar='NAME', default='validation',
                   help='dataset validation split (default: validation)')
group.add_argument('--train-num-samples', default=None, type=int, metavar='N')
group.add_argument('--val-num-samples', default=None, type=int, metavar='N')
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
group.add_argument('--grad-accum-steps', type=int, default=1, metavar='N')
group.add_argument('--grad-checkpointing', action='store_true', default=False)
group.add_argument('--fast-norm', default=False, action='store_true')
group.add_argument('--model-kwargs', nargs='*', default={}, action=utils.ParseKwargs)
group.add_argument('--head-init-scale', default=None, type=float)
group.add_argument('--head-init-bias', default=None, type=float)

scripting_group = group.add_mutually_exclusive_group()
scripting_group.add_argument('--torchscript', dest='torchscript', action='store_true')
scripting_group.add_argument('--torchcompile', nargs='?', type=str, default=None, const='inductor')

group = parser.add_argument_group('Device parameters')
group.add_argument('--device', default='cuda', type=str)
group.add_argument('--amp', action='store_true', default=False)
group.add_argument('--amp-dtype', default='float16', type=str)
group.add_argument('--model-dtype', default=None, type=str)
group.add_argument('--no-ddp-bb', action='store_true', default=False)
group.add_argument('--synchronize-step', action='store_true', default=False)
group.add_argument("--local_rank", default=0, type=int)
group.add_argument('--device-modules', default=None, type=str, nargs='+')

group = parser.add_argument_group('Optimizer parameters')
group.add_argument('--opt', default='sgd', type=str, metavar='OPTIMIZER')
group.add_argument('--opt-eps', default=None, type=float, metavar='EPSILON')
group.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA')
group.add_argument('--momentum', type=float, default=0.9, metavar='M')
group.add_argument('--weight-decay', type=float, default=2e-5)
group.add_argument('--clip-grad', type=float, default=None, metavar='NORM')
group.add_argument('--clip-mode', type=str, default='norm')
group.add_argument('--layer-decay', type=float, default=None)
group.add_argument('--opt-kwargs', nargs='*', default={}, action=utils.ParseKwargs)

group = parser.add_argument_group('Learning rate schedule parameters')
group.add_argument('--sched', type=str, default='cosine', metavar='SCHEDULER')
group.add_argument('--sched-on-updates', action='store_true', default=False)
group.add_argument('--lr', type=float, default=None, metavar='LR')
group.add_argument('--lr-base', type=float, default=0.1, metavar='LR')
group.add_argument('--lr-base-size', type=int, default=256, metavar='DIV')
group.add_argument('--lr-base-scale', type=str, default='', metavar='SCALE')
group.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct')
group.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT')
group.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV')
group.add_argument('--lr-cycle-mul', type=float, default=1.0, metavar='MULT')
group.add_argument('--lr-cycle-decay', type=float, default=0.5, metavar='MULT')
group.add_argument('--lr-cycle-limit', type=int, default=1, metavar='N')
group.add_argument('--lr-k-decay', type=float, default=1.0)
group.add_argument('--warmup-lr', type=float, default=1e-5, metavar='LR')
group.add_argument('--min-lr', type=float, default=0, metavar='LR')
group.add_argument('--epochs', type=int, default=300, metavar='N')
group.add_argument('--epoch-repeats', type=float, default=0., metavar='N')
group.add_argument('--start-epoch', default=None, type=int, metavar='N')
group.add_argument('--decay-milestones', default=[90, 180, 270], type=int, nargs='+', metavar="MILESTONES")
group.add_argument('--decay-epochs', type=float, default=90, metavar='N')
group.add_argument('--warmup-epochs', type=int, default=5, metavar='N')
group.add_argument('--warmup-prefix', action='store_true', default=False)
group.add_argument('--cooldown-epochs', type=int, default=0, metavar='N')
group.add_argument('--patience-epochs', type=int, default=10, metavar='N')
group.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE')

group = parser.add_argument_group('Augmentation and regularization parameters')
group.add_argument('--no-aug', action='store_true', default=False)
group.add_argument('--train-crop-mode', type=str, default=None)
group.add_argument('--scale', type=float, nargs='+', default=[0.08, 1.0], metavar='PCT')
group.add_argument('--ratio', type=float, nargs='+', default=[3. / 4., 4. / 3.], metavar='RATIO')
group.add_argument('--hflip', type=float, default=0.5)
group.add_argument('--vflip', type=float, default=0.)
group.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT')
group.add_argument('--color-jitter-prob', type=float, default=None, metavar='PCT')
group.add_argument('--grayscale-prob', type=float, default=None, metavar='PCT')
group.add_argument('--gaussian-blur-prob', type=float, default=None, metavar='PCT')
group.add_argument('--aa', type=str, default=None, metavar='NAME')
group.add_argument('--aug-repeats', type=float, default=0)
group.add_argument('--aug-splits', type=int, default=0)
group.add_argument('--jsd-loss', action='store_true', default=False)
group.add_argument('--bce-loss', action='store_true', default=False)
group.add_argument('--bce-sum', action='store_true', default=False)
group.add_argument('--bce-target-thresh', type=float, default=None)
group.add_argument('--bce-pos-weight', type=float, default=None)
group.add_argument('--reprob', type=float, default=0., metavar='PCT')
group.add_argument('--remode', type=str, default='pixel')
group.add_argument('--recount', type=int, default=1)
group.add_argument('--resplit', action='store_true', default=False)
group.add_argument('--mixup', type=float, default=0.0)
group.add_argument('--cutmix', type=float, default=0.0)
group.add_argument('--cutmix-minmax', type=float, nargs='+', default=None)
group.add_argument('--mixup-prob', type=float, default=1.0)
group.add_argument('--mixup-switch-prob', type=float, default=0.5)
group.add_argument('--mixup-mode', type=str, default='batch')
group.add_argument('--mixup-off-epoch', default=0, type=int, metavar='N')
group.add_argument('--smoothing', type=float, default=0.1)
group.add_argument('--train-interpolation', type=str, default='random')
group.add_argument('--drop', type=float, default=0.0, metavar='PCT')
group.add_argument('--drop-connect', type=float, default=None, metavar='PCT')
group.add_argument('--drop-path', type=float, default=None, metavar='PCT')
group.add_argument('--drop-block', type=float, default=None, metavar='PCT')

group = parser.add_argument_group('Batch norm parameters')
group.add_argument('--bn-momentum', type=float, default=None)
group.add_argument('--bn-eps', type=float, default=None)
group.add_argument('--sync-bn', action='store_true')
group.add_argument('--dist-bn', type=str, default='reduce')
group.add_argument('--split-bn', action='store_true')

group = parser.add_argument_group('Model exponential moving average parameters')
group.add_argument('--model-ema', action='store_true', default=False)
group.add_argument('--model-ema-force-cpu', action='store_true', default=False)
group.add_argument('--model-ema-decay', type=float, default=0.9998)
group.add_argument('--model-ema-warmup', action='store_true')

group = parser.add_argument_group('Miscellaneous parameters')
group.add_argument('--seed', type=int, default=42, metavar='S')
group.add_argument('--worker-seeding', type=str, default='all')
group.add_argument('--log-interval', type=int, default=50, metavar='N')
group.add_argument('--val-interval', type=int, default=1, metavar='N')
group.add_argument('--recovery-interval', type=int, default=0, metavar='N')
group.add_argument('--checkpoint-hist', type=int, default=10, metavar='N')
group.add_argument('-j', '--workers', type=int, default=4, metavar='N')
group.add_argument('--save-images', action='store_true', default=False)
group.add_argument('--pin-mem', action='store_true', default=False)
group.add_argument('--no-prefetcher', action='store_true', default=False)
group.add_argument('--output', default='', type=str, metavar='PATH')
group.add_argument('--experiment', default='', type=str, metavar='NAME')
group.add_argument('--eval-metric', default='top1', type=str, metavar='EVAL_METRIC')
group.add_argument('--tta', type=int, default=0, metavar='N')
group.add_argument('--use-multi-epochs-loader', action='store_true', default=False)
group.add_argument('--log-wandb', action='store_true', default=False)
group.add_argument('--wandb-project', default=None, type=str)

group = parser.add_argument_group('SEModule Gaussian parameters')
group.add_argument('--gaussian-loss-weight', type=float, default=1.0)
group.add_argument('--use-se-module', action='store_true', default=False)
group.add_argument('--use-sam-module', type=int, default=-1)
group.add_argument('--sam-loss-weight', type=float, default=1.0)
group.add_argument('--lbatch', type=int, default=0, metavar='M')
group.add_argument('--reverse-se', action='store_true', default=False)
group.add_argument('--vit-early-norm-types', type=int, nargs=4, default=None, choices=[0, 1, 2, 3, 4],
                   metavar=('PRE_B0', 'POST_B0', 'POST_B1', 'POST_B2'))
group.add_argument('--vit-kernel-size', type=int, default=7)
group.add_argument('--spatial-group-size', type=int, default=1)
group.add_argument('--vit-last', action='store_true', default=False)
group.add_argument('--parallel-attention', action='store_true', default=False)
group.add_argument('--vit-closed', type=str, default=None, choices=['same', 'diff'])
group.add_argument('--train-mode', type=int, default=0, choices=[0, 1, 2])

group = parser.add_argument_group('Statistics parameters')
group.add_argument('--max-batches', type=int, default=0,
                   help='Max batches to process per domain (0 = all)')
group.add_argument('--save-stats', type=str, default=None,
                   help='Save raw statistics to this JSON file')


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


class AttentionCollector:
    """Hook-based collector for ChannelAttention and SpatialAttention outputs."""

    def __init__(self):
        self.channel_attn_outputs = []
        self.spatial_mask_outputs = []
        self._hooks = []

    def register(self, model):
        base = _get_base_model(model)

        if hasattr(base, 'channel_attn'):
            h = base.channel_attn.register_forward_hook(self._channel_hook)
            self._hooks.append(h)

        if hasattr(base, 'spatial_attn'):
            h = base.spatial_attn.register_forward_hook(self._spatial_hook)
            self._hooks.append(h)

        if hasattr(base, 'channel_attn_last'):
            h = base.channel_attn_last.register_forward_hook(self._channel_hook)
            self._hooks.append(h)

        if hasattr(base, 'spatial_attn_last'):
            h = base.spatial_attn_last.register_forward_hook(self._spatial_hook)
            self._hooks.append(h)

    def _channel_hook(self, module, input, output):
        self.channel_attn_outputs.append(output.detach().cpu())

    def _spatial_hook(self, module, input, output):
        _, mask = output
        self.spatial_mask_outputs.append(mask.detach().cpu())

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def reset(self):
        self.channel_attn_outputs.clear()
        self.spatial_mask_outputs.clear()


def compute_statistics(tensor):
    """Compute descriptive statistics for a 1-D tensor."""
    t = tensor.float().numpy()
    q = np.percentile(t, [0, 25, 50, 75, 100])
    return OrderedDict([
        ('mean', float(np.mean(t))),
        ('std', float(np.std(t))),
        ('min', float(q[0])),
        ('Q1', float(q[1])),
        ('median', float(q[2])),
        ('Q3', float(q[3])),
        ('max', float(q[4])),
        ('count', len(t)),
    ])


def compute_per_channel_statistics(tensor_4d):
    """Compute per-channel statistics from (N, C, ...) tensor.
    Returns dict mapping channel_idx -> stats dict."""
    C = tensor_4d.shape[1]
    result = OrderedDict()
    for c in range(C):
        ch_vals = tensor_4d[:, c].reshape(-1)
        result[c] = compute_statistics(ch_vals)
    return result


def print_comparison_table(name, stats_target, stats_source):
    """Print a side-by-side comparison table."""
    keys = ['mean', 'std', 'min', 'Q1', 'median', 'Q3', 'max']
    print(f"\n{'='*80}")
    print(f"  {name} — Global Statistics Comparison")
    print(f"{'='*80}")
    print(f"  {'Stat':<10} {'Target':>14} {'Source':>14} {'Diff':>14} {'Diff%':>10}")
    print(f"  {'-'*10} {'-'*14} {'-'*14} {'-'*14} {'-'*10}")
    for k in keys:
        vt = stats_target[k]
        vs = stats_source[k]
        diff = vt - vs
        pct = (diff / abs(vs) * 100) if abs(vs) > 1e-12 else float('nan')
        print(f"  {k:<10} {vt:>14.6f} {vs:>14.6f} {diff:>+14.6f} {pct:>+9.2f}%")
    print(f"  {'count':<10} {stats_target['count']:>14d} {stats_source['count']:>14d}")


def print_per_channel_comparison(name, pc_target, pc_source, top_k=20):
    """Print per-channel comparison sorted by absolute mean difference."""
    channels = sorted(pc_target.keys())
    diffs = []
    for c in channels:
        mean_t = pc_target[c]['mean']
        mean_s = pc_source[c]['mean']
        diffs.append((c, mean_t, mean_s, mean_t - mean_s))

    diffs.sort(key=lambda x: abs(x[3]), reverse=True)

    print(f"\n{'='*80}")
    print(f"  {name} — Per-Channel Mean Comparison (top {top_k} by |diff|)")
    print(f"{'='*80}")
    print(f"  {'Ch':>5} {'Target Mean':>14} {'Source Mean':>14} {'Diff':>14} {'Tgt Std':>12} {'Src Std':>12}")
    print(f"  {'-'*5} {'-'*14} {'-'*14} {'-'*14} {'-'*12} {'-'*12}")
    for c, mt, ms, d in diffs[:top_k]:
        std_t = pc_target[c]['std']
        std_s = pc_source[c]['std']
        print(f"  {c:>5d} {mt:>14.6f} {ms:>14.6f} {d:>+14.6f} {std_t:>12.6f} {std_s:>12.6f}")


def _welch_ttest(a, b):
    """Welch's t-test (unequal variance) — pure numpy fallback."""
    n1, n2 = len(a), len(b)
    m1, m2 = np.mean(a), np.mean(b)
    v1, v2 = np.var(a, ddof=1), np.var(b, ddof=1)
    se = np.sqrt(v1 / n1 + v2 / n2)
    if se < 1e-15:
        return 0.0, 1.0
    t_stat = (m1 - m2) / se
    df_num = (v1 / n1 + v2 / n2) ** 2
    df_den = (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)
    df = df_num / df_den if df_den > 0 else 1.0
    # Approximate two-sided p-value using normal for large df
    if df > 100:
        from math import erfc, sqrt
        p_val = erfc(abs(t_stat) / sqrt(2))
    else:
        # Beta-function approximation for Student's t
        x = df / (df + t_stat ** 2)
        try:
            from math import lgamma, exp
            a_, b_ = df / 2, 0.5
            # Regularized incomplete beta via continued fraction (sufficient for p-value)
            log_beta = lgamma(a_) + lgamma(b_) - lgamma(a_ + b_)
            p_val = 1.0 - exp(-log_beta) * (x ** a_) * ((1 - x) ** b_) / a_
            p_val = max(0.0, min(1.0, 2 * min(p_val, 1 - p_val)))
        except Exception:
            from math import erfc, sqrt
            p_val = erfc(abs(t_stat) / sqrt(2))
    return t_stat, p_val


def _ttest_ind(a, b):
    """Welch's t-test dispatching to scipy when available."""
    if HAS_SCIPY:
        return scipy_stats.ttest_ind(a, b, equal_var=False)
    return _welch_ttest(a, b)


def _subsample(arr, max_n):
    if len(arr) > max_n:
        return arr[np.random.choice(len(arr), max_n, replace=False)]
    return arr


def run_significance_tests(name, values_target, values_source):
    """Run Welch's t-test on global distributions."""
    vt = values_target.float().numpy()
    vs = values_source.float().numpy()

    vt_sub = _subsample(vt, 100_000)
    vs_sub = _subsample(vs, 100_000)

    t_stat, t_pval = _ttest_ind(vt_sub, vs_sub)

    pooled_std = np.sqrt((np.var(vt_sub) + np.var(vs_sub)) / 2)
    d = (np.mean(vt_sub) - np.mean(vs_sub)) / pooled_std if pooled_std > 1e-12 else 0.0

    print(f"\n{'='*80}")
    print(f"  {name} — Significance Tests")
    print(f"{'='*80}")
    print(f"  Welch's t-test:  t = {t_stat:.4f},  p = {t_pval:.2e}")
    print(f"  Cohen's d (effect size): {d:.4f}")
    if t_pval < 0.001:
        print(f"  => HIGHLY significant (p < 0.001)")
    elif t_pval < 0.01:
        print(f"  => Significant (p < 0.01)")
    elif t_pval < 0.05:
        print(f"  => Marginally significant (p < 0.05)")
    else:
        print(f"  => NOT significant (p >= 0.05)")

    if abs(d) >= 0.8:
        print(f"  => Large effect size (|d| >= 0.8)")
    elif abs(d) >= 0.5:
        print(f"  => Medium effect size (|d| >= 0.5)")
    elif abs(d) >= 0.2:
        print(f"  => Small effect size (|d| >= 0.2)")
    else:
        print(f"  => Negligible effect size (|d| < 0.2)")


def run_per_channel_significance(name, tensor_target, tensor_source, top_k=20):
    """Run per-channel Welch's t-test and report channels with largest effect."""
    C = tensor_target.shape[1]
    results = []
    for c in range(C):
        vt = tensor_target[:, c].reshape(-1).float().numpy()
        vs = tensor_source[:, c].reshape(-1).float().numpy()

        vt = _subsample(vt, 50_000)
        vs = _subsample(vs, 50_000)

        t_stat, p_val = _ttest_ind(vt, vs)
        pooled_std = np.sqrt((np.var(vt) + np.var(vs)) / 2)
        d = (np.mean(vt) - np.mean(vs)) / pooled_std if pooled_std > 1e-12 else 0.0
        results.append((c, t_stat, p_val, d, np.mean(vt), np.mean(vs)))

    results.sort(key=lambda x: abs(x[3]), reverse=True)

    print(f"\n{'='*80}")
    print(f"  {name} — Per-Channel Significance (top {top_k} by |Cohen's d|)")
    print(f"{'='*80}")
    print(f"  {'Ch':>5} {'t-stat':>12} {'p-value':>14} {'Cohen d':>12} {'Tgt Mean':>12} {'Src Mean':>12}")
    print(f"  {'-'*5} {'-'*12} {'-'*14} {'-'*12} {'-'*12} {'-'*12}")
    for c, t, p, d, mt, ms in results[:top_k]:
        sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else ''))
        print(f"  {c:>5d} {t:>12.4f} {p:>14.2e} {d:>+12.4f} {mt:>12.6f} {ms:>12.6f} {sig}")

    n_sig = sum(1 for _, _, p, _, _, _ in results if p < 0.05)
    n_highly_sig = sum(1 for _, _, p, _, _, _ in results if p < 0.001)
    n_large_effect = sum(1 for _, _, _, d, _, _ in results if abs(d) >= 0.8)
    n_medium_effect = sum(1 for _, _, _, d, _, _ in results if 0.5 <= abs(d) < 0.8)
    print(f"\n  Summary: {n_sig}/{C} channels significant (p<0.05), "
          f"{n_highly_sig}/{C} highly significant (p<0.001)")
    print(f"  Effect sizes: {n_large_effect} large (|d|>=0.8), "
          f"{n_medium_effect} medium (0.5<=|d|<0.8)")


def create_eval_loader(data_dir, args, data_config, model_dtype, device):
    """Create a validation-mode data loader for the given data directory."""
    if args.input_img_mode is None:
        input_img_mode = 'RGB' if data_config['input_size'][0] == 3 else 'L'
    else:
        input_img_mode = args.input_img_mode

    dataset = create_dataset(
        args.dataset,
        root=data_dir,
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
    return loader


@torch.no_grad()
def collect_attention_stats(model, loader, collector, args, device, model_dtype=None):
    """Run inference and collect attention outputs via hooks."""
    model.eval()
    collector.reset()
    amp_autocast = suppress
    if args.amp:
        amp_dtype = torch.bfloat16 if args.amp_dtype == 'bfloat16' else torch.float16
        amp_autocast = partial(torch.autocast, device_type=device.type, dtype=amp_dtype)

    top1_m = utils.AverageMeter()
    top5_m = utils.AverageMeter()
    prefetcher = not args.no_prefetcher

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

        if batch_idx % args.log_interval == 0:
            _logger.info(f'  Batch [{batch_idx:>4d}/{len(loader)}]  '
                         f'Acc@1: {top1_m.avg:.3f}  Acc@5: {top5_m.avg:.3f}')

    _logger.info(f'  Final — Acc@1: {top1_m.avg:.3f}  Acc@5: {top5_m.avg:.3f}')

    channel_attn = None
    spatial_mask = None
    if collector.channel_attn_outputs:
        channel_attn = torch.cat(collector.channel_attn_outputs, dim=0)
    if collector.spatial_mask_outputs:
        spatial_mask = torch.cat(collector.spatial_mask_outputs, dim=0)

    return channel_attn, spatial_mask, top1_m.avg, top5_m.avg


def main():
    utils.setup_default_logging()
    args = _parse_args()

    if args.data_dir_source is None:
        raise ValueError('--data-dir_source is required (path to source/clean domain)')

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
            file=args.pretrained_path,
            num_classes=-1,
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
        resume_checkpoint(
            model,
            args.resume,
            optimizer=None,
            loss_scaler=None,
            log_info=True,
        )

    _logger.info(f'Model {safe_model_name(args.model)} loaded, '
                 f'param count: {sum(p.numel() for p in model.parameters()):,}')

    data_config = resolve_data_config(vars(args), model=model, verbose=True)

    # --- Collect for TARGET domain ---
    _logger.info(f'\n{"#"*80}')
    _logger.info(f'  Collecting attention stats for TARGET domain: {args.data_dir}')
    _logger.info(f'{"#"*80}')

    loader_target = create_eval_loader(args.data_dir, args, data_config, model_dtype, device)
    collector = AttentionCollector()
    collector.register(model)

    ca_target, sm_target, acc1_target, acc5_target = collect_attention_stats(
        model, loader_target, collector, args, device, model_dtype)
    collector.remove()

    # --- Collect for SOURCE domain ---
    _logger.info(f'\n{"#"*80}')
    _logger.info(f'  Collecting attention stats for SOURCE domain: {args.data_dir_source}')
    _logger.info(f'{"#"*80}')

    loader_source = create_eval_loader(args.data_dir_source, args, data_config, model_dtype, device)
    collector_src = AttentionCollector()
    collector_src.register(model)

    ca_source, sm_source, acc1_source, acc5_source = collect_attention_stats(
        model, loader_source, collector_src, args, device, model_dtype)
    collector_src.remove()

    # --- Print accuracy comparison ---
    print(f"\n{'='*80}")
    print(f"  Accuracy Comparison")
    print(f"{'='*80}")
    print(f"  Target domain:  Acc@1 = {acc1_target:.3f},  Acc@5 = {acc5_target:.3f}")
    print(f"  Source domain:  Acc@1 = {acc1_source:.3f},  Acc@5 = {acc5_source:.3f}")
    print(f"  Difference:     Acc@1 = {acc1_target - acc1_source:+.3f},  "
          f"Acc@5 = {acc5_target - acc5_source:+.3f}")

    # --- Channel Attention analysis ---
    if ca_target is not None and ca_source is not None:
        _logger.info(f'Channel Attention collected: target {ca_target.shape}, source {ca_source.shape}')

        ca_target_flat = ca_target.reshape(-1)
        ca_source_flat = ca_source.reshape(-1)

        stats_ca_target = compute_statistics(ca_target_flat)
        stats_ca_source = compute_statistics(ca_source_flat)
        print_comparison_table('Channel Attention (global)', stats_ca_target, stats_ca_source)

        pc_ca_target = compute_per_channel_statistics(ca_target)
        pc_ca_source = compute_per_channel_statistics(ca_source)
        print_per_channel_comparison('Channel Attention', pc_ca_target, pc_ca_source)

        run_significance_tests('Channel Attention (global)', ca_target_flat, ca_source_flat)
        run_per_channel_significance('Channel Attention', ca_target, ca_source)
    else:
        print("\n  [!] No Channel Attention outputs captured.")

    # --- Spatial Attention analysis ---
    if sm_target is not None and sm_source is not None:
        _logger.info(f'Spatial Mask collected: target {sm_target.shape}, source {sm_source.shape}')

        sm_target_flat = sm_target.reshape(-1)
        sm_source_flat = sm_source.reshape(-1)

        stats_sm_target = compute_statistics(sm_target_flat)
        stats_sm_source = compute_statistics(sm_source_flat)
        print_comparison_table('Spatial Attention Mask (global)', stats_sm_target, stats_sm_source)

        pc_sm_target = compute_per_channel_statistics(sm_target)
        pc_sm_source = compute_per_channel_statistics(sm_source)
        print_per_channel_comparison('Spatial Attention Mask', pc_sm_target, pc_sm_source)

        run_significance_tests('Spatial Attention Mask (global)', sm_target_flat, sm_source_flat)
        run_per_channel_significance('Spatial Attention Mask', sm_target, sm_source)

        pos_ratio_target = (sm_target > 0).float().mean().item()
        pos_ratio_source = (sm_source > 0).float().mean().item()
        print(f"\n{'='*80}")
        print(f"  Spatial Mask — Positive Ratio (mask > 0)")
        print(f"{'='*80}")
        print(f"  Target: {pos_ratio_target:.4f}  ({pos_ratio_target*100:.2f}%)")
        print(f"  Source: {pos_ratio_source:.4f}  ({pos_ratio_source*100:.2f}%)")
        print(f"  Diff:   {pos_ratio_target - pos_ratio_source:+.4f}")
    else:
        print("\n  [!] No Spatial Attention outputs captured.")

    # --- Combined mask*attn analysis ---
    if ca_target is not None and sm_target is not None and ca_source is not None and sm_source is not None:
        combined_target = (sm_target * ca_target).reshape(-1)
        combined_source = (sm_source * ca_source).reshape(-1)

        stats_comb_target = compute_statistics(combined_target)
        stats_comb_source = compute_statistics(combined_source)
        print_comparison_table('Combined (mask * channel_attn) (global)', stats_comb_target, stats_comb_source)
        run_significance_tests('Combined (mask * channel_attn)', combined_target, combined_source)

    # --- Save stats ---
    if args.save_stats:
        save_data = {
            'accuracy': {
                'target': {'top1': acc1_target, 'top5': acc5_target},
                'source': {'top1': acc1_source, 'top5': acc5_source},
            }
        }
        if ca_target is not None and ca_source is not None:
            save_data['channel_attention'] = {
                'target_global': stats_ca_target,
                'source_global': stats_ca_source,
            }
        if sm_target is not None and sm_source is not None:
            save_data['spatial_mask'] = {
                'target_global': stats_sm_target,
                'source_global': stats_sm_source,
            }
        with open(args.save_stats, 'w') as f:
            json.dump(save_data, f, indent=2)
        _logger.info(f'Statistics saved to {args.save_stats}')

    print(f"\n{'='*80}")
    print(f"  Done.")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()

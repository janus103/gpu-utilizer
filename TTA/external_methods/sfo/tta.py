#!/usr/bin/env python3
""" ImageNet Validation Script

This is intended to be a lean and easily modifiable ImageNet validation script for evaluating pretrained
models or training checkpoints against ImageNet or similarly organized image datasets. It prioritizes
canonical PyTorch, standard Python style, and good performance. Repurpose as you see fit.

Hacked together by Ross Wightman (https://github.com/rwightman)
"""
import argparse
import copy
import csv
import glob
import json
import logging
import os
import time
from collections import OrderedDict
from contextlib import suppress
from functools import partial

import torch
import torch.nn as nn
import torch.nn.parallel

from timm.data import create_dataset, create_loader, resolve_data_config, RealLabelsImagenet
from timm.layers import apply_test_time_pool, set_fast_norm
from timm.models import create_model, load_checkpoint, is_model, list_models
from timm.utils import accuracy, AverageMeter, natural_key, setup_default_logging, set_jit_fuser, \
    decay_batch_step, check_batch_size_retry, ParseKwargs, reparameterize_model


try:
    from functorch.compile import memory_efficient_fusion
    has_functorch = True
except ImportError as e:
    has_functorch = False

has_compile = hasattr(torch, 'compile')

try:
    from sklearn.metrics import precision_score, recall_score, f1_score
    has_sklearn = True
except ImportError:
    has_sklearn = False

_logger = logging.getLogger('validate')


parser = argparse.ArgumentParser(description='PyTorch ImageNet Validation')
parser.add_argument('data', nargs='?', metavar='DIR', const=None,
                    help='path to dataset (*deprecated*, use --data-dir)')
parser.add_argument('--data-dir', metavar='DIR',
                    help='path to dataset (root dir)')
parser.add_argument('--dataset', metavar='NAME', default='',
                    help='dataset type + name ("<type>/<name>") (default: ImageFolder or ImageTar if empty)')
parser.add_argument('--split', metavar='NAME', default='validation',
                    help='dataset split (default: validation)')
parser.add_argument('--num-samples', default=None, type=int,
                    metavar='N', help='Manually specify num samples in dataset split, for IterableDatasets.')
parser.add_argument('--dataset-download', action='store_true', default=False,
                    help='Allow download of dataset for torch/ and tfds/ datasets that support it.')
parser.add_argument('--class-map', default='', type=str, metavar='FILENAME',
                    help='path to class to idx mapping file (default: "")')
parser.add_argument('--input-key', default=None, type=str,
                   help='Dataset key for input images.')
parser.add_argument('--input-img-mode', default=None, type=str,
                   help='Dataset image conversion mode for input images.')
parser.add_argument('--target-key', default=None, type=str,
                   help='Dataset key for target labels.')
parser.add_argument('--dataset-trust-remote-code', action='store_true', default=False,
                   help='Allow huggingface dataset import to execute code downloaded from the dataset\'s repo.')

parser.add_argument('--model', '-m', metavar='NAME', default='dpn92',
                    help='model architecture (default: dpn92)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--img-size', default=None, type=int,
                    metavar='N', help='Input image dimension, uses model default if empty')
parser.add_argument('--in-chans', type=int, default=None, metavar='N',
                    help='Image input channels (default: None => 3)')
parser.add_argument('--input-size', default=None, nargs=3, type=int, metavar='N',
                    help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty')
parser.add_argument('--use-train-size', action='store_true', default=False,
                    help='force use of train input size, even when test size is specified in pretrained cfg')
parser.add_argument('--crop-pct', default=None, type=float,
                    metavar='N', help='Input image center crop pct')
parser.add_argument('--crop-mode', default=None, type=str,
                    metavar='N', help='Input image crop mode (squash, border, center). Model default if None.')
parser.add_argument('--crop-border-pixels', type=int, default=None,
                    help='Crop pixels from image border.')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float,  nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('--num-classes', type=int, default=None,
                    help='Number classes in dataset')
parser.add_argument('--gp', default=None, type=str, metavar='POOL',
                    help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
parser.add_argument('--log-freq', default=10, type=int,
                    metavar='N', help='batch logging frequency (default: 10)')
parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--num-gpu', type=int, default=1,
                    help='Number of GPUS to use')
parser.add_argument('--test-pool', dest='test_pool', action='store_true',
                    help='enable test time pool')
parser.add_argument('--no-prefetcher', action='store_true', default=False,
                    help='disable fast prefetcher')
parser.add_argument('--pin-mem', action='store_true', default=False,
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--channels-last', action='store_true', default=False,
                    help='Use channels_last memory layout')
parser.add_argument('--device', default='cuda', type=str,
                    help="Device (accelerator) to use.")
parser.add_argument('--amp', action='store_true', default=False,
                    help='use Native AMP for mixed precision inference')
parser.add_argument('--amp-dtype', default='float16', type=str,
                    help='lower precision AMP dtype (default: float16)')
parser.add_argument('--model-dtype', default=None, type=str,
                   help='Model dtype override (non-AMP) (default: float32)')
parser.add_argument('--tf-preprocessing', action='store_true', default=False,
                    help='Use Tensorflow preprocessing pipeline (require CPU TF installed')
parser.add_argument('--use-ema', dest='use_ema', action='store_true',
                    help='use ema version of weights if present')
parser.add_argument('--fuser', default='', type=str,
                    help="Select jit fuser. One of ('', 'te', 'old', 'nvfuser')")
parser.add_argument('--fast-norm', default=False, action='store_true',
                    help='enable experimental fast-norm')
parser.add_argument('--reparam', default=False, action='store_true',
                    help='Reparameterize model')
parser.add_argument('--model-kwargs', nargs='*', default={}, action=ParseKwargs)
parser.add_argument('--torchcompile-mode', type=str, default=None,
                    help="torch.compile mode (default: None).")
parser.add_argument('--latent-size', type=int, default=None,
                    help='Latent vector size used for stem AE (forward-time application).')
parser.add_argument('--ae-checkpoint', '--ae-checkpoints', dest='ae_checkpoint', default='',
                    help='Path to stem autoencoder checkpoint (.pth).')
parser.add_argument('--wo-config', type=int, nargs=3, default=[0, 0, 0], metavar=('PWC1', 'DW_DEPTH', 'PWC2'),
                    help='WhiteningObserver config [PWC1_out_channel, Depthwise_depth, PWC2_out_channel]. 0 skips that stage.')
parser.add_argument('--tta-steps', type=int, default=1, metavar='K',
                    help='Number of TTA adaptation steps per mini-batch (default: 1)')
parser.add_argument('--tta-lr', type=float, default=1e-3, metavar='LR',
                    help='Learning rate for TTA adaptation of WhiteningObserver (default: 1e-3)')
parser.add_argument('--renew-n', type=int, default=0, metavar='N',
                    help='Reset adapted weights to original every N batches. 0 = never reset (accumulate). (default: 0)')

scripting_group = parser.add_mutually_exclusive_group()
scripting_group.add_argument('--torchscript', default=False, action='store_true',
                             help='torch.jit.script the full model')
scripting_group.add_argument('--torchcompile', nargs='?', type=str, default=None, const='inductor',
                             help="Enable compilation w/ specified backend (default: inductor).")
scripting_group.add_argument('--aot-autograd', default=False, action='store_true',
                             help="Enable AOT Autograd support.")

parser.add_argument('--results-file', default='', type=str, metavar='FILENAME',
                    help='Output csv file for validation results (summary)')
parser.add_argument('--results-format', default='csv', type=str,
                    help='Format for results file one of (csv, json) (default: csv).')
parser.add_argument('--real-labels', default='', type=str, metavar='FILENAME',
                    help='Real labels JSON file for imagenet evaluation')
parser.add_argument('--valid-labels', default='', type=str, metavar='FILENAME',
                    help='Valid label indices txt file for validation of partial label space')
parser.add_argument('--retry', default=False, action='store_true',
                    help='Enable batch size decay & retry for single model validation')

parser.add_argument('--metrics-avg', type=str, default=None,
                    choices=['micro', 'macro', 'weighted'],
                    help='Enable precision, recall, F1-score calculation and specify the averaging method. '
                         'Requires scikit-learn. (default: None)')

# NaFlex loader arguments
parser.add_argument('--naflex-loader', action='store_true', default=False,
                   help='Use NaFlex loader (Requires NaFlex compatible model)')
parser.add_argument('--naflex-max-seq-len', type=int, default=576,
                   help='Fixed maximum sequence length for NaFlex loader (validation)')


def validate(args):
    # might as well try to validate something
    args.pretrained = args.pretrained or not args.checkpoint
    args.prefetcher = not args.no_prefetcher

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    device = torch.device(args.device)

    if args.metrics_avg and not has_sklearn:
        _logger.warning(
            f"scikit-learn not installed, disabling metrics calculation. Please install with 'pip install scikit-learn'.")
        args.metrics_avg = None

    model_dtype = None
    if args.model_dtype:
        assert args.model_dtype in ('float32', 'float16', 'bfloat16')
        model_dtype = getattr(torch, args.model_dtype)

    # resolve AMP arguments based on PyTorch availability
    amp_autocast = suppress
    if args.amp:
        assert model_dtype is None or model_dtype == torch.float32, 'float32 model dtype must be used with AMP'
        assert args.amp_dtype in ('float16', 'bfloat16')
        amp_dtype = torch.bfloat16 if args.amp_dtype == 'bfloat16' else torch.float16
        amp_autocast = partial(torch.autocast, device_type=device.type, dtype=amp_dtype)
        _logger.info('Validating in mixed precision with native PyTorch AMP.')
    else:
        _logger.info(f'Validating in {model_dtype or torch.float32}. AMP not enabled.')

    if args.fuser:
        set_jit_fuser(args.fuser)

    if args.fast_norm:
        set_fast_norm()

    # create model
    in_chans = 3
    if args.in_chans is not None:
        in_chans = args.in_chans
    elif args.input_size is not None:
        in_chans = args.input_size[0]

    # propagate stem AE options to model kwargs
    if args.ae_checkpoint:
        args.model_kwargs = dict(args.model_kwargs)
        args.model_kwargs.setdefault('stem_ae_checkpoint', args.ae_checkpoint)
    if args.latent_size is not None:
        args.model_kwargs = dict(args.model_kwargs)
        args.model_kwargs.setdefault('stem_ae_latent_size', args.latent_size)

    if args.model in ('resnet26', 'resnet50') and args.wo_config != [0, 0, 0]:
        args.model_kwargs = dict(args.model_kwargs)
        args.model_kwargs['wo_config'] = args.wo_config
        args.model_kwargs.setdefault('pretrained_strict', False)

    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.num_classes,
        in_chans=in_chans,
        global_pool=args.gp,
        scriptable=args.torchscript,
        **args.model_kwargs,
    )
    # Freeze stem AE parameters when an AE checkpoint is provided.
    if args.ae_checkpoint:
        for name, p in model.named_parameters():
            if name.startswith('stem_ae.'):
                p.requires_grad = False
    if args.num_classes is None:
        assert hasattr(model, 'num_classes'), 'Model must have `num_classes` attr if not set on cmd line/config.'
        args.num_classes = model.num_classes

    if args.checkpoint:
        load_checkpoint(model, args.checkpoint, args.use_ema)

    if args.reparam:
        model = reparameterize_model(model)

    param_count = sum([m.numel() for m in model.parameters()])
    ae_param_count = 0
    if hasattr(model, 'stem_ae') and model.stem_ae is not None:
        ae_param_count = sum(p.numel() for p in model.stem_ae.parameters())
    base_param_count = max(param_count - ae_param_count, 0)
    ae_base_ratio = (ae_param_count / base_param_count) if base_param_count > 0 else 0.0
    _logger.info(
        'Model %s created, param count: %d | base: %.3fM | ae: %.3fM | ae/base: %.6f'
        % (
            args.model,
            param_count,
            base_param_count / 1e6,
            ae_param_count / 1e6,
            ae_base_ratio,
        )
    )

    data_config = resolve_data_config(
        vars(args),
        model=model,
        use_test_size=not args.use_train_size,
        verbose=True,
    )
    test_time_pool = False
    if args.test_pool:
        model, test_time_pool = apply_test_time_pool(model, data_config)

    model = model.to(device=device, dtype=model_dtype)  # FIXME move model device & dtype into create_model
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    if args.torchscript:
        model = torch.jit.script(model)
    elif args.torchcompile:
        assert has_compile, 'A version of torch w/ torch.compile() is required for --compile, possibly a nightly.'
        torch._dynamo.reset()
        model = torch.compile(model, backend=args.torchcompile, mode=args.torchcompile_mode)
    elif args.aot_autograd:
        assert has_functorch, "functorch is needed for --aot-autograd"
        model = memory_efficient_fusion(model)

    if args.num_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.num_gpu)))

    criterion = nn.CrossEntropyLoss().to(device)

    root_dir = args.data or args.data_dir
    if args.input_img_mode is None:
        input_img_mode = 'RGB' if data_config['input_size'][0] == 3 else 'L'
    else:
        input_img_mode = args.input_img_mode
    dataset = create_dataset(
        args.dataset,
        root=root_dir,
        split=args.split,
        is_training=False,
        class_map=args.class_map,
        download=args.dataset_download,
        batch_size=args.batch_size,
        input_img_mode=input_img_mode,
        input_key=args.input_key,
        target_key=args.target_key,
        num_samples=args.num_samples,
        trust_remote_code=args.dataset_trust_remote_code,
    )

    if args.valid_labels:
        with open(args.valid_labels, 'r') as f:
            valid_labels = [int(line.rstrip()) for line in f]
    else:
        valid_labels = None

    if args.real_labels:
        real_labels = RealLabelsImagenet(dataset.filenames(basename=True), real_json=args.real_labels)
    else:
        real_labels = None

    if args.naflex_loader:
        model_patch_size = None
        if hasattr(model, 'embeds') and hasattr(model.embeds, 'patch_size'):
            model_patch_size = model.embeds.patch_size
        from timm.data import create_naflex_loader
        loader = create_naflex_loader(
            dataset=dataset,
            patch_size=model_patch_size or (16, 16),
            max_seq_len=args.naflex_max_seq_len,
            mean=data_config['mean'],
            std=data_config['std'],
            pin_memory=args.pin_mem,
            img_dtype=model_dtype or torch.float32,
            device=device,
            use_prefetcher=args.prefetcher,
            batch_size=args.batch_size,
            is_training=False,
            interpolation=data_config['interpolation'],
            num_workers=args.workers,
            crop_pct=data_config['crop_pct'],
        )
    else:
        loader = create_loader(
            dataset,
            input_size=data_config['input_size'],
            mean=data_config['mean'],
            std=data_config['std'],
            pin_memory=args.pin_mem,
            img_dtype=model_dtype or torch.float32,
            device=device,
            use_prefetcher=args.prefetcher,
            batch_size=args.batch_size,
            is_training=False,
            interpolation=data_config['interpolation'],
            num_workers=args.workers,
            crop_pct=data_config['crop_pct'],
        )

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    g_losses = AverageMeter()

    if args.metrics_avg:
        all_preds = []
        all_targets = []

    tta_mode = hasattr(model, 'use_wo') and model.use_wo and args.wo_config != [0, 0, 0]
    if tta_mode:
        renew_desc = f'every {args.renew_n} batches' if args.renew_n > 0 else 'never (accumulate)'
        _logger.info(
            f'TTA enabled: {args.tta_steps} adaptation steps, lr={args.tta_lr}, '
            f'adapting conv1 + whitening_observer, renew={renew_desc}'
        )

    model.eval()

    if not tta_mode:
        # Standard validation: no gradients needed at all
        eval_ctx = torch.inference_mode()
    else:
        eval_ctx = torch.no_grad()
    with eval_ctx:
        # warmup, reduce variability of first batch time, especially for comparing torchscript vs non
        if not args.naflex_loader:
            input = torch.randn((args.batch_size,) + tuple(data_config['input_size'])).to(device=device, dtype=model_dtype)
            if args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)
            with amp_autocast():
                model(input)

        if tta_mode:
            conv1_state_orig = copy.deepcopy(model.conv1.state_dict())
            wo_state_orig = copy.deepcopy(model.whitening_observer.state_dict())

        eval_start = time.time()
        end = time.time()
        for batch_idx, (input, target) in enumerate(loader):
            if not args.prefetcher:
                input = input.to(device=device, dtype=model_dtype)
                target = target.to(device=device)
            if args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)

            if tta_mode:
                if args.renew_n > 0 and batch_idx % args.renew_n == 0:
                    model.conv1.load_state_dict(conv1_state_orig)
                    model.whitening_observer.load_state_dict(wo_state_orig)

                model.conv1.eval()
                model.whitening_observer.eval()
                tta_params = list(model.conv1.parameters()) + list(model.whitening_observer.parameters())
                wo_optimizer = torch.optim.Adam(tta_params, lr=args.tta_lr)
                for _step in range(args.tta_steps):
                    wo_optimizer.zero_grad()
                    with torch.enable_grad():
                        with amp_autocast():
                            _out = model(input)
                            if isinstance(_out, (tuple, list)):
                                _, g_loss = _out
                            else:
                                g_loss = torch.tensor(0.0, device=device)
                    g_loss.backward()
                    wo_optimizer.step()
                if args.tta_steps > 0:
                    g_losses.update(g_loss.item(), input.shape[0])
                model.conv1.eval()
                model.whitening_observer.eval()

            # compute output
            with amp_autocast():
                output = model(input)
                if isinstance(output, (tuple, list)):
                    output = output[0]

                if valid_labels is not None:
                    output = output[:, valid_labels]
                loss = criterion(output, target)

            if real_labels is not None:
                real_labels.add_result(output)

            # measure accuracy and record loss
            batch_size = output.shape[0]
            acc1, acc5 = accuracy(output.detach(), target, topk=(1, 5))
            losses.update(loss.item(), batch_size)
            top1.update(acc1.item(), batch_size)
            top5.update(acc5.item(), batch_size)

            if args.metrics_avg:
                predictions = torch.argmax(output, dim=1)
                all_preds.append(predictions.cpu())
                all_targets.append(target.cpu())

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx % args.log_freq == 0:
                log_msg = (
                    'Test: [{0:>4d}/{1}]  '
                    'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                    'Acc@1: {top1.val:>7.3f} ({top1.avg:>7.3f})  '
                    'Acc@5: {top5.val:>7.3f} ({top5.avg:>7.3f})'.format(
                        batch_idx,
                        len(loader),
                        batch_time=batch_time,
                        rate_avg=batch_size / batch_time.avg,
                        loss=losses,
                        top1=top1,
                        top5=top5
                    )
                )
                if tta_mode:
                    log_msg += f'  GLoss: {g_losses.val:.4f} ({g_losses.avg:.4f})'
                _logger.info(log_msg)

        total_eval_time = time.time() - eval_start
        num_batches = len(loader) if len(loader) > 0 else 1
        avg_batch_time = total_eval_time / num_batches
        _logger.info(
            f"Eval timing - total: {total_eval_time:.3f}s, "
            f"per-batch: {avg_batch_time:.3f}s over {num_batches} batches."
        )

    if real_labels is not None:
        # real labels mode replaces topk values at the end
        top1a, top5a = real_labels.get_accuracy(k=1), real_labels.get_accuracy(k=5)
    else:
        top1a, top5a = top1.avg, top5.avg

    metric_results = {}
    if args.metrics_avg:
        all_preds = torch.cat(all_preds).numpy()
        all_targets = torch.cat(all_targets).numpy()
        precision = precision_score(all_targets, all_preds, average=args.metrics_avg, zero_division=0)
        recall = recall_score(all_targets, all_preds, average=args.metrics_avg, zero_division=0)
        f1 = f1_score(all_targets, all_preds, average=args.metrics_avg, zero_division=0)
        metric_results = {
            f'{args.metrics_avg}_precision': round(100 * precision, 4),
            f'{args.metrics_avg}_recall': round(100 * recall, 4),
            f'{args.metrics_avg}_f1_score': round(100 * f1, 4),
        }

    results = OrderedDict(
        model=args.model,
        top1=round(top1a, 4), top1_err=round(100 - top1a, 4),
        top5=round(top5a, 4), top5_err=round(100 - top5a, 4),
        **metric_results,
        eval_total_time=round(total_eval_time, 4),
        eval_avg_batch_time=round(avg_batch_time, 6),
        param_count=round(param_count / 1e6, 2),
        base_param_count=round(base_param_count / 1e6, 2),
        ae_param_count=round(ae_param_count / 1e6, 4),
        ae_base_ratio=round(ae_base_ratio, 6),
        img_size=data_config['input_size'][-1],
        crop_pct=data_config['crop_pct'],
        interpolation=data_config['interpolation'],
    )

    log_string = ' * Acc@1 {:.3f} ({:.3f}) Acc@5 {:.3f} ({:.3f})'.format(
       results['top1'], results['top1_err'], results['top5'], results['top5_err'])
    if metric_results:
        log_string += ' | Precision({avg}) {prec:.3f} | Recall({avg}) {rec:.3f} | F1-score({avg}) {f1:.3f}'.format(
            avg=args.metrics_avg,
            prec=metric_results[f'{args.metrics_avg}_precision'],
            rec=metric_results[f'{args.metrics_avg}_recall'],
            f1=metric_results[f'{args.metrics_avg}_f1_score'],
        )
    _logger.info(log_string)

    return results


def _try_run(args, initial_batch_size):
    batch_size = initial_batch_size
    results = OrderedDict()
    error_str = 'Unknown'
    while batch_size:
        args.batch_size = batch_size * args.num_gpu  # multiply by num-gpu for DataParallel case
        try:
            if 'cuda' in args.device and torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif "npu" in args.device and torch.npu.is_available():
                torch.npu.empty_cache()
            results = validate(args)
            return results
        except RuntimeError as e:
            error_str = str(e)
            _logger.error(f'"{error_str}" while running validation.')
            if not check_batch_size_retry(error_str):
                break
        batch_size = decay_batch_step(batch_size)
        _logger.warning(f'Reducing batch size to {batch_size} for retry.')
    results['model'] = args.model
    results['error'] = error_str
    _logger.error(f'{args.model} failed to validate ({error_str}).')
    return results


_NON_IN1K_FILTERS = ['*_in21k', '*_in22k', '*in12k', '*_dino', '*fcmae', '*seer']


def main():
    setup_default_logging()
    args = parser.parse_args()
    model_cfgs = []
    model_names = []
    if os.path.isdir(args.checkpoint):
        # validate all checkpoints in a path with same model
        checkpoints = glob.glob(args.checkpoint + '/*.pth.tar')
        checkpoints += glob.glob(args.checkpoint + '/*.pth')
        model_names = list_models(args.model)
        model_cfgs = [(args.model, c) for c in sorted(checkpoints, key=natural_key)]
    else:
        if args.model == 'all':
            # validate all models in a list of names with pretrained checkpoints
            args.pretrained = True
            model_names = list_models(
                pretrained=True,
                exclude_filters=_NON_IN1K_FILTERS,
            )
            model_cfgs = [(n, '') for n in model_names]
        elif not is_model(args.model):
            # model name doesn't exist, try as wildcard filter
            model_names = list_models(
                args.model,
                pretrained=True,
            )
            model_cfgs = [(n, '') for n in model_names]

        if not model_cfgs and os.path.isfile(args.model):
            with open(args.model) as f:
                model_names = [line.rstrip() for line in f]
            model_cfgs = [(n, None) for n in model_names if n]

    if len(model_cfgs):
        _logger.info('Running bulk validation on these pretrained models: {}'.format(', '.join(model_names)))
        results = []
        try:
            initial_batch_size = args.batch_size
            for m, c in model_cfgs:
                args.model = m
                args.checkpoint = c
                r = _try_run(args, initial_batch_size)
                if 'error' in r:
                    continue
                if args.checkpoint:
                    r['checkpoint'] = args.checkpoint
                results.append(r)
        except KeyboardInterrupt as e:
            pass
        results = sorted(results, key=lambda x: x['top1'], reverse=True)
    else:
        if args.retry:
            results = _try_run(args, args.batch_size)
        else:
            results = validate(args)

    if args.results_file:
        write_results(args.results_file, results, format=args.results_format)

    # output results in JSON to stdout w/ delimiter for runner script
    print(f'--result\n{json.dumps(results, indent=4)}')


def write_results(results_file, results, format='csv'):
    with open(results_file, mode='w') as cf:
        if format == 'json':
            json.dump(results, cf, indent=4)
        else:
            if not isinstance(results, (list, tuple)):
                results = [results]
            if not results:
                return
            dw = csv.DictWriter(cf, fieldnames=results[0].keys())
            dw.writeheader()
            for r in results:
                dw.writerow(r)
            cf.flush()



if __name__ == '__main__':
    main()
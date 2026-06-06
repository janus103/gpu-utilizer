#!/usr/bin/env python3
"""Expert Stem Training Script.

This script trains a specialized stem layer (conv1, bn1) for a SINGLE augmentation type.

Available augmentation types (V2 policy):
    0: IntensityIncreasing (Brightness + Contrast)
    1: SaturationIncreasing
    2: SharpnessIncreasing
    3: GaussianBlurIncreasing
    4: PosterizeIncreasing
    5: SolarizeIncreasing
    6: SaltAndPepperIncreasing

Usage:
    python train_expert.py --data-dir /path/to/imagenet --model resnet50 --pretrained \
        --aug-type 0 --sl-max 0.5 --epochs 100
"""
import argparse
import datetime
import importlib
import logging
import os
import time
from collections import OrderedDict
from contextlib import suppress
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import yaml
from torchvision import transforms

from timm import utils
from timm.data import resolve_data_config
from timm.data import augmix_sl_ops_v2, get_augmix_sl_transform_names, AUGMIX_SL_V2_NUM_TRANSFORMS
from timm.models import create_model, safe_model_name
from timm.optim import create_optimizer_v2
from timm.scheduler import create_scheduler_v2
from timm.utils import NativeScaler

try:
    import wandb
    has_wandb = True
except ImportError:
    has_wandb = False

_logger = logging.getLogger('train_expert')


class ExpertAugmentationDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform_op, sl_min=0.0, sl_max=1.0, base_transform=None, final_transform=None):
        self.dataset = dataset
        self.transform_op = transform_op
        self.sl_min = sl_min
        self.sl_max = sl_max
        self.base_transform = base_transform
        self.final_transform = final_transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        if self.base_transform is not None:
            img = self.base_transform(img)
        sl = np.random.uniform(self.sl_min, self.sl_max)
        img = self.transform_op(img, sl)
        if self.final_transform is not None:
            img = self.final_transform(img)
        return img, label


class ExpertAugmentationValidationDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform_op, sl_max=1.0, base_transform=None, final_transform=None):
        self.dataset = dataset
        self.transform_op = transform_op
        self.sl_max = sl_max
        self.base_transform = base_transform
        self.final_transform = final_transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        if self.base_transform is not None:
            img = self.base_transform(img)
        img = self.transform_op(img, self.sl_max)
        if self.final_transform is not None:
            img = self.final_transform(img)
        return img, label


config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE')

parser = argparse.ArgumentParser(description='Expert Stem Training')
group = parser.add_argument_group('Dataset parameters')
parser.add_argument('data', nargs='?', metavar='DIR', const=None)
group.add_argument('--data-dir', metavar='DIR')
group.add_argument('--train-split', default='train')
group.add_argument('--val-split', default='validation')
group = parser.add_argument_group('Model parameters')
group.add_argument('--model', default='resnet50', type=str)
group.add_argument('--pretrained', action='store_true', default=False)
group.add_argument('--initial-checkpoint', default='', type=str)
group.add_argument('--resume', default='', type=str)
group.add_argument('--num-classes', type=int, default=1000)
group.add_argument('--img-size', type=int, default=224)
group.add_argument('-b', '--batch-size', type=int, default=128)
group.add_argument('-vb', '--validation-batch-size', type=int, default=None)
group = parser.add_argument_group('Expert Augmentation parameters')
group.add_argument('--aug-type', type=int, required=True, choices=range(7))
group.add_argument('--sl-min', type=float, default=0.0,
                   help='Minimum severity level (default: 0.0)')
group.add_argument('--sl-max', type=float, default=1.0,
                   help='Maximum severity level (default: 1.0)')
group.add_argument('--train-mode', type=str, default='stem', 
                   choices=['stem', 'all-conv', 'stem-all-bn'],
                   help='Training mode: '
                        '"stem" = conv1 + bn1 only (default), '
                        '"all-conv" = all conv layers including stem, '
                        '"stem-all-bn" = conv1 + all BN layers')
group = parser.add_argument_group('Device parameters')
group.add_argument('--device', default='cuda', type=str)
group.add_argument('--amp', action='store_true', default=False)
group.add_argument('--amp-dtype', default='float16', type=str)
group.add_argument("--local_rank", default=0, type=int)
group.add_argument('--device-modules', default=None, type=str, nargs='+')
group = parser.add_argument_group('Optimizer parameters')
group.add_argument('--opt', default='sgd', type=str)
group.add_argument('--lr', type=float, default=0.01)
group.add_argument('--weight-decay', type=float, default=1e-4)
group.add_argument('--momentum', type=float, default=0.9)
group.add_argument('--clip-grad', type=float, default=None)
group = parser.add_argument_group('LR schedule parameters')
group.add_argument('--sched', type=str, default='cosine')
group.add_argument('--epochs', type=int, default=100)
group.add_argument('--warmup-epochs', type=int, default=5)
group.add_argument('--warmup-lr', type=float, default=1e-5)
group.add_argument('--min-lr', type=float, default=1e-6)
group.add_argument('--decay-rate', type=float, default=0.1)
group = parser.add_argument_group('Misc parameters')
group.add_argument('--seed', type=int, default=42)
group.add_argument('--log-interval', type=int, default=50)
group.add_argument('--val-interval', type=int, default=1)
group.add_argument('-j', '--workers', type=int, default=4)
group.add_argument('--pin-mem', action='store_true', default=False)
group.add_argument('--output', default='', type=str)
group.add_argument('--experiment', default='', type=str)
group.add_argument('--log-wandb', action='store_true', default=False)
group.add_argument('--wandb-project', default='expert-stem', type=str)


def _parse_args():
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)
    args = parser.parse_args(remaining)
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


def main():
    utils.setup_default_logging()
    args, args_text = _parse_args()
    
    if args.device_modules:
        for module in args.device_modules:
            importlib.import_module(module)
    
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    
    device = utils.init_distributed_device(args)
    if args.distributed:
        _logger.info(f'Distributed mode, Process {args.rank}/{args.world_size}')
    else:
        _logger.info(f'Single process on {args.device}.')
    
    amp_dtype = torch.float16
    amp_autocast = suppress
    loss_scaler = None
    if args.amp:
        if args.amp_dtype == 'bfloat16':
            amp_dtype = torch.bfloat16
        amp_autocast = partial(torch.autocast, device_type=device.type, dtype=amp_dtype)
        if device.type == 'cuda' and amp_dtype == torch.float16:
            loss_scaler = NativeScaler(device=device.type)
        _logger.info('Using AMP.')
    
    utils.random_seed(args.seed, args.rank)
    
    transform_names = get_augmix_sl_transform_names(version=2)
    transform_ops = augmix_sl_ops_v2()
    aug_type = args.aug_type
    aug_name = transform_names[aug_type]
    transform_op = transform_ops[aug_type]
    
    _logger.info(f'Expert training: {aug_name} (idx={aug_type}), SL=[{args.sl_min}, {args.sl_max}]')
    _logger.info(f'Train mode: {args.train_mode}')
    
    _logger.info(f'Creating model: {args.model}')
    model = create_model(args.model, pretrained=args.pretrained, num_classes=args.num_classes,
                         checkpoint_path=args.initial_checkpoint)
    model = model.to(device)
    
    # Freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False
    
    trainable_params = []
    
    if args.train_mode == 'stem':
        # Mode 1: conv1 + bn1 only (original behavior)
        _logger.info('Training mode: stem (conv1 + bn1)')
        if hasattr(model, 'conv1'):
            for param in model.conv1.parameters():
                param.requires_grad = True
                trainable_params.append(param)
            _logger.info(f'  conv1: {sum(p.numel() for p in model.conv1.parameters())} params')
        
        if hasattr(model, 'bn1'):
            for param in model.bn1.parameters():
                param.requires_grad = True
                trainable_params.append(param)
            _logger.info(f'  bn1: {sum(p.numel() for p in model.bn1.parameters())} params')
    
    elif args.train_mode == 'all-conv':
        # Mode 2: All conv layers including stem
        _logger.info('Training mode: all-conv (all Conv2d layers)')
        conv_count = 0
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                for param in module.parameters():
                    param.requires_grad = True
                    trainable_params.append(param)
                conv_count += 1
        _logger.info(f'  Total Conv2d layers: {conv_count}')
        _logger.info(f'  Conv params: {sum(p.numel() for p in trainable_params)}')
    
    elif args.train_mode == 'stem-all-bn':
        # Mode 3: conv1 + all BN layers
        _logger.info('Training mode: stem-all-bn (conv1 + all BatchNorm layers)')
        if hasattr(model, 'conv1'):
            for param in model.conv1.parameters():
                param.requires_grad = True
                trainable_params.append(param)
            _logger.info(f'  conv1: {sum(p.numel() for p in model.conv1.parameters())} params')
        
        bn_count = 0
        bn_params = 0
        for name, module in model.named_modules():
            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                for param in module.parameters():
                    param.requires_grad = True
                    trainable_params.append(param)
                    bn_params += param.numel()
                bn_count += 1
        _logger.info(f'  Total BatchNorm layers: {bn_count}')
        _logger.info(f'  BatchNorm params: {bn_params}')
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in trainable_params)
    _logger.info(f'Total={total_params/1e6:.2f}M, Trainable={trainable_count} ({100*trainable_count/total_params:.2f}%)')
    
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = create_optimizer_v2(trainable_params, opt=args.opt, lr=args.lr,
                                     weight_decay=args.weight_decay, momentum=args.momentum)
    
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_state = model.state_dict()
        src = checkpoint.get('model', checkpoint)
        for key in ['conv1.weight', 'bn1.weight', 'bn1.bias', 'bn1.running_mean', 'bn1.running_var']:
            if key in src:
                model_state[key] = src[key]
        model.load_state_dict(model_state)
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        _logger.info(f'Resumed from epoch {start_epoch}')
    
    data_config = resolve_data_config(vars(args), model=model, verbose=utils.is_primary(args))
    if args.data and not args.data_dir:
        args.data_dir = args.data
    
    base_transform = transforms.Compose([
        transforms.Resize(int(args.img_size / data_config['crop_pct'])),
        transforms.CenterCrop(args.img_size),
    ])
    final_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=data_config['mean'], std=data_config['std']),
    ])
    
    from torchvision.datasets import ImageFolder
    train_dir = os.path.join(args.data_dir, args.train_split)
    val_dir = os.path.join(args.data_dir, args.val_split)
    raw_train = ImageFolder(train_dir)
    raw_val = ImageFolder(val_dir)
    
    dataset_train = ExpertAugmentationDataset(raw_train, transform_op, args.sl_min, args.sl_max, base_transform, final_transform)
    dataset_val = ExpertAugmentationValidationDataset(raw_val, transform_op, args.sl_max, base_transform, final_transform)
    
    _logger.info(f'Train: {len(dataset_train)}, Val: {len(dataset_val)} (train SL=[{args.sl_min}, {args.sl_max}], val SL={args.sl_max})')
    
    sampler_train = sampler_val = None
    if args.distributed:
        sampler_train = torch.utils.data.distributed.DistributedSampler(dataset_train)
        sampler_val = torch.utils.data.distributed.DistributedSampler(dataset_val, shuffle=False)
    
    loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size,
        shuffle=(sampler_train is None), sampler=sampler_train, num_workers=args.workers,
        pin_memory=args.pin_mem, drop_last=True)
    loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=args.validation_batch_size or args.batch_size,
        shuffle=False, sampler=sampler_val, num_workers=args.workers, pin_memory=args.pin_mem)
    
    lr_scheduler = None
    if args.sched:
        lr_scheduler, num_epochs = create_scheduler_v2(optimizer, sched=args.sched, num_epochs=args.epochs,
            warmup_epochs=args.warmup_epochs, warmup_lr=args.warmup_lr, min_lr=args.min_lr, decay_rate=args.decay_rate)
    else:
        num_epochs = args.epochs
    
    base_dir = args.output or './output'
    if args.experiment:
        output_dir = os.path.join(base_dir, args.experiment)
    else:
        output_dir = os.path.join(base_dir, f'expert_{aug_name}_{args.model}_{args.train_mode}_sl{args.sl_min}-{args.sl_max}')
    
    if utils.is_primary(args):
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
            f.write(args_text)
        _logger.info(f'Output: {output_dir}')
        
        # Write header to results file
        results_file = os.path.join(output_dir, 'results.txt')
        with open(results_file, 'a') as f:
            f.write(f'\n=== Training started at {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} ===\n')
            f.write(f'model={args.model}, aug={aug_name}, train_mode={args.train_mode}, sl=[{args.sl_min}, {args.sl_max}]\n')
            f.write(f'epochs={args.epochs}, lr={args.lr}, batch_size={args.batch_size}\n')
    
    if args.log_wandb and utils.is_primary(args) and has_wandb:
        wandb.init(project=args.wandb_project, name=f'expert_{aug_name}', config=args)
    
    # Track top 3 checkpoints: list of (accuracy, epoch) tuples
    # Priority: higher accuracy first, then later epoch
    top_checkpoints = []
    
    try:
        for epoch in range(start_epoch, num_epochs):
            if args.distributed:
                loader_train.sampler.set_epoch(epoch)
            
            train_metrics = train_one_epoch(epoch, model, loader_train, optimizer, criterion,
                                            args, device, amp_autocast, loss_scaler)
            
            if lr_scheduler is not None:
                lr_scheduler.step(epoch + 1)
            
            if (epoch + 1) % args.val_interval == 0 or epoch == num_epochs - 1:
                eval_metrics = validate(model, loader_val, criterion, args, device, amp_autocast)
                
                if utils.is_primary(args):
                    # Save trainable parameters based on train_mode
                    trained_state = {}
                    
                    if args.train_mode == 'stem':
                        # Save conv1 + bn1
                        for name, param in model.named_parameters():
                            if name.startswith('conv1') or name.startswith('bn1'):
                                trained_state[name] = param.data.clone()
                        if hasattr(model, 'bn1'):
                            trained_state['bn1.running_mean'] = model.bn1.running_mean.clone()
                            trained_state['bn1.running_var'] = model.bn1.running_var.clone()
                            trained_state['bn1.num_batches_tracked'] = model.bn1.num_batches_tracked.clone()
                    
                    elif args.train_mode == 'all-conv':
                        # Save all conv parameters
                        for name, param in model.named_parameters():
                            if param.requires_grad:
                                trained_state[name] = param.data.clone()
                    
                    elif args.train_mode == 'stem-all-bn':
                        # Save conv1 + all BN parameters and running stats
                        for name, param in model.named_parameters():
                            if param.requires_grad:
                                trained_state[name] = param.data.clone()
                        # Also save BN running stats
                        for name, module in model.named_modules():
                            if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
                                trained_state[f'{name}.running_mean'] = module.running_mean.clone()
                                trained_state[f'{name}.running_var'] = module.running_var.clone()
                                trained_state[f'{name}.num_batches_tracked'] = module.num_batches_tracked.clone()
                    
                    ckpt = {'epoch': epoch, 'model': trained_state, 'optimizer': optimizer.state_dict(),
                            'args': args, 'aug_type': aug_type, 'aug_name': aug_name, 
                            'train_mode': args.train_mode, 'sl_min': args.sl_min, 'sl_max': args.sl_max}
                    torch.save(ckpt, os.path.join(output_dir, 'last.pth.tar'))
                    
                    # Append validation results to txt file
                    results_file = os.path.join(output_dir, 'results.txt')
                    with open(results_file, 'a') as f:
                        f.write(f'epoch={epoch:03d}, '
                                f'train_loss={train_metrics["loss"]:.4f}, train_top1={train_metrics["top1"]:.2f}, '
                                f'val_loss={eval_metrics["loss"]:.4f}, val_top1={eval_metrics["top1"]:.2f}, val_top5={eval_metrics["top5"]:.2f}, '
                                f'lr={optimizer.param_groups[0]["lr"]:.6f}\n')
                    
                    # Update top 3 checkpoints
                    current_acc = eval_metrics['top1']
                    top_checkpoints.append((current_acc, epoch))
                    # Sort by accuracy (desc), then by epoch (desc) for tie-breaking
                    top_checkpoints.sort(key=lambda x: (x[0], x[1]), reverse=True)
                    
                    # Keep only top 3
                    old_top3 = set((acc, ep) for acc, ep in top_checkpoints[:3])
                    if len(top_checkpoints) > 3:
                        # Remove checkpoints that fell out of top 3
                        for acc, ep in top_checkpoints[3:]:
                            old_file = os.path.join(output_dir, f'best_ep{ep:03d}_{acc:.2f}.pth.tar')
                            if os.path.exists(old_file):
                                os.remove(old_file)
                        top_checkpoints = top_checkpoints[:3]
                    
                    # Save current checkpoint if it's in top 3
                    if (current_acc, epoch) in old_top3:
                        ckpt['top1'] = current_acc
                        torch.save(ckpt, os.path.join(output_dir, f'best_ep{epoch:03d}_{current_acc:.2f}.pth.tar'))
                        rank = [i for i, (a, e) in enumerate(top_checkpoints) if a == current_acc and e == epoch][0] + 1
                        _logger.info(f'Top {rank}: {current_acc:.2f}% @ epoch {epoch}')
                    
                    if (epoch + 1) % 10 == 0:
                        torch.save(ckpt, os.path.join(output_dir, f'checkpoint-{epoch}.pth.tar'))
                    
                    if args.log_wandb and has_wandb:
                        wandb.log({'epoch': epoch, 'train_loss': train_metrics['loss'],
                                   'val_top1': eval_metrics['top1'], 'val_top5': eval_metrics['top5'],
                                   'lr': optimizer.param_groups[0]['lr']})
    except KeyboardInterrupt:
        pass
    
    if args.distributed:
        torch.distributed.destroy_process_group()
    
    if top_checkpoints:
        best_acc, best_ep = top_checkpoints[0]
        _logger.info(f'*** Best: {best_acc:.2f}% (epoch {best_ep})')
        _logger.info(f'*** Expert: {aug_name}, mode={args.train_mode}, SL=[{args.sl_min}, {args.sl_max}]')
        _logger.info(f'*** Top 3: {[(f"{acc:.2f}%@ep{ep}" ) for acc, ep in top_checkpoints]}')
        
        # Write final summary to results file
        if utils.is_primary(args):
            results_file = os.path.join(output_dir, 'results.txt')
            with open(results_file, 'a') as f:
                f.write(f'=== Training finished at {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} ===\n')
                f.write(f'Best: {best_acc:.2f}% @ epoch {best_ep}\n')
                f.write(f'Top 3: {[(f"{acc:.2f}%@ep{ep}") for acc, ep in top_checkpoints]}\n\n')


def train_one_epoch(epoch, model, loader, optimizer, criterion, args, device, amp_autocast=suppress, loss_scaler=None):
    losses_m = utils.AverageMeter()
    top1_m = utils.AverageMeter()
    top5_m = utils.AverageMeter()
    batch_time_m = utils.AverageMeter()
    
    model.train()
    
    # Keep BatchNorm in eval mode to use pretrained running statistics
    # This ensures fair comparison between train and validation
    # (both use the same BN statistics)
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            module.eval()
    
    end = time.time()
    num_batches = len(loader)
    
    for batch_idx, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)
        
        with amp_autocast():
            output = model(images)
            loss = criterion(output, labels)
        
        optimizer.zero_grad()
        if loss_scaler is not None:
            loss_scaler(loss, optimizer, clip_grad=args.clip_grad,
                        parameters=[p for p in model.parameters() if p.requires_grad])
        else:
            loss.backward()
            if args.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.clip_grad)
            optimizer.step()
        
        acc1, acc5 = utils.accuracy(output, labels, topk=(1, 5))
        losses_m.update(loss.item(), images.size(0))
        top1_m.update(acc1.item(), images.size(0))
        top5_m.update(acc5.item(), images.size(0))
        batch_time_m.update(time.time() - end)
        end = time.time()
        
        if batch_idx % args.log_interval == 0 or batch_idx == num_batches - 1:
            lr = optimizer.param_groups[0]['lr']
            if utils.is_primary(args):
                _logger.info(f'Train: {epoch} [{batch_idx:>4d}/{num_batches}]  '
                             f'Loss: {losses_m.val:.4f} ({losses_m.avg:.4f})  '
                             f'Acc@1: {top1_m.val:.2f} ({top1_m.avg:.2f})  '
                             f'Acc@5: {top5_m.val:.2f} ({top5_m.avg:.2f})  '
                             f'Time: {batch_time_m.val:.3f}s  LR: {lr:.2e}')
    
    return OrderedDict([('loss', losses_m.avg), ('top1', top1_m.avg), ('top5', top5_m.avg)])


def validate(model, loader, criterion, args, device, amp_autocast=suppress):
    losses_m = utils.AverageMeter()
    top1_m = utils.AverageMeter()
    top5_m = utils.AverageMeter()
    batch_time_m = utils.AverageMeter()
    
    # Set model to eval mode (uses pretrained running statistics for BN)
    # This matches training behavior where BN is also in eval mode
    model.eval()
    
    end = time.time()
    num_batches = len(loader)
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(loader):
            images = images.to(device)
            labels = labels.to(device)
            
            with amp_autocast():
                output = model(images)
                loss = criterion(output, labels)
            
            acc1, acc5 = utils.accuracy(output, labels, topk=(1, 5))
            
            if args.distributed:
                loss = utils.reduce_tensor(loss, args.world_size)
                acc1 = utils.reduce_tensor(acc1, args.world_size)
                acc5 = utils.reduce_tensor(acc5, args.world_size)
            
            losses_m.update(loss.item(), images.size(0))
            top1_m.update(acc1.item(), images.size(0))
            top5_m.update(acc5.item(), images.size(0))
            batch_time_m.update(time.time() - end)
            end = time.time()
            
            if batch_idx % args.log_interval == 0 or batch_idx == num_batches - 1:
                if utils.is_primary(args):
                    _logger.info(f'Val: [{batch_idx:>4d}/{num_batches}]  '
                                 f'Loss: {losses_m.val:.4f} ({losses_m.avg:.4f})  '
                                 f'Acc@1: {top1_m.val:.2f} ({top1_m.avg:.2f})  '
                                 f'Acc@5: {top5_m.val:.2f} ({top5_m.avg:.2f})  '
                                 f'Time: {batch_time_m.val:.3f}s')
    
    if utils.is_primary(args):
        _logger.info(f'Validation: Acc@1={top1_m.avg:.2f}% Acc@5={top5_m.avg:.2f}%')
    
    return OrderedDict([('loss', losses_m.avg), ('top1', top1_m.avg), ('top5', top5_m.avg)])


if __name__ == '__main__':
    main()

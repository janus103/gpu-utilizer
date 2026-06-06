#!/usr/bin/env python3
"""Augmentation Prediction Training Script.

This script trains a model to predict which augmentations were applied to an image
and their severity levels. The task uses Feature Space Centroid (FSC) differences
as input to an augmentation classifier.

Key concepts:
- Backbone (e.g., ResNet50) is frozen and used only for feature extraction
- FSC_diff = current_feature - FSC[predicted_label] is computed
- Aug classifier predicts [Aug_type, Normalized_SL] from FSC_diff
- Soft labels: SL value itself is the label (0 = not applied, >0 = applied with that SL)

Usage:
    python train_augmix.py --data-dir /path/to/imagenet --model resnet50 --pretrained \
        --fsc-path ./FSC/resnet50_FSC.pth --new-depth 3

Hacked together by / Copyright 2020 Ross Wightman (https://github.com/rwightman)
Modified for augmentation prediction task.
"""
import argparse
import copy
import importlib
import json
import logging
import os
import time
from collections import OrderedDict
from contextlib import suppress
from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils
import yaml

from timm import utils
from timm.data import (
    create_dataset,
    create_loader,
    resolve_data_config,
    create_augmix_sl_transform,
    create_augmix_sl_validation_transforms,
    get_augmix_sl_num_transforms,
    get_augmix_sl_transform_names,
    AUGMIX_SL_NUM_TRANSFORMS,
)
from timm.layers import set_fast_norm
from timm.models import create_model, safe_model_name, resume_checkpoint, load_checkpoint
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler_v2, scheduler_kwargs
from timm.utils import NativeScaler


try:
    import wandb
    has_wandb = True
except ImportError:
    has_wandb = False

has_compile = hasattr(torch, 'compile')


_logger = logging.getLogger('train_augmix')


# =============================================================================
# Augmentation Classifier Module
# =============================================================================

class AugClassifier(nn.Module):
    """Classifier that predicts augmentation types and severity levels from FSC_diff.
    
    Input: FSC_diff tensor of shape [batch_size, feature_dim]
    Output: Predictions of shape [batch_size, num_transforms]
    
    The output represents soft labels where:
    - 0 means the transform was not applied
    - >0 means the transform was applied with that severity level
    
    Args:
        feature_dim: Dimension of input features (e.g., 2048 for ResNet50).
        num_transforms: Number of transform types to predict.
        hidden_dims: List of hidden layer dimensions.
        dropout: Dropout probability.
        use_sigmoid: If True, apply sigmoid to output (for BCE loss).
    """
    
    def __init__(
        self,
        feature_dim: int,
        num_transforms: int = AUGMIX_SL_NUM_TRANSFORMS,
        hidden_dims: list = None,
        dropout: float = 0.1,
        use_sigmoid: bool = False,
    ):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [512, 256]
        
        self.feature_dim = feature_dim
        self.num_transforms = num_transforms
        self.use_sigmoid = use_sigmoid
        
        # Build MLP
        layers = []
        in_dim = feature_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(in_dim, num_transforms))
        
        self.mlp = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: FSC_diff tensor of shape [batch_size, feature_dim].
        
        Returns:
            Predictions of shape [batch_size, num_transforms].
        """
        out = self.mlp(x)
        if self.use_sigmoid:
            out = torch.sigmoid(out)
        return out


# =============================================================================
# Dataset Wrapper for Augmentation Labels
# =============================================================================

class AugMixSLDataset(torch.utils.data.Dataset):
    """Dataset wrapper that applies AugMixSL augmentation and returns labels.
    
    This wrapper:
    1. Loads the original image and label
    2. Applies the base transforms (resize, crop, etc.)
    3. Applies AugMixSL augmentation
    4. Returns (image, original_label, aug_labels)
    
    Args:
        dataset: Original dataset (e.g., ImageFolder).
        augmix_sl_transform: AugMixSLAugment instance.
        base_transform: Base transforms to apply before augmentation (resize, crop).
        final_transform: Final transforms (ToTensor, Normalize).
    """
    
    def __init__(
        self,
        dataset,
        augmix_sl_transform,
        base_transform=None,
        final_transform=None,
    ):
        self.dataset = dataset
        self.augmix_sl_transform = augmix_sl_transform
        self.base_transform = base_transform
        self.final_transform = final_transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # Get original image and label
        img, label = self.dataset[idx]
        
        # Apply base transform (resize, crop) if provided
        if self.base_transform is not None:
            img = self.base_transform(img)
        
        # Apply AugMixSL augmentation - returns (augmented_img, aug_labels)
        img, aug_labels = self.augmix_sl_transform(img)
        
        # Apply final transform (ToTensor, Normalize) if provided
        if self.final_transform is not None:
            img = self.final_transform(img)
        
        # Convert aug_labels to tensor
        aug_labels = torch.from_numpy(aug_labels).float()
        
        return img, label, aug_labels


class AugMixSLValidationDataset(torch.utils.data.Dataset):
    """Validation dataset with fixed augmentation groups.
    
    Divides the validation set into groups, each receiving a fixed augmentation.
    This ensures consistent evaluation across epochs.
    
    Args:
        dataset: Original validation dataset.
        validation_transforms: List of AugMixSLAugmentFixed instances.
        base_transform: Base transforms to apply before augmentation.
        final_transform: Final transforms (ToTensor, Normalize).
    """
    
    def __init__(
        self,
        dataset,
        validation_transforms,
        base_transform=None,
        final_transform=None,
    ):
        self.dataset = dataset
        self.validation_transforms = validation_transforms
        self.num_groups = len(validation_transforms)
        self.base_transform = base_transform
        self.final_transform = final_transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        
        # Determine which group this sample belongs to
        group_idx = idx % self.num_groups
        transform = self.validation_transforms[group_idx]
        
        if self.base_transform is not None:
            img = self.base_transform(img)
        
        # Apply fixed augmentation
        img, aug_labels = transform(img)
        
        if self.final_transform is not None:
            img = self.final_transform(img)
        
        aug_labels = torch.from_numpy(aug_labels).float()
        
        return img, label, aug_labels


# =============================================================================
# Custom Collate Function
# =============================================================================

def collate_aug_labels(batch):
    """Custom collate function for augmentation label dataset."""
    images = []
    labels = []
    aug_labels = []
    
    for img, label, aug_label in batch:
        images.append(img)
        labels.append(label)
        aug_labels.append(aug_label)
    
    images = torch.stack(images, dim=0)
    labels = torch.tensor(labels, dtype=torch.long)
    aug_labels = torch.stack(aug_labels, dim=0)
    
    return images, labels, aug_labels


# =============================================================================
# Argument Parser
# =============================================================================

config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')


parser = argparse.ArgumentParser(description='Augmentation Prediction Training')

# Dataset parameters
group = parser.add_argument_group('Dataset parameters')
parser.add_argument('data', nargs='?', metavar='DIR', const=None,
                    help='path to dataset (positional is *deprecated*, use --data-dir)')
group.add_argument('--data-dir', metavar='DIR',
                    help='path to dataset (root dir)')
group.add_argument('--dataset', metavar='NAME', default='',
                    help='dataset type + name ("<type>/<name>") (default: ImageFolder)')
group.add_argument('--train-split', metavar='NAME', default='train',
                   help='dataset train split (default: train)')
group.add_argument('--val-split', metavar='NAME', default='validation',
                   help='dataset validation split (default: validation)')
group.add_argument('--class-map', default='', type=str, metavar='FILENAME',
                   help='path to class to idx mapping file (default: "")')

# Model parameters
group = parser.add_argument_group('Model parameters')
group.add_argument('--model', default='resnet50', type=str, metavar='MODEL',
                   help='Name of backbone model (default: "resnet50")')
group.add_argument('--pretrained', action='store_true', default=False,
                   help='Start with pretrained version of specified network')
group.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
                   help='Load this checkpoint into model after initialization')
group.add_argument('--resume', default='', type=str, metavar='PATH',
                   help='Resume full model and optimizer state from checkpoint')
group.add_argument('--num-classes', type=int, default=1000, metavar='N',
                   help='number of label classes (default: 1000 for ImageNet)')
group.add_argument('--img-size', type=int, default=224, metavar='N',
                   help='Image size (default: 224)')
group.add_argument('-b', '--batch-size', type=int, default=128, metavar='N',
                   help='Input batch size for training (default: 128)')
group.add_argument('-vb', '--validation-batch-size', type=int, default=None, metavar='N',
                   help='Validation batch size override')

# Augmentation Prediction Task parameters
group = parser.add_argument_group('Augmentation Prediction parameters')
group.add_argument('--fsc-path', type=str, required=True,
                   help='Path to Feature Space Centroid file (e.g., ./FSC/resnet50_FSC.pth)')
group.add_argument('--new-depth', type=int, default=3,
                   help='Maximum number of transforms to apply per image (k value, default: 3)')
group.add_argument('--sl-loss-type', type=str, default='mse', choices=['mse', 'bce'],
                   help='Loss type for SL prediction: "mse" for regression, "bce" for probability (default: mse)')
group.add_argument('--aug-classifier-hidden', type=int, nargs='+', default=[512, 256],
                   help='Hidden layer dimensions for aug classifier (default: 512 256)')
group.add_argument('--aug-classifier-dropout', type=float, default=0.1,
                   help='Dropout rate for aug classifier (default: 0.1)')
group.add_argument('--min-sl', type=float, default=0.1,
                   help='Minimum severity level for augmentation (default: 0.1)')
group.add_argument('--max-sl', type=float, default=1.0,
                   help='Maximum severity level for augmentation (default: 1.0)')
group.add_argument('--val-groups', type=int, default=6,
                   help='Number of validation groups with different fixed augmentations (default: 6)')

# Device & distributed
group = parser.add_argument_group('Device parameters')
group.add_argument('--device', default='cuda', type=str,
                    help="Device (accelerator) to use.")
group.add_argument('--amp', action='store_true', default=False,
                   help='use AMP for mixed precision training')
group.add_argument('--amp-dtype', default='float16', type=str,
                   help='lower precision AMP dtype (default: float16)')
group.add_argument("--local_rank", default=0, type=int)
group.add_argument('--device-modules', default=None, type=str, nargs='+',
                    help="Python imports for device backend modules.")

# Optimizer parameters
group = parser.add_argument_group('Optimizer parameters')
group.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                   help='Optimizer (default: "adamw")')
group.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                   help='learning rate (default: 1e-3)')
group.add_argument('--weight-decay', type=float, default=1e-4,
                   help='weight decay (default: 1e-4)')
group.add_argument('--momentum', type=float, default=0.9, metavar='M',
                   help='Optimizer momentum (default: 0.9)')
group.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                   help='Clip gradient norm (default: None)')

# Learning rate schedule parameters
group = parser.add_argument_group('Learning rate schedule parameters')
group.add_argument('--sched', type=str, default='cosine', metavar='SCHEDULER',
                   help='LR scheduler (default: "cosine"')
group.add_argument('--epochs', type=int, default=100, metavar='N',
                   help='number of epochs to train (default: 100)')
group.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                   help='epochs to warmup LR')
group.add_argument('--warmup-lr', type=float, default=1e-5, metavar='LR',
                   help='warmup learning rate (default: 1e-5)')
group.add_argument('--min-lr', type=float, default=1e-6, metavar='LR',
                   help='lower lr bound for cyclic schedulers (default: 1e-6)')
group.add_argument('--decay-rate', type=float, default=0.1, metavar='RATE',
                   help='LR decay rate (default: 0.1)')

# Misc
group = parser.add_argument_group('Miscellaneous parameters')
group.add_argument('--seed', type=int, default=42, metavar='S',
                   help='random seed (default: 42)')
group.add_argument('--log-interval', type=int, default=50, metavar='N',
                   help='how many batches to wait before logging training status')
group.add_argument('--val-interval', type=int, default=1, metavar='N',
                   help='how many epochs between validation')
group.add_argument('--checkpoint-hist', type=int, default=10, metavar='N',
                   help='number of checkpoints to keep (default: 10)')
group.add_argument('-j', '--workers', type=int, default=4, metavar='N',
                   help='how many training processes to use (default: 4)')
group.add_argument('--pin-mem', action='store_true', default=False,
                   help='Pin CPU memory in DataLoader')
group.add_argument('--output', default='', type=str, metavar='PATH',
                   help='path to output folder (default: none, current dir)')
group.add_argument('--experiment', default='', type=str, metavar='NAME',
                   help='name of train experiment')
group.add_argument('--log-wandb', action='store_true', default=False,
                   help='log training and validation metrics to wandb')
group.add_argument('--wandb-project', default='aug-prediction', type=str,
                   help='wandb project name')


def _parse_args():
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)
    
    args = parser.parse_args(remaining)
    args_text = yaml.safe_dump(args.__dict__, default_flow_style=False)
    return args, args_text


# =============================================================================
# Main Training Functions
# =============================================================================

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
        _logger.info(
            f'Training in distributed mode with multiple processes, 1 device per process. '
            f'Process {args.rank}, total {args.world_size}, device {args.device}.')
    else:
        _logger.info(f'Training with a single process on 1 device ({args.device}).')
    
    # Setup AMP
    amp_dtype = torch.float16
    amp_autocast = suppress
    loss_scaler = None
    if args.amp:
        if args.amp_dtype == 'bfloat16':
            amp_dtype = torch.bfloat16
        amp_autocast = partial(torch.autocast, device_type=device.type, dtype=amp_dtype)
        if device.type == 'cuda' and amp_dtype == torch.float16:
            loss_scaler = NativeScaler(device=device.type)
        _logger.info('Using native Torch AMP.')
    
    utils.random_seed(args.seed, args.rank)
    
    # ==========================================================================
    # Load FSC (Feature Space Centroid)
    # ==========================================================================
    _logger.info(f'Loading FSC from: {args.fsc_path}')
    fsc_data = torch.load(args.fsc_path, map_location='cpu')
    fsc_centroids = fsc_data['centroids'].to(device)  # [num_classes, feature_dim]
    feature_dim = fsc_data['feature_dim']
    _logger.info(f'FSC loaded: {fsc_centroids.shape}, feature_dim={feature_dim}')
    
    # ==========================================================================
    # Create Backbone Model (Frozen)
    # ==========================================================================
    _logger.info(f'Creating backbone model: {args.model}')
    backbone = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.num_classes,
        checkpoint_path=args.initial_checkpoint,
    )
    backbone = backbone.to(device)
    backbone.eval()
    
    # Freeze all backbone parameters
    for param in backbone.parameters():
        param.requires_grad = False
    
    param_count = sum(p.numel() for p in backbone.parameters())
    _logger.info(f'Backbone {args.model} created, param count: {param_count / 1e6:.2f}M (all frozen)')
    
    # Get classifier for label prediction
    backbone_classifier = backbone.get_classifier() if hasattr(backbone, 'get_classifier') else backbone.fc
    
    # ==========================================================================
    # Create Aug Classifier (Trainable)
    # ==========================================================================
    num_transforms = get_augmix_sl_num_transforms()
    transform_names = get_augmix_sl_transform_names()
    _logger.info(f'Number of SL transforms: {num_transforms}')
    _logger.info(f'Transform names: {transform_names}')
    
    # Note: use_sigmoid=False because we use BCEWithLogitsLoss (handles sigmoid internally)
    aug_classifier = AugClassifier(
        feature_dim=feature_dim,
        num_transforms=num_transforms,
        hidden_dims=args.aug_classifier_hidden,
        dropout=args.aug_classifier_dropout,
        use_sigmoid=False,  # Always False - BCEWithLogitsLoss handles sigmoid internally
    )
    aug_classifier = aug_classifier.to(device)
    
    aug_param_count = sum(p.numel() for p in aug_classifier.parameters())
    _logger.info(f'Aug classifier created, param count: {aug_param_count / 1e6:.2f}M (trainable)')
    
    # ==========================================================================
    # Setup Loss Function
    # ==========================================================================
    if args.sl_loss_type == 'bce':
        # BCEWithLogitsLoss is AMP-safe and numerically stable
        # It applies sigmoid internally, so model outputs raw logits
        criterion = nn.BCEWithLogitsLoss()
    else:
        # MSE loss for regression
        criterion = nn.MSELoss()
    criterion = criterion.to(device)
    _logger.info(f'Using {args.sl_loss_type.upper()} loss')
    
    # ==========================================================================
    # Create Optimizer (only for aug_classifier)
    # ==========================================================================
    optimizer = create_optimizer_v2(
        aug_classifier,
        opt=args.opt,
        lr=args.lr,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
    )
    _logger.info(f'Created optimizer: {args.opt}, lr={args.lr}')
    
    # ==========================================================================
    # Resume from checkpoint if specified
    # ==========================================================================
    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        aug_classifier.load_state_dict(checkpoint['aug_classifier'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        _logger.info(f'Resumed from epoch {start_epoch}')
    
    # ==========================================================================
    # Create Datasets and DataLoaders
    # ==========================================================================
    data_config = resolve_data_config(vars(args), model=backbone, verbose=utils.is_primary(args))
    
    if args.data and not args.data_dir:
        args.data_dir = args.data
    
    # Create base transforms
    from torchvision import transforms
    from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
    
    base_transform = transforms.Compose([
        transforms.Resize(int(args.img_size / data_config['crop_pct'])),
        transforms.CenterCrop(args.img_size),
    ])
    
    final_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=data_config['mean'], std=data_config['std']),
    ])
    
    # Create AugMixSL transforms
    augmix_sl_transform = create_augmix_sl_transform(
        max_depth=args.new_depth,
        min_sl=args.min_sl,
        max_sl=args.max_sl,
    )
    
    validation_transforms = create_augmix_sl_validation_transforms(
        num_groups=args.val_groups,
        max_depth=args.new_depth,
    )
    
    # Create raw datasets (without transforms)
    _logger.info(f'Loading dataset from: {args.data_dir}')
    
    # Use ImageFolder directly for more control over transforms
    from torchvision.datasets import ImageFolder
    
    train_dir = os.path.join(args.data_dir, args.train_split)
    val_dir = os.path.join(args.data_dir, args.val_split)
    
    raw_train_dataset = ImageFolder(train_dir)
    raw_val_dataset = ImageFolder(val_dir)
    
    # Wrap with AugMixSL datasets
    dataset_train = AugMixSLDataset(
        raw_train_dataset,
        augmix_sl_transform=augmix_sl_transform,
        base_transform=base_transform,
        final_transform=final_transform,
    )
    
    dataset_eval = AugMixSLValidationDataset(
        raw_val_dataset,
        validation_transforms=validation_transforms,
        base_transform=base_transform,
        final_transform=final_transform,
    )
    
    _logger.info(f'Train dataset size: {len(dataset_train)}')
    _logger.info(f'Val dataset size: {len(dataset_eval)} ({args.val_groups} groups)')
    
    # Create DataLoaders
    sampler_train = None
    sampler_eval = None
    if args.distributed:
        sampler_train = torch.utils.data.distributed.DistributedSampler(dataset_train)
        sampler_eval = torch.utils.data.distributed.DistributedSampler(dataset_eval, shuffle=False)
    
    loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=(sampler_train is None),
        sampler=sampler_train,
        num_workers=args.workers,
        pin_memory=args.pin_mem,
        collate_fn=collate_aug_labels,
        drop_last=True,
    )
    
    loader_eval = torch.utils.data.DataLoader(
        dataset_eval,
        batch_size=args.validation_batch_size or args.batch_size,
        shuffle=False,
        sampler=sampler_eval,
        num_workers=args.workers,
        pin_memory=args.pin_mem,
        collate_fn=collate_aug_labels,
    )
    
    # ==========================================================================
    # Setup Learning Rate Scheduler
    # ==========================================================================
    updates_per_epoch = len(loader_train)
    lr_scheduler, num_epochs = create_scheduler_v2(
        optimizer,
        sched=args.sched,
        num_epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        warmup_lr=args.warmup_lr,
        min_lr=args.min_lr,
        decay_rate=args.decay_rate,
        updates_per_epoch=updates_per_epoch,
    )
    
    if start_epoch > 0 and lr_scheduler is not None:
        lr_scheduler.step(start_epoch)
    
    _logger.info(f'Scheduled epochs: {num_epochs}')
    
    # ==========================================================================
    # Setup Checkpoint Saver
    # ==========================================================================
    best_metric = None
    best_epoch = None
    output_dir = None
    
    if utils.is_primary(args):
        if args.experiment:
            exp_name = args.experiment
        else:
            exp_name = '-'.join([
                datetime.now().strftime("%Y%m%d-%H%M%S"),
                'aug-pred',
                safe_model_name(args.model),
            ])
        output_dir = utils.get_outdir(args.output if args.output else './output/train_augmix', exp_name)
        
        with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
            f.write(args_text)
        
        if args.log_wandb and has_wandb:
            wandb.init(project=args.wandb_project, name=exp_name, config=args)
    
    # ==========================================================================
    # Training Loop
    # ==========================================================================
    results = []
    try:
        for epoch in range(start_epoch, num_epochs):
            if args.distributed:
                sampler_train.set_epoch(epoch)
            
            train_metrics = train_one_epoch(
                epoch=epoch,
                backbone=backbone,
                backbone_classifier=backbone_classifier,
                aug_classifier=aug_classifier,
                fsc_centroids=fsc_centroids,
                loader=loader_train,
                optimizer=optimizer,
                criterion=criterion,
                args=args,
                device=device,
                lr_scheduler=lr_scheduler,
                amp_autocast=amp_autocast,
                loss_scaler=loss_scaler,
            )
            
            if lr_scheduler is not None:
                lr_scheduler.step(epoch + 1)
            
            # Validation
            if (epoch + 1) % args.val_interval == 0 or epoch == num_epochs - 1:
                eval_metrics = validate(
                    backbone=backbone,
                    backbone_classifier=backbone_classifier,
                    aug_classifier=aug_classifier,
                    fsc_centroids=fsc_centroids,
                    loader=loader_eval,
                    criterion=criterion,
                    args=args,
                    device=device,
                    amp_autocast=amp_autocast,
                    transform_names=transform_names,
                )
                
                # Save checkpoint
                if utils.is_primary(args) and output_dir:
                    checkpoint = {
                        'epoch': epoch,
                        'aug_classifier': aug_classifier.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'args': args,
                    }
                    
                    # Save latest
                    torch.save(checkpoint, os.path.join(output_dir, 'last.pth.tar'))
                    
                    # Save best
                    current_metric = eval_metrics['aug_acc']
                    if best_metric is None or current_metric > best_metric:
                        best_metric = current_metric
                        best_epoch = epoch
                        torch.save(checkpoint, os.path.join(output_dir, 'best.pth.tar'))
                        _logger.info(f'New best: {best_metric:.2f}% at epoch {epoch}')
                    
                    # Save periodic
                    if (epoch + 1) % 10 == 0:
                        torch.save(checkpoint, os.path.join(output_dir, f'checkpoint-{epoch}.pth.tar'))
                    
                    # Log to wandb
                    if args.log_wandb and has_wandb:
                        wandb.log({
                            'epoch': epoch,
                            'train_loss': train_metrics['loss'],
                            'val_loss': eval_metrics['loss'],
                            'val_aug_acc': eval_metrics['aug_acc'],
                            'val_sl_mae': eval_metrics['sl_mae'],
                            'lr': optimizer.param_groups[0]['lr'],
                        })
                
                results.append({
                    'epoch': epoch,
                    'train': train_metrics,
                    'validation': eval_metrics,
                })
    
    except KeyboardInterrupt:
        pass
    
    if args.distributed:
        torch.distributed.destroy_process_group()
    
    if best_metric is not None:
        _logger.info(f'*** Best aug_acc: {best_metric:.2f}% (epoch {best_epoch})')


def train_one_epoch(
    epoch,
    backbone,
    backbone_classifier,
    aug_classifier,
    fsc_centroids,
    loader,
    optimizer,
    criterion,
    args,
    device,
    lr_scheduler=None,
    amp_autocast=suppress,
    loss_scaler=None,
):
    """Train for one epoch."""
    
    losses_m = utils.AverageMeter()
    batch_time_m = utils.AverageMeter()
    data_time_m = utils.AverageMeter()
    
    backbone.eval()  # Keep backbone in eval mode
    aug_classifier.train()
    
    end = time.time()
    num_batches = len(loader)
    
    for batch_idx, (images, labels, aug_labels) in enumerate(loader):
        data_time_m.update(time.time() - end)
        
        images = images.to(device)
        labels = labels.to(device)
        aug_labels = aug_labels.to(device)
        
        with amp_autocast():
            # Extract features from frozen backbone
            with torch.no_grad():
                features = backbone.forward_features(images)
                features = backbone.forward_head(features, pre_logits=True)
                
                # Get predicted labels
                logits = backbone_classifier(features)
                pred_labels = logits.argmax(dim=1)
            
            # Compute FSC_diff
            # fsc_centroids: [num_classes, feature_dim]
            # features: [batch_size, feature_dim]
            # pred_labels: [batch_size]
            fsc_for_batch = fsc_centroids[pred_labels]  # [batch_size, feature_dim]
            fsc_diff = features - fsc_for_batch  # [batch_size, feature_dim]
            
            # Predict augmentation labels
            aug_pred = aug_classifier(fsc_diff)
            
            # Compute loss
            loss = criterion(aug_pred, aug_labels)
        
        # Backward pass
        optimizer.zero_grad()
        if loss_scaler is not None:
            loss_scaler(
                loss,
                optimizer,
                clip_grad=args.clip_grad,
                parameters=aug_classifier.parameters(),
            )
        else:
            loss.backward()
            if args.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(aug_classifier.parameters(), args.clip_grad)
            optimizer.step()
        
        losses_m.update(loss.item(), images.size(0))
        batch_time_m.update(time.time() - end)
        end = time.time()
        
        if batch_idx % args.log_interval == 0 or batch_idx == num_batches - 1:
            lr = optimizer.param_groups[0]['lr']
            if utils.is_primary(args):
                _logger.info(
                    f'Train: {epoch} [{batch_idx:>4d}/{num_batches}]  '
                    f'Loss: {losses_m.val:.4f} ({losses_m.avg:.4f})  '
                    f'Time: {batch_time_m.val:.3f}s  '
                    f'LR: {lr:.2e}'
                )
    
    return OrderedDict([('loss', losses_m.avg)])


def validate(
    backbone,
    backbone_classifier,
    aug_classifier,
    fsc_centroids,
    loader,
    criterion,
    args,
    device,
    amp_autocast=suppress,
    transform_names=None,
):
    """Validate the model."""
    
    losses_m = utils.AverageMeter()
    aug_correct_m = utils.AverageMeter()  # Augmentation type accuracy
    sl_mae_m = utils.AverageMeter()  # Severity level MAE (for applied augs)
    
    backbone.eval()
    aug_classifier.eval()
    
    num_batches = len(loader)
    
    with torch.no_grad():
        for batch_idx, (images, labels, aug_labels) in enumerate(loader):
            images = images.to(device)
            labels = labels.to(device)
            aug_labels = aug_labels.to(device)
            
            with amp_autocast():
                # Extract features
                features = backbone.forward_features(images)
                features = backbone.forward_head(features, pre_logits=True)
                
                # Get predicted labels
                logits = backbone_classifier(features)
                pred_labels = logits.argmax(dim=1)
                
                # Compute FSC_diff
                fsc_for_batch = fsc_centroids[pred_labels]
                fsc_diff = features - fsc_for_batch
                
                # Predict augmentation labels
                aug_pred = aug_classifier(fsc_diff)
                
                # Compute loss
                loss = criterion(aug_pred, aug_labels)
            
            losses_m.update(loss.item(), images.size(0))
            
            # Compute augmentation type accuracy
            # Ground truth: aug_labels > 0 means transform was applied
            gt_applied = (aug_labels > 0).float()
            
            # For BCE loss, apply sigmoid to get probabilities; for MSE, threshold at 0.05
            if args.sl_loss_type == 'bce':
                # aug_pred is raw logits, apply sigmoid for probability
                aug_pred_prob = torch.sigmoid(aug_pred)
                pred_applied = (aug_pred_prob > 0.5).float()
            else:
                aug_pred_prob = aug_pred  # Already in [0, 1] range for MSE
                pred_applied = (aug_pred > 0.05).float()
            
            # Per-transform accuracy
            correct = (pred_applied == gt_applied).float().mean() * 100
            aug_correct_m.update(correct.item(), images.size(0))
            
            # Compute SL MAE for applied transforms only
            # Use aug_pred_prob (sigmoid-applied for BCE, raw for MSE)
            mask = aug_labels > 0
            if mask.sum() > 0:
                sl_mae = (aug_pred_prob[mask] - aug_labels[mask]).abs().mean()
                sl_mae_m.update(sl_mae.item(), mask.sum().item())
    
    # Log final metrics
    if utils.is_primary(args):
        _logger.info(
            f'Validation: Loss: {losses_m.avg:.4f}  '
            f'Aug Acc: {aug_correct_m.avg:.2f}%  '
            f'SL MAE: {sl_mae_m.avg:.4f}'
        )
    
    return OrderedDict([
        ('loss', losses_m.avg),
        ('aug_acc', aug_correct_m.avg),
        ('sl_mae', sl_mae_m.avg),
    ])


if __name__ == '__main__':
    main()

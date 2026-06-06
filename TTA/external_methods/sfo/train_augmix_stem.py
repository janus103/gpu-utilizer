#!/usr/bin/env python3
"""Augmentation Prediction Training Script (Stem Feature Version).

This script trains a model to predict which augmentations were applied to an image
and their severity levels. Unlike train_augmix.py which uses final layer features,
this script uses STEM (first conv layer) features for FSC computation.

Key concepts:
- Backbone (e.g., ResNet50) is frozen and used only for feature extraction
- StemFeatureExtractor extracts 1024-dim features from first conv layer
- FSC_diff = current_stem_feature - FSC_stem[predicted_label] is computed
- Aug classifier predicts [Aug_type, Normalized_SL] from FSC_diff
- Uses V2 augmentation policy (IntensityIncreasing, SaturationIncreasing, etc.)

Usage:
    python train_augmix_stem.py --data-dir /path/to/imagenet --model resnet50 --pretrained \
        --fsc-path ./FSC/resnet50_FSC_stem.pth --new-depth 3

Hacked together by / Copyright 2020 Ross Wightman (https://github.com/rwightman)
Modified for augmentation prediction task with stem features and V2 policy.
"""
import argparse
import copy
import gc
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
    AUGMIX_SL_V2_NUM_TRANSFORMS,
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


_logger = logging.getLogger('train_augmix_stem')


# =============================================================================
# Stem Feature Extractor (from compute_fsc_stem.py)
# =============================================================================

class StemFeatureExtractor(nn.Module):
    """Extract features from the first conv layer (stem) of a ResNet model.
    
    For ResNet50 with 224x224 input:
    - conv1 (7x7, stride=2, padding=3): 3 -> 64 channels, 112x112 spatial
    - AdaptiveAvgPool2d(4, 4): 64 channels, 4x4 spatial
    - Flatten: 64 * 4 * 4 = 1024-dim vector
    """
    
    def __init__(self, model):
        super().__init__()
        # Get the first conv layer from the model
        if hasattr(model, 'conv1'):
            self.conv1 = model.conv1
        else:
            raise ValueError("Model does not have conv1 layer")
        
        # Get batch norm and activation after conv1
        if hasattr(model, 'bn1'):
            self.bn1 = model.bn1
        else:
            self.bn1 = nn.Identity()
        
        if hasattr(model, 'act1'):
            self.act1 = model.act1
        else:
            self.act1 = nn.ReLU(inplace=True)
        
        # Adaptive average pooling to get 4x4 spatial output
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        
    def forward(self, x):
        # Ensure input is contiguous (required for cuDNN with AMP)
        x = x.contiguous()
        
        # Apply first conv layer
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        
        # Pool to 4x4 spatial
        x = self.pool(x)
        
        # Flatten to [batch, 64 * 4 * 4] = [batch, 1024]
        x = x.flatten(1)
        
        return x


# =============================================================================
# FSC Difference Computation Functions
# =============================================================================

def compute_fsc_diff(features: torch.Tensor, fsc: torch.Tensor, mode: str = 'subtract') -> torch.Tensor:
    """Compute the difference between features and FSC centroids.
    
    Args:
        features: Current image features [batch_size, feature_dim].
        fsc: FSC centroids for the batch [batch_size, feature_dim].
        mode: Difference computation mode:
            - 'subtract': Simple subtraction (features - fsc)
            - 'orthogonal': Orthogonal component of (features - fsc) w.r.t. fsc direction
              This captures the deviation perpendicular to the class prototype.
    
    Returns:
        fsc_diff: Difference tensor [batch_size, feature_dim].
    """
    if mode == 'subtract':
        # Simple subtraction: captures total deviation from centroid
        return features - fsc
    
    elif mode == 'orthogonal':
        # Orthogonal projection: captures deviation perpendicular to FSC direction
        # This isolates the component that is NOT explained by the class prototype
        #
        # diff = features - fsc
        # diff_parallel = proj_{fsc}(diff) = (diff · fsc / ||fsc||^2) * fsc
        # diff_orthogonal = diff - diff_parallel
        #
        # This captures augmentation-induced changes that are orthogonal to
        # the class-specific direction, potentially more informative for
        # augmentation detection.
        
        diff = features - fsc
        
        # Compute ||fsc||^2 for each sample
        fsc_norm_sq = (fsc * fsc).sum(dim=1, keepdim=True) + 1e-8  # [batch, 1]
        
        # Compute projection coefficient: (diff · fsc) / ||fsc||^2
        proj_coeff = (diff * fsc).sum(dim=1, keepdim=True) / fsc_norm_sq  # [batch, 1]
        
        # Compute parallel component
        diff_parallel = proj_coeff * fsc  # [batch, feature_dim]
        
        # Compute orthogonal component
        diff_orthogonal = diff - diff_parallel  # [batch, feature_dim]
        
        return diff_orthogonal
    
    else:
        raise ValueError(f"Unknown fsc_diff_mode: {mode}. Choose 'subtract' or 'orthogonal'.")


# =============================================================================
# Soft Cross Entropy Loss for Probability Distributions
# =============================================================================

class SoftCrossEntropyLoss(nn.Module):
    """Cross Entropy Loss for soft labels (probability distributions).
    
    Computes: -sum(target * log_softmax(pred)) averaged over batch.
    
    This is equivalent to CE = H(target) + KL(target || softmax(pred)),
    where H(target) is the entropy of the target distribution.
    
    Since H(target) is constant w.r.t. model parameters, this produces
    the same gradients as KLDivLoss, but with different loss values.
    
    Args:
        reduction: 'mean' (default) or 'sum'.
    """
    
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Raw logits [batch_size, num_classes].
            target: Soft labels (probability distribution) [batch_size, num_classes].
                    Should sum to 1 along dim=1.
        
        Returns:
            Cross entropy loss.
        """
        log_probs = F.log_softmax(pred, dim=1)
        loss = -(target * log_probs).sum(dim=1)
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


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
        feature_dim: Dimension of input features (1024 for stem features).
        num_transforms: Number of transform types to predict (7 for V2 policy).
        hidden_dims: List of hidden layer dimensions.
        dropout: Dropout probability.
        use_sigmoid: If True, apply sigmoid to output (for BCE loss).
    """
    
    def __init__(
        self,
        feature_dim: int,
        num_transforms: int = AUGMIX_SL_V2_NUM_TRANSFORMS,
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


class SingleTransformValidationDataset(torch.utils.data.Dataset):
    """Validation dataset that applies a SINGLE transform with fixed SL to all images.
    
    This is used for per-transform accuracy evaluation.
    Each transform is tested individually across all images.
    
    Args:
        dataset: Original validation dataset.
        transform_op: Single AugMixSLOp instance to apply.
        transform_idx: Index of this transform in the transform list.
        num_transforms: Total number of transforms (for label tensor size).
        severity_level: Fixed severity level to apply (default: 0.5 = average).
        max_sl: Maximum severity level for scaling (SL is scaled by max_sl so max_sl maps to 1.0).
        base_transform: Base transforms to apply before augmentation.
        final_transform: Final transforms (ToTensor, Normalize).
    """
    
    def __init__(
        self,
        dataset,
        transform_op,
        transform_idx: int,
        num_transforms: int,
        severity_level: float = 0.5,
        max_sl: float = 1.0,
        base_transform=None,
        final_transform=None,
    ):
        self.dataset = dataset
        self.transform_op = transform_op
        self.transform_idx = transform_idx
        self.num_transforms = num_transforms
        self.severity_level = severity_level
        self.max_sl = max_sl
        self.base_transform = base_transform
        self.final_transform = final_transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        
        # Apply base transform (resize, crop)
        if self.base_transform is not None:
            img = self.base_transform(img)
        
        # Apply single transform with fixed SL
        img = self.transform_op(img, self.severity_level)
        
        # Create ground truth label tensor
        # Scale SL by max_sl so that max_sl maps to 1.0 (consistent with training)
        scaled_sl = self.severity_level / self.max_sl
        aug_labels = np.zeros(self.num_transforms, dtype=np.float32)
        aug_labels[self.transform_idx] = scaled_sl
        
        # Apply final transform (ToTensor, Normalize)
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


parser = argparse.ArgumentParser(description='Augmentation Prediction Training (Stem Features + V2 Policy)')

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
                   help='Path to Stem FSC file (e.g., ./FSC/resnet50_FSC_stem.pth)')
group.add_argument('--new-depth', type=int, default=3,
                   help='Maximum number of transforms to apply per image (k value, default: 3)')
group.add_argument('--sl-loss-type', type=str, default='mse', choices=['mse', 'bce', 'kldiv', 'ce'],
                   help='Loss type for SL prediction: "mse" for regression, "bce" for multi-label probability, '
                        '"kldiv" for KL divergence, "ce" for cross entropy '
                        '(kldiv/ce recommended with --normalize-aug-labels) (default: mse)')
group.add_argument('--aug-classifier-hidden', type=int, nargs='+', default=[512, 256],
                   help='Hidden layer dimensions for aug classifier (default: 512 256)')
group.add_argument('--aug-classifier-dropout', type=float, default=0.1,
                   help='Dropout rate for aug classifier (default: 0.1)')
group.add_argument('--min-sl', type=float, default=0.1,
                   help='Minimum severity level for augmentation (default: 0.1)')
group.add_argument('--max-sl', type=float, default=1.0,
                   help='Maximum severity level for augmentation (default: 1.0)')
group.add_argument('--normalize-aug-labels', action='store_true', default=False,
                   help='Normalize aug_labels to sum to 1 (probability distribution). '
                        'Useful for cross-entropy style learning where labels represent softmax probabilities.')
group.add_argument('--val-groups', type=int, default=6,
                   help='Number of validation groups with different fixed augmentations (default: 6)')
group.add_argument('--fsc-diff-mode', type=str, default='subtract', choices=['subtract', 'orthogonal'],
                   help='FSC difference computation mode: "subtract" for simple subtraction, '
                        '"orthogonal" for orthogonal projection (default: subtract)')

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
group.add_argument('-j', '--workers', type=int, default=None, metavar='N',
                   help='how many training processes to use (default: half of CPU cores)')
group.add_argument('--pin-mem', action='store_true', default=False,
                   help='Pin CPU memory in DataLoader')
group.add_argument('--output', default='', type=str, metavar='PATH',
                   help='path to output folder (default: none, current dir)')
group.add_argument('--experiment', default='', type=str, metavar='NAME',
                   help='name of train experiment')
group.add_argument('--log-wandb', action='store_true', default=False,
                   help='log training and validation metrics to wandb')
group.add_argument('--wandb-project', default='aug-prediction-stem', type=str,
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
    
    # Auto-set workers to half of CPU cores if not specified (capped at 16 to avoid deadlock)
    if args.workers is None:
        max_workers = 16  # Upper limit to prevent shared memory issues / deadlock
        args.workers = min(max(1, os.cpu_count() // 2), max_workers)
        _logger.info(
            f'Auto-setting workers to {args.workers} '
            f'(half of {os.cpu_count()} CPU cores, capped at {max_workers})'
        )
    
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
    # Load FSC (Feature Space Centroid) from Stem
    # ==========================================================================
    _logger.info(f'Loading Stem FSC from: {args.fsc_path}')
    fsc_data = torch.load(args.fsc_path, map_location='cpu')
    fsc_centroids = fsc_data['centroids'].to(device)  # [num_classes, stem_feature_dim]
    feature_dim = fsc_data['feature_dim']  # Should be 1024 for stem
    _logger.info(f'Stem FSC loaded: {fsc_centroids.shape}, feature_dim={feature_dim}')
    
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
    
    # ==========================================================================
    # Create Stem Feature Extractor (Frozen)
    # ==========================================================================
    _logger.info('Creating stem feature extractor...')
    stem_extractor = StemFeatureExtractor(backbone)
    stem_extractor = stem_extractor.to(device)
    stem_extractor.eval()
    
    # Freeze stem extractor (it uses backbone's conv1, bn1, act1 which are already frozen)
    for param in stem_extractor.parameters():
        param.requires_grad = False
    
    _logger.info(f'Stem feature extractor created (output dim: {feature_dim})')
    
    # ==========================================================================
    # Create Aug Classifier (Trainable) - Using V2 Policy
    # ==========================================================================
    num_transforms = get_augmix_sl_num_transforms(version=2)
    transform_names = get_augmix_sl_transform_names(version=2)
    _logger.info(f'Using V2 policy with {num_transforms} transforms: {transform_names}')
    
    # Note: use_sigmoid=False because we use BCEWithLogitsLoss (handles sigmoid internally)
    aug_classifier = AugClassifier(
        feature_dim=feature_dim,  # 1024 for stem
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
        if args.normalize_aug_labels:
            _logger.warning(
                'Using BCE loss with --normalize-aug-labels may not be ideal. '
                'Consider using --sl-loss-type kldiv or ce for probability distributions.'
            )
    elif args.sl_loss_type == 'kldiv':
        # KL Divergence for comparing probability distributions
        # Model should output log_softmax, target should be normalized (sum to 1)
        criterion = nn.KLDivLoss(reduction='batchmean')
        if not args.normalize_aug_labels:
            _logger.warning(
                'Using KL divergence loss without --normalize-aug-labels. '
                'Labels may not sum to 1, which is required for proper KL divergence.'
            )
    elif args.sl_loss_type == 'ce':
        # Soft Cross Entropy for probability distributions
        # CE = H(target) + KL(target || pred), same gradients as KL but different loss values
        criterion = SoftCrossEntropyLoss(reduction='mean')
        if not args.normalize_aug_labels:
            _logger.warning(
                'Using Cross Entropy loss without --normalize-aug-labels. '
                'Labels may not sum to 1, which is required for proper cross entropy.'
            )
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
    
    # Create AugMixSL transforms with V2 policy
    augmix_sl_transform = create_augmix_sl_transform(
        max_depth=args.new_depth,
        min_sl=args.min_sl,
        max_sl=args.max_sl,
        version=2,  # Use V2 policy
        normalize_labels=args.normalize_aug_labels,
    )
    
    # Get V2 transform ops for per-transform validation
    from timm.data import augmix_sl_ops_v2
    val_transform_ops = augmix_sl_ops_v2()
    val_severity_level = args.max_sl  # Use max SL for validation (stronger signal)
    
    # Create raw datasets (without transforms)
    _logger.info(f'Loading dataset from: {args.data_dir}')
    
    # Use ImageFolder directly for more control over transforms
    from torchvision.datasets import ImageFolder
    
    train_dir = os.path.join(args.data_dir, args.train_split)
    val_dir = os.path.join(args.data_dir, args.val_split)
    
    raw_train_dataset = ImageFolder(train_dir)
    raw_val_dataset = ImageFolder(val_dir)
    
    # Wrap training dataset with AugMixSL
    dataset_train = AugMixSLDataset(
        raw_train_dataset,
        augmix_sl_transform=augmix_sl_transform,
        base_transform=base_transform,
        final_transform=final_transform,
    )
    
    _logger.info(f'Train dataset: {len(dataset_train)} samples')
    _logger.info(f'Val dataset: {len(raw_val_dataset)} samples (per-transform evaluation)')
    _logger.info(f'Validation transforms ({num_transforms}): {transform_names}')
    _logger.info(f'Validation SL: {val_severity_level}')
    
    # Create training data loader
    sampler_train = None
    if args.distributed:
        sampler_train = torch.utils.data.distributed.DistributedSampler(dataset_train)
    
    loader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=(sampler_train is None),
        sampler=sampler_train,
        num_workers=args.workers,
        pin_memory=args.pin_mem,
        collate_fn=collate_aug_labels,
        drop_last=True,
        persistent_workers=True if args.workers > 0 else False,  # Reuse workers (faster startup)
        prefetch_factor=2 if args.workers > 0 else None,  # Reduced to prevent memory pressure
    )
    
    # Note: Validation loaders are created per-transform inside validate_per_transform()
    
    # ==========================================================================
    # Setup Learning Rate Scheduler
    # ==========================================================================
    lr_scheduler = None
    if args.sched:
        lr_scheduler, num_epochs = create_scheduler_v2(
            optimizer,
            sched=args.sched,
            num_epochs=args.epochs,
            warmup_epochs=args.warmup_epochs,
            warmup_lr=args.warmup_lr,
            min_lr=args.min_lr,
            decay_rate=args.decay_rate,
        )
    else:
        num_epochs = args.epochs
    
    # ==========================================================================
    # Setup Output Directory
    # ==========================================================================
    output_dir = None
    if args.output:
        output_dir = Path(args.output)
    else:
        exp_name = args.experiment or f'augmix_stem_{args.model}'
        output_dir = Path(f'./output/{exp_name}')
    
    if utils.is_primary(args):
        output_dir.mkdir(parents=True, exist_ok=True)
        # Save args
        with open(output_dir / 'args.yaml', 'w') as f:
            f.write(args_text)
        _logger.info(f'Output directory: {output_dir}')
    
    # ==========================================================================
    # Setup Wandb
    # ==========================================================================
    if args.log_wandb and utils.is_primary(args) and has_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.experiment or f'augmix_stem_{args.model}',
            config=args,
        )
    
    # ==========================================================================
    # Training Loop
    # ==========================================================================
    _logger.info(f'Starting training for {num_epochs} epochs')
    
    best_metric = None
    best_epoch = None
    results = []
    
    # ==========================================================================
    # Initialize validation results file (tab-separated table)
    # ==========================================================================
    val_results_path = None
    if utils.is_primary(args) and output_dir:
        val_results_path = os.path.join(output_dir, 'val_results.txt')
        # Write header (overwrite if starting fresh, append if resuming)
        if start_epoch == 0 or not os.path.exists(val_results_path):
            with open(val_results_path, 'w') as f:
                header_cols = ['epoch', 'train_loss', 'lr'] + list(transform_names) + ['mean_acc', 'best']
                f.write('\t'.join(header_cols) + '\n')
            _logger.info(f'Validation results will be saved to: {val_results_path}')
    
    try:
        for epoch in range(start_epoch, num_epochs):
            if args.distributed:
                loader_train.sampler.set_epoch(epoch)
            
            # Train
            train_metrics = train_one_epoch(
                epoch=epoch,
                stem_extractor=stem_extractor,
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
            
            # Validation (per-transform evaluation)
            if (epoch + 1) % args.val_interval == 0 or epoch == num_epochs - 1:
                eval_metrics = validate_per_transform(
                    stem_extractor=stem_extractor,
                    aug_classifier=aug_classifier,
                    fsc_centroids=fsc_centroids,
                    raw_val_dataset=raw_val_dataset,
                    val_transform_ops=val_transform_ops,
                    transform_names=transform_names,
                    num_transforms=num_transforms,
                    severity_level=val_severity_level,
                    base_transform=base_transform,
                    final_transform=final_transform,
                    criterion=criterion,
                    args=args,
                    device=device,
                    amp_autocast=amp_autocast,
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
                    
                    # Save best (based on mean accuracy across all transforms)
                    current_metric = eval_metrics['mean_acc']
                    if best_metric is None or current_metric > best_metric:
                        best_metric = current_metric
                        best_epoch = epoch
                        torch.save(checkpoint, os.path.join(output_dir, 'best.pth.tar'))
                        _logger.info(f'New best mean_acc: {best_metric:.2f}% at epoch {epoch}')
                    
                    # Save periodic
                    if (epoch + 1) % 10 == 0:
                        torch.save(checkpoint, os.path.join(output_dir, f'checkpoint-{epoch}.pth.tar'))
                    
                    # Log to wandb
                    if args.log_wandb and has_wandb:
                        log_dict = {
                            'epoch': epoch,
                            'train_loss': train_metrics['loss'],
                            'val_mean_acc': eval_metrics['mean_acc'],
                            'lr': optimizer.param_groups[0]['lr'],
                        }
                        # Add per-transform accuracies
                        for t_name in transform_names:
                            log_dict[f'val_acc_{t_name}'] = eval_metrics['per_transform_acc'].get(t_name, 0)
                        wandb.log(log_dict)
                
                    # Append validation results to table file
                    if val_results_path is not None:
                        is_best = '*' if (best_epoch == epoch) else ''
                        row_cols = [
                            str(epoch),
                            f'{train_metrics["loss"]:.4f}',
                            f'{optimizer.param_groups[0]["lr"]:.2e}',
                        ]
                        for t_name in transform_names:
                            row_cols.append(f'{eval_metrics["per_transform_acc"].get(t_name, 0):.2f}')
                        row_cols.append(f'{eval_metrics["mean_acc"]:.2f}')
                        row_cols.append(is_best)
                        with open(val_results_path, 'a') as f:
                            f.write('\t'.join(row_cols) + '\n')
                
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
        _logger.info(f'*** Best mean_acc: {best_metric:.2f}% (epoch {best_epoch})')


def train_one_epoch(
    epoch,
    stem_extractor,
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
    """Train for one epoch using stem features.
    
    Note: Uses ground truth labels for FSC selection (no backbone forward needed).
    """
    
    losses_m = utils.AverageMeter()
    batch_time_m = utils.AverageMeter()
    data_time_m = utils.AverageMeter()
    
    stem_extractor.eval()  # Keep stem extractor in eval mode
    aug_classifier.train()
    
    end = time.time()
    num_batches = len(loader)
    
    for batch_idx, (images, labels, aug_labels) in enumerate(loader):
        data_time_m.update(time.time() - end)
        
        images = images.to(device)
        labels = labels.to(device)
        aug_labels = aug_labels.to(device)
        
        with amp_autocast():
            # Extract stem features from frozen stem extractor
            # Note: During training, we use ground truth labels for FSC selection
            # This avoids running the full backbone network for classification
            with torch.no_grad():
                # Get stem features (1024-dim)
                stem_features = stem_extractor(images)
            
            # Select FSC centroids using GROUND TRUTH labels
            fsc_for_batch = fsc_centroids[labels]  # [batch_size, stem_feature_dim]
            
            # Compute FSC_diff based on specified mode (subtract or orthogonal)
            fsc_diff = compute_fsc_diff(stem_features, fsc_for_batch, mode=args.fsc_diff_mode)
            
            # Predict augmentation labels
            aug_pred = aug_classifier(fsc_diff)
            
            # Compute loss
            if args.sl_loss_type == 'kldiv':
                # KL divergence expects log probabilities vs probability targets
                aug_pred_log = F.log_softmax(aug_pred, dim=1)
                loss = criterion(aug_pred_log, aug_labels)
            else:
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


def validate_per_transform(
    stem_extractor,
    aug_classifier,
    fsc_centroids,
    raw_val_dataset,
    val_transform_ops,
    transform_names,
    num_transforms,
    severity_level,
    base_transform,
    final_transform,
    criterion,
    args,
    device,
    amp_autocast=suppress,
):
    """Validate the model by testing each transform individually.
    
    For each transform:
    1. Apply that single transform (with fixed SL) to all validation images
    2. Measure whether the model correctly identifies which transform was applied
    
    Returns mean accuracy across all transforms.
    
    Args:
        stem_extractor: Stem feature extractor.
        aug_classifier: Augmentation classifier.
        fsc_centroids: FSC centroids tensor.
        raw_val_dataset: Raw validation ImageFolder dataset.
        val_transform_ops: List of AugMixSLOp instances for V2 transforms.
        transform_names: List of transform names.
        num_transforms: Number of transforms.
        severity_level: Fixed severity level for validation.
        base_transform: Base image transforms.
        final_transform: Final image transforms (ToTensor, Normalize).
        criterion: Loss function.
        args: Training arguments.
        device: Device to use.
        amp_autocast: AMP autocast context.
    
    Returns:
        OrderedDict with 'mean_acc' and 'per_transform_acc'.
    """
    
    stem_extractor.eval()
    aug_classifier.eval()
    
    per_transform_acc = {}
    per_transform_correct = {}
    per_transform_total = {}
    
    if utils.is_primary(args):
        _logger.info(f'Per-transform validation (SL={severity_level:.2f}):')
    
    with torch.no_grad():
        for t_idx, (t_op, t_name) in enumerate(zip(val_transform_ops, transform_names)):
            # Create dataset for this single transform
            single_transform_dataset = SingleTransformValidationDataset(
                dataset=raw_val_dataset,
                transform_op=t_op,
                transform_idx=t_idx,
                num_transforms=num_transforms,
                severity_level=severity_level,
                max_sl=args.max_sl,
                base_transform=base_transform,
                final_transform=final_transform,
            )
            
            # Create data loader
            sampler = None
            if args.distributed:
                sampler = torch.utils.data.distributed.DistributedSampler(
                    single_transform_dataset, shuffle=False
                )
            
            loader = torch.utils.data.DataLoader(
                single_transform_dataset,
                batch_size=args.validation_batch_size or args.batch_size,
                shuffle=False,
                sampler=sampler,
                num_workers=args.workers,
                pin_memory=args.pin_mem,
                collate_fn=collate_aug_labels,
                persistent_workers=False,  # Must be False: new DataLoader created each iteration
                prefetch_factor=2 if args.workers > 0 else None,
            )
            
            # Evaluate on this transform
            correct_count = 0
            total_count = 0
            
            for batch_idx, (images, labels, aug_labels) in enumerate(loader):
                images = images.to(device)
                labels = labels.to(device)
                aug_labels = aug_labels.to(device)
                
                with amp_autocast():
                    # Extract stem features
                    stem_features = stem_extractor(images)
                    
                    # Select FSC centroids using GROUND TRUTH labels
                    fsc_for_batch = fsc_centroids[labels]
                    
                    # Compute FSC_diff
                    fsc_diff = compute_fsc_diff(stem_features, fsc_for_batch, mode=args.fsc_diff_mode)
                    
                    # Predict augmentation labels
                    aug_pred = aug_classifier(fsc_diff)
                
                # Get predictions for the current transform only
                # For this single-transform validation, we check if the model
                # correctly identifies that THIS transform was applied
                if args.sl_loss_type == 'bce':
                    aug_pred_prob = torch.sigmoid(aug_pred)
                elif args.sl_loss_type in ('kldiv', 'ce'):
                    # Both KL divergence and Cross Entropy use softmax for probability output
                    aug_pred_prob = F.softmax(aug_pred, dim=1)
                else:
                    aug_pred_prob = aug_pred
                
                # Check if the correct transform has highest prediction
                # (since only one transform is applied, it should be predicted highest)
                pred_transform_idx = aug_pred_prob.argmax(dim=1)
                correct = (pred_transform_idx == t_idx).sum().item()
                
                correct_count += correct
                total_count += images.size(0)
            
            # Calculate accuracy for this transform
            transform_acc = 100.0 * correct_count / total_count if total_count > 0 else 0.0
            per_transform_acc[t_name] = transform_acc
            per_transform_correct[t_name] = correct_count
            per_transform_total[t_name] = total_count
            
            if utils.is_primary(args):
                _logger.info(f'  {t_name}: {transform_acc:.2f}% ({correct_count}/{total_count})')
            
            # Explicit memory cleanup to prevent OOM across transform iterations
            del loader
            del single_transform_dataset
            del sampler
            gc.collect()
            torch.cuda.empty_cache()
    
    # Calculate mean accuracy
    mean_acc = sum(per_transform_acc.values()) / len(per_transform_acc) if per_transform_acc else 0.0
    
    if utils.is_primary(args):
        _logger.info(f'  Mean Accuracy: {mean_acc:.2f}%')
    
    return OrderedDict([
        ('mean_acc', mean_acc),
        ('per_transform_acc', per_transform_acc),
    ])


if __name__ == '__main__':
    main()

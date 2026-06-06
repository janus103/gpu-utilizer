#!/usr/bin/env python3
""" ImageNet Training Script

This is intended to be a lean and easily modifiable ImageNet training script that reproduces ImageNet
training results with some of the latest networks and training techniques. It favours canonical PyTorch
and standard Python style over trying to be able to 'do it all.' That said, it offers quite a few speed
and training result improvements over the usual PyTorch example scripts. Repurpose as you see fit.

This script was started from an early version of the PyTorch ImageNet example
(https://github.com/pytorch/examples/tree/master/imagenet)

NVIDIA CUDA specific speedups adopted from NVIDIA Apex examples
(https://github.com/NVIDIA/apex/tree/master/examples/imagenet)

Hacked together by / Copyright 2020 Ross Wightman (https://github.com/rwightman)
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
from typing import Tuple

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.utils
import yaml

from timm import utils
from timm.data import create_dataset, create_loader, create_naflex_loader, resolve_data_config, \
    Mixup, FastCollateMixup, AugMixDataset
from timm.data.auto_augment import AugmentOp, _LEVEL_DENOM, _HPARAMS_DEFAULT
from timm.layers import convert_splitbn_model, convert_sync_batchnorm, set_fast_norm
from timm.loss import JsdCrossEntropy, SoftTargetCrossEntropy, BinaryCrossEntropy, LabelSmoothingCrossEntropy
from timm.models import create_model, safe_model_name, resume_checkpoint, load_checkpoint, model_parameters
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler_v2, scheduler_kwargs
from timm.utils import NativeScaler
from timm.task import (
    ClassificationTask,
    LogitDistillationTask,
    FeatureDistillationTask,
    TokenDistillationTask,
)


try:
    import wandb
    has_wandb = True
except ImportError:
    has_wandb = False

try:
    from functorch.compile import memory_efficient_fusion
    has_functorch = True
except ImportError as e:
    has_functorch = False

has_compile = hasattr(torch, 'compile')


_logger = logging.getLogger('train')


class AugmentationManager:
    def __init__(self, sensitivity: float = 1.0, included_augs: list = None):
        self.sensitivity = sensitivity
        # Default augmentation list based on the provided table, using Increasing variants where applicable
        self.default_aug_types = [
            'Rotate', 
            'ShearX', 
            'ShearY', 
            'TranslateXRel', 
            'TranslateYRel',
            'PosterizeIncreasing', 
            'SolarizeIncreasing', 
            'ColorIncreasing',
            'ContrastIncreasing', 
            'BrightnessIncreasing', 
            'SharpnessIncreasing',
            'SolarizeAdd'
        ]
        self.aug_types = included_augs if included_augs is not None else self.default_aug_types
        
    def get_ops(self):
        # Map sensitivity (0.0-1.0) to magnitude (0-_LEVEL_DENOM, usually 10)
        magnitude = self.sensitivity * _LEVEL_DENOM
        ops = []
        for name in self.aug_types:
            # We use prob=1.0 to ensure it's always applied when called
            # We use the default hparams from timm
            ops.append(AugmentOp(name, prob=1.0, magnitude=magnitude, hparams=_HPARAMS_DEFAULT))
        return ops

    def __len__(self):
        return len(self.aug_types)


class PerSampleAugmentDataset(torch.utils.data.Dataset):
    """Wrap a dataset to emit multiple augmented views of the same sample, with labels."""

    def __init__(self, dataset, aug_manager: AugmentationManager, pre_transform=None, post_transform=None):
        self.dataset = dataset
        self.aug_manager = aug_manager
        self.ops = aug_manager.get_ops()
        self.n_aug = len(self.ops)
        self.pre_transform = pre_transform
        self.post_transform = post_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, target = self.dataset[idx]
        
        # Apply Pre-Transform (RandomResizedCrop, Flip, etc.)
        # IMPORTANT: We need deterministic transforms for all N augmentations?
        # The user asked: "Augmented Images should be identical for Pre-Transform?"
        # If pre_transform contains Random ops, calling it once creates one random view.
        # Then we apply N different "style" augmentations on this SAME random view.
        # This seems to be the intended behavior:
        # One geometric view (Crop/Flip) -> N Style variations (Color, etc)
        # + 1 Original (Geometric view only, no style aug)
        
        if self.pre_transform:
            img = self.pre_transform(img)
            
        imgs = []
        aug_labels = []
        
        # Apply each augmentation defined in the manager
        # Since pre_transform is already applied, img is a PIL Image (cropped/flipped)
        # ops apply color/style augs on this PIL Image.
        
        for i, op in enumerate(self.ops):
            # Apply augmentation (Style only)
            aug_img = op(img)
            
            # Apply Post-Transform (ToTensor, Normalize)
            if self.post_transform:
                aug_img = self.post_transform(aug_img)
                
            imgs.append(aug_img)
            
            # Create label vector: [0, ..., sensitivity, ..., 0]
            label = torch.zeros(self.n_aug)
            label[i] = self.aug_manager.sensitivity
            aug_labels.append(label)
        
        # Append "Original" Image (Geometric view but NO style aug)
        # Apply Post-Transform to this base image as well
        # We need a fresh copy of post-transform if it's stateful? No, ToTensor/Norm are not.
        
        base_img_tensor = img
        if self.post_transform:
            base_img_tensor = self.post_transform(img)
            
        imgs.append(base_img_tensor)
        # Label for original image (all zeros)
        aug_labels.append(torch.zeros(self.n_aug))
            
        return imgs, aug_labels, target


def collate_per_sample_aug(batch):
    """Collate multiple augmented views of each sample into a flat batch."""
    images = []
    aug_labels = []
    targets = []
    
    for imgs, labels, target in batch:
        images.extend(imgs)
        aug_labels.extend(labels)
        # Use target for all augmented views? 
        # Usually for training we assume the label is invariant.
        # However, for the 'Main Task' we explicitly use the original image.
        # But `train_one_epoch` usually expands targets.
        # We will expand targets for all N+1 images to keep batch dimension consistent for standard loaders,
        # but inside the wrapper we might only use the target for the main task (or all).
        targets.extend([target] * len(imgs))
        
    images = torch.stack(images, dim=0)
    aug_labels = torch.stack(aug_labels, dim=0)
    targets = torch.tensor(targets)
    
    return images, aug_labels, targets


class StemAutoEncoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Linear(latent_dim, input_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z


class AugAwareWrapper(nn.Module):
    def __init__(self, model, stem_ae: StemAutoEncoder, lalp: nn.Parameter, n_aug: int, use_aug_main_loss: bool = False):
        super().__init__()
        self.model = model
        self.stem_ae = stem_ae
        self.lalp = lalp
        self.n_aug = n_aug
        self.use_aug_main_loss = use_aug_main_loss
        
        # Identify Stem Layer (Conv1)
        if hasattr(model, "conv1"):
            self.stem = model.conv1
        elif hasattr(model, "stem") and hasattr(model.stem, "conv1"):
            self.stem = model.stem.conv1
        else:
            raise AttributeError("Could not find stem conv (conv1) in the model.")
            
        # Get Stem Config
        self.in_channels = self.stem.in_channels
        self.out_channels = self.stem.out_channels
        self.kernel_size = self.stem.kernel_size
        self.stride = self.stem.stride
        self.padding = self.stem.padding
        self.dilation = self.stem.dilation
        self.groups = self.stem.groups
        self.bias = self.stem.bias
        
        # Encode Base Latent (z_base) from initial weights
        # We use a buffer so it moves with device but isn't updated by optimizer directly (unless we want to)
        with torch.no_grad():
            weight = self.stem.weight.detach() # [Out, In, kH, kW]
            means = weight.mean(dim=(2, 3))  # [Out, In]
            vars_ = weight.var(dim=(2, 3), unbiased=False)  # [Out, In]
            flat_stats = torch.cat([means.flatten(), vars_.flatten()], dim=0).unsqueeze(0) # [1, Input_Dim]
            
            # Ensure flat_stats is on the same device as stem_ae
            if next(self.stem_ae.parameters()).is_cuda:
                flat_stats = flat_stats.to(next(self.stem_ae.parameters()).device)
            
            _, z_base = self.stem_ae(flat_stats)
            self.register_buffer('z_base', z_base)
            
            # Store original weight statistics for reconstruction baseline
            self.register_buffer('weight_stats_mean', means)
            self.register_buffer('weight_stats_std', vars_.sqrt() + 1e-6) # approximate std for reconstruction
            
            # Keep a copy of the original weight for reference or fallback
            self.register_buffer('base_weight', weight.clone())

        # SSL Header
        # Input: Pooled OFM (Out_Channels dim) -> Output: N_Aug probabilities
        self.ssl_header = nn.Sequential(
            nn.Linear(self.out_channels, self.out_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.out_channels // 2, n_aug)
        )

    def _generate_weight(self, latent_vector):
        """
        Reconstruct Stem Weight from Latent Vector.
        The AE learns mapping from (Mean, Var) stats to Latent.
        Decoder maps Latent -> (Mean, Var) stats.
        We need to reconstruct the 4D weight tensor from these stats.
        Since AE only predicts stats, we use the original weight's structure (normalized) 
        and rescale it with the predicted stats.
        """
        # latent_vector: [1, Latent_Dim] or [N, Latent_Dim]
        stats = self.stem_ae.decoder(latent_vector) # [N, 2 * Out * In]
        
        # Split stats into Mean and Var
        split_idx = self.out_channels * self.in_channels
        pred_means = stats[:, :split_idx].view(-1, self.out_channels, self.in_channels) # [N, Out, In]
        
        # Fix: Ensure variance is positive to avoid NaN in sqrt
        raw_vars = stats[:, split_idx:].view(-1, self.out_channels, self.in_channels) # [N, Out, In]
        pred_vars = torch.nn.functional.softplus(raw_vars)
        pred_stds = (pred_vars + 1e-6).sqrt()
        
        # Normalize Base Weight: (W - Mean) / Std
        # We use the stored base weights and its stats
        std_broadcast = self.weight_stats_std.unsqueeze(-1).unsqueeze(-1)
        mean_broadcast = self.weight_stats_mean.unsqueeze(-1).unsqueeze(-1)
        
        w_normalized = (self.base_weight - mean_broadcast) / std_broadcast
        
        # Rescale with Predicted Stats
        # pred_means: [N, Out, In], pred_stds: [N, Out, In]
        # w_normalized: [Out, In, kH, kW]
        # Output: [N, Out, In, kH, kW]
        new_weights = w_normalized.unsqueeze(0) * pred_stds.unsqueeze(-1).unsqueeze(-1) + pred_means.unsqueeze(-1).unsqueeze(-1)
        
        return new_weights

    def forward_stem(self, x, weights):
        """
        Apply specific weights to x.
        x: [B, C, H, W]
        weights: [Out, In, kH, kW] or [B, Out, In, kH, kW] (not supported by F.conv2d directly for per-sample)
        
        If we have N distinct weights and N distinct images (or 1 image), 
        we might need grouped conv or loop.
        
        Case Training: N augmented images (B*N total in batch, but we handle per augment type).
        We have N LALPs -> N Weights.
        Image batch is (B, N_Aug+1, C, H, W).
        We want to apply Weight_i to Image_i across the batch.
        """
        return torch.nn.functional.conv2d(
            x, weights, self.bias, self.stride, self.padding, self.dilation, self.groups
        )

    def forward(self, x):
        # x: [Batch * (N_Aug + 1), C, H, W] or [Batch, C, H, W] (if simple inference)
        # But data loader produces flattened batch if n_aug_mode is True and collate flattens it.
        # However, our modified dataset produces N+1 items per sample.
        # collate_per_sample_aug flattens it to (Batch * (N+1)).
        
        B_total = x.shape[0]
        
        if self.training:
            # We assume x is collated batch: [Sample1_Aug1, S1_Aug2, ..., S1_Orig, Sample2_Aug1, ...]
            # Reshape to [Batch, N_Aug + 1, C, H, W]
            num_views = self.n_aug + 1
            B = B_total // num_views
            x = x.view(B, num_views, self.in_channels, x.shape[-2], x.shape[-1])
            
            aug_imgs = x[:, :self.n_aug] # [B, N, C, H, W]
            clean_imgs = x[:, self.n_aug] # [B, C, H, W] - Clean Image
            
            # --- Stem Weight Generation ---
            # 1. Generate N Stem Weights from LALP
            # z_base: [1, L]
            # lalp: [N, L]
            z_combined = self.z_base + self.lalp # [N, L] (Broadcast add)
            stem_weights = self._generate_weight(z_combined) # [N, Out, In, kH, kW]
            
            # --- Path 1: Augmented Image Processing ---
            # 2. Process N Aug Images with N Stem Weights
            # We want: for each i in 0..N-1, Apply stem_weights[i] to aug_imgs[:, i]
            # Output: [B, N, Out, H', W']
            aug_ofm_list = []
            for i in range(self.n_aug):
                feat = self.forward_stem(aug_imgs[:, i], stem_weights[i]) # [B, Out, H', W']
                aug_ofm_list.append(feat)
            aug_ofms = torch.stack(aug_ofm_list, dim=1) # [B, N, Out, H', W']
            
            # 3. SSL Header Prediction (For Aug Images)
            # Pool: [B, N, Out, H', W'] -> [B, N, Out]
            aug_pooled_ofms = aug_ofms.mean(dim=(3, 4)) 
            
            # Predict: [B, N, Out] -> [B, N, N_Aug]
            # Flatten batch and N for Linear layer
            aug_ssl_logits = self.ssl_header(aug_pooled_ofms.view(-1, self.out_channels))
            aug_ssl_logits = aug_ssl_logits.view(B, self.n_aug, self.n_aug)
            
            # 4. Optional: Main Backbone for Aug Images
            aug_main_logits = None
            if self.use_aug_main_loss:
                # Need to pass aug_ofms through the rest of the model
                # aug_ofms: [B, N, Out, H', W'] -> Flatten to [B*N, Out, H', W']
                aug_feat_flat = aug_ofms.view(B * self.n_aug, self.out_channels, aug_ofms.shape[-2], aug_ofms.shape[-1])
                
                # Monkey Patching for Main Model
                original_stem_forward = self.stem.forward
                def custom_forward(x_in):
                    return aug_feat_flat
                self.stem.forward = custom_forward
                
                try:
                    # Input to model doesn't matter as stem ignores it, but shape must be compatible for check
                    # We pass a dummy input or just repeat clean_imgs? 
                    # model(x) checks input shape usually.
                    # We need input of size [B*N, C, H, W] to match batch size
                    dummy_input = clean_imgs.repeat_interleave(self.n_aug, dim=0) # [B*N, C, H, W]
                    aug_main_logits = self.model(dummy_input) # [B*N, Num_Classes]
                finally:
                    self.stem.forward = original_stem_forward
            
            # --- Path 2: Clean Image Processing ---
            # 5. Process Clean Image with ALL N Stem Weights (For Consistency / Uniformity)
            # We want to see if Clean Image produces Uniform distribution from SSL Header
            clean_ofm_list = []
            for i in range(self.n_aug):
                feat = self.forward_stem(clean_imgs, stem_weights[i]) # [B, Out, H', W']
                clean_ofm_list.append(feat)
            clean_ofms = torch.stack(clean_ofm_list, dim=1) # [B, N, Out, H', W']
            
            # 6. SSL Header Prediction (For Clean Image)
            clean_pooled_ofms = clean_ofms.mean(dim=(3, 4))
            clean_ssl_logits = self.ssl_header(clean_pooled_ofms.view(-1, self.out_channels))
            clean_ssl_logits = clean_ssl_logits.view(B, self.n_aug, self.n_aug)
            
            # 7. Weight Calculation & Ensemble (Dynamic Stem for Main Task)
            # "Clean Image should produce Uniform Weights" is enforced by Loss
            # But here we just use what SSL Header predicts to form the ensemble.
            
            # Softmax over prediction classes (last dim)
            # But which output do we use? The diagonal?
            # Clean image has no "ground truth" augmentation.
            # However, the logic is: "How much does this image (processed by Stem i) look like Augmentation i?"
            # So we still look at the diagonal.
            clean_ssl_probs = torch.softmax(clean_ssl_logits, dim=-1) # [B, N, N_Aug]
            weights = torch.diagonal(clean_ssl_probs, dim1=1, dim2=2) # [B, N]
            
            # Normalize weights
            weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-6) # [B, N]
            
            # Weighted Average of LALP (Reparameterization)
            z_agg = torch.matmul(weights, self.lalp) # [B, L]
            
            # Reconstruct Stem for Main Task
            z_final = self.z_base + z_agg # [B, L]
            main_stem_weights = self._generate_weight(z_final) # [B, Out, In, kH, kW]
            
            # 8. Main Pass (Final Prediction with Clean Image + Ensemble Stem)
            # Reshape input to [1, B*C, H, W], Weights to [B*Out, In, kH, kW], Groups=B
            x_main = clean_imgs.reshape(1, B * self.in_channels, clean_imgs.shape[-2], clean_imgs.shape[-1])
            w_main = main_stem_weights.reshape(B * self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1])
            
            main_feat = torch.nn.functional.conv2d(
                x_main, w_main, 
                bias=self.bias.repeat(B) if self.bias is not None else None, 
                stride=self.stride, padding=self.padding, dilation=self.dilation, groups=B * self.groups
            )
            # Output: [1, B*Out, H', W'] -> [B, Out, H', W']
            main_feat = main_feat.view(B, self.out_channels, main_feat.shape[-2], main_feat.shape[-1])
            
            # Monkey patch for Main Pass
            original_stem_forward = self.stem.forward
            def custom_forward(x_in):
                return main_feat
            self.stem.forward = custom_forward
            
            try:
                main_logits = self.model(clean_imgs)
            finally:
                self.stem.forward = original_stem_forward
                
            return main_logits, aug_ssl_logits, clean_ssl_logits, aug_main_logits
            
        else:
            # Inference Mode
            # x: [B, C, H, W] (Original Images)
            B = x.shape[0]
            
            # 1. Generate N Stem Weights (Standard LALPs)
            z_combined = self.z_base + self.lalp # [N, L]
            stem_weights = self._generate_weight(z_combined) # [N, Out, In, kH, kW]
            
            # 2. Process Original Image with ALL N Stems
            # We need to run x through N stems.
            # Grouped conv trick: Input repeated N times?
            # Or loop. Loop N=12 is fine.
            ofm_list = []
            for i in range(self.n_aug):
                # Use Weight i for entire batch
                feat = self.forward_stem(x, stem_weights[i])
                ofm_list.append(feat)
            ofms = torch.stack(ofm_list, dim=1) # [B, N, Out, H', W']
            
            # 3. SSL Header & Weights
            pooled_ofms = ofms.mean(dim=(3, 4))
            ssl_logits = self.ssl_header(pooled_ofms.view(-1, self.out_channels))
            ssl_logits = ssl_logits.view(B, self.n_aug, self.n_aug)
            ssl_probs = torch.softmax(ssl_logits, dim=-1)
            
            # "SSL Header를 통해 어떤 LALP를 사용할지 확인 후... 중요도 값 처럼 대입"
            # Here we use the probability that the image (processed by Stem i) looks like Aug i.
            # i.e. Diagonal again.
            weights = torch.diagonal(ssl_probs, dim1=1, dim2=2) # [B, N]
            weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-6)
            
            # 4. Aggregate LALP and Reconstruct Stem
            z_agg = torch.matmul(weights, self.lalp) # [B, L]
            z_final = self.z_base + z_agg
            main_stem_weights = self._generate_weight(z_final) # [B, Out, In, kH, kW]
            
            # 5. Main Pass
            # Per-sample weights again. Grouped conv trick.
            x_main = x.reshape(1, B * self.in_channels, x.shape[-2], x.shape[-1])
            w_main = main_stem_weights.reshape(B * self.out_channels, self.in_channels, self.kernel_size[0], self.kernel_size[1])
            main_feat = torch.nn.functional.conv2d(
                x_main, w_main, 
                bias=self.bias.repeat(B) if self.bias is not None else None, 
                stride=self.stride, padding=self.padding, dilation=self.dilation, groups=B * self.groups
            )
            main_feat = main_feat.view(B, self.out_channels, main_feat.shape[-2], main_feat.shape[-1])
            
            # Monkey patch
            original_stem_forward = self.stem.forward
            def custom_forward(x_in):
                return main_feat
            self.stem.forward = custom_forward
            
            try:
                main_logits = self.model(x)
            finally:
                self.stem.forward = original_stem_forward
                
            # For validation, we might just return main_logits
            # But the train loop might expect tuple if we returned tuple in training
            # We should probably return same structure or handle in validate
            return main_logits


# The first arg parser parses out only the --config argument, this argument is used to
# load a yaml file containing key-values that override the defaults for the main parser below
config_parser = parser = argparse.ArgumentParser(description='Training Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE',
                    help='YAML config file specifying default arguments')


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

# Dataset parameters
group = parser.add_argument_group('Dataset parameters')
# Keep this argument outside the dataset group because it is positional.
parser.add_argument('data', nargs='?', metavar='DIR', const=None,
                    help='path to dataset (positional is *deprecated*, use --data-dir)')
group.add_argument('--data-dir', metavar='DIR',
                    help='path to dataset (root dir)')
group.add_argument('--dataset', metavar='NAME', default='',
                    help='dataset type + name ("<type>/<name>") (default: ImageFolder or ImageTar if empty)')
group.add_argument('--train-split', metavar='NAME', default='train',
                   help='dataset train split (default: train)')
group.add_argument('--val-split', metavar='NAME', default='validation',
                   help='dataset validation split (default: validation)')
group.add_argument('--train-num-samples', default=None, type=int,
                    metavar='N', help='Manually specify num samples in train split, for IterableDatasets.')
group.add_argument('--val-num-samples', default=None, type=int,
                    metavar='N', help='Manually specify num samples in validation split, for IterableDatasets.')
group.add_argument('--dataset-download', action='store_true', default=False,
                   help='Allow download of dataset for torch/ and tfds/ datasets that support it.')
group.add_argument('--class-map', default='', type=str, metavar='FILENAME',
                   help='path to class to idx mapping file (default: "")')
group.add_argument('--input-img-mode', default=None, type=str,
                   help='Dataset image conversion mode for input images.')
group.add_argument('--input-key', default=None, type=str,
                   help='Dataset key for input images.')
group.add_argument('--target-key', default=None, type=str,
                   help='Dataset key for target labels.')
group.add_argument('--dataset-trust-remote-code', action='store_true', default=False,
                   help='Allow huggingface dataset import to execute code downloaded from the dataset\'s repo.')

# Model parameters
group = parser.add_argument_group('Model parameters')
group.add_argument('--model', default='resnet50', type=str, metavar='MODEL',
                   help='Name of model to train (default: "resnet50")')
group.add_argument('--pretrained', action='store_true', default=False,
                   help='Start with pretrained version of specified network (if avail)')
group.add_argument('--pretrained-path', default=None, type=str,
                   help='Load this checkpoint as if they were the pretrained weights (with adaptation).')
group.add_argument('--latent-size', type=int, default=None,
                   help='Latent vector size for stem AE (forward-time).')
group.add_argument('--ae-checkpoint', '--ae-checkpoints', dest='ae_checkpoint', default='',
                   help='Path to stem autoencoder checkpoint (.pth).')
group.add_argument('--initial-checkpoint', default='', type=str, metavar='PATH',
                   help='Load this checkpoint into model after initialization (default: none)')
group.add_argument('--resume', default='', type=str, metavar='PATH',
                   help='Resume full model and optimizer state from checkpoint (default: none)')
group.add_argument('--no-resume-opt', action='store_true', default=False,
                   help='prevent resume of optimizer state when resuming model')
group.add_argument('--num-classes', type=int, default=None, metavar='N',
                   help='number of label classes (Model default if None)')
group.add_argument('--gp', default=None, type=str, metavar='POOL',
                   help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
group.add_argument('--img-size', type=int, default=None, metavar='N',
                   help='Image size (default: None => model default)')
group.add_argument('--in-chans', type=int, default=None, metavar='N',
                   help='Image input channels (default: None => 3)')
group.add_argument('--input-size', default=None, nargs=3, type=int, metavar='N',
                   help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty')
group.add_argument('--crop-pct', default=None, type=float,
                   metavar='N', help='Input image center crop percent (for validation only)')
group.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                   help='Override mean pixel value of dataset')
group.add_argument('--std', type=float, nargs='+', default=None, metavar='STD',
                   help='Override std deviation of dataset')
group.add_argument('--interpolation', default='', type=str, metavar='NAME',
                   help='Image resize interpolation type (overrides model)')
group.add_argument('-b', '--batch-size', type=int, default=128, metavar='N',
                   help='Input batch size for training (default: 128)')
group.add_argument('-vb', '--validation-batch-size', type=int, default=None, metavar='N',
                   help='Validation batch size override (default: None)')
group.add_argument('--channels-last', action='store_true', default=False,
                   help='Use channels_last memory layout')
group.add_argument('--fuser', default='', type=str,
                   help="Select jit fuser. One of ('', 'te', 'old', 'nvfuser')")
group.add_argument('--grad-accum-steps', type=int, default=1, metavar='N',
                   help='The number of steps to accumulate gradients (default: 1)')
group.add_argument('--grad-checkpointing', action='store_true', default=False,
                   help='Enable gradient checkpointing through model blocks/stages')
group.add_argument('--fast-norm', default=False, action='store_true',
                   help='enable experimental fast-norm')
group.add_argument('--model-kwargs', nargs='*', default={}, action=utils.ParseKwargs)
group.add_argument('--head-init-scale', default=None, type=float,
                   help='Head initialization scale')
group.add_argument('--head-init-bias', default=None, type=float,
                   help='Head initialization bias value')
group.add_argument('--torchcompile-mode', type=str, default=None,
                    help="torch.compile mode (default: None).")

# scripting / codegen
scripting_group = group.add_mutually_exclusive_group()
scripting_group.add_argument('--torchscript', dest='torchscript', action='store_true',
                             help='torch.jit.script the full model')
scripting_group.add_argument('--torchcompile', nargs='?', type=str, default=None, const='inductor',
                             help="Enable compilation w/ specified backend (default: inductor).")

# Device & distributed
group = parser.add_argument_group('Device parameters')
group.add_argument('--device', default='cuda', type=str,
                    help="Device (accelerator) to use.")
group.add_argument('--amp', action='store_true', default=False,
                   help='use AMP for mixed precision training')
group.add_argument('--amp-dtype', default='float16', type=str,
                   help='lower precision AMP dtype (default: float16)')
group.add_argument('--model-dtype', default=None, type=str,
                   help='Model dtype override (non-AMP) (default: float32)')
group.add_argument('--no-ddp-bb', action='store_true', default=False,
                   help='Force broadcast buffers for native DDP to off.')
group.add_argument('--synchronize-step', action='store_true', default=False,
                   help='torch.cuda.synchronize() end of each step')
group.add_argument("--local_rank", default=0, type=int)
group.add_argument('--device-modules', default=None, type=str, nargs='+',
                    help="Python imports for device backend modules.")

# Optimizer parameters
group = parser.add_argument_group('Optimizer parameters')
group.add_argument('--opt', default='sgd', type=str, metavar='OPTIMIZER',
                   help='Optimizer (default: "sgd")')
group.add_argument('--opt-eps', default=None, type=float, metavar='EPSILON',
                   help='Optimizer Epsilon (default: None, use opt default)')
group.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                   help='Optimizer Betas (default: None, use opt default)')
group.add_argument('--momentum', type=float, default=0.9, metavar='M',
                   help='Optimizer momentum (default: 0.9)')
group.add_argument('--weight-decay', type=float, default=2e-5,
                   help='weight decay (default: 2e-5)')
group.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                   help='Clip gradient norm (default: None, no clipping)')
group.add_argument('--clip-mode', type=str, default='norm',
                   help='Gradient clipping mode. One of ("norm", "value", "agc")')
group.add_argument('--layer-decay', type=float, default=None,
                   help='layer-wise learning rate decay (default: None)')
group.add_argument('--layer-decay-min-scale', type=float, default=0,
                   help='layer-wise lr decay minimum scale clamp (default: 0)')
group.add_argument('--layer-decay-no-opt-scale', type=float, default=None,
                   help='layer-wise lr decay no optimization scale (default: None)')
group.add_argument('--opt-kwargs', nargs='*', default={}, action=utils.ParseKwargs)

# Learning rate schedule parameters
group = parser.add_argument_group('Learning rate schedule parameters')
group.add_argument('--sched', type=str, default='cosine', metavar='SCHEDULER',
                   help='LR scheduler (default: "cosine"')
group.add_argument('--sched-on-updates', action='store_true', default=False,
                   help='Apply LR scheduler step on update instead of epoch end.')
group.add_argument('--lr', type=float, default=None, metavar='LR',
                   help='learning rate, overrides lr-base if set (default: None)')
group.add_argument('--lr-base', type=float, default=0.1, metavar='LR',
                   help='base learning rate: lr = lr_base * global_batch_size / base_size')
group.add_argument('--lr-base-size', type=int, default=256, metavar='DIV',
                   help='base learning rate batch size (divisor, default: 256).')
group.add_argument('--lr-base-scale', type=str, default='', metavar='SCALE',
                   help='base learning rate vs batch_size scaling ("linear", "sqrt", based on opt if empty)')
group.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                   help='learning rate noise on/off epoch percentages')
group.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                   help='learning rate noise limit percent (default: 0.67)')
group.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                   help='learning rate noise std-dev (default: 1.0)')
group.add_argument('--lr-cycle-mul', type=float, default=1.0, metavar='MULT',
                   help='learning rate cycle len multiplier (default: 1.0)')
group.add_argument('--lr-cycle-decay', type=float, default=0.5, metavar='MULT',
                   help='amount to decay each learning rate cycle (default: 0.5)')
group.add_argument('--lr-cycle-limit', type=int, default=1, metavar='N',
                   help='learning rate cycle limit, cycles enabled if > 1')
group.add_argument('--lr-k-decay', type=float, default=1.0,
                   help='learning rate k-decay for cosine/poly (default: 1.0)')
group.add_argument('--warmup-lr', type=float, default=1e-5, metavar='LR',
                   help='warmup learning rate (default: 1e-5)')
group.add_argument('--min-lr', type=float, default=0, metavar='LR',
                   help='lower lr bound for cyclic schedulers that hit 0 (default: 0)')
group.add_argument('--epochs', type=int, default=300, metavar='N',
                   help='number of epochs to train (default: 300)')
group.add_argument('--epoch-repeats', type=float, default=0., metavar='N',
                   help='epoch repeat multiplier (number of times to repeat dataset epoch per train epoch).')
group.add_argument('--start-epoch', default=None, type=int, metavar='N',
                   help='manual epoch number (useful on restarts)')
group.add_argument('--decay-milestones', default=[90, 180, 270], type=int, nargs='+', metavar="MILESTONES",
                   help='list of decay epoch indices for multistep lr. must be increasing')
group.add_argument('--decay-epochs', type=float, default=90, metavar='N',
                   help='epoch interval to decay LR')
group.add_argument('--warmup-epochs', type=int, default=1, metavar='N',
                   help='epochs to warmup LR, if scheduler supports')
group.add_argument('--warmup-prefix', action='store_true', default=False,
                   help='Exclude warmup period from decay schedule.'),
group.add_argument('--cooldown-epochs', type=int, default=0, metavar='N',
                   help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
group.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                   help='patience epochs for Plateau LR scheduler (default: 10)')
group.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                   help='LR decay rate (default: 0.1)')

# Augmentation & regularization parameters
group = parser.add_argument_group('Augmentation and regularization parameters')
group.add_argument('--no-aug', action='store_true', default=False,
                   help='Disable all training augmentation, override other train aug args')
group.add_argument('--train-crop-mode', type=str, default=None,
                   help='Crop-mode in train'),
group.add_argument('--scale', type=float, nargs='+', default=[0.08, 1.0], metavar='PCT',
                   help='Random resize scale (default: 0.08 1.0)')
group.add_argument('--ratio', type=float, nargs='+', default=[3. / 4., 4. / 3.], metavar='RATIO',
                   help='Random resize aspect ratio (default: 0.75 1.33)')
group.add_argument('--hflip', type=float, default=0.5,
                   help='Horizontal flip training aug probability')
group.add_argument('--vflip', type=float, default=0.,
                   help='Vertical flip training aug probability')
group.add_argument('--color-jitter', type=float, default=0.4, metavar='PCT',
                   help='Color jitter factor (default: 0.4)')
group.add_argument('--color-jitter-prob', type=float, default=None, metavar='PCT',
                   help='Probability of applying any color jitter.')
group.add_argument('--grayscale-prob', type=float, default=None, metavar='PCT',
                   help='Probability of applying random grayscale conversion.')
group.add_argument('--gaussian-blur-prob', type=float, default=None, metavar='PCT',
                   help='Probability of applying gaussian blur.')
group.add_argument('--aa', type=str, default=None, metavar='NAME',
                   help='Use AutoAugment policy. "v0" or "original". (default: None)'),
group.add_argument('--aug-repeats', type=float, default=0,
                   help='Number of augmentation repetitions (distributed training only) (default: 0)')
group.add_argument('--aug-splits', type=int, default=0,
                   help='Number of augmentation splits (default: 0, valid: 0 or >=2)')
group.add_argument('--jsd-loss', action='store_true', default=False,
                   help='Enable Jensen-Shannon Divergence + CE loss. Use with `--aug-splits`.')
group.add_argument('--bce-loss', action='store_true', default=False,
                   help='Enable BCE loss w/ Mixup/CutMix use.')
group.add_argument('--ssl-lambda', type=float, default=12.0,
                   help='Weight lambda for SSL loss in Augmentation Aware training (default: 12.0)')
group.add_argument('--use-aug-main-loss', action='store_true', default=False,
                   help='Use Main Classification Loss for Augmented Images (default: False)')
group.add_argument('--aug-main-loss-weight', type=float, default=1.0,
                   help='Weight for Augmented Main Loss (default: 1.0)')
group.add_argument('--kl-loss-weight', type=float, default=1.0,
                   help='Weight for Clean Image KL Divergence Loss (default: 1.0)')
group.add_argument('--aug-sensitivity', type=float, default=1.0,
                   help='Sensitivity (intensity) of augmentations (0.0 to 1.0). Default: 1.0 (Max)')
group.add_argument('--freeze-non-selfsup', action='store_true', default=False,
                   help='Freeze all parameters except SSL FC head (ssl_header) and LALP so only self-supervised components learn.')
group.add_argument('--bce-sum', action='store_true', default=False,
                   help='Sum over classes when using BCE loss.')
group.add_argument('--bce-target-thresh', type=float, default=None,
                   help='Threshold for binarizing softened BCE targets (default: None, disabled).')
group.add_argument('--bce-pos-weight', type=float, default=None,
                   help='Positive weighting for BCE loss.')
group.add_argument('--reprob', type=float, default=0., metavar='PCT',
                   help='Random erase prob (default: 0.)')
group.add_argument('--remode', type=str, default='pixel',
                   help='Random erase mode (default: "pixel")')
group.add_argument('--recount', type=int, default=1,
                   help='Random erase count (default: 1)')
group.add_argument('--resplit', action='store_true', default=False,
                   help='Do not random erase first (clean) augmentation split')
group.add_argument('--mixup', type=float, default=0.0,
                   help='mixup alpha, mixup enabled if > 0. (default: 0.)')
group.add_argument('--cutmix', type=float, default=0.0,
                   help='cutmix alpha, cutmix enabled if > 0. (default: 0.)')
group.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                   help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
group.add_argument('--mixup-prob', type=float, default=1.0,
                   help='Probability of performing mixup or cutmix when either/both is enabled')
group.add_argument('--mixup-switch-prob', type=float, default=0.5,
                   help='Probability of switching to cutmix when both mixup and cutmix enabled')
group.add_argument('--mixup-mode', type=str, default='batch',
                   help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')
group.add_argument('--mixup-off-epoch', default=0, type=int, metavar='N',
                   help='Turn off mixup after this epoch, disabled if 0 (default: 0)')
group.add_argument('--smoothing', type=float, default=0.1,
                   help='Label smoothing (default: 0.1)')
group.add_argument('--train-interpolation', type=str, default='random',
                   help='Training interpolation (random, bilinear, bicubic default: "random")')
group.add_argument('--drop', type=float, default=0.0, metavar='PCT',
                   help='Dropout rate (default: 0.)')
group.add_argument('--drop-connect', type=float, default=None, metavar='PCT',
                   help='Drop connect rate, DEPRECATED, use drop-path (default: None)')
group.add_argument('--drop-path', type=float, default=None, metavar='PCT',
                   help='Drop path rate (default: None)')
group.add_argument('--drop-block', type=float, default=None, metavar='PCT',
                   help='Drop block rate (default: None)')

# Batch norm parameters (only works with gen_efficientnet based models currently)
group = parser.add_argument_group('Batch norm parameters', 'Only works with gen_efficientnet based models currently.')
group.add_argument('--bn-momentum', type=float, default=None,
                   help='BatchNorm momentum override (if not None)')
group.add_argument('--bn-eps', type=float, default=None,
                   help='BatchNorm epsilon override (if not None)')
group.add_argument('--sync-bn', action='store_true',
                   help='Enable synchronized BatchNorm.')
group.add_argument('--dist-bn', type=str, default='reduce',
                   help='Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")')
group.add_argument('--split-bn', action='store_true',
                   help='Enable separate BN layers per augmentation split.')

# Model Exponential Moving Average
group = parser.add_argument_group('Model exponential moving average parameters')
group.add_argument('--model-ema', action='store_true', default=False,
                   help='Enable tracking moving average of model weights.')
group.add_argument('--model-ema-force-cpu', action='store_true', default=False,
                   help='Force ema to be tracked on CPU, rank=0 node only. Disables EMA validation.')
group.add_argument('--model-ema-decay', type=float, default=0.9998,
                   help='Decay factor for model weights moving average (default: 0.9998)')
group.add_argument('--model-ema-warmup', action='store_true',
                   help='Enable warmup for model EMA decay.')

# Misc
group = parser.add_argument_group('Miscellaneous parameters')
group.add_argument('--seed', type=int, default=42, metavar='S',
                   help='random seed (default: 42)')
group.add_argument('--worker-seeding', type=str, default='all',
                   help='worker seed mode (default: all)')
group.add_argument('--log-interval', type=int, default=50, metavar='N',
                   help='how many batches to wait before logging training status')
group.add_argument('--val-interval', type=int, default=1, metavar='N',
                   help='how many epochs between validation and checkpointing')
group.add_argument('--recovery-interval', type=int, default=0, metavar='N',
                   help='how many batches to wait before writing recovery checkpoint')
group.add_argument('--checkpoint-hist', type=int, default=10, metavar='N',
                   help='number of checkpoints to keep (default: 10)')
group.add_argument('-j', '--workers', type=int, default=4, metavar='N',
                   help='how many training processes to use (default: 4)')
group.add_argument('--save-images', action='store_true', default=False,
                   help='save images of input batches every log interval for debugging')
group.add_argument('--pin-mem', action='store_true', default=False,
                   help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
group.add_argument('--no-prefetcher', action='store_true', default=False,
                   help='disable fast prefetcher')
group.add_argument('--output', default='', type=str, metavar='PATH',
                   help='path to output folder (default: none, current dir)')
group.add_argument('--experiment', default='', type=str, metavar='NAME',
                   help='name of train experiment, name of sub-folder for output')
group.add_argument('--eval-metric', default='top1', type=str, metavar='EVAL_METRIC',
                   help='Best metric (default: "top1"')
group.add_argument('--tta', type=int, default=0, metavar='N',
                   help='Test/inference time augmentation (oversampling) factor. 0=None (default: 0)')
group.add_argument('--use-multi-epochs-loader', action='store_true', default=False,
                   help='use the multi-epochs-loader to save time at the beginning of every epoch')
group.add_argument('--log-wandb', action='store_true', default=False,
                   help='log training and validation metrics to wandb')
group.add_argument('--wandb-project', default=None, type=str,
                   help='wandb project name')
group.add_argument('--wandb-tags', default=[], type=str, nargs='+',
                   help='wandb tags')
group.add_argument('--wandb-resume-id', default='', type=str, metavar='ID',
                   help='If resuming a run, the id of the run in wandb')

# NaFlex scheduled loader arguments
group.add_argument('--naflex-loader', action='store_true', default=False,
                   help='Use NaFlex loader (Requires NaFlex compatible model)')
group.add_argument('--naflex-train-seq-lens', type=int, nargs='+', default=[128, 256, 576, 784, 1024],
                   help='Sequence lengths to use for NaFlex loader')
group.add_argument('--naflex-max-seq-len', type=int, default=576,
                   help='Fixed maximum sequence length for NaFlex loader (validation)')
group.add_argument('--naflex-patch-sizes', type=int, nargs='+', default=None,
                   help='List of patch sizes for variable patch size training (e.g., 8 12 16 24 32)')
group.add_argument('--naflex-patch-size-probs', type=float, nargs='+', default=None,
                   help='Probabilities for each patch size (must sum to 1.0, uniform if not specified)')
group.add_argument('--naflex-loss-scale', default='linear', type=str,
                   help='Scale loss (gradient) by batch_size ("none", "sqrt", or "linear")')

# Knowledge Distillation parameters
parser.add_argument('--kd-model-name', default=None, type=str,
                    help='Name of teacher model for knowledge distillation')
parser.add_argument('--kd-distill-type', default='logit', type=str, choices=['logit', 'feature', 'token'],
                    help='Type of distillation: "logit" for output distillation, "feature" for intermediate features, "token" for models with distillation heads (default: logit)')
parser.add_argument('--kd-loss-type', default='kl', type=str,
                    help='Loss function for logit distillation (default: kl). Currently only "kl" supported, reserved for future extensions.')
parser.add_argument('--distill-loss-weight', default=None, type=float,
                    help='Weight for distillation loss. If both weights specified: loss = task_weight * task + distill_weight * distill. '
                         'If only task_weight: loss = task_weight * task + (1-task_weight) * distill. Default: 1.0 if only this specified.')
parser.add_argument('--task-loss-weight', default=None, type=float,
                    help='Weight for task (classification) loss. See --distill-loss-weight for weighting modes. Default: 1.0 if unspecified.')
parser.add_argument('--kd-temperature', default=4.0, type=float,
                    help='Temperature for softmax in distillation (default: 4.0, typical range: 1-4)')
parser.add_argument('--kd-student-feature-dim', default=None, type=int,
                    help='Student model feature dimension (auto-detected from model.head_hidden_size or model.num_features if not specified)')
parser.add_argument('--kd-teacher-feature-dim', default=None, type=int,
                    help='Teacher model feature dimension (auto-detected from model.head_hidden_size or model.num_features if not specified)')
parser.add_argument('--kd-token-distill-type', default='soft', type=str, choices=['soft', 'hard'],
                    help='Token distillation type: "soft" for KL-div with temperature, "hard" for CE with teacher argmax (default: soft)')


def _parse_args():
    # Do we have a config file to parse?
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)

    # The main arg parser parses the rest of the args, the usual
    # defaults will have been overridden if config file specified.
    args = parser.parse_args(remaining)

    # Cache the args as a text string to save them in the output dir later
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
        _logger.info(
            'Training in distributed mode with multiple processes, 1 device per process.'
            f'Process {args.rank}, total {args.world_size}, device {args.device}.')
    else:
        _logger.info(f'Training with a single process on 1 device ({args.device}).')
    assert args.rank >= 0

    args.prefetcher = not args.no_prefetcher
    args.grad_accum_steps = max(1, args.grad_accum_steps)
    
    # Initialize AugmentationManager
    aug_manager = AugmentationManager(sensitivity=args.aug_sensitivity)
    if not args.no_aug:
        # Override n_aug with the number of augmentations in the manager
        args.n_aug = len(aug_manager)
        
    n_aug_mode = args.n_aug and args.n_aug > 1
    if n_aug_mode and args.naflex_loader:
        parser.error('--n-aug is not supported with --naflex-loader')
    if n_aug_mode and args.aug_splits > 1:
        parser.error('--n-aug cannot be combined with --aug-splits/JS D mode')
    if n_aug_mode and args.prefetcher:
        if utils.is_primary(args):
            _logger.warning('Disabling prefetcher for --n-aug mode (custom collate required).')
        args.prefetcher = False
    base_batch_size = args.batch_size
    
    # In AugAware Mode, we process N augmented images only through Stem+SSL Header (lightweight)
    # and 1 main image through Full Backbone (heavy).
    # Previous logic forced batch_size=1, which is too conservative and slow.
    # We allow standard batch size, but user should be aware that Stem memory usage increases by (N+1)x.
    # If OOM occurs, user should reduce --batch-size manually.
    
    train_loader_batch_size = args.batch_size
    
    # train_loader_batch_size = 1 if n_aug_mode else args.batch_size
    # if n_aug_mode:
    #     args.grad_accum_steps = max(args.grad_accum_steps, base_batch_size)
    #     if utils.is_primary(args):
    #         eff_global_batch = base_batch_size * args.n_aug * args.world_size
    #         _logger.info(
    #             f'Enabling per-sample multi-augmentation: n_aug={args.n_aug}, '
    #             f'loader_batch=1, accum_steps={args.grad_accum_steps}, '
    #             f'effective_global_batch={eff_global_batch}'
    #         )

    model_dtype = None
    if args.model_dtype:
        assert args.model_dtype in ('float32', 'float16', 'bfloat16')
        model_dtype = getattr(torch, args.model_dtype)
        if model_dtype == torch.float16:
            _logger.warning('float16 is not recommended for training, for half precision bfloat16 is recommended.')

    # resolve AMP arguments based on PyTorch availability
    amp_dtype = torch.float16
    if args.amp:
        assert model_dtype is None or model_dtype == torch.float32, 'float32 model dtype must be used with AMP'
        assert args.amp_dtype in ('float16', 'bfloat16')
        if args.amp_dtype == 'bfloat16':
            amp_dtype = torch.bfloat16

    utils.random_seed(args.seed, args.rank)

    if args.fuser:
        utils.set_jit_fuser(args.fuser)
    if args.fast_norm:
        set_fast_norm()

    in_chans = 3
    if args.in_chans is not None:
        in_chans = args.in_chans
    elif args.input_size is not None:
        in_chans = args.input_size[0]

    factory_kwargs = {}
    if args.pretrained_path:
        # merge with pretrained_cfg of model, 'file' has priority over 'url' and 'hf_hub'.
        factory_kwargs['pretrained_cfg_overlay'] = dict(
            file=args.pretrained_path,
            num_classes=-1,  # force head adaptation
        )

    # propagate stem AE options to model kwargs and relax strict loading when needed
    if args.ae_checkpoint:
        args.model_kwargs = dict(args.model_kwargs)
        args.model_kwargs.setdefault('stem_ae_checkpoint', args.ae_checkpoint)
    if args.latent_size is not None:
        args.model_kwargs = dict(args.model_kwargs)
        args.model_kwargs.setdefault('stem_ae_latent_size', args.latent_size)
    if args.ae_checkpoint or args.latent_size is not None or args.model.endswith('_ae'):
        args.model_kwargs = dict(args.model_kwargs)
        args.model_kwargs.setdefault('pretrained_strict', False)

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
        **factory_kwargs,
        **args.model_kwargs,
    )
    # Always freeze stem AE parameters if present (regardless of checkpoint).
    stem_ae_in_model = getattr(model, 'stem_ae', None)
    if stem_ae_in_model is not None:
        for p in stem_ae_in_model.parameters():
            p.requires_grad = False
        stem_ae_in_model.eval()
                
    # Register LALP parameter if we are in n_aug_mode (Augmentation Aware mode)
    if n_aug_mode:
        if args.latent_size is None:
             # Default to 128 if not specified, though it should ideally match the AE latent size
             if utils.is_primary(args):
                 _logger.warning("Latent size not specified for LALP! Using default 128.")
             args.latent_size = 128
        
        # Define LALP: (N_Aug_Types, Latent_Dim)
        # We initialize it with small random values to start close to the base stem
        # Initialize LALP to a tiny epsilon (close to zero but not denormal).
        lalp_init = torch.full(
            (len(aug_manager), args.latent_size),
            torch.finfo(torch.float32).eps,
        )
        model.lalp = nn.Parameter(lalp_init)
        model.register_parameter('lalp', model.lalp)
        if utils.is_primary(args):
            _logger.info(f'Registered LALP parameter with shape: {model.lalp.shape}')
            
    if args.head_init_scale is not None:
        with torch.no_grad():
            model.get_classifier().weight.mul_(args.head_init_scale)
            model.get_classifier().bias.mul_(args.head_init_scale)
    if args.head_init_bias is not None:
        nn.init.constant_(model.get_classifier().bias, args.head_init_bias)

    if args.num_classes is None:
        assert hasattr(model, 'num_classes'), 'Model must have `num_classes` attr if not set on cmd line/config.'
        args.num_classes = model.num_classes  # FIXME handle model default vs config num_classes more elegantly

    if args.grad_checkpointing:
        model.set_grad_checkpointing(enable=True)

    # Create training task (classification or distillation)
    task = None
    
    # Initialize Stem AutoEncoder if needed
    if n_aug_mode:
        # Reuse a single stem AE (shared, frozen). Prefer the one already in the model.
        stem_ae = getattr(model, 'stem_ae', None)
        if stem_ae is None:
            # Fallback: create from checkpoint/latent args if model didn't build one.
            if args.latent_size is None and not args.ae_checkpoint:
                raise ValueError("Stem AE is required in n_aug_mode but not provided.")
            stem_ae = StemAutoEncoder(
                input_dim=2 * model.conv1.out_channels * model.conv1.in_channels,  # Assuming ResNet stem
                latent_dim=args.latent_size,
            )
            if args.ae_checkpoint:
                checkpoint = torch.load(args.ae_checkpoint, map_location='cpu')
                stem_ae.load_state_dict(checkpoint['state_dict'])
                if utils.is_primary(args):
                    _logger.info(f'Loaded Stem AE from {args.ae_checkpoint} (fallback init)')
            # Attach to model so it moves with model.to(...)
            model.stem_ae = stem_ae

        # Freeze AE: only LALP should learn.
        for p in stem_ae.parameters():
            p.requires_grad = False
        stem_ae.eval()

        # Wrap model with AugAwareWrapper
        model = AugAwareWrapper(
            model, 
            stem_ae, 
            model.lalp, 
            n_aug=len(aug_manager),
            use_aug_main_loss=args.use_aug_main_loss
        )
        if utils.is_primary(args):
            _logger.info(f'Wrapped model with AugAwareWrapper. Use Aug Main Loss: {args.use_aug_main_loss}')

        # Optionally freeze everything except self-supervised components (LALP + SSL FC head).
        if args.freeze_non_selfsup:
            frozen = []
            def _keep(name: str) -> bool:
                return name.endswith('lalp') or 'ssl_header' in name

            for name, p in model.named_parameters():
                if _keep(name):
                    continue
                p.requires_grad = False
                frozen.append(name)

            if utils.is_primary(args):
                _logger.info(
                    f'freeze-non-selfsup enabled. Frozen params: {len(frozen)} (kept LALP + ssl_header).'
                )

    if utils.is_primary(args):
        _logger.info(
            f'Model {safe_model_name(args.model)} created, param count:{sum([m.numel() for m in model.parameters()])}')

    data_config = resolve_data_config(vars(args), model=model, verbose=utils.is_primary(args))

    # setup augmentation batch splits for contrastive loss or split bn
    num_aug_splits = 0
    if args.aug_splits > 0:
        assert args.aug_splits > 1, 'A split of 1 makes no sense'
        num_aug_splits = args.aug_splits

    # enable split bn (separate bn stats per batch-portion)
    if args.split_bn:
        assert num_aug_splits > 1 or args.resplit
        model = convert_splitbn_model(model, max(num_aug_splits, 2))

    # move model to GPU, enable channels last layout if set
    model.to(device=device, dtype=model_dtype)  # FIXME move model device & dtype into create_model
    if args.channels_last:
        model.to(memory_format=torch.channels_last)

    # setup synchronized BatchNorm for distributed training
    if args.distributed and args.sync_bn:
        args.dist_bn = ''  # disable dist_bn when sync BN active
        assert not args.split_bn
        model = convert_sync_batchnorm(model)
        if utils.is_primary(args):
            _logger.info(
                'Converted model to use Synchronized BatchNorm. WARNING: You may have issues if using '
                'zero initialized BN layers (enabled by default for ResNets) while sync-bn enabled.')

    model_patch_size = None
    if args.naflex_loader:
        # NaFlexVit models have embeds.patch_size. Needs to be extracted here before mutating the model.
        model_patch_size = getattr(getattr(model, "embeds", None), "patch_size", None)

    if args.torchscript:
        assert not args.torchcompile
        assert not args.sync_bn, 'Cannot use SyncBatchNorm with torchscripted model'
        model = torch.jit.script(model)

    if not args.lr:
        if n_aug_mode:
            global_batch_size = base_batch_size * args.n_aug * args.world_size
        else:
            global_batch_size = args.batch_size * args.world_size * args.grad_accum_steps
        batch_ratio = global_batch_size / args.lr_base_size
        if not args.lr_base_scale:
            on = args.opt.lower()
            args.lr_base_scale = 'sqrt' if any([o in on for o in ('ada', 'lamb')]) else 'linear'
        if args.lr_base_scale == 'sqrt':
            batch_ratio = batch_ratio ** 0.5
        args.lr = args.lr_base * batch_ratio
        if utils.is_primary(args):
            _logger.info(
                f'Learning rate ({args.lr}) calculated from base learning rate ({args.lr_base}) '
                f'and effective global batch size ({global_batch_size}) with {args.lr_base_scale} scaling.')

    optimizer = create_optimizer_v2(
        model,
        **optimizer_kwargs(cfg=args),
        **args.opt_kwargs,
    )
    if utils.is_primary(args):
        defaults = copy.deepcopy(optimizer.defaults)
        defaults['weight_decay'] = args.weight_decay  # this isn't stored in optimizer.defaults
        defaults = ', '.join([f'{k}: {v}' for k, v in defaults.items()])
        logging.info(
            f'Created {type(optimizer).__name__} ({args.opt}) optimizer: {defaults}'
        )

    # setup automatic mixed-precision (AMP) loss scaling and op casting
    amp_autocast = suppress  # do nothing
    loss_scaler = None
    if args.amp:
        amp_autocast = partial(torch.autocast, device_type=device.type, dtype=amp_dtype)
        if device.type in ('cuda',) and amp_dtype == torch.float16:
            # loss scaler only used for float16 (half) dtype, bfloat16 does not need it
            loss_scaler = NativeScaler(device=device.type)
        if utils.is_primary(args):
            _logger.info('Using native Torch AMP. Training in mixed precision.')
    else:
        if utils.is_primary(args):
            _logger.info(f'AMP not enabled. Training in {model_dtype or torch.float32}.')

    # optionally resume from a checkpoint
    resume_epoch = None
    if args.resume:
        resume_epoch = resume_checkpoint(
            model,
            args.resume,
            optimizer=None if args.no_resume_opt else optimizer,
            loss_scaler=None if args.no_resume_opt else loss_scaler,
            log_info=utils.is_primary(args),
        )

    # setup exponential moving average of model weights, SWA could be used here too
    model_ema = None
    if args.model_ema:
        # Important to create EMA model after cuda(), DP wrapper, and AMP but before DDP wrapper
        model_ema = utils.ModelEmaV3(
            model,
            decay=args.model_ema_decay,
            use_warmup=args.model_ema_warmup,
            device='cpu' if args.model_ema_force_cpu else None,
        )
        if args.resume:
            load_checkpoint(model_ema.module, args.resume, use_ema=True)
        if args.torchcompile:
            model_ema = torch.compile(
                model_ema,
                backend=args.torchcompile,
                mode=args.torchcompile_mode,
            )

    # create the train and eval datasets
    if args.data and not args.data_dir:
        args.data_dir = args.data
    if args.input_img_mode is None:
        input_img_mode = 'RGB' if data_config['input_size'][0] == 3 else 'L'
    else:
        input_img_mode = args.input_img_mode

    dataset_train = create_dataset(
        args.dataset,
        root=args.data_dir,
        split=args.train_split,
        is_training=True,
        class_map=args.class_map,
        download=args.dataset_download,
        batch_size=args.batch_size,
        seed=args.seed,
        repeats=args.epoch_repeats,
        input_img_mode=input_img_mode,
        input_key=args.input_key,
        target_key=args.target_key,
        num_samples=args.train_num_samples,
        trust_remote_code=args.dataset_trust_remote_code,
    )

    dataset_eval = None
    if args.val_split:
        dataset_eval = create_dataset(
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
            num_samples=args.val_num_samples,
            trust_remote_code=args.dataset_trust_remote_code,
        )

    # create data loaders w/ augmentation pipeline
    train_interpolation = args.train_interpolation
    if args.no_aug or not train_interpolation:
        train_interpolation = data_config['interpolation']
        
    # Check if we should use the NaFlex scheduled loader
    common_loader_kwargs = dict(
        mean=data_config['mean'],
        std=data_config['std'],
        pin_memory=args.pin_mem,
        img_dtype=model_dtype or torch.float32,
        device=device,
        distributed=args.distributed,
        use_prefetcher=args.prefetcher,
    )

    train_loader_kwargs = dict(
        batch_size=train_loader_batch_size,
        is_training=True,
        no_aug=args.no_aug,
        re_prob=args.reprob,
        re_mode=args.remode,
        re_count=args.recount,
        re_split=args.resplit,
        train_crop_mode=args.train_crop_mode,
        scale=args.scale,
        ratio=args.ratio,
        hflip=args.hflip,
        vflip=args.vflip,
        color_jitter=args.color_jitter,
        color_jitter_prob=args.color_jitter_prob,
        grayscale_prob=args.grayscale_prob,
        gaussian_blur_prob=args.gaussian_blur_prob,
        auto_augment=args.aa,
        num_aug_repeats=args.aug_repeats,
        num_aug_splits=num_aug_splits,
        interpolation=train_interpolation,
        num_workers=args.workers,
        worker_seeding=args.worker_seeding,
    )

    mixup_fn = None
    mixup_args = {}
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_args = dict(
            mixup_alpha=args.mixup,
            cutmix_alpha=args.cutmix,
            cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob,
            switch_prob=args.mixup_switch_prob,
            mode=args.mixup_mode,
            label_smoothing=args.smoothing,
            num_classes=args.num_classes
        )

    naflex_mode = False
    if args.naflex_loader:
        if utils.is_primary(args):
            _logger.info('Using NaFlex loader')

        assert num_aug_splits <= 1, 'Augmentation splits not supported in NaFlex mode'
        naflex_mixup_fn = None
        if mixup_active:
            from timm.data import NaFlexMixup
            mixup_args.pop('mode')  # not supported
            mixup_args.pop('cutmix_minmax')  # not supported
            naflex_mixup_fn = NaFlexMixup(**mixup_args)

        # Check if we have model's patch size for NaFlex mode
        if model_patch_size is None:
            # Fallback to default
            model_patch_size = (16, 16)
            if utils.is_primary(args):
                _logger.warning(f'Could not determine model patch size, using default: {model_patch_size}')

        # Configure patch sizes for NaFlex loader
        patch_loader_kwargs = {}
        if args.naflex_patch_sizes:
            # Variable patch size mode
            patch_loader_kwargs['patch_size_choices'] = args.naflex_patch_sizes
            if args.naflex_patch_size_probs:
                if len(args.naflex_patch_size_probs) != len(args.naflex_patch_sizes):
                    parser.error('--naflex-patch-size-probs must have same length as --naflex-patch-sizes')
                patch_loader_kwargs['patch_size_choice_probs'] = args.naflex_patch_size_probs
            if utils.is_primary(args):
                _logger.info(f'Using variable patch sizes: {args.naflex_patch_sizes}')
        else:
            # Single patch size mode - use model's patch size
            patch_loader_kwargs['patch_size'] = model_patch_size
            if utils.is_primary(args):
                _logger.info(f'Using model patch size: {model_patch_size}')

        naflex_mode = True
        loader_train = create_naflex_loader(
            dataset=dataset_train,
            train_seq_lens=args.naflex_train_seq_lens,
            mixup_fn=naflex_mixup_fn,
            rank=args.rank,
            world_size=args.world_size,
            **patch_loader_kwargs,
            **common_loader_kwargs,
            **train_loader_kwargs,
        )
    else:
        # setup mixup / cutmix
        if n_aug_mode:
            collate_fn = collate_per_sample_aug
            
            # Define specific transforms for AugAware Mode
            # We want RRC + Flip -> Custom Augs -> ToTensor + Normalize
            
            mean = data_config['mean']
            std = data_config['std']
            input_size = data_config['input_size']
            
            # Pre-Augmentation: Geometric transforms that should apply to base image
            # Note: We use args.scale and args.ratio which are parsed from command line
            pre_transform = transforms.Compose([
                transforms.RandomResizedCrop(input_size[-2:], scale=args.scale, ratio=args.ratio, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(p=args.hflip),
            ])
            
            # Post-Augmentation: Convert to Tensor and Normalize
            post_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
            
            if utils.is_primary(args):
                _logger.info(f"AugAware Mode Transforms:\nPre: {pre_transform}\nPost: {post_transform}")
            
            dataset_train = PerSampleAugmentDataset(
                dataset_train, 
                aug_manager=aug_manager,
                pre_transform=pre_transform,
                post_transform=post_transform
            )
        else:
            collate_fn = None
            if mixup_active:
                if args.prefetcher:
                    assert not num_aug_splits  # collate conflict (need to support de-interleaving in collate mixup)
                    collate_fn = FastCollateMixup(**mixup_args)
                else:
                    mixup_fn = Mixup(**mixup_args)

            # wrap dataset in AugMix helper
            if num_aug_splits > 1:
                dataset_train = AugMixDataset(dataset_train, num_splits=num_aug_splits)

        # Use standard loader
        loader_train = create_loader(
            dataset_train,
            input_size=data_config['input_size'],
            collate_fn=collate_fn,
            use_multi_epochs_loader=args.use_multi_epochs_loader,
            **common_loader_kwargs,
            **train_loader_kwargs,
        )

    loader_eval = None
    if args.val_split:
        assert dataset_eval is not None
        eval_workers = args.workers
        if args.distributed and ('tfds' in args.dataset or 'wds' in args.dataset):
            # FIXME reduces validation padding issues when using TFDS, WDS w/ workers and distributed training
            eval_workers = min(2, args.workers)

        eval_loader_kwargs = dict(
            batch_size=args.validation_batch_size or args.batch_size,
            is_training=False,
            interpolation=data_config['interpolation'],
            num_workers=eval_workers,
            crop_pct=data_config['crop_pct'],
        )

        if args.naflex_loader:
            # Use largest sequence length for validation
            loader_eval = create_naflex_loader(
                dataset=dataset_eval,
                patch_size=model_patch_size,  # Use model's native patch size (already determined above)
                max_seq_len=args.naflex_max_seq_len,
                **common_loader_kwargs,
                **eval_loader_kwargs
            )
        else:
            # Use standard loader
            loader_eval = create_loader(
                dataset_eval,
                input_size=data_config['input_size'],
                **common_loader_kwargs,
                **eval_loader_kwargs,
            )

    # setup loss function
    if args.jsd_loss:
        assert num_aug_splits > 1  # JSD only valid with aug splits set
        train_loss_fn = JsdCrossEntropy(num_splits=num_aug_splits, smoothing=args.smoothing)
    elif mixup_active:
        # smoothing is handled with mixup target transform which outputs sparse, soft targets
        if args.bce_loss:
            train_loss_fn = BinaryCrossEntropy(
                target_threshold=args.bce_target_thresh,
                sum_classes=args.bce_sum,
                pos_weight=args.bce_pos_weight,
            )
        else:
            train_loss_fn = SoftTargetCrossEntropy()
    elif args.smoothing:
        if args.bce_loss:
            train_loss_fn = BinaryCrossEntropy(
                smoothing=args.smoothing,
                target_threshold=args.bce_target_thresh,
                sum_classes=args.bce_sum,
                pos_weight=args.bce_pos_weight,
            )
        else:
            train_loss_fn = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        train_loss_fn = nn.CrossEntropyLoss()
    
    # SSL Loss Function
    # We use SoftTargetCrossEntropy from timm to handle probability vector labels
    ssl_loss_fn = SoftTargetCrossEntropy().to(device)
    
    train_loss_fn = train_loss_fn.to(device=device)
    validate_loss_fn = nn.CrossEntropyLoss().to(device=device)

    # Setup training task (classification or distillation)
    if args.kd_model_name is not None:
        # Create distillation task (teacher created internally from model name)
        if args.kd_distill_type == 'logit':
            task = LogitDistillationTask(
                student_model=model,
                teacher_model=args.kd_model_name,
                criterion=train_loss_fn,
                loss_type=args.kd_loss_type,
                distill_loss_weight=args.distill_loss_weight,
                task_loss_weight=args.task_loss_weight,
                temperature=args.kd_temperature,
                device=device,
                dtype=model_dtype,
                verbose=utils.is_primary(args),
            )
        elif args.kd_distill_type == 'feature':
            task = FeatureDistillationTask(
                student_model=model,
                teacher_model=args.kd_model_name,
                criterion=train_loss_fn,
                distill_loss_weight=args.distill_loss_weight,
                task_loss_weight=args.task_loss_weight,
                student_feature_dim=args.kd_student_feature_dim,
                teacher_feature_dim=args.kd_teacher_feature_dim,
                device=device,
                dtype=model_dtype,
                verbose=utils.is_primary(args),
            )
        elif args.kd_distill_type == 'token':
            task = TokenDistillationTask(
                student_model=model,
                teacher_model=args.kd_model_name,
                criterion=train_loss_fn,
                distill_type=args.kd_token_distill_type,
                distill_loss_weight=args.distill_loss_weight,
                task_loss_weight=args.task_loss_weight,
                temperature=args.kd_temperature,
                device=device,
                dtype=model_dtype,
                verbose=utils.is_primary(args),
            )
        else:
            raise ValueError(f"Unknown distillation type: {args.kd_distill_type}")
    else:
        # Standard classification task
        task = ClassificationTask(
            model=model,
            criterion=train_loss_fn,
            device=device,
            dtype=model_dtype,
            verbose=utils.is_primary(args),
        )

    # Prepare task for distributed training
    if args.distributed:
        if utils.is_primary(args):
            _logger.info("Preparing task for distributed training")
        task.prepare_distributed(device_ids=[device])

    # Compile task if requested (should be done after DDP)
    if args.torchcompile:
        assert has_compile, 'A version of torch w/ torch.compile() is required for --compile, possibly a nightly.'
        if utils.is_primary(args):
            _logger.info(f"Compiling task with backend={args.torchcompile}, mode={args.torchcompile_mode}")
        task = torch.compile(task, backend=args.torchcompile, mode=args.torchcompile_mode)

    # setup checkpoint saver and eval metric tracking
    eval_metric = args.eval_metric if loader_eval is not None else 'loss'
    decreasing_metric = eval_metric == 'loss'
    best_metric = None
    best_epoch = None
    saver = None
    output_dir = None
    if utils.is_primary(args):
        if args.experiment:
            exp_name = args.experiment
        else:
            exp_name = '-'.join([
                datetime.now().strftime("%Y%m%d-%H%M%S"),
                safe_model_name(args.model),
                str(data_config['input_size'][-1])
            ])
        output_dir = utils.get_outdir(args.output if args.output else './output/train', exp_name)
        saver = utils.CheckpointSaver(
            model=model,
            optimizer=optimizer,
            args=args,
            model_ema=model_ema,
            amp_scaler=loss_scaler,
            checkpoint_dir=output_dir,
            recovery_dir=output_dir,
            decreasing=decreasing_metric,
            max_history=args.checkpoint_hist
        )
        with open(os.path.join(output_dir, 'args.yaml'), 'w') as f:
            f.write(args_text)

        if args.log_wandb:
            if has_wandb:
                assert not args.wandb_resume_id or args.resume
                wandb.init(
                    project=args.wandb_project,
                    name=exp_name,
                    config=args,
                    tags=args.wandb_tags,
                    resume="must" if args.wandb_resume_id else None,
                    id=args.wandb_resume_id if args.wandb_resume_id else None,
                )
            else:
                _logger.warning(
                    "You've requested to log metrics to wandb but package not found. "
                    "Metrics not being logged to wandb, try `pip install wandb`")

    # setup learning rate schedule and starting epoch
    updates_per_epoch = (len(loader_train) + args.grad_accum_steps - 1) // args.grad_accum_steps
    lr_scheduler, num_epochs = create_scheduler_v2(
        optimizer,
        **scheduler_kwargs(args, decreasing_metric=decreasing_metric),
        updates_per_epoch=updates_per_epoch,
    )
    start_epoch = 0
    if args.start_epoch is not None:
        # a specified start_epoch will always override the resume epoch
        start_epoch = args.start_epoch
    elif resume_epoch is not None:
        start_epoch = resume_epoch
    if lr_scheduler is not None and start_epoch > 0:
        if args.sched_on_updates:
            lr_scheduler.step_update(start_epoch * updates_per_epoch)
        else:
            lr_scheduler.step(start_epoch)

    if utils.is_primary(args):
        if args.warmup_prefix:
            sched_explain = '(warmup_epochs + epochs + cooldown_epochs). Warmup added to total when warmup_prefix=True'
        else:
            sched_explain = '(epochs + cooldown_epochs). Warmup within epochs when warmup_prefix=False'
        _logger.info(
            f'Scheduled epochs: {num_epochs} {sched_explain}. '
            f'LR stepped per {"epoch" if lr_scheduler.t_in_epochs else "update"}.')

    results = []
    try:
        for epoch in range(start_epoch, num_epochs):
            if hasattr(dataset_train, 'set_epoch'):
                dataset_train.set_epoch(epoch)
            elif args.distributed and hasattr(loader_train.sampler, 'set_epoch'):
                loader_train.sampler.set_epoch(epoch)

            train_metrics = train_one_epoch(
                epoch,
                model,
                loader_train,
                optimizer,
                args,
                task=task,
                device=device,
                lr_scheduler=lr_scheduler,
                saver=saver,
                output_dir=output_dir,
                amp_autocast=amp_autocast,
                loss_scaler=loss_scaler,
                model_dtype=model_dtype,
                model_ema=model_ema,
                mixup_fn=mixup_fn,
                num_updates_total=num_epochs * updates_per_epoch,
                naflex_mode=naflex_mode,
                ssl_loss_fn=ssl_loss_fn, # Pass SSL Loss Fn
            )

            if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                if utils.is_primary(args):
                    _logger.info("Distributing BatchNorm running means and vars")
                utils.distribute_bn(model, args.world_size, args.dist_bn == 'reduce')

            epoch_p_1 = epoch + 1
            if epoch_p_1 % args.val_interval != 0 and epoch_p_1 != num_epochs:
                if utils.is_primary(args):
                    _logger.info("Skipping eval and checkpointing ")
                if lr_scheduler is not None:
                    # step LR for next epoch, take care when using metric dependent lr_scheduler
                    lr_scheduler.step(epoch_p_1, metric=None)
                # Skip validation and metric logic
                # FIXME we could make the logic below able to handle no eval metrics more gracefully,
                #  but for simplicity opting to just skip for now.
                continue

            if loader_eval is not None:
                eval_metrics = validate(
                    model,
                    loader_eval,
                    validate_loss_fn,
                    args,
                    device=device,
                    amp_autocast=amp_autocast,
                    model_dtype=model_dtype,
                )

                if model_ema is not None and not args.model_ema_force_cpu:
                    if args.distributed and args.dist_bn in ('broadcast', 'reduce'):
                        utils.distribute_bn(model_ema, args.world_size, args.dist_bn == 'reduce')

                    ema_eval_metrics = validate(
                        model_ema,
                        loader_eval,
                        validate_loss_fn,
                        args,
                        device=device,
                        amp_autocast=amp_autocast,
                        log_suffix=' (EMA)',
                    )
                    eval_metrics = ema_eval_metrics
            else:
                eval_metrics = None

            if output_dir is not None:
                lrs = [param_group['lr'] for param_group in optimizer.param_groups]
                utils.update_summary(
                    epoch,
                    train_metrics,
                    eval_metrics,
                    filename=os.path.join(output_dir, 'summary.csv'),
                    lr=sum(lrs) / len(lrs),
                    write_header=best_metric is None,
                    log_wandb=args.log_wandb and has_wandb,
                )

            if eval_metrics is not None:
                latest_metric = eval_metrics[eval_metric]
            else:
                latest_metric = train_metrics[eval_metric]

            if saver is not None:
                # save proper checkpoint with eval metric
                best_metric, best_epoch = saver.save_checkpoint(epoch, metric=latest_metric)

            if lr_scheduler is not None:
                # step LR for next epoch
                lr_scheduler.step(epoch_p_1, latest_metric)

            latest_results = {
                'epoch': epoch,
                'train': train_metrics,
            }
            if eval_metrics is not None:
                latest_results['validation'] = eval_metrics
            results.append(latest_results)

    except KeyboardInterrupt:
        pass

    if args.distributed:
        torch.distributed.destroy_process_group()

    if best_metric is not None:
        # log best metric as tracked by checkpoint saver
        _logger.info('*** Best metric: {0} (epoch {1})'.format(best_metric, best_epoch))

    if utils.is_primary(args):
        # for parsable results display, dump top-10 summaries to avoid excess console spam
        display_results = sorted(
            results,
            key=lambda x: x.get('validation', x.get('train')).get(eval_metric, 0),
            reverse=decreasing_metric,
        )
        print(f'--result\n{json.dumps(display_results[-10:], indent=4)}')


def train_one_epoch(
        epoch,
        model,
        loader,
        optimizer,
        args,
        task=None,
        device=torch.device('cuda'),
        lr_scheduler=None,
        saver=None,
        output_dir=None,
        amp_autocast=suppress,
        loss_scaler=None,
        model_dtype=None,
        model_ema=None,
        mixup_fn=None,
        num_updates_total=None,
        naflex_mode=False,
        ssl_loss_fn=None,
):
    if args.mixup_off_epoch and epoch >= args.mixup_off_epoch:
        if args.prefetcher and loader.mixup_enabled:
            loader.mixup_enabled = False
        elif mixup_fn is not None:
            mixup_fn.mixup_enabled = False

    second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
    has_no_sync = hasattr(model, "no_sync")
    update_time_m = utils.AverageMeter()
    data_time_m = utils.AverageMeter()
    losses_m = utils.AverageMeter()
    # Add meters for Main Loss and SSL Loss
    main_losses_m = utils.AverageMeter()
    ssl_losses_m = utils.AverageMeter()
    kl_losses_m = utils.AverageMeter()
    aug_main_losses_m = utils.AverageMeter()

    model.train()

    accum_steps = args.grad_accum_steps
    last_accum_steps = len(loader) % accum_steps
    updates_per_epoch = (len(loader) + accum_steps - 1) // accum_steps
    num_updates = epoch * updates_per_epoch
    last_batch_idx = len(loader) - 1
    last_batch_idx_to_accum = len(loader) - last_accum_steps

    data_start_time = update_start_time = time.time()
    optimizer.zero_grad()
    update_sample_count = 0
    for batch_idx, batch_data in enumerate(loader):
        # Unpack based on loader type (Standard vs AugAware)
        if hasattr(loader.dataset, 'aug_manager') or (isinstance(batch_data, (tuple, list)) and len(batch_data) == 3):
            # AugAware Mode: (input, aug_labels, target)
            input, aug_labels, target = batch_data
            aug_labels = aug_labels.to(device=device)
        else:
            # Standard Mode: (input, target)
            input, target = batch_data
            aug_labels = None
            
        last_batch = batch_idx == last_batch_idx
        need_update = last_batch or (batch_idx + 1) % accum_steps == 0
        update_idx = batch_idx // accum_steps
        if batch_idx >= last_batch_idx_to_accum:
            accum_steps = last_accum_steps

        if not args.prefetcher:
            input, target = input.to(device=device, dtype=model_dtype), target.to(device=device)
            if mixup_fn is not None:
                input, target = mixup_fn(input, target)
        if args.channels_last:
            input = input.contiguous(memory_format=torch.channels_last)

        # multiply by accum steps to get equivalent for full update
        data_time_m.update(accum_steps * (time.time() - data_start_time))

        def _forward():
            with amp_autocast():
                # AugAwareWrapper returns (main_logits, ssl_logits) if training
                # But 'task' might wrap it if using Distillation
                # If we are using standard classification task with AugAwareWrapper, task(input, target) calls model(input)
                # We need to handle the tuple return if model is wrapped
                
                if aug_labels is not None:
                    # Augmentation Aware Training
                    
                    output = model(input)
                    if isinstance(output, tuple) and len(output) == 4:
                        main_logits, aug_ssl_logits, clean_ssl_logits, aug_main_logits = output
                        
                        # --- 1. Main Loss (Classification of Clean Image) ---
                        # input has batch size B * (N+1).
                        # main_logits has batch size B
                        
                        B = main_logits.shape[0]
                        # Assume last element of each group is the clean image target.
                        # target: [T1, ..., T1, T2, ...]
                        views = input.shape[0] // B
                        main_targets = target[views-1::views] # Pick the last one (Clean)
                        
                        main_loss = task.criterion(main_logits, main_targets)
                        
                        # --- 2. SSL Loss (Augmentation Type Prediction for Aug Images) ---
                        # aug_ssl_logits: [B, N, N]
                        # aug_labels: [B, N+1, N] -> we need [B, N, N]
                        
                        aug_labels_reshaped = aug_labels.view(B, views, -1)
                        # Take first N (Augmented) labels
                        aug_labels_for_loss = aug_labels_reshaped[:, :aug_ssl_logits.shape[1], :] # [B, N, N]
                        
                        ssl_loss = ssl_loss_fn(aug_ssl_logits.reshape(-1, aug_ssl_logits.shape[-1]), 
                                             aug_labels_for_loss.reshape(-1, aug_labels_for_loss.shape[-1]))
                        
                        # --- 3. KL Divergence Loss (Uniformity for Clean Image) ---
                        # clean_ssl_logits: [B, N, N] (Batch, N_Stems, N_Classes)
                        # We want each stem's prediction for clean image to be Uniform Distribution
                        # Target: Uniform distribution over N classes (1/N)
                        
                        # Flatten to [B*N, N_Classes]
                        clean_log_probs = torch.log_softmax(clean_ssl_logits.reshape(-1, clean_ssl_logits.shape[-1]), dim=-1)
                        
                        # Create Uniform Target: [B*N, N_Classes]
                        # We can use kl_div with target as probabilities
                        num_classes = clean_ssl_logits.shape[-1]
                        # uniform_target = torch.full_as(clean_log_probs, 1.0 / num_classes) # torch.full_as is available in newer torch
                        # Use full_like or full
                        uniform_target = torch.full_like(clean_log_probs, 1.0 / num_classes)
                        
                        # F.kl_div(input, target, reduction='batchmean')
                        # input should be log-probabilities, target is probabilities
                        kl_loss = torch.nn.functional.kl_div(clean_log_probs, uniform_target, reduction='batchmean')
                        
                        # --- 4. Optional: Augmented Main Loss (Classification of Aug Images) ---
                        aug_main_loss = torch.tensor(0.0, device=device)
                        if aug_main_logits is not None:
                            # aug_main_logits: [B*N, Num_Classes]
                            # We need targets for all augmented images.
                            # target: [T1, T1(Aug1), ..., T1(AugN), T1(Clean), T2...]
                            # We want first N targets for each sample.
                            
                            target_reshaped = target.view(B, views)
                            # Take first N targets: [B, N]
                            aug_targets = target_reshaped[:, :aug_ssl_logits.shape[1]].reshape(-1) # [B*N]
                            
                            aug_main_loss = task.criterion(aug_main_logits, aug_targets)
                        
                        # --- Total Loss ---
                        _loss = main_loss + \
                                (args.ssl_lambda * ssl_loss) + \
                                (args.kl_loss_weight * kl_loss) + \
                                (args.aug_main_loss_weight * aug_main_loss)
                        
                        # Update meters
                        main_losses_m.update(main_loss.item() * accum_steps, B)
                        ssl_losses_m.update(ssl_loss.item() * accum_steps, B)
                        kl_losses_m.update(kl_loss.item() * accum_steps, B)
                        if aug_main_logits is not None:
                            aug_main_losses_m.update(aug_main_loss.item() * accum_steps, B * aug_ssl_logits.shape[1])
                        
                        # Create result dict for logging
                        result = {'loss': _loss}
                    else:
                        # Fallback if something is wrong (e.g. inference mode in train loop?)
                        result = task(input, target)
                        _loss = result['loss']
                else:
                    # Standard Training
                    result = task(input, target)
                    _loss = result['loss']

            if accum_steps > 1:
                _loss /= accum_steps
            return _loss, result

        def _backward(_loss):
            if loss_scaler is not None:
                loss_scaler(
                    _loss,
                    optimizer,
                    clip_grad=args.clip_grad,
                    clip_mode=args.clip_mode,
                    parameters=model_parameters(model, exclude_head='agc' in args.clip_mode),
                    create_graph=second_order,
                    need_update=need_update,
                )
            else:
                _loss.backward(create_graph=second_order)
                if need_update:
                    if args.clip_grad is not None:
                        utils.dispatch_clip_grad(
                            model_parameters(model, exclude_head='agc' in args.clip_mode),
                            value=args.clip_grad,
                            mode=args.clip_mode,
                        )
                    optimizer.step()

        if naflex_mode:
            assert isinstance(input, dict)
            batch_size = input['patches'].shape[0]

            # scale gradient vs the minimum batch size (for max seq len)
            if not args.naflex_loss_scale or args.naflex_loss_scale == 'none':
                local_scale = 1.0
            else:
                local_scale = (batch_size / args.batch_size)
                if local_scale == 'sqrt':
                    local_scale = local_scale ** 0.5

            if args.distributed:
                # scale gradient btw distributed ranks, each one can have different batch size
                global_batch_size = utils.reduce_tensor(
                    torch.tensor(batch_size, device=device, dtype=torch.float32),
                    1 # SUM
                )
                dist_scale = args.world_size * batch_size / global_batch_size
            else:
                dist_scale = None
                global_batch_size = batch_size

            if has_no_sync and not need_update:
                with model.no_sync():
                    loss, result = _forward()
                    scaled_loss = local_scale * loss
                    if dist_scale is not None:
                        scaled_loss *= dist_scale
                    _backward(scaled_loss)
            else:
                loss, result = _forward()
                scaled_loss = local_scale * loss
                if dist_scale is not None:
                    scaled_loss *= dist_scale
                _backward(scaled_loss)
        else:
            global_batch_size = batch_size = input.shape[0]
            if args.distributed:
                global_batch_size *= args.world_size

            if has_no_sync and not need_update:
                with model.no_sync():
                    loss, result = _forward()
                    _backward(loss)
            else:
                loss, result = _forward()
                _backward(loss)

        losses_m.update(loss.item() * accum_steps, batch_size)
        update_sample_count += global_batch_size

        if not need_update:
            data_start_time = time.time()
            continue

        num_updates += 1
        optimizer.zero_grad()
        if model_ema is not None:
            model_ema.update(model, step=num_updates)

        if args.synchronize_step:
            if device.type == 'cuda':
                torch.cuda.synchronize()
            elif device.type == 'npu':
                torch.npu.synchronize()
        time_now = time.time()

        update_time_m.update(time.time() - update_start_time)
        update_start_time = time_now

        if update_idx % args.log_interval == 0 or last_batch:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            loss_avg, loss_now = losses_m.avg, losses_m.val
            if args.distributed:
                # synchronize current step and avg loss, each process keeps its own running avg
                loss_avg = utils.reduce_tensor(loss.new([loss_avg]), args.world_size).item()
                loss_now = utils.reduce_tensor(loss.new([loss_now]), args.world_size).item()

            if utils.is_primary(args):
                _logger.info(
                    f'Train: {epoch} [{update_idx:>4d}/{updates_per_epoch} '
                    f'({100. * (update_idx + 1) / updates_per_epoch:>3.0f}%)]  '
                    f'Loss: {loss_now:#.3g} ({loss_avg:#.3g})  '
                    f'Main Loss: {main_losses_m.val:#.3g} ({main_losses_m.avg:#.3g})  '
                    f'SSL Loss: {ssl_losses_m.val:#.3g} ({ssl_losses_m.avg:#.3g})  '
                    f'KL Loss: {kl_losses_m.val:#.3g} ({kl_losses_m.avg:#.3g})  '
                    f'Aug Main Loss: {aug_main_losses_m.val:#.3g} ({aug_main_losses_m.avg:#.3g})  '
                    f'Time: {update_time_m.val:.3f}s, {update_sample_count / update_time_m.val:>7.2f}/s  '
                    f'({update_time_m.avg:.3f}s, {update_sample_count / update_time_m.avg:>7.2f}/s)  '
                    f'LR: {lr:.3e}  '
                    f'Data: {data_time_m.val:.3f} ({data_time_m.avg:.3f})'
                )

                if args.save_images and output_dir:
                    torchvision.utils.save_image(
                        input,
                        os.path.join(output_dir, 'train-batch-%d.jpg' % batch_idx),
                        padding=0,
                        normalize=True
                    )

        if saver is not None and args.recovery_interval and (
                (update_idx + 1) % args.recovery_interval == 0):
            saver.save_recovery(epoch, batch_idx=update_idx)

        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

        update_sample_count = 0
        data_start_time = time.time()
        # end for

    if hasattr(optimizer, 'sync_lookahead'):
        optimizer.sync_lookahead()

    loss_avg = losses_m.avg
    if args.distributed:
        # synchronize avg loss, each process keeps its own running avg
        loss_avg = torch.tensor([loss_avg], device=device, dtype=torch.float32)
        loss_avg = utils.reduce_tensor(loss_avg, args.world_size).item()
    return OrderedDict([('loss', loss_avg)])


def validate(
        model,
        loader,
        loss_fn,
        args,
        device=torch.device('cuda'),
        amp_autocast=suppress,
        model_dtype=None,
        log_suffix=''
):
    batch_time_m = utils.AverageMeter()
    losses_m = utils.AverageMeter()
    top1_m = utils.AverageMeter()
    top5_m = utils.AverageMeter()

    model.eval()

    end = time.time()
    last_idx = len(loader) - 1
    with torch.inference_mode():
        for batch_idx, batch_data in enumerate(loader):
            # Validation loop might receive different structure depending on loader
            # But standard validation dataset (created in main) is likely ImageFolder-like.
            # If we used PerSampleAugmentDataset for valid, it would return tuples.
            # But main() L762 creates standard dataset for eval.
            # So batch_data is (input, target).
            
            if isinstance(batch_data, (tuple, list)) and len(batch_data) == 3:
                 # Just in case validation loader is also wrapped (unlikely based on main)
                 input, _, target = batch_data
            else:
                 input, target = batch_data
                 
            last_batch = batch_idx == last_idx
            if not args.prefetcher:
                input = input.to(device=device, dtype=model_dtype)
                target = target.to(device=device)
            if args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)

            with amp_autocast():
                output = model(input)
                if isinstance(output, (tuple, list)):
                    # If model returns tuple in eval mode (it shouldn't if implemented correctly in forward), pick first
                    output = output[0]

                # augmentation reduction
                reduce_factor = args.tta
                if reduce_factor > 1:
                    output = output.unfold(0, reduce_factor, reduce_factor).mean(dim=2)
                    target = target[0:target.size(0):reduce_factor]

                loss = loss_fn(output, target)
            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))

            if args.distributed:
                reduced_loss = utils.reduce_tensor(loss.data, args.world_size)
                acc1 = utils.reduce_tensor(acc1, args.world_size)
                acc5 = utils.reduce_tensor(acc5, args.world_size)
            else:
                reduced_loss = loss.data

            if device.type == 'cuda':
                torch.cuda.synchronize()
            elif device.type == "npu":
                torch.npu.synchronize()

            batch_size = output.shape[0]
            losses_m.update(reduced_loss.item(), batch_size)
            top1_m.update(acc1.item(), batch_size)
            top5_m.update(acc5.item(), batch_size)

            batch_time_m.update(time.time() - end)
            end = time.time()
            if utils.is_primary(args) and (last_batch or batch_idx % args.log_interval == 0):
                log_name = 'Test' + log_suffix
                _logger.info(
                    f'{log_name}: [{batch_idx:>4d}/{last_idx}]  '
                    f'Time: {batch_time_m.val:.3f} ({batch_time_m.avg:.3f})  '
                    f'Loss: {losses_m.val:>7.3f} ({losses_m.avg:>6.3f})  '
                    f'Acc@1: {top1_m.val:>7.3f} ({top1_m.avg:>7.3f})  '
                    f'Acc@5: {top5_m.val:>7.3f} ({top5_m.avg:>7.3f})'
                )

    metrics = OrderedDict([('loss', losses_m.avg), ('top1', top1_m.avg), ('top5', top5_m.avg)])

    return metrics


if __name__ == '__main__':
    main()

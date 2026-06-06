#!/usr/bin/env python3
"""Compute Feature Space Centroid (FSC) from stem layer.

Supports configurable stem depth via ``--stem-mode``:

  CNN (ResNet, etc.):
    - ``conv1``:          conv1 only (raw, unnormalized).          → 1,024-dim
    - ``conv1_bn1``:      conv1 + bn1 (batch-normalized).         → 1,024-dim
    - ``conv1_bn1_act1``: conv1 + bn1 + act1 (normalized + act).  → 1,024-dim

  ViT:
    - ``patch_embed``:      patch_embed.proj only (raw).           → 12,288-dim (ViT-B/16)
    - ``patch_embed_norm``: patch_embed.proj + norm.               → 12,288-dim (ViT-B/16)

All modes: stem → AdaptiveAvgPool2d(4,4) → Flatten → feature vector.

Usage (ResNet):
    python compute_fsc_conv1_only.py --data-dir /path/to/imagenet/train \\
        --model resnet50 --pretrained --output-dir ./ZOA_FSC \\
        --stem-mode conv1

Usage (ViT):
    python compute_fsc_conv1_only.py --data-dir /path/to/imagenet/train \\
        --model vit_base_patch16_224 --pretrained --output-dir ./ZOA_FSC \\
        --stem-mode patch_embed

Output:
    {output_dir}/{model_name}_FSC_{stem_mode}.pth
"""

import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
from tqdm import tqdm

from timm.data import create_dataset, create_loader, resolve_data_config
from timm.models import create_model, load_checkpoint
from timm.utils import setup_default_logging

_logger = logging.getLogger("compute_fsc_stem")


def parse_args():
    parser = argparse.ArgumentParser(description="Compute Feature Space Centroid (FSC) from stem layer")
    
    # Data arguments
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to training dataset (e.g., ImageNet train folder)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="",
        help='Dataset type + name ("<type>/<name>") (default: ImageFolder)',
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to use (default: train)",
    )
    
    # Model arguments
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="resnet50",
        help="Model architecture (default: resnet50)",
    )
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Use pretrained model weights",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="",
        help="Path to model checkpoint (optional)",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=1000,
        help="Number of classes (default: 1000 for ImageNet)",
    )
    
    # Processing arguments
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for inference (default: 256)",
    )
    parser.add_argument(
        "-j",
        "--workers",
        type=int,
        default=8,
        help="Number of data loading workers (default: 8)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (default: cuda)",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use automatic mixed precision",
    )
    
    # Stem mode
    parser.add_argument(
        "--stem-mode",
        type=str,
        default="conv1",
        choices=[
            "conv1", "conv1_bn1", "conv1_bn1_act1",    # CNN (ResNet, etc.)
            "patch_embed", "patch_embed_norm",           # ViT
        ],
        help="Stem depth: conv1* for CNN, patch_embed* for ViT (default: conv1)",
    )
    
    # Output arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./FSC",
        help="Output directory for FSC files (default: ./FSC)",
    )
    
    return parser.parse_args()


class Conv1OnlyFeatureExtractor(nn.Module):
    """Legacy: Extract features from conv1 ONLY.  Kept for backward compat."""

    def __init__(self, model):
        super().__init__()
        if hasattr(model, 'conv1'):
            self.conv1 = model.conv1
        else:
            raise ValueError("Model does not have conv1 layer")
        self.pool = nn.AdaptiveAvgPool2d((4, 4))

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = x.flatten(1)
        return x


class StemFeatureExtractor(nn.Module):
    """Configurable stem feature extractor.

    CNN modes (ResNet, etc.):
        ``conv1``:          conv1 only (raw, unnormalized).
        ``conv1_bn1``:      conv1 + bn1 (batch-normalized).
        ``conv1_bn1_act1``: conv1 + bn1 + act1 (normalized + activated).

    ViT modes:
        ``patch_embed``:      patch_embed.proj only (Conv2d projection, raw).
        ``patch_embed_norm``: patch_embed.proj + patch_embed.norm.

    Output: ``[B, feature_dim]`` flattened vector.
        CNN (ResNet50): 64*4*4 = 1,024-dim.
        ViT-B/16:       768*4*4 = 12,288-dim.
    """

    CNN_MODES = ('conv1', 'conv1_bn1', 'conv1_bn1_act1')
    VIT_MODES = ('patch_embed', 'patch_embed_norm')

    def __init__(self, model, stem_mode='conv1'):
        super().__init__()
        self.stem_mode = stem_mode

        if stem_mode in self.CNN_MODES:
            self._init_cnn(model, stem_mode)
        elif stem_mode in self.VIT_MODES:
            self._init_vit(model, stem_mode)
        else:
            raise ValueError(
                f"Unknown stem_mode='{stem_mode}'. "
                f"Choose from {self.CNN_MODES + self.VIT_MODES}"
            )

        self.pool = nn.AdaptiveAvgPool2d((4, 4))

    def _init_cnn(self, model, stem_mode):
        if not hasattr(model, 'conv1'):
            raise ValueError("Model does not have conv1 layer")
        self.conv1 = model.conv1

        if stem_mode in ('conv1_bn1', 'conv1_bn1_act1'):
            if not hasattr(model, 'bn1'):
                raise ValueError(f"stem_mode='{stem_mode}' requires bn1")
            self.bn1 = model.bn1

        if stem_mode == 'conv1_bn1_act1':
            if not hasattr(model, 'act1'):
                raise ValueError(f"stem_mode='{stem_mode}' requires act1")
            self.act1 = model.act1

    def _init_vit(self, model, stem_mode):
        if not hasattr(model, 'patch_embed'):
            raise ValueError("Model does not have patch_embed layer")
        pe = model.patch_embed
        if not hasattr(pe, 'proj'):
            raise ValueError("patch_embed does not have proj (Conv2d) layer")
        self.proj = pe.proj

        if stem_mode == 'patch_embed_norm':
            if hasattr(pe, 'norm') and not isinstance(pe.norm, nn.Identity):
                self.patch_norm = pe.norm
            else:
                _logger.warning(
                    "patch_embed_norm requested but patch_embed.norm is Identity — "
                    "falling back to patch_embed (no norm)."
                )
                self.patch_norm = None
        else:
            self.patch_norm = None

    def forward(self, x):
        if self.stem_mode in self.CNN_MODES:
            x = self.conv1(x)
            if self.stem_mode in ('conv1_bn1', 'conv1_bn1_act1'):
                x = self.bn1(x)
            if self.stem_mode == 'conv1_bn1_act1':
                x = self.act1(x)
        else:
            # ViT: patch_embed.proj → [B, embed_dim, H', W'] (NCHW)
            x = self.proj(x)
            if self.patch_norm is not None:
                B, C, H, W = x.shape
                x = x.flatten(2).transpose(1, 2)
                x = self.patch_norm(x)
                x = x.transpose(1, 2).view(B, C, H, W)

        x = self.pool(x)
        x = x.flatten(1)
        return x


def extract_stem_features_and_predictions(model, stem_extractor, loader, device, amp=False):
    """Extract stem features and predictions from model.
    
    Args:
        model: The pretrained model (for classification).
        stem_extractor: StemFeatureExtractor for extracting stem features.
        loader: DataLoader for the dataset.
        device: Device to use.
        amp: Whether to use automatic mixed precision.
    
    Returns:
        all_features: List of feature tensors [N, 1024].
        all_predictions: List of prediction tensors [N].
        all_targets: List of target tensors [N].
    """
    model.eval()
    stem_extractor.eval()
    
    all_features = []
    all_predictions = []
    all_targets = []
    
    num_batches = len(loader)
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(tqdm(loader, desc="Extracting stem features", total=num_batches, ncols=80)):
            images = images.to(device)
            targets = targets.to(device)
            
            with torch.cuda.amp.autocast(enabled=amp):
                # Extract stem features (1024-dim)
                stem_features = stem_extractor(images)
                
                # Get predictions using full model
                logits = model(images)
                predictions = logits.argmax(dim=1)
            
            # Move to CPU to save GPU memory
            all_features.append(stem_features.float().cpu())
            all_predictions.append(predictions.cpu())
            all_targets.append(targets.cpu())
    
    return (
        torch.cat(all_features, dim=0),
        torch.cat(all_predictions, dim=0),
        torch.cat(all_targets, dim=0),
    )


def compute_fsc(features, predictions, targets, num_classes):
    """Compute Feature Space Centroid for each class.
    
    Only uses correctly classified samples to compute the centroid.
    
    Args:
        features: Feature tensor [N, feature_dim].
        predictions: Prediction tensor [N].
        targets: Target tensor [N].
        num_classes: Number of classes.
    
    Returns:
        fsc: Dictionary containing:
            - 'centroids': Tensor [num_classes, feature_dim] - centroid for each class
            - 'counts': Tensor [num_classes] - number of correct samples per class
            - 'total_correct': Total number of correctly classified samples
            - 'total_samples': Total number of samples
    """
    feature_dim = features.shape[1]
    
    # Initialize accumulators
    class_feature_sum = torch.zeros(num_classes, feature_dim, dtype=torch.float64)
    class_counts = torch.zeros(num_classes, dtype=torch.long)
    
    # Find correctly classified samples
    correct_mask = (predictions == targets)
    correct_indices = correct_mask.nonzero(as_tuple=True)[0]
    
    _logger.info(f"Total samples: {len(targets)}")
    _logger.info(f"Correctly classified: {len(correct_indices)} ({100 * len(correct_indices) / len(targets):.2f}%)")
    
    # Accumulate features for each class (only correct predictions)
    for idx in tqdm(correct_indices, desc="Computing centroids"):
        cls = targets[idx].item()
        class_feature_sum[cls] += features[idx].double()
        class_counts[cls] += 1
    
    # Compute mean (centroid) for each class
    centroids = torch.zeros(num_classes, feature_dim, dtype=torch.float32)
    for cls in range(num_classes):
        if class_counts[cls] > 0:
            centroids[cls] = (class_feature_sum[cls] / class_counts[cls]).float()
        else:
            _logger.warning(f"Class {cls} has no correctly classified samples!")
    
    return {
        "centroids": centroids,
        "counts": class_counts,
        "total_correct": len(correct_indices),
        "total_samples": len(targets),
        "feature_dim": feature_dim,
    }


def main():
    setup_default_logging()
    args = parse_args()
    
    # Setup device
    device = torch.device(args.device)
    _logger.info(f"Using device: {device}")
    
    # Create model
    _logger.info(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.num_classes,
    )
    
    if args.checkpoint:
        _logger.info(f"Loading checkpoint: {args.checkpoint}")
        load_checkpoint(model, args.checkpoint)
    
    model = model.to(device)
    model.eval()
    
    param_count = sum(p.numel() for p in model.parameters())
    _logger.info(f"Model {args.model} created, param count: {param_count / 1e6:.2f}M")
    
    # Create stem feature extractor with configurable depth
    _logger.info(f"Creating stem feature extractor (stem_mode={args.stem_mode})...")
    stem_extractor = StemFeatureExtractor(model, stem_mode=args.stem_mode)
    stem_extractor = stem_extractor.to(device)
    stem_extractor.eval()
    
    # Verify stem output dimensions
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        dummy_output = stem_extractor(dummy_input)
        _logger.info(f"Stem feature dimension: {dummy_output.shape[1]}")
    
    # Resolve data config from model
    data_config = resolve_data_config(vars(args), model=model, verbose=True)
    
    # Create dataset and loader (NO AUGMENTATION - is_training=False)
    _logger.info(f"Loading dataset from: {args.data_dir}")
    _logger.info("Using no augmentation (validation transforms)")
    dataset = create_dataset(
        root=args.data_dir,
        name=args.dataset,
        split=args.split,
    )
    _logger.info(f"Dataset size: {len(dataset)}")
    
    # Create loader with is_training=False to disable augmentation
    loader = create_loader(
        dataset,
        input_size=data_config["input_size"],
        batch_size=args.batch_size,
        is_training=False,  # No augmentation
        no_aug=True,  # Explicitly disable augmentation
        use_prefetcher=True,
        interpolation=data_config["interpolation"],
        mean=data_config["mean"],
        std=data_config["std"],
        num_workers=args.workers,
        crop_pct=data_config["crop_pct"],
        crop_mode=data_config["crop_mode"],
        pin_memory=True,
        device=device,
    )
    
    # Extract stem features and predictions
    _logger.info("Extracting stem features from model...")
    features, predictions, targets = extract_stem_features_and_predictions(
        model, stem_extractor, loader, device, amp=args.amp
    )
    
    # Compute FSC
    _logger.info("Computing Feature Space Centroids from stem features...")
    fsc = compute_fsc(features, predictions, targets, args.num_classes)
    
    # Add metadata
    fsc["model_name"] = args.model
    fsc["checkpoint"] = args.checkpoint if args.checkpoint else "pretrained"
    fsc["data_dir"] = args.data_dir
    fsc["stem_mode"] = args.stem_mode
    fsc["feature_source"] = f"stem_{args.stem_mode}"
    fsc["pooling"] = "AdaptiveAvgPool2d(4, 4)"
    
    # Create output directory and save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Filename reflects stem_mode: e.g. resnet50_FSC_conv1.pth, resnet50_FSC_conv1_bn1.pth
    output_path = output_dir / f"{args.model}_FSC_{args.stem_mode}.pth"
    torch.save(fsc, output_path)
    
    _logger.info(f"FSC saved to: {output_path}")
    _logger.info(f"Centroid shape: {fsc['centroids'].shape}")
    _logger.info(f"Feature dimension: {fsc['feature_dim']} (stem_mode={args.stem_mode})")
    _logger.info(f"Classes with valid centroids: {(fsc['counts'] > 0).sum().item()}/{args.num_classes}")
    
    # Print per-class statistics
    _logger.info("\nPer-class statistics (first 10 classes):")
    for cls in range(min(10, args.num_classes)):
        _logger.info(f"  Class {cls}: {fsc['counts'][cls].item()} correct samples")


if __name__ == "__main__":
    main()

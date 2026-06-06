#!/usr/bin/env python3
"""Validate FSC-based label prediction accuracy using stem features.

This script evaluates how well FSC (Feature Space Centroid) can predict labels
by computing similarity/distance between stem features and FSC centroids.

Since ground truth labels are not available during inference, we select the
FSC centroid that is most similar to the current feature and check if it
matches the actual label.

Supported similarity/distance methods:
- cosine: Cosine similarity (higher is better)
- l2: L2 (Euclidean) distance (lower is better)
- dot: Dot product (higher is better)
- l1: L1 (Manhattan) distance (lower is better)

Usage:
    python validate_fsc_stem.py \
        --fsc-path ./FSC/resnet50_FSC_stem.pth \
        --data-dir /path/to/imagenet/val \
        --model resnet50 \
        --pretrained \
        --similarity cosine

Output:
    Prints accuracy for each similarity method and overall statistics.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from timm.data import create_dataset, create_loader, resolve_data_config
from timm.models import create_model, load_checkpoint
from timm.utils import setup_default_logging, AverageMeter

_logger = logging.getLogger("validate_fsc_stem")

# Available similarity/distance methods
SIMILARITY_METHODS = ['cosine', 'l2', 'dot', 'l1']


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate FSC-based label prediction accuracy using stem features"
    )
    
    # FSC arguments
    parser.add_argument(
        "--fsc-path",
        type=str,
        required=True,
        help="Path to FSC file (e.g., ./FSC/resnet50_FSC_stem.pth)",
    )
    
    # Data arguments
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Path to validation dataset (e.g., ImageNet val folder)",
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
        default="validation",
        help="Dataset split to use (default: validation)",
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
    
    # Similarity method arguments
    parser.add_argument(
        "--similarity",
        type=str,
        default="all",
        choices=SIMILARITY_METHODS + ["all"],
        help="Similarity method to use (default: all - run all methods)",
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
    
    return parser.parse_args()


class StemFeatureExtractor(nn.Module):
    """Extract features from the first conv layer (stem) of a ResNet model.
    
    For ResNet50 with 224x224 input:
    - conv1 (7x7, stride=2, padding=3): 3 -> 64 channels, 112x112 spatial
    - AdaptiveAvgPool2d(4, 4): 64 channels, 4x4 spatial
    - Flatten: 64 * 4 * 4 = 1024-dim vector
    """
    
    def __init__(self, model):
        super().__init__()
        if hasattr(model, 'conv1'):
            self.conv1 = model.conv1
        else:
            raise ValueError("Model does not have conv1 layer")
        
        if hasattr(model, 'bn1'):
            self.bn1 = model.bn1
        else:
            self.bn1 = nn.Identity()
        
        if hasattr(model, 'act1'):
            self.act1 = model.act1
        else:
            self.act1 = nn.ReLU(inplace=True)
        
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.pool(x)
        x = x.flatten(1)
        return x


# =============================================================================
# Similarity/Distance Functions
# =============================================================================

def compute_similarity_cosine(features: torch.Tensor, fsc_centroids: torch.Tensor) -> torch.Tensor:
    """Compute cosine similarity between features and all FSC centroids.
    
    Args:
        features: [batch_size, feature_dim]
        fsc_centroids: [num_classes, feature_dim]
    
    Returns:
        similarity: [batch_size, num_classes] - higher is more similar
    """
    features_norm = F.normalize(features, dim=1)
    fsc_norm = F.normalize(fsc_centroids, dim=1)
    similarity = torch.mm(features_norm, fsc_norm.t())
    return similarity


def compute_similarity_dot(features: torch.Tensor, fsc_centroids: torch.Tensor) -> torch.Tensor:
    """Compute dot product between features and all FSC centroids.
    
    Args:
        features: [batch_size, feature_dim]
        fsc_centroids: [num_classes, feature_dim]
    
    Returns:
        similarity: [batch_size, num_classes] - higher is more similar
    """
    return torch.mm(features, fsc_centroids.t())


def compute_distance_l2(features: torch.Tensor, fsc_centroids: torch.Tensor) -> torch.Tensor:
    """Compute L2 (Euclidean) distance between features and all FSC centroids.
    
    Uses the identity: ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a·b
    
    Args:
        features: [batch_size, feature_dim]
        fsc_centroids: [num_classes, feature_dim]
    
    Returns:
        distance: [batch_size, num_classes] - lower is more similar
    """
    features_sq = (features ** 2).sum(dim=1, keepdim=True)  # [B, 1]
    fsc_sq = (fsc_centroids ** 2).sum(dim=1).unsqueeze(0)   # [1, C]
    dot = torch.mm(features, fsc_centroids.t())              # [B, C]
    dist_sq = features_sq + fsc_sq - 2 * dot                 # [B, C]
    return dist_sq.sqrt()


def compute_distance_l1(features: torch.Tensor, fsc_centroids: torch.Tensor) -> torch.Tensor:
    """Compute L1 (Manhattan) distance between features and all FSC centroids.
    
    Args:
        features: [batch_size, feature_dim]
        fsc_centroids: [num_classes, feature_dim]
    
    Returns:
        distance: [batch_size, num_classes] - lower is more similar
    """
    # features: [B, D] -> [B, 1, D]
    # fsc_centroids: [C, D] -> [1, C, D]
    # diff: [B, C, D]
    diff = features.unsqueeze(1) - fsc_centroids.unsqueeze(0)
    return diff.abs().sum(dim=2)  # [B, C]


def predict_labels(
    features: torch.Tensor,
    fsc_centroids: torch.Tensor,
    method: str,
) -> torch.Tensor:
    """Predict labels based on similarity/distance to FSC centroids.
    
    Args:
        features: [batch_size, feature_dim]
        fsc_centroids: [num_classes, feature_dim]
        method: One of 'cosine', 'l2', 'dot', 'l1'
    
    Returns:
        predictions: [batch_size] - predicted class indices
    """
    if method == 'cosine':
        scores = compute_similarity_cosine(features, fsc_centroids)
        return scores.argmax(dim=1)
    
    elif method == 'dot':
        scores = compute_similarity_dot(features, fsc_centroids)
        return scores.argmax(dim=1)
    
    elif method == 'l2':
        distances = compute_distance_l2(features, fsc_centroids)
        return distances.argmin(dim=1)
    
    elif method == 'l1':
        distances = compute_distance_l1(features, fsc_centroids)
        return distances.argmin(dim=1)
    
    else:
        raise ValueError(f"Unknown method: {method}. Choose from {SIMILARITY_METHODS}")


def validate_fsc(
    stem_extractor: nn.Module,
    fsc_centroids: torch.Tensor,
    loader,
    device: torch.device,
    methods: List[str],
    amp: bool = False,
) -> Dict[str, Dict[str, float]]:
    """Validate FSC-based label prediction.
    
    Args:
        stem_extractor: StemFeatureExtractor module
        fsc_centroids: [num_classes, feature_dim] FSC centroids
        loader: DataLoader for validation set
        device: Device to use
        methods: List of similarity methods to evaluate
        amp: Whether to use automatic mixed precision
    
    Returns:
        results: Dictionary with accuracy metrics for each method
    """
    stem_extractor.eval()
    fsc_centroids = fsc_centroids.to(device)
    
    # Initialize meters for each method
    meters = {method: {'top1': AverageMeter(), 'top5': AverageMeter()} for method in methods}
    total_samples = 0
    
    with torch.no_grad():
        for images, targets in tqdm(loader, desc="Validating FSC"):
            images = images.to(device)
            targets = targets.to(device)
            batch_size = images.size(0)
            total_samples += batch_size
            
            # Extract stem features
            with torch.cuda.amp.autocast(enabled=amp):
                stem_features = stem_extractor(images).float()
            
            # Evaluate each method
            for method in methods:
                if method in ['cosine', 'dot']:
                    # Higher score = more similar
                    if method == 'cosine':
                        scores = compute_similarity_cosine(stem_features, fsc_centroids)
                    else:
                        scores = compute_similarity_dot(stem_features, fsc_centroids)
                    
                    # Top-1 accuracy
                    pred = scores.argmax(dim=1)
                    correct_top1 = (pred == targets).float().sum().item()
                    
                    # Top-5 accuracy
                    _, top5_pred = scores.topk(5, dim=1)
                    correct_top5 = (top5_pred == targets.unsqueeze(1)).any(dim=1).float().sum().item()
                
                else:
                    # Lower distance = more similar
                    if method == 'l2':
                        distances = compute_distance_l2(stem_features, fsc_centroids)
                    else:
                        distances = compute_distance_l1(stem_features, fsc_centroids)
                    
                    # Top-1 accuracy (argmin for distance)
                    pred = distances.argmin(dim=1)
                    correct_top1 = (pred == targets).float().sum().item()
                    
                    # Top-5 accuracy (smallest 5 distances)
                    _, top5_pred = distances.topk(5, dim=1, largest=False)
                    correct_top5 = (top5_pred == targets.unsqueeze(1)).any(dim=1).float().sum().item()
                
                meters[method]['top1'].update(correct_top1 / batch_size * 100, batch_size)
                meters[method]['top5'].update(correct_top5 / batch_size * 100, batch_size)
    
    # Compile results
    results = {}
    for method in methods:
        results[method] = {
            'top1': meters[method]['top1'].avg,
            'top5': meters[method]['top5'].avg,
        }
    
    return results, total_samples


def main():
    setup_default_logging()
    args = parse_args()
    
    device = torch.device(args.device)
    _logger.info(f"Using device: {device}")
    
    # Load FSC
    _logger.info(f"Loading FSC from: {args.fsc_path}")
    fsc_data = torch.load(args.fsc_path, map_location='cpu')
    fsc_centroids = fsc_data['centroids']
    feature_dim = fsc_data['feature_dim']
    _logger.info(f"FSC loaded: {fsc_centroids.shape}, feature_dim={feature_dim}")
    _logger.info(f"FSC metadata - model: {fsc_data.get('model_name', 'unknown')}, "
                 f"source: {fsc_data.get('feature_source', 'unknown')}")
    
    # Create model (only need stem)
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
    
    # Create stem feature extractor
    stem_extractor = StemFeatureExtractor(model)
    stem_extractor = stem_extractor.to(device)
    stem_extractor.eval()
    
    # Verify stem output dimensions
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        dummy_output = stem_extractor(dummy_input)
        _logger.info(f"Stem feature dimension: {dummy_output.shape[1]}")
        
        if dummy_output.shape[1] != feature_dim:
            _logger.warning(
                f"Stem feature dim ({dummy_output.shape[1]}) != FSC feature dim ({feature_dim})"
            )
    
    # Resolve data config
    data_config = resolve_data_config(vars(args), model=model, verbose=True)
    
    # Create dataset and loader
    _logger.info(f"Loading validation dataset from: {args.data_dir}")
    dataset = create_dataset(
        root=args.data_dir,
        name=args.dataset,
        split=args.split,
    )
    _logger.info(f"Dataset size: {len(dataset)}")
    
    loader = create_loader(
        dataset,
        input_size=data_config["input_size"],
        batch_size=args.batch_size,
        is_training=False,
        no_aug=True,
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
    
    # Determine which methods to evaluate
    if args.similarity == "all":
        methods = SIMILARITY_METHODS
    else:
        methods = [args.similarity]
    
    _logger.info(f"Evaluating similarity methods: {methods}")
    
    # Validate
    results, total_samples = validate_fsc(
        stem_extractor=stem_extractor,
        fsc_centroids=fsc_centroids,
        loader=loader,
        device=device,
        methods=methods,
        amp=args.amp,
    )
    
    # Print results
    _logger.info("\n" + "=" * 60)
    _logger.info("FSC-based Label Prediction Results")
    _logger.info("=" * 60)
    _logger.info(f"Total validation samples: {total_samples}")
    _logger.info(f"FSC path: {args.fsc_path}")
    _logger.info("-" * 60)
    _logger.info(f"{'Method':<12} {'Top-1 Acc (%)':<15} {'Top-5 Acc (%)':<15}")
    _logger.info("-" * 60)
    
    for method in methods:
        _logger.info(
            f"{method:<12} {results[method]['top1']:<15.2f} {results[method]['top5']:<15.2f}"
        )
    
    _logger.info("=" * 60)
    
    # Find best method
    best_method = max(methods, key=lambda m: results[m]['top1'])
    _logger.info(f"Best method: {best_method} (Top-1: {results[best_method]['top1']:.2f}%)")


if __name__ == "__main__":
    main()

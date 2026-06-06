#!/usr/bin/env python3
"""Compute Feature Space Centroid (FSC) from ViT patch embedding (stem) for each class.

This script computes the mean feature vector (centroid) for each class using only
correctly classified samples from the training dataset. The ViT patch embedding output
(196 patches, 768-dim each) is compressed by taking mean of 128-unit chunks to create
1176-dim features (196 patches * 6 mean values).

For ViT-B/16:
- Patch embedding: 16x16 conv, stride 16 -> 196 patches (14x14), 768-dim each
- Compression: 768-dim -> 6-dim (mean of 128-unit chunks)
- Flatten: 196 * 6 = 1176-dim vector

Usage:
    python compute_fsc_vit_stem.py --data-dir /path/to/imagenet/train --model vit_base_patch16_224 --pretrained

Output:
    ./FSC/{model_name}_FSC_vit_stem.pth
"""

import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from timm.data import create_dataset, create_loader, resolve_data_config
from timm.models import create_model, load_checkpoint
from timm.utils import setup_default_logging

_logger = logging.getLogger("compute_fsc_vit_stem")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute Feature Space Centroid (FSC) from ViT patch embedding"
    )
    
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
        default="vit_base_patch16_224",
        help="Model architecture (default: vit_base_patch16_224)",
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
    
    # ViT-specific arguments
    parser.add_argument(
        "--after-block0",
        action="store_true",
        help="Extract features after Block 0 (with LayerNorm). Default: before Block 0",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=128,
        help="Chunk size for mean compression (default: 128, 768/128=6 means per patch)",
    )
    
    # Processing arguments
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for inference (default: 128, smaller for ViT)",
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
    
    # Output arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./FSC",
        help="Output directory for FSC files (default: ./FSC)",
    )
    
    return parser.parse_args()


class ViTStemFeatureExtractor(nn.Module):
    """Extract features from ViT patch embedding (stem equivalent).
    
    For ViT-B/16 with 224x224 input:
    - patch_embed (16x16, stride=16): 3 -> 768 channels, 14x14 patches = 196 patches
    - Each patch: 768-dim vector
    - Compression: 768-dim -> (768/chunk_size)-dim by taking mean of chunks
    - Output: [B, 196 * (768/chunk_size)] = [B, 1176] for chunk_size=128
    
    Options:
    - before_block0: Raw patch embedding output (no normalization)
    - after_block0: After first transformer block's LayerNorm (normalized)
    """
    
    def __init__(self, model, after_block0=False, chunk_size=128):
        super().__init__()
        self.patch_embed = model.patch_embed
        self.after_block0 = after_block0
        self.chunk_size = chunk_size
        
        # For after_block0 option
        if after_block0:
            self.cls_token = model.cls_token
            self.pos_embed = model.pos_embed
            self.pos_drop = model.pos_drop
            self.block0 = model.blocks[0]
        
        # Calculate compression parameters
        self.embed_dim = model.embed_dim  # 768 for ViT-B
        self.num_chunks = self.embed_dim // chunk_size  # 768 // 128 = 6
        
        if self.embed_dim % chunk_size != 0:
            raise ValueError(
                f"embed_dim ({self.embed_dim}) must be divisible by chunk_size ({chunk_size})"
            )
        
    def forward(self, x):
        B = x.shape[0]
        
        # Patch embedding: [B, 3, 224, 224] -> [B, 196, 768]
        x = self.patch_embed(x)
        num_patches = x.shape[1]  # 196
        
        if self.after_block0:
            # Add CLS token and positional embedding
            cls_token = self.cls_token.expand(B, -1, -1)  # [B, 1, 768]
            x = torch.cat((cls_token, x), dim=1)  # [B, 197, 768]
            x = x + self.pos_embed  # [B, 197, 768]
            x = self.pos_drop(x)
            
            # Pass through Block 0
            x = self.block0(x)  # [B, 197, 768]
            
            # Remove CLS token, keep only patch tokens
            x = x[:, 1:, :]  # [B, 196, 768]
        
        # x shape: [B, 196, 768]
        # Compress each 768-dim vector to (768/chunk_size)-dim by taking mean of chunks
        # Reshape: [B, 196, 768] -> [B, 196, 6, 128]
        x = x.view(B, num_patches, self.num_chunks, self.chunk_size)
        
        # Take mean of each chunk: [B, 196, 6, 128] -> [B, 196, 6]
        x = x.mean(dim=-1)
        
        # Flatten: [B, 196, 6] -> [B, 1176]
        x = x.flatten(1)
        
        return x


def extract_vit_stem_features_and_predictions(
    model, stem_extractor, loader, device, amp=False
):
    """Extract ViT stem features and predictions from model.
    
    Args:
        model: The pretrained ViT model (for classification).
        stem_extractor: ViTStemFeatureExtractor for extracting stem features.
        loader: DataLoader for the dataset.
        device: Device to use.
        amp: Whether to use automatic mixed precision.
    
    Returns:
        all_features: Tensor [N, feature_dim].
        all_predictions: Tensor [N].
        all_targets: Tensor [N].
    """
    model.eval()
    stem_extractor.eval()
    
    all_features = []
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(
            tqdm(loader, desc="Extracting ViT stem features")
        ):
            images = images.to(device)
            targets = targets.to(device)
            
            with torch.cuda.amp.autocast(enabled=amp):
                # Extract stem features (1176-dim)
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
    _logger.info(
        f"Correctly classified: {len(correct_indices)} "
        f"({100 * len(correct_indices) / len(targets):.2f}%)"
    )
    
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


def compute_cosine_similarity_matrix(centroids, counts):
    """Compute cosine similarity matrix between all class centroids.
    
    Args:
        centroids: Tensor [num_classes, feature_dim] - centroid for each class
        counts: Tensor [num_classes] - number of samples per class (for filtering)
    
    Returns:
        similarity_matrix: Tensor [num_classes, num_classes] - pairwise cosine similarities
        stats: Dictionary with similarity statistics
    """
    num_classes = centroids.shape[0]
    
    # Normalize centroids for cosine similarity
    # Only normalize classes with valid centroids (count > 0)
    valid_mask = counts > 0
    
    # Compute L2 norm
    norms = centroids.norm(dim=1, keepdim=True)
    
    # Avoid division by zero
    norms = norms.clamp(min=1e-8)
    normalized_centroids = centroids / norms
    
    # Compute cosine similarity matrix
    similarity_matrix = torch.mm(normalized_centroids, normalized_centroids.t())
    
    # Set diagonal to 1 (self-similarity)
    similarity_matrix.fill_diagonal_(1.0)
    
    # Set invalid class similarities to 0
    invalid_mask = ~valid_mask
    similarity_matrix[invalid_mask, :] = 0.0
    similarity_matrix[:, invalid_mask] = 0.0
    
    # Compute statistics (only for valid classes, excluding diagonal)
    valid_indices = valid_mask.nonzero(as_tuple=True)[0]
    num_valid = len(valid_indices)
    
    if num_valid > 1:
        # Extract upper triangular part (excluding diagonal) for valid classes
        valid_similarities = []
        for i in range(num_valid):
            for j in range(i + 1, num_valid):
                idx_i = valid_indices[i]
                idx_j = valid_indices[j]
                valid_similarities.append(similarity_matrix[idx_i, idx_j].item())
        
        valid_similarities = torch.tensor(valid_similarities)
        
        stats = {
            "mean": valid_similarities.mean().item(),
            "std": valid_similarities.std().item(),
            "min": valid_similarities.min().item(),
            "max": valid_similarities.max().item(),
            "median": valid_similarities.median().item(),
            "num_pairs": len(valid_similarities),
        }
    else:
        stats = {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "median": 0.0,
            "num_pairs": 0,
        }
    
    return similarity_matrix, stats


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
    
    # Verify it's a ViT model
    if not hasattr(model, 'patch_embed'):
        raise ValueError(f"Model {args.model} does not have patch_embed. Is it a ViT model?")
    
    # Create ViT stem feature extractor
    feature_location = "after Block 0" if args.after_block0 else "before Block 0"
    _logger.info(f"Creating ViT stem feature extractor ({feature_location})...")
    _logger.info(f"Chunk size: {args.chunk_size} (768 -> {768 // args.chunk_size} means per patch)")
    
    stem_extractor = ViTStemFeatureExtractor(
        model,
        after_block0=args.after_block0,
        chunk_size=args.chunk_size,
    )
    stem_extractor = stem_extractor.to(device)
    stem_extractor.eval()
    
    # Verify stem output dimensions
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        dummy_output = stem_extractor(dummy_input)
        expected_dim = 196 * (768 // args.chunk_size)  # 196 * 6 = 1176
        _logger.info(f"ViT stem feature dimension: {dummy_output.shape[1]} (expected: {expected_dim})")
    
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
    
    # Extract ViT stem features and predictions
    _logger.info("Extracting ViT stem features from model...")
    features, predictions, targets = extract_vit_stem_features_and_predictions(
        model, stem_extractor, loader, device, amp=args.amp
    )
    
    # Compute FSC
    _logger.info("Computing Feature Space Centroids from ViT stem features...")
    fsc = compute_fsc(features, predictions, targets, args.num_classes)
    
    # Compute cosine similarity between class FSCs
    _logger.info("Computing cosine similarity between class FSCs...")
    similarity_matrix, similarity_stats = compute_cosine_similarity_matrix(
        fsc["centroids"], fsc["counts"]
    )
    
    # Add similarity results to FSC
    fsc["similarity_matrix"] = similarity_matrix
    fsc["similarity_stats"] = similarity_stats
    
    # Log similarity statistics
    _logger.info("\n=== FSC Cosine Similarity Statistics ===")
    _logger.info(f"  Number of valid class pairs: {similarity_stats['num_pairs']}")
    _logger.info(f"  Mean similarity: {similarity_stats['mean']:.4f}")
    _logger.info(f"  Std similarity:  {similarity_stats['std']:.4f}")
    _logger.info(f"  Min similarity:  {similarity_stats['min']:.4f}")
    _logger.info(f"  Max similarity:  {similarity_stats['max']:.4f}")
    _logger.info(f"  Median similarity: {similarity_stats['median']:.4f}")
    
    # Find most similar and most dissimilar class pairs
    valid_mask = fsc["counts"] > 0
    valid_indices = valid_mask.nonzero(as_tuple=True)[0]
    num_valid = len(valid_indices)
    
    if num_valid > 1:
        # Find top 5 most similar pairs (excluding self)
        sim_matrix_copy = similarity_matrix.clone()
        sim_matrix_copy.fill_diagonal_(-1)  # Exclude diagonal
        sim_matrix_copy[~valid_mask, :] = -1
        sim_matrix_copy[:, ~valid_mask] = -1
        
        # Get upper triangular indices only
        triu_indices = torch.triu_indices(args.num_classes, args.num_classes, offset=1)
        triu_values = sim_matrix_copy[triu_indices[0], triu_indices[1]]
        
        # Top 5 most similar
        top_k = min(5, (triu_values > -1).sum().item())
        if top_k > 0:
            top_values, top_indices = triu_values.topk(top_k)
            _logger.info(f"\nTop {top_k} most similar class pairs:")
            for i in range(top_k):
                idx = top_indices[i]
                cls_i = triu_indices[0][idx].item()
                cls_j = triu_indices[1][idx].item()
                sim = top_values[i].item()
                _logger.info(f"  Class {cls_i} <-> Class {cls_j}: {sim:.4f}")
            
            # Top 5 least similar
            valid_triu = triu_values[triu_values > -1]
            if len(valid_triu) >= top_k:
                bottom_values, bottom_indices_local = valid_triu.topk(top_k, largest=False)
                valid_indices_list = (triu_values > -1).nonzero(as_tuple=True)[0]
                bottom_indices = valid_indices_list[bottom_indices_local]
                
                _logger.info(f"\nTop {top_k} least similar class pairs:")
                for i in range(top_k):
                    idx = bottom_indices[i]
                    cls_i = triu_indices[0][idx].item()
                    cls_j = triu_indices[1][idx].item()
                    sim = bottom_values[i].item()
                    _logger.info(f"  Class {cls_i} <-> Class {cls_j}: {sim:.4f}")
    
    # Add metadata
    fsc["model_name"] = args.model
    fsc["checkpoint"] = args.checkpoint if args.checkpoint else "pretrained"
    fsc["data_dir"] = args.data_dir
    fsc["feature_source"] = "vit_patch_embed"
    fsc["feature_location"] = "after_block0" if args.after_block0 else "before_block0"
    fsc["chunk_size"] = args.chunk_size
    fsc["num_patches"] = 196
    fsc["embed_dim"] = 768
    fsc["compression"] = f"mean of {args.chunk_size}-unit chunks"
    
    # Create output directory and save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Include feature location in filename
    location_suffix = "after_block0" if args.after_block0 else "before_block0"
    output_path = output_dir / f"{args.model}_FSC_vit_stem_{location_suffix}.pth"
    torch.save(fsc, output_path)
    
    _logger.info(f"\nFSC saved to: {output_path}")
    _logger.info(f"Centroid shape: {fsc['centroids'].shape}")
    _logger.info(
        f"Feature dimension: {fsc['feature_dim']} "
        f"(196 patches × {768 // args.chunk_size} means = {196 * (768 // args.chunk_size)})"
    )
    _logger.info(
        f"Classes with valid centroids: {(fsc['counts'] > 0).sum().item()}/{args.num_classes}"
    )
    
    # Print per-class statistics
    _logger.info("\nPer-class statistics (first 10 classes):")
    for cls in range(min(10, args.num_classes)):
        _logger.info(f"  Class {cls}: {fsc['counts'][cls].item()} correct samples")


if __name__ == "__main__":
    main()

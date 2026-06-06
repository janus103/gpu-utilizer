#!/usr/bin/env python3
"""Measure aug_classifier diagnostics (p_clean, znorm, dist) on clean ImageNet val."""

import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from timm.data import create_transform, resolve_data_config, get_augmix_sl_num_transforms, AUGMIX_SL_V2_NUM_TRANSFORMS
from timm.models import create_model
from timm.utils import AverageMeter, setup_default_logging
from torchvision.datasets import ImageFolder

_logger = logging.getLogger("measure_clean")


class DirectAugClassifier(nn.Module):
    def __init__(self, stem_channels, output_channels=64, num_dw_stages=2,
                 num_transforms=AUGMIX_SL_V2_NUM_TRANSFORMS, hidden_dims=None,
                 dropout=0.1, dw_init_mode='fan_in'):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]
        self.num_classes = num_transforms + 1
        self.feature_dim = output_channels * 4 * 4
        self.inst_norm = nn.InstanceNorm2d(stem_channels, affine=True)
        dw_layers = []
        for _ in range(num_dw_stages):
            dw_layers.extend([
                nn.Conv2d(stem_channels, stem_channels, 3, stride=2, padding=1, groups=stem_channels),
                nn.ReLU(inplace=True),
            ])
        self.dw_stages = nn.Sequential(*dw_layers)
        self.clean_ref = nn.Parameter(torch.zeros(1, stem_channels, 4, 4))
        self.pw_conv = nn.Conv2d(stem_channels, output_channels, 1)
        self.log_r = nn.Parameter(torch.full((num_transforms,), 3.0))
        encoder_input_dim = self.feature_dim + 1
        layers = []
        in_dim = encoder_input_dim
        for h_dim in hidden_dims:
            layers.extend([nn.Linear(in_dim, h_dim), nn.BatchNorm1d(h_dim),
                           nn.ReLU(inplace=True), nn.Dropout(dropout)])
            in_dim = h_dim
        self.shared_encoder = nn.Sequential(*layers)
        self.aug_head = nn.Linear(hidden_dims[-1], self.num_classes)
        self.dist_head = nn.Sequential(nn.Linear(hidden_dims[-1], 1), nn.Softplus())

    def encode(self, features_spatial):
        normed = self.inst_norm(features_spatial)
        reduced = self.dw_stages(normed)
        diff = reduced - self.clean_ref
        z = self.pw_conv(diff)
        return z.flatten(1)

    def forward(self, z_flat):
        dist = z_flat.norm(dim=1, keepdim=True)
        x = torch.cat([z_flat, dist], dim=1)
        shared = self.shared_encoder(x)
        return self.aug_head(shared), self.dist_head(shared).squeeze(-1)


def main():
    setup_default_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagenet-val-dir", required=True, type=str)
    parser.add_argument("--aug-ckpt", required=True, type=str)
    parser.add_argument("--model", default="resnet50", type=str)
    parser.add_argument("--initial-checkpoint", default="", type=str)
    parser.add_argument("--stem-channels", default=64, type=int)
    parser.add_argument("--num-dw-stages", default=5, type=int)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("-j", "--workers", default=4, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--max-batches", default=50, type=int,
                        help="Max batches to evaluate (0=all)")
    args = parser.parse_args()

    device = torch.device(args.device)

    model_kwargs = dict(pretrained=True, num_classes=1000)
    if args.initial_checkpoint:
        model_kwargs['pretrained'] = False
        model_kwargs['checkpoint_path'] = args.initial_checkpoint
    model = create_model(args.model, **model_kwargs).to(device).eval()

    aug_ckpt = torch.load(args.aug_ckpt, map_location="cpu")
    train_args = aug_ckpt.get("args", {})
    hidden_dims = train_args.get("hidden_dims", [512, 256, 128])
    dropout = train_args.get("dropout", 0.1)
    dw_init_mode = train_args.get("dw_init_mode", "fan_in")
    num_transforms = get_augmix_sl_num_transforms(version=2)

    aug_classifier = DirectAugClassifier(
        stem_channels=args.stem_channels, output_channels=64,
        num_dw_stages=args.num_dw_stages, num_transforms=num_transforms,
        hidden_dims=hidden_dims, dropout=dropout, dw_init_mode=dw_init_mode,
    )
    aug_classifier.load_state_dict(aug_ckpt["aug_classifier"])
    aug_classifier.to(device).eval()

    data_config = resolve_data_config(vars(args), model=model)
    transform = create_transform(**data_config, is_training=False)
    dataset = ImageFolder(args.imagenet_val_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True)

    # Determine stem forward
    is_vit = hasattr(model, 'patch_embed') and hasattr(model.patch_embed, 'proj')
    if is_vit:
        stem_fn = lambda imgs: model.patch_embed.proj(imgs)
        _logger.info("Stem: patch_embed.proj (ViT)")
    else:
        stem_fn = lambda imgs: model.conv1(imgs)
        _logger.info("Stem: conv1 (ResNet)")

    p_clean_m = AverageMeter()
    znorm_m = AverageMeter()
    dist_m = AverageMeter()
    entropy_m = AverageMeter()
    top1_m = AverageMeter()

    _logger.info("Measuring clean validation diagnostics...")
    with torch.no_grad():
        for batch_idx, (images, target) in enumerate(loader):
            if args.max_batches > 0 and batch_idx >= args.max_batches:
                break
            images = images.to(device)
            target = target.to(device)

            features = stem_fn(images)
            z = aug_classifier.encode(features)
            aug_out, dist_out = aug_classifier(z)
            probs = F.softmax(aug_out, dim=1)

            bs = images.size(0)
            p_clean_m.update(probs[:, 0].mean().item(), bs)
            znorm_m.update(z.norm(dim=1).mean().item(), bs)
            dist_m.update(dist_out.mean().item(), bs)
            ent = -(probs * torch.log(probs + 1e-10)).sum(1).mean().item()
            entropy_m.update(ent, bs)

            output = model(images)
            _, pred = output.topk(1, 1, True, True)
            top1_m.update(pred.eq(target.view(-1, 1)).float().mean().item() * 100, bs)

            if batch_idx % 10 == 0:
                _logger.info(
                    "  Batch %d: top1=%.1f%% | p_clean=%.4f, znorm=%.2f, dist=%.2f, entropy=%.4f",
                    batch_idx, top1_m.val, p_clean_m.val, znorm_m.val, dist_m.val, entropy_m.val,
                )

    print("\n" + "=" * 60)
    print(f"Clean Validation Baseline ({args.model})")
    print(f"  Samples: {p_clean_m.count}")
    print(f"  Top-1 Accuracy: {top1_m.avg:.2f}%")
    print(f"  p_clean:  {p_clean_m.avg:.4f}")
    print(f"  znorm:    {znorm_m.avg:.2f}")
    print(f"  dist:     {dist_m.avg:.2f}")
    print(f"  entropy:  {entropy_m.avg:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()

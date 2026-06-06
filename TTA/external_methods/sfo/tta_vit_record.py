#!/usr/bin/env python3
"""TTA Recording (ViT): Run TTA for classes 0-999 and save per-epoch metrics to TSV.

ViT version of tta_record.py — uses patch_embed.proj as trainable stem (instead of conv1).
Records metrics (loss, dist, entropy, entropy_var, backbone_acc) for:
- Before TTA (epoch=-1)
- After each TTA epoch (epoch=0, 1, ..., epochs-1)

Usage:
    python tta_vit_record.py \\
        --data-dir /home/oem/jin/datasets/imagenet-c \\
        --val-split pixelate/5 \\
        --backbone-checkpoint ./ZOA_WEIGHT/ZOA_vit_base_timm_format.pth \\
        --aug-classifier-checkpoint ./ZOA_FSC/phase1_vit_base_direct_warmrestart/best.pth.tar \\
        --target-aug-logits-file ./output/eval_analyze_per_class.tsv \\
        --tta-norm l2 --epochs 20 \\
        --output-tsv ./output/tta_vit_record.tsv

    # Or use direct 8 values (same target for all classes):
    python tta_vit_record.py ... --target-aug-logits "5.46,-3.67,3.05,3.14,-9.32,-11.48,-1.54,-11.04"
"""
import argparse
import csv
import logging
import os
import re

import torch
import torch.nn.functional as F

from timm import utils
from timm.data import get_augmix_sl_num_transforms

from train_phase1_vit import (
    StemFeatureExtractor,
    DirectAugClassifier,
    _SimpleTransformDataset,
)
from tta import train_tta_one_epoch

_logger = logging.getLogger('tta_vit_record')


def load_target_aug_logits_for_class(file_path: str, target_class: int) -> torch.Tensor:
    """Load target aug_logits for target_class from TSV."""
    with open(file_path) as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            if int(row['class_id']) == target_class:
                logits = [
                    float(row[f'aug_logit_{i}'])
                    for i in range(8)
                ]
                return torch.tensor(logits, dtype=torch.float32)
    raise ValueError(
        f'Target class {target_class} not found in {file_path}'
    )


def parse_target_aug_logits(logits_str: str) -> torch.Tensor:
    """Parse comma-separated 8 values to tensor."""
    parts = [float(x.strip()) for x in logits_str.split(',')]
    if len(parts) != 8:
        raise ValueError(
            f'--target-aug-logits must have 8 values, got {len(parts)}'
        )
    return torch.tensor(parts, dtype=torch.float32)


def get_target_aug_logits(args, target_class: int) -> torch.Tensor:
    """Get target aug_logits from file (per class) or direct string (same for all)."""
    if args.target_aug_logits_file:
        return load_target_aug_logits_for_class(
            args.target_aug_logits_file, target_class
        )
    if args.target_aug_logits:
        return parse_target_aug_logits(args.target_aug_logits)
    raise ValueError(
        'Must provide --target-aug-logits-file or --target-aug-logits'
    )


def eval_tta_metrics(
    backbone,
    stem_extractor,
    aug_classifier,
    loader,
    norm_type: str,
    target_aug_logits: torch.Tensor,
    device: torch.device,
) -> dict:
    """Compute TTA metrics (loss, dist, entropy, backbone_acc) without training."""
    stem_extractor.eval()
    aug_classifier.eval()
    backbone.eval()

    target = target_aug_logits.to(device).unsqueeze(0)

    loss_m = utils.AverageMeter()
    dist_m = utils.AverageMeter()
    entropy_m = utils.AverageMeter()
    entropy_var_m = utils.AverageMeter()
    backbone_acc_m = utils.AverageMeter()

    with torch.no_grad():
        for imgs, labels in loader:
            B = imgs.size(0)
            imgs = imgs.to(device)
            labels = labels.to(device)

            f = stem_extractor(imgs)
            z = aug_classifier.encode(f)
            pred_aug_logits, _ = aug_classifier(z)

            if norm_type == 'l2':
                loss = F.mse_loss(pred_aug_logits, target)
                dist = (pred_aug_logits - target).norm(dim=1).mean().item()
            else:
                loss = F.l1_loss(pred_aug_logits, target)
                dist = (pred_aug_logits - target).abs().sum(dim=1).mean().item()

            probs = F.softmax(pred_aug_logits, dim=1)
            entropy_per_sample = -(probs * (probs + 1e-10).log()).sum(dim=1)
            entropy_mean = entropy_per_sample.mean().item()
            entropy_var = entropy_per_sample.var(unbiased=(B > 1)).item()

            backbone_out = backbone(imgs)
            backbone_pred = backbone_out.argmax(dim=1)
            backbone_acc = (backbone_pred == labels).float().mean().item() * 100.0

            loss_m.update(loss.item(), B)
            dist_m.update(dist, B)
            entropy_m.update(entropy_mean, B)
            entropy_var_m.update(entropy_var, B)
            backbone_acc_m.update(backbone_acc, B)

    return {
        'loss': loss_m.avg,
        'dist': dist_m.avg,
        'entropy': entropy_m.avg,
        'entropy_var': entropy_var_m.avg,
        'backbone_acc': backbone_acc_m.avg,
    }


def _parse_args():
    parser = argparse.ArgumentParser(
        description='TTA Record (ViT): Run TTA for classes 0-999, save per-epoch metrics to TSV'
    )
    parser.add_argument('--data-dir', type=str, required=True)
    parser.add_argument('--val-split', default='val', type=str)
    parser.add_argument(
        '--backbone-checkpoint', type=str, required=True,
        help='Path to backbone checkpoint.',
    )
    parser.add_argument(
        '--aug-classifier-checkpoint', type=str, required=True,
        help='Path to phase-1 aug classifier checkpoint (from train_phase1_vit.py).',
    )
    parser.add_argument(
        '--target-aug-logits-file', type=str, default='',
        help='Path to eval_analyze TSV (aug_logit_0..7 per class). '
             'Mutually exclusive with --target-aug-logits.',
    )
    parser.add_argument(
        '--target-aug-logits', type=str, default='',
        help='Comma-separated 8 values, e.g. "5.4583,-3.6745,3.0471,...". '
             'Same target for all classes. Mutually exclusive with --target-aug-logits-file.',
    )
    parser.add_argument(
        '--tta-norm', type=str, default='l2', choices=['l1', 'l2'],
    )
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument(
        '--output-tsv', type=str, default='',
        help='Output TSV path. Default: output_tta_vit_record_mean/tta_vit_record_{val_safe}_{norm}_{epochs}.tsv',
    )
    parser.add_argument('--model', default='vit_base_patch16_224', type=str)
    parser.add_argument('--num-classes', type=int, default=1000)
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--hidden-dims', type=int, nargs='+', default=[512, 256, 128])
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--dw-init-mode', type=str, default='fan_in')
    parser.add_argument('--opt', default='adamw', type=str)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--clip-grad', type=float, default=1.0)
    parser.add_argument('--sched', type=str, default='cosine')
    parser.add_argument('--warmup-epochs', type=int, default=0)
    parser.add_argument('--min-lr', type=float, default=1e-6)
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('-j', '--workers', type=int, default=None)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--pin-mem', action='store_true')
    parser.add_argument('--log-interval', type=int, default=10)
    parser.add_argument(
        '--start-class', type=int, default=0,
        help='Start target class (default: 0).',
    )
    parser.add_argument(
        '--end-class', type=int, default=1000,
        help='End target class exclusive (default: 1000).',
    )
    return parser.parse_args()


def main():
    utils.setup_default_logging()
    args = _parse_args()

    if args.workers is None:
        args.workers = min(4, max(1, os.cpu_count() // 4))

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    device = torch.device(args.device)
    utils.random_seed(args.seed, 0)

    if bool(args.target_aug_logits_file) == bool(args.target_aug_logits):
        raise ValueError(
            'Provide exactly one of --target-aug-logits-file or --target-aug-logits'
        )

    val_safe = re.sub(r'[^a-zA-Z0-9_]', '_', args.val_split)
    if not args.output_tsv:
        args.output_tsv = (
            f'./output_tta_vit_record_mean/tta_vit_record_{val_safe}_{args.tta_norm}_ep{args.epochs}.tsv'
        )
    os.makedirs(os.path.dirname(args.output_tsv) or '.', exist_ok=True)

    from timm.data import resolve_data_config
    from timm.models import create_model
    from torchvision import transforms
    from torchvision.datasets import ImageFolder
    from torch.utils.data import Subset
    from timm.optim import create_optimizer_v2
    from timm.scheduler import create_scheduler_v2

    # -------------------------------------------------------------------------
    # Load backbone and aug classifier once (ViT: patch_embed.proj trainable)
    # -------------------------------------------------------------------------
    _logger.info(f'Creating backbone: {args.model}')
    backbone = create_model(
        args.model,
        pretrained=False,
        num_classes=args.num_classes,
        checkpoint_path=args.backbone_checkpoint,
    ).to(device)

    # ViT: patch_embed.proj is the stem (analogous to ResNet conv1)
    for name, p in backbone.named_parameters():
        if name in ('patch_embed.proj.weight', 'patch_embed.proj.bias'):
            p.requires_grad = True
        else:
            p.requires_grad = False

    stem_extractor = StemFeatureExtractor(backbone).to(device)
    for p in stem_extractor.parameters():
        p.requires_grad = True

    with torch.no_grad():
        dummy = torch.randn(1, 3, args.img_size, args.img_size, device=device)
        dummy_out = stem_extractor(dummy)
        stem_channels = dummy_out.shape[1]
        stem_h = dummy_out.shape[2]

    num_dw_stages = 0
    h = stem_h
    while h > 4:
        h = (h - 3 + 2) // 2 + 1
        num_dw_stages += 1

    num_transforms = get_augmix_sl_num_transforms(version=2)
    aug_classifier = DirectAugClassifier(
        stem_channels=stem_channels,
        output_channels=64,
        num_dw_stages=num_dw_stages,
        num_transforms=num_transforms,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
        dw_init_mode=args.dw_init_mode,
    ).to(device)

    _logger.info(f'Loading aug classifier from {args.aug_classifier_checkpoint}')
    aug_ckpt = torch.load(args.aug_classifier_checkpoint, map_location=device)
    aug_classifier.load_state_dict(aug_ckpt['aug_classifier'], strict=False)
    aug_classifier.eval()
    for p in aug_classifier.parameters():
        p.requires_grad = False

    # Save initial patch_embed.proj state for reset between classes (ViT stem)
    patch_embed_proj_state = {
        k: v.cpu().clone()
        for k, v in backbone.patch_embed.proj.state_dict().items()
    }

    data_config = resolve_data_config(vars(args), model=backbone, verbose=False)
    base_transform = transforms.Compose([
        transforms.Resize(int(args.img_size / data_config['crop_pct'])),
        transforms.CenterCrop(args.img_size),
    ])
    final_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=data_config['mean'], std=data_config['std']),
    ])

    val_dir = os.path.join(args.data_dir, args.val_split)
    raw_dataset = ImageFolder(val_dir)

    # Open TSV for writing
    fieldnames = [
        'target_class', 'epoch', 'loss', 'dist', 'entropy', 'entropy_var',
        'backbone_acc',
    ]
    write_header = True

    for target_class in range(args.start_class, args.end_class):
        _logger.info(f'=== Target class {target_class} ===')

        # Reset stem (patch_embed.proj) to initial state
        backbone.patch_embed.proj.load_state_dict(
            {k: v.to(device) for k, v in patch_embed_proj_state.items()}
        )

        try:
            target_aug_logits = get_target_aug_logits(args, target_class)
        except ValueError as e:
            _logger.warning(f'Skipping class {target_class}: {e}')
            continue

        target_indices = [
            i for i in range(len(raw_dataset))
            if raw_dataset.targets[i] == target_class
        ]
        if not target_indices:
            _logger.warning(
                f'No images for class {target_class} in {val_dir}, skipping'
            )
            continue

        filtered_dataset = Subset(raw_dataset, target_indices)
        dataset_tta = _SimpleTransformDataset(
            filtered_dataset,
            base_transform=base_transform,
            final_transform=final_transform,
        )

        loader_tta = torch.utils.data.DataLoader(
            dataset_tta,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=args.pin_mem,
            drop_last=len(dataset_tta) >= args.batch_size,
        )
        loader_eval = torch.utils.data.DataLoader(
            dataset_tta,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=args.pin_mem,
            drop_last=False,
        )

        optimizer = create_optimizer_v2(
            stem_extractor,
            opt=args.opt,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        lr_scheduler, num_epochs = create_scheduler_v2(
            optimizer,
            sched=args.sched,
            num_epochs=args.epochs,
            warmup_epochs=args.warmup_epochs,
            min_lr=args.min_lr,
        )

        # Before TTA (epoch=-1)
        metrics_before = eval_tta_metrics(
            backbone,
            stem_extractor,
            aug_classifier,
            loader_eval,
            args.tta_norm,
            target_aug_logits,
            device,
        )
        row_before = {
            'target_class': target_class,
            'epoch': -1,
            **metrics_before,
        }
        with open(args.output_tsv, 'a', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')
            if write_header:
                w.writeheader()
                write_header = False
            w.writerow(row_before)

        # TTA training loop
        global_step = 0
        for epoch in range(num_epochs):
            train_metrics, global_step = train_tta_one_epoch(
                epoch,
                backbone,
                stem_extractor,
                aug_classifier,
                loader_tta,
                optimizer,
                args.tta_norm,
                target_aug_logits,
                global_step,
                args,
                device,
            )
            if lr_scheduler is not None:
                lr_scheduler.step(epoch + 1)

            row = {
                'target_class': target_class,
                'epoch': epoch,
                **train_metrics,
            }
            with open(args.output_tsv, 'a', newline='') as f:
                w = csv.DictWriter(f, fieldnames=fieldnames, delimiter='\t')
                w.writerow(row)

        _logger.info(
            f'Class {target_class}: before acc={metrics_before["backbone_acc"]:.1f}%, '
            f'after acc={train_metrics["backbone_acc"]:.1f}%'
        )

    _logger.info(f'TTA record complete. Output: {args.output_tsv}')


if __name__ == '__main__':
    main()

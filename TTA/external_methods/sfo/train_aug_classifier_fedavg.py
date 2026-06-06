#!/usr/bin/env python3
"""Joint Batch AugClassifier Training — All 9 Variants Per Image

Each image is augmented by all 8 photometric transforms at maximum severity,
plus the original, yielding 9 variants per image in a single batch.

With batch_size=512, each step processes 512//9 = 56 source images × 9 = 504
image variants simultaneously, ensuring all augmentation types are balanced
within every gradient step.
"""
import argparse
import csv
import importlib
import logging
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import ImageFolder

from timm import utils
from timm.data.transforms import RandomResizedCropAndInterpolation
from timm.data.auto_augment import (
    _AUGMIX_SL_TRANSFORMS_V2, augmix_sl_ops_v2,
    _HPARAMS_DEFAULT,
)

from train_aug_classifier import AugClassifier, NUM_AUG_TRANSFORMS

_logger = logging.getLogger('train_aug_cls_joint')

_SAFE_MAG_CAPS = {
    'NegativeIntensity': 0.8,
    'SolarizeIncreasing': 0.7,
}

AUG_NAMES = list(_AUGMIX_SL_TRANSFORMS_V2)

_DISPLAY_NAMES = {
    'PositiveIntensity': 'PosIntensity',
    'NegativeIntensity': 'NegIntensity',
    'SaturationV2': 'Saturation',
    'SharpnessV2': 'Sharpness',
    'GaussianBlurIncreasing': 'GaussBlur',
    'PosterizeIncreasing': 'Posterize',
    'SolarizeIncreasing': 'Solarize',
    'SaltAndPepperIncreasing': 'S&P',
}


# =============================================================================
# Dataset: all 9 variants (original + 8 augmentations) per image
# =============================================================================

class AllAugDataset(torch.utils.data.Dataset):
    """Returns 1 original + 8 augmented variants per image.

    All variants share the same geometric transform (RandomResizedCrop + HFlip).
    Each augmentation is applied at maximum severity (respecting safe caps).

    __getitem__ returns:
        images:  (9, 3, H, W)           — [original, aug0, ..., aug7]
        targets: (9, num_aug_classes)    — soft labels
        class_label: int                 — ImageNet class (for reference)
    """

    def __init__(
        self,
        root: str,
        aug_ops: list,
        img_size: int = 224,
        smoothing: float = 0.1,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ):
        self.dataset = ImageFolder(root)
        self.aug_ops = aug_ops
        self.num_augs = len(aug_ops)
        self.smoothing = smoothing

        self.safe_caps = [
            _SAFE_MAG_CAPS.get(AUG_NAMES[i], 1.0) for i in range(self.num_augs)
        ]

        self.geometric = transforms.Compose([
            RandomResizedCropAndInterpolation(
                img_size, scale=(0.08, 1.0), ratio=(3./4., 4./3.)),
            transforms.RandomHorizontalFlip(p=0.5),
        ])

        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        self._clean_target = self._make_clean_target()
        self._aug_targets = self._make_aug_targets()

    def _make_clean_target(self):
        eps = self.smoothing / (self.num_augs - 1)
        target = torch.full((self.num_augs,), eps)
        return target / target.sum()

    def _make_aug_targets(self):
        targets = []
        eps = self.smoothing / (self.num_augs - 1)
        for i in range(self.num_augs):
            t = torch.full((self.num_augs,), eps)
            t[i] = 1.0
            targets.append(t / t.sum())
        return torch.stack(targets)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        pil_img, class_label = self.dataset[idx]
        pil_img = pil_img.convert('RGB')

        geom_img = self.geometric(pil_img)

        images = [self.to_tensor(geom_img)]

        for i in range(self.num_augs):
            aug_img = self.aug_ops[i](geom_img.copy(), self.safe_caps[i])
            images.append(self.to_tensor(aug_img))

        images = torch.stack(images)                        # (9, 3, H, W)
        targets = torch.cat([
            self._clean_target.unsqueeze(0),                # (1, num_augs)
            self._aug_targets,                              # (8, num_augs)
        ], dim=0)                                           # (9, num_augs)

        return images, targets, class_label


# =============================================================================
# Label construction & loss (kept for evaluation compatibility)
# =============================================================================

def make_aug_target(aug_idx, severity, num_classes, smoothing=0.1):
    eps = smoothing / (num_classes - 1)
    target = torch.full((num_classes,), eps)
    target[aug_idx] = severity
    return target / target.sum()


def make_clean_target(num_classes, smoothing=0.1):
    eps = smoothing / (num_classes - 1)
    target = torch.full((num_classes,), eps)
    return target / target.sum()


def soft_cross_entropy(logits, targets):
    log_probs = F.log_softmax(logits, dim=-1)
    return -(targets * log_probs).sum(dim=-1).mean()


# =============================================================================
# LR scheduler
# =============================================================================

def cosine_lr(base_lr, min_lr, current_step, total_steps):
    return min_lr + 0.5 * (base_lr - min_lr) * (
        1 + math.cos(math.pi * current_step / total_steps)
    )


# =============================================================================
# Training: one epoch
# =============================================================================

def train_one_epoch(model, loader, optimizer, device, model_dtype,
                    num_classes, clip_grad=None, log_interval=50):
    model.train()
    losses = utils.AverageMeter()
    top1 = utils.AverageMeter()

    for batch_idx, (images_9, targets_9, _) in enumerate(loader):
        B = images_9.shape[0]
        imgs = images_9.reshape(-1, *images_9.shape[2:]).to(
            device=device, dtype=model_dtype)                   # (B*9, 3, H, W)
        tgts = targets_9.reshape(-1, targets_9.shape[-1]).to(
            device=device, dtype=model_dtype)                   # (B*9, num_classes)

        logits = model(imgs)
        loss = soft_cross_entropy(logits, tgts)

        optimizer.zero_grad()
        loss.backward()
        if clip_grad is not None:
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()

        with torch.no_grad():
            pred = logits.argmax(dim=-1)
            hard_label = tgts.argmax(dim=-1)
            acc1 = (pred == hard_label).float().mean().item() * 100.0

        losses.update(loss.item(), imgs.shape[0])
        top1.update(acc1, imgs.shape[0])

        if batch_idx % log_interval == 0:
            _logger.info(
                f'    [{batch_idx:>4d}/{len(loader)}]  '
                f'loss={losses.val:.4f}({losses.avg:.4f})  '
                f'acc={top1.val:.1f}%({top1.avg:.1f}%)')

    return losses.avg, top1.avg


# =============================================================================
# Evaluation: same structure as training, but on a different data split
# =============================================================================

@torch.no_grad()
def evaluate(model, eval_loader, device, model_dtype, num_classes, max_steps):
    """Evaluate on eval_loader for up to max_steps batches.

    eval_loader yields the same (images_9, targets_9, class_label) as training.
    Returns per-aug-type accuracy and clean prediction distribution.
    """
    model.eval()

    per_aug_correct = torch.zeros(num_classes)
    per_aug_total = torch.zeros(num_classes)
    clean_correct = 0
    clean_total = 0
    clean_pred_counts = torch.zeros(num_classes)
    losses = utils.AverageMeter()

    for step, (images_9, targets_9, _) in enumerate(eval_loader):
        if step >= max_steps:
            break

        B = images_9.shape[0]
        num_var = images_9.shape[1]

        imgs = images_9.reshape(-1, *images_9.shape[2:]).to(
            device=device, dtype=model_dtype)
        tgts = targets_9.reshape(-1, targets_9.shape[-1]).to(
            device=device, dtype=model_dtype)

        logits = model(imgs)
        loss = soft_cross_entropy(logits, tgts)
        losses.update(loss.item(), imgs.shape[0])

        pred = logits.argmax(dim=-1)
        hard_label = tgts.argmax(dim=-1)

        pred = pred.reshape(B, num_var)
        hard_label = hard_label.reshape(B, num_var)

        for aug_idx in range(num_classes):
            col = aug_idx + 1
            per_aug_correct[aug_idx] += (pred[:, col] == aug_idx).sum().item()
            per_aug_total[aug_idx] += B

        clean_preds = pred[:, 0]
        for p in clean_preds:
            clean_pred_counts[p] += 1
        clean_total += B

    results = {}
    for aug_idx in range(num_classes):
        aug_name = AUG_NAMES[aug_idx]
        disp = _DISPLAY_NAMES.get(aug_name, aug_name)
        total = per_aug_total[aug_idx].item()
        acc = 100.0 * per_aug_correct[aug_idx].item() / total if total > 0 else 0.0
        results[disp] = acc

    clean_dist = {
        _DISPLAY_NAMES.get(AUG_NAMES[i], AUG_NAMES[i]):
        f'{100.0 * clean_pred_counts[i].item() / clean_total:.1f}%'
        for i in range(num_classes)
    }
    results['clean_dist'] = clean_dist
    results['eval_loss'] = losses.avg

    return results


# =============================================================================
# Argument parsing
# =============================================================================

parser = argparse.ArgumentParser(description='Joint Batch AugClassifier Training')

group = parser.add_argument_group('Data')
group.add_argument('--data-dir', type=str, default='/data/imagenet/imagenet/val',
                   help='Training data (ImageNet val)')
group.add_argument('--eval-data-dir', type=str, default='/data/imagenet/imagenet/train',
                   help='Evaluation data (ImageNet train)')
group.add_argument('--img-size', type=int, default=224)

group = parser.add_argument_group('Model')
group.add_argument('--embed-dim', type=int, default=768,
                   help='ViT embedding dim for projection alignment')

group = parser.add_argument_group('Training')
group.add_argument('--epochs', type=int, default=20)
group.add_argument('--lr', type=float, default=1e-3)
group.add_argument('--min-lr', type=float, default=1e-5)
group.add_argument('--weight-decay', type=float, default=1e-4)
group.add_argument('--clip-grad', type=float, default=1.0)
group.add_argument('--smoothing', type=float, default=0.1,
                   help='Label smoothing epsilon')
group.add_argument('-b', '--batch-size', type=int, default=512,
                   help='Total batch size (images_per_step = batch_size // 9 * 9)')
group.add_argument('-j', '--workers', type=int, default=8)
group.add_argument('--pin-mem', action='store_true', default=False)
group.add_argument('--device', default='cuda', type=str)
group.add_argument('--seed', type=int, default=42)
group.add_argument('--amp', action='store_true', default=False)
group.add_argument('--log-interval', type=int, default=50)
group.add_argument('--device-modules', default=None, type=str, nargs='+')

group = parser.add_argument_group('Output')
group.add_argument('--output-dir', type=str, default=None)
group.add_argument('--resume', type=str, default=None,
                   help='Resume AugClassifier from checkpoint')


def main():
    utils.setup_default_logging()
    args = parser.parse_args()

    num_variants = 1 + NUM_AUG_TRANSFORMS   # 9
    imgs_per_batch = args.batch_size // num_variants

    if args.output_dir is None:
        args.output_dir = (
            f'./output/aug_cls_joint_e{args.epochs}'
            f'_lr{args.lr}_b{args.batch_size}'
        )
    os.makedirs(args.output_dir, exist_ok=True)

    if args.device_modules:
        for module in args.device_modules:
            importlib.import_module(module)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    device = torch.device(args.device)
    utils.random_seed(args.seed, 0)

    model_dtype = torch.float32

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    # ── Create AugClassifier ──
    num_classes = NUM_AUG_TRANSFORMS  # 8
    _logger.info(f'Creating AugClassifier: {num_classes} aug types, embed_dim={args.embed_dim}')
    _logger.info(f'Aug types: {AUG_NAMES}')

    model = AugClassifier(
        in_chans=3,
        num_classes=num_classes,
        embed_dim=args.embed_dim,
    ).to(device=device, dtype=model_dtype)

    start_epoch = 1
    if args.resume:
        _logger.info(f'Resuming from: {args.resume}')
        ckpt = torch.load(args.resume, map_location='cpu')
        sd = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
        model.load_state_dict(sd, strict=False)
        if 'epoch' in ckpt:
            start_epoch = ckpt['epoch'] + 1
            _logger.info(f'  Resuming from epoch {start_epoch}')

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    _logger.info(f'Parameters: {total_params:,} total, {trainable_params:,} trainable')

    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    if num_gpus > 1:
        model = nn.DataParallel(model)
        _logger.info(f'Using DataParallel on {num_gpus} GPUs')

    # ── Create augmentation ops ──
    aug_ops = augmix_sl_ops_v2(hparams=_HPARAMS_DEFAULT)
    assert len(aug_ops) == num_classes, \
        f'Expected {num_classes} ops, got {len(aug_ops)}'

    # ── Training Dataset & DataLoader (ImageNet val) ──
    train_dataset = AllAugDataset(
        root=args.data_dir,
        aug_ops=aug_ops,
        img_size=args.img_size,
        smoothing=args.smoothing,
        mean=mean, std=std,
    )
    loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=imgs_per_batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    train_steps = len(loader)

    _logger.info(f'Train data: {len(train_dataset)} images from {args.data_dir}')
    _logger.info(f'  {imgs_per_batch} source imgs/batch × {num_variants} variants '
                 f'= {imgs_per_batch * num_variants} imgs/step, '
                 f'{train_steps} steps/epoch')

    # ── Evaluation Dataset & DataLoader (ImageNet train) ──
    eval_dataset = AllAugDataset(
        root=args.eval_data_dir,
        aug_ops=aug_ops,
        img_size=args.img_size,
        smoothing=args.smoothing,
        mean=mean, std=std,
    )
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset,
        batch_size=imgs_per_batch,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    _logger.info(f'Eval data:  {len(eval_dataset)} images from {args.eval_data_dir}')
    _logger.info(f'  Eval runs {train_steps} steps per epoch (same as training)')

    # ── Optimizer ──
    base_model = model.module if isinstance(model, nn.DataParallel) else model
    optimizer = torch.optim.AdamW(
        base_model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # ── CSV logger ──
    csv_path = os.path.join(args.output_dir, 'training_log.csv')
    write_csv_header = (start_epoch == 1)
    csv_file = open(csv_path, 'a' if start_epoch > 1 else 'w', newline='')
    csv_writer = csv.writer(csv_file)
    if write_csv_header:
        aug_disp_names = [_DISPLAY_NAMES.get(n, n) for n in AUG_NAMES]
        csv_writer.writerow([
            'epoch', 'lr', 'train_loss', 'train_acc',
            *aug_disp_names, 'avg_acc',
        ])

    best_avg_acc = 0.0

    _logger.info(f'\n{"="*70}')
    _logger.info(f'Joint Batch Training: {args.epochs} epochs, '
                 f'smoothing={args.smoothing}')
    _logger.info(f'{"="*70}\n')

    # ── Training Loop ──
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()
        lr = cosine_lr(args.lr, args.min_lr, epoch - 1, args.epochs)
        for pg in optimizer.param_groups:
            pg['lr'] = lr

        _logger.info(f'\n{"="*60}')
        _logger.info(f'  Epoch {epoch}/{args.epochs}  (lr={lr:.6f})')
        _logger.info(f'{"="*60}')

        train_loss, train_acc = train_one_epoch(
            model, loader, optimizer, device, model_dtype,
            num_classes, clip_grad=args.clip_grad,
            log_interval=args.log_interval,
        )

        epoch_time = time.time() - epoch_start
        _logger.info(f'\n  Epoch {epoch} done | loss={train_loss:.4f} | '
                     f'acc={train_acc:.1f}% | time={epoch_time:.1f}s')

        # ── Evaluation (ImageNet train, same # of steps as training) ──
        _logger.info(f'\n  Evaluating on ImageNet train ({train_steps} steps)...')
        eval_results = evaluate(
            model, eval_loader, device, model_dtype,
            num_classes, max_steps=train_steps,
        )

        _logger.info(f'  Eval loss: {eval_results["eval_loss"]:.4f}')
        aug_accs = []
        for aug_name in AUG_NAMES:
            disp = _DISPLAY_NAMES.get(aug_name, aug_name)
            acc = eval_results.get(disp, 0.0)
            aug_accs.append(acc)
            _logger.info(f'    {disp:>12s}: {acc:>5.1f}%')

        avg_acc = sum(aug_accs) / len(aug_accs) if aug_accs else 0.0
        _logger.info(f'\n  Average accuracy: {avg_acc:.1f}%')

        if 'clean_dist' in eval_results:
            _logger.info(f'  Clean pred distribution: {eval_results["clean_dist"]}')

        # ── CSV logging ──
        csv_writer.writerow([
            epoch, f'{lr:.6f}', f'{train_loss:.4f}', f'{train_acc:.1f}',
            *[f'{a:.1f}' for a in aug_accs],
            f'{avg_acc:.1f}',
        ])
        csv_file.flush()

        # ── Checkpoint ──
        is_best = avg_acc > best_avg_acc
        if is_best:
            best_avg_acc = avg_acc

        ckpt = {
            'epoch': epoch,
            'state_dict': base_model.state_dict(),
            'num_transforms': num_classes,
            'embed_dim': args.embed_dim,
            'aug_names': AUG_NAMES,
            'avg_acc': avg_acc,
            'best_avg_acc': best_avg_acc,
        }

        torch.save(ckpt, os.path.join(args.output_dir, 'last.pth'))
        if is_best:
            torch.save(ckpt, os.path.join(args.output_dir, 'best.pth'))
            _logger.info(f'  ** New best: {avg_acc:.1f}% (saved)')

        _logger.info('')

    csv_file.close()
    _logger.info(f'\nTraining complete. Best avg accuracy: {best_avg_acc:.1f}%')
    _logger.info(f'Output: {args.output_dir}/')


if __name__ == '__main__':
    main()

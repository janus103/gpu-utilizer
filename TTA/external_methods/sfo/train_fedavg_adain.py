#!/usr/bin/env python3
"""FedAvg Training with AdaIN Auxiliary Network for ViT

Federated Averaging over 15 ImageNet-C corruption clients.
Each client trains a shared auxiliary network that:
  1. Takes channel attention output (B, 768) from the frozen ViT's last-block attention
  2. Produces per-patch AdaIN parameters (B, 196, 2) — mean and std
  3. Applies AdaIN to the patch embedding output before the ViT backbone

Two-pass forward:
  Pass 1: frozen model forward → collect channel attention (B, 768)
  Pass 2: channel attention → aux network → AdaIN on patch embedding → backbone → CE loss
  Backprop updates only the aux network.
"""
import argparse
import copy
import csv
import importlib
import logging
import math
import os
import time
from collections import OrderedDict
from contextlib import suppress
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from timm import utils
from timm.data import create_dataset, create_loader, resolve_data_config
from timm.layers import set_fast_norm
from timm.models import create_model, safe_model_name, resume_checkpoint

_logger = logging.getLogger('fedavg')

CORRUPTIONS = [
    'gaussian_noise', 'shot_noise', 'impulse_noise',
    'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur',
    'snow', 'frost', 'fog', 'brightness',
    'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
]


ACT_LAYERS = {
    'relu': lambda: nn.ReLU(inplace=True),
    'gelu': lambda: nn.GELU(),
    'silu': lambda: nn.SiLU(inplace=True),
    'leaky_relu': lambda: nn.LeakyReLU(inplace=True),
    'tanh': lambda: nn.Tanh(),
    'none': lambda: nn.Identity(),
}


class AdaINAuxNet(nn.Module):
    """Auxiliary network: channel attention (768) → AdaIN params (196 × 2)."""

    def __init__(self, in_dim=768, num_patches=196, hidden_dim=256, act='relu'):
        super().__init__()
        self.num_patches = num_patches
        act_fn = ACT_LAYERS[act]
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            act_fn(),
            nn.Linear(hidden_dim, hidden_dim),
            act_fn(),
            nn.Linear(hidden_dim, num_patches * 2),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, ca):
        # ca: (B, 768)
        out = self.net(ca)                          # (B, 196*2)
        out = out.view(-1, self.num_patches, 2)     # (B, 196, 2)
        mean_shift = out[:, :, 0]                   # (B, 196)
        std_scale = out[:, :, 1]                    # (B, 196)
        return mean_shift, std_scale


def apply_adain(patch_tokens, mean_shift, std_scale):
    """Apply AdaIN to patch tokens.

    Args:
        patch_tokens: (B, N, C) — patch embedding output (after pos_embed, before blocks)
        mean_shift: (B, N) — predicted mean shift per patch
        std_scale: (B, N) — predicted std scale per patch (added to 1.0 for residual)

    Returns:
        (B, N, C) — AdaIN-transformed tokens
    """
    eps = 1e-6
    # Normalize each patch token across the channel dimension
    mu = patch_tokens.mean(dim=-1, keepdim=True)     # (B, N, 1)
    sigma = patch_tokens.std(dim=-1, keepdim=True)    # (B, N, 1)
    x_norm = (patch_tokens - mu) / (sigma + eps)      # (B, N, C)

    # Apply predicted affine transform (residual: starts at identity)
    new_std = (1.0 + std_scale).unsqueeze(-1)          # (B, N, 1)
    new_mean = mean_shift.unsqueeze(-1)                # (B, N, 1)

    return x_norm * (sigma * new_std) + (mu + new_mean)


def _get_base_model(model):
    m = model
    if hasattr(m, 'module'):
        m = m.module
    if hasattr(m, '_orig_mod'):
        m = m._orig_mod
    return m


class ChannelAttnCollectorBatch:
    """Hook to capture per-batch channel attention output."""

    def __init__(self):
        self._ca = None
        self._hook = None

    def register(self, model):
        base = _get_base_model(model)
        if hasattr(base, 'channel_attn'):
            self._hook = base.channel_attn.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, input, output):
        self._ca = output.detach()  # (B, C, 1, 1), stays on device

    def pop(self):
        ca = self._ca
        self._ca = None
        return ca

    def remove(self):
        if self._hook is not None:
            self._hook.remove()
            self._hook = None


def two_pass_forward(model, aux_net, images, collector, device):
    """Two-pass forward: collect CA, then apply AdaIN and run again.

    Returns logits from the second pass.
    """
    base = _get_base_model(model)

    # --- Pass 1: frozen forward to get channel attention ---
    with torch.no_grad():
        _ = model(images)
    ca_batch = collector.pop()  # (B, 768, 1, 1)
    ca_input = ca_batch.squeeze(-1).squeeze(-1)  # (B, 768)

    # --- Aux network: predict AdaIN params ---
    mean_shift, std_scale = aux_net(ca_input)  # (B, 196), (B, 196)

    # --- Pass 2: patch embed → AdaIN → backbone → head ---
    # Step 1: patch embedding (frozen)
    with torch.no_grad():
        x = base.patch_embed(images)        # (B, 196, 768) after flatten+transpose
        x = base._pos_embed(x)              # add positional embedding
        x = base.patch_drop(x)
        x = base.norm_pre(x)

    # Step 2: separate CLS token from patch tokens
    if base.num_prefix_tokens > 0:
        prefix_tokens = x[:, :base.num_prefix_tokens].detach()  # (B, 1, 768) CLS
        patch_tokens = x[:, base.num_prefix_tokens:]             # (B, 196, 768)
    else:
        prefix_tokens = None
        patch_tokens = x

    # Step 3: apply AdaIN (this is the only differentiable path for aux_net)
    patch_tokens = apply_adain(patch_tokens, mean_shift, std_scale)

    # Step 4: reassemble
    if prefix_tokens is not None:
        x = torch.cat((prefix_tokens, patch_tokens), dim=1)
    else:
        x = patch_tokens

    # Step 5: transformer blocks (frozen, but gradients flow through for aux_net)
    x = base._apply_early_token_norm(x, 0)
    for i, blk in enumerate(base.blocks):
        x = blk(x)
        if i < 3:
            x = base._apply_early_token_norm(x, i + 1)

    # Step 6: last-position parallel attention (frozen)
    apply_last = base.parallel_attention and (base.vit_last or base.vit_closed is not None)
    if apply_last:
        B, N, C = x.shape
        if base.num_prefix_tokens > 0:
            pf = x[:, :base.num_prefix_tokens]
            pt = x[:, base.num_prefix_tokens:]
        else:
            pf = None
            pt = x
        H = W = int(math.sqrt(pt.shape[1]))
        pt = pt.transpose(1, 2).reshape(B, C, H, W)
        if base.vit_closed is not None:
            pt = base._apply_parallel_attention_last(pt)
        else:
            pt = base._apply_parallel_attention(pt)
        pt = pt.flatten(2).transpose(1, 2)
        if pf is not None:
            x = torch.cat((pf, pt), dim=1)
        else:
            x = pt

    # Step 7: final norm + head
    x = base.norm(x)
    x = base.forward_head(x)
    return x


def create_client_loader(corruption, args, data_config, model_dtype, device):
    """Create validation-style loader for one corruption client."""
    data_dir = os.path.join(args.data_root, corruption, str(args.severity))

    if args.input_img_mode is None:
        input_img_mode = 'RGB' if data_config['input_size'][0] == 3 else 'L'
    else:
        input_img_mode = args.input_img_mode

    dataset = create_dataset(
        '',
        root=data_dir,
        split=args.val_split,
        is_training=True,
        class_map=args.class_map,
        download=False,
        batch_size=args.batch_size,
        seed=args.seed,
        input_img_mode=input_img_mode,
    )

    loader = create_loader(
        dataset,
        input_size=data_config['input_size'],
        batch_size=args.batch_size,
        is_training=True,
        no_aug=True,
        num_workers=args.workers,
        mean=data_config['mean'],
        std=data_config['std'],
        pin_memory=args.pin_mem,
        img_dtype=model_dtype or torch.float32,
        device=device,
        distributed=False,
        use_prefetcher=not args.no_prefetcher,
    )
    return loader


def create_eval_loader(corruption, args, data_config, model_dtype, device):
    """Create evaluation loader for one corruption."""
    data_dir = os.path.join(args.data_root, corruption, str(args.severity))

    if args.input_img_mode is None:
        input_img_mode = 'RGB' if data_config['input_size'][0] == 3 else 'L'
    else:
        input_img_mode = args.input_img_mode

    dataset = create_dataset(
        '',
        root=data_dir,
        split=args.val_split,
        is_training=False,
        class_map=args.class_map,
        download=False,
        batch_size=args.batch_size,
        input_img_mode=input_img_mode,
    )

    loader = create_loader(
        dataset,
        input_size=data_config['input_size'],
        batch_size=args.batch_size,
        is_training=False,
        interpolation=data_config['interpolation'],
        num_workers=args.workers,
        crop_pct=data_config['crop_pct'],
        mean=data_config['mean'],
        std=data_config['std'],
        pin_memory=args.pin_mem,
        img_dtype=model_dtype or torch.float32,
        device=device,
        distributed=False,
        use_prefetcher=not args.no_prefetcher,
    )
    return loader


def local_train_one_epoch(model, aux_net, loader, optimizer, collector, args, device, model_dtype):
    """One local epoch of training for a single client."""
    model.eval()
    aux_net.train()

    losses_m = utils.AverageMeter()
    top1_m = utils.AverageMeter()
    prefetcher = not args.no_prefetcher

    for batch_idx, (images, target) in enumerate(loader):
        if not prefetcher:
            images = images.to(device=device, dtype=model_dtype)
            target = target.to(device=device)

        logits = two_pass_forward(model, aux_net, images, collector, device)
        loss = F.cross_entropy(logits, target)

        optimizer.zero_grad()
        loss.backward()
        if args.clip_grad is not None:
            nn.utils.clip_grad_norm_(aux_net.parameters(), args.clip_grad)
        optimizer.step()

        acc1, = utils.accuracy(logits.detach(), target, topk=(1,))
        losses_m.update(loss.item(), images.shape[0])
        top1_m.update(acc1.item(), images.shape[0])

    return losses_m.avg, top1_m.avg


@torch.no_grad()
def evaluate(model, aux_net, loader, collector, args, device, model_dtype):
    """Evaluate on one corruption dataset."""
    model.eval()
    aux_net.eval()

    losses_m = utils.AverageMeter()
    top1_m = utils.AverageMeter()
    top5_m = utils.AverageMeter()
    prefetcher = not args.no_prefetcher

    for images, target in loader:
        if not prefetcher:
            images = images.to(device=device, dtype=model_dtype)
            target = target.to(device=device)

        logits = two_pass_forward(model, aux_net, images, collector, device)
        loss = F.cross_entropy(logits, target)

        acc1, acc5 = utils.accuracy(logits, target, topk=(1, 5))
        losses_m.update(loss.item(), images.shape[0])
        top1_m.update(acc1.item(), images.shape[0])
        top5_m.update(acc5.item(), images.shape[0])

    return losses_m.avg, top1_m.avg, top5_m.avg


def fedavg_aggregate(global_state, client_states):
    """FedAvg: average all client model states."""
    avg_state = OrderedDict()
    num_clients = len(client_states)
    for key in global_state.keys():
        avg_state[key] = sum(cs[key] for cs in client_states) / num_clients
    return avg_state


# ── Argument parsing ──

config_parser = parser = argparse.ArgumentParser(description='Config', add_help=False)
parser.add_argument('-c', '--config', default='', type=str, metavar='FILE')

parser = argparse.ArgumentParser(description='FedAvg AdaIN Training')

group = parser.add_argument_group('Data')
group.add_argument('--data-root', type=str, default='/home/oem/servers/imagenet-c',
                   help='Root path for ImageNet-C (contains corruption_name/severity/)')
group.add_argument('--severity', type=int, default=5)
group.add_argument('--val-split', type=str, default='validation')
group.add_argument('--class-map', default='', type=str)
group.add_argument('--input-img-mode', default=None, type=str)

group = parser.add_argument_group('Model')
group.add_argument('--model', default='vit_base_patch16_224', type=str)
group.add_argument('--resume', default='./VIT_IMG_PAR/Normal_parallel_train_1_kernel_size_1_last/model_best.pth.tar',
                   type=str, help='Path to frozen model checkpoint')
group.add_argument('--num-classes', type=int, default=1000)
group.add_argument('--input-size', default=[3, 224, 224], nargs=3, type=int)
group.add_argument('--parallel-attention', action='store_true', default=True)
group.add_argument('--vit-kernel-size', type=int, default=1)
group.add_argument('--spatial-group-size', type=int, default=1)
group.add_argument('--vit-last', action='store_true', default=True)
group.add_argument('--vit-closed', type=str, default=None, choices=['same', 'diff'])

group = parser.add_argument_group('Auxiliary Network')
group.add_argument('--aux-hidden-dim', type=int, default=256,
                   help='Hidden dimension of AdaIN auxiliary network')
group.add_argument('--aux-act', type=str, default='relu',
                   choices=list(ACT_LAYERS.keys()),
                   help='Activation function for auxiliary network '
                        f'(choices: {", ".join(ACT_LAYERS.keys())})')
group.add_argument('--num-patches', type=int, default=196,
                   help='Number of patch tokens (14*14 for ViT-B/16 with 224 input)')

group = parser.add_argument_group('FedAvg')
group.add_argument('--rounds', type=int, default=10, help='Number of FedAvg communication rounds')
group.add_argument('--local-epochs', type=int, default=1, help='Local epochs per round per client')

group = parser.add_argument_group('Optimizer')
group.add_argument('--lr', type=float, default=1e-3)
group.add_argument('--weight-decay', type=float, default=1e-4)
group.add_argument('--clip-grad', type=float, default=1.0)

group = parser.add_argument_group('Training')
group.add_argument('-b', '--batch-size', type=int, default=64)
group.add_argument('--device', default='cuda', type=str)
group.add_argument('--seed', type=int, default=42)
group.add_argument('-j', '--workers', type=int, default=4)
group.add_argument('--pin-mem', action='store_true', default=False)
group.add_argument('--no-prefetcher', action='store_true', default=False)
group.add_argument('--amp', action='store_true', default=False)
group.add_argument('--amp-dtype', default='float16', type=str)
group.add_argument('--model-dtype', default=None, type=str)
group.add_argument('--log-interval', type=int, default=50)
group.add_argument("--local_rank", default=0, type=int)
group.add_argument('--device-modules', default=None, type=str, nargs='+')

group = parser.add_argument_group('Output')
group.add_argument('--output-dir', type=str, default=None,
                   help='Output directory (default: ./output/fedavg_adain_{aux_act})')


def _parse_args():
    args_config, remaining = config_parser.parse_known_args()
    if args_config.config:
        with open(args_config.config, 'r') as f:
            cfg = yaml.safe_load(f)
            parser.set_defaults(**cfg)
    args = parser.parse_args(remaining)
    return args


def main():
    utils.setup_default_logging()
    args = _parse_args()

    if args.device_modules:
        for module in args.device_modules:
            importlib.import_module(module)

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    if args.output_dir is None:
        args.output_dir = f'./output/fedavg_adain_{args.aux_act}'

    device = torch.device(args.device)
    utils.random_seed(args.seed, 0)

    model_dtype = None
    if args.model_dtype:
        model_dtype = getattr(torch, args.model_dtype)

    # ── Create frozen ViT model ──
    vit_norm_kwargs = {
        'sam_kernel_size': args.vit_kernel_size,
        'spatial_group_size': args.spatial_group_size,
    }
    if args.vit_last:
        vit_norm_kwargs['vit_last'] = True
    if args.vit_closed is not None:
        vit_norm_kwargs['vit_closed'] = args.vit_closed

    in_chans = args.input_size[0] if args.input_size else 3

    model = create_model(
        args.model,
        num_classes=args.num_classes,
        in_chans=in_chans,
        parallel_attention=args.parallel_attention,
        **vit_norm_kwargs,
    )
    model.to(device=device, dtype=model_dtype)

    if args.resume:
        resume_checkpoint(model, args.resume, optimizer=None, loss_scaler=None, log_info=True)

    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    _logger.info(f'Frozen model loaded: {safe_model_name(args.model)}, '
                 f'params: {sum(p.numel() for p in model.parameters()):,}')

    data_config = resolve_data_config(vars(args) | {'pretrained': False}, model=model, verbose=True)

    # ── Register channel attention collector ──
    collector = ChannelAttnCollectorBatch()
    collector.register(model)

    # ── Create auxiliary network ──
    embed_dim = _get_base_model(model).embed_dim
    aux_net = AdaINAuxNet(
        in_dim=embed_dim,
        num_patches=args.num_patches,
        hidden_dim=args.aux_hidden_dim,
        act=args.aux_act,
    ).to(device)

    aux_params = sum(p.numel() for p in aux_net.parameters())
    _logger.info(f'AdaIN auxiliary network: {aux_params:,} parameters, activation: {args.aux_act}')

    # ── Output dir ──
    os.makedirs(args.output_dir, exist_ok=True)
    summary_path = os.path.join(args.output_dir, 'summary.csv')

    # ── FedAvg training loop ──
    _logger.info(f'\nStarting FedAvg: {args.rounds} rounds, {len(CORRUPTIONS)} clients, '
                 f'local_epochs={args.local_epochs}')

    for rnd in range(1, args.rounds + 1):
        round_start = time.time()
        _logger.info(f'\n{"="*60}')
        _logger.info(f'  Round {rnd}/{args.rounds}')
        _logger.info(f'{"="*60}')

        global_state = copy.deepcopy(aux_net.state_dict())
        client_states = []

        for ci, corruption in enumerate(CORRUPTIONS):
            # Initialize client aux_net from global
            aux_net.load_state_dict(global_state)
            optimizer = torch.optim.AdamW(
                aux_net.parameters(), lr=args.lr, weight_decay=args.weight_decay,
            )

            loader = create_client_loader(corruption, args, data_config, model_dtype, device)

            for local_ep in range(args.local_epochs):
                loss, acc1 = local_train_one_epoch(
                    model, aux_net, loader, optimizer, collector, args, device, model_dtype,
                )

            _logger.info(f'  Client {ci:>2d} [{corruption:<22s}]  '
                         f'loss={loss:.4f}  acc@1={acc1:.2f}%')

            client_states.append(copy.deepcopy(aux_net.state_dict()))

        # FedAvg aggregation
        aggregated = fedavg_aggregate(global_state, client_states)
        aux_net.load_state_dict(aggregated)

        round_time = time.time() - round_start
        _logger.info(f'  Aggregated {len(client_states)} clients ({round_time:.1f}s)')

        # ── Evaluate after each round ──
        _logger.info(f'\n  Evaluating round {rnd}...')
        eval_results = OrderedDict()
        acc1_sum = 0.0

        for corruption in CORRUPTIONS:
            eval_loader = create_eval_loader(corruption, args, data_config, model_dtype, device)
            eval_loss, eval_acc1, eval_acc5 = evaluate(
                model, aux_net, eval_loader, collector, args, device, model_dtype,
            )
            eval_results[corruption] = {'loss': eval_loss, 'top1': eval_acc1, 'top5': eval_acc5}
            acc1_sum += eval_acc1
            _logger.info(f'    {corruption:<22s}  Acc@1={eval_acc1:.3f}%  Acc@5={eval_acc5:.3f}%')

        mean_acc1 = acc1_sum / len(CORRUPTIONS)
        _logger.info(f'  Round {rnd} Mean Acc@1: {mean_acc1:.3f}%')

        # ── Write summary CSV ──
        write_header = (rnd == 1)
        with open(summary_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if write_header:
                header = ['round', 'mean_acc1'] + [c for c in CORRUPTIONS]
                writer.writerow(header)
            row = [rnd, f'{mean_acc1:.3f}']
            for c in CORRUPTIONS:
                row.append(f'{eval_results[c]["top1"]:.3f}')
            writer.writerow(row)

        # ── Save checkpoint ──
        ckpt_path = os.path.join(args.output_dir, f'aux_net_round_{rnd:03d}.pth')
        torch.save({
            'round': rnd,
            'aux_net_state_dict': aux_net.state_dict(),
            'mean_acc1': mean_acc1,
        }, ckpt_path)
        _logger.info(f'  Checkpoint saved: {ckpt_path}')

    # ── Final summary ──
    best_ckpt = os.path.join(args.output_dir, 'aux_net_best.pth')
    torch.save({
        'round': args.rounds,
        'aux_net_state_dict': aux_net.state_dict(),
        'mean_acc1': mean_acc1,
    }, best_ckpt)

    collector.remove()

    _logger.info(f'\n{"="*60}')
    _logger.info(f'  Training complete. Summary: {summary_path}')
    _logger.info(f'  Final Mean Acc@1: {mean_acc1:.3f}%')
    _logger.info(f'{"="*60}')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Entropy consistency variant built on top of train_gaussian4.py.

This script reuses the original training pipeline and augments half-mode
training with an optional consistency loss between the original and augmented
halves of each batch.
"""

from contextlib import suppress
from typing import Optional

import torch
import torch.nn.functional as F

import train_gaussian4 as base
from timm import utils
from timm.models import model_parameters


group = base.parser.add_argument_group('Consistency parameters')
group.add_argument(
    '--consistency-mode',
    type=str,
    default='none',
    choices=['none', 'logit_kl', 'logit_jsd', 'embedding_cosine'],
    help='Consistency loss mode for half-mode pairs (default: none)',
)
group.add_argument(
    '--consistency-loss-weight',
    type=float,
    default=0.0,
    help='Weight for the selected consistency loss (default: 0.0)',
)


_base_parse_args = base._parse_args


def _parse_args():
    args, args_text = _base_parse_args()
    if args.consistency_loss_weight < 0:
        raise ValueError('--consistency-loss-weight must be non-negative.')
    if args.consistency_mode != 'none':
        if not args.half_mode:
            raise ValueError('--consistency-mode requires --half-mode.')
        if args.consistency_loss_weight == 0:
            raise ValueError(
                '--consistency-mode is enabled but --consistency-loss-weight is 0. '
                'Set a positive loss weight or use --consistency-mode none.'
            )
    return args, args_text


base._parse_args = _parse_args


def _unwrap_output(output: torch.Tensor) -> torch.Tensor:
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


def _split_half_mode_tensor(x: torch.Tensor):
    batch_size = x.shape[0]
    if batch_size % 2 != 0:
        raise ValueError(f'Half-mode consistency requires an even batch size, got {batch_size}.')
    half_batch = batch_size // 2
    return x[:half_batch], x[half_batch:]


def _flatten_features(features: torch.Tensor) -> torch.Tensor:
    if isinstance(features, (tuple, list)):
        features = features[0]
    if features.ndim > 2:
        features = features.flatten(1)
    return features


def _symmetric_kl_from_logits(orig_logits: torch.Tensor, aug_logits: torch.Tensor) -> torch.Tensor:
    orig_log_prob = F.log_softmax(orig_logits, dim=-1)
    aug_log_prob = F.log_softmax(aug_logits, dim=-1)
    orig_prob = orig_log_prob.exp()
    aug_prob = aug_log_prob.exp()
    return 0.5 * (
        F.kl_div(orig_log_prob, aug_prob, reduction='batchmean') +
        F.kl_div(aug_log_prob, orig_prob, reduction='batchmean')
    )


def _jsd_from_logits(orig_logits: torch.Tensor, aug_logits: torch.Tensor) -> torch.Tensor:
    orig_log_prob = F.log_softmax(orig_logits, dim=-1)
    aug_log_prob = F.log_softmax(aug_logits, dim=-1)
    orig_prob = orig_log_prob.exp()
    aug_prob = aug_log_prob.exp()
    mean_prob = 0.5 * (orig_prob + aug_prob)
    mean_prob = mean_prob.clamp_min(1e-8)
    return 0.5 * (
        F.kl_div(orig_log_prob, mean_prob, reduction='batchmean') +
        F.kl_div(aug_log_prob, mean_prob, reduction='batchmean')
    )


def _cosine_embedding_loss(orig_features: torch.Tensor, aug_features: torch.Tensor) -> torch.Tensor:
    orig_features = _flatten_features(orig_features)
    aug_features = _flatten_features(aug_features)
    return 1.0 - F.cosine_similarity(orig_features, aug_features, dim=1).mean()


def _compute_consistency_loss(
        args,
        logits: torch.Tensor,
        captured_features: Optional[torch.Tensor],
        device: torch.device,
) -> torch.Tensor:
    if args.consistency_mode == 'none' or args.consistency_loss_weight == 0:
        return torch.tensor(0.0, device=device)

    orig_logits, aug_logits = _split_half_mode_tensor(logits)

    if args.consistency_mode == 'logit_kl':
        return _symmetric_kl_from_logits(orig_logits, aug_logits)
    if args.consistency_mode == 'logit_jsd':
        return _jsd_from_logits(orig_logits, aug_logits)
    if args.consistency_mode == 'embedding_cosine':
        if captured_features is None:
            raise RuntimeError('Embedding consistency requested, but classifier features were not captured.')
        orig_features, aug_features = _split_half_mode_tensor(captured_features)
        return _cosine_embedding_loss(orig_features, aug_features)

    raise ValueError(f'Unsupported consistency mode: {args.consistency_mode}')


def train_one_epoch(
        epoch,
        model,
        loader,
        optimizer,
        loss_fn,
        args,
        device=torch.device('cuda'),
        lr_scheduler=None,
        saver=None,
        output_dir=None,
        amp_autocast=suppress,
        loss_scaler=None,
        model_dtype=None,
        model_ema=None,
        mixup_fn=None,
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
    cls_losses_m = utils.AverageMeter()
    gaussian_losses_m = utils.AverageMeter()
    half_mode_losses_m = utils.AverageMeter()
    consistency_losses_m = utils.AverageMeter()

    model.train()

    accum_steps = args.grad_accum_steps
    last_accum_steps = len(loader) % accum_steps
    updates_per_epoch = (len(loader) + accum_steps - 1) // accum_steps
    num_updates = epoch * updates_per_epoch
    last_batch_idx = len(loader) - 1
    last_batch_idx_to_accum = len(loader) - last_accum_steps

    data_start_time = update_start_time = base.time.time()
    optimizer.zero_grad()
    update_sample_count = 0
    for batch_idx, (input, target) in enumerate(loader):
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

        data_time_m.update(accum_steps * (base.time.time() - data_start_time))

        def _forward():
            with amp_autocast():
                base_model = base._get_base_model(model)
                captured = {}
                handle = None
                if args.half_mode and args.consistency_mode == 'embedding_cosine':
                    classifier = base_model.get_classifier()

                    def _capture_features(_module, module_inputs):
                        if module_inputs:
                            captured['features'] = module_inputs[0]

                    handle = classifier.register_forward_pre_hook(_capture_features)

                try:
                    output = _unwrap_output(model(input))
                finally:
                    if handle is not None:
                        handle.remove()

                _cls_loss = loss_fn(output, target)
                _loss = _cls_loss.clone()

                _gaussian_loss = torch.tensor(0.0, device=device)
                _half_mode_loss = torch.tensor(0.0, device=device)
                _consistency_loss = torch.tensor(0.0, device=device)

                nll_loss = base_model.get_nll_loss()
                if nll_loss is not None:
                    _gaussian_loss = nll_loss
                    _loss = _loss + args.gaussian_loss_weight * nll_loss

                nll_sam_loss = base_model.get_nll_sam_loss()
                if nll_sam_loss is not None:
                    _gaussian_loss = _gaussian_loss + args.sam_loss_weight * nll_sam_loss
                    _loss = _loss + args.sam_loss_weight * nll_sam_loss

                if args.half_mode:
                    spatial_logits_list = base_model.get_spatial_logits()
                    if spatial_logits_list is not None:
                        half_batch = input.shape[0] // 2
                        for logits in spatial_logits_list:
                            orig_logits = logits[:half_batch]
                            target_zeros = torch.zeros(
                                orig_logits.shape[0], *orig_logits.shape[2:],
                                dtype=torch.long, device=device,
                            )
                            _half_mode_loss = _half_mode_loss + F.cross_entropy(orig_logits, target_zeros)
                        _loss = _loss + args.half_mode_loss_weight * _half_mode_loss

                    _consistency_loss = _compute_consistency_loss(
                        args=args,
                        logits=output,
                        captured_features=captured.get('features'),
                        device=device,
                    )
                    _loss = _loss + args.consistency_loss_weight * _consistency_loss

            if accum_steps > 1:
                _loss /= accum_steps
            return _loss, _cls_loss, _gaussian_loss, _half_mode_loss, _consistency_loss

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

        batch_size = input.shape[0]
        global_batch_size = batch_size
        if args.distributed:
            global_batch_size *= args.world_size

        if has_no_sync and not need_update:
            with model.no_sync():
                loss, cls_loss, gaussian_loss, half_mode_loss, consistency_loss = _forward()
                _backward(loss)
        else:
            loss, cls_loss, gaussian_loss, half_mode_loss, consistency_loss = _forward()
            _backward(loss)

        losses_m.update(loss.item() * accum_steps, batch_size)
        cls_losses_m.update(cls_loss.item(), batch_size)
        gaussian_losses_m.update(gaussian_loss.item(), batch_size)
        if args.half_mode:
            half_mode_losses_m.update(half_mode_loss.item(), batch_size)
            consistency_losses_m.update(consistency_loss.item(), batch_size)
        update_sample_count += global_batch_size

        if not need_update:
            data_start_time = base.time.time()
            continue

        num_updates += 1
        optimizer.zero_grad()
        if model_ema is not None:
            model_ema.update(model, step=num_updates)

        if args.synchronize_step and device.type == 'cuda':
            torch.cuda.synchronize()
        time_now = base.time.time()

        update_time_m.update(base.time.time() - update_start_time)
        update_start_time = time_now

        if update_idx % args.log_interval == 0 or last_batch:
            lrl = [param_group['lr'] for param_group in optimizer.param_groups]
            lr = sum(lrl) / len(lrl)

            loss_avg, loss_now = losses_m.avg, losses_m.val
            if args.distributed:
                loss_avg = utils.reduce_tensor(loss.new([loss_avg]), args.world_size).item()
                loss_now = utils.reduce_tensor(loss.new([loss_now]), args.world_size).item()

            if utils.is_primary(args):
                hm_str = ''
                if args.half_mode:
                    hm_str = (
                        f'  HM-Loss: {half_mode_losses_m.val:#.3g} ({half_mode_losses_m.avg:#.3g})'
                        f'  Cons: {consistency_losses_m.val:#.3g} ({consistency_losses_m.avg:#.3g})'
                    )
                base._logger.info(
                    f'Train: {epoch} [{update_idx:>4d}/{updates_per_epoch} '
                    f'({100. * (update_idx + 1) / updates_per_epoch:>3.0f}%)]  '
                    f'Loss: {loss_now:#.3g} ({loss_avg:#.3g})  '
                    f'CLS: {cls_losses_m.val:#.3g} ({cls_losses_m.avg:#.3g})  '
                    f'G-Loss: {gaussian_losses_m.val:#.3g} ({gaussian_losses_m.avg:#.3g})'
                    f'{hm_str}  '
                    f'Time: {update_time_m.val:.3f}s, {update_sample_count / update_time_m.val:>7.2f}/s  '
                    f'({update_time_m.avg:.3f}s, {update_sample_count / update_time_m.avg:>7.2f}/s)  '
                    f'LR: {lr:.3e}  '
                    f'Data: {data_time_m.val:.3f} ({data_time_m.avg:.3f})'
                )

                if args.save_images and output_dir:
                    base.torchvision.utils.save_image(
                        input,
                        base.os.path.join(output_dir, 'train-batch-%d.jpg' % batch_idx),
                        padding=0,
                        normalize=True,
                    )

        if saver is not None and args.recovery_interval and (
                (update_idx + 1) % args.recovery_interval == 0):
            saver.save_recovery(epoch, batch_idx=update_idx)

        if lr_scheduler is not None:
            lr_scheduler.step_update(num_updates=num_updates, metric=losses_m.avg)

        update_sample_count = 0
        data_start_time = base.time.time()

    if hasattr(optimizer, 'sync_lookahead'):
        optimizer.sync_lookahead()

    loss_avg = losses_m.avg
    if args.distributed:
        loss_avg = torch.tensor([loss_avg], device=device, dtype=torch.float32)
        loss_avg = utils.reduce_tensor(loss_avg, args.world_size).item()
    return {'loss': loss_avg}


base.train_one_epoch = train_one_epoch


if __name__ == '__main__':
    base.main()

#!/usr/bin/env python3
"""
Save timm pretrained weights as ZOA-format checkpoint files.

Produces plain state_dict .pth files (same format as existing ZOA_WEIGHT/*.pth)
that can be used with:
    --pretrained-path ./ZOA_WEIGHT/ZOA_resnet50_timm_format.pth
    --pretrained-path ./ZOA_WEIGHT/ZOA_vit_base_timm_format.pth
    --initial-checkpoint ./ZOA_WEIGHT/...
    --resume ./ZOA_WEIGHT/...

Usage:
    # Save both ResNet50 and ViT-B/16 pretrained weights
    python save_pretrained_as_zoa.py

    # Save only ResNet50
    python save_pretrained_as_zoa.py --model resnet50

    # Save only ViT-B/16
    python save_pretrained_as_zoa.py --model vit_base_patch16_224

    # Custom output directory
    python save_pretrained_as_zoa.py --output-dir ./MY_WEIGHTS

    # With SE/SAM/parallel-attention: save backbone-only (strip extra modules)
    python save_pretrained_as_zoa.py --backbone-only

    # Verify saved weights load correctly into ZOA models
    python save_pretrained_as_zoa.py --verify
"""

import argparse
import os
import sys
from collections import OrderedDict

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import timm


MODELS = {
    'resnet50': {
        'timm_name': 'resnet50',
        'output_name': 'ZOA_resnet50_timm_format.pth',
        'num_classes': 1000,
        'extra_module_prefixes': ('se_module.', 'sam_module.', 'channel_attn.', 'spatial_attn.'),
    },
    'vit_base_patch16_224': {
        'timm_name': 'vit_base_patch16_224',
        'output_name': 'ZOA_vit_base_timm_format.pth',
        'num_classes': 1000,
        'extra_module_prefixes': (
            'channel_attn.', 'spatial_attn.',
            'channel_attn_last.', 'spatial_attn_last.',
            'vit_early_norms.',
        ),
    },
}


def save_pretrained_weights(model_key, output_dir, backbone_only=False, verbose=True):
    """Load timm pretrained model and save its state_dict in ZOA format."""
    cfg = MODELS[model_key]
    timm_name = cfg['timm_name']
    output_name = cfg['output_name']
    output_path = os.path.join(output_dir, output_name)

    print(f'\n{"="*60}')
    print(f'Model: {timm_name}')
    print(f'Output: {output_path}')
    print(f'{"="*60}')

    print(f'Loading pretrained {timm_name} from timm...')
    model = timm.create_model(timm_name, pretrained=True, num_classes=cfg['num_classes'])
    model.eval()

    sd = model.state_dict()

    if backbone_only:
        prefixes = cfg['extra_module_prefixes']
        original_count = len(sd)
        sd = OrderedDict(
            (k, v) for k, v in sd.items()
            if not any(k.startswith(p) for p in prefixes)
        )
        removed = original_count - len(sd)
        if removed > 0:
            print(f'Removed {removed} extra module keys (SE/SAM/attention)')

    print(f'State dict keys: {len(sd)}')
    if verbose:
        print(f'  First 5: {list(sd.keys())[:5]}')
        print(f'  Last 5:  {list(sd.keys())[-5:]}')

    os.makedirs(output_dir, exist_ok=True)
    torch.save(sd, output_path)

    file_size = os.path.getsize(output_path)
    print(f'Saved: {output_path} ({file_size / 1024 / 1024:.1f} MB)')
    return output_path


def verify_weights(model_key, weight_path, verbose=True):
    """Verify that saved weights can be loaded by the model used in train_gaussian*.py."""
    cfg = MODELS[model_key]
    timm_name = cfg['timm_name']

    print(f'\n--- Verifying {timm_name} with {weight_path} ---')

    saved_sd = torch.load(weight_path, map_location='cpu', weights_only=False)
    if not isinstance(saved_sd, dict):
        print(f'  ERROR: Expected dict, got {type(saved_sd)}')
        return False

    model_vanilla = timm.create_model(timm_name, pretrained=False, num_classes=cfg['num_classes'])
    model_sd = model_vanilla.state_dict()

    saved_keys = set(saved_sd.keys())
    model_keys = set(model_sd.keys())
    missing = model_keys - saved_keys
    unexpected = saved_keys - model_keys

    if missing:
        print(f'  Missing from saved weights ({len(missing)}):')
        for k in sorted(missing)[:10]:
            print(f'    {k}')
    if unexpected:
        print(f'  Unexpected in saved weights ({len(unexpected)}):')
        for k in sorted(unexpected)[:10]:
            print(f'    {k}')

    shape_mismatches = []
    for k in saved_keys & model_keys:
        if saved_sd[k].shape != model_sd[k].shape:
            shape_mismatches.append((k, saved_sd[k].shape, model_sd[k].shape))

    if shape_mismatches:
        print(f'  Shape mismatches ({len(shape_mismatches)}):')
        for k, s1, s2 in shape_mismatches[:10]:
            print(f'    {k}: saved={s1} vs model={s2}')

    result = model_vanilla.load_state_dict(saved_sd, strict=False)
    print(f'  load_state_dict result: missing={len(result.missing_keys)}, unexpected={len(result.unexpected_keys)}')

    if not missing and not unexpected and not shape_mismatches:
        print(f'  OK: Perfect match for vanilla {timm_name}')
    else:
        print(f'  WARN: Loading with strict=False will work (for --pretrained-path usage)')

    if model_key == 'resnet50':
        print(f'\n  Testing with SE+SAM modules (train_gaussian2 style)...')
        model_se_sam = timm.create_model(
            timm_name, pretrained=False, num_classes=cfg['num_classes'],
            use_se_module=True, use_sam_module=0,
        )
        result2 = model_se_sam.load_state_dict(saved_sd, strict=False)
        extra_keys = set(model_se_sam.state_dict().keys()) - saved_keys
        print(f'  SE+SAM extra params (randomly initialized): {len(extra_keys)}')
        if verbose and extra_keys:
            for k in sorted(extra_keys)[:15]:
                print(f'    {k}')
        print(f'  load_state_dict: missing={len(result2.missing_keys)}, unexpected={len(result2.unexpected_keys)}')

    elif model_key == 'vit_base_patch16_224':
        print(f'\n  Testing with parallel_attention (train_gaussian3 style)...')
        model_pa = timm.create_model(
            timm_name, pretrained=False, num_classes=cfg['num_classes'],
            parallel_attention=True, sam_kernel_size=1, spatial_group_size=1,
        )
        result2 = model_pa.load_state_dict(saved_sd, strict=False)
        extra_keys = set(model_pa.state_dict().keys()) - saved_keys
        print(f'  parallel_attention extra params (randomly initialized): {len(extra_keys)}')
        if verbose and extra_keys:
            for k in sorted(extra_keys)[:15]:
                print(f'    {k}')
        print(f'  load_state_dict: missing={len(result2.missing_keys)}, unexpected={len(result2.unexpected_keys)}')

    return True


def compare_with_existing(model_key, new_path, existing_dir='./ZOA_WEIGHT'):
    """Compare newly saved weights with existing ZOA weights."""
    cfg = MODELS[model_key]
    existing_path = os.path.join(existing_dir, cfg['output_name'])

    if not os.path.exists(existing_path):
        print(f'\n  No existing ZOA weights at {existing_path}, skipping comparison')
        return

    print(f'\n--- Comparing with existing: {existing_path} ---')
    new_sd = torch.load(new_path, map_location='cpu', weights_only=False)
    old_sd = torch.load(existing_path, map_location='cpu', weights_only=False)

    new_keys = set(new_sd.keys())
    old_keys = set(old_sd.keys())

    if new_keys == old_keys:
        print(f'  Key sets are identical ({len(new_keys)} keys)')
    else:
        only_new = new_keys - old_keys
        only_old = old_keys - new_keys
        if only_new:
            print(f'  Keys only in new: {sorted(only_new)[:5]}')
        if only_old:
            print(f'  Keys only in existing: {sorted(only_old)[:5]}')

    diff_count = 0
    for k in new_keys & old_keys:
        if new_sd[k].shape != old_sd[k].shape:
            print(f'  Shape diff: {k} new={new_sd[k].shape} old={old_sd[k].shape}')
            diff_count += 1
        elif not torch.equal(new_sd[k], old_sd[k]):
            max_diff = (new_sd[k].float() - old_sd[k].float()).abs().max().item()
            diff_count += 1
            if diff_count <= 5:
                print(f'  Value diff: {k} max_abs_diff={max_diff:.6e}')

    if diff_count == 0:
        print(f'  Weights are numerically identical!')
    else:
        print(f'  Total parameters with different values: {diff_count}/{len(new_keys & old_keys)}')
        print(f'  (This is expected if pretrained source differs from original ZOA training)')


def main():
    parser = argparse.ArgumentParser(description='Save timm pretrained weights in ZOA format')
    parser.add_argument('--model', type=str, default=None, choices=list(MODELS.keys()),
                        help='Which model to save (default: both)')
    parser.add_argument('--output-dir', type=str, default='./ZOA_WEIGHT',
                        help='Output directory (default: ./ZOA_WEIGHT)')
    parser.add_argument('--backbone-only', action='store_true',
                        help='Strip SE/SAM/attention module keys (save backbone only)')
    parser.add_argument('--verify', action='store_true',
                        help='Verify saved weights load correctly')
    parser.add_argument('--compare', action='store_true',
                        help='Compare with existing ZOA weights')
    parser.add_argument('--suffix', type=str, default=None,
                        help='Add suffix to output filename (e.g. "_v2" -> ZOA_resnet50_timm_format_v2.pth)')
    parser.add_argument('--quiet', action='store_true')
    args = parser.parse_args()

    models_to_save = [args.model] if args.model else list(MODELS.keys())

    saved_paths = {}
    for model_key in models_to_save:
        if args.suffix:
            original_name = MODELS[model_key]['output_name']
            base, ext = os.path.splitext(original_name)
            MODELS[model_key]['output_name'] = f'{base}{args.suffix}{ext}'

        path = save_pretrained_weights(
            model_key, args.output_dir,
            backbone_only=args.backbone_only,
            verbose=not args.quiet,
        )
        saved_paths[model_key] = path

    if args.verify:
        print(f'\n{"="*60}')
        print('VERIFICATION')
        print(f'{"="*60}')
        for model_key, path in saved_paths.items():
            verify_weights(model_key, path, verbose=not args.quiet)

    if args.compare:
        print(f'\n{"="*60}')
        print('COMPARISON WITH EXISTING ZOA WEIGHTS')
        print(f'{"="*60}')
        for model_key, path in saved_paths.items():
            compare_with_existing(model_key, path)

    print(f'\n{"="*60}')
    print('DONE')
    print(f'{"="*60}')
    print('\nSaved files:')
    for model_key, path in saved_paths.items():
        print(f'  {model_key}: {path}')
    print(f'\nUsage in train_gaussian2.py / train_gaussian3.py:')
    for model_key, path in saved_paths.items():
        cfg = MODELS[model_key]
        print(f'  python train_gaussian3.py --model {cfg["timm_name"]} --pretrained-path {path} \\')
        print(f'    --use-se-module --use-sam-module 0 --parallel-attention ...')
    print()


if __name__ == '__main__':
    main()

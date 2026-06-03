from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn

from schema import STANDARD_COLUMNS, UNKNOWN


def classify_conv(module: nn.Conv2d) -> str:
    if module.groups == module.in_channels and module.in_channels == module.out_channels:
        return "depthwise_conv"
    if module.kernel_size == (1, 1):
        return "pointwise_conv"
    return "standard_conv"


def empty_row(seq: int, layer_type: str, name: str, batch: int, dtype: str, repeat: int, warmup: int) -> Dict[str, str]:
    row = {column: UNKNOWN for column in STANDARD_COLUMNS}
    row.update(
        {
            "layer_seq": str(seq),
            "layer_type": layer_type,
            "layer_name": name,
            "batch": str(batch),
            "dtype": dtype,
            "repeat": str(repeat),
            "warmup": str(warmup),
        }
    )
    return row


def extract_layers(model_name: str, batch: int, image_size: int, dtype: str, repeat: int, warmup: int) -> List[Dict[str, str]]:
    import timm

    model = timm.create_model(model_name, pretrained=False).eval()
    rows: List[Dict[str, str]] = []
    hooks = []

    def hook(name: str, module: nn.Module):
        def inner(_module: nn.Module, inputs, _output):
            seq = len(rows)
            if isinstance(module, nn.Conv2d):
                x = inputs[0]
                row = empty_row(seq, classify_conv(module), name, batch, dtype, repeat, warmup)
                row.update(
                    {
                        "in_channels": str(module.in_channels),
                        "out_channels": str(module.out_channels),
                        "input_h": str(int(x.shape[-2])),
                        "input_w": str(int(x.shape[-1])),
                        "kernel_h": str(module.kernel_size[0]),
                        "kernel_w": str(module.kernel_size[1]),
                        "stride_h": str(module.stride[0]),
                        "stride_w": str(module.stride[1]),
                        "pad_h": str(module.padding[0]),
                        "pad_w": str(module.padding[1]),
                        "groups": str(module.groups),
                        "bias": str(module.bias is not None).lower(),
                    }
                )
                rows.append(row)
            elif isinstance(module, nn.Linear):
                row = empty_row(seq, "fully_connected", name, batch, dtype, repeat, warmup)
                row.update(
                    {
                        "in_features": str(module.in_features),
                        "out_features": str(module.out_features),
                        "bias": str(module.bias is not None).lower(),
                    }
                )
                rows.append(row)

        return inner

    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            hooks.append(module.register_forward_hook(hook(name, module)))

    with torch.no_grad():
        x = torch.randn(batch, 3, image_size, image_size)
        model(x)

    for handle in hooks:
        handle.remove()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract timm Conv/Linear layers to gpu_utilizer CSV schema.")
    parser.add_argument("--model", default="mobilenetv2_100")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--output", default="examples/mobilenetv2_100_bs1_layers.csv")
    parser.add_argument("--timm-root", default="")
    args = parser.parse_args()

    if args.repeat >= 50:
        raise ValueError("--repeat must be less than 50 for the requested validation.")
    if args.timm_root:
        sys.path.insert(0, str(Path(args.timm_root).resolve()))

    rows = extract_layers(args.model, args.batch, args.image_size, args.dtype, args.repeat, args.warmup)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=STANDARD_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} layers to {output_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn

from schema import LayerSpec


@dataclass
class Workload:
    module: nn.Module
    inputs: Tuple[torch.Tensor, ...]
    mode: str

    def run(self) -> torch.Tensor:
        return self.module(*self.inputs)


class MatrixModule(nn.Module):
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.matmul(a, b)


def dtype_from_name(name: str) -> torch.dtype:
    normalized = str(name).lower()
    if normalized in {"-1", "", "fp32", "float32"}:
        return torch.float32
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype '{name}'")


def positive(value: int, default: int) -> int:
    return value if value > 0 else default


def build_workload(layer: LayerSpec, device: torch.device, default_dtype: str) -> Workload:
    layer_type = layer.get("layer_type")
    dtype = dtype_from_name(layer.get("dtype", default_dtype) if layer.get("dtype") != "-1" else default_dtype)
    if layer_type in {"standard_conv", "depthwise_conv", "pointwise_conv"}:
        return build_conv(layer, device, dtype)
    if layer_type == "fully_connected":
        return build_fully_connected(layer, device, dtype)
    if layer_type == "matrix":
        return build_matrix(layer, device, dtype)
    if layer_type == "self_attention":
        return build_self_attention(layer, device, dtype)
    raise ValueError(f"Unsupported layer_type '{layer_type}'")


def build_conv(layer: LayerSpec, device: torch.device, dtype: torch.dtype) -> Workload:
    batch = positive(layer.int("batch"), 1)
    in_channels = positive(layer.int("in_channels"), 3)
    input_h = positive(layer.int("input_h"), 224)
    input_w = positive(layer.int("input_w"), input_h)

    layer_type = layer.get("layer_type")
    if layer_type == "pointwise_conv":
        kernel_h = kernel_w = 1
    else:
        kernel_h = positive(layer.int("kernel_h"), 3)
        kernel_w = positive(layer.int("kernel_w"), kernel_h)

    if layer_type == "depthwise_conv":
        groups = in_channels
        out_channels = positive(layer.int("out_channels"), in_channels)
    else:
        out_channels = positive(layer.int("out_channels"), 64)
        groups = positive(layer.int("groups"), 1)

    stride_h = positive(layer.int("stride_h"), 1)
    stride_w = positive(layer.int("stride_w"), stride_h)
    pad_h = max(layer.int("pad_h"), 0)
    pad_w = max(layer.int("pad_w"), pad_h)
    bias = layer.bool("bias", default=False)

    module = nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=(kernel_h, kernel_w),
        stride=(stride_h, stride_w),
        padding=(pad_h, pad_w),
        groups=groups,
        bias=bias,
    ).to(device=device, dtype=dtype)
    x = torch.randn(batch, in_channels, input_h, input_w, device=device, dtype=dtype)
    return Workload(module.eval(), (x,), "inference")


def build_fully_connected(layer: LayerSpec, device: torch.device, dtype: torch.dtype) -> Workload:
    batch = positive(layer.int("batch"), 1)
    in_features = positive(layer.int("in_features"), positive(layer.int("k"), 1024))
    out_features = positive(layer.int("out_features"), positive(layer.int("n"), 1000))
    bias = layer.bool("bias", default=False)
    module = nn.Linear(in_features, out_features, bias=bias).to(device=device, dtype=dtype)
    x = torch.randn(batch, in_features, device=device, dtype=dtype)
    return Workload(module.eval(), (x,), "inference")


def build_matrix(layer: LayerSpec, device: torch.device, dtype: torch.dtype) -> Workload:
    batch = layer.int("batch")
    m = positive(layer.int("m"), 1024)
    n = positive(layer.int("n"), 1024)
    k = positive(layer.int("k"), 1024)
    module = MatrixModule().to(device=device)
    if batch > 1:
        a = torch.randn(batch, m, k, device=device, dtype=dtype)
        b = torch.randn(batch, k, n, device=device, dtype=dtype)
    else:
        a = torch.randn(m, k, device=device, dtype=dtype)
        b = torch.randn(k, n, device=device, dtype=dtype)
    return Workload(module.eval(), (a, b), "inference")


def build_self_attention(layer: LayerSpec, device: torch.device, dtype: torch.dtype) -> Workload:
    batch = positive(layer.int("batch"), 1)
    seq_len = positive(layer.int("seq_len"), 128)
    embed_dim = positive(layer.int("embed_dim"), 768)
    num_heads = positive(layer.int("num_heads"), 12)
    bias = layer.bool("qkv_bias", default=False)
    module = nn.MultiheadAttention(embed_dim, num_heads, bias=bias, batch_first=True).to(device=device, dtype=dtype)
    x = torch.randn(batch, seq_len, embed_dim, device=device, dtype=dtype)
    return Workload(module.eval(), (x, x, x), "inference")

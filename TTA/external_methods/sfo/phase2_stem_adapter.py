#!/usr/bin/env python3
"""Stem-only LoRA adapter utilities for Phase-2 coupled meta-TTA.

This module provides a lightweight adapter that modifies only the stem weight:
  - ResNet family: ``conv1.weight``
  - ViT family: ``patch_embed.proj.weight``

The effective stem weight is:
    W_eff = W0 + scale * sum_m c_m * (B_m @ A_m)

where ``c_m`` are per-basis coefficients updated in the TTA inner loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch.func import functional_call as _functional_call
except Exception:  # pragma: no cover
    from torch.nn.utils.stateless import functional_call as _functional_call


@dataclass
class StemTargetSpec:
    """Resolved stem target spec."""

    weight_name: str
    module: nn.Module
    affine_weight_name: Optional[str] = None
    affine_bias_name: Optional[str] = None


def resolve_stem_target(model: nn.Module, target: str = "auto") -> StemTargetSpec:
    """Resolve supported stem parameter names from model structure."""
    if target == "auto":
        if hasattr(model, "conv1") and isinstance(model.conv1, nn.Conv2d):
            target = "conv1"
        elif hasattr(model, "patch_embed") and hasattr(model.patch_embed, "proj"):
            target = "patch_embed.proj"
        else:
            raise ValueError("Could not resolve stem target automatically.")

    if target == "conv1":
        if not hasattr(model, "conv1") or not isinstance(model.conv1, nn.Conv2d):
            raise ValueError("target=conv1 requires model.conv1 Conv2d.")
        aw, ab = None, None
        if hasattr(model, "bn1") and hasattr(model.bn1, "weight") and hasattr(model.bn1, "bias"):
            aw, ab = "bn1.weight", "bn1.bias"
        return StemTargetSpec(
            weight_name="conv1.weight",
            module=model.conv1,
            affine_weight_name=aw,
            affine_bias_name=ab,
        )

    if target == "patch_embed.proj":
        pe = getattr(model, "patch_embed", None)
        if pe is None or not hasattr(pe, "proj") or not isinstance(pe.proj, nn.Conv2d):
            raise ValueError("target=patch_embed.proj requires model.patch_embed.proj Conv2d.")
        aw, ab = None, None
        # Prefer patch embedding norm when it exists.
        if hasattr(pe, "norm") and isinstance(pe.norm, nn.Module) and not isinstance(pe.norm, nn.Identity):
            if hasattr(pe.norm, "weight") and hasattr(pe.norm, "bias"):
                aw, ab = "patch_embed.norm.weight", "patch_embed.norm.bias"
        # Fallback to model norm.
        elif hasattr(model, "norm") and hasattr(model.norm, "weight") and hasattr(model.norm, "bias"):
            aw, ab = "norm.weight", "norm.bias"
        return StemTargetSpec(
            weight_name="patch_embed.proj.weight",
            module=pe.proj,
            affine_weight_name=aw,
            affine_bias_name=ab,
        )

    raise ValueError(f"Unsupported stem target: {target}")


class StemLoRAAdapter(nn.Module):
    """LoRA basis adapter for a single stem Conv2d weight."""

    def __init__(
        self,
        model: nn.Module,
        target: str = "auto",
        num_bases: int = 4,
        rank: int = 4,
        scale: float = 1.0,
        init_std: float = 1e-3,
        train_affine: bool = False,
    ):
        super().__init__()
        self.spec = resolve_stem_target(model, target=target)
        self.num_bases = int(num_bases)
        self.rank = int(rank)
        self.scale = float(scale)
        self.train_affine = bool(train_affine)

        stem_w = dict(model.named_parameters())[self.spec.weight_name]
        self.register_buffer("w0", stem_w.detach().clone())
        self.weight_shape = tuple(stem_w.shape)
        self.out_dim = self.weight_shape[0]
        self.in_dim = int(stem_w.numel() // self.out_dim)

        # LoRA factorization for each basis m:
        # delta_m(flat) = B_m @ A_m, B:[out,rank], A:[rank,in]
        self.lora_A = nn.Parameter(torch.empty(self.num_bases, self.rank, self.in_dim))
        self.lora_B = nn.Parameter(torch.empty(self.num_bases, self.out_dim, self.rank))

        nn.init.normal_(self.lora_A, mean=0.0, std=init_std)
        nn.init.normal_(self.lora_B, mean=0.0, std=init_std)

        self.register_buffer("coeff_state", torch.zeros(self.num_bases))

        # Optional affine deltas for stem-adjacent normalization layer.
        self.affine_delta_weight: Optional[nn.Parameter] = None
        self.affine_delta_bias: Optional[nn.Parameter] = None
        if self.train_affine and self.spec.affine_weight_name and self.spec.affine_bias_name:
            named_params = dict(model.named_parameters())
            aw = named_params[self.spec.affine_weight_name]
            ab = named_params[self.spec.affine_bias_name]
            self.register_buffer("affine_w0", aw.detach().clone())
            self.register_buffer("affine_b0", ab.detach().clone())
            self.affine_delta_weight = nn.Parameter(torch.zeros_like(aw))
            self.affine_delta_bias = nn.Parameter(torch.zeros_like(ab))

    def extra_repr(self) -> str:
        return (
            f"target={self.spec.weight_name}, bases={self.num_bases}, rank={self.rank}, "
            f"scale={self.scale}, train_affine={self.train_affine}"
        )

    def zero_state(self) -> None:
        self.coeff_state.zero_()

    def set_state(self, coeff: torch.Tensor) -> None:
        self.coeff_state.copy_(coeff.detach())

    def carry_state(self, decay: float = 1.0) -> None:
        self.coeff_state.mul_(float(decay))

    def delta_from_coeff(self, coeff: torch.Tensor) -> torch.Tensor:
        """Return weight delta with shape matching stem weight."""
        if coeff.ndim != 1 or coeff.numel() != self.num_bases:
            raise ValueError(f"Expected coeff shape [{self.num_bases}], got {tuple(coeff.shape)}")
        mats = torch.matmul(self.lora_B, self.lora_A)  # [M, out, in]
        delta_flat = torch.tensordot(coeff, mats, dims=([0], [0]))  # [out, in]
        return delta_flat.view(*self.weight_shape)

    def compose_weight(self, coeff: Optional[torch.Tensor] = None) -> torch.Tensor:
        if coeff is None:
            coeff = self.coeff_state
        return self.w0 + self.scale * self.delta_from_coeff(coeff)

    def functional_params(self, coeff: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        params = {self.spec.weight_name: self.compose_weight(coeff)}
        if self.affine_delta_weight is not None and self.affine_delta_bias is not None:
            params[self.spec.affine_weight_name] = self.affine_w0 + self.affine_delta_weight
            params[self.spec.affine_bias_name] = self.affine_b0 + self.affine_delta_bias
        return params

    def forward_logits(self, model: nn.Module, images: torch.Tensor, coeff: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward full model logits with functional stem replacement."""
        return _functional_call(model, self.functional_params(coeff), (images,))

    def forward_stem_feature(
        self,
        images: torch.Tensor,
        coeff: Optional[torch.Tensor] = None,
        pooled_size: int = 4,
    ) -> torch.Tensor:
        """Forward only stem conv using composed weight and return pooled feature."""
        module = self.spec.module
        weight = self.compose_weight(coeff)
        x = F.conv2d(
            images.contiguous(),
            weight,
            bias=module.bias,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
        )
        x = F.adaptive_avg_pool2d(x, (pooled_size, pooled_size))
        return x.flatten(1)

    def forward_stem_spatial(
        self,
        images: torch.Tensor,
        coeff: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward only stem conv using composed weight, returning raw spatial map."""
        module = self.spec.module
        weight = self.compose_weight(coeff)
        x = F.conv2d(
            images.contiguous(),
            weight,
            bias=module.bias,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            groups=module.groups,
        )
        return x

    @torch.no_grad()
    def apply_weight_to_model_(self, model: nn.Module, coeff: Optional[torch.Tensor] = None) -> None:
        """In-place helper for debugging / non-functional usage."""
        named_params = dict(model.named_parameters())
        named_params[self.spec.weight_name].copy_(self.compose_weight(coeff))
        if self.affine_delta_weight is not None and self.affine_delta_bias is not None:
            named_params[self.spec.affine_weight_name].copy_(self.affine_w0 + self.affine_delta_weight)
            named_params[self.spec.affine_bias_name].copy_(self.affine_b0 + self.affine_delta_bias)

    def coeff_norm(self, coeff: Optional[torch.Tensor] = None) -> torch.Tensor:
        if coeff is None:
            coeff = self.coeff_state
        return coeff.norm()


def build_stem_adapter(
    model: nn.Module,
    num_bases: int = 4,
    rank: int = 4,
    scale: float = 1.0,
    target: str = "auto",
    init_std: float = 1e-3,
    train_affine: bool = False,
) -> StemLoRAAdapter:
    """Factory wrapper for explicit callsites."""
    return StemLoRAAdapter(
        model=model,
        target=target,
        num_bases=num_bases,
        rank=rank,
        scale=scale,
        init_std=init_std,
        train_affine=train_affine,
    )

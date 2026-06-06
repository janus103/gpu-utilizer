# pv_utils.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import io
import random
import torch
import torchvision.transforms as T
from PIL import Image


# -----------------------
# Augmentations (offline)
# -----------------------

class RandomJPEG:
    """
    Offline-only JPEG compression artifact.
    """
    def __init__(self, quality_min: int = 10, quality_max: int = 60, p: float = 0.5):
        self.quality_min = quality_min
        self.quality_max = quality_max
        self.p = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        q = random.randint(self.quality_min, self.quality_max)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=q)
        buf.seek(0)
        return Image.open(buf).convert("RGB")


def build_photometric_aug_family(name: str) -> Callable:
    """
    Returns a torchvision/PIL transform for a given family.
    """
    if name == "color":
        return T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
    if name == "blur":
        return T.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))
    if name == "noise":
        # implement as tensor-space noise later (keep here as identity; use add_noise in tensor space)
        return T.Lambda(lambda img: img)
    if name == "jpeg":
        return RandomJPEG(quality_min=10, quality_max=60, p=1.0)
    raise ValueError(f"Unknown aug family: {name}")


def add_tensor_noise(x: torch.Tensor, sigma: float = 0.03) -> torch.Tensor:
    """
    x: float tensor in [0,1] (or normalized - not ideal). Use before normalize if possible.
    """
    return x + sigma * torch.randn_like(x)


# -----------------------
# V computation utilities
# -----------------------

@torch.no_grad()
def upper_tri_values(M: torch.Tensor, exclude_diag: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    M: (C,C)
    return (vals, iu, ju) for upper triangle (i<j)
    """
    C = M.size(0)
    offset = 1 if exclude_diag else 0
    iu, ju = torch.triu_indices(C, C, offset=offset, device=M.device)
    vals = M[iu, ju]
    return vals, iu, ju


@torch.no_grad()
def kmeans_1d(vals: torch.Tensor, k: int = 2, iters: int = 30, seed: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Simple 1D k-means in torch.
    vals: (N,) float
    returns:
      centers: (k,)
      labels: (N,) in {0..k-1}
    """
    assert vals.dim() == 1
    torch.manual_seed(seed)

    # init centers by sampling
    idx = torch.randperm(vals.numel(), device=vals.device)[:k]
    centers = vals[idx].clone()

    for _ in range(iters):
        # assign
        d = (vals.unsqueeze(1) - centers.unsqueeze(0)).abs()
        labels = d.argmin(dim=1)
        # update
        new_centers = []
        for ci in range(k):
            mask = labels == ci
            if mask.any():
                new_centers.append(vals[mask].mean())
            else:
                new_centers.append(centers[ci])  # keep old
        new_centers = torch.stack(new_centers, dim=0)
        if torch.allclose(new_centers, centers, atol=1e-6):
            centers = new_centers
            break
        centers = new_centers
    # final assign
    d = (vals.unsqueeze(1) - centers.unsqueeze(0)).abs()
    labels = d.argmin(dim=1)
    return centers, labels


@torch.no_grad()
def make_P_from_V(
    V: torch.Tensor,
    mode: str = "kmeans",
    topk: int = 256,
    thr: Optional[float] = None,
    kmeans_iters: int = 30,
    seed: int = 0,
    soft: bool = False,
) -> torch.Tensor:
    """
    V: (C,C) sensitivity matrix (>=0)
    Returns P: (C,C) prior mask/score.

    mode:
      - 'kmeans': cluster upper-tri vals into 2 clusters; pick higher-mean cluster as style.
      - 'topk': choose topk edges by V
      - 'thr':  choose edges with V > thr
      - 'soft': normalize V to [0,1]
    If soft=True with kmeans/topk/thr: P is still binary; soft normalization can be applied separately.
    """
    C = V.size(0)
    V2 = V.clone()
    V2.fill_diagonal_(0.0)

    if mode == "soft":
        return V2 / (V2.max() + 1e-12)

    vals, iu, ju = upper_tri_values(V2, exclude_diag=True)

    if mode == "topk":
        k = min(topk, vals.numel())
        _, idx = torch.topk(vals, k=k, largest=True)
        P = torch.zeros_like(V2)
        P[iu[idx], ju[idx]] = 1.0
        P[ju[idx], iu[idx]] = 1.0
        return P

    if mode == "thr":
        if thr is None:
            raise ValueError("thr must be provided for mode='thr'")
        P = (V2 > thr).float()
        P.fill_diagonal_(0.0)
        return P

    if mode == "kmeans":
        centers, labels = kmeans_1d(vals.float(), k=2, iters=kmeans_iters, seed=seed)
        style_cluster = centers.argmax().item()  # higher-mean cluster = more sensitive
        idx = (labels == style_cluster).nonzero(as_tuple=False).squeeze(1)
        P = torch.zeros_like(V2)
        P[iu[idx], ju[idx]] = 1.0
        P[ju[idx], iu[idx]] = 1.0
        if soft:
            # optional: weight by normalized V while keeping sparsity pattern
            P = P * (V2 / (V2.max() + 1e-12))
        return P

    raise ValueError(f"Unknown mode: {mode}")


@dataclass
class VAccumulator:
    """
    Accumulate dataset mean of V where:
      V_batch = mean_k (R(aug_k) - R(orig))^2
    """
    running_V: Optional[torch.Tensor] = None
    n: int = 0

    @torch.no_grad()
    def update(self, R0: torch.Tensor, R_aug_list: List[torch.Tensor]):
        """
        R0: (B,C,C)
        R_aug_list: list of (B,C,C)
        """
        diffs = [(Ra - R0) ** 2 for Ra in R_aug_list]
        Vb = torch.stack(diffs, dim=0).mean(dim=0)  # (B,C,C)
        V_mean = Vb.mean(dim=0)                    # (C,C)
        B = R0.size(0)

        if self.running_V is None:
            self.running_V = V_mean.clone().float()
            self.n = B
        else:
            n_new = self.n + B
            self.running_V = (self.running_V * self.n + V_mean.float() * B) / float(n_new)
            self.n = n_new
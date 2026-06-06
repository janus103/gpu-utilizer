# example_calibrate.py
from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Optional

import torch
import timm

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from timm.data import resolve_data_config, create_transform

import torchvision.transforms as T

from stat_utils import attach_hook, compute_batch_stats_from_captured, running_update
from pv_utils import (
    build_photometric_aug_family,
    add_tensor_noise,
    VAccumulator,
    make_P_from_V,
)


def build_base_transform(model) -> T.Compose:
    """
    timm val transform (resize/crop/normalize).
    We'll also need a "pre-normalize" transform for noise if you want to inject noise before normalize;
    to keep code simple, we inject noise after ToTensor but before Normalize using a wrapper below.
    """
    cfg = resolve_data_config({}, model=model)
    return create_transform(**cfg, is_training=False)


def split_transform(transform: T.Compose):
    """
    Split timm transform into:
      - pre_tensor (PIL -> PIL-ish)
      - to_tensor (PIL -> Tensor)
      - post_tensor (Tensor -> normalized Tensor)

    timm create_transform returns Compose that usually includes Resize/Crop/ToTensor/Normalize.
    We'll parse it crudely by type.
    """
    pre = []
    to_tensor = None
    post = []
    for tr in transform.transforms:
        name = tr.__class__.__name__.lower()
        if "totensor" in name:
            to_tensor = tr
        elif to_tensor is None:
            pre.append(tr)
        else:
            post.append(tr)
    if to_tensor is None:
        raise RuntimeError("Could not find ToTensor in transform; cannot split.")
    return T.Compose(pre), to_tensor, T.Compose(post)


def make_augmented_view_pil(img_pil, family_names: List[str]):
    """
    Apply a random choice of photometric families in PIL space.
    You can extend this to compose multiple families if desired.
    """
    out = img_pil
    for fam in family_names:
        out = build_photometric_aug_family(fam)(out)
    return out


@torch.no_grad()
def calibrate(
    model_name: str,
    pretrained: bool,
    data_dir: str,
    batch_size: int,
    workers: int,
    device: str,
    hook_path: str,
    drop_cls: bool,
    max_images: int,
    eps: float,
    amp: bool,
    pv_families: List[str],
    pv_k: int,
    pv_mode: str,
    pv_topk: int,
    pv_thr: Optional[float],
    pv_kmeans_iters: int,
    pv_seed: int,
    pv_soft_weight: bool,
    out_path: str,
):
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=1000)
    model.eval().to(device)

    # Hook
    cap, handle = attach_hook(model, hook_path)

    # Data
    base_tfm = build_base_transform(model)
    pre_tfm, to_tensor, post_tfm = split_transform(base_tfm)

    dataset = ImageFolder(root=data_dir, transform=pre_tfm)  # PIL output after resize/crop, before ToTensor/Normalize
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True, drop_last=False)

    # Running stats
    mu_s = None
    var_s = None
    R_s = None
    n = 0

    # V accumulator
    vacc = VAccumulator()

    autocast = torch.cuda.amp.autocast if (amp and device.type == "cuda") else torch.cpu.amp.autocast
    dtype = torch.float16 if device.type == "cuda" else torch.bfloat16

    seen = 0
    for batch in loader:
        imgs_pil, _ = batch  # imgs_pil is a list of PIL images or tensors depending on ImageFolder pipeline
        # ImageFolder + pre_tfm returns PIL transformed to PIL; but DataLoader will give list-like objects
        # Ensure a list of PIL images:
        if isinstance(imgs_pil, torch.Tensor):
            raise RuntimeError("Unexpected tensor from dataset; expected PIL list. Check transform splitting.")

        # Build original tensor batch (ToTensor -> (optional noise) -> Normalize)
        x_list = []
        for im in imgs_pil:
            t = to_tensor(im)   # [0,1]
            # Note: If you want noise as a PV family, we will handle it per-aug-view below instead.
            t = post_tfm(t)
            x_list.append(t)
        x0 = torch.stack(x_list, dim=0).to(device, non_blocking=True)  # (B,3,H,W)

        # Forward original
        cap.tensor = None
        with autocast(enabled=(amp and device.type == "cuda"), dtype=dtype):
            _ = model(x0)
        if cap.tensor is None:
            raise RuntimeError("Hook did not capture tensor. Check hook_path.")
        mu_b, var_b, R_b = compute_batch_stats_from_captured(cap.tensor, eps=eps, drop_cls=drop_cls)
        B = x0.size(0)

        # Update running mu/var/R with batch means
        mu_mean = mu_b.mean(dim=0, keepdim=True)      # (1,C,1)
        var_mean = var_b.mean(dim=0, keepdim=True)    # (1,C,1)
        R_mean = R_b.mean(dim=0)                      # (C,C)

        mu_s, n_mu = running_update(mu_s, mu_mean, n, B)
        var_s, _   = running_update(var_s, var_mean, n, B)
        R_s, _     = running_update(R_s, R_mean, n, B)
        n = n_mu

        # --- Build augmented views for V computation
        R_aug_list = []
        for _k in range(pv_k):
            xk_list = []
            for im in imgs_pil:
                imk = make_augmented_view_pil(im, [torch.choice(torch.tensor([0]))] if False else pv_families)
                # The above line is a placeholder; we apply all families in pv_families.
                # If you want random family per view, change make_augmented_view_pil to sample.
                tk = to_tensor(imk)  # [0,1]
                if "noise" in pv_families:
                    tk = add_tensor_noise(tk, sigma=0.03)
                tk = post_tfm(tk)
                xk_list.append(tk)
            xk = torch.stack(xk_list, dim=0).to(device, non_blocking=True)

            cap.tensor = None
            with autocast(enabled=(amp and device.type == "cuda"), dtype=dtype):
                _ = model(xk)
            if cap.tensor is None:
                raise RuntimeError("Hook did not capture tensor for augmented view.")
            _, _, Rk = compute_batch_stats_from_captured(cap.tensor, eps=eps, drop_cls=drop_cls)
            R_aug_list.append(Rk)

        vacc.update(R_b, R_aug_list)

        seen += B
        if max_images > 0 and seen >= max_images:
            break

    handle.remove()

    # Symmetrize R_s, V_s
    R_s = 0.5 * (R_s + R_s.t())
    V_s = vacc.running_V
    V_s = 0.5 * (V_s + V_s.t())
    V_s.fill_diagonal_(0.0)

    # Prior mask P from V_s
    P = make_P_from_V(
        V_s,
        mode=pv_mode,
        topk=pv_topk,
        thr=pv_thr,
        kmeans_iters=pv_kmeans_iters,
        seed=pv_seed,
        soft=pv_soft_weight,
    )

    out = {
        "model_name": model_name,
        "hook_path": hook_path,
        "drop_cls": drop_cls,
        "n": n,
        "mu_s": mu_s,     # (1,C,1)
        "var_s": var_s,   # (1,C,1)
        "R_s": R_s,       # (C,C)
        "V_s": V_s,       # (C,C)
        "P": P,           # (C,C)
        "pv": {
            "families": pv_families,
            "k": pv_k,
            "mode": pv_mode,
            "topk": pv_topk,
            "thr": pv_thr,
            "kmeans_iters": pv_kmeans_iters,
            "seed": pv_seed,
            "soft_weight": pv_soft_weight,
        }
    }

    out_path = str(Path(out_path).resolve())
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_path)

    # quick sanity
    diag = torch.diag(R_s)
    print(f"[saved] {out_path}")
    print(f" n={n}  C={R_s.size(0)}")
    print(f" R_s diag mean={diag.mean().item():.4g} std={diag.std().item():.4g}")
    print(f" V_s mean={V_s.mean().item():.4g} max={V_s.max().item():.4g}")
    print(f" P nnz={(P>0).sum().item()} / {P.numel()}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=["resnet50", "vit_base_patch16_224"])
    p.add_argument("--pretrained", action="store_true")
    p.add_argument("--data-dir", required=True, help="ImageFolder root, e.g., imagenet/val")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--amp", action="store_true")

    # Hook control
    p.add_argument("--hook", type=str, default="", help="Module path for hook, e.g., 'maxpool', 'patch_embed', 'layer1.0.conv1'")
    p.add_argument("--drop-cls", action="store_true", help="If hook output includes CLS token, drop it when computing stats.")

    p.add_argument("--max-images", type=int, default=0)
    p.add_argument("--eps", type=float, default=1e-5)

    # PV/P controls
    p.add_argument("--pv-families", type=str, default="color,blur,jpeg", help="comma-separated: color,blur,noise,jpeg")
    p.add_argument("--pv-k", type=int, default=4, help="number of augmented views per sample (offline)")
    p.add_argument("--pv-mode", type=str, default="kmeans", choices=["kmeans", "topk", "thr", "soft"])
    p.add_argument("--pv-topk", type=int, default=256)
    p.add_argument("--pv-thr", type=float, default=None)
    p.add_argument("--pv-kmeans-iters", type=int, default=30)
    p.add_argument("--pv-seed", type=int, default=0)
    p.add_argument("--pv-soft-weight", action="store_true", help="For kmeans: weight selected edges by normalized V.")

    p.add_argument("--out", type=str, default="artifacts/calib.pt")
    args = p.parse_args()

    if args.hook == "":
        # sensible defaults
        if args.model == "resnet50":
            args.hook = "maxpool"
        else:
            args.hook = "patch_embed"

    pv_families = [s.strip() for s in args.pv_families.split(",") if s.strip()]

    calibrate(
        model_name=args.model,
        pretrained=args.pretrained,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        workers=args.workers,
        device=args.device,
        hook_path=args.hook,
        drop_cls=args.drop_cls,
        max_images=args.max_images,
        eps=args.eps,
        amp=args.amp,
        pv_families=pv_families,
        pv_k=args.pv_k,
        pv_mode=args.pv_mode,
        pv_topk=args.pv_topk,
        pv_thr=args.pv_thr,
        pv_kmeans_iters=args.pv_kmeans_iters,
        pv_seed=args.pv_seed,
        pv_soft_weight=args.pv_soft_weight,
        out_path=args.out,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Phase-1 Direct TTA for ViT-B/16: ImageNet-C evaluation.

Uses Phase-1's DirectAugClassifier (no FSC, no Phase-2 meta-learning).
Adapts patch_embed.proj.weight to filter corruption at test time.

Loss terms (configurable weights):
    A (--w-clean):   P(clean) maximization   — "classify as clean"
    B (--w-entropy): Entropy maximization     — "erase augmentation signal"
    C (--w-dist):    dist_output minimization — "reduce corruption magnitude"
    D (--w-znorm):   z-norm minimization      — "pull toward origin (clean center)"
    E (--w-repel):   Repel from original      — "push away from corrupted features"

TTA modes:
    static:   reset patch_embed.proj to pretrained each mini-batch (independent).
    continue: carry adapted patch_embed.proj across batches (cumulative).

Usage:
    python validate_phase1_tta_vit.py \\
        --imagenet-c-dir /path/to/ImageNet-C \\
        --imagenet-val-dir /path/to/imagenet/val \\
        --aug-ckpt ./output3/phase1_vitb_direct_fanout_warmrestart/best.pth.tar \\
        --initial-checkpoint ./ZOA_WEIGHT/ZOA_vit_base_timm_format.pth \\
        --tta-mode static --inner-lr 0.005 --inner-steps 5 \\
        --w-clean 1.0
"""

import argparse
import copy
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from timm.data import (
    create_transform,
    resolve_data_config,
    get_augmix_sl_num_transforms,
    AUGMIX_SL_V2_NUM_TRANSFORMS,
)
from timm.models import create_model
from timm.utils import AverageMeter, setup_default_logging

_logger = logging.getLogger("validate_phase1_tta_vit")

DEFAULT_CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog", "brightness", "contrast",
    "elastic_transform", "pixelate", "jpeg_compression",
]

# ViT-B/16 patch_embed.proj → [B, 768, 14, 14] → 2 DW stages → 4×4
STEM_CHANNELS = 768
NUM_DW_STAGES = 2


# =============================================================================
# DirectAugClassifier (inline — matches train_phase1_vit.py)
# =============================================================================

class DirectAugClassifier(nn.Module):
    """Reference-free augmentation classifier.

    Architecture:
        stem features → InstanceNorm → DW stride stages → (normed − clean_ref) → PW → MLP
    """

    def __init__(
        self,
        stem_channels: int = STEM_CHANNELS,
        output_channels: int = 64,
        num_dw_stages: int = NUM_DW_STAGES,
        num_transforms: int = AUGMIX_SL_V2_NUM_TRANSFORMS,
        hidden_dims: list = None,
        dropout: float = 0.1,
        dw_init_mode: str = 'fan_in',
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256, 128]

        self.stem_channels = stem_channels
        self.output_channels = output_channels
        self.num_dw_stages = num_dw_stages
        self.num_transforms = num_transforms
        self._dw_init_mode = dw_init_mode
        self.num_classes = num_transforms + 1
        self.feature_dim = output_channels * 4 * 4

        self.inst_norm = nn.InstanceNorm2d(stem_channels, affine=True)

        dw_layers = []
        for _ in range(num_dw_stages):
            dw_layers.extend([
                nn.Conv2d(
                    stem_channels, stem_channels, 3,
                    stride=2, padding=1, groups=stem_channels,
                ),
                nn.ReLU(inplace=True),
            ])
        self.dw_stages = nn.Sequential(*dw_layers)

        self.clean_ref = nn.Parameter(torch.zeros(1, stem_channels, 4, 4))
        self.pw_conv = nn.Conv2d(stem_channels, output_channels, 1)
        self.log_r = nn.Parameter(torch.full((num_transforms,), 3.0))

        encoder_input_dim = self.feature_dim + 1
        layers = []
        in_dim = encoder_input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim
        self.shared_encoder = nn.Sequential(*layers)

        self.aug_head = nn.Linear(hidden_dims[-1], self.num_classes)
        self.dist_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], 1),
            nn.Softplus(),
        )

    def encode(self, features_spatial: torch.Tensor) -> torch.Tensor:
        normed = self.inst_norm(features_spatial)
        reduced = self.dw_stages(normed)
        diff = reduced - self.clean_ref
        z = self.pw_conv(diff)
        return z.flatten(1)

    def forward(self, z_flat: torch.Tensor):
        dist = z_flat.norm(dim=1, keepdim=True)
        x = torch.cat([z_flat, dist], dim=1)
        shared = self.shared_encoder(x)
        aug_output = self.aug_head(shared)
        dist_output = self.dist_head(shared).squeeze(-1)
        return aug_output, dist_output


# =============================================================================
# Dataset
# =============================================================================

class ImageNetCDataset(Dataset):
    """ImageNet-C folder that may contain class subdirs or flat files."""

    def __init__(self, root: Path, filename_label_map: Dict[str, int],
                 class_to_idx: Dict[str, int], transform) -> None:
        self.root = root
        self.transform = transform

        entries = list(root.iterdir())
        has_class_dirs = any(p.is_dir() for p in entries)
        kept_paths: List[Path] = []
        targets: List[int] = []
        patterns = ("*.JPEG", "*.jpeg", "*.jpg", "*.png")

        if has_class_dirs:
            for class_dir in sorted(p for p in entries if p.is_dir()):
                target = class_to_idx.get(class_dir.name)
                if target is None:
                    continue
                for pattern in patterns:
                    for path in sorted(class_dir.glob(pattern)):
                        kept_paths.append(path)
                        targets.append(target)
        else:
            paths: List[Path] = []
            for pattern in patterns:
                paths.extend(sorted(root.glob(pattern)))
            for path in paths:
                target = filename_label_map.get(path.name)
                if target is None:
                    continue
                kept_paths.append(path)
                targets.append(target)

        if not kept_paths:
            raise RuntimeError(f"No usable images found in {root}")
        self.paths = kept_paths
        self.targets = targets

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        path = self.paths[index]
        target = self.targets[index]
        img = Image.open(path).convert("RGB")
        return self.transform(img), target


def _build_label_map(val_dir: Path) -> Tuple[Dict[str, int], Dict[str, int]]:
    from torchvision import datasets
    val_dataset = datasets.ImageFolder(val_dir)
    filename_label_map = {
        Path(p).name: target for p, target in val_dataset.samples
    }
    class_to_idx = val_dataset.class_to_idx
    _logger.info("Loaded label map for %d validation images", len(filename_label_map))
    return filename_label_map, class_to_idx


# =============================================================================
# Stem layer accessor
# =============================================================================

def _get_stem_layer(model: nn.Module) -> nn.Conv2d:
    """Return patch_embed.proj Conv2d from a ViT model."""
    if hasattr(model, 'patch_embed') and hasattr(model.patch_embed, 'proj'):
        return model.patch_embed.proj
    raise ValueError("Model does not have patch_embed.proj")


def _stem_forward(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    """Raw patch_embed.proj output [B, 768, 14, 14] without reshaping."""
    return _get_stem_layer(model)(images)


# =============================================================================
# TTA Inner Loop
# =============================================================================

def phase1_tta_inner_loop(
    images: torch.Tensor,
    model: nn.Module,
    aug_classifier: DirectAugClassifier,
    inner_lr: float,
    inner_steps: int,
    w_clean: float = 1.0,
    w_entropy: float = 0.0,
    w_dist: float = 0.0,
    w_znorm: float = 0.0,
    w_repel: float = 0.0,
    momentum: float = 0.0,
    lam: float = 0.0,
    clip_grad: float = 0.0,
) -> dict:
    """Adapt patch_embed.proj.weight using Phase-1 DirectAugClassifier signal.

    Returns diagnostics dict and stores ``original_weight`` for reset.
    """
    device = images.device
    stem_layer = _get_stem_layer(model)

    original_weight = stem_layer.weight.data.clone()

    for p in model.parameters():
        p.requires_grad = False
    stem_layer.weight.requires_grad = True

    with torch.no_grad():
        z_orig = aug_classifier.encode(_stem_forward(model, images))

    optimizer = torch.optim.SGD(
        [stem_layer.weight], lr=inner_lr, momentum=momentum,
    )

    # Diagnostics: before
    with torch.no_grad():
        z_0 = aug_classifier.encode(_stem_forward(model, images))
        aug_out_0, dist_out_0 = aug_classifier(z_0)
        probs_0 = F.softmax(aug_out_0, dim=1)
        p_clean_before = probs_0[:, 0].mean().item()
        entropy_before = -(probs_0 * torch.log(probs_0 + 1e-10)).sum(1).mean().item()
        znorm_before = z_0.norm(dim=1).mean().item()
        dist_before = dist_out_0.mean().item()

    # Inner loop
    num_updates = 0
    for _k in range(inner_steps):
        optimizer.zero_grad()

        features = _stem_forward(model, images)
        z = aug_classifier.encode(features)
        aug_output, dist_output = aug_classifier(z)

        loss = torch.tensor(0.0, device=device, requires_grad=True)

        if w_clean > 0:
            loss = loss + w_clean * (-F.log_softmax(aug_output, dim=1)[:, 0].mean())

        if w_entropy > 0:
            probs = F.softmax(aug_output, dim=1)
            ent = -(probs * F.log_softmax(aug_output, dim=1)).sum(1).mean()
            loss = loss + w_entropy * (-ent)

        if w_dist > 0:
            loss = loss + w_dist * dist_output.mean()

        if w_znorm > 0:
            loss = loss + w_znorm * z.norm(dim=1).mean()

        if w_repel > 0:
            loss = loss + w_repel * F.cosine_similarity(z, z_orig, dim=1).mean()

        if lam > 0:
            loss = loss + lam * (stem_layer.weight - original_weight).pow(2).sum()

        loss.backward()

        if clip_grad > 0:
            torch.nn.utils.clip_grad_norm_([stem_layer.weight], clip_grad)

        optimizer.step()
        num_updates += 1

    # Diagnostics: after
    with torch.no_grad():
        z_k = aug_classifier.encode(_stem_forward(model, images))
        aug_out_k, dist_out_k = aug_classifier(z_k)
        probs_k = F.softmax(aug_out_k, dim=1)
        p_clean_after = probs_k[:, 0].mean().item()
        entropy_after = -(probs_k * torch.log(probs_k + 1e-10)).sum(1).mean().item()
        znorm_after = z_k.norm(dim=1).mean().item()
        dist_after = dist_out_k.mean().item()

    param_delta = (stem_layer.weight.data - original_weight).norm().item()

    return {
        'original_weight': original_weight,
        'num_updates': num_updates,
        'p_clean_before': p_clean_before,
        'p_clean_after': p_clean_after,
        'entropy_before': entropy_before,
        'entropy_after': entropy_after,
        'znorm_before': znorm_before,
        'znorm_after': znorm_after,
        'dist_before': dist_before,
        'dist_after': dist_after,
        'param_delta_norm': param_delta,
    }


# =============================================================================
# Evaluation
# =============================================================================

@torch.no_grad()
def _eval_batch(images: torch.Tensor, target: torch.Tensor,
                model: nn.Module) -> Tuple[float, float]:
    output = model(images)
    _, pred = output.topk(5, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    n = images.size(0)
    top1 = correct[:1].reshape(-1).float().sum(0).item() / n * 100
    top5 = correct[:5].reshape(-1).float().sum(0).item() / n * 100
    return top1, top5


def _evaluate_corruption(
    model: nn.Module,
    loader: DataLoader,
    aug_classifier: DirectAugClassifier,
    pretrained_state: Dict[str, torch.Tensor],
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[float, float, float, float, str]:
    pre_top1_m = AverageMeter()
    pre_top5_m = AverageMeter()
    top1_m = AverageMeter()
    top5_m = AverageMeter()

    agg = {k: 0.0 for k in [
        'p_clean_before', 'p_clean_after', 'entropy_before', 'entropy_after',
        'znorm_before', 'znorm_after', 'dist_before', 'dist_after',
        'param_delta_norm',
    ]}
    total_batches = 0

    for batch_idx, (images, target) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        if args.tta_mode == "static" or batch_idx == 0:
            model.load_state_dict(pretrained_state)
        model.eval()

        pre_t1, pre_t5 = _eval_batch(images, target, model)
        pre_top1_m.update(pre_t1, images.size(0))
        pre_top5_m.update(pre_t5, images.size(0))

        diag = phase1_tta_inner_loop(
            images, model, aug_classifier,
            inner_lr=args.inner_lr,
            inner_steps=args.inner_steps,
            w_clean=args.w_clean,
            w_entropy=args.w_entropy,
            w_dist=args.w_dist,
            w_znorm=args.w_znorm,
            w_repel=args.w_repel,
            momentum=args.inner_momentum,
            lam=args.lam,
            clip_grad=args.clip_grad,
        )

        for k in agg:
            agg[k] += diag[k]
        total_batches += 1

        model.eval()
        ada_t1, ada_t5 = _eval_batch(images, target, model)
        top1_m.update(ada_t1, images.size(0))
        top5_m.update(ada_t5, images.size(0))

        if batch_idx % 20 == 0 or batch_idx == len(loader) - 1:
            _logger.info(
                "  Batch %d: pre=%.2f%% → ada=%.2f%% | "
                "p_clean=%.4f→%.4f, znorm=%.1f→%.1f, dist=%.1f→%.1f",
                batch_idx, pre_t1, ada_t1,
                diag['p_clean_before'], diag['p_clean_after'],
                diag['znorm_before'], diag['znorm_after'],
                diag['dist_before'], diag['dist_after'],
            )

        if args.tta_mode == "static":
            model.load_state_dict(pretrained_state)

    avg = {k: v / max(total_batches, 1) for k, v in agg.items()}
    desc = (
        f"mode={args.tta_mode}, "
        f"p_clean={avg['p_clean_before']:.4f}→{avg['p_clean_after']:.4f}, "
        f"entropy={avg['entropy_before']:.4f}→{avg['entropy_after']:.4f}, "
        f"znorm={avg['znorm_before']:.1f}→{avg['znorm_after']:.1f}, "
        f"dist={avg['dist_before']:.1f}→{avg['dist_after']:.1f}, "
        f"param_delta={avg['param_delta_norm']:.6f}"
    )
    return pre_top1_m.avg, pre_top5_m.avg, top1_m.avg, top5_m.avg, desc


# =============================================================================
# Results I/O
# =============================================================================

def _generate_results_filename(args: argparse.Namespace) -> str:
    parts = ["p1tta", args.tta_mode, f"lr{args.inner_lr}", f"s{args.inner_steps}"]
    for name, val in [("c", args.w_clean), ("e", args.w_entropy),
                      ("d", args.w_dist), ("z", args.w_znorm), ("r", args.w_repel)]:
        if val > 0:
            parts.append(f"{name}{val}")
    if args.lam > 0:
        parts.append(f"lam{args.lam}")
    if args.clip_grad > 0:
        parts.append(f"cg{args.clip_grad}")
    if args.inner_momentum > 0:
        parts.append(f"mom{args.inner_momentum}")
    parts.append(f"sev{args.severity}")
    return "_".join(parts) + ".txt"


def _save_results(
    results: List[Dict],
    args: argparse.Namespace,
    tta_summaries: Dict[str, str],
) -> None:
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    filename = _generate_results_filename(args)
    filepath = results_dir / filename

    with filepath.open("w") as f:
        f.write("Corruption\tPre_Top1\tPre_Top5\tAdapted_Top1\tAdapted_Top5\tDelta_Top1\n")

        sums = {'pre1': 0, 'pre5': 0, 'ada1': 0, 'ada5': 0}
        for r in results:
            delta = r["adapted_top1"] - r["pre_top1"]
            f.write(
                f"{r['corruption']}\t"
                f"{r['pre_top1']:.3f}\t{r['pre_top5']:.3f}\t"
                f"{r['adapted_top1']:.3f}\t{r['adapted_top5']:.3f}\t"
                f"{delta:+.3f}\n"
            )
            sums['pre1'] += r['pre_top1']
            sums['pre5'] += r['pre_top5']
            sums['ada1'] += r['adapted_top1']
            sums['ada5'] += r['adapted_top5']

        if results:
            n = len(results)
            md = sums['ada1'] / n - sums['pre1'] / n
            f.write(
                f"mean\t{sums['pre1']/n:.3f}\t{sums['pre5']/n:.3f}\t"
                f"{sums['ada1']/n:.3f}\t{sums['ada5']/n:.3f}\t{md:+.3f}\n"
            )

        f.write("\n# --- Phase-1 TTA Summary per Corruption ---\n")
        for corr, desc in tta_summaries.items():
            f.write(f"# {corr}: {desc}\n")

        f.write("\n# --- Run Configuration ---\n")
        f.write(f"# model: {args.model}\n")
        f.write(f"# tta_mode: {args.tta_mode}\n")
        f.write(f"# inner_lr: {args.inner_lr}\n")
        f.write(f"# inner_steps: {args.inner_steps}\n")
        f.write(f"# inner_momentum: {args.inner_momentum}\n")
        f.write(f"# w_clean: {args.w_clean}\n")
        f.write(f"# w_entropy: {args.w_entropy}\n")
        f.write(f"# w_dist: {args.w_dist}\n")
        f.write(f"# w_znorm: {args.w_znorm}\n")
        f.write(f"# w_repel: {args.w_repel}\n")
        f.write(f"# lam: {args.lam}\n")
        f.write(f"# clip_grad: {args.clip_grad}\n")
        f.write(f"# severity: {args.severity}\n")
        f.write(f"# batch_size: {args.batch_size}\n")
        f.write(f"# aug_ckpt: {args.aug_ckpt}\n")

    _logger.info("Results saved to: %s", filepath.resolve())


# =============================================================================
# Argument Parser
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase-1 Direct TTA for ViT-B/16 on ImageNet-C",
    )

    g = parser.add_argument_group("Data")
    g.add_argument("--imagenet-c-dir", required=True, type=str)
    g.add_argument("--imagenet-val-dir", required=True, type=str)
    g.add_argument("--corruptions", nargs="+", default=None, type=str)
    g.add_argument("--severity", default=5, type=int)

    g = parser.add_argument_group("Model")
    g.add_argument("--model", default="vit_base_patch16_224", type=str)
    g.add_argument("--pretrained", action="store_true", default=True)
    g.add_argument("--initial-checkpoint", default="", type=str,
                   help="Path to model checkpoint (e.g., ZOA weights).")
    g.add_argument("--num-classes", default=1000, type=int)
    g.add_argument("--mean", nargs="+", default=None, type=float)
    g.add_argument("--std", nargs="+", default=None, type=float)
    g.add_argument("--crop-pct", default=None, type=float)
    g.add_argument("--interpolation", default="", type=str)

    g = parser.add_argument_group("Phase-1 Aug Classifier")
    g.add_argument("--aug-ckpt", required=True, type=str,
                   help="Path to Phase-1 DirectAugClassifier checkpoint.")

    g = parser.add_argument_group("TTA Configuration")
    g.add_argument("--tta-mode", default="static", type=str,
                   choices=["static", "continue"])
    g.add_argument("--inner-lr", default=0.005, type=float)
    g.add_argument("--inner-steps", default=5, type=int)
    g.add_argument("--inner-momentum", default=0.0, type=float)

    g = parser.add_argument_group("Loss Weights")
    g.add_argument("--w-clean", default=0.0, type=float,
                   help="P(clean) maximization weight.")
    g.add_argument("--w-entropy", default=0.0, type=float,
                   help="Entropy maximization weight.")
    g.add_argument("--w-dist", default=0.0, type=float,
                   help="dist_output minimization weight.")
    g.add_argument("--w-znorm", default=0.0, type=float,
                   help="z-norm minimization weight.")
    g.add_argument("--w-repel", default=0.0, type=float,
                   help="Repel from original features weight.")
    g.add_argument("--lam", default=0.0, type=float,
                   help="Trust region weight: lam * ||theta - theta_0||^2.")
    g.add_argument("--clip-grad", default=0.0, type=float,
                   help="Max gradient norm for stem weight (0 = disabled).")

    g = parser.add_argument_group("Runtime")
    g.add_argument("--batch-size", default=64, type=int)
    g.add_argument("-j", "--workers", default=8, type=int)
    g.add_argument("--device", default="cuda", type=str)

    g = parser.add_argument_group("Output")
    g.add_argument("--results-dir", default="./results_phase1_tta", type=str)

    return parser.parse_args()


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    setup_default_logging()
    args = parse_args()

    active = sum(1 for w in [args.w_clean, args.w_entropy, args.w_dist,
                             args.w_znorm, args.w_repel] if w > 0)
    if active == 0:
        raise ValueError("At least one loss weight must be > 0")

    device = torch.device(args.device)
    if torch.cuda.is_available() and args.device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    # ---- Pretrained model ----
    _logger.info("Creating pretrained %s ...", args.model)
    model_kwargs = dict(
        pretrained=args.pretrained,
        num_classes=args.num_classes,
    )
    if args.initial_checkpoint:
        model_kwargs['pretrained'] = False
        model_kwargs['checkpoint_path'] = args.initial_checkpoint
    model = create_model(args.model, **model_kwargs)
    model.to(device).eval()
    pretrained_state = copy.deepcopy(model.state_dict())

    # ---- Phase-1 DirectAugClassifier ----
    aug_ckpt_path = Path(args.aug_ckpt)
    assert aug_ckpt_path.exists(), f"Not found: {aug_ckpt_path}"
    _logger.info("Loading Phase-1 aug classifier from: %s", aug_ckpt_path)
    aug_ckpt = torch.load(aug_ckpt_path, map_location="cpu")

    train_args = aug_ckpt.get("args", {})
    hidden_dims = train_args.get("hidden_dims", [512, 256, 128])
    dropout = train_args.get("dropout", 0.1)
    dw_init_mode = train_args.get("dw_init_mode", "fan_out")

    num_transforms = get_augmix_sl_num_transforms(version=2)
    aug_classifier = DirectAugClassifier(
        stem_channels=STEM_CHANNELS,
        output_channels=64,
        num_dw_stages=NUM_DW_STAGES,
        num_transforms=num_transforms,
        hidden_dims=hidden_dims,
        dropout=dropout,
        dw_init_mode=dw_init_mode,
    )
    aug_classifier.load_state_dict(aug_ckpt["aug_classifier"])
    aug_classifier.to(device).eval()
    for p in aug_classifier.parameters():
        p.requires_grad = False
    _logger.info(
        "Phase-1 aug classifier loaded (epoch=%d, metric=%.2f, hidden=%s)",
        aug_ckpt.get("epoch", -1), aug_ckpt.get("metric", 0), hidden_dims,
    )

    # ---- Data ----
    filename_label_map, class_to_idx = _build_label_map(Path(args.imagenet_val_dir))
    data_config = resolve_data_config(vars(args), model=model)
    transform = create_transform(**data_config, is_training=False)

    # ---- Evaluate ----
    corruptions = args.corruptions or DEFAULT_CORRUPTIONS
    results: List[Dict] = []
    tta_summaries: Dict[str, str] = {}

    loss_desc = []
    for name, val in [("clean", args.w_clean), ("entropy", args.w_entropy),
                      ("dist", args.w_dist), ("znorm", args.w_znorm),
                      ("repel", args.w_repel)]:
        if val > 0:
            loss_desc.append(f"{name}={val}")

    _logger.info(
        "\n%s\n"
        "Phase-1 Direct TTA (ViT): %d corruptions, severity=%d\n"
        "  tta_mode=%s, inner_lr=%.4f, inner_steps=%d, momentum=%.2f\n"
        "  loss: %s\n"
        "%s",
        "=" * 70, len(corruptions), args.severity,
        args.tta_mode, args.inner_lr, args.inner_steps, args.inner_momentum,
        ", ".join(loss_desc), "=" * 70,
    )

    for corr_idx, corruption in enumerate(corruptions):
        corr_dir = Path(args.imagenet_c_dir) / corruption / str(args.severity)
        if not corr_dir.exists():
            corr_dir = Path(args.imagenet_c_dir) / corruption
        if not corr_dir.exists():
            _logger.warning("Skip missing corruption: %s", corruption)
            continue

        _logger.info(
            "\n[%d/%d] Evaluating: %s (severity=%d)",
            corr_idx + 1, len(corruptions), corruption, args.severity,
        )

        dataset = ImageNetCDataset(corr_dir, filename_label_map, class_to_idx, transform)
        loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True,
        )

        pre_t1, pre_t5, ada_t1, ada_t5, desc = _evaluate_corruption(
            model, loader, aug_classifier, pretrained_state, args, device,
        )

        delta = ada_t1 - pre_t1
        _logger.info(
            ">>> %s: Pre=%.3f%% → Ada=%.3f%% (delta=%+.3f%%)",
            corruption, pre_t1, ada_t1, delta,
        )
        results.append({
            "corruption": corruption,
            "pre_top1": pre_t1, "pre_top5": pre_t5,
            "adapted_top1": ada_t1, "adapted_top5": ada_t5,
        })
        tta_summaries[corruption] = desc

    _save_results(results, args, tta_summaries)

    if results:
        n = len(results)
        mp1 = sum(r["pre_top1"] for r in results) / n
        ma1 = sum(r["adapted_top1"] for r in results) / n
        _logger.info(
            "\n%s\nFinal Mean: Pre=%.3f%% → Ada=%.3f%% (delta=%+.3f%%)\n%s",
            "=" * 70, mp1, ma1, ma1 - mp1, "=" * 70,
        )


if __name__ == "__main__":
    main()

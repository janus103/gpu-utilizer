#!/usr/bin/env python3
"""
Export representative (low-entropy) samples per class + augmented variants.

Purpose:
- Scan a dataset split with a given model checkpoint
- Select per-class representative samples (default: correct prediction + min entropy)
- Save original image (mandatory) + augmentation-policy images
- Policies are defined in a YAML file and can be mixed (multiple ops applied sequentially)

Output layout (ImageFolder-style):
  <out_dir>/<split>/<class_name>/
    original.<ext>
    aug_<policy_id>.<ext>
  <out_dir>/manifest.json
"""

import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from PIL import Image

from timm.data import create_dataset, create_loader, resolve_data_config
from timm.data.aug2 import AugmentOp, NAME_TO_OP
from timm.models import create_model, load_checkpoint


@dataclass(frozen=True)
class PolicyOp:
    name: str
    level: float


@dataclass(frozen=True)
class Policy:
    policy_id: str
    ops: List[PolicyOp]


def _set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _unwrap_target(target: Any) -> Any:
    """Handle various target batch structures produced by timm loaders."""
    if isinstance(target, (list, tuple)) and len(target) > 0:
        # common case in this repo: [target, aux]
        return target[0]
    return target


def _extract_input_target(batch: Any) -> Tuple[torch.Tensor, Any]:
    """Handle PrefetchLoader (tuple len>=2) and other batch structures."""
    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        return batch[0], batch[1]
    if isinstance(batch, dict):
        x = batch.get("input", batch.get("img", batch.get("image")))
        y = batch.get("target", batch.get("label", batch.get("labels")))
        return x, y
    raise ValueError(f"Unexpected batch type: {type(batch)}")


def _entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=1)
    return -torch.sum(probs * torch.log(probs + 1e-8), dim=1)


def _tensor_to_pil(img_chw: torch.Tensor, mean: Tuple[float, ...], std: Tuple[float, ...]) -> Image.Image:
    """
    Convert a normalized CHW float tensor to a PIL RGB image.
    Expects img_chw in model space (normalized by mean/std), range typically ~N(0,1).
    """
    if img_chw.ndim != 3:
        raise ValueError(f"Expected CHW tensor, got shape {tuple(img_chw.shape)}")

    t = img_chw.detach().cpu().float()
    mean_t = torch.tensor(mean, dtype=t.dtype).view(-1, 1, 1)
    std_t = torch.tensor(std, dtype=t.dtype).view(-1, 1, 1)
    t = t * std_t + mean_t
    t = torch.clamp(t, 0.0, 1.0)
    hwc = (t.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(hwc, mode="RGB")


def _load_checkpoint_state_dict_for_inspect(path: str) -> Dict[str, torch.Tensor]:
    ckpt = torch.load(path, map_location="cpu")
    state_dict = None

    if isinstance(ckpt, dict):
        for k in ("state_dict", "model", "model_state", "model_state_dict"):
            if k in ckpt and isinstance(ckpt[k], dict):
                state_dict = ckpt[k]
                break
        if state_dict is None:
            # sometimes checkpoints are already a state_dict-like dict
            state_dict = ckpt
    elif isinstance(ckpt, (dict,)):
        state_dict = ckpt

    if not isinstance(state_dict, dict):
        raise ValueError(f"Unsupported checkpoint format at {path}")

    # Normalize keys (strip common prefixes)
    normalized: Dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        if not isinstance(k, str):
            continue
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module."):]
        if nk.startswith("model."):
            nk = nk[len("model."):]
        normalized[nk] = v
    return normalized


def _infer_num_classes_from_checkpoint(path: str) -> Optional[int]:
    sd = _load_checkpoint_state_dict_for_inspect(path)
    for key in ("fc.weight", "head.fc.weight", "classifier.weight", "head.weight"):
        w = sd.get(key, None)
        if isinstance(w, torch.Tensor) and w.ndim == 2 and w.shape[0] > 0:
            return int(w.shape[0])
    return None


def _load_policies(policy_file: str) -> List[Policy]:
    with open(policy_file, "r") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict) or "policies" not in cfg:
        raise ValueError("Policy file must be a YAML dict with top-level key: policies")

    policies_raw = cfg["policies"]
    if not isinstance(policies_raw, list) or len(policies_raw) == 0:
        raise ValueError("Policy file 'policies' must be a non-empty list")

    policies: List[Policy] = []
    seen_ids = set()
    for p in policies_raw:
        if not isinstance(p, dict):
            raise ValueError("Each policy must be a dict")
        pid = p.get("id", "")
        ops_raw = p.get("ops", None)
        if not pid or not isinstance(pid, str):
            raise ValueError("Each policy must have a non-empty string 'id'")
        if pid in seen_ids:
            raise ValueError(f"Duplicate policy id: {pid}")
        if not isinstance(ops_raw, list) or len(ops_raw) == 0:
            raise ValueError(f"Policy '{pid}' must have a non-empty 'ops' list")

        ops: List[PolicyOp] = []
        for op in ops_raw:
            if not isinstance(op, dict):
                raise ValueError(f"Policy '{pid}': each op must be a dict with keys {{name, level}}")
            name = op.get("name", "")
            level = op.get("level", None)
            if not name or not isinstance(name, str):
                raise ValueError(f"Policy '{pid}': op.name must be a non-empty string")
            if name not in NAME_TO_OP:
                raise ValueError(f"Policy '{pid}': invalid op name '{name}'. Check timm.data.aug2.NAME_TO_OP keys.")
            if level is None:
                raise ValueError(f"Policy '{pid}': op.level is required")
            ops.append(PolicyOp(name=name, level=float(level)))

        policies.append(Policy(policy_id=pid, ops=ops))
        seen_ids.add(pid)

    return policies


def _apply_policy(img: Image.Image, policy: Policy) -> Image.Image:
    out = img
    for op in policy.ops:
        aug = AugmentOp(op.name, prob=1.0, magnitude=op.level, hparams=None)
        out = aug(out)
    return out


def _build_idx_to_class_name(dataset: Any) -> Dict[int, str]:
    class_to_idx = None
    if hasattr(dataset, "reader") and hasattr(dataset.reader, "class_to_idx"):
        class_to_idx = getattr(dataset.reader, "class_to_idx")
    elif hasattr(dataset, "class_to_idx"):
        class_to_idx = getattr(dataset, "class_to_idx")

    if not isinstance(class_to_idx, dict):
        return {}
    inv = {int(v): str(k) for k, v in class_to_idx.items()}
    return inv


def main() -> None:
    ap = argparse.ArgumentParser(description="Export representative low-entropy samples + augmented variants")
    ap.add_argument("--data-dir", required=True, type=str, help="Dataset root directory")
    ap.add_argument("--dataset", default="imagenet", type=str, help="Dataset type + name (default: imagenet)")
    ap.add_argument("--split", default="train", type=str, help="Dataset split to scan/export (default: train)")

    ap.add_argument("--model", required=True, type=str, help="timm model name")
    ap.add_argument("--model-checkpoint", required=True, type=str, help="Model checkpoint path (passed to timm checkpoint_path)")
    ap.add_argument("--input-size", required=True, nargs=3, type=int, metavar=("C", "H", "W"),
                    help="Model input size as 3 ints: C H W (e.g., 3 224 224)")
    ap.add_argument("--num-classes", default=None, type=int,
                    help="Override model num_classes (default: infer from checkpoint if possible)")
    ap.add_argument("--checkpoint-strict", action="store_true", default=True,
                    help="Load checkpoint with strict=True (default: true)")
    ap.add_argument("--no-checkpoint-strict", dest="checkpoint_strict", action="store_false",
                    help="Load checkpoint with strict=False (allows missing/unexpected keys)")

    ap.add_argument("--out-dir", required=True, type=str, help="Output directory to write exported images")
    ap.add_argument("--policy-file", required=True, type=str, help="YAML file defining augmentation policies")

    ap.add_argument("--batch-size", default=64, type=int, help="Batch size for scanning (default: 64)")
    ap.add_argument("--workers", default=4, type=int, help="DataLoader workers (default: 4)")
    ap.add_argument("--seed", default=42, type=int, help="Random seed for deterministic augmentation (default: 42)")
    ap.add_argument("--device", default="cuda", type=str, help="Device: cuda or cpu (default: cuda)")
    ap.add_argument("--img-format", default="jpg", type=str, choices=["jpg", "png"], help="Output image format")
    ap.add_argument("--require-correct", action="store_true", default=True,
                    help="Require predicted class == target class when selecting representative samples (default: true)")
    ap.add_argument("--no-require-correct", dest="require_correct", action="store_false",
                    help="Disable correctness filter; select min-entropy sample per class regardless of prediction")
    ap.add_argument("--max-classes", default=0, type=int,
                    help="If >0, limit export to first N class_ids (after sorting) for quick tests")
    ap.add_argument("--copy-raw-original", action="store_true", default=False,
                    help="Also copy the original dataset file to the class folder (if filename() is available)")

    args = ap.parse_args()
    _set_seeds(args.seed)

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    policies = _load_policies(args.policy_file)

    _ensure_dir(args.out_dir)

    # Infer num_classes if not provided (helps with custom checkpoints, e.g. fc.weight shape != 1000)
    inferred_num_classes = None
    if args.num_classes is None:
        try:
            inferred_num_classes = _infer_num_classes_from_checkpoint(args.model_checkpoint)
        except Exception:
            inferred_num_classes = None
    num_classes = int(args.num_classes) if args.num_classes is not None else inferred_num_classes

    # Load model from checkpoint
    in_chans, in_h, in_w = args.input_size
    model = create_model(
        args.model,
        pretrained=False,
        in_chans=in_chans,
        num_classes=num_classes,
    )
    load_checkpoint(model, args.model_checkpoint, strict=bool(args.checkpoint_strict))
    model.to(device=device)
    model.eval()

    # Data config + dataset/loader
    data_config = resolve_data_config(
        {"input_size": tuple(args.input_size), "crop_pct": None, "interpolation": "", "mean": None, "std": None},
        model=model,
        verbose=True,
    )

    dataset = create_dataset(
        args.dataset,
        root=args.data_dir,
        split=args.split,
        is_training=False,
        download=False,
        batch_size=1,
    )

    loader = create_loader(
        dataset,
        input_size=data_config["input_size"],
        batch_size=args.batch_size,
        is_training=False,
        interpolation=data_config["interpolation"],
        mean=data_config["mean"],
        std=data_config["std"],
        num_workers=args.workers,
        persistent_workers=bool(args.workers and args.workers > 0),
        distributed=False,
        crop_pct=data_config["crop_pct"],
        pin_memory=False,
        device=device,
        use_prefetcher=True,
    )

    idx_to_class = _build_idx_to_class_name(dataset)

    # Scan: per-class minimum entropy (optionally only among correct predictions)
    best: Dict[int, Dict[str, Any]] = {}
    global_idx = 0

    with torch.no_grad():
        for batch in loader:
            x, y = _extract_input_target(batch)
            y = _unwrap_target(y)

            if not isinstance(x, torch.Tensor):
                raise ValueError(f"Unexpected input type from loader: {type(x)}")
            if not isinstance(y, torch.Tensor):
                y = torch.as_tensor(y)

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).long()

            out = model(x)
            logits = out[0] if isinstance(out, tuple) else out

            ent = _entropy_from_logits(logits)
            pred = torch.argmax(logits, dim=1)

            bs = x.shape[0]
            for i in range(bs):
                class_id = int(y[i].item())
                ok = True
                if args.require_correct:
                    ok = (int(pred[i].item()) == class_id)
                if not ok:
                    global_idx += 1
                    continue

                e = float(ent[i].item())
                prev = best.get(class_id)
                if prev is None or e < float(prev["entropy"]):
                    src = None
                    if hasattr(dataset, "filename"):
                        try:
                            src = dataset.filename(global_idx, absolute=True)
                        except Exception:
                            src = None
                    best[class_id] = {
                        "class_id": class_id,
                        "class_name": idx_to_class.get(class_id, f"class{class_id}"),
                        "dataset_index": int(global_idx),
                        "entropy": e,
                        "pred_class": int(pred[i].item()),
                        "source": src,
                        # store model-input tensor (normalized, CHW) for export without reloading
                        "input_chw": x[i].detach().cpu(),
                    }
                global_idx += 1

    class_ids = sorted(best.keys())
    if args.max_classes and args.max_classes > 0:
        class_ids = class_ids[: args.max_classes]

    split_out_dir = os.path.join(args.out_dir, args.split)
    _ensure_dir(split_out_dir)

    mean = tuple(float(m) for m in data_config["mean"])
    std = tuple(float(s) for s in data_config["std"])

    for class_id in class_ids:
        info = best[class_id]
        class_name = info["class_name"]
        class_dir = os.path.join(split_out_dir, class_name)
        _ensure_dir(class_dir)

        base_pil = _tensor_to_pil(info["input_chw"], mean=mean, std=std)

        original_name = f"original.{args.img_format}"
        original_path = os.path.join(class_dir, original_name)
        base_pil.save(original_path)

        for p in policies:
            out_img = _apply_policy(base_pil, p)
            aug_name = f"aug_{p.policy_id}.{args.img_format}"
            aug_path = os.path.join(class_dir, aug_name)
            out_img.save(aug_path)
        
        if args.copy_raw_original and info.get("source"):
            try:
                raw_src = info["source"]
                _, ext = os.path.splitext(raw_src)
                raw_dst = os.path.join(class_dir, f"original_raw{ext or '.jpg'}")
                with open(raw_src, "rb") as rf, open(raw_dst, "wb") as wf:
                    wf.write(rf.read())
            except Exception:
                pass

    print(f"[EXPORT] Exported classes: {len(class_ids)} (split={args.split})")


if __name__ == "__main__":
    main()



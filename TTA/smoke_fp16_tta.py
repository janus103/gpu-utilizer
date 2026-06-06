from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn

GPU_UTILIZER_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = GPU_UTILIZER_ROOT.parent
TTA_DIR = Path(__file__).resolve().parent
EXTERNAL_METHODS_ROOT = TTA_DIR / "external_methods"
DEFAULT_EATA_ROOT = EXTERNAL_METHODS_ROOT / "eata"
for path in (GPU_UTILIZER_ROOT, TTA_DIR, DEFAULT_EATA_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from model_factory import build_model  # noqa: E402
from profile_tta import configure_model_for_adaptation, parse_adapt_param_types  # noqa: E402

import eata  # type: ignore  # noqa: E402
import tent  # type: ignore  # noqa: E402


MODEL_CONFIGS = [
    ("resnet50", "auto", 224, "bn"),
    ("mobilenetv2_100", "timm", 224, "bn"),
    ("mobilevit_xxs", "timm", 256, "bn"),
    ("vit_base_patch16_224", "timm", 224, "ln"),
]

FIELDNAMES = [
    "model",
    "model_source",
    "algorithm",
    "precision",
    "pretrained",
    "batch_size",
    "image_size",
    "adapt_param_types",
    "trainable_param_tensors",
    "trainable_param_elements",
    "fishers",
    "status",
    "failure_reason",
    "error",
    "output_dtype",
    "output_all_finite",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test AMP FP16 TTA execution without profiling.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--pretrained", action="store_true", help="Use pretrained weights.")
    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--e-margin", type=float, default=math.log(1000) * 0.40)
    parser.add_argument("--d-margin", type=float, default=0.05)
    parser.add_argument("--fisher-alpha", type=float, default=2000.0)
    parser.add_argument("--output", default=str(GPU_UTILIZER_ROOT / "Results" / "TTA" / "fp16_smoke.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)
    torch.cuda.set_device(device)

    rows: List[Dict[str, object]] = []
    for model_name, model_source, image_size, adapt_param_types in MODEL_CONFIGS:
        for algorithm in ("baseline", "tent", "eata"):
            row = run_case(args, model_name, model_source, image_size, adapt_param_types, algorithm, device)
            rows.append(row)
            write_rows(Path(args.output), rows)
            print(
                f"{row['model']} {row['algorithm']} status={row['status']} "
                f"finite={row['output_all_finite']} dtype={row['output_dtype']} reason={row['failure_reason']}"
            )
    print(f"wrote FP16 smoke results: {Path(args.output).resolve()}")


def run_case(
    args: argparse.Namespace,
    model_name: str,
    model_source: str,
    image_size: int,
    adapt_param_types: str,
    algorithm: str,
    device: torch.device,
) -> Dict[str, object]:
    row: Dict[str, object] = {
        "model": model_name,
        "model_source": model_source,
        "algorithm": algorithm,
        "precision": "amp_fp16",
        "pretrained": args.pretrained,
        "batch_size": args.batch_size,
        "image_size": image_size,
        "adapt_param_types": "" if algorithm == "baseline" else adapt_param_types,
        "trainable_param_tensors": 0,
        "trainable_param_elements": 0,
        "fishers": 0,
        "status": "pending",
        "failure_reason": "",
        "error": "",
        "output_dtype": "",
        "output_all_finite": "",
    }
    try:
        model_info = build_model(
            model_name=model_name,
            model_source=model_source,
            pretrained=args.pretrained,
            eata_root=DEFAULT_EATA_ROOT,
        )
        model = model_info.model.to(device)
        x = torch.randn(args.batch_size, 3, image_size, image_size, device=device)
        y = torch.randn_like(x)

        if algorithm == "baseline":
            model.eval()
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
                output = model(x)
        else:
            adapt_model, param_names, fishers = build_adapt_model(
                args,
                algorithm,
                model,
                parse_adapt_param_types(adapt_param_types),
                x,
            )
            row["trainable_param_tensors"] = len(param_names)
            row["trainable_param_elements"] = sum(p.numel() for p in trainable_params(adapt_model.model))
            row["fishers"] = len(fishers)
            with torch.cuda.amp.autocast(dtype=torch.float16):
                adapt_model(x)
                with torch.no_grad():
                    output = adapt_model.model(y)

        torch.cuda.synchronize()
        row["output_dtype"] = str(output.dtype)
        row["output_all_finite"] = bool(torch.isfinite(output).all().item())
        row["status"] = "ok" if row["output_all_finite"] else "error"
        row["failure_reason"] = "" if row["output_all_finite"] else "non_finite_output"
    except Exception as exc:
        row["status"] = "error"
        row["failure_reason"] = classify_failure(exc)
        row["error"] = repr(exc)
    finally:
        torch.cuda.empty_cache()
    return row


def build_adapt_model(
    args: argparse.Namespace,
    algorithm: str,
    model: nn.Module,
    adapt_param_types: Sequence[str],
    fisher_x: torch.Tensor,
):
    model, params, param_names = configure_model_for_adaptation(model, adapt_param_types)
    if not params:
        raise RuntimeError("no adaptable parameters")
    optimizer = torch.optim.SGD(params, args.lr, momentum=args.momentum)
    if algorithm == "tent":
        return tent.Tent(model, optimizer, steps=1, episodic=False), param_names, {}

    fishers = compute_fishers_amp(model, fisher_x, param_names)
    return (
        eata.EATA(
            model,
            optimizer,
            fishers=fishers,
            fisher_alpha=args.fisher_alpha,
            steps=1,
            episodic=False,
            e_margin=args.e_margin,
            d_margin=args.d_margin,
        ),
        param_names,
        fishers,
    )


def compute_fishers_amp(
    model: nn.Module,
    x: torch.Tensor,
    param_names: Sequence[str],
) -> Dict[str, List[torch.Tensor]]:
    tracked = set(param_names)
    fishers: Dict[str, List[torch.Tensor]] = {}
    criterion = nn.CrossEntropyLoss()
    model.train()
    with torch.cuda.amp.autocast(dtype=torch.float16):
        outputs = model(x)
        targets = outputs.detach().argmax(dim=1)
        loss = criterion(outputs.float(), targets)
    loss.backward()
    for name, param in model.named_parameters():
        if name not in tracked or param.grad is None:
            continue
        fishers[name] = [param.grad.detach().clone().pow(2), param.detach().clone()]
    model.zero_grad(set_to_none=True)
    return fishers


def trainable_params(model: nn.Module) -> List[nn.Parameter]:
    return [param for param in model.parameters() if param.requires_grad]


def classify_failure(exc: Exception) -> str:
    message = repr(exc).lower()
    if "out of memory" in message:
        return "oom"
    if "nan" in message or "inf" in message:
        return "non_finite"
    return "error"


def write_rows(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List

import torch

GPU_UTILIZER_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = GPU_UTILIZER_ROOT.parent
TTA_DIR = Path(__file__).resolve().parent
EXTERNAL_METHODS_ROOT = TTA_DIR / "external_methods"
ZOA_ROOT = EXTERNAL_METHODS_ROOT / "zoa"

for path in (GPU_UTILIZER_ROOT, ZOA_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from gpu_metrics import MetricConfig, profile_callable  # noqa: E402


FIELDNAMES = [
    "model",
    "model_source",
    "precision",
    "batch_size",
    "image_size",
    "status",
    "failure_reason",
    "error",
    "output_dtype",
    "output_all_finite",
    "cuda_elapsed_ms_total",
    "latency_ms_per_iter",
    "estimated_sm_cycles_per_iter",
    "energy_j_per_iter",
    "energy_j_per_iter_est_from_power",
    "avg_power_w_from_energy",
    "avg_power_w_sampled",
    "energy_source",
    "clock_source",
    "power_source",
]


MODEL_SPECS = [
    ("resnet50", "zoa.models.resnet", 224),
    ("vit_base_patch16_224", "timm", 224),
    ("mobilevit_xxs", "timm", 256),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke test ZOA/timm models with gpu_utilizer AMP FP16 profiling.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--no-idle-check", action="store_true")
    parser.add_argument("--output", default=str(GPU_UTILIZER_ROOT / "Results" / "TTA" / "zoa_fp16_gpu_utilizer_smoke.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    device = torch.device(args.device)
    torch.cuda.set_device(device)

    metric_config = MetricConfig(
        device_index=args.gpu_index,
        allow_nvidia_smi_fallback=False,
        require_idle_gpu=not args.no_idle_check,
        allow_zero_energy_estimate=True,
    )

    rows: List[Dict[str, object]] = []
    for model_name, source, image_size in MODEL_SPECS:
        row = smoke_one(args, metric_config, model_name, source, image_size, device)
        rows.append(row)
        write_rows(Path(args.output), rows)
        print(
            f"{model_name} source={source} status={row['status']} "
            f"dtype={row['output_dtype']} finite={row['output_all_finite']} "
            f"latency_ms={row.get('latency_ms_per_iter', '')}"
        )
    print(f"wrote ZOA FP16 gpu_utilizer smoke: {Path(args.output).resolve()}")


def smoke_one(
    args: argparse.Namespace,
    metric_config: MetricConfig,
    model_name: str,
    source: str,
    image_size: int,
    device: torch.device,
) -> Dict[str, object]:
    row: Dict[str, object] = {
        "model": model_name,
        "model_source": source,
        "precision": "amp_fp16",
        "batch_size": args.batch_size,
        "image_size": image_size,
        "status": "pending",
        "failure_reason": "",
        "error": "",
        "output_dtype": "",
        "output_all_finite": "",
    }
    try:
        model = build_model(model_name, source, pretrained=args.pretrained).to(device).eval()
        x = torch.randn(args.batch_size, 3, image_size, image_size, device=device)
        output_holder: Dict[str, torch.Tensor] = {}

        def run_once():
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
                output = model(x)
            output_holder["output"] = output
            return output

        metrics = profile_callable(
            run_once,
            warmup=args.warmup,
            repeat=args.repeat,
            config=metric_config,
            nvtx_name=f"zoa_fp16_{model_name}",
        )
        output = output_holder["output"]
        row.update(metrics)
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


def build_model(model_name: str, source: str, pretrained: bool):
    if source == "zoa.models.resnet":
        import models.resnet as zoa_resnet

        if model_name != "resnet50":
            raise ValueError(f"Unsupported ZOA ResNet model: {model_name}")
        return zoa_resnet.resnet50(pretrained=pretrained)

    if source == "timm":
        import timm

        return timm.create_model(model_name, pretrained=pretrained, num_classes=1000)

    raise ValueError(f"Unsupported source: {source}")


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
        writer = csv.DictWriter(stream, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()

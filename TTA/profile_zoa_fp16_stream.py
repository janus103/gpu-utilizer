from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

GPU_UTILIZER_ROOT = Path(__file__).resolve().parents[1]
TTA_DIR = Path(__file__).resolve().parent
EXTERNAL_METHODS_ROOT = TTA_DIR / "external_methods"
ZOA_ROOT = EXTERNAL_METHODS_ROOT / "zoa"
for path in (GPU_UTILIZER_ROOT, ZOA_ROOT, TTA_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from gpu_metrics import MetricConfig, assert_gpu_idle, read_sample, require_nvml  # noqa: E402
from smoke_zoa_adapt_fp16_gpu_utilizer import (  # noqa: E402
    build_mobilevit_zoa_like,
    build_zoa_resnet,
    build_zoa_vit,
)


FIELDS = [
    "status",
    "failure_reason",
    "error",
    "model",
    "algorithm",
    "precision",
    "batch_size",
    "image_size",
    "corruption",
    "level",
    "num_samples",
    "num_batches",
    "cuda_elapsed_ms_total",
    "latency_ms_per_batch",
    "latency_ms_per_sample",
    "estimated_sm_cycles_total",
    "estimated_sm_cycles_per_sample",
    "energy_j_total",
    "energy_j_per_sample",
    "energy_j_total_est_from_power",
    "energy_j_per_sample_est_from_power",
    "avg_power_w_from_energy",
    "avg_power_w_sampled",
    "energy_source",
    "clock_source",
    "power_source",
    "model_forward_gops_per_sample",
    "forward_equiv_factor",
    "workload_gops_total",
    "gops",
    "gops_per_watt",
    "output_all_finite",
]

BUILDERS = {
    "resnet50": (build_zoa_resnet, 224, "zoa_resnet"),
    "vit_base_patch16_224": (build_zoa_vit, 224, "zoa_vit"),
    "mobilevit_xxs": (build_mobilevit_zoa_like, 256, "zoa_like_mobilevit_middle_norm"),
}
DEFAULT_FORWARD_GOPS_PER_SAMPLE = {
    "resnet50": 8.178,
    "vit_base_patch16_224": 35.200,
    "mobilevit_xxs": 0.800,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Full ImageNet-C ZOA/ZOA-like AMP FP16 stream profiler.")
    p.add_argument("--models", default="resnet50,vit_base_patch16_224,mobilevit_xxs")
    p.add_argument("--batch-sizes", default="1,2,4,8,16,32,64,128")
    p.add_argument("--data-corruption", default="/home/oem/servers/imagenet-c")
    p.add_argument("--corruption", default="gaussian_noise")
    p.add_argument("--level", type=int, default=5)
    p.add_argument("--gpu-index", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--forward-equiv-factor", type=float, default=2.0)
    p.add_argument("--max-samples", type=int, default=0, help="Debug only. 0 means full split.")
    p.add_argument("--no-idle-check", action="store_true")
    p.add_argument("--keep-going", action="store_true")
    p.add_argument("--output", default=str(GPU_UTILIZER_ROOT / "Results" / "TTA" / "zoa_fp16_stream.csv"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    metric_config = MetricConfig(
        device_index=args.gpu_index,
        allow_nvidia_smi_fallback=False,
        require_idle_gpu=not args.no_idle_check,
        allow_zero_energy_estimate=True,
    )
    require_nvml(args.gpu_index)
    assert_gpu_idle(metric_config)

    out = Path(args.output).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []
    for model in split_csv(args.models):
        for batch_size in [int(x) for x in split_csv(args.batch_sizes)]:
            row = run_case(args, metric_config, model, batch_size, device)
            rows.append(row)
            write_rows(out, rows)
            if row["status"] != "ok" and not args.keep_going:
                raise RuntimeError(row["error"])
    print(f"wrote ZOA FP16 stream results: {out}")


def run_case(args, metric_config: MetricConfig, model_name: str, batch_size: int, device) -> Dict[str, object]:
    row = base_row(args, model_name, batch_size)
    try:
        if model_name not in BUILDERS:
            raise ValueError(f"Unsupported model: {model_name}")
        builder, image_size, algorithm = BUILDERS[model_name]
        row["image_size"] = image_size
        row["algorithm"] = algorithm
        adapt_model, _ = builder(batch_size, device)
        loader = build_loader(args, batch_size, image_size)

        def run_stream():
            samples = 0
            batches = 0
            finite = True
            for images, _target in loader:
                images = images.to(device=device, non_blocking=True)
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    output = adapt_model(images)
                finite = finite and bool(torch.isfinite(output).all().item())
                samples += images.size(0)
                batches += 1
                if args.max_samples and samples >= args.max_samples:
                    break
            return samples, batches, finite

        metrics = measure_stream(run_stream, metric_config)
        row.update(metrics)
        row["output_all_finite"] = metrics.pop("output_all_finite")
        add_gops(row, args, model_name)
        row["status"] = "ok" if row["output_all_finite"] else "error"
        row["failure_reason"] = "" if row["output_all_finite"] else "non_finite_output"
        print(f"profiled {model_name} bs={batch_size} samples={row['num_samples']} status={row['status']}")
    except Exception as exc:
        row["status"] = "error"
        row["failure_reason"] = classify_failure(exc)
        row["error"] = repr(exc)
        torch.cuda.empty_cache()
        print(f"profiled {model_name} bs={batch_size} status=error reason={row['failure_reason']}")
    finally:
        torch.cuda.empty_cache()
    return row


def build_loader(args, batch_size: int, image_size: int):
    root = Path(args.data_corruption) / args.corruption / str(args.level)
    if not root.exists():
        raise FileNotFoundError(f"Input dataset directory does not exist: {root}")
    transform = transforms.Compose([
        transforms.Resize(image_size + 32) if image_size == 224 else transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = datasets.ImageFolder(str(root), transform=transform)
    if args.max_samples:
        dataset = Subset(dataset, range(min(args.max_samples, len(dataset))))
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)


def measure_stream(run_stream, config: MetricConfig) -> Dict[str, object]:
    start_sample = read_sample(config)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    wall_start = time.perf_counter()
    start_event.record()
    samples, batches, finite = run_stream()
    end_event.record()
    torch.cuda.synchronize()
    wall_end = time.perf_counter()
    end_sample = read_sample(config)
    elapsed_ms = float(start_event.elapsed_time(end_event))
    elapsed_s = elapsed_ms / 1000.0
    avg_clock = avg(start_sample.sm_clock_mhz, end_sample.sm_clock_mhz)
    cycles_total = elapsed_ms * avg_clock * 1000.0 if avg_clock is not None else None
    avg_power_sampled = avg(start_sample.power_w, end_sample.power_w)
    energy_total = None
    energy_est = None
    avg_power_from_energy = None
    energy_source = "unavailable"
    if start_sample.energy_mj is not None and end_sample.energy_mj is not None:
        delta_mj = end_sample.energy_mj - start_sample.energy_mj
        if delta_mj > 0:
            energy_total = delta_mj / 1000.0
            avg_power_from_energy = energy_total / elapsed_s if elapsed_s > 0 else None
            energy_source = "nvml_total_energy"
        elif avg_power_sampled is not None:
            energy_est = avg_power_sampled * elapsed_s
            energy_source = "power_estimate_zero_delta"
    elif avg_power_sampled is not None:
        energy_est = avg_power_sampled * elapsed_s
        energy_source = "power_estimate_no_energy_counter"
    return {
        "num_samples": samples,
        "num_batches": batches,
        "cuda_elapsed_ms_total": elapsed_ms,
        "latency_ms_per_batch": div(elapsed_ms, batches),
        "latency_ms_per_sample": div(elapsed_ms, samples),
        "estimated_sm_cycles_total": cycles_total,
        "estimated_sm_cycles_per_sample": div(cycles_total, samples),
        "energy_j_total": energy_total,
        "energy_j_per_sample": div(energy_total, samples),
        "energy_j_total_est_from_power": energy_est,
        "energy_j_per_sample_est_from_power": div(energy_est, samples),
        "avg_power_w_from_energy": avg_power_from_energy,
        "avg_power_w_sampled": avg_power_sampled,
        "energy_source": energy_source,
        "clock_source": join(start_sample.clock_source, end_sample.clock_source),
        "power_source": join(start_sample.power_source, end_sample.power_source),
        "output_all_finite": finite,
        "wall_s_total": wall_end - wall_start,
    }


def base_row(args, model: str, batch_size: int) -> Dict[str, object]:
    return {
        "status": "pending",
        "failure_reason": "",
        "error": "",
        "model": model,
        "algorithm": "",
        "precision": "amp_fp16",
        "batch_size": batch_size,
        "image_size": "",
        "corruption": args.corruption,
        "level": args.level,
        "num_samples": 0,
        "num_batches": 0,
        "model_forward_gops_per_sample": DEFAULT_FORWARD_GOPS_PER_SAMPLE.get(model, ""),
        "forward_equiv_factor": args.forward_equiv_factor,
        "workload_gops_total": "",
        "gops": "",
        "gops_per_watt": "",
        "output_all_finite": "",
    }


def add_gops(row: Dict[str, object], args, model: str) -> None:
    forward_gops = DEFAULT_FORWARD_GOPS_PER_SAMPLE.get(model)
    elapsed_ms = row.get("cuda_elapsed_ms_total")
    samples = row.get("num_samples")
    if not forward_gops or not elapsed_ms or not samples:
        return
    workload_gops = float(forward_gops) * float(samples) * float(args.forward_equiv_factor)
    elapsed_s = float(elapsed_ms) / 1000.0
    gops = workload_gops / elapsed_s if elapsed_s > 0 else None
    power = row.get("avg_power_w_from_energy") or row.get("avg_power_w_sampled")
    row["model_forward_gops_per_sample"] = forward_gops
    row["forward_equiv_factor"] = args.forward_equiv_factor
    row["workload_gops_total"] = workload_gops
    row["gops"] = gops
    row["gops_per_watt"] = (gops / float(power)) if gops is not None and power else None


def classify_failure(exc: Exception) -> str:
    msg = repr(exc).lower()
    if "out of memory" in msg:
        return "oom"
    if "does not exist" in msg:
        return "missing_data"
    if "nan" in msg or "inf" in msg:
        return "non_finite"
    return "error"


def split_csv(value: str) -> List[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def avg(a, b):
    if a is None or b is None:
        return None
    return (float(a) + float(b)) / 2.0


def div(value, divisor):
    if value is None or not divisor:
        return None
    return float(value) / float(divisor)


def join(a: str, b: str) -> str:
    return a if a == b else f"{a}->{b}"


def write_rows(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()

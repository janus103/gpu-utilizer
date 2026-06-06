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
if str(GPU_UTILIZER_ROOT) not in sys.path:
    sys.path.insert(0, str(GPU_UTILIZER_ROOT))
if str(TTA_DIR) not in sys.path:
    sys.path.insert(0, str(TTA_DIR))

from gpu_metrics import MetricConfig, assert_gpu_idle, read_sample, require_nvml  # noqa: E402
from profile_tta import DEFAULT_FORWARD_GOPS_PER_SAMPLE  # noqa: E402


FIELDS = [
    "status",
    "failure_reason",
    "error",
    "model",
    "precision",
    "mode",
    "pretrained",
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
    "avg_power_w_from_energy",
    "avg_power_w_sampled",
    "energy_source",
    "model_forward_gops_per_sample",
    "forward_equiv_factor",
    "workload_gops_total",
    "gops",
    "gops_per_watt",
]

MODEL_IMAGE_SIZE = {
    "resnet50": 224,
    "mobilevit_xxs": 256,
    "vit_base_patch16_224": 224,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Profile FP32 inference versus supervised training step.")
    p.add_argument("--models", default="resnet50,mobilevit_xxs,vit_base_patch16_224")
    p.add_argument("--batch-sizes", default="1,2,4,8,16,32,64,128")
    p.add_argument("--data-corruption", default="/home/oem/servers/imagenet-c")
    p.add_argument("--corruption", default="gaussian_noise")
    p.add_argument("--level", type=int, default=5)
    p.add_argument("--gpu-index", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--pretrained", action="store_true", default=True)
    p.add_argument("--no-pretrained", action="store_false", dest="pretrained")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--no-idle-check", action="store_true")
    p.add_argument("--keep-going", action="store_true")
    p.add_argument("--output", default=str(GPU_UTILIZER_ROOT / "Results" / "TTA" / "fp32_train_vs_infer.csv"))
    return p.parse_args()


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
    require_nvml(args.gpu_index)
    assert_gpu_idle(metric_config)

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []
    for model in split_csv(args.models):
        for batch_size in [int(x) for x in split_csv(args.batch_sizes)]:
            for mode in ("inference", "training"):
                row = run_case(args, metric_config, model, batch_size, mode, device)
                rows.append(row)
                write_rows(output, rows)
                if row["status"] != "ok" and not args.keep_going:
                    raise RuntimeError(row["error"])
    print(f"wrote FP32 train-vs-infer results: {output}")


def run_case(args, metric_config: MetricConfig, model_name: str, batch_size: int, mode: str, device) -> Dict[str, object]:
    row = base_row(args, model_name, batch_size, mode)
    try:
        import timm

        image_size = MODEL_IMAGE_SIZE[model_name]
        model = timm.create_model(model_name, pretrained=args.pretrained, num_classes=1000).to(device)
        loader = build_loader(args, batch_size, image_size)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)

        def run_stream():
            samples = 0
            batches = 0
            if mode == "inference":
                model.eval()
            else:
                model.train()
            for images, target in loader:
                images = images.to(device=device, non_blocking=True)
                target = target.to(device=device, non_blocking=True)
                if mode == "inference":
                    with torch.no_grad():
                        model(images)
                else:
                    optimizer.zero_grad(set_to_none=True)
                    output = model(images)
                    loss = criterion(output.float(), target)
                    loss.backward()
                    optimizer.step()
                samples += images.size(0)
                batches += 1
                if args.max_samples and samples >= args.max_samples:
                    break
            return samples, batches

        metrics = measure_stream(run_stream, metric_config)
        row.update(metrics)
        add_gops(row, model_name, mode)
        row["status"] = "ok"
    except Exception as exc:
        row["status"] = "error"
        row["failure_reason"] = classify_failure(exc)
        row["error"] = repr(exc)
        torch.cuda.empty_cache()
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
    start_event.record()
    wall_start = time.perf_counter()
    samples, batches = run_stream()
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
    avg_power_from_energy = None
    energy_source = "unavailable"
    if start_sample.energy_mj is not None and end_sample.energy_mj is not None:
        delta_mj = end_sample.energy_mj - start_sample.energy_mj
        if delta_mj > 0:
            energy_total = delta_mj / 1000.0
            avg_power_from_energy = energy_total / elapsed_s if elapsed_s > 0 else None
            energy_source = "nvml_total_energy"
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
        "avg_power_w_from_energy": avg_power_from_energy,
        "avg_power_w_sampled": avg_power_sampled,
        "energy_source": energy_source,
        "wall_s_total": wall_end - wall_start,
    }


def base_row(args, model: str, batch_size: int, mode: str) -> Dict[str, object]:
    return {
        "status": "pending",
        "failure_reason": "",
        "error": "",
        "model": model,
        "precision": "float32",
        "mode": mode,
        "pretrained": args.pretrained,
        "batch_size": batch_size,
        "image_size": MODEL_IMAGE_SIZE.get(model, ""),
        "corruption": args.corruption,
        "level": args.level,
        "num_samples": 0,
        "num_batches": 0,
    }


def add_gops(row: Dict[str, object], model: str, mode: str) -> None:
    forward_gops = DEFAULT_FORWARD_GOPS_PER_SAMPLE.get(model)
    elapsed_ms = row.get("cuda_elapsed_ms_total")
    samples = row.get("num_samples")
    factor = 1.0 if mode == "inference" else 3.0
    if not forward_gops or not elapsed_ms or not samples:
        return
    workload_gops = float(forward_gops) * float(samples) * factor
    elapsed_s = float(elapsed_ms) / 1000.0
    gops = workload_gops / elapsed_s if elapsed_s > 0 else None
    power = row.get("avg_power_w_from_energy") or row.get("avg_power_w_sampled")
    row["model_forward_gops_per_sample"] = forward_gops
    row["forward_equiv_factor"] = factor
    row["workload_gops_total"] = workload_gops
    row["gops"] = gops
    row["gops_per_watt"] = (gops / float(power)) if gops is not None and power else None


def classify_failure(exc: Exception) -> str:
    msg = repr(exc).lower()
    if "out of memory" in msg:
        return "oom"
    if "does not exist" in msg:
        return "missing_data"
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


def write_rows(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import csv
import math
import random
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

GPU_UTILIZER_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = GPU_UTILIZER_ROOT.parent
TTA_DIR = Path(__file__).resolve().parent
EXTERNAL_METHODS_ROOT = TTA_DIR / "external_methods"
DEFAULT_EATA_ROOT = EXTERNAL_METHODS_ROOT / "eata"
if str(GPU_UTILIZER_ROOT) not in sys.path:
    sys.path.insert(0, str(GPU_UTILIZER_ROOT))
if str(TTA_DIR) not in sys.path:
    sys.path.insert(0, str(TTA_DIR))

from gpu_metrics import MetricConfig, assert_gpu_idle, read_sample, require_nvml  # noqa: E402
from model_factory import build_model  # noqa: E402
from profile_tta import (  # noqa: E402
    DEFAULT_FORWARD_GOPS_PER_SAMPLE,
    add_gops_metrics,
    classify_failure,
    collect_trainable_params,
    configure_model_for_adaptation,
    get_forward_gops_per_sample,
    parse_adapt_param_types,
    parse_algorithms,
    preferred_power_w,
    workload_forward_equiv_factor,
)


RESULT_COLUMNS = [
    "status",
    "error",
    "failure_reason",
    "measurement",
    "algorithm",
    "model",
    "model_source",
    "pretrained",
    "requested_batch_size",
    "image_size",
    "input_mode",
    "corruption",
    "level",
    "num_samples",
    "num_batches",
    "drop_last",
    "adapt_steps",
    "repeat",
    "warmup",
    "device",
    "gpu_index",
    "dtype",
    "lr",
    "momentum",
    "episodic",
    "adapt_param_types",
    "fisher_samples",
    "fisher_param_tensors",
    "unfrozen_param_tensors",
    "unfrozen_param_elements",
    "tta_updates_reliable",
    "tta_updates_reliable_non_redundant",
    "cuda_elapsed_ms_total",
    "latency_ms_per_batch",
    "latency_ms_per_sample",
    "wall_s_total",
    "estimated_sm_cycles_total",
    "estimated_sm_cycles_per_batch",
    "estimated_sm_cycles_per_sample",
    "sm_clock_mhz_start",
    "sm_clock_mhz_end",
    "sm_clock_mhz_avg_used",
    "power_w_start",
    "power_w_end",
    "avg_power_w_sampled",
    "energy_j_total",
    "energy_j_per_batch",
    "energy_j_per_sample",
    "energy_j_total_est_from_power",
    "energy_j_per_batch_est_from_power",
    "energy_j_per_sample_est_from_power",
    "avg_power_w_from_energy",
    "energy_source",
    "clock_source",
    "power_source",
    "util_source",
    "nvml_available",
    "nvidia_smi_fallback_allowed",
    "gops_count_source",
    "model_forward_gops_per_sample",
    "workload_forward_equiv_factor",
    "estimated_workload_gops_total",
    "estimated_gops",
    "estimated_gops_per_w",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile a full ImageNet-C TTA stream once.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-source", default="auto", choices=["auto", "eata_resnet", "torchvision", "timm"])
    parser.add_argument("--eata-root", default=str(DEFAULT_EATA_ROOT))
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--no-pretrained", action="store_false", dest="pretrained")
    parser.add_argument("--algorithms", default="tent,eata")
    parser.add_argument("--no-baseline", action="store_true")
    parser.add_argument("--batch-sizes", default="1,2,4,8,16,32,64,128")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--adapt-steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--episodic", action="store_true")
    parser.add_argument("--adapt-param-types", default="bn")
    parser.add_argument("--e-margin", type=float, default=math.log(1000) * 0.40)
    parser.add_argument("--d-margin", type=float, default=0.05)
    parser.add_argument("--fisher-alpha", type=float, default=2000.0)
    parser.add_argument("--fisher-samples", type=int, default=128)
    parser.add_argument("--input-mode", default="imagenet-c", choices=["imagenet", "imagenet-c"])
    parser.add_argument("--data", default="")
    parser.add_argument("--data-corruption", default="")
    parser.add_argument("--corruption", default="gaussian_noise")
    parser.add_argument("--level", type=int, default=5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--drop-last", action="store_true", help="Skip incomplete tail batch.")
    parser.add_argument("--max-samples", type=int, default=0, help="Debug only. 0 means full split.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--dtype", default="amp_fp16", choices=["float32", "amp_fp16"])
    parser.add_argument("--forward-gops-per-sample", type=float, default=None)
    parser.add_argument("--gops-count-source", default="static_forward_gops_mac2")
    parser.add_argument("--tta-forward-equiv-factor", type=float, default=4.0)
    parser.add_argument("--eata-forward-equiv-factor", type=float, default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--nvidia-smi-fallback", action="store_true")
    parser.add_argument("--no-idle-check", action="store_true")
    parser.add_argument("--no-zero-energy-estimate", action="store_true")
    parser.add_argument("--no-cudnn-benchmark", action="store_true")
    parser.add_argument("--keep-going", action="store_true")
    parser.add_argument("--seed", type=int, default=2020)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_args(args)
    set_reproducible(args.seed)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = not args.no_cudnn_benchmark

    metric_config = MetricConfig(
        device_index=args.gpu_index,
        allow_nvidia_smi_fallback=args.nvidia_smi_fallback,
        require_idle_gpu=not args.no_idle_check,
        allow_zero_energy_estimate=not args.no_zero_energy_estimate,
    )
    if not metric_config.allow_nvidia_smi_fallback:
        require_nvml(metric_config.device_index)
    assert_gpu_idle(metric_config)

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []

    algorithms = parse_algorithms(args.algorithms)
    for batch_size in parse_batch_sizes(args.batch_sizes):
        if not args.no_baseline:
            rows.append(run_case(args, "none", batch_size, metric_config, device))
            write_results(output_path, rows)
        for algorithm in algorithms:
            rows.append(run_case(args, algorithm, batch_size, metric_config, device))
            write_results(output_path, rows)

    print(f"wrote stream TTA profile results: {output_path}")


def validate_args(args: argparse.Namespace) -> None:
    if args.input_mode == "imagenet" and not args.data:
        raise ValueError("--data is required for --input-mode imagenet.")
    if args.input_mode == "imagenet-c" and not args.data_corruption:
        raise ValueError("--data-corruption is required for --input-mode imagenet-c.")
    parse_adapt_param_types(args.adapt_param_types)
    if get_forward_gops_per_sample(args) is None:
        raise ValueError(
            f"No default GOP count for model '{args.model}'. Pass --forward-gops-per-sample."
        )


def parse_batch_sizes(value: str) -> List[int]:
    sizes = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not sizes or any(size <= 0 for size in sizes):
        raise ValueError("--batch-sizes must contain positive integers.")
    return sizes


def set_reproducible(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_case(
    args: argparse.Namespace,
    algorithm: str,
    batch_size: int,
    metric_config: MetricConfig,
    device: torch.device,
) -> Dict[str, object]:
    row = base_row(args, algorithm, batch_size)
    try:
        loader = build_loader(args, batch_size)
        model_info = build_model(
            model_name=args.model,
            model_source=args.model_source,
            pretrained=args.pretrained,
            eata_root=Path(args.eata_root),
            num_classes=args.num_classes,
        )
        model = model_info.model.to(device)
        row["model_source"] = model_info.source

        adapt_model = None
        use_amp = args.dtype == "amp_fp16"
        if algorithm == "none":
            model.eval()
            run_once = lambda: run_baseline_stream(model, loader, device, args.max_samples, use_amp)
        else:
            tent_mod, eata_mod = import_tta_modules(Path(args.eata_root))
            adapt_model, param_names, fishers = build_adapt_model(args, algorithm, model, loader, device, tent_mod, eata_mod)
            row["fisher_param_tensors"] = len(fishers)
            row["unfrozen_param_tensors"] = len(param_names)
            row["unfrozen_param_elements"] = sum(p.numel() for p in collect_trainable_params(adapt_model.model))
            run_once = lambda: run_tta_stream(adapt_model, loader, device, args.max_samples, use_amp)

        metrics = profile_stream(run_once, metric_config)
        row.update(metrics)
        row["num_samples"] = metrics.pop("num_samples")
        row["num_batches"] = metrics.pop("num_batches")
        if adapt_model is not None and algorithm == "eata":
            row["tta_updates_reliable"] = getattr(adapt_model, "num_samples_update_1", "")
            row["tta_updates_reliable_non_redundant"] = getattr(adapt_model, "num_samples_update_2", "")
        add_stream_gops(row, args, algorithm)
        row["status"] = "ok"
        row["error"] = ""
        row["failure_reason"] = ""
        print(
            f"profiled stream model={args.model} algorithm={algorithm} "
            f"batch={batch_size} samples={row['num_samples']} status=ok"
        )
    except Exception as exc:
        row["status"] = "error"
        row["error"] = repr(exc)
        row["failure_reason"] = classify_failure(exc)
        torch.cuda.empty_cache()
        print(f"profiled stream model={args.model} algorithm={algorithm} batch={batch_size} status=error reason={row['failure_reason']}")
        if not args.keep_going:
            raise
    finally:
        torch.cuda.empty_cache()
    return row


def build_loader(args: argparse.Namespace, batch_size: int):
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if args.input_mode == "imagenet":
        root = Path(args.data) / "val"
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        root = Path(args.data_corruption) / args.corruption / str(args.level)
        transform = transforms.Compose([
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
            normalize,
        ])
    if not root.exists():
        raise FileNotFoundError(f"Input dataset directory does not exist: {root}")
    dataset = datasets.ImageFolder(str(root), transform=transform)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=args.drop_last,
    )


def run_baseline_stream(model: nn.Module, loader, device: torch.device, max_samples: int, use_amp: bool) -> Tuple[int, int]:
    num_samples = 0
    num_batches = 0
    with torch.no_grad():
        for images, _targets in loader:
            images = images.to(device=device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):
                model(images)
            num_samples += images.size(0)
            num_batches += 1
            if max_samples and num_samples >= max_samples:
                break
    return num_samples, num_batches


def run_tta_stream(adapt_model, loader, device: torch.device, max_samples: int, use_amp: bool) -> Tuple[int, int]:
    num_samples = 0
    num_batches = 0
    for images, _targets in loader:
        images = images.to(device=device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):
            adapt_model(images)
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):
                adapt_model.model(images)
        num_samples += images.size(0)
        num_batches += 1
        if max_samples and num_samples >= max_samples:
            break
    return num_samples, num_batches


def profile_stream(run_once, config: MetricConfig) -> Dict[str, object]:
    start_sample = read_sample(config)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    wall_start = time.perf_counter()
    start_event.record()
    num_samples, num_batches = run_once()
    end_event.record()
    torch.cuda.synchronize()
    wall_end = time.perf_counter()
    end_sample = read_sample(config)

    elapsed_ms_total = float(start_event.elapsed_time(end_event))
    elapsed_s_total = elapsed_ms_total / 1000.0
    avg_clock = average_optional(start_sample.sm_clock_mhz, end_sample.sm_clock_mhz)
    estimated_cycles_total = elapsed_ms_total * avg_clock * 1000.0 if avg_clock is not None else None
    avg_power_sampled = average_optional(start_sample.power_w, end_sample.power_w)

    energy_source = "unavailable"
    energy_j_total = None
    energy_j_total_est = None
    avg_power_from_energy = None
    if start_sample.energy_mj is not None and end_sample.energy_mj is not None:
        delta_mj = end_sample.energy_mj - start_sample.energy_mj
        if delta_mj > 0:
            energy_j_total = delta_mj / 1000.0
            avg_power_from_energy = energy_j_total / elapsed_s_total if elapsed_s_total > 0 else None
            energy_source = "nvml_total_energy"
        elif config.allow_zero_energy_estimate and avg_power_sampled is not None:
            energy_j_total_est = avg_power_sampled * elapsed_s_total
            energy_source = "power_estimate_zero_delta"
        else:
            energy_source = "nvml_zero_delta"
    elif avg_power_sampled is not None:
        energy_j_total_est = avg_power_sampled * elapsed_s_total
        energy_source = "power_estimate_no_energy_counter"

    return {
        "num_samples": num_samples,
        "num_batches": num_batches,
        "cuda_elapsed_ms_total": elapsed_ms_total,
        "latency_ms_per_batch": safe_div(elapsed_ms_total, num_batches),
        "latency_ms_per_sample": safe_div(elapsed_ms_total, num_samples),
        "wall_s_total": wall_end - wall_start,
        "estimated_sm_cycles_total": estimated_cycles_total,
        "estimated_sm_cycles_per_batch": safe_div(estimated_cycles_total, num_batches),
        "estimated_sm_cycles_per_sample": safe_div(estimated_cycles_total, num_samples),
        "sm_clock_mhz_start": start_sample.sm_clock_mhz,
        "sm_clock_mhz_end": end_sample.sm_clock_mhz,
        "sm_clock_mhz_avg_used": avg_clock,
        "power_w_start": start_sample.power_w,
        "power_w_end": end_sample.power_w,
        "avg_power_w_sampled": avg_power_sampled,
        "energy_j_total": energy_j_total,
        "energy_j_per_batch": safe_div(energy_j_total, num_batches),
        "energy_j_per_sample": safe_div(energy_j_total, num_samples),
        "energy_j_total_est_from_power": energy_j_total_est,
        "energy_j_per_batch_est_from_power": safe_div(energy_j_total_est, num_batches),
        "energy_j_per_sample_est_from_power": safe_div(energy_j_total_est, num_samples),
        "avg_power_w_from_energy": avg_power_from_energy,
        "energy_source": energy_source,
        "clock_source": join_sources(start_sample.clock_source, end_sample.clock_source),
        "power_source": join_sources(start_sample.power_source, end_sample.power_source),
        "util_source": join_sources(start_sample.util_source, end_sample.util_source),
        "nvml_available": True,
        "nvidia_smi_fallback_allowed": config.allow_nvidia_smi_fallback,
    }


def import_tta_modules(eata_root: Path):
    eata_root = eata_root.resolve()
    if str(eata_root) not in sys.path:
        sys.path.insert(0, str(eata_root))
    import tent  # type: ignore
    import eata  # type: ignore
    return tent, eata


def build_adapt_model(args, algorithm: str, model: nn.Module, loader, device: torch.device, tent_mod, eata_mod):
    model, params, param_names = configure_model_for_adaptation(model, parse_adapt_param_types(args.adapt_param_types))
    if not params:
        raise RuntimeError(f"{algorithm} found no adaptable parameters in model '{args.model}'.")
    fishers = {}
    if algorithm == "eata" and args.fisher_samples > 0:
        fishers = compute_fishers_from_loader(
            model,
            loader,
            device,
            args.fisher_samples,
            param_names,
            use_amp=args.dtype == "amp_fp16",
        )
    optimizer = torch.optim.SGD(params, args.lr, momentum=args.momentum)
    if algorithm == "tent":
        return tent_mod.Tent(model, optimizer, steps=args.adapt_steps, episodic=args.episodic), param_names, {}
    return (
        eata_mod.EATA(
            model,
            optimizer,
            fishers=fishers or None,
            fisher_alpha=args.fisher_alpha,
            steps=args.adapt_steps,
            episodic=args.episodic,
            e_margin=args.e_margin,
            d_margin=args.d_margin,
        ),
        param_names,
        fishers,
    )


def compute_fishers_from_loader(
    model: nn.Module,
    loader,
    device: torch.device,
    fisher_samples: int,
    param_names: Sequence[str],
    use_amp: bool,
) -> Dict[str, List[torch.Tensor]]:
    criterion = nn.CrossEntropyLoss()
    fishers: Dict[str, List[torch.Tensor]] = {}
    tracked = set(param_names)
    seen = 0
    model.train()
    for images, _targets in loader:
        images = images.to(device=device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=use_amp, dtype=torch.float16):
            outputs = model(images)
        targets = outputs.detach().argmax(dim=1)
        loss = criterion(outputs.float(), targets)
        loss.backward()
        batch = images.size(0)
        for name, param in model.named_parameters():
            if name not in tracked or param.grad is None:
                continue
            fisher = param.grad.detach().clone().pow(2) * batch
            if name in fishers:
                fishers[name][0].add_(fisher)
            else:
                fishers[name] = [fisher, param.detach().clone()]
        model.zero_grad(set_to_none=True)
        seen += batch
        if seen >= fisher_samples:
            break
    divisor = max(1, seen)
    for name in fishers:
        fishers[name][0].div_(divisor)
    return fishers


def base_row(args, algorithm: str, batch_size: int) -> Dict[str, object]:
    return {
        "status": "pending",
        "error": "",
        "failure_reason": "",
        "measurement": "baseline_stream" if algorithm == "none" else "tta_stream_adapt_plus_infer",
        "algorithm": algorithm,
        "model": args.model,
        "model_source": "",
        "pretrained": args.pretrained,
        "requested_batch_size": batch_size,
        "image_size": args.image_size,
        "input_mode": args.input_mode,
        "corruption": args.corruption if args.input_mode == "imagenet-c" else "",
        "level": args.level if args.input_mode == "imagenet-c" else "",
        "num_samples": 0,
        "num_batches": 0,
        "drop_last": args.drop_last,
        "adapt_steps": args.adapt_steps if algorithm != "none" else 0,
        "repeat": 1,
        "warmup": 0,
        "device": args.device,
        "gpu_index": args.gpu_index,
        "dtype": args.dtype,
        "lr": args.lr if algorithm != "none" else "",
        "momentum": args.momentum if algorithm != "none" else "",
        "episodic": args.episodic if algorithm != "none" else "",
        "adapt_param_types": args.adapt_param_types if algorithm != "none" else "",
        "fisher_samples": args.fisher_samples if algorithm == "eata" else 0,
        "fisher_param_tensors": 0,
        "unfrozen_param_tensors": 0,
        "unfrozen_param_elements": 0,
        "tta_updates_reliable": "",
        "tta_updates_reliable_non_redundant": "",
    }


def add_stream_gops(row: Dict[str, object], args: argparse.Namespace, algorithm: str) -> None:
    forward_gops = get_forward_gops_per_sample(args)
    factor = workload_forward_equiv_factor(args, algorithm)
    num_samples = row.get("num_samples")
    elapsed_ms = row.get("cuda_elapsed_ms_total")
    row["gops_count_source"] = args.gops_count_source
    row["model_forward_gops_per_sample"] = forward_gops
    row["workload_forward_equiv_factor"] = factor
    row["estimated_workload_gops_total"] = ""
    row["estimated_gops"] = ""
    row["estimated_gops_per_w"] = ""
    if not isinstance(forward_gops, (int, float)) or not isinstance(num_samples, int):
        return
    workload_gops = float(forward_gops) * num_samples * factor
    row["estimated_workload_gops_total"] = workload_gops
    if isinstance(elapsed_ms, (int, float)) and elapsed_ms > 0:
        gops = workload_gops / (float(elapsed_ms) / 1000.0)
        row["estimated_gops"] = gops
        power = preferred_power_w(row)
        if power is not None and power > 0:
            row["estimated_gops_per_w"] = gops / power


def safe_div(value, divisor):
    if value is None or divisor in {0, None}:
        return None
    return float(value) / float(divisor)


def average_optional(first, second):
    if first is None or second is None:
        return None
    return (float(first) + float(second)) / 2.0


def join_sources(first: str, second: str) -> str:
    return first if first == second else f"{first}->{second}"


def write_results(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()

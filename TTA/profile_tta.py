from __future__ import annotations

import argparse
import csv
import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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

from gpu_metrics import MetricConfig, assert_gpu_idle, profile_callable  # noqa: E402

try:
    from .model_factory import ModelInfo, build_model
except ImportError:
    from model_factory import ModelInfo, build_model  # type: ignore


METRIC_COLUMNS = [
    "cuda_elapsed_ms_total",
    "latency_ms_per_iter",
    "wall_s_total",
    "sm_clock_mhz_start",
    "sm_clock_mhz_end",
    "sm_clock_mhz_avg_used",
    "estimated_sm_cycles_per_iter",
    "power_w_start",
    "power_w_end",
    "avg_power_w_sampled",
    "energy_j_total",
    "energy_j_per_iter",
    "energy_j_total_est_from_power",
    "energy_j_per_iter_est_from_power",
    "avg_power_w_from_energy",
    "energy_source",
    "clock_source",
    "power_source",
    "util_source",
    "nvml_available",
    "nvidia_smi_fallback_allowed",
]

GOPS_COLUMNS = [
    "gops_count_source",
    "model_forward_gops_per_sample",
    "workload_forward_equiv_factor",
    "estimated_workload_gops_per_iter",
    "estimated_gops",
    "estimated_gops_per_w",
]

PER_SAMPLE_COLUMNS = [
    "latency_ms_per_sample",
    "estimated_sm_cycles_per_sample",
    "energy_j_per_sample",
    "energy_j_per_sample_est_from_power",
]

RESULT_COLUMNS = [
    "status",
    "error",
    "failure_reason",
    "measurement",
    "algorithm",
    "model",
    "model_source",
    "pretrained",
    "batch_size",
    "image_size",
    "input_mode",
    "corruption",
    "level",
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
    "unfrozen_param_names",
] + METRIC_COLUMNS + GOPS_COLUMNS + PER_SAMPLE_COLUMNS

DEFAULT_FORWARD_GOPS_PER_SAMPLE = {
    # MACs are counted as 2 ops. Override from the CLI if using a different convention.
    "resnet50": 8.178,
    "mobilenetv2_100": 0.600,
    "mobilevit_xxs": 0.800,
    "vit_base_patch16_224": 35.200,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile full-model TTA overhead with GPU Utilizer metrics."
    )
    parser.add_argument("--model", default="resnet50", help="Target model name.")
    parser.add_argument(
        "--model-source",
        default="auto",
        choices=["auto", "eata_resnet", "torchvision", "timm"],
        help="Model zoo to use. auto prefers EATA ResNet for ResNet/ResNeXt names.",
    )
    parser.add_argument("--eata-root", default=str(DEFAULT_EATA_ROOT))
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--no-pretrained", action="store_false", dest="pretrained")
    parser.add_argument(
        "--algorithms",
        default="tent,eata",
        help="Comma-separated algorithms: tent, eta, eata. Baseline is controlled separately.",
    )
    parser.add_argument("--no-baseline", action="store_true", help="Skip baseline inference rows.")
    parser.add_argument(
        "--batch-sizes",
        default="1,2,4,8,16,32,64,128",
        help="Comma-separated mini-batch sizes to sweep.",
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--adapt-steps", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--episodic", action="store_true")
    parser.add_argument(
        "--adapt-param-types",
        default="bn",
        help="Comma-separated trainable norm affine types: bn, ln. Use ln for ViT-style models.",
    )
    parser.add_argument("--e-margin", type=float, default=math.log(1000) * 0.40)
    parser.add_argument("--d-margin", type=float, default=0.05)
    parser.add_argument("--fisher-alpha", type=float, default=2000.0)
    parser.add_argument(
        "--fisher-samples",
        type=int,
        default=0,
        help="Precompute EATA fisher terms outside the measured region. 0 disables fisher.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--dtype", default="float32", choices=["float32"])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2020)
    parser.add_argument(
        "--input-mode",
        default="synthetic",
        choices=["synthetic", "imagenet", "imagenet-c"],
        help="synthetic excludes data loading from the measured GPU workload.",
    )
    parser.add_argument("--data", default="", help="ImageNet root containing val/.")
    parser.add_argument("--data-corruption", default="", help="ImageNet-C root.")
    parser.add_argument("--corruption", default="gaussian_noise")
    parser.add_argument("--level", type=int, default=5)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--forward-gops-per-sample",
        type=float,
        default=None,
        help="Static forward GOP/image with MAC=2 ops. Defaults are used for known models.",
    )
    parser.add_argument(
        "--gops-count-source",
        default="static_forward_gops_mac2",
        help="Label recorded in CSV for the GOP count source.",
    )
    parser.add_argument(
        "--tta-forward-equiv-factor",
        type=float,
        default=4.0,
        help="Forward-equivalent factor for adaptation+inference GOP estimates.",
    )
    parser.add_argument(
        "--eata-forward-equiv-factor",
        type=float,
        default=None,
        help="Optional EATA-specific forward-equivalent factor. Defaults to --tta-forward-equiv-factor.",
    )
    parser.add_argument("--output", default=str(GPU_UTILIZER_ROOT / "Results" / "TTA" / "tta_profile.csv"))
    parser.add_argument("--nvidia-smi-fallback", action="store_true")
    parser.add_argument("--no-idle-check", action="store_true")
    parser.add_argument("--no-zero-energy-estimate", action="store_true")
    parser.add_argument("--no-cudnn-benchmark", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true", default=True)
    parser.add_argument("--keep-going", action="store_false", dest="stop_on_error")
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
    assert_gpu_idle(metric_config)

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, object]] = []

    algorithms = parse_algorithms(args.algorithms)
    batch_sizes = parse_batch_sizes(args.batch_sizes)

    for batch_size in batch_sizes:
        try:
            adapt_x, infer_x = make_input_pair(args, batch_size, device)
        except Exception as exc:
            torch.cuda.empty_cache()
            rows.extend(make_batch_error_rows(args, batch_size, algorithms, exc))
            write_results(output_path, rows)
            if args.stop_on_error:
                raise
            continue

        if not args.no_baseline:
            rows.append(profile_baseline(args, batch_size, infer_x, metric_config))
            write_results(output_path, rows)

        for algorithm in algorithms:
            try:
                rows.append(profile_tta_algorithm(args, algorithm, batch_size, adapt_x, infer_x, metric_config))
            except Exception as exc:
                row = base_row(args, batch_size, algorithm, measurement="tta_adapt_plus_infer")
                row["status"] = "error"
                row["error"] = repr(exc)
                row["failure_reason"] = classify_failure(exc)
                rows.append(row)
                write_results(output_path, rows)
                torch.cuda.empty_cache()
                if args.stop_on_error:
                    raise
            else:
                write_results(output_path, rows)

    print(f"wrote TTA profile results: {output_path}")


def validate_args(args: argparse.Namespace) -> None:
    if args.repeat <= 0:
        raise ValueError("--repeat must be positive.")
    if args.warmup < 0:
        raise ValueError("--warmup must be non-negative.")
    if args.adapt_steps <= 0:
        raise ValueError("--adapt-steps must be positive.")
    if args.input_mode == "imagenet" and not args.data:
        raise ValueError("--data is required for --input-mode imagenet.")
    if args.input_mode == "imagenet-c" and not args.data_corruption:
        raise ValueError("--data-corruption is required for --input-mode imagenet-c.")
    parse_adapt_param_types(args.adapt_param_types)
    if get_forward_gops_per_sample(args) is None:
        raise ValueError(
            f"No default GOP count for model '{args.model}'. "
            "Pass --forward-gops-per-sample with MAC counted as 2 ops."
        )


def parse_batch_sizes(value: str) -> List[int]:
    sizes = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not sizes or any(size <= 0 for size in sizes):
        raise ValueError("--batch-sizes must contain positive integers.")
    return sizes


def parse_algorithms(value: str) -> List[str]:
    algorithms = [item.strip().lower() for item in value.split(",") if item.strip()]
    supported = {"tent", "eta", "eata"}
    unknown = [item for item in algorithms if item not in supported]
    if unknown:
        raise ValueError(f"Unsupported algorithms: {unknown}. Supported: {sorted(supported)}")
    return algorithms


def parse_adapt_param_types(value: str) -> List[str]:
    types = [item.strip().lower() for item in value.split(",") if item.strip()]
    supported = {"bn", "ln"}
    unknown = [item for item in types if item not in supported]
    if unknown:
        raise ValueError(f"Unsupported adapt param types: {unknown}. Supported: {sorted(supported)}")
    if not types:
        raise ValueError("--adapt-param-types must contain at least one type.")
    return types


def set_reproducible(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_input_pair(args: argparse.Namespace, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    if args.input_mode == "synthetic":
        shape = (batch_size, 3, args.image_size, args.image_size)
        adapt_x = torch.randn(shape, device=device, dtype=torch.float32)
        infer_x = torch.randn(shape, device=device, dtype=torch.float32)
        return adapt_x, infer_x
    return load_image_input_pair(args, batch_size, device)


def load_image_input_pair(args: argparse.Namespace, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
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
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )
    batches: List[torch.Tensor] = []
    for images, _targets in loader:
        batches.append(images.to(device=device, non_blocking=True))
        if len(batches) == 2:
            break
    if not batches:
        raise RuntimeError(f"No full batch of size {batch_size} could be loaded from {root}")
    if len(batches) == 1:
        batches.append(batches[0].clone())
    return batches[0], batches[1]


def profile_baseline(
    args: argparse.Namespace,
    batch_size: int,
    infer_x: torch.Tensor,
    metric_config: MetricConfig,
) -> Dict[str, object]:
    model_info = create_model(args)
    model = model_info.model.to(device=infer_x.device)
    model.eval()

    def run_once() -> torch.Tensor:
        with torch.no_grad():
            return model(infer_x)

    row = base_row(args, batch_size, "none", measurement="baseline_inference")
    row["model_source"] = model_info.source
    try:
        metrics = profile_callable(
            run_once,
            warmup=args.warmup,
            repeat=args.repeat,
            config=metric_config,
            nvtx_name=f"baseline_{args.model}_bs{batch_size}",
        )
        row.update(metrics)
        add_per_sample_metrics(row, batch_size)
        add_gops_metrics(row, args, batch_size, "none")
        row["status"] = "ok"
        row["error"] = ""
        row["failure_reason"] = ""
    except Exception as exc:
        row["status"] = "error"
        row["error"] = repr(exc)
        row["failure_reason"] = classify_failure(exc)
        if args.stop_on_error:
            raise
    finally:
        del model
        torch.cuda.empty_cache()
    print(f"profiled baseline model={args.model} batch={batch_size} status={row['status']}")
    return row


def profile_tta_algorithm(
    args: argparse.Namespace,
    algorithm: str,
    batch_size: int,
    adapt_x: torch.Tensor,
    infer_x: torch.Tensor,
    metric_config: MetricConfig,
) -> Dict[str, object]:
    tent_mod, eata_mod = import_tta_modules(Path(args.eata_root))
    model_info = create_model(args)
    model = model_info.model.to(device=adapt_x.device)
    adapt_model, param_names, fishers = build_adapt_model(args, algorithm, model, tent_mod, eata_mod, adapt_x)

    def run_once() -> torch.Tensor:
        adapt_model(adapt_x)
        with torch.no_grad():
            return adapt_model.model(infer_x)

    row = base_row(args, batch_size, algorithm, measurement="tta_adapt_plus_infer")
    row["model_source"] = model_info.source
    row["fisher_param_tensors"] = len(fishers)
    row["unfrozen_param_tensors"] = len(param_names)
    row["unfrozen_param_elements"] = sum(p.numel() for p in collect_trainable_params(adapt_model.model))
    row["unfrozen_param_names"] = "|".join(param_names)

    metrics = profile_callable(
        run_once,
        warmup=args.warmup,
        repeat=args.repeat,
        config=metric_config,
        nvtx_name=f"{algorithm}_{args.model}_bs{batch_size}",
    )
    row.update(metrics)
    add_per_sample_metrics(row, batch_size)
    add_gops_metrics(row, args, batch_size, algorithm)
    row["status"] = "ok"
    row["error"] = ""
    row["failure_reason"] = ""
    print(f"profiled {algorithm} model={args.model} batch={batch_size} status=ok")

    del adapt_model
    torch.cuda.empty_cache()
    return row


def import_tta_modules(eata_root: Path):
    eata_root = eata_root.resolve()
    if str(eata_root) not in sys.path:
        sys.path.insert(0, str(eata_root))
    import tent  # type: ignore
    import eata  # type: ignore

    return tent, eata


def create_model(args: argparse.Namespace) -> ModelInfo:
    return build_model(
        model_name=args.model,
        model_source=args.model_source,
        pretrained=args.pretrained,
        eata_root=Path(args.eata_root),
        num_classes=args.num_classes,
    )


def build_adapt_model(
    args: argparse.Namespace,
    algorithm: str,
    model: nn.Module,
    tent_mod,
    eata_mod,
    fisher_x: torch.Tensor,
):
    adapt_param_types = parse_adapt_param_types(args.adapt_param_types)
    if algorithm == "tent":
        model, params, param_names = configure_model_for_adaptation(model, adapt_param_types)
        ensure_adaptable_params(args.model, algorithm, params)
        optimizer = torch.optim.SGD(params, args.lr, momentum=args.momentum)
        return tent_mod.Tent(model, optimizer, steps=args.adapt_steps, episodic=args.episodic), param_names, {}

    model, params, param_names = configure_model_for_adaptation(model, adapt_param_types)
    ensure_adaptable_params(args.model, algorithm, params)
    fishers = {}
    if algorithm == "eata" and args.fisher_samples > 0:
        fishers = compute_fishers(model, fisher_x, args.fisher_samples, param_names)
    optimizer = torch.optim.SGD(params, args.lr, momentum=args.momentum)
    adapt_model = eata_mod.EATA(
        model,
        optimizer,
        fishers=fishers or None,
        fisher_alpha=args.fisher_alpha,
        steps=args.adapt_steps,
        episodic=args.episodic,
        e_margin=args.e_margin,
        d_margin=args.d_margin,
    )
    return adapt_model, param_names, fishers


def configure_model_for_adaptation(
    model: nn.Module,
    adapt_param_types: Sequence[str],
) -> Tuple[nn.Module, List[nn.Parameter], List[str]]:
    model.train()
    model.requires_grad_(False)

    params: List[nn.Parameter] = []
    names: List[str] = []
    adapt_bn = "bn" in adapt_param_types
    adapt_ln = "ln" in adapt_param_types

    for module_name, module in model.named_modules():
        should_adapt = False
        if adapt_bn and isinstance(module, nn.BatchNorm2d):
            should_adapt = True
            module.track_running_stats = False
            module.running_mean = None
            module.running_var = None
        elif adapt_ln and isinstance(module, nn.LayerNorm):
            should_adapt = True

        if not should_adapt:
            continue

        for param_name, param in module.named_parameters(recurse=False):
            if param_name not in {"weight", "bias"}:
                continue
            param.requires_grad_(True)
            params.append(param)
            names.append(f"{module_name}.{param_name}" if module_name else param_name)

    return model, params, names


def ensure_adaptable_params(model_name: str, algorithm: str, params: Sequence[nn.Parameter]) -> None:
    if params:
        return
    raise RuntimeError(
        f"{algorithm} found no BatchNorm2d affine parameters in model '{model_name}'. "
        "Use --adapt-param-types ln for ViT/LayerNorm models."
    )


def collect_trainable_params(model: nn.Module) -> List[nn.Parameter]:
    return [param for param in model.parameters() if param.requires_grad]


def compute_fishers(
    model: nn.Module,
    input_x: torch.Tensor,
    fisher_samples: int,
    param_names: Sequence[str],
) -> Dict[str, List[torch.Tensor]]:
    model.train()
    criterion = nn.CrossEntropyLoss()
    iterations = max(1, math.ceil(fisher_samples / input_x.size(0)))
    fishers: Dict[str, List[torch.Tensor]] = {}
    tracked = set(param_names)

    for iter_idx in range(iterations):
        outputs = model(input_x)
        targets = outputs.detach().argmax(dim=1)
        loss = criterion(outputs, targets)
        loss.backward()
        for name, param in model.named_parameters():
            if name not in tracked or param.grad is None:
                continue
            fisher = param.grad.detach().clone().pow(2)
            if name in fishers:
                fishers[name][0].add_(fisher)
            else:
                fishers[name] = [fisher, param.detach().clone()]
        model.zero_grad(set_to_none=True)

    for name in fishers:
        fishers[name][0].div_(iterations)
    return fishers


def base_row(args: argparse.Namespace, batch_size: int, algorithm: str, measurement: str) -> Dict[str, object]:
    return {
        "status": "pending",
        "error": "",
        "failure_reason": "",
        "measurement": measurement,
        "algorithm": algorithm,
        "model": args.model,
        "model_source": "",
        "pretrained": args.pretrained,
        "batch_size": batch_size,
        "image_size": args.image_size,
        "input_mode": args.input_mode,
        "corruption": args.corruption if args.input_mode == "imagenet-c" else "",
        "level": args.level if args.input_mode == "imagenet-c" else "",
        "adapt_steps": args.adapt_steps if algorithm != "none" else 0,
        "repeat": args.repeat,
        "warmup": args.warmup,
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
        "unfrozen_param_names": "",
    }


def make_batch_error_rows(
    args: argparse.Namespace,
    batch_size: int,
    algorithms: Sequence[str],
    exc: Exception,
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    reason = classify_failure(exc)
    if not args.no_baseline:
        row = base_row(args, batch_size, "none", measurement="baseline_inference")
        row["status"] = "error"
        row["error"] = repr(exc)
        row["failure_reason"] = reason
        rows.append(row)
    for algorithm in algorithms:
        row = base_row(args, batch_size, algorithm, measurement="tta_adapt_plus_infer")
        row["status"] = "error"
        row["error"] = repr(exc)
        row["failure_reason"] = reason
        rows.append(row)
    return rows


def classify_failure(exc: Exception) -> str:
    message = repr(exc).lower()
    if "out of memory" in message or "cuda oom" in message or "cublas_status_alloc_failed" in message:
        return "oom"
    if "no full batch" in message:
        return "insufficient_data"
    if "does not exist" in message:
        return "missing_data"
    return "error"


def get_forward_gops_per_sample(args: argparse.Namespace) -> Optional[float]:
    if args.forward_gops_per_sample is not None:
        return args.forward_gops_per_sample
    return DEFAULT_FORWARD_GOPS_PER_SAMPLE.get(args.model)


def workload_forward_equiv_factor(args: argparse.Namespace, algorithm: str) -> float:
    if algorithm == "none":
        return 1.0
    if algorithm == "eata" and args.eata_forward_equiv_factor is not None:
        return args.eata_forward_equiv_factor
    return args.tta_forward_equiv_factor


def add_gops_metrics(
    row: Dict[str, object],
    args: argparse.Namespace,
    batch_size: int,
    algorithm: str,
) -> None:
    forward_gops = get_forward_gops_per_sample(args)
    factor = workload_forward_equiv_factor(args, algorithm)
    row["gops_count_source"] = args.gops_count_source
    row["model_forward_gops_per_sample"] = forward_gops
    row["workload_forward_equiv_factor"] = factor

    if forward_gops is None:
        row["estimated_workload_gops_per_iter"] = ""
        row["estimated_gops"] = ""
        row["estimated_gops_per_w"] = ""
        return

    workload_gops = forward_gops * batch_size * factor
    latency_ms = row.get("latency_ms_per_iter")
    avg_power_w = preferred_power_w(row)
    row["estimated_workload_gops_per_iter"] = workload_gops
    row["estimated_gops"] = ""
    row["estimated_gops_per_w"] = ""

    if isinstance(latency_ms, (int, float)) and latency_ms > 0:
        gops = workload_gops / (float(latency_ms) / 1000.0)
        row["estimated_gops"] = gops
        if avg_power_w is not None and avg_power_w > 0:
            row["estimated_gops_per_w"] = gops / avg_power_w


def preferred_power_w(row: Dict[str, object]) -> Optional[float]:
    for key in ("avg_power_w_from_energy", "avg_power_w_sampled"):
        value = row.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def add_per_sample_metrics(row: Dict[str, object], batch_size: int) -> None:
    row["latency_ms_per_sample"] = divide_metric(row.get("latency_ms_per_iter"), batch_size)
    row["estimated_sm_cycles_per_sample"] = divide_metric(row.get("estimated_sm_cycles_per_iter"), batch_size)
    row["energy_j_per_sample"] = divide_metric(row.get("energy_j_per_iter"), batch_size)
    row["energy_j_per_sample_est_from_power"] = divide_metric(row.get("energy_j_per_iter_est_from_power"), batch_size)


def divide_metric(value: object, divisor: int) -> object:
    if isinstance(value, (int, float)) and divisor > 0:
        return float(value) / divisor
    return ""


def write_results(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    main()

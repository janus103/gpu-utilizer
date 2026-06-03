from __future__ import annotations

import argparse
import configparser
import csv
from pathlib import Path
from typing import Dict, List

import torch

from gpu_metrics import MetricConfig, assert_gpu_idle, profile_callable
from layer_templates import build_workload
from schema import LayerSpec, STANDARD_COLUMNS, load_layer_csv, write_normalized_csv


RESULT_COLUMNS = STANDARD_COLUMNS + [
    "status",
    "error",
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


def str_to_bool(value: str, default: bool = False) -> bool:
    if value is None:
        return default
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return default


def load_config(path: Path) -> configparser.ConfigParser:
    config = configparser.ConfigParser()
    read_files = config.read(path)
    if not read_files:
        raise FileNotFoundError(f"Unable to read config file: {path}")
    return config


def resolve_path(base_dir: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base_dir / path).resolve()


def merged_layer_value(layer: LayerSpec, key: str, default: str) -> str:
    value = layer.get(key)
    return default if value in {"-1", ""} else value


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile GPU metrics for layer CSV rows.")
    parser.add_argument("--config", default="config.ini")
    parser.add_argument("--nvidia-smi-fallback", action="store_true", help="Explicitly allow nvidia-smi fallback.")
    parser.add_argument("--no-idle-check", action="store_true", help="Disable GPU idle check for diagnostics.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    config_path = resolve_path(root, args.config)
    config = load_config(config_path)

    run_cfg = config["run"] if config.has_section("run") else {}
    layer_csv = resolve_path(root, run_cfg.get("layer_csv", "examples/layers_example.csv"))
    output_csv = resolve_path(root, run_cfg.get("output_csv", "Results/gpu_profile_results.csv"))
    normalized_csv = resolve_path(root, run_cfg.get("normalized_csv", "Results/normalized_layers.csv"))

    device_str = run_cfg.get("device", "cuda:0")
    device_index = int(device_str.split(":")[1]) if ":" in device_str else int(run_cfg.get("gpu_index", "0"))
    repeat_default = int(run_cfg.get("repeat", run_cfg.get("iters", "200")))
    warmup_default = int(run_cfg.get("warmup", "50"))
    dtype_default = run_cfg.get("dtype", "float32")
    stop_on_error = str_to_bool(run_cfg.get("stop_on_error", "true"), True)
    allow_fallback = args.nvidia_smi_fallback or str_to_bool(run_cfg.get("allow_nvidia_smi_fallback", "false"))
    require_idle = (not args.no_idle_check) and str_to_bool(run_cfg.get("require_idle_gpu", "true"), True)
    allow_zero_energy_estimate = str_to_bool(run_cfg.get("allow_zero_energy_estimate", "true"), True)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    device = torch.device(device_str)
    torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = str_to_bool(run_cfg.get("cudnn_benchmark", "true"), True)

    metric_config = MetricConfig(
        device_index=device_index,
        allow_nvidia_smi_fallback=allow_fallback,
        require_idle_gpu=require_idle,
        allow_zero_energy_estimate=allow_zero_energy_estimate,
    )
    assert_gpu_idle(metric_config)

    layers = load_layer_csv(layer_csv)
    write_normalized_csv(layers, normalized_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, object]] = []
    for layer in layers:
        layer_row: Dict[str, object] = layer.as_row()
        repeat = int(merged_layer_value(layer, "repeat", str(repeat_default)))
        warmup = int(merged_layer_value(layer, "warmup", str(warmup_default)))
        try:
            workload = build_workload(layer, device, dtype_default)

            def run_once():
                with torch.no_grad():
                    return workload.run()

            metrics = profile_callable(
                run_once,
                warmup=warmup,
                repeat=repeat,
                config=metric_config,
                nvtx_name=f"layer_{layer.get('layer_seq')}_{layer.get('layer_type')}",
            )
            layer_row.update(metrics)
            layer_row["status"] = "ok"
            layer_row["error"] = ""
        except Exception as exc:
            layer_row["status"] = "error"
            layer_row["error"] = repr(exc)
            if stop_on_error:
                rows.append(layer_row)
                write_results(output_csv, rows)
                raise
        rows.append(layer_row)
        write_results(output_csv, rows)
        print(f"profiled layer_seq={layer.get('layer_seq')} layer_type={layer.get('layer_type')} status={layer_row['status']}")

    print(f"wrote results: {output_csv}")
    print(f"wrote normalized layers: {normalized_csv}")


def write_results(path: Path, rows: List[Dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    main()

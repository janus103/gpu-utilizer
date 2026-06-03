from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from typing import Callable, Dict, Optional

import torch


@dataclass
class MetricConfig:
    device_index: int
    allow_nvidia_smi_fallback: bool = False
    require_idle_gpu: bool = True
    allow_zero_energy_estimate: bool = True


@dataclass
class GpuSample:
    sm_clock_mhz: Optional[int]
    power_w: Optional[float]
    energy_mj: Optional[int]
    gpu_util_percent: Optional[int]
    clock_source: str
    power_source: str
    energy_source: str
    util_source: str


def nvml_handle(device_index: int):
    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        return pynvml.nvmlDeviceGetHandleByIndex(device_index), pynvml
    except Exception:
        return None, None


def nvidia_smi_query(device_index: int, field: str) -> Optional[str]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                f"--query-gpu={field}",
                "--format=csv,noheader,nounits",
                "-i",
                str(device_index),
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return output.strip().splitlines()[0].strip()
    except Exception:
        return None


def require_nvml(device_index: int) -> None:
    handle, pynvml = nvml_handle(device_index)
    if handle is None or pynvml is None:
        raise RuntimeError(
            "NVML Python API is required in default mode, but pynvml/NVML is unavailable. "
            "Install nvidia-ml-py or explicitly set allow_nvidia_smi_fallback=true for diagnostic runs."
        )


def read_sample(config: MetricConfig) -> GpuSample:
    handle, pynvml = nvml_handle(config.device_index)
    sm_clock_mhz = None
    power_w = None
    energy_mj = None
    gpu_util_percent = None
    clock_source = "unavailable"
    power_source = "unavailable"
    energy_source = "unavailable"
    util_source = "unavailable"

    if handle is not None and pynvml is not None:
        try:
            sm_clock_mhz = int(pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM))
            clock_source = "nvml"
        except Exception:
            pass
        try:
            power_w = float(pynvml.nvmlDeviceGetPowerUsage(handle)) / 1000.0
            power_source = "nvml"
        except Exception:
            pass
        try:
            energy_mj = int(pynvml.nvmlDeviceGetTotalEnergyConsumption(handle))
            energy_source = "nvml"
        except Exception:
            pass
        try:
            utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_util_percent = int(utilization.gpu)
            util_source = "nvml"
        except Exception:
            pass

    if config.allow_nvidia_smi_fallback:
        if sm_clock_mhz is None:
            value = nvidia_smi_query(config.device_index, "clocks.sm")
            if value is not None:
                sm_clock_mhz = int(float(value))
                clock_source = "nvidia-smi"
        if power_w is None:
            value = nvidia_smi_query(config.device_index, "power.draw")
            if value is not None:
                power_w = float(value)
                power_source = "nvidia-smi"
        if gpu_util_percent is None:
            value = nvidia_smi_query(config.device_index, "utilization.gpu")
            if value is not None:
                gpu_util_percent = int(float(value))
                util_source = "nvidia-smi"

    return GpuSample(
        sm_clock_mhz=sm_clock_mhz,
        power_w=power_w,
        energy_mj=energy_mj,
        gpu_util_percent=gpu_util_percent,
        clock_source=clock_source,
        power_source=power_source,
        energy_source=energy_source,
        util_source=util_source,
    )


def assert_gpu_idle(config: MetricConfig) -> None:
    if not config.require_idle_gpu:
        return
    sample = read_sample(config)
    if sample.gpu_util_percent is None:
        raise RuntimeError("Unable to read GPU utilization before profiling.")
    if sample.gpu_util_percent != 0:
        raise RuntimeError(
            f"Selected GPU {config.device_index} is not idle: volatile GPU-Util={sample.gpu_util_percent}%. "
            "Stop other GPU workloads or disable require_idle_gpu only for diagnostic runs."
        )


def profile_callable(
    run_once: Callable[[], object],
    warmup: int,
    repeat: int,
    config: MetricConfig,
    nvtx_name: str,
) -> Dict[str, object]:
    if not config.allow_nvidia_smi_fallback:
        require_nvml(config.device_index)

    for _ in range(warmup):
        run_once()
    torch.cuda.synchronize()

    start_sample = read_sample(config)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    wall_start = time.perf_counter()

    start_event.record()
    for idx in range(repeat):
        torch.cuda.nvtx.range_push(f"{nvtx_name}_{idx}")
        run_once()
        torch.cuda.nvtx.range_pop()
    end_event.record()
    torch.cuda.synchronize()

    wall_end = time.perf_counter()
    end_sample = read_sample(config)
    elapsed_ms_total = float(start_event.elapsed_time(end_event))
    elapsed_s_total = elapsed_ms_total / 1000.0
    latency_ms_per_iter = elapsed_ms_total / repeat

    avg_clock = average_optional(start_sample.sm_clock_mhz, end_sample.sm_clock_mhz)
    estimated_cycles = latency_ms_per_iter * avg_clock * 1000.0 if avg_clock is not None else None
    avg_power_sampled = average_optional(start_sample.power_w, end_sample.power_w)

    energy_source = "unavailable"
    energy_j_total = None
    energy_j_per_iter = None
    energy_j_total_est = None
    energy_j_per_iter_est = None
    avg_power_from_energy = None

    if start_sample.energy_mj is not None and end_sample.energy_mj is not None:
        delta_mj = end_sample.energy_mj - start_sample.energy_mj
        if delta_mj > 0:
            energy_j_total = delta_mj / 1000.0
            energy_j_per_iter = energy_j_total / repeat
            avg_power_from_energy = energy_j_total / elapsed_s_total if elapsed_s_total > 0 else None
            energy_source = "nvml_total_energy"
        elif config.allow_zero_energy_estimate and avg_power_sampled is not None:
            energy_j_total_est = avg_power_sampled * elapsed_s_total
            energy_j_per_iter_est = energy_j_total_est / repeat
            energy_source = "power_estimate_zero_delta"
        else:
            energy_source = "nvml_zero_delta"
    elif avg_power_sampled is not None:
        energy_j_total_est = avg_power_sampled * elapsed_s_total
        energy_j_per_iter_est = energy_j_total_est / repeat
        energy_source = "power_estimate_no_energy_counter"

    return {
        "cuda_elapsed_ms_total": elapsed_ms_total,
        "latency_ms_per_iter": latency_ms_per_iter,
        "wall_s_total": wall_end - wall_start,
        "sm_clock_mhz_start": start_sample.sm_clock_mhz,
        "sm_clock_mhz_end": end_sample.sm_clock_mhz,
        "sm_clock_mhz_avg_used": avg_clock,
        "estimated_sm_cycles_per_iter": estimated_cycles,
        "power_w_start": start_sample.power_w,
        "power_w_end": end_sample.power_w,
        "avg_power_w_sampled": avg_power_sampled,
        "energy_j_total": energy_j_total,
        "energy_j_per_iter": energy_j_per_iter,
        "energy_j_total_est_from_power": energy_j_total_est,
        "energy_j_per_iter_est_from_power": energy_j_per_iter_est,
        "avg_power_w_from_energy": avg_power_from_energy,
        "energy_source": energy_source,
        "clock_source": join_sources(start_sample.clock_source, end_sample.clock_source),
        "power_source": join_sources(start_sample.power_source, end_sample.power_source),
        "util_source": join_sources(start_sample.util_source, end_sample.util_source),
        "nvml_available": nvml_handle(config.device_index)[0] is not None,
        "nvidia_smi_fallback_allowed": config.allow_nvidia_smi_fallback,
    }


def average_optional(first, second):
    if first is None or second is None:
        return None
    return (float(first) + float(second)) / 2.0


def join_sources(first: str, second: str) -> str:
    return first if first == second else f"{first}->{second}"

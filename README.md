# GPU Utilizer Library

GPU Utilizer is a portable layer-level profiler for PyTorch operators. It is designed to be copied into another GPU server, pointed at a layer CSV or a local `timm` model checkout, and used to produce per-layer GPU frequency, latency, estimated cycle, energy, and power measurements.

The intended use case is paper-facing GPU baseline measurement for layers such as standard convolution, depthwise convolution, pointwise convolution, fully connected layers, matrix multiplication, and self-attention.

## Why This Exists

PIM/CXL/PIMFlow-style studies often need a fair GPU baseline for the same layer shape. GPU Utilizer provides a small and reproducible way to measure those layers directly on a real GPU instead of relying only on trace-driven GPU simulation.

The default profiling path is strict:

- Latency is measured with CUDA events through PyTorch.
- SM clock, power, energy, and GPU utilization are read through NVML Python API (`pynvml` / `nvidia-ml-py`).
- `nvidia-smi` is not used unless explicitly enabled with `--nvidia-smi-fallback`.
- Profiling aborts if the selected GPU volatile utilization is not `0%`.

## Directory Structure

```text
gpu_utilizer/
  README.md
  config.ini
  config_mobilenetv2_timm.ini
  profile_layers.py
  extract_timm_layers.py
  gpu_metrics.py
  layer_templates.py
  schema.py
  examples/
    layers_example.csv
    mobilenetv2_100_bs1_layers.csv
  Results/
    gpu_profile_results.csv
    mobilenetv2_100_bs1_timm_profile.csv
    mobilenetv2_100_bs1_timm_normalized.csv
  Explains/
    metric_methodology.md
    layer_schema.md
    portability_review.md
    agent_transfer_prompt.md
  Dependencies/
    requirements.txt
    environment.robust.yml
    INSTALL_FOR_AGENTS.md
```

Generated CSV files in `Results/` are ignored by `.gitignore` by default. Keep them only when intentionally publishing example outputs.

## Installation

If PyTorch is already installed and CUDA works, install the minimal additional dependency:

```bash
python3 -m pip install nvidia-ml-py pandas numpy
```

Or install from the provided requirements:

```bash
python3 -m pip install -r Dependencies/requirements.txt
```

Validate CUDA and NVML:

```bash
python3 - <<'PY'
import torch
import pynvml

print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device0", torch.cuda.get_device_name(0))
pynvml.nvmlInit()
h = pynvml.nvmlDeviceGetHandleByIndex(0)
print("power_mw", pynvml.nvmlDeviceGetPowerUsage(h))
print("sm_clock_mhz", pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM))
print("gpu_util", pynvml.nvmlDeviceGetUtilizationRates(h).gpu)
PY
```

## Quick Start

```bash
cd gpu_utilizer
python3 profile_layers.py --config config.ini
```

This reads `examples/layers_example.csv` and writes:

```text
Results/gpu_profile_results.csv
Results/normalized_layers.csv
```

## Input Layer CSV

`config.ini` points to a layer CSV. The standard identifiers are:

- `layer_seq`: actual execution order.
- `layer_type`: one of `standard_conv`, `depthwise_conv`, `pointwise_conv`, `fully_connected`, `matrix`, `self_attention`.
- `layer_name`: optional label.

Unsupported or unused fields should be `-1`. See:

```text
examples/layers_example.csv
Explains/layer_schema.md
```

## Output Metrics

The result CSV contains the normalized layer spec plus metrics:

- `latency_ms_per_iter`
- `sm_clock_mhz_start`, `sm_clock_mhz_end`, `sm_clock_mhz_avg_used`
- `estimated_sm_cycles_per_iter`
- `energy_j_per_iter` or `energy_j_per_iter_est_from_power`
- `avg_power_w_from_energy` or `avg_power_w_sampled`
- `energy_source`, `clock_source`, `power_source`, `util_source`
- `nvml_available`, `nvidia_smi_fallback_allowed`

The estimated cycle is computed as:

```text
estimated_sm_cycles_per_iter = latency_ms_per_iter * sm_clock_mhz_avg_used * 1000
```

This is a time-normalized GPU SM-cycle proxy, not an Accel-Sim internal cycle and not a PIM command cycle.

## MobileNetV2 A100 Profiling Example

This repository was validated on an NVIDIA A100 80GB PCIe server using local `timm` from `pytorch-image-models`.

Layer extraction:

```bash
python3 extract_timm_layers.py \
  --timm-root /home/oem/TETC/pytorch-image-models \
  --model mobilenetv2_100 \
  --batch 1 \
  --image-size 224 \
  --repeat 20 \
  --warmup 5 \
  --output examples/mobilenetv2_100_bs1_layers.csv
```

Profiling:

```bash
python3 profile_layers.py --config config_mobilenetv2_timm.ini
```

Validation summary:

```text
model: mobilenetv2_100
batch size: 1
repeat per layer: 20
extracted layers: 53
successful profiles: 53 / 53
standard_conv: 1
depthwise_conv: 17
pointwise_conv: 34
fully_connected: 1
sum of per-layer latency_ms_per_iter: ~2.1468 ms
fallback: nvidia_smi_fallback_allowed=False
```

Generated files:

```text
Results/mobilenetv2_100_bs1_timm_profile.csv
Results/mobilenetv2_100_bs1_timm_normalized.csv
```

Example first layer result:

```text
layer_name=conv_stem
layer_type=standard_conv
latency_ms_per_iter=0.0420848
estimated_sm_cycles_per_iter=59339.57
energy_source=power_estimate_zero_delta
clock_source=nvml
power_source=nvml
```

The energy source is `power_estimate_zero_delta` for many short layers because the NVML total energy counter can have zero delta over sub-ms windows. In that case GPU Utilizer reports the zero-delta explicitly and provides a power-based estimate.

## timm Model Extraction

Use `extract_timm_layers.py` to convert a local timm model into GPU Utilizer's layer schema:

```bash
python3 extract_timm_layers.py \
  --timm-root /path/to/pytorch-image-models \
  --model mobilenetv2_100 \
  --batch 1 \
  --repeat 20 \
  --warmup 5 \
  --output examples/mobilenetv2_100_bs1_layers.csv
```

Currently the extractor records `nn.Conv2d` and `nn.Linear` modules. It classifies convolutions as:

- `standard_conv`
- `depthwise_conv`
- `pointwise_conv`

## nvidia-smi Fallback Policy

For paper measurements, do not use fallback:

```bash
python3 profile_layers.py --config config.ini
```

For diagnostics only:

```bash
python3 profile_layers.py --config config.ini --nvidia-smi-fallback
```

The output always records:

```text
nvml_available
nvidia_smi_fallback_allowed
clock_source
power_source
energy_source
```

## Clone and Auto-Run on Another Server

On a target GPU server, clone this repository and ask a coding agent to read this README before changing anything:

```bash
git clone https://github.com/janus103/gpu-utilizer.git
cd gpu-utilizer
```

Agent instruction:

```text
You are on a GPU server. I cloned `https://github.com/janus103/gpu-utilizer`.

Goal:
Read `README.md`, set up only the missing dependencies, validate CUDA/NVML, and run layer-level GPU profiling. Save per-layer Frequency, estimated Cycle, Energy, and Power to CSV.

Rules:
1. Do not use `nvidia-smi` fallback for paper measurements.
2. Use strict NVML Python API mode by default.
3. If NVML Python API is missing, install `nvidia-ml-py` or stop and report the issue.
4. Before profiling, confirm the selected GPU volatile utilization is 0%.
5. If GPU utilization is not 0%, stop and report which process/GPU is busy.
6. Do not reinstall PyTorch unless CUDA compatibility is broken.
7. Do not modify the user's model code unless explicitly requested.

Steps:
1. Inspect the environment:
   `python3 -c "import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"`
2. Validate NVML:
   `python3 - <<'PY'
import pynvml
pynvml.nvmlInit()
h = pynvml.nvmlDeviceGetHandleByIndex(0)
print('power_mw', pynvml.nvmlDeviceGetPowerUsage(h))
print('sm_clock_mhz', pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM))
print('gpu_util', pynvml.nvmlDeviceGetUtilizationRates(h).gpu)
PY`
3. If dependencies are missing, install only missing packages:
   `python3 -m pip install nvidia-ml-py pandas numpy`
4. Run the built-in example:
   `python3 profile_layers.py --config config.ini`
5. If profiling a local timm model, generate the layer CSV:
   `python3 extract_timm_layers.py --timm-root <PATH_TO_PYTORCH_IMAGE_MODELS_OR_TIMM_PROJECT> --model mobilenetv2_100 --batch 1 --repeat 20 --warmup 5 --output examples/mobilenetv2_100_bs1_layers.csv`
6. Profile the MobileNetV2 layer CSV:
   `python3 profile_layers.py --config config_mobilenetv2_timm.ini`
7. Summarize:
   - Number of profiled layers.
   - Status counts.
   - Layer type counts.
   - Output CSV path.
   - Whether `nvidia_smi_fallback_allowed` is false in the result.

If the server is Docker-based, confirm the container was launched with GPU support, for example `--gpus all`. Do not use `--nvidia-smi-fallback` unless the user explicitly asks for diagnostic mode.
```

The shorter standalone version of this prompt is also kept in `Explains/agent_transfer_prompt.md`.

## Notes and Limitations

- The profiler creates synthetic operator templates from layer shapes; it does not execute the full original model graph unless you first extract/convert the layers.
- Estimated SM cycles are latency proxies based on measured time and observed SM clock.
- Very short layers should use enough repetitions to avoid unstable energy readings.
- `nvidia-smi` fallback is explicit and diagnostic-only.

# Agent Prompt for Transfer and Auto-Run

Copy the prompt below to Cursor, Claude, or another coding agent on the target GPU server.

```text
You are on a GPU server. I copied a directory named `gpu_utilizer/` into my project.

Goal:
Run layer-level GPU profiling for a target model using `gpu_utilizer`, and save per-layer Frequency, estimated Cycle, Energy, and Power to CSV.

Rules:
1. Do not use `nvidia-smi` fallback for paper measurements.
2. Use strict NVML Python API mode by default.
3. If NVML Python API is missing, install `nvidia-ml-py` or stop and report the issue.
4. Before profiling, confirm the selected GPU volatile utilization is 0%.
5. If GPU utilization is not 0%, stop and report which process/GPU is busy.
6. Do not modify the user's model code unless explicitly requested.

Steps:
1. Enter the directory:
   `cd gpu_utilizer`
2. Inspect environment:
   `python3 -c "import torch; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"`
3. Validate NVML:
   `python3 - <<'PY'
import pynvml
pynvml.nvmlInit()
h = pynvml.nvmlDeviceGetHandleByIndex(0)
print('power_mw', pynvml.nvmlDeviceGetPowerUsage(h))
print('sm_clock_mhz', pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM))
print('gpu_util', pynvml.nvmlDeviceGetUtilizationRates(h).gpu)
PY`
4. If dependencies are missing, install only missing packages:
   `python3 -m pip install nvidia-ml-py pandas numpy`
   Do not reinstall PyTorch unless CUDA compatibility is broken.
5. If profiling a timm model, generate the layer CSV:
   `python3 extract_timm_layers.py --timm-root <PATH_TO_PYTORCH_IMAGE_MODELS_OR_TIMM_PROJECT> --model mobilenetv2_100 --batch 1 --repeat 20 --warmup 5 --output examples/mobilenetv2_100_bs1_layers.csv`
6. Create or edit a config file:
   `config_mobilenetv2_timm.ini`
   Set:
   `layer_csv = examples/mobilenetv2_100_bs1_layers.csv`
   `output_csv = Results/mobilenetv2_100_bs1_timm_profile.csv`
   `device = cuda:0`
   `repeat = 20`
   `warmup = 5`
   `require_idle_gpu = true`
   `allow_nvidia_smi_fallback = false`
7. Run profiling:
   `python3 profile_layers.py --config config_mobilenetv2_timm.ini`
8. Summarize:
   - Number of profiled layers.
   - Status counts.
   - Layer type counts.
   - Output CSV path.
   - Whether `nvidia_smi_fallback_allowed` is false in the result.

If the server is Docker-based:
1. Confirm the container was launched with GPU support, e.g. `--gpus all`.
2. Confirm Python can import `pynvml`.
3. If `pynvml` works, run the same strict NVML path.
4. Do not use `--nvidia-smi-fallback` unless the user explicitly asks for diagnostic mode.
```


# Install Instructions for Agents

Use these steps when moving `gpu_utilizer/` to another GPU server.

## 1. Detect Environment

Check whether conda exists:

```bash
conda env list
```

If a suitable environment such as `robust` or `robust2` already exists, activate it:

```bash
conda activate robust
```

If no conda environment is available, use the system Python or a Docker container Python.

## 2. Install Dependencies

Conda path:

```bash
conda env create -f Dependencies/environment.robust.yml
conda activate gpu-utilizer
```

Existing conda or pip-only path:

```bash
python3 -m pip install -r Dependencies/requirements.txt
```

If the server already has PyTorch installed, avoid reinstalling it unless CUDA compatibility is broken:

```bash
python3 -m pip install nvidia-ml-py pandas numpy
```

## 3. Validate CUDA and NVML

```bash
python3 - <<'PY'
import torch
import pynvml

print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device0", torch.cuda.get_device_name(0))
pynvml.nvmlInit()
handle = pynvml.nvmlDeviceGetHandleByIndex(0)
print("power_mw", pynvml.nvmlDeviceGetPowerUsage(handle))
print("sm_clock_mhz", pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM))
PY
```

If NVML import fails, install:

```bash
python3 -m pip install nvidia-ml-py
```

Do not enable `--nvidia-smi-fallback` for paper measurements. Use it only for diagnostics.

## 3-1. Export Current Environment When Needed

If the target server already has a working environment such as `robust` or `robust2`, export it locally for reproducibility:

```bash
conda env export -n robust --from-history > Dependencies/environment.robust.from-history.yml
```

Avoid committing a full `conda env export` with `prefix:` and hundreds of transitive packages unless you intentionally want a machine-specific lock file. The provided `environment.robust.yml` is a portable template, not a full server lock.

## 4. Run Smoke Test

```bash
python3 profile_layers.py --config config.ini
```

If the selected GPU is not idle, the profiler will abort. Stop other GPU jobs or use a different `device` in `config.ini`.

## 5. Docker Notes

For Docker, make sure the container is launched with GPU and NVML access:

```bash
docker run --gpus all ...
```

Inside the container, `nvidia-smi` should work and Python should import `pynvml`. If not, the host/container NVIDIA runtime is not configured correctly.

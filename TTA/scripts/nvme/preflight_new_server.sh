#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GPU_UTILIZER_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${GPU_UTILIZER_DIR}"

if [[ $# -ge 1 ]]; then
  export DATA_CORRUPTION="$1"
fi

source TTA/configs/nvme_env.sh
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"

CONDA_ENV="${CONDA_ENV:-robust}"
if [[ "${SKIP_CONDA_ACTIVATE:-0}" != "1" ]]; then
  set +u
  source "${HOME}/miniconda3/etc/profile.d/conda.sh" 2>/dev/null \
    || source "${HOME}/anaconda3/etc/profile.d/conda.sh" 2>/dev/null \
    || source "/opt/conda/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
  set -u
fi

echo "[preflight] gpu_utilizer=${GPU_UTILIZER_DIR}"
echo "[preflight] conda_env=${CONDA_DEFAULT_ENV:-unknown}"
echo "[preflight] data_corruption=${DATA_CORRUPTION}"
echo "[preflight] corruption=${CORRUPTION} level=${LEVEL}"
echo "[preflight] gpu=${GPU} device=${DEVICE} workers=${WORKERS}"

python3 - <<'PY'
from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from pathlib import Path

import torch

root = Path.cwd()
tta_dir = root / "TTA"
external = tta_dir / "external_methods"
data_root = Path(os.environ["DATA_CORRUPTION"])
corruption = os.environ.get("CORRUPTION", "gaussian_noise")
level = os.environ.get("LEVEL", "5")
device_name = os.environ.get("DEVICE", "cuda:0")

errors: list[str] = []


def ok(msg: str) -> None:
    print(f"[OK] {msg}")


def fail(msg: str) -> None:
    print(f"[FAIL] {msg}")
    errors.append(msg)


def warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def check_path(path: Path, label: str) -> None:
    if path.exists():
        ok(f"{label}: {path}")
    else:
        fail(f"{label} missing: {path}")


check_path(data_root / corruption / str(level), "ImageNet-C split")
for method in ("eata", "sar", "zoa", "ltta", "sfo"):
    check_path(external / method, f"vendored {method}")

print(f"[info] python={sys.version.split()[0]}")
print(f"[info] torch={torch.__version__}")
if not torch.cuda.is_available():
    fail("CUDA is not available in this environment")
else:
    device = torch.device(device_name)
    torch.cuda.set_device(device)
    ok(f"CUDA available: {torch.cuda.get_device_name(device)}")

for package in ("torchvision", "timm"):
    try:
        mod = importlib.import_module(package)
        ok(f"import {package} {getattr(mod, '__version__', 'unknown')}")
    except Exception as exc:
        fail(f"import {package}: {exc!r}")

sys.path.insert(0, str(root))
sys.path.insert(0, str(tta_dir))
sys.path.insert(0, str(external / "eata"))
sys.path.insert(0, str(external / "zoa"))

for module in ("gpu_metrics", "model_factory", "tent", "eata"):
    try:
        importlib.import_module(module)
        ok(f"import {module}")
    except Exception as exc:
        fail(f"import {module}: {exc!r}")

for module_name, path in (
    ("vendored_sar", external / "sar" / "sar.py"),
    ("vendored_sam", external / "sar" / "sam.py"),
):
    try:
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        ok(f"import {module_name}")
    except Exception as exc:
        fail(f"import {module_name}: {exc!r}")

for path, label in (
    (external / "ltta" / "train_dwt_tta_se_online.py", "L-TTA entrypoint"),
    (external / "sfo" / "eval_bn_adapt.py", "SFO ResNet entrypoint"),
    (external / "sfo" / "eval_vit.py", "SFO ViT/MobileViT entrypoint"),
):
    check_path(path, label)

if torch.cuda.is_available():
    try:
        import timm

        model_specs = (
            ("resnet50", 224),
            ("mobilevit_xxs", 256),
            ("vit_base_patch16_224", 224),
        )
        device = torch.device(device_name)
        for model_name, image_size in model_specs:
            model = timm.create_model(model_name, pretrained=False, num_classes=1000).to(device).eval()
            x = torch.randn(1, 3, image_size, image_size, device=device)
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
                y = model(x)
            if torch.isfinite(y).all().item():
                ok(f"AMP FP16 forward {model_name} image_size={image_size} output={tuple(y.shape)}")
            else:
                fail(f"AMP FP16 forward {model_name}: non-finite output")
            del model, x, y
            torch.cuda.empty_cache()
    except Exception as exc:
        fail(f"model forward smoke: {exc!r}")

if errors:
    print("[preflight] FAILED")
    for item in errors:
        print(f" - {item}")
    raise SystemExit(1)

print("[preflight] PASSED")
PY

echo
echo "Next step if PASSED:"
echo "  bash TTA/scripts/nvme/run_all_nvme_fp16_single_model.sh"

# TTA GPU Profiling

This directory profiles full-model test-time adaptation workloads with the same
CUDA event and NVML metric path used by `gpu_utilizer/gpu_metrics.py`.

The measured TTA unit is:

```text
mini-batch adaptation 1 time + adapted-model inference 1 time
```

This is intentionally not layer-by-layer. TENT/EATA update the model at test
time, so the profiler measures the full ongoing overhead that would be paid in
the field.

## ResNet50 Command

```bash
cd /home/oem/TETC/gpu_utilizer
bash TTA/run_resnet50_tta.sh
```

Equivalent explicit command:

```bash
export LD_LIBRARY_PATH=""
python3 TTA/profile_tta.py \
  --model resnet50 \
  --model-source auto \
  --pretrained \
  --algorithms tent,eata \
  --batch-sizes 1,2,4,8,16,32,64,128 \
  --image-size 224 \
  --adapt-steps 1 \
  --fisher-samples 128 \
  --warmup 5 \
  --repeat 20 \
  --device cuda:0 \
  --gpu-index 0 \
  --input-mode synthetic \
  --output Results/TTA/resnet50_tent_eata_bs1_128.csv
```

For a quick debug run without pretrained weight download:

```bash
export LD_LIBRARY_PATH=""
python3 TTA/profile_tta.py \
  --model resnet50 \
  --no-pretrained \
  --algorithms tent,eata \
  --batch-sizes 1 \
  --warmup 1 \
  --repeat 1 \
  --no-idle-check \
  --output Results/TTA/debug_resnet50_tta.csv
```

## Robust Data Input

Synthetic input is the default and excludes data loading from the measured GPU
window. To profile on ImageNet-C tensors loaded before measurement:

```bash
export LD_LIBRARY_PATH=""
python3 TTA/profile_tta.py \
  --model resnet50 \
  --pretrained \
  --algorithms tent,eata \
  --batch-sizes 1,2,4,8,16,32,64,128 \
  --fisher-samples 128 \
  --input-mode imagenet-c \
  --data-corruption /path/to/imagenet-c \
  --corruption gaussian_noise \
  --level 5 \
  --output Results/TTA/resnet50_imagenet_c_gaussian_l5.csv
```

## Extension Notes

`--model-source auto` currently prefers the bundled EATA ResNet definitions for
ResNet/ResNeXt names, then falls back to `timm` or `torchvision`.

The bundled TENT/EATA implementation collects only `BatchNorm2d` affine
parameters. MobileViT models with BatchNorm can work through `timm` once `timm`
is installed. Pure ViT/LayerNorm models need a new parameter collector before
TENT/EATA adaptation can be measured correctly.

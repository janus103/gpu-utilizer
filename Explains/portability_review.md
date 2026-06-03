# Portability Review

## Copy/Paste Readiness

The `gpu_utilizer/` directory is designed to be copied into another server directory:

```text
A/
  main.py
  gpu_utilizer/
    config.ini
    profile_layers.py
    schema.py
    layer_templates.py
    gpu_metrics.py
    examples/
    Explains/
    Dependencies/
    Results/
```

The current profiler does not require importing or modifying `A/main.py`. It reads a layer CSV and creates supported PyTorch layer templates directly.

## Machine-Specific Paths

No Python source file requires an absolute path. `config.ini` uses paths relative to `gpu_utilizer/` by default. Users can override paths in `config.ini`.

## Public GitHub Suitability

The directory is safe to publish as a small standalone repository if generated outputs are excluded.

Recommended before publishing:

- Keep `.gitignore`.
- Do not commit `Results/*.csv` unless intentionally publishing example outputs.
- Do not commit machine-specific conda exports with private paths.
- Keep `Dependencies/environment.robust.yml` as a generic template.
- Add a license if this becomes a public repository.

## Measurement Caveats

- Default measurement requires NVML Python API and does not silently call `nvidia-smi`.
- `--nvidia-smi-fallback` is diagnostic only.
- Estimated SM cycles are time-normalized proxies, not simulator internal cycles.
- Very short workloads can produce zero NVML energy delta; those rows are labeled with an energy source such as `power_estimate_zero_delta`.

## Smoke Test Result

On the current A100 server, `examples/layers_example.csv` profiled all six supported layer types successfully:

- `standard_conv`
- `depthwise_conv`
- `pointwise_conv`
- `fully_connected`
- `matrix`
- `self_attention`

Results were written to `Results/gpu_profile_results.csv`.

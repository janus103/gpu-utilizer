# Metric Methodology

## What Is Measured

The profiler measures GPU execution of synthetic PyTorch operator templates generated from layer CSV rows. It is intended for layer-level GPU baselines, not for full application tracing.

## Latency

Latency uses CUDA Runtime event timing through PyTorch:

```python
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
...
end.record()
torch.cuda.synchronize()
elapsed_ms = start.elapsed_time(end)
```

Reported value:

```text
latency_ms_per_iter = cuda_elapsed_ms_total / repeat
```

## Frequency and Estimated Cycles

SM clock is read through NVML at the beginning and end of the profiling window.

```text
sm_clock_mhz_avg_used = (sm_clock_mhz_start + sm_clock_mhz_end) / 2
estimated_sm_cycles_per_iter = latency_ms_per_iter * sm_clock_mhz_avg_used * 1000
```

This is a time-normalized cycle proxy for the evaluated GPU. It is not claimed to be identical to Accel-Sim internal cycles or PIM command cycles.

## Energy and Power

Preferred energy source is NVML total energy counter:

```text
energy_j_per_iter = (energy_mj_end - energy_mj_start) / 1000 / repeat
avg_power_w_from_energy = energy_j_total / elapsed_s_total
```

For very short workloads, the NVML total energy counter can have zero delta. In that case the profiler reports:

```text
energy_source = power_estimate_zero_delta
energy_j_per_iter_est_from_power = avg_power_w_sampled * elapsed_s_total / repeat
```

This estimate is useful for diagnostics but should be clearly labeled in papers.

## NVML vs nvidia-smi

Default mode requires NVML Python API through `pynvml` / `nvidia-ml-py`.

`nvidia-smi` is only used when explicitly enabled:

```bash
python3 profile_layers.py --nvidia-smi-fallback
```

This is intentionally not the default because CLI sampling is coarser and can be viewed less favorably in paper methodology.

## Reviewer-Facing Claim

For a single-layer GPU baseline, this profiler uses direct hardware execution on the evaluated GPU rather than trace-driven GPU simulation. Latency is measured with CUDA events after warmup, SM-cycle estimates are normalized from measured latency and observed SM clock, and energy/power are obtained from NVML. This avoids simulator replay overhead and simulator-configuration mismatch. We do not claim that estimated SM cycles are identical to simulator internal cycles; we use them only as a time-normalized latency proxy.

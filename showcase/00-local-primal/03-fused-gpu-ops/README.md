# 03 — Fused GPU Operations

Demonstrates barraCuda's fused single-dispatch GPU operations and zero-readback
GpuView persistent buffer chains.

## What It Shows

- **Fused Welford**: mean + variance in a single GPU dispatch (vs 2-pass naive)
- **Fused Correlation**: 5-accumulator Pearson r from a single kernel launch
- **GpuView**: Data stays on GPU between operations — zero host-device round-trips
- **Zero-readback chains**: Compute stats on GPU-resident data without downloading

## Why This Matters

Each GPU dispatch has overhead (~50-200us for command encoding + submission).
Fused operations eliminate intermediate readbacks:

```
Naive:       upload -> mean -> download -> upload -> variance -> download
Fused:       upload -> mean+variance -> download (1 dispatch, 1 readback)
GpuView:     upload -> mean_variance -> correlation -> download (data stays on GPU)
```

## Run

```bash
cargo run --release
```

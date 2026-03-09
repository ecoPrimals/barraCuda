# 01 — Device Discovery

Demonstrates barraCuda's runtime GPU discovery and capability-based routing.

## What It Shows

- Automatic GPU detection via wgpu (NVIDIA, AMD, Intel, Apple, software)
- Capability scoring: f64 support, workgroup sizes, memory limits
- Precision routing advice (`Fp64Strategy`: Native, Hybrid, Concurrent, Sovereign)
- Vendor-specific optimization insights
- Workload-aware workgroup sizing

## Run

```bash
cargo run --release
```

## No Hardcoding

All values are discovered at runtime. Zero hardcoded vendor names, port numbers,
or performance estimates. The same binary adapts to any WebGPU-capable device.

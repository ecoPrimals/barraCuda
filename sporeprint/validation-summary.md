+++
title = "barraCuda Validation Summary"
description = "GPU-accelerated scientific computing engine — 4,393+ tests, 90 IPC methods, 15-tier precision ladder, pure safe Rust, A+ grade"
date = 2026-05-20

[taxonomies]
primals = ["barracuda"]
springs = ["hotspring", "primalspring", "wetspring", "airspring"]
+++

## Status

- **4,393+ tests** (nextest CI profile), 0 failed, 80.54% line / 83.45% function coverage
- **87 registered IPC methods** across 22 semantic namespaces
- **1,163 Rust source files**, zero unsafe in production (`#![forbid(unsafe_code)]`)
- **v0.4.0** — stadial gate cleared (Wave 22), all checklist items green
- **A+ grade** — zero unwrap/panic/expect in production, zero println in library, zero `Result<T, String>`, zero mocks in production
- **Pure Rust** — zero C dependencies (`deny.toml` bans ring, openssl, aws-lc-sys)
- **15-tier precision ladder** — F16 → BF16 → TF32 → F32 → DF64 → F64 → F64Precise → DF128 → QF128 → FP8 → INT2 → Binary → Quantized4 → Quantized8
- **Compute Trio member** — sovereign pipeline: barraCuda (workload) → coralReef (compiler) → toadStool (hardware)

## Key Capabilities

| Domain | Methods | Description |
|--------|---------|-------------|
| `tensor.*` | 9 | GPU/CPU tensor ops (create, matmul, add, scale, clamp, reduce, sigmoid, batch.submit) |
| `stats.*` | 22 | Statistics, regression, ecology (mean, variance, correlation, chi_squared, fit_quadratic/exp/log, simpson, bray_curtis, hill, rarefaction, gamma_fit/cdf) |
| `linalg.*` | 5 | Linear algebra (solve, eigenvalues, svd, qr, graph_laplacian) |
| `ml.*` | 4 | Machine learning (mlp_forward, mlp_train, attention, esn_predict) |
| `spectral.*` | 3 | FFT, power spectrum, STFT |
| `fhe.*` | 2 | Fully homomorphic encryption (NTT, pointwise_mul) |
| `precision.*` | 1 | Precision routing advisory with dispatch_path differentiation |
| `nautilus.*` | 6 | Anomaly detection sessions (Path B server-side) |
| `signal.*` | 3 | Signal processing (detect_peaks, bandpass, derivative) |
| `health.*` | 4 | Liveness, readiness, check, version |
| `btsp.*` | 2 | Cipher negotiation + capabilities |

## Architecture

- **4-layer validation**: llvmpipe → NagaExecutor → coralReef CPU → real GPU
- **WGSL-as-truth**: 337 op test files, GPU shaders validated at all layers
- **TensorSession**: fused multi-op GPU pipeline (add, sub, mul, negate, fma, scale)
- **PrecisionBrain**: self-routing precision advisory, dispatch_path differentiation
- **Sovereign dispatch**: `KernelTarget::Sovereign` routes tensor-core-eligible workloads to toadStool via JSON-RPC
- **OOM detection**: `is_oom()` + `is_retriable()` classification for multi-GPU scenarios
- **BTSP Phase 3**: ChaCha20-Poly1305 stream encryption on all accept loops

## Downstream Consumers

- **hotSpring** — 3-GPU sovereign compute (Titan V, K80, RTX 5060), lattice QCD, nuclear EOS
- **wetSpring** — Tenaillon 2016 (264-clone LTEE), pairwise L2 GPU dispatch
- **airSpring** — cross-tier parity benchmarks
- **primalSpring** — composition validation, typed extractors

## Workload TOMLs

Not yet created — contribute to `projectNUCLEUS/workloads/barracuda/`.

## See Also

- [Primal Catalog](https://primals.eco/architecture/) on primals.eco
- `specs/TENSOR_WIRE_CONTRACT.md` — full IPC wire contract + stability tiers
- `specs/REMAINING_WORK.md` — evolution roadmap

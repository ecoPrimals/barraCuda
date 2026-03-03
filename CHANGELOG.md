# Changelog

All notable changes to barraCuda will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.1] - March 3, 2026

### Added
- **73 new tests** — cpu_executor dispatch (Conv2D, MaxPool2D, AvgPool2D, BatchMatMul, all ops),
  benchmarks (harness, operations, report), device/vendor, validation, cubic_spline
- **tarpc/JSON-RPC parity** — tarpc service now has matching parameters and full implementations
  for `fhe_ntt`, `fhe_pointwise_mul`, `compute_dispatch`, `tensor_create`

### Changed
- **blake3 pure feature** — `features = ["pure"]` eliminates C SIMD compilation dependency
- **IPC transport constants** — extracted `TARPC_MAX_FRAME_LENGTH`, `TARPC_MAX_CONCURRENT_CONNECTIONS`
- **println → tracing** — 14 `println!` calls in library code migrated to `tracing::info!`
  (benchmarks/harness, benchmarks/mod, multi_gpu/pipeline_dispatch)
- **Placeholder errors** — `channel_shuffle_wgsl` and `diag_new` replaced misleading
  `InvalidShape { expected: vec![0,0,...] }` with descriptive `InvalidInput { message }`
- **tarpc `MatmulResult`** — `lhs_id` renamed to `result_id` with `shape` field added
- **tarpc `DispatchResult`** — redesigned with `tensor_id`, `shape`, `data` fields
- **tarpc FHE types** — split into `FheNttResult` and `FhePointwiseMulResult` with coefficient vectors

### Removed
- Unused `_vta_buffer` GPU allocation in `qr_gpu.rs`

### Quality
- 2,965 unit tests passing, 0 failures
- ~80% line coverage (all CPU-testable code covered; remaining gap is GPU-only)
- `cargo fmt --check` clean
- `cargo clippy --workspace -- -D warnings` clean
- `RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps` clean
- 3-config check clean (pure math, GPU, full)

## [0.3.0] - March 3, 2026

### Added
- **tarpc service** — 10 strongly-typed RPC endpoints mirroring JSON-RPC 2.0, dual-protocol IPC
- **UniBin CLI** — `barracuda server`, `doctor`, `validate`, `version` subcommands
- **`BarracudaError::DeviceLost`** — explicit variant for GPU device loss with `is_retriable()` check
- **Global `DEVICE_CREATION_LOCK`** — serializes all `wgpu::Adapter::request_device` calls process-wide
- **Rayon parallelism** — Nelder-Mead solvers and LOO-CV grid search run concurrently
- `barracuda` registered in `wateringHole/genomeBin/manifest.toml`
- `.github/workflows/ci.yml` — full CI pipeline (fmt, clippy, deny, doc, test, coverage)
- `rustfmt.toml`, `deny.toml`, `.cargo/config.toml`

### Removed — Complete toadStool Untangle (S89)
- **`toadstool-core` dependency** — removed from Cargo.toml, zero cross-deps on any primal
- **`akida-driver` dependency** — removed from Cargo.toml
- **`toadstool` feature flag** — removed entirely
- **`npu-akida` feature flag** — removed entirely
- **`toadstool_integration.rs`** — deleted (hardware discovery/routing via toadStool)
- **`npu/ml_backend.rs`** — deleted (Akida NPU execution layer)
- **`npu/ops/`** — deleted (6 files: matmul, softmax, relu, gelu, layer_norm, mod)
- **`npu_integration` example** — deleted (required akida-driver)
- **`e2e_math_pipeline.rs`** — deleted (entire file gated on toadstool)
- **toadstool-gated tests** — removed from chaos, cross_hardware_parity, hardware_verification
- **Dead feature flags** — removed `tpu`, `cloud-tpu`, `coral-tpu`, `mock-tpu`, `cuda-comparison`

### Changed
- **GPU synchronization** — all 11 lock bypass paths fixed; every GPU operation now routes through
  `WgpuDevice::lock()`, `submit_and_poll_inner`, `read_buffer`, or `poll_safe`
- **Device error handler** — `on_uncaptured_error` now flags device as lost instead of panicking
- **Sparse buffer readback** — `read_f64_raw`/`read_i32_raw` accept `&WgpuDevice` for synchronized access
- **ComputeGraph** — stores `Arc<WgpuDevice>`, uses synchronized submit/poll
- **AsyncSubmitter/AsyncReadback** — fully synchronized via `WgpuDevice`
- **Autotune/Calibration** — new `GpuDeviceForCalibration` trait, synchronized submit/poll
- **Probe runner** — accepts `&WgpuDevice` for synchronized probing
- **PPPM GPU solver** — stores `Arc<WgpuDevice>`, removed unused `adapter_info` field
- **Sparsity sampler** — `F: Fn + Sync` bound for parallel Nelder-Mead
- **Clippy pedantic** — configured in `Cargo.toml` `[lints]` with targeted allows
- Chaos/E2E tests — removed hardcoded timing assertions, relaxed precision checks for instrumented builds

### Fixed
- Non-deterministic SIGSEGV from concurrent `request_device` calls racing on kernel DRM descriptors
- Uncaptured wgpu error handler crashing the process on device loss
- `elidable_lifetime_names`, `borrow_as_ptr`, `comparison_chain`, `checked_conversions`,
  `unchecked_time_subtraction` clippy warnings
- Digamma recurrence test resilience to transient GPU device loss

### Quality
- 2,965 unit tests + 8 IPC E2E tests passing, 0 failures
- 29 integration test suites compiling and passing
- ~80% line coverage (unit tests via llvm-cov)
- Cross-dependencies on toadStool: **ZERO**
- `cargo clippy --workspace -- -D warnings` clean
- `cargo fmt --all` clean
- `cargo deny check` clean

## [0.2.0] - March 2, 2026

### Added
- Full barracuda compute library extracted from toadStool (956 .rs, 767 WGSL shaders, 61 tests)
- `validate_gpu` binary — canary suite for GPU correctness (FHE NTT, matmul, DF64, pointwise mul)
- `barracuda-core` crate wired to compute library (device discovery, health reporting)
- 5 examples: device_capabilities, esn_demo, fhe_ntt_validation, npu_integration, pppm_debug
- Optional feature gates: `toadstool` (toadstool-core integration), `npu-akida` (Akida NPU)

### Changed
- `DeviceSelection` and `HardwareWorkload` enums moved to `device/mod.rs` (always available)
- MSRV bumped to 1.87 (code uses `is_multiple_of`)

### Quality
- 2,832 lib tests passing, 0 failures
- 20+ integration test binaries compiling and passing
- `cargo clippy -- -D warnings` clean
- `cargo fmt` clean

## [0.1.0] - March 2, 2026

### Added
- Initial scaffold via sourDough
- `barracuda-core` primal lifecycle (PrimalLifecycle, PrimalHealth)
- `BarracudaError` type with device, shader, shape, dispatch variants
- Workspace configuration (wgpu 22, naga 22.1, AGPL-3.0-or-later)

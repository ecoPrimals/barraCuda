# Changelog

All notable changes to barraCuda will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Fused mean+variance shader** (`shaders/reduce/mean_variance_f64.wgsl`) — single-pass
  Welford algorithm with grid-stride loop and workgroup tree reduction. Computes both
  mean and variance in one GPU dispatch, eliminating intermediate CPU round-trips.
  Absorbed from Kokkos `parallel_reduce` patterns
- **Fused correlation shader** (`shaders/stats/correlation_full_f64.wgsl`) — 5-accumulator
  single-pass Pearson correlation (sum_x, sum_y, sum_xx, sum_yy, sum_xy). Returns
  mean_x, mean_y, var_x, var_y, and pearson_r from a single kernel launch. Absorbed
  from Kokkos `parallel_reduce` with `JoinOp` patterns
- **`CorrelationResult` struct** — rich return type from fused correlation with all
  five statistics (means, variances, Pearson r) from a single dispatch
- **`VarianceF64::mean_variance()`** — returns `[mean, variance]` from a single fused
  GPU pass
- **`TensorContext::acquire_pooled_output_f64()`** — f64-sized pooled buffer allocation
- **`TensorContext::acquire_pooled_bytes()`** — raw byte-sized pooled buffer allocation

### Changed
- **Stats ops evolved to TensorContext path** — `mean.rs`, `sum.rs`, `prod.rs`
  migrated from raw `ComputeDispatch` with per-call buffer allocation to
  `TensorContext` with pooled buffers, pipeline cache, and bind group cache.
  Eliminates per-op buffer allocation overhead in steady state
- **`VarianceF64` fused dispatch** — evolved from 2-pass (mean → deviation) via
  `ComputeDispatch` to single-pass Welford via `TensorContext` + pipeline cache
- **`CorrelationF64` fused dispatch** — evolved from multi-dispatch via
  `ComputeDispatch` to single 5-accumulator pass via `TensorContext` + pipeline cache
- **Comprehensive codebase audit** — full pass across all quality gates, sovereignty,
  documentation, error handling, and idiomatic Rust patterns (736 files changed)
- **Documentation completeness** — added `///` doc comments to all undocumented `pub`
  items across ~300 files, resolving all `missing_docs` warnings. `RUSTDOCFLAGS="-D warnings"`
  now passes clean
- **Sovereignty compliance** — replaced all hardcoded primal names (`hotSpring`,
  `wetSpring`, `neuralSpring`, `toadStool`) in production code and tests with
  capability-based identifiers (`lattice_qcd`, `marine_bio`, `ml_inference`,
  `orchestration layer`)
- **Error handling evolution** — replaced `expect()`/`panic!()` in production code
  with `Result<T, BarracudaError>` returning `InvalidInput` or `Internal` variants
- **Magic number extraction** — replaced bare numeric literals with named constants
  (`BYTES_PER_MB`, `LARGE_INPUT_BUFFER_MB`, etc.) in staging and GPU executor
- **`Arc<WgpuDevice>` removal** — `BarraCudaPrimal` now stores `Option<WgpuDevice>`
  directly, cloning only where `Tensor` APIs require `Arc`
- **Lint cleanup** — fixed all unfulfilled `#[expect]` annotations, resolved
  `inclusive_range` and `large_stack_arrays` diagnostics, added `cfg_attr(test, ...)`
  for test-only lint suppressions
- **CI coverage enforcement** — added `--fail-under-lines 80` to `cargo llvm-cov`
  and artifact upload for `lcov.info`
- **`deny.toml` cleanup** — removed unused license allowances (`AGPL-3.0`,
  `BSD-3-Clause`, `BSL-1.0`, `MPL-2.0`, `Unicode-DFS-2016`)

### Quality
- `cargo fmt --all -- --check` — clean
- `cargo clippy --workspace --all-targets --all-features -- -D warnings` — zero warnings
- `RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps` — clean
- `cargo deny check` — advisories/bans/licenses/sources OK
- 3,471 test functions across 62 integration test suites
- 80%+ line coverage enforced in CI

## [0.3.3] - March 4, 2026

### Changed
- **wgpu 22 → 28 + naga 22.1 → 28** — major GPU stack upgrade. All wgpu API
  changes propagated across the codebase (~800 call-site updates):
  - `Maintain::Wait` / `MaintainBase::Poll` → `PollType::Wait` / `PollType::Poll`
  - `create_shader_module_spirv` → `create_shader_module_passthrough`
  - `push_constant_ranges` removed; `immediate_size` added to `PipelineLayoutDescriptor`
  - `entry_point` now `Option<&str>` in pipeline descriptors
  - `set_bind_group` second argument now `Option<&BindGroup>`
  - `request_adapter` returns `Result` (was `Option`)
  - `DeviceDescriptor` gains `experimental_features` and `trace` fields
  - `on_uncaptured_error` handler evolved to `Arc<dyn UncapturedErrorHandler>`
  - `pop_error_scope` → `ErrorScopeGuard` pattern via `push_error_scope().pop()`
  - Naga IR: new `Statement` / `Expression` variants for barriers, atomics, ray queries
- **`Arc<wgpu::Device>` / `Arc<wgpu::Queue>` removed** — wgpu 28 makes `Device` and
  `Queue` internally `Clone`. Removed redundant `Arc` wrappers from `GuardedDeviceHandle`,
  `WgpuDevice`, `BufferPool`, `PppmGpu`, `ComputeGraph`, and `PppmPipelines`.
  `device_arc()` → `device_clone()`, `queue_arc()` → `queue_clone()`,
  `inner_arc()` removed, `from_existing()` takes plain types
- **tokio 1.40 → 1.49** — workspace dependency bumped to current stable
- **Dependency alignment** — `serde_json` now uses `workspace = true` in barracuda
  crate; tokio dev-dependency aligned with workspace (was pinned to 1.35)
- **Workgroup size constants** — introduced `WORKGROUP_SIZE_COMPACT = 64` alongside
  existing `WORKGROUP_SIZE_1D = 256` in `device::capabilities`. Replaced ~80 bare
  `div_ceil(64)` and `div_ceil(256)` magic numbers across 68 files with named constants
- **Lint cleanup** — fixed 33 unfulfilled `#[expect]` annotations: removed stale
  `dead_code` / `unused_imports` expectations, correctly classified dead entry-point
  functions vs. transitively-live helpers, removed unused `wgpu::util::DeviceExt` imports

### Fixed
- `wgpu::Id` removed in wgpu 28 — replaced `buffer.global_id()` with stable hash and
  `device.global_id()` with `format!("{device:?}")` / `device.hash()`
- `wgpu::Features::SPIRV_SHADER_PASSTHROUGH` constant removed — `has_spirv_passthrough()`
  now checks `adapter_info.backend == Backend::Vulkan` (SPIR-V passthrough is a Cargo feature)
- `enumerate_adapters()` now async — all call sites updated with `.await` or `pollster::block_on`
- `AdapterInfo` new required fields (`device_pci_bus_id`, `subgroup_min_size`,
  `subgroup_max_size`, `transient_saves_memory`) — populated in all manual constructors

### Quality
- `cargo check --workspace --all-features` clean
- `cargo clippy --workspace --all-features` — zero warnings
- `cargo deny check` — advisories/bans/licenses/sources OK
- `cargo fmt --all` clean
- 112/112 device tests passing
- Zero unfulfilled `#[expect]` annotations in test profile

## [0.3.2] - March 3, 2026

### Added
- **3 new ET₀ operations** — `MakkinkEt0` (op 14), `TurcEt0` (op 15), `HamonEt0` (op 16)
  with WGSL shader implementations and CPU reference functions
- **`GuardedDeviceHandle`** — RAII-wrapped `wgpu::Device` that automatically protects all
  `create_*` calls with atomic encoder barriers, eliminating wgpu-core races codebase-wide

### Removed
- **`sourdough-core` dependency** — lifecycle (`PrimalLifecycle`, `PrimalState`) and health
  (`PrimalHealth`, `HealthStatus`, `HealthReport`) traits internalized into `barracuda-core`.
  barraCuda is now fully standalone with zero cross-primal dependencies
- **`async-trait` dependency** — replaced with native `BoxFuture` type alias and `Box::pin`
  for object-safe async trait methods
- **Dead feature flags** — `tpu`, `cloud-tpu`, `coral-tpu`, `mock-tpu`, `unidirectional`
- **`tpu.rs` module and `unidirectional_benchmark.rs`** — dead code removed
- **`sourDough` CI checkout** — removed all 6 `actions/checkout@v4` steps from CI

### Changed
- **GPU concurrency overhaul** — replaced `WgpuDevice::lock()` RwLock with a three-layer model:
  `active_encoders: AtomicU32` for lock-free encoder tracking, `gpu_lock: Mutex<()>` for
  submit/poll serialization, and a bounded yield loop (`brief_encoder_wait`) before poll
- **`GuardedEncoder` redesign** — now an RAII wrapper holding `Option<CommandEncoder>` and the
  `active_encoders` Arc; auto-decrements on finish or drop, making the barrier leak-proof
- **`encoding_guard()` / `encoding_complete()`** — explicit atomic increment/decrement pair
  applied to all `WgpuDevice` buffer creation, shader compilation, and `ComputeDispatch::submit`
  to prevent wgpu-core races between resource creation and `device.poll()`
- **Device-lost discrimination** — `on_uncaptured_error`, `submit_commands`, `poll_safe`, and
  `submit_and_poll_inner` now only flag `lost = true` for genuine device-lost errors; validation
  errors are logged or re-panicked without poisoning the shared device for other threads
- **`BufferPool` concurrency** — `poll_lock` changed to `Mutex`, `drain_pending` checks
  `active_encoders` before attempting non-blocking poll, `allocate_new` protected with
  encoding guard
- **`AsyncSubmitter` / `AsyncReadback`** — updated from `RwLock::write()` to `Mutex::lock()`,
  added `brief_encoder_wait()` before submissions
- **`#[allow]` → `#[expect]`** — converted all clippy suppressions to `#[expect(reason)]`
  for compile-time verification of necessity
- **`rand` 0.8 → 0.9** — updated to latest rand crate
- **Clippy tightening** — reduced bulk `Cargo.toml` allows, fixed `type_complexity` with
  `BoxFuture` type alias, resolved `deref`, `range_plus_one`, struct field order warnings

### Fixed
- wgpu-core "Buffer does not exist" panics under concurrent GPU access
- Cascading `DeviceLost` failures from transient validation errors on shared test devices
- `RwLock` convoy effect causing test hangs at 16+ threads on llvmpipe
- Unprotected `device.device.create_*()` calls in `expand`, `ComputeDispatch`, buffer and
  shader creation racing with `device.poll()`
- NVK reciprocal bug in 3 WGSL shaders — replaced `/ f64(4294967296.0)` with reciprocal
  multiplication `* f64(2.3283064365386963e-10)` for numerical stability on NVIDIA Vulkan

### Quality
- 1,791+ test functions, 0 concurrency-related failures at 16 threads on llvmpipe
- ~80% line coverage (all CPU-testable code covered; remaining gap is GPU-only)
- `cargo fmt --check` clean
- `cargo clippy --workspace` clean (zero warnings)
- `RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps` clean
- 3-config check clean (pure math, GPU, full)

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
- Workspace configuration (wgpu 22, naga 22.1, AGPL-3.0-or-later) — upgraded to wgpu 28 + naga 28 in 0.3.3

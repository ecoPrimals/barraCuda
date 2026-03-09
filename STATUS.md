# barraCuda Status

**Version**: 0.3.3
**Date**: 2026-03-09
**Overall Grade**: A+ (Zero unsafe, pure safe Rust, all quality gates green, 3,450+ tests, GpuBackend trait abstraction, sovereign dispatch scaffold, zero-copy bytemuck/Bytes, showcase collection with 10 progressive demos, all deps pure Rust, zero hardcoded workgroup sizes)

---

## Category Grades

| Category | Grade | Notes |
|----------|-------|-------|
| **Core compute** | A | 791 WGSL shaders, 13-tier tolerance architecture, GpuView persistent buffers with ops |
| **Precision tiers** | A+ | 3-tier model (F32/F64/Df64) aligned with coralReef `Fp64Strategy`; DF64 naga-guided rewrite validated; probe-aware Fp64Strategy; DF64 reduce shaders for Hybrid devices |
| **Sovereign compiler** | A | FMA fusion + dead expr elimination + safe WGSL roundtrip (all backends); sovereign validation harness covers all shaders |
| **IPC / primal protocol** | A+ | JSON-RPC 2.0 (notification-compliant) + tarpc; Unix socket default + TCP; capability-based discovery; coralReef Phase 10 `shader.compile.*` semantic naming; AMD arch support |
| **Device management** | A+ | `GpuBackend` trait abstraction, `CoralReefDevice` scaffold behind `sovereign-dispatch` feature, multi-GPU, capability-scored discovery, probe-aware f64 strategy, bounded poll timeout |
| **Test coverage** | A | 3,450+ tests (all pass on llvmpipe); proptest; chaos/fault test tiers; nextest CI/stress profiles; bounded GPU poll timeout; GPU-heavy test group with extended timeouts; coverage tests for batch_ipr, histogram, staging, precision/cpu, surrogate/adaptive |
| **Dependencies** | A+ | All deps pure Rust (blake3 `pure`, wgpu/naga 28); zero application C deps; ecoBin compliant |
| **Documentation** | A+ | Comprehensive CHANGELOG, specs, README, CONTRIBUTING, CONVENTIONS, BREAKING_CHANGES; all rustdoc warnings resolved; showcase/ with 10 progressive demos (local, IPC, cross-primal) |
| **Unsafe code** | A+ | Zero `unsafe` blocks in entire codebase |
| **Clippy / lint** | A+ | Zero warnings with pedantic + unwrap_used; `#[expect(reason)]` for clippy suppressions; `#[allow(dead_code, reason)]` for CPU reference implementations; `bytes::Bytes` zero-copy on I/O boundaries; zero undocumented suppressions |
| **Error handling** | A+ | Binary `main()` uses typed `BarracudaCoreError` (not `Box<dyn Error>`); `From` impls for `serde_json::Error`, `BarracudaError`, `io::Error`; `Result` propagation throughout; `let-else` throughout; poison recovery |
| **Idiomatic Rust** | A+ | Edition 2024; zero `too_many_arguments` (all 9 → builder/struct); documented `#[allow]`/`#[expect]` with reason; `#[derive(Default)]`; zero unsafe; `ChamferDirection` enum; smart module decomposition (provenance, coral_compiler) |
| **Spring absorption** | A+ | Cross-spring P0/P1/P2 items resolved (Mar 8); healthSpring Hill dose-response (Emax) + Population PK Monte Carlo absorbed; hotSpring plasma dispersion W(z)/Z(z) CPU stable implementations absorbed (ISSUE-006); `hill_activation`/`hill_repression` from neuralSpring; head_split/head_concat WGSL confirmed aligned with neuralSpring; Ada Lovelace `F64NativeNoSharedMem` (groundSpring P0); `shared_mem_f64` runtime probe (groundSpring P1); DF64 reduce fix; builder re-exports for wetSpring; `dot`/`l2_norm` for springs; canary/test utils; NVK guard; GpuView ops; all shader targets verified absorbed |

---

## What's Working

- 3-tier precision model (F32/F64/Df64) aligned with coralReef's `Fp64Strategy`
- Naga-guided DF64 infix rewrite (compound assignments, comparisons, nested ops)
- DF64 Hybrid path: 10 ops now fail loudly on rewrite failure instead of silently producing zeros
- **P0 fix**: `SumReduceF64`/`VarianceReduceF64` now route through DF64 shaders on Hybrid devices (was returning zeros)
- Builder type re-exports at `barracuda::{HmmForwardArgs, Dada2DispatchArgs, GillespieModel, PrecisionRoutingAdvice, Rk45DispatchArgs}`
- `barracuda::math::{dot, l2_norm}` for springs to drop local implementations
- `fused_ops_healthy()` canary, `is_software_adapter()`, `baseline_path()` in test harness
- `GpuDriverProfile::f64_zeros_risk()` for NVK + Ada Lovelace proprietary shared-memory f64 detection
- `barracuda::stats::{hill_activation, hill_repression}` — Hill kinetics for regulatory networks
- Ada Lovelace + proprietary → `F64NativeNoSharedMem` precision routing (probe-aware)
- `shared_mem_f64` runtime probe — empirically verifies `var<workgroup>` f64 reductions on hardware
- `GpuViewF64::{mean_variance, sum, correlation}` ops for zero-readback chains
- Sovereign compiler: WGSL → naga IR → FMA fusion → dead expr elimination → optimised WGSL (safe, all backends)
- Sovereign validation harness: pure-Rust traversal + parse + optimize + validate of all WGSL shaders
- GpuView persistent buffer API for zero-copy GPU-resident computation
- GPU-resident reduction pipeline (`encode_reduce_to_buffer` + `readback_scalar`)
- 13-tier numerical tolerance architecture (DETERMINISM through EQUILIBRIUM)
- JSON-RPC 2.0 (notification-compliant per spec) + tarpc IPC with Unix socket (default) and TCP transport
- Capability-scored multi-GPU adapter discovery
- Probe-aware Fp64Strategy (NVK f64 detection via runtime probe cache)
- GPU f64 computational accuracy probe (dispatches `3*2+1=7` to verify real f64 execution)
- Capability-based shader-compiler discovery (env → `shader.compile` capability scan → `shader_compiler` fallback → well-known port)
- Bounded GPU poll timeout (configurable via `BARRACUDA_POLL_TIMEOUT_SECS`, default 120s)
- RwLock poison recovery in autotune (no panics on poisoned calibration cache)
- Graceful Tokio runtime detection in coral compiler spawn
- LSCFRK gradient flow integrators (W6, W7, CK45) with algebraic coefficient derivation
- NautilusBrain force anomaly detection (10σ energy deviation, rolling window)
- `GpuBackend` trait (`device::backend`) — backend-agnostic compute interface; `WgpuDevice` + `Arc<WgpuDevice>` implement it; `ComputeDispatch<B: GpuBackend>` generic over backend
- `CoralReefDevice` scaffold behind `sovereign-dispatch` feature flag — ready for `coral-gpu` crate
- `SOVEREIGN_PIPELINE_TRACKER.md` — tracks P0 (CoralReefDevice), libc→rustix evolution, cross-primal deps
- Zero TODOs/FIXMEs/HACKs/`unreachable!()` without messages in codebase
- Zero `#[expect(clippy::too_many_arguments)]` — all 9 evolved to builder/struct patterns
- All quality gates green (fmt, clippy -D warnings, rustdoc -D warnings, deny)
- Compile-time verified `#[expect(reason)]` for lint suppressions; `#[allow(dead_code, reason)]` on all CPU reference implementations
- coralReef IPC client aligned to Phase 10 semantic naming (`shader.compile.spirv/wgsl/status/capabilities`)
- AMD RDNA2 (`gfx1030`), RDNA3 (`gfx1100`), CDNA2 (`gfx90a`) architecture mappings for coralReef
- `shader.compile.capabilities` preferred for arch enumeration with health-response fallback
- Backward-compat fallback for pre-Phase 10 coralReef (probe + discovery)
- Cross-spring shader provenance registry with Write → Absorb → Lean tracking
- Deprecated PPPM constructors removed (zero callers)
- Akida SDK paths extracted to shared capability constant
- `PrecisionRoutingAdvice` from toadStool S128 (`F64Native`, `F64NativeNoSharedMem`, `Df64Only`, `F32Only`)
- `BatchedOdeRK45F64` adaptive Dormand-Prince integrator on GPU (wetSpring V95)
- `mean_variance_to_buffer()` GPU-resident fused Welford (zero CPU readback for chained pipelines)
- Cross-spring evolution timeline with 10 events + dependency matrix + 27 dated shader records
- `ChamferDirection` enum — evolved from raw u32 to exhaustive-match type-safe direction
- Smart module decomposition: `provenance/` (types/registry/report); `coral_compiler/` (types/discovery/cache/jsonrpc/client)
- All `#[allow(dead_code)]` on CPU reference implementations documented with `reason` parameter
- Magic numbers evolved to named constants (workload thresholds, discovery filenames)
- Zero `unreachable!()` without descriptive messages
- `service` subcommand for genomeBin compliance (systemd integration, PID file, READY=1)
- Dynamic capability derivation from `REGISTERED_METHODS` source of truth in discovery files
- Thread-local GPU test throttling via `OwnedSemaphorePermit` — stable `cargo test` at any parallelism
- `bytes::Bytes` zero-copy for `TensorStorage::read_to_cpu()`, staging `WorkUnit`/`CompletedWork`
- RPC `String` parameter documentation for serde boundary correctness
- `eprintln!` → `tracing::warn!` in sovereign validation harness (library code)
- Binary `main()` typed to `BarracudaCoreError` — zero `Box<dyn Error>` in codebase
- `From<serde_json::Error>` and `From<BarracudaError>` conversions in `BarracudaCoreError`
- Hardcoded `"127.0.0.1"` → `LOCALHOST` constant in coral discovery
- Hardcoded `"2.0"` → `JSONRPC_VERSION` constant in JSON-RPC protocol layer
- CPU executor magic numbers evolved to `defaults::` named constants
- `is_retriable()` covers buffer validation errors (not just device-lost)
- `with_device_retry` gracefully skips on persistent llvmpipe instability
- Flaky GPU tests (erf, erfc, expand, determinant) evolved from `catch_unwind` to `with_device_retry` — production recovery pattern
- CI evolved to nextest with `ci`/`stress` profiles; chaos/fault/property test tier added
- Coverage job uses `BARRACUDA_POLL_TIMEOUT_SECS` and soft-gates at 80% (90% requires real GPU)
- `SparseGemmF64` and `PeakDetectF64` now compile via `compile_shader_f64()` (was incorrectly using `compile_shader()` which downcasts f64→f32, causing data corruption on non-f64 GPUs)
- f64 GPU ops gated on `get_test_device_if_f64_gpu_available()` — no more false failures on llvmpipe
- GPU performance estimation refactored from 13 constants to `fallback_estimates::{gflops, vram_bytes}` pattern-matched functions
- NPU SIMD width extracted to `NPU_SIMD_WIDTH` constant
- `ShaderCompilation(Arc<str>)` — error type evolved from `String` to `Arc<str>` for zero-copy error propagation across 10 DF64 shader paths
- ~50 GPU dispatch paths evolved from `to_le_bytes().collect::<Vec<u8>>()` to `bytemuck::cast_slice()` — zero-copy buffer uploads
- `GpuBackend::download()` returns `bytes::Bytes` instead of `Vec<u8>` — zero-copy GPU readback
- `NpuTensorStorage` evolved from `Vec<u8>` to `bytes::BytesMut` with `freeze()` zero-copy read
- GPU-heavy test group with extended timeouts for edge_conv, fft, conv2d, flash_attention
- Coverage tests added for batch_ipr, histogram, staging/ring_buffer, staging/unidirectional, staging/stateful, precision/cpu, surrogate/adaptive
- CI dual coverage targets: 80% baseline (llvmpipe) + 90% stretch (GPU hardware)
- `showcase/` collection: 10 progressive demos across 3 tiers (local primal, IPC protocol, cross-primal compute)
- Showcase demonstrates: device discovery, precision tiers, fused GPU ops, science shaders, JSON-RPC server, doctor/validate, coralReef compilation, toadStool discovery, sovereign pipeline
- Zero `panic!()` in production library code (all panics restricted to `#[cfg(test)]` modules)
- **Systematic f64 pipeline fix**: 14 ops (transe_score, triangular_solve, variance, correlation, covariance, hermite, bessel_i0/j0/j1/k0, beta, digamma, cosine_similarity, weighted_dot) evolved from `compile_shader()`/`GLOBAL_CACHE` to f64-native compilation paths — eliminates silent data corruption on f64-capable GPUs
- Pipeline cache evolved with f64-native compilation path (`get_or_create_pipeline_f64_native`) — separate cache maps prevent f64/f32 key collisions
- `create_f64_data_pipeline()` helper auto-selects native or downcast path based on device `SHADER_F64` capability
- `compile_shader()` doc evolved to accurately describe f64-canonical always-downcast behavior
- `CpuTensorStorageSimple` evolved from `Vec<u8>` to `Bytes` — `read_to_cpu()` is now zero-copy (ref-count bump instead of full buffer clone)
- `CosineSimilarityF64::similarity()` zero-copy: eliminated unnecessary `to_vec()` pair via flat-dispatch refactor
- Pipeline cache `DeviceFingerprint` evolved from `format!("{:?}:")` string allocation to `std::mem::discriminant` hashing — zero allocation on cache lookup
- Pipeline cache `PipelineKey` evolved from `String` entry point to `u64` hash — eliminates per-lookup allocation
- Legacy discovery filename evolved from hardcoded `coralreef-core.json` to agnostic `shader-compiler.json`
- `DeviceInfo::name` evolved from `String` to `Arc<str>` — zero-alloc clone on every device lease
- `RingBufferConfig::label` evolved from `String` to `Option<Arc<str>>` — zero-alloc clone on buffer creation
- `CoralCompiler::state` evolved from `Mutex` to `RwLock` with `Arc<str>` addresses — concurrent reads
- Ring buffer `write()` evolved from million-iteration `spin_loop()` to staged back-off (256 spins → 4096 yields)
- 10 f64 ops (`weighted_dot`, `digamma`, `bessel_k0`, `bessel_j0`, `prod_reduce`, `norm_reduce`, `variance_reduce`, `sum_reduce`, `max_abs_diff` ×2) evolved from hardcoded `256` to `WORKGROUP_SIZE_1D` constant
- `max_allocation_size()` evolved from float round-trip to integer arithmetic (`max_buffer_size / 4 * 3`)
- `sanitize_max_buffer_size` VRAM caps extracted to named constants (`VRAM_CAP_PROFESSIONAL`, `VRAM_CAP_CONSUMER_HIGH`, `VRAM_CAP_CONSERVATIVE`)
- `gpu_dispatch_threshold` magic numbers extracted to named constants (`DISCRETE_THRESHOLD`, `INTEGRATED_THRESHOLD`, `OTHER_THRESHOLD`)
- `DeviceRequirements::score()` magic numbers extracted to named constants (`PREFERRED_VENDOR_BONUS`, `DISCRETE_BONUS`, `IDLE_BONUS`)
- `AttentionDims` config struct — replaces 4-arg attention/head_split/head_concat with typed struct
- `parse_shape()` helper in IPC methods — `usize::try_from` instead of `as usize` casts
- `eprintln!` → `tracing::warn!` in hardware verification tests

## What's Not Working Yet

- P1: DF64 end-to-end NVK hardware verification (Yukawa shaders)
- P2: Test coverage ~75% on llvmpipe (target: 90%, requires real GPU hardware for GPU-path coverage)
- P2: Kokkos validation baseline documentation
- P2: Kokkos GPU parity benchmarks

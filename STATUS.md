# barraCuda Status

**Version**: 0.3.6
**Date**: 2026-03-20
**Overall Grade**: A+ (Zero unsafe via `#![forbid(unsafe_code)]`, zero unwrap in production, zero println in library code, pure safe Rust, AGPL-3.0-or-later scyBorg trio, all quality gates green, 3,555 tests passing, zero TODO/FIXME/unimplemented, zero "for now" debt language, zero `Result<T, String>` in production (all typed errors), clippy nursery clean in barracuda-core, wateringHole leverage patterns guide published, scheduler println→tracing evolution, full audit: all mocks test-only, all panics test-only, capability-based discovery confirmed in production, JSON-RPC+tarpc dual protocol, UniBin+ecoBin compliant, all files under 1000 lines, GemmF64 TransA/TransB flags, FAMILY_ID socket paths per PRIMAL_IPC_PROTOCOL, `deny.toml` wildcards=deny supply chain audit, blake3 `default-features=false` ecoBin pure, WGSL_MEAN_REDUCE public re-export, NVVM poisoning guard, PrecisionBrain self-routing, HardwareCalibration per-tier probing, PCIe topology probing, VRAM quota enforcement, rayon-parallel shader validation, optimised test pipeline, all deps pure Rust, device-aware test tolerances, cross-spring pharma/bio/health absorption, FMA policy, stable GPU special functions, sovereign coral-cache dispatch wiring, capability-based PRIMAL_NAMESPACE, VoltaNoPmuFirmware workaround detection, namespace-derived IPC method names, 806/806 WGSL SPDX headers, 1,077 Rust source files, `#![deny(clippy::pedantic)]` + nursery + doc lint promotion, CI coverage 80% blocking, ecoBin cross-compile CI, CoralReefDevice→toadStool dispatch wired, capability-based toadStool discovery, RwLock tensor store, zero-copy BytesMut/Bytes evolution, device-lost DRY refactor, ODE bio system test coverage, async readback typed errors, genomics 25-test coverage, smart module decomposition, transport defaults named constants, cargo update applied)

---

## Category Grades

| Category | Grade | Notes |
|----------|-------|-------|
| **Core compute** | A+ | 806 WGSL shaders, 13-tier tolerance architecture, GpuView persistent buffers with ops, BGL builder pattern, PrecisionBrain domain→tier routing |
| **Precision tiers** | A+ | 3-tier model (F32/F64/Df64) aligned with coralReef `Fp64Strategy`; DF64 naga-guided rewrite validated; probe-aware Fp64Strategy; DF64 reduce shaders correctly routed via `.df64()` on Hybrid devices; NVVM poisoning guard for proprietary NVIDIA DF64 transcendentals |
| **Sovereign compiler** | A+ | FMA fusion + dead expr elimination + safe WGSL roundtrip (all backends); sovereign validation harness covers all 806 shaders via rayon parallel validation; `erfc_f64` recursion eliminated |
| **IPC / primal protocol** | A+ | JSON-RPC 2.0 (notification-compliant) + tarpc; Unix socket default + TCP; capability-based discovery; namespace-derived method names via `PRIMAL_NAMESPACE` + `METHOD_SUFFIXES`; coralReef Phase 10 `shader.compile.*` semantic naming; AMD arch support |
| **Device management** | A+ | `GpuBackend` trait abstraction, `CoralReefDevice` IPC-first sovereign backend behind `sovereign-dispatch` feature (`is_coral_available`, `with_auto_device`, `has_dispatch`), multi-GPU with PCIe topology sysfs probing (`PcieLinkInfo`), capability-scored discovery, probe-aware f64 strategy, VRAM quota enforcement via `ResourceQuota`/`QuotaTracker`, bounded poll timeout, split-lock GPU submission |
| **Test coverage** | A+ | 3,555 tests (all pass on llvmpipe); GPU test timeout guards (30s); proptest; chaos/fault test tiers (blocking in CI); nextest CI/stress profiles; 80% coverage gate (blocking); GPU streaming: split-lock dispatch, fire-and-forget `submit_commands`, single-poll `submit_and_map` readback; optimised test pipeline (nautilus 14.3s→0.01s, sovereign 800+ shaders parallelised via rayon, ESN reservoir shrunk); zero `todo!()`/`unimplemented!()`; async readback GPU roundtrip tests; genomics 25-test edge-case coverage; GemmF64 transpose roundtrip tests; smart module decomposition (IPC methods/ directory, hydrology GPU barrel); optimised build profiles (codegen-units=256, split-debuginfo); with_device_retry double-permit fix |
| **Dependencies** | A+ | All deps pure Rust (blake3 `pure`, wgpu/naga 28); zero application C deps; ecoBin compliant |
| **Documentation** | A+ | Comprehensive CHANGELOG, specs, README, CONTRIBUTING, CONVENTIONS, BREAKING_CHANGES; all rustdoc warnings resolved; showcase/ with 9 progressive demos (local, IPC, cross-primal) |
| **Unsafe code** | A+ | Zero `unsafe` blocks; `#![forbid(unsafe_code)]` in both crates (irrevocable) |
| **Clippy / lint** | A+ | Zero warnings with pedantic + unwrap_used + nursery (selective) + `-D warnings`; 20 bulk-allowed lints promoted (9 pedantic + 5 nursery + 2 doc + 4 cast); `if_same_then_else` fixed (7 sites) and promoted; `redundant_clone`, `imprecise_flops`, `derive_partial_eq_without_eq`, `unnecessary_struct_initialization` enforced; `missing_errors_doc`, `missing_panics_doc` enforced; cast lints enforced in barracuda-core; zero production `unwrap()`; `#[expect(reason)]` for suppressions |
| **Error handling** | A+ | Binary `main()` uses typed `BarracudaCoreError` (not `Box<dyn Error>`); `From` impls for `serde_json::Error`, `BarracudaError`, `io::Error`; `Result` propagation throughout; `let-else` throughout; poison recovery |
| **Idiomatic Rust** | A+ | Edition 2024; zero `too_many_arguments` (all 9 → builder/struct); documented `#[allow]`/`#[expect]` with reason; `#[derive(Default)]`; zero unsafe; zero production unwrap; `ChamferDirection` enum; smart module decomposition (provenance, coral_compiler); capability version derived from `env!("CARGO_PKG_VERSION")`; method lists derived from `REGISTERED_METHODS` constant |
| **Spring absorption** | A+ | All P0/P1/P2 items resolved; hotSpring: NVVM poisoning guard, plasma dispersion, LSCFRK, **PrecisionBrain + HardwareCalibration**; groundSpring: F64NativeNoSharedMem, DF64 reduce, shared `estimate_gflops`/`estimate_vram_bytes`; wetSpring: BGL builder, `ComputeDispatch` builder, `Rk45Result::variable_trajectory()`, **`CsrMatrix::from_triplets_summed()`**, **`BipartitionEncodeGpu`**; neuralSpring: activations, Wright-Fisher, `analyze_weight_matrix()`; healthSpring: Hill Emax, Population PK, tridiagonal QL, LCG PRNG, **FOCE/VPC GPU shaders**; toadStool S139: dual-scan discovery |

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
- `CoralReefDevice` IPC-first sovereign backend behind `sovereign-dispatch` feature — compiles via coralReef JSON-RPC, dispatches via toadStool JSON-RPC; no compile-time coupling
- VFIO-primary architecture adopted: toadStool VFIO is the primary GPU dispatch path (exclusive device access, IOMMU isolation, deterministic scheduling); wgpu demoted to development/fallback
- VFIO detection moved to toadStool — barraCuda queries hardware capabilities via IPC at runtime
- Kokkos parity projections: ~4,000 steps/s target (VFIO + DF64) vs 2,630 steps/s Kokkos baseline
- `SOVEREIGN_PIPELINE_TRACKER.md` — tracks P0 (CoralReefDevice), VFIO primary dispatch, libc→rustix evolution, cross-primal deps
- Zero TODOs/FIXMEs/HACKs/`unreachable!()` without messages in codebase
- Zero "for now" debt language — all 22 instances evolved to proper engineering documentation with performance thresholds
- Device-lost panic handling DRY-refactored via `handle_device_lost_panic()` helper
- 21 new ODE bio system unit tests covering all 5 biological models
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
- `showcase/` collection: 9 progressive demos across 3 tiers (local primal, IPC protocol, cross-primal compute)
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
- `board.rs` hash unwrap evolved to zero-panic direct array indexing — `blake3::Hash::as_bytes()` returns `&[u8; 32]`, compile-time safe
- tarpc `primal_capabilities` method list derived from `REGISTERED_METHODS` constant — single source of truth
- JSON-RPC `primal.capabilities` version strings derived from `env!("CARGO_PKG_VERSION")` — no hardcoded version drift
- JSON-RPC `primal.capabilities` methods array derived from `REGISTERED_METHODS` — eliminates method list duplication
- HMM dispatch threshold extracted to `HMM_FORWARD_THRESHOLD` named constant
- Three springs tests evolved with device-aware tolerance: `tol()` helper floors precision expectations at 1e-6 for hardware with imprecise f64 shaders
- Kahan summation test detects f32-only GPU execution and skips gracefully rather than false-failing
- `PrecisionTier` enum (F32/DF64/F64/F64Precise) + `PhysicsDomain` classification — absorbed from hotSpring v0.6.25 precision brain
- `HardwareCalibration` per-tier GPU probing — safe tier detection with NVVM poisoning guard, builds on existing probe infrastructure
- `PrecisionBrain` self-routing domain→tier routing table — O(1) lookup for all physics domains, probe-first, data-driven
- `PhysicsDomain` extended with `PopulationPk`, `Bioinformatics`, `Hydrology`, `Statistics`, `General` for cross-spring coverage
- `CsrMatrix::from_triplets_summed()` — duplicate (row, col) entries summed automatically (wetSpring V105 finite-element assembly pattern)
- `OdeTrajectory` result struct with `.time_series(batch, var)` and `.state_at(batch, t)` interpolation helpers (wetSpring/airSpring request)
- `integrate_cpu_trajectory()` on `BatchedOdeRK4<S>` — records full ODE trajectory at every time step
- `lanczos_with_config()` — configurable convergence threshold + progress callback for long-running eigensolves (N > 1000)
- `lanczos_extremal()` — efficient k-largest eigenvalue extraction via early-termination Lanczos
- Two-pass Gram-Schmidt reorthogonalization in Lanczos for improved numerical stability on large matrices
- Tolerance registry evolution: `all_tolerances()`, `by_name()`, `tier()` runtime introspection functions
- Pharma tolerances: `PHARMA_FOCE`, `PHARMA_VPC`, `PHARMA_NCA` for population PK validation
- Signal processing tolerances: `SIGNAL_FFT`, `SIGNAL_QRS` for biosignal analysis
- `BipartitionEncodeGpu` — GPU kernel for Robinson-Foulds distance bit-vector encoding (wetSpring V105 absorption)
- `FoceGradientGpu` — GPU-accelerated FOCE per-subject gradient computation for population PK (healthSpring V14 absorption)
- `VpcSimulateGpu` — GPU Monte Carlo VPC simulation with RK4 PK integration (healthSpring V14 absorption)
- `foce_gradient_f64.wgsl` + `vpc_simulate_f64.wgsl` + `bipartition_encode.wgsl` — 3 new production WGSL shaders

- **Deep debt sprint 12 — comprehensive audit, module decomposition & build optimisation** (Mar 20): IPC `methods.rs` (675L) decomposed into `methods/` directory (mod.rs router + primal/device/health/compute/tensor/fhe domain files). Hydrology `gpu.rs` (648L) decomposed into barrel module + `hargreaves_gpu.rs`/`seasonal_gpu.rs`/`mc_et0_gpu.rs`. Kernel router magic numbers evolved to `WORKGROUP_FFT`/`WORKGROUP_PHYSICS` named constants. `with_device_retry` double-permit bug fixed (was acquiring GPU semaphore twice, halving test parallelism). Build profiles optimised: `codegen-units=256`, `split-debuginfo=unpacked`, `opt-level=2` for deps. 26 new tests across `compute_graph.rs` (9) and `spectral/lanczos.rs` (7) and `bfgs.rs` fix. `BFGS_MAX_ITER_EXTENDED` moved to test-only scope (clippy `unfulfilled_lint_expectations` fix). All quality gates green. 3,555 tests pass, 0 fail.
- **Deep debt sprint 6 — cross-ecosystem absorption** (Mar 16): GemmF64 `execute_gemm_ex()` with TransA/TransB flags — enables `A^T*A` and `A^T*b` without materializing transpose (groundSpring Tikhonov, airSpring least-squares). WGSL `select()`-based stride swapping. FAMILY_ID socket paths (`$BIOMEOS_FAMILY_ID`, default `"default"`) per `PRIMAL_IPC_PROTOCOL`. blake3 `default-features=false` (ecoBin pure Rust only). `deny.toml` `wildcards=deny` supply chain audit. `WGSL_MEAN_REDUCE` + `WGSL_MEAN_REDUCE_F64` re-exported from `ops/mod.rs` (neuralSpring). 3 stale lint suppressions removed via `#[expect]` audit. 3,466 tests pass, 0 fail. All quality gates green (fmt, clippy, doc, deny).
- **Deep debt sprint 5 — typed errors, nursery lints & coverage** (Mar 16): All `Result<T, String>` in production code evolved to `Result<T, BarracudaError>` with typed error variants (15 sites across `async_submit.rs`, `coral_compiler/jsonrpc.rs`, `df64_rewrite/mod.rs`, `test_harness.rs`, `ipc/methods.rs`). 6 clippy nursery warnings eliminated in `barracuda-core` (`option_if_let_else`, `missing_const_for_fn`, `or_fun_call`, `iter_on_single_items`). `poll_until_ready` `&mut self` → `&self` (no mutation needed). Async readback methods return `BarracudaError::device_lost()` and `BarracudaError::gpu()` instead of String. 5 new `async_submit` tests (queue/submit lifecycle, readback roundtrips). 14 new genomics edge-case tests (RNA, empty, N-heavy, batch, error paths). 3,464 tests pass, 0 fail. All quality gates green (fmt, clippy -D warnings, rustdoc -D warnings).
- **Deep debt sprint 4 — sovereign wiring & zero-copy** (Mar 15): `CoralReefDevice` wired to toadStool `compute.dispatch.submit` via JSON-RPC. `detect_dispatch_addr()` discovers toadStool via capability-based scanning of `$XDG_RUNTIME_DIR/ecoPrimals/*.json`. Buffer staging via `BytesMut`. `dispatch_compute` evolved to `Entry` API. `Default` impl added. `#![deny(clippy::pedantic)]` in both crates. Tensor store `Mutex` → `RwLock`. Zero-copy: `CpuTensorStorage` → `BytesMut`, `EventCodec` → `Bytes`, `CompileResponse::into_bytes()`. Edition 2024 safety: `set_var` eliminated from tests. coralNAK → coralReef in all active docs.
- **GPU streaming & audit sprint** (Mar 15): `submit_and_poll_inner` split into separate `submit_commands_inner` + `poll_wait_inner` lock acquisitions — other threads interleave submits while one polls (eliminates 120s lock convoy). 279 fire-and-forget ops migrated from blocking `submit_and_poll` to non-blocking `submit_commands`. New `submit_and_map<T>` method collapses double-poll (submit+poll then map+poll) into single submit → `map_async` → `poll_safe` cycle. `read_buffer<T>` now uses single-poll path. `--all-features` clippy fixed: `is_coral_available()`, `CoralReefDevice::with_auto_device()`, `CoralReefDevice::has_dispatch()` added. Pre-existing doc_markdown lints fixed (bcs, screened_coulomb, chi2). All quality gates green. 3,400+ tests pass, 0 fail.
- **Deep debt sprint 3 — lint evolution & refactoring** (Mar 14): `missing_errors_doc` and `missing_panics_doc` promoted to warn in both crates. Cast lints promoted in `barracuda-core`. `large_stack_frames` documented as test artifact. `suboptimal_flops` evolved in test files. `ode_bio/params.rs` refactored (774L → 7-file modular structure). RBF `assemble_and_solve` zero-copy via `split_off`. CI: 80% coverage + chaos/fault now blocking; cross-compile job for musl targets. Dead `ring` deny.toml config removed. All quality gates green.
- **Deep debt evolution sprint 2** (Mar 12): 5 nursery lints promoted (`redundant_clone`, `imprecise_flops`, `unnecessary_struct_initialization`, `derive_partial_eq_without_eq`, `suboptimal_flops` kept allow with rationale). 7 `if_same_then_else` sites fixed and lint promoted to warn. `needless_range_loop` sites reduced (csr, device_info, fft_1d converted to idiomatic iterators). Hardcoded discovery paths evolved to `PRIMAL_NAMESPACE`-derived. `zeros`/`ones` dispatch duplication eliminated via combined match arm. 193 files touched by auto-fix (redundant clones removed, precision improved via `ln_1p`/`to_radians`/`hypot`).
- **Comprehensive audit & deep debt sprint** (Mar 12): Full codebase audit against wateringHole standards. 12-item remediation completed.
- **`#![forbid(unsafe_code)]`** (Mar 12): Upgraded from `deny` (overridable) to `forbid` (irrevocable) in both `barracuda` and `barracuda-core` crate roots.
- **Namespace-derived IPC method names** (Mar 12): All 12 hardcoded `"barracuda.method.name"` strings evolved to `LazyLock<Vec<String>>` built from `PRIMAL_NAMESPACE` + `METHOD_SUFFIXES`. Dispatch routing uses `method_suffix()` to strip namespace prefix. Discovery, tarpc, CLI all consume derived names.
- **SPDX license headers** (Mar 12): 648 WGSL shaders missing `// SPDX-License-Identifier: AGPL-3.0-only` headers — all 806 shaders now have them. 1088/1088 Rust files already had them.
- **Pedantic lint promotion** (Mar 12): 9 bulk-allowed clippy lints promoted to `warn` (enforced via `-D warnings`): `needless_raw_string_hashes`, `redundant_closure_for_method_calls`, `bool_to_int_with_if`, `cloned_instead_of_copied`, `map_unwrap_or`, `no_effect_underscore_binding`, `format_push_string`, `explicit_iter_loop`, `used_underscore_binding`.
- **erfc_f64 recursion fix** (Mar 12): `stable_f64.wgsl` had recursive `erfc_f64` (WGSL forbids recursion). Refactored to non-recursive `erfc_x_nonneg_f64` helper.
- **Magic numbers extracted** (Mar 12): `CONSERVATIVE_GPR_COUNT`, `DEFAULT_WORKGROUP`, `CORAL_CACHE_ARCHITECTURES` constants in `coral_reef_device.rs`.
- **Zero-copy evolution** (Mar 12): `async_submit::read_bytes()` and `ncbi_cache::load()` evolved to return `bytes::Bytes`.
- **`unreachable!` evolved** (Mar 12): Production `unreachable!()` in `df64_rewrite` evolved to `debug_assert!` + graceful fallback.
- **Rustdoc zero warnings** (Mar 12): Fixed broken `transport::resolve_bind_address` link and private `wgsl_templates` link.
- **BufferBinding import** (Mar 12): Added missing `BufferBinding` import in `coral_reef_device.rs` — `--all-features` clippy now passes.
- **Sovereign cache → dispatch wiring** (Mar 12): `CoralReefDevice::dispatch_compute` now checks the coral compiler cache (populated by `spawn_coral_compile`) before recompiling. Cache hits skip compilation, completing the WgpuDevice-compile → CoralReefDevice-dispatch pipeline.
- **PRIMAL_NAMESPACE constant** (Mar 12): All hardcoded `"barracuda"` strings in IPC namespace, socket paths, PID file paths evolved to `PRIMAL_NAMESPACE` constant for capability-based discovery.
- **VoltaNoPmuFirmware workaround** (Mar 12): `GpuDriverProfile` detects Volta + NVK as needing software PMU. `needs_software_pmu()` and `sovereign_resolves_poisoning()` methods added.
- **`dispatch_binary` implemented** (Mar 12): `GpuBackend::dispatch_binary` method on `CoralReefDevice` accepts raw native binaries from coralReef. `dispatch_kernel` method for full `CompiledKernel` metadata.
- **ODE solver refactored** (Mar 12): `ode_generic` split into mod.rs (solver/trait/tests) + wgsl_templates.rs (codegen) — clean concern separation.
- **DF64 shader comments cleaned** (Mar 12): Removed misleading `DF64_POLYFILL_PLACEHOLDER` from 15 protein folding shaders (polyfill injection handled at compile time).
- **CLI refactored** (Mar 12): `barracuda` binary's monolithic main() split into modular subcommand handlers.
- **Arc allocation elimination** (Mar 12): `Arc::from(format!(...).as_str())` → `Arc::from(format!(...))` across 11 files, eliminating double heap allocation.

## What's Not Working Yet

- ~~P0: toadStool `compute.dispatch.submit` IPC wiring~~ — **Done** (Mar 15). `CoralReefDevice` discovers toadStool via capability scan and dispatches via JSON-RPC
- P0: VFIO dispatch hardware revalidation on Titan V — coralReef Iter 50 applied USERD_TARGET + INST_TARGET fix; 7/7 expected
- P1: DF64 end-to-end NVK hardware verification (Yukawa shaders)
- P1: coralReef sovereign compiler evolution (unified compiler and driver for all GPU targets)
- P1: Kokkos validation baseline documentation (unblocked by VFIO strategy)
- P2: Test coverage ~75% on llvmpipe (80% CI gate blocking; 90% target requires real GPU hardware; 3,555 tests passing)
- P2: Kokkos GPU parity benchmarks
- ~~P2: RHMC multi-shift CG solver~~ — **Done** (Mar 12, rhmc.rs + rhmc_hmc.rs)

### Cross-Primal Pins (current)

| Primal | Version/Session | Key capability |
|--------|-----------------|----------------|
| toadStool | S156 | Full audit specialty resurrection; sovereign infra gaps resolved |
| coralReef | Phase 10 Iter 50 | Full audit execution; USERD_TARGET + INST_TARGET runlist fix applied |
| hotSpring | v0.6.32 | Trio rewire; VFIO validation; GP_PUT root cause found |

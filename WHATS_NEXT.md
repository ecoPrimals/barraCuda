# barraCuda — What's Next

Prioritized work items, ordered by impact. Updated 2026-03-12.

---

## Recently Completed

- **Comprehensive audit & deep debt sprint (Mar 12)**: Full codebase audit against
  wateringHole standards (uniBin, ecoBin, semantic naming, sovereignty, zero-copy,
  license compliance, code quality). 12-item remediation: `#![forbid(unsafe_code)]`
  in both crates; namespace-derived IPC method names via `PRIMAL_NAMESPACE` +
  `METHOD_SUFFIXES` (LazyLock); 648 WGSL SPDX headers added (805/805 complete);
  9 bulk-allowed pedantic lints promoted to warn (enforced); erfc_f64 recursion
  fix in stable_f64.wgsl; magic numbers extracted (CONSERVATIVE_GPR_COUNT,
  DEFAULT_WORKGROUP, CORAL_CACHE_ARCHITECTURES); zero-copy evolution
  (async_submit::read_bytes, ncbi_cache::load -> bytes::Bytes); unreachable! ->
  debug_assert! + graceful fallback; rustdoc zero warnings; BufferBinding import
  for --all-features clippy. 3,688 tests pass, 0 fail, 15 skip.
- **Sovereign dispatch wiring & deep debt evolution (Mar 11-12)**: Wired coral
  compiler cache → `CoralReefDevice::dispatch_compute` (sovereign cache hits
  skip recompilation). Implemented `dispatch_binary` and `dispatch_kernel` on
  `CoralReefDevice`. Added `PRIMAL_NAMESPACE` constant, replacing all hardcoded
  `"barracuda"` strings in IPC/socket/PID paths. Refactored `ode_generic` (890L →
  613L + 290L WGSL codegen). Cleaned 15 DF64 shader placeholder comments.
  Refactored CLI into modular subcommand handlers. Added `VoltaNoPmuFirmware`
  workaround detection. Eliminated double heap allocation in `Arc::from` across
  11 files. All clippy pedantic clean. External deps (pollster, futures, half)
  audited and justified. Zero production unwrap/expect confirmed.
- **Cross-spring absorption & deep evolution (Mar 10)**: PrecisionTier/PhysicsDomain
  enums for domain-aware precision selection. HardwareCalibration safe per-tier GPU
  probing with NVVM poisoning guard (synthesizes from cached driver profile — no device
  poisoning risk). PrecisionBrain self-routing domain→tier O(1) table with `route()`,
  `route_advice()`, `compile()`. Lanczos extended with `lanczos_with_config()`,
  two-pass Gram-Schmidt reorth, `lanczos_extremal()`. CsrMatrix::from_triplets_summed
  (wetSpring V105). OdeTrajectory with `time_series()`, `state_at()`, `final_state()`.
  BipartitionEncodeGpu for Robinson-Foulds bit-vector encoding (wetSpring). FOCE
  gradient GPU shader for population PK (healthSpring V14). VPC Monte Carlo GPU
  simulation with RK4 + Box-Muller (healthSpring V14). Tolerance registry evolved
  with `all_tolerances()`, `by_name()`, `tier()` runtime introspection — 36 registered
  tolerances. All 6 springs absorbed. 3,348 tests pass, 0 fail, 13 ignored.
- **Deep debt & test pipeline (Mar 10)**: Unified GFLOPS/VRAM estimation across
  GpuPool and MultiDevicePool. Fixed Fp64Strategy routing in 4 reduce ops (DF64
  shaders now correctly compiled via `.df64()` on Hybrid devices). PCIe topology
  via Linux sysfs probing (`PcieLinkInfo`). VRAM quota enforcement wired into all
  buffer allocation paths. BGL builder for declarative bind-group layout. Sovereign
  shader validation parallelised via rayon. Nautilus test pipeline 1430× faster
  (14.3s→0.01s) — tests now validate dispatch mechanics, not full computation.
  Board hash evolved from Debug formatting to zero-alloc blake3 incremental hashing.
  ESN test shrunk from 200→16 reservoir. Full suite: 3,348 pass, 0 fail, 21.5s.
- **Deep cleanup sprint**: Removed 4 orphaned test directories (`tests/chaos/`, `tests/fault/`,
  `tests/e2e/`, `tests/precision/`) — ~4,000 lines of dead code that drifted to 84-125 compilation
  errors each. Wired in `three_springs/` (was compiling but never linked). Cleaned stale
  informal TODO comments from `ops/mod.rs`. Corrected doc counts to match actual codebase
  (3,348 lib tests, 28 integration suites, 1,060+ .rs files, 9 showcase demos).
- **`Rk45Result::variable_trajectory()`**: Convenience method extracting single-variable
  trajectory across all ODE time steps. Eliminates `y_history[step][var_idx]` boilerplate
  used by 5+ ODE scenario builders in wetSpring. Also added `n_vars()`. 2 tests.
- **`spectral::analyze_weight_matrix()`**: Composite primitive combining `eigh_f64` eigensolve
  with bandwidth, condition number, phase classification, mean IPR, level spacing ratio,
  and spectral entropy — single call for neural network weight diagnostics. 4 tests.
- **`histogram_u32_to_f64()`**: Conversion utility for GPU k-mer histogram readback.
  Spectrum/visualization channels require `Vec<f64>`; GPU histograms produce `Vec<u32>`.
  Eliminates repeated manual casting. 2 tests.
- **toadStool S139 discovery alignment**: `discover_from_file()` now scans both
  `$XDG_RUNTIME_DIR/ecoPrimals/` (flat) and `$XDG_RUNTIME_DIR/ecoPrimals/discovery/`
  (canonical) for primal manifests. Aligns with toadStool S139 dual-write discovery.
- **Tridiagonal QL eigensolver**: `special::tridiagonal_ql` — symmetric tridiagonal
  eigenvalue/eigenvector solver with `anderson_diagonalize()`. healthSpring absorption.
  Fixed EISPACK sub-diagonal convention bug. 6 tests.
- **LCG PRNG module**: `rng` — centralized Knuth LCG (`lcg_step`, `state_to_f64`,
  `uniform_f64_sequence`). Replaces constant duplication across 4+ springs. 6 tests.
- **Public activations API**: `activations` — canonical CPU f64 `sigmoid`, `relu`, `gelu`,
  `swish`, `mish`, `softplus`, `leaky_relu` + batch variants. Consolidates 7 duplicates. 8 tests.
- **Wright-Fisher population genetics**: `ops::wright_fisher_f32` — GPU-vectorized allele
  frequency evolution with selection + drift + xoshiro128** PRNG. neuralSpring absorption.
  3 GPU tests (neutral drift, strong selection, fixation). New WGSL shader.
- **healthSpring Hill dose-response absorption**: `HillFunctionF64` evolved to support
  Emax parameter (`dose_response()` constructor). Output now `[0, Emax]` instead of
  fixed `[0, 1]`. Backward compatible — `new()` defaults to `emax = 1.0`. 3 new tests.
- **healthSpring Population PK Monte Carlo**: New `PopulationPkF64` op with
  `population_pk_f64.wgsl` shader. GPU-vectorized virtual patient simulation with
  Wang hash + xorshift32 PRNG. Fully parameterized (no hardcoded CL ranges). 6 tests.
- **hotSpring plasma dispersion W(z)/Z(z)**: `special::plasma_dispersion` module with
  CPU-side numerically stable implementations. Direct asymptotic expansion for |z| ≥ 4
  avoids catastrophic cancellation (ISSUE-006). 8 tests. `Complex64` promoted to runtime.
- **neuralSpring head_split/head_concat alignment**: Confirmed equivalent index math —
  both compute `[B,S,D] ↔ [B,H,S,D/H]` with identical memory layout. No changes needed.
- **P0: `SumReduceF64`/`VarianceReduceF64` `Fp64Strategy` fix**: Created DF64 shader
  variants (`sum_reduce_df64.wgsl`, `variance_reduce_df64.wgsl`) that use `vec2<f32>`
  workgroup memory instead of native f64. Added `Fp64Strategy` routing so Hybrid
  devices auto-select the DF64 path. Fixes zeros-from-reductions bug reported by
  groundSpring V96 and neuralSpring S131.
- **Re-export builder types**: `HmmForwardArgs`, `Dada2DispatchArgs`, `Dada2Buffers`,
  `Dada2Dimensions`, `GillespieModel`, `PrecisionRoutingAdvice`, `Rk45DispatchArgs`
  now accessible at `barracuda::` level (wetSpring P1 request).
- **`barracuda::math::{dot, l2_norm}`**: Re-exported from `stats::metrics` —
  15+ wetSpring binaries can drop local 5-line implementations.
- **`fused_ops_healthy()` canary**: `device::test_harness::fused_ops_healthy(&device)`
  runs minimal variance probe, returns `false` if shared-memory reductions fail.
  neuralSpring canary pattern absorbed.
- **NVK zero-output detection**: `GpuDriverProfile::f64_zeros_risk()` flags
  NVK + Full/Throttled FP64 as shared-memory-unreliable (airSpring V071 request).
- **`GpuViewF64` ops**: `mean_variance(ddof)`, `sum()`, `correlation(a, b)` —
  stepping-stone API for zero-readback chains (groundSpring P2 request).
- **Test utilities absorbed**: `is_software_adapter(&device)`, `baseline_path(relative)`
  in `test_harness`, re-exported in `test_prelude`.
- **DF64 Hybrid fallback bug**: 10 ops (covariance, weighted_dot, hermite, digamma,
  cosine_similarity, beta, bessel_i0/j0/j1/k0) now return `ShaderCompilation` error
  instead of silently producing zeros on Hybrid devices.
- **Systematic f64 pipeline fix**: 14 ops evolved from `compile_shader()`/`GLOBAL_CACHE`
  (which silently downcast f64→f32) to f64-native compilation paths —
  `compile_shader_f64()` for direct callers, `create_f64_data_pipeline()` for
  GLOBAL_CACHE users. Pipeline cache evolved with f64-native path (`shaders_f64`,
  `pipelines_f64` maps).
- **Zero-copy `CpuTensorStorageSimple`**: `Vec<u8>` → `Bytes` — `read_to_cpu()` is
  now a cheap ref-count bump.
- **Pipeline cache hot-path**: `DeviceFingerprint` discriminant hashing (no `format!`),
  `PipelineKey` hash (no `String`).
- **Legacy discovery filename**: `coralreef-core.json` → `shader-compiler.json`.
- **GPU f64 computational accuracy probe**: `get_test_device_if_f64_gpu_available()`
  now runs a runtime probe (`3*2+1=7`) to verify real f64 execution, gating 58 tests
  that were failing on software rasterizers.
- **Zero-copy upload evolution**: ~50 GPU dispatch paths evolved from
  `to_le_bytes().collect::<Vec<u8>>()` to `bytemuck::cast_slice()` — eliminates
  per-dispatch allocation across pipeline, MD, linalg, reduce, optimize, PDE, grid ops.
- **`GpuBackend::download()` → `Bytes`**: Trait return type `Vec<u8>` → `bytes::Bytes`.
- **`NpuTensorStorage` → `BytesMut`**: `Vec<u8>` → `bytes::BytesMut` with `freeze()`.
- **`ShaderCompilation(Arc<str>)`**: Error variant `String` → `Arc<str>` across 10 ops.
- **GPU fallback estimates**: 13 hardcoded constants refactored to `fallback_estimates::{gflops, vram_bytes}` pattern-matched by vendor/device-type.
- **Coverage expansion**: batch_ipr, histogram, staging (ring_buffer, unidirectional, stateful), precision/cpu, surrogate/adaptive — targeting 0% and <30% coverage modules.
- **GPU-heavy test timeouts**: Extended slow-timeout overrides for edge_conv, fft, conv2d, flash_attention; fixed edge_conv 60s timeout failure.
- **CI 90% coverage**: Dual target — 80% baseline (llvmpipe), 90% stretch (GPU hardware, continue-on-error).
- **Showcase collection**: 9 progressive demos across 3 tiers (local primal, IPC
  protocol, cross-primal compute). Demonstrates device discovery, precision tiers
  (F32/F64/DF64), fused GPU ops (Welford, correlation, GpuView), science shaders
  (Hill kinetics, tolerance architecture), JSON-RPC server, doctor/validate,
  coralReef shader compilation, toadStool hardware discovery, sovereign pipeline
  capstone. All Cargo crates compile zero warnings. Each cross-primal demo
  degrades gracefully when toadStool/coralReef are absent.
- **Bounded GPU poll timeout**: `BARRACUDA_POLL_TIMEOUT_SECS` (default 120s) prevents
  indefinite hangs under llvm-cov instrumentation.
- **LSCFRK gradient flow integrators**: Absorbed from hotSpring — `derive_lscfrk3`
  const fn, W6/W7/CK45 coefficient sets, `find_t0`/`find_w0`/`compute_w_function`.
- **NautilusBrain force anomaly**: 10σ energy anomaly detection with rolling delta-H
  window, 4th training head.
- **GPU-resident reduction**: `encode_reduce_to_buffer` + `readback_scalar` for
  zero-CPU-roundtrip multi-kernel pipelines.
- **Idiomatic Rust evolution**: `#[expect]` over `#[allow]`, `#[derive(Default)]`,
  `is_none_or`, iterator `collect()`, Option combinators.
- **airSpring 6 ops**: Confirmed all 6 (MakkinkEt0, TurcEt0, HamonEt0, ScsCnRunoff,
  StewartYieldWater, BlaneyCriddleEt0) already absorbed.
- **Sovereign validation harness**: Pure-Rust shader pipeline coverage without GPU.
- **Tokio runtime graceful detection**: `coral_compiler` module no longer panics without runtime.
- **Zero `too_many_arguments`**: All 9 instances evolved to builder/struct patterns (CG solver,
  Gillespie, seasonal params, HMM, RK45, DADA2, spin-orbit, leapfrog, RBF).
- **Deprecated PPPM constructors removed**: `new()` / `new_with_driver()` had zero callers.
- **Akida SDK paths → capability constant**: `AKIDA_SDK_SYSTEM_DIRS` shared between device modules.
- **Cross-spring provenance registry**: `shaders::provenance` (types/registry/report modules)
  tracks Write → Absorb → Lean evolution with `ShaderRecord` and `SpringDomain` taxonomy.
- **coralReef Phase 10 rewire**: `shader.compile.*` semantic IPC, `capabilities()`, AMD RDNA2+,
  backward-compat fallback; `coral_compiler` decomposed into types/discovery/cache/jsonrpc/client.
- **`PrecisionRoutingAdvice`** from toadStool S128: `F64Native`, `F64NativeNoSharedMem`, `Df64Only`,
  `F32Only` routing in `GpuDriverProfile::precision_routing()`.
- **3-tier precision lean-out**: Removed `Precision::F16` (aspirational, zero production callers),
  `templates.rs` (411-line `{{SCALAR}}` system, zero production callers), `compile_shader_universal`,
  `compile_op_shader`, `compile_template` (all zero callers). Net -798 lines. Precision model now
  explicitly 3-tier (F32/F64/Df64), directly aligned with coralReef's `Fp64Strategy`. IPC
  `CompileWgslRequest` now sends `fp64_strategy` hint alongside legacy `fp64_software` flag.
  `precision_to_coral_strategy()` maps barraCuda's `Precision` → coralReef's strategy string.
- **`hill_activation` / `hill_repression`**: Absorbed from neuralSpring `primitives.rs`.
  Amplitude-scaled Hill functions for gene regulatory networks. 9 unit tests.
  `barracuda::stats::{hill_activation, hill_repression}`.
- **Ada Lovelace `F64NativeNoSharedMem` reclassification**: RTX 4000-series + proprietary
  driver now routes to `F64NativeNoSharedMem` instead of `Df64Only`. `f64_zeros_risk()` extended
  to cover Ada + proprietary. groundSpring P0 request resolved.
- **`shared_mem_f64` runtime probe**: New empirical probe in `device::probe` verifies
  `var<workgroup> array<f64, 4>` reductions produce correct results. `precision_routing()`
  and heuristic seed now use probe result. `needs_shared_mem_f64_workaround()` added to
  `F64BuiltinCapabilities`. Native count: 9 → 10. groundSpring P1 request resolved.
- **`BatchedOdeRK45F64`**: Full-trajectory adaptive Dormand-Prince integrator on GPU with
  host-side step-size control (wetSpring V95, 18.5× fewer steps than RK4).
- **`mean_variance_to_buffer()`**: GPU-resident fused Welford — output stays as buffer for
  chained multi-kernel pipelines (zero CPU readback).
- **Evolution timeline**: 10 chronological events + dependency matrix in provenance registry,
  27 shaders with created/absorbed dates.
- **`service` subcommand**: genomeBin compliance — systemd/init integration with PID file,
  READY=1 notification, Unix socket transport, graceful shutdown.
- **Dynamic discovery**: capabilities, provides, and methods arrays derived from
  `REGISTERED_METHODS` source of truth instead of hardcoded values.
- **Thread-local GPU throttling**: `OwnedSemaphorePermit` in test pool transparently limits
  concurrent GPU access during `cargo test` — reduced intermittent failures from ~103 to 2.
- **`bytes::Bytes` zero-copy**: `TensorStorage::read_to_cpu()`, `WorkUnit.data`,
  `CompletedWork.data` now return `Bytes` instead of `Vec<u8>`.
- **Deep audit**: lint migration (#[allow] → #[expect] where valid), eprintln → tracing,
  RPC String parameter documentation, CI coverage with --ignore-run-fail.
- **Hot-path clone elimination**: `DeviceInfo::name` (`String` → `Arc<str>`),
  `RingBufferConfig::label` (`String` → `Option<Arc<str>>`), `CoralCompiler::state`
  (`Mutex` → `RwLock` with `Arc<str>` addresses).
- **Ring buffer back-off**: `write()` spin-wait evolved from million-iteration `spin_loop()`
  to staged back-off (256 spins → 4096 `yield_now()` calls, ~100ms budget).
- **Workgroup size consolidation**: 10 f64 ops (`weighted_dot`, `digamma`, `bessel_k0`,
  `bessel_j0`, `prod_reduce`, `norm_reduce`, `variance_reduce`, `sum_reduce`,
  `max_abs_diff` ×2) evolved from hardcoded `256` to `WORKGROUP_SIZE_1D` constant.
- **Magic number extraction**: `max_allocation_size()` float round-trip → integer arithmetic;
  `sanitize_max_buffer_size` VRAM caps, `gpu_dispatch_threshold` levels, and
  `DeviceRequirements::score()` weights all extracted to named constants.
- **Test fragility resolved**: GPU tests (erf, erfc, expand, determinant) evolved from
  `catch_unwind` to `with_device_retry` — production recovery pattern.
- **Streaming pipeline completion**: `GpuRingBuffer::read()`, `advance_write()`, and
  `UnidirectionalPipeline::poll_results()` implemented for GPU→CPU data flow.
- **`AttentionDims` config struct**: Replaces 4-argument attention/head_split/head_concat
  with typed struct (builder pattern).
- **IPC `as` casts → `try_from`**: `parse_shape()` helper with safe `usize::try_from`.
- **External dependency audit**: All deps confirmed pure Rust — fully ecoBin compliant.

---

## Immediate (P1)

- **DF64 NVK end-to-end verification**: Run df64 compilation on Yukawa force kernels through
  NVK/NAK on hardware. Validate sovereign compiler's safe WGSL roundtrip produces correct
  numerical results across all backends. Probe-aware `fp64_strategy()` is now in place to
  auto-fallback if native f64 fails.
- **coralNAK extraction**: When org repo fork lands, create the sovereign NVIDIA shader
  compiler primal.
- **Dedicated DF64 shaders for covariance + weighted_dot**: The auto-rewrite works and
  the native f64 path is now fixed via `create_f64_data_pipeline()`. Hand-written DF64
  shaders (like variance/correlation already have) would be more robust on Hybrid devices.
- **`BatchedTridiagEigh` GPU op**: groundSpring local QL implicit eigensolver is a candidate
  for absorption as a batched GPU tridiagonal eigenvector solver.
- **Multi-GPU OOM recovery**: `QuotaTracker` is wired into buffer allocation; next step
  is automatic workload migration when a device hits VRAM quota.

## Near-term (P2)

- **Test coverage to 90%**: Evolve CI `--fail-under` from 80 to 90. Add GPU-conditional
  tests for new ops (SCS-CN, Stewart, Blaney-Criddle, autocorrelation).
- **Kokkos validation baseline**: Document `sarkas_gpu` validation results, extract PPPM
  shader performance numbers for apples-to-apples comparison.
- **Kokkos GPU parity benchmarks**: Run barraCuda GPU benchmarks on matching hardware,
  publish comparison data.
- **WGSL optimizer annotation coverage**: Expand `@ilp_region` / `@unroll_hint` annotations
  across science shaders for architecture-specific ILP optimization.
- **RHMC multi-shift CG absorb**: hotSpring has RHMC with Hasenbusch preconditioning;
  barraCuda already has the CG solver but multi-shift variant is pending.

## Medium-term (P3)

- **Multi-GPU dispatch**: Evolve GpuView to span multiple devices with automatic work
  distribution across primary/secondary adapters.
- **Pipeline cache re-enable**: When wgpu provides a safe `create_pipeline_cache` API
  (or safe wrapper for `data: None`), re-enable in-memory pipeline caching. The field +
  accessor are preserved, `make_pipeline_cache` returns `None` until then.
- **Shader hot-reload**: File watcher for `.wgsl` files during development, automatic
  recompilation through sovereign pipeline.
- **Zero-copy evolution**: `bytes::Bytes` on I/O boundaries + `CpuTensorStorageSimple` +
  `CosineSimilarityF64` done; remaining: pre-allocated buffers for `domain_ops.rs` CPU
  fallback clones, LSTM hidden state clones, RBF assembly allocations.

## Long-term (P4)

See `SOVEREIGN_PIPELINE_TRACKER.md` for the full sovereign pipeline tracker
including cross-primal dependencies, libc/musl → rustix evolution, and
cross-compilation target matrix.

- **Sovereign Compute Evolution**: Replace entire non-Rust GPU stack with coral-prefixed
  pure Rust components (coralNak, coralDriver, coralMem, coralQueue, coralGpu).
- **WebGPU browser target**: Compile barraCuda shaders for browser execution via wasm-pack
  and wgpu's WebGPU backend.
- **Distributed compute**: Cross-node GPU dispatch via primal-to-primal IPC for HPC clusters.

---

## C Dependency Chain — Evolution Map

**barraCuda has zero unsafe code and zero application-level C dependencies.**

The remaining C boundary is the OS/driver interface via transitive dependencies of
`wgpu` and `tokio`. These are system-level and do not constitute application C deps.

### barraCuda dependency chain (what touches C)

| Dependency | What it does | C boundary | Who evolves it |
|------------|-------------|------------|----------------|
| `wgpu` → `wgpu-hal` → `ash` → `libloading` | Vulkan FFI: dynamically loads `libvulkan.so` and calls the Vulkan C API | Vulkan driver (OS/GPU vendor) | **coralReef** (sovereign driver replaces Vulkan path) |
| `wgpu` → `wgpu-hal` → `renderdoc-sys` | RenderDoc debug capture FFI | Debug-only, never hits production | Can be feature-gated out of wgpu |
| `wgpu` → `wgpu-core` → `parking_lot_core` → `libc` | Futex/condvar syscalls for GPU synchronization | Kernel ABI, not a C library | Rust std evolves (already uses libc internally) |
| `tokio` → `mio` → `libc` | epoll/kqueue/io_uring syscalls | Kernel ABI | Rust std evolves |
| `tokio` → `signal-hook-registry` → `libc` | Signal handler registration | Kernel ABI | Rust std evolves |
| `getrandom` → `libc` | `/dev/urandom` or `getrandom(2)` syscall | Kernel ABI | Rust std evolves |
| `blake3` | Hashing (with `pure` feature) | **None** — `pure` flag = no C SIMD asm | Already pure Rust |

### coralReef dependency chain (what touches C)

| Dependency | What it does | C boundary | Who evolves it |
|------------|-------------|------------|----------------|
| `jsonrpsee` → `hyper` → `tokio` → `libc` | HTTP/WS transport + async runtime | Kernel ABI | Rust std evolves |
| `nak-ir-proc` (2 unsafe blocks) | `from_raw_parts` on `#[repr(C)]` struct fields with compile-time contiguity proofs | **None** — pure Rust, unsafe for performance | **coralReef** evolves: array-field pattern or `bytemuck` cast |

### The path to pure Rust end-to-end

Math is universal. A shader is just math. The execution substrate (GPU, CPU, NPU, Android
ARM core) is a hardware implementation detail — not a difference in universal math.

**Layer 1 — barraCuda (DONE)**: Zero unsafe, zero application C deps. WGSL shaders
express the math. The sovereign compiler optimises at the naga IR level in pure Rust.
Compilation flows through safe `create_shader_module`. The math layer is pure Rust today.

**Layer 2 — coralReef (2 unsafe blocks remain)**: The `nak-ir-proc` proc macro uses
`slice::from_raw_parts` on `#[repr(C)]` structs with compile-time contiguity proofs.
Evolution path: store matched fields as `[T; N]` arrays with named accessors, or use
`bytemuck::cast_ref`/`cast_mut` on Pod types. This is an internal coralReef evolution —
the IPC interface is unaffected.

**Layer 3 — GPU drivers (external, OS-level)**: `wgpu → ash → libvulkan.so` is the
system driver boundary. This is where the sovereign compute evolution eliminates the
last C dependency: coralReef's pure-Rust NVIDIA codegen (coralNak) replaces NAK, then
coralDriver replaces the Vulkan loader. The math never changes — only the substrate.

**Layer 4 — Kernel ABI (`libc`)**: Every Rust program on Linux calls the kernel through
`libc` (syscalls for memory, I/O, signals). This evolves via `rustix` (pure Rust syscalls
using `linux-raw-sys`) — see `SOVEREIGN_PIPELINE_TRACKER.md` for the phased evolution
from libc/musl to zero-package cross-compilation.

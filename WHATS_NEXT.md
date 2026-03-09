# barraCuda — What's Next

Prioritized work items, ordered by impact. Updated 2026-03-09.

---

## Recently Completed

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
- **Showcase collection**: 10 progressive demos across 3 tiers (local primal, IPC
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

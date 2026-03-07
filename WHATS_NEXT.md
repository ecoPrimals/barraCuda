# barraCuda — What's Next

Prioritized work items, ordered by impact. Updated 2026-03-07.

---

## Recently Completed

- **DF64 Hybrid fallback bug**: 10 ops (covariance, weighted_dot, hermite, digamma,
  cosine_similarity, beta, bessel_i0/j0/j1/k0) now return `ShaderCompilation` error
  instead of silently producing zeros on Hybrid devices.
- **GPU f64 computational accuracy probe**: `get_test_device_if_f64_gpu_available()`
  now runs a runtime probe (`3*2+1=7`) to verify real f64 execution, gating 58 tests
  that were failing on software rasterizers.
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
- **`BatchedOdeRK45F64`**: Full-trajectory adaptive Dormand-Prince integrator on GPU with
  host-side step-size control (wetSpring V95, 18.5× fewer steps than RK4).
- **`mean_variance_to_buffer()`**: GPU-resident fused Welford — output stays as buffer for
  chained multi-kernel pipelines (zero CPU readback).
- **Evolution timeline**: 10 chronological events + dependency matrix in provenance registry,
  27 shaders with created/absorbed dates.

---

## Immediate (P1)

- **DF64 NVK end-to-end verification**: Run `compile_shader_universal(Precision::Df64)` on
  Yukawa force kernels through NVK/NAK on hardware. Validate sovereign compiler's safe WGSL
  roundtrip produces correct numerical results across all backends. Probe-aware `fp64_strategy()`
  is now in place to auto-fallback if native f64 fails.
- **coralNAK extraction**: When org repo fork lands, create the sovereign NVIDIA shader
  compiler primal.
- **Dedicated DF64 shaders for covariance + weighted_dot**: The auto-rewrite works but
  hand-written DF64 shaders (like variance/correlation already have) would be more robust.

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
- **Zero-copy evolution**: Pre-allocated buffers for `domain_ops.rs` CPU fallback clones,
  LSTM hidden state clones, RBF assembly allocations (see zero-copy audit).

## Long-term (P4)

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
`libc` (syscalls for memory, I/O, signals). This is the OS boundary, not a C dependency.
Rust's `std` uses it internally. It's unavoidable and irrelevant to sovereignty — the
kernel is a platform, not a dependency.

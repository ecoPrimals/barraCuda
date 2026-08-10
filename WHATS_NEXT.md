# barraCuda — What's Next

Prioritized work items, ordered by impact. Updated 2026-08-10.

---

## Recently Completed

### Wave 157g — Gossip Client + G72 Dep Audit (Aug 10, 2026)
Fire-and-forget `gossip.inject` client wired in `barracuda-core/src/ipc/gossip.rs`.
Injects `compute.device.created`, `tower.endpoint.alive`, and
`tower.health.readiness_changed` at primal startup. 5 public injection helpers
cover device lifecycle, health, and capacity events. Socket discovery:
`SWARMVINE_SOCKET` env → `$XDG_RUNTIME_DIR/biomeos/swarmvine.sock` → silent no-op.
7 new tests (5,038 total). G72 Dependency Pandemic Tier 1 audit confirms barraCuda
is already clean: zero `pollster`, tokio features already trimmed (not `["full"]`),
all 19 direct deps actively used, wgpu 28 canonical, `deny.toml` bans C crypto.
231 transitive deps — duplicate chains (tarpc→rand 0.8, wgpu→hashbrown) are
transitive and uncontrollable.

### Wave 157e — Sovereign GEMM Executor + Gossip Injection Points (Aug 10, 2026)
Sovereign executor bridge wired: `SovereignDevice::compile_gemm(m, n, k, precision)` →
`GLOBAL_CORAL.compile_gemm()` IPC with binary caching → `dispatch_gemm()` combines
compilation + `submit_dispatch()` with `HardwareHint::TensorCore`. Completes the
`KernelRouter::Sovereign` → `shader.compile.gemm` → dispatch pipeline that was P2 from
Wave 157d. 20 gossip injection points documented in `capability_registry.toml` across 6
categories (device lifecycle, shader compilation, health state, capacity/load, precision
routing, systemic errors) — pending swarmVine UDS wiring for actual injection hooks.
Deep debt scan confirmed clean: 0 production `unwrap()`/`expect()`, 0 files >800L, 0 stale
TODOs, 5,031 tests pass, zero clippy warnings.

### Wave 157d — Deep Debt Evolution: Zero-Panic + Decomposition (Aug 9, 2026)
17 GPU dispatch `.expect()` sites evolved to `?` / `Result` propagation —
restores zero-panic production guarantee across bio ops (14), MD observables (2),
and RK45 adaptive (1). `method_descriptor()` 512-line match decomposed into
10 per-namespace helpers. 3 `#[allow(dead_code)]` → `#[expect]`. 5 AKIDA env
vars centralized in `env_keys.rs`. 5,031 tests pass. Zero clippy warnings.

### Wave 157d — Node Atomic Trio Wiring (Aug 9, 2026)
Tier 2 items: PTXAS/NAK routing + GEMM IPC client.

- **Compiler-aware coralReef routing**: `DeviceCapabilities::compiler_prefers_coral()`
  detects NAK/PTXAS/RADV codegen defects (probe-first when available, vendor
  heuristic fallback). `PrecisionBrain::should_bypass_local_compiler()` returns
  true when local compiler has known bugs AND coralReef is available — superset
  of `needs_sovereign_compile()`.
- **`shader.compile.gemm` IPC client**: `CoralCompiler::compile_gemm(m, n, k,
  precision, arch)` calls coralReef's tiled MMA kernel generator. Wire type
  `GemmCompileRequest` + `GemmCompileResponse`. Graceful `None` fallback.
- **Registry cleanup**: `capability_registry.toml` updated — added
  `shader.compile.gemm` + `shader.compile.spirv` (was used but undeclared),
  removed dead `shader.compile.module`.
- **5,031 tests pass.** Zero clippy warnings. Zero failures.

### Wave 157d — Pattern Abstraction + Deep Debt Evolution (Aug 9, 2026)
Three "by design" patterns evolved into proper abstractions:

- **DF64 source centralization**: `DF64_CORE` and `DF64_TRANSCENDENTALS` centralized
  in `shaders/mod.rs` as public constants. 21 `include_str!` duplicates removed from
  production code. `df64_source()` and `df64_f64_source()` helpers replace 5 identical
  `format!` patterns. `LazyLock<String>` caching retained (irreducible — caches
  runtime concat).
- **Deprecated batch functions evolved**: 10 `_batch()` functions (`relu_batch`,
  `sigmoid_batch`, `gelu_batch`, `swish_batch`, `erf_batch`, `erfc_batch`,
  `bessel_j0_batch`, `bessel_j1_batch`, `bessel_i0_batch`, `bessel_k0_batch`)
  had zero production callers — removed from public API, moved to `#[cfg(test)]`.
  Re-exports removed from `lib.rs`.
- **BEARDOG env aliases evolved**: `env_with_deprecated_fallback()` helper in
  `env_keys.rs` replaces 3 repeated fallback-chain patterns across `btsp.rs`,
  `btsp_client.rs`, `btsp_discovery.rs`. 3 `#[deprecated]` constants retained
  for test backward compat only.
- **CachedPipeline abstraction**: New `CachedPipeline` + `BindingKind` in
  `compute_pipeline.rs` — build once, dispatch many. 4 bio genomics ops migrated
  from ad-hoc `snp.rs` helpers. 6 `pub(super)` helpers deleted from `snp.rs`.
  `create_bind_group_layout` now zero in ops/ (was 3).
- **coralReef IPC wire verified complete**: Full pipeline already wired —
  `CoralCompiler` → `shader.compile.wgsl` via `compile_wgsl_direct()` +
  `compile_wgsl_with_advice()`. `SovereignDevice::live_compile()` uses
  precision-routed path. No stubs remaining.
- **5,025 tests pass.** Zero clippy warnings. Zero failures.

### Wave 157d — Fmt Cleanup (Aug 9, 2026)
- 142-file `cargo fmt` correction (−177 net lines).
- `sourdough validate ecobin` PASS.

### Wave 157d — Silicon Fold Absorption + Binary Size Fix (Aug 9, 2026)
- **Buffer limit negotiation** (P1): Static `SCIENCE_MAX_BUFFER_SIZE` evolved to
  `negotiate_buffer_limits(adapter)` — queries hardware, takes `min(hw, desired)`.
  New `NegotiatedLimits` struct + `science_limits_from_adapter()` /
  `high_capacity_limits_from_adapter()`. Existing const functions retained for
  test-without-adapter backward compat.
- **SiliconRouter trait**: Free `route_workload()` evolved to trait with
  `WorkloadRequirements`-based routing. Cache-aware (IC vs L2), F16 vendor-aware
  (AMD 1.32x native, NVIDIA 0.99x skip), atomic/precision routing. Implemented
  for `SiliconProfile`.
- **TileDecomposer**: N-dimensional cache-aligned domain decomposition. Sizes
  tiles to fit GPU cache (128 MB AMD IC → 16^4, 6 MB NVIDIA L2 → 6^4). Halo
  width for boundary exchange.
- **RiverScheduler**: PCIe/VRAM bandwidth as schedulable resource. Double-buffer
  staging model. Transfer planning with estimated utilization (current: 1.7%,
  target: 50%+).
- **VideoCodec trait + IPC**: `VideoCodec` trait (encode/decode), `NullCodec`
  fallback, `detect_codecs()` ffmpeg probe (NVENC/VAAPI/SW detection).
  `device.video_codecs` IPC method registered (101 methods total).
- **Release profile**: Added `lto = true`, `codegen-units = 1`, `strip = true`
  to address P3 binary size bloat (4.4x on Windows).
- **5,025 tests pass.** Zero clippy warnings. Zero failures.

### Wave 157a — Vertebrate Self-Audit (Aug 9, 2026)
- **RPC surface audit**: 3-way cross-reference — `REGISTERED_METHODS` (code),
  `capability_registry.toml` (biomeOS), and dispatch match arms.
- **2 gaps found and fixed**: `method.describe` and `stats.eigh` alias missing
  from capability_registry.toml. Now documented with `[domains.method]` and
  `aliases` entry.
- **Zero phantom APIs**: every registered method dispatches to a real handler.
  Unknown methods return `METHOD_NOT_FOUND` (not health stubs).
- **2 new tests**: `every_registered_method_dispatches` (runtime proof no
  phantom APIs) and `registry_toml_covers_registered_methods` (compile-time
  proof TOML matches code). 4,996 tests pass.
- **12-axis deep debt re-scan**: all green. Zero files >800L, zero unsafe,
  zero `TODO/FIXME`, zero production `unwrap()`, zero `Result<T, String>` in
  production, zero hardcoded primal names.

### Wave 157a — G68 Platform Substrate Compliance (Aug 7, 2026)
- New `platform_substrate` module in barracuda-core following sourDough
  reference pattern. `platform_link()` (symlink/Unix, hard-link/Windows)
  and `is_symlink()` helper.
- `server.rs::create_legacy_symlink()` evolved from raw
  `std::os::unix::fs::symlink` to `platform_link()`.
- 4 new tests. 4,996 tests pass (incl. 2 vertebrate self-audit tests). Windows cross-compile clean.
- **0 L2 violations remaining.** barraCuda is G68 compliant.

### Wave 157a — ComputeDispatch P1 Migration (Aug 7, 2026)
**Commit 4** — ComputeDispatch P1 Migration (225 non-WGSL ops, −26,373 LOC):
- **All 225 remaining non-WGSL ops** migrated to `ComputeDispatch` builder.
  Categories: simple binary (4), simple unary (4), unary_with_params (55),
  binary_with_params (49), multi_buffer (48), complex (65).
- 107 false `.f64()` additions corrected (f32 ops incorrectly marked as f64).
- Dead BGL helpers, shader constants, and unused imports cleaned.
- Only 3 files retain manual BGL (intentional cached pipeline patterns:
  `snp.rs`, `fd_common.rs`, `gemm_f64.rs`).
- Combined with P0: **317 ops migrated, −37,144 LOC total**.
- 4,990 tests pass, 0 failures. Zero clippy warnings.
- Zero files over 800 lines (largest: 783L test file).

**Commit 3** — ComputeDispatch P0 Migration (92 files, −10,771 LOC):
- **All 92 `*_wgsl.rs` ops** migrated from manual BGL→pipeline→encoder→submit
  boilerplate to `ComputeDispatch::new().shader().storage_read().storage_rw()
  .uniform().dispatch_1d().submit()` builder pattern. Includes:
  - 12 simple unary math ops (sin, cos, tan, exp, sqrt, log, abs, sign,
    trunc, floor, ceil, round)
  - 34 script-migrated unary ops (trig inverses, bessels, activations, etc.)
  - 16 parameterized ops (pow, clamp, pool1d, cumsum, norm, etc.)
  - 16 multi-constructor ops (pad variants, interpolate, PRNG, polynomials)
  - 14 complex multi-binding ops (gather, sparse_matvec, rnn_cell, trace, etc.)
- **`create_uniform_buffer<T: Pod>()` helper** already existed — all ops now
  use it for shader parameter structs (correct UNIFORM usage instead of
  STORAGE usage from `create_buffer_f32_init`).
- 4,990 tests pass, 0 failures. Zero clippy warnings.

### Wave 157a — Deep Debt Sweep Phase 4 (Aug 7, 2026)
**Commit 1** — Error Idiom Evolution Phase 3 + Magic Number Centralization:
- Verbose error constructor cleanup across 20 files. `device_ctx()` helper.
  40 `Gpu(format!)` → `gpu_ctx()` in domain_ops. Magic numbers centralized
  (SOFTPLUS_UPPER_THRESHOLD, DEFAULT_FITTS_HICK_B). Protocol negotiation
  zero-alloc. G68 Platform Substrate audit: CLEAN.

**Commit 2** — Deprecated Batch Unwiring + Pool2D Dedup + Env Centralization:
- **IPC `activation.gelu` evolved** — Removed `#[expect(deprecated)]` suppression.
  Handler now uses scalar `gelu()` directly instead of deprecated `gelu_batch()`.
- **Pool2D dispatch dedup** — Extracted `pool2d_dispatch()` helper consolidating
  MaxPool2D and AvgPool2D match arms (~90 LOC → single function). Both pool
  variants share routing logic, GPU/CPU fallback, and NCHW reshape.
- **Env var centralization Phase 2** — Added `TRANSPORT_ENDPOINT` and
  `BARRACUDA_TARPC_SOCKET` to `env_keys.rs`. Updated `transport_endpoint.rs`,
  `transport_config.rs`, `main.rs`, `bench_wgsize_nvk.rs`, `bench_f64_builtins.rs`.
- **12-axis deep debt scan**: all axes clean. Zero hardcoded env vars remaining
  (AKIDA vendor-specific keys intentionally inline). 5 `LazyLock<String>` (Pattern C).
- 5,011 tests pass, 0 failures. Zero clippy warnings.

### Wave 156l — Fmt Drift + G65 Readiness Review (Aug 6, 2026)
- **182-file `cargo fmt` correction** — systematic line-length and indentation drift
  accumulated across the barracuda crate. All formatting now clean.
- **G65 readiness assessment** — server accept loop architecture confirmed ready for
  Phase 3 protocol negotiation. No structural changes needed.
- **Full quality gate verification**:
  - Zero clippy warnings (including `suboptimal_flops`, `inefficient_to_string`,
    `needless_pass_by_value`, `cloned_instead_of_copied`, `implicit_clone`,
    `match_same_arms`, `unused_self`, `redundant_else`, `map_unwrap_or`)
  - Zero `#[allow]`, zero TODO/FIXME/HACK, zero `#[ignore]` on unit tests
  - All `unsafe` confined to test code (env var mutation, serialized by ENV_MUTEX)
  - `#![forbid(unsafe_code)]` on all 4 crates
  - All production files under 800 LOC
  - Cross-architecture: Windows `x86_64-pc-windows-gnu` compiles clean
- 4,984 tests pass, 0 failed.

### Wave 156k — GPU Buffer Alignment Fix (Aug 6, 2026)

### Wave 155u — Deep Idiom Evolution (Aug 4, 2026)
- **LazyLock\<String\> → const &str Phase 2** — Converted remaining 323 shader
  statics across 310 files. Total migration: 374 statics across 328 files
  (51 in Wave 155p + 323 in 155u). Only 5 `LazyLock<String>` remain — all
  genuine DF64 `format!` concatenation (Pattern C) that requires runtime
  string assembly.
- **Error constructor helpers** — Added `invalid_input()`, `numerical()`,
  `execution()`, `internal()`, `not_implemented()` constructor methods to
  `BarracudaError`. Migrated 518 call sites from verbose struct literal
  syntax (`BarracudaError::InvalidInput { message: "...".to_string() }`)
  to concise helper calls (`BarracudaError::invalid_input("...")`).
  Eliminates `.to_string()` / `.into()` noise at call sites while keeping
  the typed error enum unchanged.
- **Environment variable centralization** — Added 9 new constants to
  `env_keys.rs`: `BARRACUDA_GPU_ADAPTER`, `BARRACUDA_CONCURRENCY_BUDGET`,
  `BARRACUDA_MATMUL_SMALL_THRESHOLD`, `BARRACUDA_MATMUL_GPU_THRESHOLD`,
  `BARRACUDA_SHADER_COMPILER_ADDR`, `BARRACUDA_SHADER_COMPILER_PORT`,
  `XDG_CACHE_HOME`, `XDG_DATA_HOME`, `HOME`. Updated all scattered inline
  string literals in `creation.rs`, `dispatch.rs`, `session/types.rs`,
  `coral_compiler/discovery.rs`, `autotune.rs`, `ncbi_cache.rs`, `akida.rs`
  to reference centralized constants.
- **12-axis deep debt scan** confirmed clean:
  - Files >800L: **0** (max 783L, test file)
  - Production unsafe: **1** (barracuda-spirv passthrough, feature-gated)
  - Production unwrap: **0** in src/
  - todo/unimplemented: **0**
  - Bare `#[allow(`: **0**
  - Hardcoded primal names: **0**
  - Production mocks: **0**
  - Cross-primal deps: **0**
  - `Result<T, String>`: **0**
  - println in lib: **0**
  - Hardcoded env var literals: **0** (all centralized in `env_keys.rs`)
  - Remaining `LazyLock<String>`: **5** (all Pattern C — format! concatenation)
- Net -1,504 LOC total (-478 shader statics + -1,026 error constructors + env_keys).
  Zero clippy warnings. 4,984 tests pass. All quality gates green.

### Wave 155p — PRNG Validation + Shader Static Evolution + Magic Numbers (Aug 3, 2026)
- **LazyLock\<String\> → const &str Phase 1** — Converted 51 shader statics across 18
  files from heap-allocating `LazyLock<String>` to zero-cost `const &str`.
  Eliminates ~51 one-time heap allocations and `LazyLock` synchronization
  overhead on the shader compilation hot path. `format!` concatenation
  instances (Pattern C) intentionally preserved.
- **Protocol version unified** — `"jsonrpc-2.0"` vs `"json-rpc-2.0"` inconsistency
  in `primal.info` vs `primal.capabilities` resolved. Single `PROTOCOL_ID` constant.
- **Magic numbers extracted** — `BTSP_WIRE_VERSION` (was inline `1` in 3 locations),
  `IPC_PROBE_TIMEOUT` (was `Duration::from_secs(5)` inline).
- **Dependency analysis** — confirmed 100% RustCrypto, no openssl/ring, blake3 `pure`
  feature enabled, no build.rs in workspace, tarpc optional (default-on for binary).
- **Production stubs audited** — 3 stubs all correctly feature-gated with documentation.
  SPIRV passthrough exists behind feature. BTSP relay is Unix-only by design. Discovery
  socket returns sentinel on non-Unix.
- **12-axis deep debt scan** confirmed clean (0 critical findings).
- Net -182 LOC. Zero clippy warnings. 4,984 tests pass.

#### PRNG Fixes (same wave)
- **CPU PRNG half-range bug FIXED** — `state_to_f64()` extracted 31 bits (>> 33)
  but divided by `u32::MAX` (2^32-1), producing values in [0, 0.5) instead of
  [0, 1). Fixed to 53-bit extraction matching `LcgRng::uniform()` and lattice
  `lcg_uniform_f64()`. The `rng.uniform` IPC method was affected — now correctly
  covers full [min, max) range.
- **GPU xoshiro PRNG half-range bug FIXED** — `prng_xoshiro_f64.wgsl` combined
  26+26=52 bits but divided by 2^53, producing [0, 0.5). Fixed to 27+26=53 bits
  for full [0, 1) coverage.
- **Statistical PRNG validation harness added** — 11 new tests covering:
  - Uniform mean/variance for f64, f32, and LcgRng (expected 0.5 / 1/12)
  - Chi-squared goodness-of-fit (10-bin, p<0.001)
  - CPU Box-Muller Gaussian moments (mean, variance, skewness, kurtosis)
  - Gaussian chi-squared (20-bin vs N(0,1) CDF)
  - GPU xoshiro statistical validation (mean, variance, chi-squared)
  - GPU seed independence
  - Multi-seed independence
  - Lattice LCG uniform mean/variance and Gaussian moments
- **PRNG YELLOW → GREEN**: Statistical validation harness is now in-repo.
  CPU generators (LCG, Box-Muller) and GPU generators (xoshiro128**) validated
  against expected distributions. Lattice PCG uniform confirmed correct (uses
  `(v + 0.5) / 2^32` — proper [0, 1) range).
- **12-axis deep debt scan** confirmed clean (0 critical findings).
- Zero clippy warnings. 4,984 tests pass. All quality gates green.

### Waves 155f–155n (Jul 28 – Aug 3, 2026)
SIGSEGV fix (GPU_TEST_GUARD), ESN BindGroupLayout fix, BTSP env races fixed,
BatchError→thiserror, wgpu backend target-gating, RTX 3090 profiling (103.97
TFLOPS FP64), MultiDevicePool wired, silicon utilization AAR, PRNG half-range
fixes (YELLOW→GREEN), RK4 zero-alloc, LazyLock→const Phase 1 (51 statics),
error helpers (518 sites), env centralization. See CHANGELOG.md for details.

### Waves 107–129 (Jun 10 – Jun 28, 2026)
12-axis deep debt audits, GNU depot validation, RTX 5070 E2E pipeline,
LSTM zero-copy, OOM auto-migration, spring absorption (100 IPC methods),
bincode→postcard, 238 mul_add evolutions, transport module decomposition.
See `CHANGELOG.md` for full details.


Older completions (Waves 44–128, Mar–Jun 2026) documented in `CHANGELOG.md`.

---

## Immediate (P1)

- **PrecisionBrain → coralReef → SovereignDevice CI integration**: Mock trio E2E validated
  (Sprint 57). Next: CI with live coralReef instance for full pipeline validation.
- **DF64 hardware verification**: GPU-dispatched DF64 E2E tests validated on RTX 3090
  (Wave 155i). DF64 at 91.89 TFLOPS on strandGate. Remaining: Yukawa force kernels
  through coralReef-compiled ISA on physical hardware.
- **Tensor core GEMM codegen**: `kernel_router` routes F16/BF16/TF32 `DenseMatmul` to
  `KernelTarget::Sovereign` with `HardwareHint::TensorCore` (Sprint 64). `dispatch_gemm()`
  bridge wired (Wave 157e): `compile_gemm()` → `GLOBAL_CORAL.compile_gemm()` IPC → binary
  cache → `submit_dispatch()`. Next: coralReef HMMA/WGMMA emission.
- **Kokkos parity validation**: SciPy cdist 65x faster on RTX 3090. Remaining: document
  `sarkas_gpu` PPPM shader comparison numbers.

## Near-term (P2)

- **Gossip injection expansion**: 5/20 events wired at startup (device.created,
  endpoint.alive, readiness_changed, device.lost, capacity). Remaining 15 need
  runtime injection at compilation, OOM, precision routing, and error sites.
- **Test coverage to 90%**: Currently 80.54% line on llvmpipe. CI 80% gate blocking.
  Evolve to 90 with real GPU hardware. Remaining gaps are GPU-dependent code paths.
- **Kokkos GPU parity benchmarks**: Publish comparison data on matching hardware.
- **Optional tensor encryption via `tensor` purpose key**: Per
  `NUCLEUS_TWO_TIER_CRYPTO_MODEL.md` — encrypt sensitive tensor data in transit.
  Opt-in for medical/financial workloads.

## Medium-term (P3)

- **Multi-GPU dispatch**: Evolve GpuView to span multiple devices with automatic work
  distribution across primary/secondary adapters.
- **Pipeline cache re-enable**: When wgpu provides a safe `create_pipeline_cache` API
  (or safe wrapper for `data: None`), re-enable in-memory pipeline caching. The field +
  accessor are preserved, `make_pipeline_cache` returns `None` until then.
- **Shader hot-reload**: File watcher for `.wgsl` files during development, automatic
  recompilation through sovereign pipeline.
- **Zero-copy evolution**: `bytes::Bytes` on I/O boundaries + `CpuTensorStorageSimple` +
  `CosineSimilarityF64` + RBF `assemble_and_solve` + `CpuTensorStorage` → `BytesMut` +
  `EventCodec` → `Bytes` + `CompileResponse::into_bytes()` done. **LSTM zero-copy
  shipped (Wave 120)**: `forward_into` + `GateBuffers` pre-allocated scratch — eliminates
  per-timestep `Vec<f64>` clone and 4×`Vec<f64>` gate allocations per layer per step.
  `domain_ops.rs` CPU fallback confirmed clone-free; GPU f64→f32 conversion is inherent.

## Long-term (P4)

See `SOVEREIGN_PIPELINE_TRACKER.md` for the full sovereign pipeline tracker
including cross-primal dependencies, libc/musl → rustix evolution, and
cross-compilation target matrix.

- **Sovereign Compute Evolution**: Replace entire non-Rust GPU stack with coralReef
  as the unified compiler and driver for all GPU targets (eventually also the Rust
  compiler) via VFIO primary dispatch path (toadStool VFIO GPU backend + IOMMU isolation).
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
last C dependency: coralReef's pure-Rust NVIDIA codegen replaces NAK, then
coralReef's driver layer replaces the Vulkan loader. The math never changes — only the substrate.

**Layer 4 — Kernel ABI (`libc`)**: Every Rust program on Linux calls the kernel through
`libc` (syscalls for memory, I/O, signals). This evolves via `rustix` (pure Rust syscalls
using `linux-raw-sys`) — see `SOVEREIGN_PIPELINE_TRACKER.md` for the phased evolution
from libc/musl to zero-package cross-compilation.

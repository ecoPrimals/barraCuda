# barraCuda — What's Next

Prioritized work items, ordered by impact. Updated 2026-03-15.

---

## Recently Completed

- **Deep debt sprint 4 — sovereign wiring & zero-copy (Mar 15)**: CoralReefDevice
  wired to toadStool `compute.dispatch.submit` via capability-based discovery
  (`detect_dispatch_addr` scans `$XDG_RUNTIME_DIR/ecoPrimals/*.json` for
  `compute.dispatch`). Buffer staging via `BytesMut`. `dispatch_compute` evolved
  to `Entry` API. `Default` impl added. `# Errors` doc sections added. Pedantic
  lint promoted to `deny` in both crates. Tensor store `Mutex` → `RwLock`.
  Zero-copy: `CpuTensorStorage` → `BytesMut`, `EventCodec` → `Bytes`,
  `CompileResponse::into_bytes()`. Edition 2024 safety: `set_var` eliminated
  from tests. coralNAK → coralReef in all active docs. All quality gates green.
- **GPU streaming & comprehensive audit sprint (Mar 15)**: Full GPU submission
  pipeline refactored to eliminate blocking. `submit_and_poll_inner` split into
  separate `submit_commands_inner` + `poll_wait_inner` lock acquisitions —
  other threads interleave submits while one thread polls (eliminates 120s
  lock convoy on 16-thread nextest). 279 fire-and-forget GPU ops migrated from
  blocking `submit_and_poll` to non-blocking `submit_commands` (zero CPU wait
  for GPU-resident results). New `submit_and_map<T>` single-poll readback method
  collapses old double-poll into one `poll_safe` cycle. `read_buffer<T>` optimized
  internally. `--all-features` clippy compilation fixed (`is_coral_available`,
  `CoralReefDevice::with_auto_device`, `has_dispatch`). Pre-existing lints fixed
  (doc_markdown in bcs/screened_coulomb/chi2, approx_constant in kinetics test,
  double_must_use in critical_screening/chi_squared). Full codebase audit: zero
  archive code, zero dead scripts, zero TODO/FIXME in production, zero files over
  1000 lines, zero .bak/.tmp debris. All quality gates green. 3,407 tests, 0 failures.
- **Deep debt sprint 3 — lint evolution & refactoring (Mar 14)**:
  `missing_errors_doc` and `missing_panics_doc` promoted to warn in both crates
  (zero violations). Cast lints (`cast_possible_truncation`, `cast_sign_loss`,
  `cast_precision_loss`, `cast_lossless`) promoted in barracuda-core.
  `large_stack_frames` documented as test framework artifact. `suboptimal_flops`
  evolved in all test files (mul_add with type annotations). `ode_bio/params.rs`
  refactored into 7-file modular structure. RBF `assemble_and_solve` zero-copy
  via `split_off`. CI: 80% coverage gate and chaos/fault tests now blocking;
  cross-compile job added for musl targets. Dead `ring` config removed from
  deny.toml. All quality gates green.
- **VFIO-primary architecture adoption (Mar 13)**: VFIO via toadStool adopted as
  primary GPU dispatch path. All root docs and specs updated. CoralReefDevice
  evolved to IPC-first architecture (no coral-gpu dependency). VFIO detection
  responsibility moved to toadStool (barraCuda queries via IPC). wgpu demoted to
  development/fallback.
- **Sovereign pipeline deep debt sprint (Mar 12)**: Hand-written `weighted_dot_df64.wgsl`
  (6 kernels with DF64 workgroup accumulators) replaces auto-rewrite for Hybrid devices.
  RHMC multi-shift CG + rational approximation + RHMC HMC absorbed from hotSpring into
  `ops::lattice::rhmc` and `ops::lattice::rhmc_hmc`. `@ilp_region` annotations added to
  high-value DF64 reduction shaders (variance_reduce_df64, weighted_dot_df64,
  mean_variance_df64, covariance_f64). Covariance f64 confirmed auto-rewrite safe
  (thread-local accumulators only). All quality gates green.
- **Deep debt sprint 2 — nursery lints & iterator evolution (Mar 12)**: 5 nursery
  lints promoted (redundant_clone, imprecise_flops, unnecessary_struct_initialization,
  derive_partial_eq_without_eq; suboptimal_flops kept allow with rationale). 193 files
  auto-fixed. All 7 if_same_then_else sites fixed and lint promoted to warn. Iterator
  evolution: csr diagonal, device_info NPU scan, fft_1d twiddle gen converted from
  range loops to idiomatic iterators. Discovery file paths derived from
  PRIMAL_NAMESPACE (3 sites). zeros/ones dispatch duplication eliminated via combined
  match arm. Total: 14 bulk-allowed lints now promoted (9 pedantic + 5 nursery).
  All quality gates green.
- **Comprehensive audit & deep debt sprint (Mar 12)**: Full codebase audit against
  wateringHole standards (uniBin, ecoBin, semantic naming, sovereignty, zero-copy,
  license compliance, code quality). 12-item remediation: `#![forbid(unsafe_code)]`
  in both crates; namespace-derived IPC method names via `PRIMAL_NAMESPACE` +
  `METHOD_SUFFIXES` (LazyLock); 648 WGSL SPDX headers added (806/806 complete);
  9 bulk-allowed pedantic lints promoted to warn (enforced); erfc_f64 recursion
  fix in stable_f64.wgsl; magic numbers extracted (CONSERVATIVE_GPR_COUNT,
  DEFAULT_WORKGROUP, CORAL_CACHE_ARCHITECTURES); zero-copy evolution
  (async_submit::read_bytes, ncbi_cache::load -> bytes::Bytes); unreachable! ->
  debug_assert! + graceful fallback; rustdoc zero warnings; BufferBinding import
  for --all-features clippy.
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

Earlier completions (Mar 7–10) are documented in `CHANGELOG.md` and
`specs/REMAINING_WORK.md`.

---

## Immediate (P1)

- **DF64 NVK end-to-end verification**: Run df64 compilation on Yukawa force kernels through
  NVK/NAK on hardware. Validate sovereign compiler's safe WGSL roundtrip produces correct
  numerical results across all backends. Probe-aware `fp64_strategy()` is now in place to
  auto-fallback if native f64 fails.
- **coralReef sovereign compiler evolution**: coralReef is the unified primal compiler
  and driver for all GPU targets — eventually also the Rust compiler for the ecosystem.
- **~~Dedicated DF64 shaders for covariance + weighted_dot~~**: Done (Mar 12). Hand-written
  `weighted_dot_df64.wgsl` with 6 kernels. Covariance confirmed safe with auto-rewrite
  (thread-local only — no `var<workgroup> array<f64, N>`).
- **`BatchedTridiagEigh` GPU op**: groundSpring local QL implicit eigensolver is a candidate
  for absorption as a batched GPU tridiagonal eigenvector solver.
- **Multi-GPU OOM recovery**: `QuotaTracker` is wired into buffer allocation; next step
  is automatic workload migration when a device hits VRAM quota.
- **Kokkos parity validation baseline**: Document `sarkas_gpu` validation results, extract
  PPPM shader performance numbers for apples-to-apples comparison. Now unblocked by VFIO
  strategy — projected ~4,000 steps/s vs Kokkos 2,630 steps/s.

## Near-term (P2)

- **Test coverage to 90%**: CI 80% gate now blocking (Sprint 3). Evolve `--fail-under`
  from 80 to 90 with real GPU hardware. Add GPU-conditional tests for new ops
  (SCS-CN, Stewart, Blaney-Criddle, autocorrelation).
- **Kokkos GPU parity benchmarks**: Run barraCuda GPU benchmarks on matching hardware,
  publish comparison data.
- **~~WGSL optimizer annotation coverage~~**: Done (Mar 12). `@ilp_region` added to
  variance_reduce_df64, weighted_dot_df64, mean_variance_df64, covariance_f64.
- **~~RHMC multi-shift CG absorb~~**: Done (Mar 12). `rhmc.rs` (RationalApproximation,
  multi_shift_cg_solve, Remez exchange) + `rhmc_hmc.rs` (RhmcConfig, heatbath, action, force).

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
  `EventCodec` → `Bytes` + `CompileResponse::into_bytes()` done; remaining: pre-allocated
  buffers for `domain_ops.rs` CPU fallback clones, LSTM hidden state clones.

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

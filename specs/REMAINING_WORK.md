# barraCuda — Remaining Work

**Version**: 0.3.3+
**Date**: March 9, 2026
**Status**: Active — tracks all open work items for barraCuda evolution

---

## Achieved (March 9, 2026)

### GpuBackend Trait + Sovereign Dispatch Scaffold
- **`GpuBackend` trait** (`device::backend`): Backend-agnostic GPU compute interface —
  9 required methods, 12 default typed convenience methods, blanket `Arc<B>` impl.
- **`WgpuDevice` implements `GpuBackend`**: `dispatch_compute()` encapsulates the full
  wgpu bind→pipeline→dispatch→submit cycle.
- **`ComputeDispatch<'a, B: GpuBackend>`**: Generic over backend, defaults to `WgpuDevice`.
  Zero changes to existing callers.
- **`CoralReefDevice`** scaffold behind `sovereign-dispatch` feature flag.
- **3097 tests pass**, zero clippy warnings, both default and sovereign-dispatch features.

## Achieved (March 7, 2026)

### Zero Unsafe
barraCuda has **zero `unsafe` blocks** in its entire codebase. Every prior
`unsafe` was evolved to safe Rust:

| Former Unsafe | Evolution | Technique |
|---------------|-----------|-----------|
| `create_pipeline_cache` (wgpu FFI) | Deferred until wgpu safe API | Return `None`, field preserved |
| `create_shader_module_passthrough` (SPIR-V) | Safe WGSL roundtrip | naga `wgsl-out` → `create_shader_module` |
| `env::set_var` / `remove_var` (tests) | Pure function testing | `parse_gpu_required(Option<&str>)` |
| `env::remove_var` (device test) | Direct path testing | `with_adapter_selector("auto")` |

### Zero Clippy Warnings
Pedantic + `unwrap_used` — zero warnings across all targets (re-verified Mar 8).

### Deep Debt Audit (March 8-9, 2026)
- **352 formatting violations** fixed (`cargo fmt`)
- **36 clippy warnings** resolved (missing doc backticks, `# Errors`, auto-deref, `#[must_use]`, inline format vars)
- **f64 shader compilation bug** fixed: `SparseGemmF64` and `PeakDetectF64` were using `compile_shader()` (downcasts f64→f32) instead of `compile_shader_f64()`, causing data corruption on non-f64 GPUs. Tests now gated on `get_test_device_if_f64_gpu_available()`.
- **Magic numbers** extracted to named constants: 16 constants across `npu_executor`, `multi_device_pool`, `cpu_executor`, `bfgs`
- **Zero production `panic!()`**: all `panic!()` calls confirmed restricted to `#[cfg(test)]` modules

### Zero-Copy and Coverage Sprint (March 9, 2026)
- **~50 GPU dispatch paths**: `to_le_bytes().collect::<Vec<u8>>()` → `bytemuck::cast_slice()` across pipeline, MD, linalg, reduce, optimize, PDE, grid, lattice ops
- **`GpuBackend::download()`**: Return type `Vec<u8>` → `bytes::Bytes` for zero-copy readback
- **`NpuTensorStorage`**: `Vec<u8>` → `bytes::BytesMut` with `freeze()` zero-copy read path
- **`ShaderCompilation(Arc<str>)`**: Error variant `String` → `Arc<str>` — eliminates clone allocation on 10 DF64 shader error paths
- **GPU fallback estimates**: 13 hardcoded constants → `fallback_estimates::{gflops, vram_bytes}` pattern-matched by vendor and device type
- **Coverage tests**: batch_ipr (3), histogram (4), precision/cpu (22+), staging/ring_buffer (8), staging/unidirectional (7), staging/stateful (3), surrogate/adaptive (4 GPU tests)
- **GPU-heavy test timeouts**: Extended slow-timeout overrides for edge_conv, fft, conv2d, flash_attention
- **CI dual coverage**: 80% baseline + 90% stretch target (continue-on-error)
- **Doc collision fix**: `barracuda-core` binary `doc = false` resolves Cargo #6313

### Systematic f64 Pipeline Evolution (March 8, 2026)
- **14 additional f64 ops** fixed: `transe_score_f64`, `triangular_solve/f64`, `variance_f64`, `correlation_f64`, `covariance_f64`, `hermite_f64`, `bessel_i0/j0/j1/k0`, `beta_f64`, `digamma_f64`, `cosine_similarity_f64`, `weighted_dot_f64` — all were silently producing corrupted data on f64-capable GPUs
- **Pipeline cache f64-native path**: `get_or_create_pipeline_f64_native()` preserves f64 types with separate cache maps; `create_f64_data_pipeline()` auto-selects native vs downcast based on `SHADER_F64` capability
- **`compile_shader()` doc corrected**: now accurately describes f64-canonical always-downcast behavior
- **Zero-copy `CpuTensorStorageSimple`**: evolved from `Vec<u8>` to `Bytes` — `read_to_cpu()` is ref-count bump, not full clone
- **Zero-copy `CosineSimilarityF64::similarity()`**: eliminated `to_vec()` pair via flat-dispatch refactor
- **Pipeline cache hot-path allocations eliminated**: `DeviceFingerprint` uses `std::mem::discriminant` instead of `format!`; `PipelineKey` uses hash instead of `String` for entry point
- **Legacy discovery filename** evolved from hardcoded `coralreef-core.json` to agnostic `shader-compiler.json`
- **Hardcoding audit**: zero hardcoded primal names in production code, zero hardcoded ports, zero hardcoded URLs — all env-var or capability-based

### Sovereign Compiler — All Backends
The sovereign compiler (FMA fusion, dead expression elimination) now runs on
**all backends** (Vulkan, Metal, DX12, WebGPU) via safe WGSL roundtrip.
Previously limited to Vulkan with SPIR-V passthrough.

---

## Remaining Work

### P1 — Immediate

#### DF64 NVK End-to-End Verification
- Run DF64 compilation on Yukawa force kernels through NVK/NAK on hardware
- Validate the sovereign compiler's safe WGSL roundtrip produces correct
  numerical results across all backends
- Probe-aware `fp64_strategy()` is in place for auto-fallback

#### coralNAK Extraction
- When org repo fork lands, create the sovereign NVIDIA shader compiler primal
- See `ecoPrimals/wateringHole/SOVEREIGN_COMPUTE_EVOLUTION.md` Level 2-3

#### coralReef Phase 10 — Verified
- IPC method names evolved to semantic naming (`shader.compile.*`) per wateringHole standard
- `shader.compile.capabilities` endpoint added (preferred over health-embedded arch list)
- AMD RDNA2/RDNA3/CDNA2 architecture mappings added (`gfx1030`, `gfx1100`, `gfx90a`)
- Backward-compat fallback retained for pre-Phase 10 coralReef instances
- Discovery scans for `shader.compile` capability (Phase 10) with `shader_compiler` fallback

### P2 — Near-term

#### Test Coverage to 90%
- Current: 3,450+ total tests, 31 integration suites
- Evolve CI `--fail-under` from 80 to 90
- Add GPU-conditional tests for new ops
- GPU_TEST_TIMEOUT (60s) prevents hangs; coordination harness with
  coralReef + toadStool needed for efficient shader-on-GPU testing

#### Kokkos Validation
- Document `sarkas_gpu` validation results
- Extract PPPM shader performance numbers
- Run GPU benchmarks on matching hardware, publish comparison data
- Gap currently 3.7× (down from 27×); remaining gap is dispatch overhead

#### WGSL Optimizer Annotation Coverage
- Expand `@ilp_region` / `@unroll_hint` annotations across science shaders
- Architecture-specific ILP optimization benefits all backends now

### P3 — Medium-term

#### Pipeline Cache Re-enable
- When wgpu provides a safe `create_pipeline_cache` API, re-enable
- Field + accessor preserved in `WgpuDevice`; `make_pipeline_cache`
  returns `None` until then
- Track wgpu upstream for safe API evolution

#### Multi-GPU Dispatch
- Evolve GpuView to span multiple devices
- Automatic work distribution across primary/secondary adapters

#### Shader Hot-reload
- File watcher for `.wgsl` files during development
- Automatic recompilation through sovereign pipeline

#### Zero-copy Evolution
- Pre-allocated buffers for `domain_ops.rs` CPU fallback clones
- LSTM hidden state clones
- RBF assembly allocations

### P4 — Long-term (Sovereign Compute)

See `SOVEREIGN_PIPELINE_TRACKER.md` for the full sovereign pipeline tracker
including cross-primal dependencies, libc/musl → rustix evolution, and
cross-compilation target matrix.

#### barraCuda's Layer 1 Contribution
barraCuda owns the math. Its remaining long-term contributions to the
sovereign compute stack:

| Item | Description | Depends On |
|------|-------------|------------|
| naga IR optimisations | Deeper FMA patterns, loop unrolling at IR level | naga upstream or fork |
| WGSL → ISA direct path | Bypass SPIR-V entirely for known hardware | coralReef Level 3 |
| CPU shader interpreter | Execute WGSL on CPU without GPU driver | naga + cranelift or custom |
| WebGPU browser target | Compile barraCuda shaders for browser via wasm-pack | wgpu WebGPU backend |
| Distributed compute | Cross-node GPU dispatch via primal-to-primal IPC | songBird + toadStool |

#### Cross-Primal Integration
barraCuda solves the math. coralReef solves the compiler. toadStool solves the
hardware. Each primal contributes its portion to a stable solution:

```
barraCuda (Layer 1 — WHAT to compute)
    WGSL shaders → naga IR → optimise → WGSL
    Zero unsafe, zero C deps, all backends
    ↓
coralReef (Layer 2-3 — HOW to compile) — Phase 10
    SPIR-V/WGSL → native GPU binary (SASS, RDNA2+)
    shader.compile.* semantic IPC, 856 tests
    ↓
toadStool (Layer 3-4 — WHERE to run)
    Hardware discovery, GPU driver, DMA, dispatch
    Vulkan FFI → evolves to coralDriver (pure Rust)
    ↓
Hardware (any GPU, CPU, NPU, Android ARM)
```

---

## C Dependency Chain Status

**barraCuda**: Zero application C deps. Zero unsafe.

Transitive C boundaries (all via wgpu/tokio, not barraCuda code):

| Boundary | Type | Evolves Via |
|----------|------|------------|
| `ash` → `libvulkan.so` | GPU driver FFI | coralReef/toadStool sovereign driver |
| `renderdoc-sys` | Debug capture | Feature-gate out of wgpu |
| `libc` (mio, signal, getrandom) | Kernel ABI (syscalls) | rustix Phase 1-2, then Rust std Phase 3 |

**blake3**: Already pure Rust (`pure` feature flag).

See `SOVEREIGN_PIPELINE_TRACKER.md` for the full libc → rustix evolution
path and cross-compilation target matrix.

---

## Quality Gates — All Green

| Gate | Status | Command |
|------|--------|---------|
| Format | Pass | `cargo fmt --check` |
| Clippy | Pass (zero warnings) | `cargo clippy --all-targets` |
| Rustdoc | Pass | `cargo doc --no-deps` |
| Deny | Pass | `cargo deny check` |
| Check (no GPU) | Pass | `cargo check --no-default-features` |
| Check (GPU only) | Pass | `cargo check --no-default-features --features gpu` |
| Check (all) | Pass | `cargo check` |

---

## References

- `SOVEREIGN_PIPELINE_TRACKER.md` — sovereign pipeline tracker (P0 blocker, libc evolution, cross-primal deps)
- `STATUS.md` — current grade (A+)
- `WHATS_NEXT.md` — prioritised work items + C dependency evolution map
- `CONVENTIONS.md` — coding standards
- `specs/BARRACUDA_SPECIFICATION.md` — crate architecture
- `specs/ARCHITECTURE_DEMARCATION.md` — primal ownership boundaries
- `ecoPrimals/wateringHole/SOVEREIGN_COMPUTE_EVOLUTION.md` — full stack evolution plan
- `ecoPrimals/wateringHole/PURE_RUST_SOVEREIGN_STACK_GUIDANCE.md` — coralReef Layer 2-4 guidance

### Cross-Primal Handoffs Absorbed (Mar 5-7, 2026)

- **coralReef Phase 10 Iter 6** (Mar 7): semantic `shader.compile.*` IPC, AMD RDNA2+, 856 tests
- **toadStool S128-S130** (Mar 6-7): PrecisionRoutingAdvice, coralReef proxy, cross-spring provenance, C dep removal
- **wateringHole groundSpring** (Mar 5-7): rewiring guidance, precision evolution, coralReef integration contracts

# Contributing to barraCuda

Welcome. barraCuda is the sovereign math engine for the ecoPrimals ecosystem.
This guide covers everything you need to start contributing.

### Architecture Note

barraCuda uses a **dual-path GPU dispatch architecture**:
- **VFIO primary** (via toadStool): exclusive device access, IOMMU isolation,
  deterministic scheduling through `SovereignDevice` → [JSON-RPC] shader.compile primal → [JSON-RPC] compute.dispatch primal → GPU
- **wgpu fallback**: Vulkan/Metal/DX12 for development, non-VFIO environments,
  and platforms without IOMMU support

For local development, wgpu (or llvmpipe software rendering) is all you need.
VFIO dispatch requires toadStool's VFIO GPU backend and IOMMU-capable hardware.

---

## Quick Start

```bash
# Prerequisites: Rust 1.87+, GPU drivers or llvmpipe, cargo-deny
git clone https://github.com/ecoPrimals/barraCuda.git
cd barraCuda

# No sibling repos required — barraCuda is fully standalone

# Build
cargo build --workspace

# Test (fully concurrent — no thread limiting needed)
cargo test -p barracuda --lib

# Quality gate
cargo fmt --all -- --check
cargo clippy --workspace -- -D warnings
cargo deny check
cargo doc --workspace --no-deps
```

---

## Repository Layout

| Path | Purpose |
|------|---------|
| `crates/barracuda/` | Umbrella crate — all math, GPU ops, compute fabric |
| `crates/barracuda-core/` | Primal lifecycle, IPC, tarpc, UniBin CLI |
| `crates/barracuda/src/shaders/` | 826 WGSL shaders (see `shaders/README.md`) |
| `crates/barracuda/examples/` | Runnable examples |
| `crates/barracuda/tests/` | 42 integration test files |
| `crates/barracuda/src/bin/` | Binaries (validate_gpu, bench_*) |
| `crates/barracuda-core/src/bin/` | `barracuda` UniBin CLI binary |
| `specs/` | Architecture specs and design documents |

---

## Adding a New Math Operation

### 1. Write the WGSL shader

Create your shader in the appropriate `src/shaders/` subdirectory. Follow
the conventions in `src/shaders/README.md`:

- Name: `snake_case.wgsl` (e.g., `shaders/math/my_op.wgsl`)
- Binding group 0 for inputs, binding group 0 for outputs (sequential bindings)
- Workgroup size: `@workgroup_size(256)` for general elementwise,
  `@workgroup_size(64)` for physics/lattice/memory-heavy (see `CONVENTIONS.md`)
- Include `fn main(@builtin(global_invocation_id) gid: vec3<u32>)` entry point
- Write shaders in f64-canonical form; the precision compiler handles F32/DF64 transforms

### 2. Register the shader

Add a `LazyLock<&str>` constant in the relevant ops module:

```rust
static MY_OP_SHADER: LazyLock<&str> = LazyLock::new(|| {
    include_str!("../shaders/math/my_op.wgsl")
});
```

### 3. Write the Rust dispatch function

```rust
use crate::device::capabilities::WORKGROUP_SIZE_1D;

pub async fn my_op(device: &WgpuDevice, input: &Tensor) -> Result<Tensor> {
    let workgroups = (input.len() as u32).div_ceil(WORKGROUP_SIZE_1D);
    let dispatch = ComputeDispatch::new(device)
        .shader(&MY_OP_SHADER)
        .input_tensor(input)
        .output_size(input.len())
        .dispatch()
        .await?;
    dispatch.into_tensor(input.shape())
}
```

### 4. Add CPU fallback (if appropriate)

In `src/dispatch/`, add a size threshold so small inputs skip the GPU:

```rust
("my_op", 512),
```

### 5. Write tests

- **Unit test** in the same module (`#[cfg(test)]`)
- **Integration test** in `tests/` if the op participates in a pipeline
- Test with `cargo test -p barracuda --lib my_op`
- Verify on GPU: `cargo test -p barracuda --lib my_op -- --nocapture`

### 6. Document

Add a doc comment with math notation, complexity, and a usage example:

```rust
/// Computes element-wise [f(x)] for each element in the input tensor.
///
/// Complexity: O(N) on GPU, dispatched in ceil(N/256) workgroups.
///
/// ```rust,ignore
/// let result = my_op(&device, &input).await?;
/// ```
pub async fn my_op(/* ... */) -> Result<Tensor> { /* ... */ }
```

---

## Adding a New Domain Module

Domain modules (`nn`, `pde`, `genomics`, etc.) are feature-gated. To add one:

1. Create `src/my_domain/mod.rs` (or `src/my_domain.rs`)
2. Add a feature flag in `Cargo.toml`:
   ```toml
   domain-my-domain = []
   ```
3. Add it to the `domain-models` umbrella:
   ```toml
   domain-models = [..., "domain-my-domain"]
   ```
4. Gate the module in `lib.rs`:
   ```rust
   #[cfg(all(feature = "gpu", feature = "domain-my-domain"))]
   pub mod my_domain;
   ```
5. Add prelude re-exports (also gated)
6. Verify: `cargo check --no-default-features --features gpu` (must pass without your domain)

---

## GPU Concurrency Protocol

barraCuda's GPU access uses a three-layer concurrency model. All contributors
must follow these rules to prevent wgpu-core races:

### The three layers

1. **`active_encoders: AtomicU32`** — lock-free counter incremented before any
   wgpu-core activity (buffer creation, shader compilation, command encoding)
   and decremented afterward. Call `device.encoding_guard()` /
   `device.encoding_complete()`, or use `GuardedEncoder` for RAII.
2. **`gpu_lock: Mutex<()>`** — serializes `queue.submit()` and `device.poll()`.
   Submit and poll use **separate** lock acquisitions so other threads can
   interleave submits while one thread polls. Before proceeding,
   `brief_encoder_wait()` yields until active encoders reach zero
   (microsecond-scale CPU work, never blocks).
3. **`dispatch_semaphore`** — hardware-aware cap (2 for CPU/llvmpipe, 8 for
   discrete GPU) preventing driver overload.

### Rules

- **Fire-and-forget**: `device.submit_commands(...)` — for ops returning `Tensor` (GPU-resident result, no readback needed)
- **Submit + readback**: `device.submit_and_map::<T>(commands, &staging, count)` — single-poll path for reading back GPU results
- **Read back**: `device.read_buffer::<T>(buffer, count)` — never map buffers manually
- **Poll**: `device.poll_safe()` — never call `device.device().poll()` directly
- **Resource creation**: `GuardedDeviceHandle` auto-protects all `device.device.create_*()` calls
  with atomic encoder barriers — no manual guarding needed
- **Device creation**: serialized via global `DEVICE_CREATION_LOCK`

Bypassing these methods causes non-deterministic crashes under concurrent load.
If you need raw access for a new pattern, add a method to `WgpuDevice`.

---

## Running Tests

```bash
# All unit tests (fully concurrent, no thread limiting needed)
cargo test -p barracuda --lib

# Specific test suite
cargo test -p barracuda --test hardware_verification

# Specific test function
cargo test -p barracuda --lib test_matmul_f64

# FHE tests only
cargo test -p barracuda --lib fhe

# With output
cargo test -p barracuda --lib my_test -- --nocapture

# GPU validation canary
cargo run -p barracuda --bin validate_gpu --features gpu

# Coverage report
cargo llvm-cov --workspace --lib --html

# No-GPU build check (must always pass)
cargo check -p barracuda --no-default-features

# Tiered test dispatch (local dev — fast iteration)
./scripts/test-tiered.sh           # Tiers 1-3 (clippy + core + targeted)
./scripts/test-tiered.sh full      # All tests via nextest
./scripts/test-tiered.sh gpu       # Full suite + GPU workload tests
```

### Test categories

| Category | Location | What it covers |
|----------|----------|---------------|
| Unit tests | `src/**/*.rs` (`#[cfg(test)]`) | Individual functions, shaders |
| Integration | `tests/*.rs` | Cross-module pipelines |
| E2E (compute) | `tests/scientific_e2e_tests.rs` | Full device-to-result flows |
| E2E (IPC) | `barracuda-core/tests/ipc_e2e.rs` | TCP server, JSON-RPC wire protocol, multi-request |
| Chaos | `tests/scientific_chaos_tests.rs` | Random failures, recovery |
| Fault injection | `tests/fhe_fault_tests.rs` | Error paths, graceful degradation |
| Cross-hardware | `tests/cross_hardware_parity.rs` | Multi-adapter parity |
| Property | `tests/property_tests.rs` | Statistical invariants |
| Pooling | `tests/pooling_tests.rs` | Multi-device resource management |

### GPU vs CPU testing

Tests run on whatever `wgpu` discovers. On machines without a GPU, `llvmpipe`
provides software rendering. To force CPU-only:

```bash
WGPU_BACKEND=vulkan WGPU_ADAPTER_NAME=llvmpipe cargo test -p barracuda
```

---

## Code Style

- Follow `CONVENTIONS.md`
- `#![forbid(unsafe_code)]` in both crates — zero unsafe is irrevocable
- `cargo fmt` before committing
- `cargo clippy --workspace -- -D warnings` must be clean
- Suppressions: `#[expect(clippy::lint, reason = "...")]` — compile-time verified; `#[allow]` only when lint may or may not fire depending on context
- No `anyhow` — use `thiserror` with `BarracudaError`
- No `println!` in library code — use `tracing`
- No `Arc<wgpu::Device>` / `Arc<wgpu::Queue>` — wgpu 28 types are `Clone`

### Error handling

- Return `Result<T, BarracudaError>` — never `unwrap()` or `expect()` in library code
- Prefer `let-else` or `?` over `expect()`; for `RwLock`, use `unwrap_or_else(PoisonError::into_inner)` to recover
- Use `BarracudaError::DeviceLost` for GPU device loss — enables retry logic
- Check `error.is_retriable()` before retrying operations
- In tests: use `with_device_retry` from `test_pool::test_prelude` for GPU resilience

### Naming

- Modules: `snake_case`
- Shaders: `snake_case.wgsl`
- GPU ops: `async fn op_name(device: &WgpuDevice, ...) -> Result<Tensor>`
- CPU math: `fn op_name(...) -> Result<T>`
- Feature flags: `domain-{name}` for domain modules
- IPC methods: `barracuda.{namespace}.{action}` (semantic, dot-separated)

---

## Feature Flag Discipline

- `gpu` gates all GPU-dependent code (device, tensor, shaders, ops)
- `domain-*` gates domain-specific models independently

When adding new code, ask: "Does this compile with `--no-default-features`?"
Pure math modules must work without GPU. GPU modules must work without
domain models.

```bash
# Verify all three configurations
cargo check --no-default-features                    # pure math
cargo check --no-default-features --features gpu     # math + GPU
cargo check                                          # everything
```

---

## Commit Messages

Follow conventional commits:

```
feat(ops): add element-wise sinh shader
fix(fhe): correct modular reduction overflow at degree 8192
refactor(device): serialize device creation globally
perf(sample): parallelize Nelder-Mead solvers with rayon
test(pde): add Richards equation convergence integration test
docs(shaders): update README with audio shader section
```

---

## Pull Request Process

1. Branch from `main`: `git checkout -b feat/my-feature`
2. Make changes, add tests
3. Run the quality gate:
   ```bash
   cargo fmt --all -- --check
   cargo clippy --workspace -- -D warnings
   cargo deny check
   cargo check --no-default-features
   cargo check --no-default-features --features gpu
   cargo check
   cargo test -p barracuda --lib
   ```
4. Push and open PR
5. PR description: what changed, why, how to test

---

## Architecture Quick Reference

barraCuda is organized in layers:

**Layer 1 — Pure Math** (no GPU, always available):
`linalg`, `special`, `numerical`, `spectral`, `stats`, `sample`, `nautilus`

**Layer 2 — GPU Math** (requires `gpu` feature):
`ops`, `tensor`, `shaders`, `interpolate`, `optimize`, `unified_math`

**Layer 3 — Compute Fabric** (requires `gpu` feature):
`device`, `staging`, `pipeline`, `dispatch`, `multi_gpu`, `compute_graph`,
`scheduler`, `session`, `unified_hardware`

**Layer 4 — Domain Models** (requires `gpu` + `domain-*` features):
`nn`, `snn`, `esn_v2`, `pde`, `genomics`, `vision`, `timeseries`

**Layer 5 — Primal Lifecycle** (`barracuda-core`):
IPC (JSON-RPC 2.0), tarpc, UniBin CLI, lifecycle/health traits

Code in layer N may depend on layers 1..N-1 but never on N+1.

---

## Where to Find Help

- `specs/BARRACUDA_SPECIFICATION.md` — crate architecture, IPC, shader pipeline
- `specs/ARCHITECTURE_DEMARCATION.md` — barraCuda vs toadStool boundaries
- `crates/barracuda/src/shaders/README.md` — shader guide
- `BREAKING_CHANGES.md` — migration notes
- `CHANGELOG.md` — version history
- `wateringHole/` — ecosystem-wide standards (JSON-RPC, UniBin, genomeBin)

# Coding Conventions

This primal follows the ecoPrimals coding conventions, internalized from
the sourDough scaffold. barraCuda owns its own standards.

## Quick Reference

- **Edition**: 2024
- **MSRV**: 1.87
- **GPU stack**: wgpu 28, naga 28 — `Device` and `Queue` are `Clone` (no `Arc` wrappers)
- **Linting**: `warn(clippy::all, clippy::pedantic, clippy::nursery)` — configured in `Cargo.toml` `[lints]`
- **Promoted lints**: pedantic + nursery (blanket) + `missing_errors_doc` + `missing_panics_doc` + cast lints in `barracuda-core` + `suboptimal_flops` + `use_self` + `tuple_array_conversions` + `needless_range_loop` — all enforced via `-D warnings`. Scientific/GPU false positives (`missing_const_for_fn`, `suspicious_operation_groupings`, `future_not_send`, etc.) selectively allowed with rationale in `Cargo.toml`.
- **Suppressions**: `#[expect(clippy::lint, reason = "...")]` — compile-time verified; `#[allow]` only for context-dependent lints in the main `barracuda` crate (e.g. `suspicious_arithmetic_impl` in complex division, `unwrap_used` in integration tests outside `cfg_attr(test)` scope). Zero `#[allow(` in `barracuda-naga-exec`.
- **Docs**: `#![warn(missing_docs, missing_errors_doc, missing_panics_doc)]` — `RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps` clean
- **Unsafe**: `#![forbid(unsafe_code)]`
- **Max file size**: 1000 LOC
- **Test coverage**: 90%+ target (currently ~72% line / ~78% function on llvmpipe; GPU hardware needed for 90%)

## Error Handling

- Return `Result<T, BarracudaError>` — never `unwrap()` or `expect()` in library code
- Use `BarracudaError::DeviceLost` for GPU device loss
- Check `error.is_retriable()` before automatic retries
- In tests: use `with_device_retry` from `test_pool::test_prelude`

## GPU Concurrency

barraCuda uses a three-layer concurrency model for wgpu-core safety:

1. **`active_encoders: AtomicU32`** — lock-free counter. Call `encoding_guard()`
   before any wgpu-core activity (buffer creation, shader compilation, command
   encoding) and `encoding_complete()` when done. Multiple threads increment
   concurrently with zero contention.
2. **`gpu_lock: Mutex<()>`** — serializes `queue.submit()` and `device.poll()`.
   Before proceeding, a bounded yield loop (`brief_encoder_wait`) waits for
   active encoders to reach zero.
3. **`dispatch_semaphore`** — hardware-aware cap (2 for CPU/llvmpipe, 8 for
   discrete GPU) preventing driver overload.

### Rules

- All GPU access through `WgpuDevice` synchronized methods or `ComputeDispatch`
- Never call `queue.submit()`, `device.poll()`, or `buffer.map_async()` directly
- `GuardedDeviceHandle` auto-protects all `device.create_*()` calls — no manual
  `encoding_guard()` / `encoding_complete()` needed for resource creation
- Use `GuardedEncoder` (via `create_encoder_guarded()`) for RAII encoder lifecycle
- Device creation serialized via `DEVICE_CREATION_LOCK`
- Device-lost flag only set for genuine device-lost errors, not validation errors

## Workgroup Sizes

Dispatch workgroup counts must use named constants from `device::capabilities`,
never magic number literals:

| Constant | Value | Use case |
|----------|-------|----------|
| `WORKGROUP_SIZE_1D` | 256 | General-purpose 1D elementwise shaders |
| `WORKGROUP_SIZE_COMPACT` | 64 | Physics, lattice, and memory-heavy shaders |
| `WORKGROUP_SIZE_2D` | 16 | 2D dispatch (16×16 = 256 threads) |

These constants define the Rust-side `div_ceil()` divisor. The matching
`@workgroup_size(N)` in each WGSL shader must agree — changing a constant
requires updating the corresponding shader.

```rust
use crate::device::capabilities::{WORKGROUP_SIZE_1D, WORKGROUP_SIZE_COMPACT};

let workgroups = (n as u32).div_ceil(WORKGROUP_SIZE_1D);       // 256-thread shaders
let workgroups = (n as u32).div_ceil(WORKGROUP_SIZE_COMPACT);  // 64-thread shaders
```

## Device Handle Patterns

wgpu 28 makes `Device` and `Queue` internally `Clone` (cheap `Arc` under the hood).
Do **not** wrap them in `Arc<wgpu::Device>` or `Arc<wgpu::Queue>`.

```rust
let device: wgpu::Device = wgpu_device.device_clone();
let queue: wgpu::Queue = wgpu_device.queue_clone();
```

`GuardedDeviceHandle` derefs to `wgpu::Device` and auto-protects all `create_*`
calls with the atomic encoder barrier.

## Device capabilities

Use **`DeviceCapabilities`** as the canonical type for device feature and capability
queries. `GpuDriverProfile` was removed in v0.3.8 — all driver-profile logic now
lives in `DeviceCapabilities` and `PrecisionBrain`.

## IPC

- JSON-RPC 2.0 primary, tarpc secondary
- Methods: `{domain}.{operation}` per wateringHole semantic naming standard
- Legacy `barracuda.{domain}.{operation}` accepted for backward compatibility
- Errors: standard JSON-RPC error codes

---

*Consistency is the foundation of collaboration.*

# Coding Conventions

This primal follows the ecoPrimals coding conventions, internalized from
the sourDough scaffold. barraCuda owns its own standards.

## Quick Reference

- **Edition**: 2021
- **MSRV**: 1.87
- **Linting**: `warn(clippy::all, clippy::pedantic)` — configured in `Cargo.toml` `[lints]`
- **Suppressions**: `#[expect(clippy::lint, reason = "...")]` — compile-time verified, never `#[allow]`
- **Docs**: `#![warn(missing_docs)]` — `cargo doc --workspace --no-deps` clean with `-D warnings`
- **Unsafe**: `#![deny(unsafe_code)]`
- **Max file size**: 1000 LOC
- **Test coverage**: 90%+ target (currently ~80% unit, growing; remaining gap is GPU-only code)

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

## IPC

- JSON-RPC 2.0 primary, tarpc secondary
- Methods: `barracuda.{namespace}.{action}` (semantic, dot-separated)
- Errors: standard JSON-RPC error codes

---

*Consistency is the foundation of collaboration.*

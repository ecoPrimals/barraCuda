# Coding Conventions

This primal follows the ecoPrimals coding conventions.

See: `../sourDough/CONVENTIONS.md` for complete guidelines.

## Quick Reference

- **Edition**: 2021
- **MSRV**: 1.87
- **Linting**: `warn(clippy::all, clippy::pedantic)` — configured in `Cargo.toml` `[lints]`
- **Docs**: `#![warn(missing_docs)]` — `cargo doc --workspace --no-deps` clean with `-D warnings`
- **Unsafe**: `#![deny(unsafe_code)]`
- **Max file size**: 1000 LOC
- **Test coverage**: 90%+ target (currently ~80% unit, growing; remaining gap is GPU-only code)

## Error Handling

- Return `Result<T, BarracudaError>` — never `unwrap()` or `expect()` in library code
- Use `BarracudaError::DeviceLost` for GPU device loss
- Check `error.is_retriable()` before automatic retries
- In tests: use `with_device_retry` from `test_pool::test_prelude`

## GPU Access

- All GPU access through `WgpuDevice` synchronized methods
- Never call `queue.submit()`, `device.poll()`, or `buffer.map_async()` directly
- Device creation serialized via `DEVICE_CREATION_LOCK`

## IPC

- JSON-RPC 2.0 primary, tarpc secondary
- Methods: `barracuda.{namespace}.{action}` (semantic, dot-separated)
- Errors: standard JSON-RPC error codes

---

*Consistency is the foundation of collaboration.*

# Breaking Changes

Track API-breaking changes here. Each entry includes the version, what changed,
and the migration path.

## Pre-1.0 (current)

### 0.3.3

| Change | Migration |
|--------|-----------|
| **wgpu 22 → 28** — `Maintain::Wait` / `MaintainBase::Poll` removed | Replace `Maintain::Wait` with `PollType::Wait { timeout }` and `MaintainBase::Poll` with `PollType::Poll`. `device.poll()` now returns `Result<PollStatus, PollError>` — match on the `Result` and call `.is_queue_empty()` on the `PollStatus`. |
| **Bounded GPU poll timeout** — `poll_safe()` and `submit_and_poll_inner()` now use configurable timeout (default 120s, env `BARRACUDA_POLL_TIMEOUT_SECS`) | Indefinite `PollType::Wait` hangs (the `llvm-cov` GPU hang) are replaced by a bounded timeout that surfaces as `BarracudaError::execution_failed`. Set `BARRACUDA_POLL_TIMEOUT_SECS` for CI/coverage environments. |
| **NVK SPIR-V exclusion** — NVK (Nouveau Vulkan) is excluded from SPIR-V passthrough paths | NVK claims Vulkan 1.3 but lacks reliable f64 SPIR-V support on Volta. barraCuda routes NVK through WGSL-only paths. If you were relying on SPIR-V passthrough for NVK, switch to WGSL shaders. |
| **DF64 Hybrid fallback removed** — ops no longer silently fall back to native f64 when DF64 rewrite fails on Hybrid devices | On Hybrid devices (e.g., Volta via NVK), if the sovereign DF64 rewrite cannot transform a shader, operations now return `BarracudaError::ShaderCompilation` instead of silently producing zeros. Affected ops: covariance, weighted_dot, hermite, digamma, cosine_similarity, beta, bessel_i0, bessel_j0, bessel_j1, bessel_k0. |
| **GPU f64 computational accuracy probe** — `get_test_device_if_f64_gpu_available()` now requires real hardware + `SHADER_F64` + a computation probe | Tests using f64 GPU paths are gated by a runtime probe that dispatches `3.0 * 2.0 + 1.0` and verifies the result is `7.0`. Software rasterizers (llvmpipe/lavapipe) that claim `SHADER_F64` but produce incorrect results are correctly excluded. |
| **naga 22.1 → 28** — new IR variants | If matching on `naga::Statement` / `naga::Expression`, add wildcard arms for new variants (`SubgroupBallot`, `SubgroupCollectiveOperation`, `RayQuery`, etc.). |
| `device_arc()` → `device_clone()` returns `wgpu::Device` (was `Arc<wgpu::Device>`) | wgpu 28 `Device` implements `Clone` internally. Replace `.device_arc()` with `.device_clone()`. The returned type is `wgpu::Device`, not `Arc<wgpu::Device>`. |
| `queue_arc()` → `queue_clone()` returns `wgpu::Queue` (was `Arc<wgpu::Queue>`) | Same as above. Replace `.queue_arc()` with `.queue_clone()`. |
| `inner_arc()` removed from `GuardedDeviceHandle` | `GuardedDeviceHandle` derefs to `wgpu::Device`. Use `device.device.clone()` if you need an owned `wgpu::Device` (cheap — internally `Arc`-wrapped). |
| `PppmGpu::new()` / `new_with_driver()` take `wgpu::Device` + `wgpu::Queue` (were `Arc<...>`) | Pass plain `wgpu::Device` and `wgpu::Queue` values. Remove `Arc::new(...)` wrappers. |
| `ComputeGraph` stores `wgpu::Queue` (was `Arc<wgpu::Queue>`) | No migration needed for most callers. Direct field access changes type from `Arc<wgpu::Queue>` to `wgpu::Queue`. |
| `create_shader_module_spirv()` → `create_shader_module_passthrough()` | Rename call sites. Function semantics are identical. |
| `on_uncaptured_error` handler is `Arc<dyn UncapturedErrorHandler>` (was `Box<dyn Fn(...)>`) | Implement `UncapturedErrorHandler` trait on your error handler struct and wrap in `Arc`. |
| Workgroup dispatch uses named constants | If you used `div_ceil(64)` or `div_ceil(256)` directly, import `WORKGROUP_SIZE_COMPACT` (64) or `WORKGROUP_SIZE_1D` (256) from `device::capabilities`. |
| tokio workspace version → 1.49 | If your crate uses `tokio = { workspace = true }`, you get 1.49 automatically. |

### 0.3.2

| Change | Migration |
|--------|-----------|
| `sourdough-core` dependency removed; `PrimalLifecycle`, `PrimalHealth`, `PrimalState`, `HealthStatus`, `HealthReport` are now in `barracuda_core::lifecycle` and `barracuda_core::health` | Replace `use sourdough_core::{PrimalLifecycle, ...}` with `use barracuda_core::lifecycle::PrimalLifecycle` and `use barracuda_core::health::PrimalHealth`. Error type is now `BarracudaCoreError` instead of `PrimalError`. |
| `WgpuDevice::lock()` removed; GPU serialization now uses `gpu_lock: Mutex<()>` + `active_encoders: AtomicU32` | If you called `device.lock()`, switch to `device.encoding_guard()` / `device.encoding_complete()` pairs, or use `ComputeDispatch` which handles this automatically. |
| `GuardedEncoder` no longer has a lifetime parameter; `encoder` field is `Option<CommandEncoder>` | Call `guarded.finish()` to get `CommandBuffer` (consumes the guard). Drop also auto-decrements the encoder count. |
| `encoding_guard()` returns `()` (was previously a guard struct) | Remove `let _guard = device.encoding_guard()` patterns. Call `device.encoding_guard()` directly, and pair with `device.encoding_complete()` when done. |
| `gpu_lock_arc()` returns `Arc<Mutex<()>>` (was `Arc<RwLock<()>>`) | Change `.write()` to `.lock()` and `.try_write()` to `.try_lock()`. |
| `BufferPool::new()` takes additional `Arc<AtomicU32>` parameter for active encoder count | Pass `device.active_encoders_arc()` as the third argument. |
| `WgpuDevice::device` field is now `GuardedDeviceHandle` (was `Arc<wgpu::Device>`) | `GuardedDeviceHandle` derefs to `wgpu::Device`. All `create_*` calls are auto-guarded — remove manual `encoding_guard()` / `encoding_complete()` pairs around resource creation. Use `.device_clone()` (0.3.3+) for an owned handle. |
| `async-trait` removed — trait methods return `BoxFuture` | Replace `#[async_trait] async fn` with `fn method() -> BoxFuture<'_, Result<T>>` using `Box::pin(async move { ... })`. Import `BoxFuture` from `barracuda_core::unified_hardware::traits`. |
| `ComputeGraph::device` field is now `GuardedDeviceHandle` | Use `.device_clone()` (0.3.3+) for an owned `wgpu::Device`. |

### 0.3.1

| Change | Migration |
|--------|-----------|
| `MatmulResult` fields renamed: `lhs_id`/`rhs_id` → `result_id`/`shape` | Update any code reading `MatmulResult.lhs_id` to use `.result_id`. The `.shape` field now contains the result tensor shape. |
| `FheResult` split into `FheNttResult` and `FhePointwiseMulResult` | Replace `FheResult` with the specific type. Both new types have a `result: Vec<u64>` field containing the output coefficients. |
| `DispatchResult` fields changed: `shader`/`entry_point` → `tensor_id`/`shape`/`data` | The result now carries the actual tensor state instead of echoing the input operation names. |
| `compute_dispatch` tarpc signature changed to `(op, shape, tensor_id)` | Was `(shader, entry_point)`. The new signature matches the JSON-RPC handler and actually creates/reads tensors. |
| `fhe_ntt` tarpc signature expanded: `(modulus, degree, root_of_unity, coefficients)` | Was `(modulus, degree)`. Now matches JSON-RPC handler and executes the NTT. |
| `fhe_pointwise_mul` tarpc signature expanded: `(modulus, degree, a, b)` | Was `(modulus)`. Now matches JSON-RPC handler and executes the multiplication. |
| `tensor_create` tarpc signature expanded: `(shape, dtype, data)` | `data` is `Option<Vec<f32>>`. Was `(shape, dtype)`. Allows initializing tensors with data. |

### 0.3.0

| Change | Migration |
|--------|-----------|
| `toadstool` feature flag removed | barraCuda no longer depends on `toadstool-core`. Remove `features = ["toadstool"]` from your Cargo.toml. Device discovery uses `WgpuDevice::new()` / `Auto::new()` directly. |
| `npu-akida` feature flag removed | `NpuMlBackend` and `npu::ops::*` are no longer available. NPU execution is the responsibility of the consuming primal or orchestrator. `EventCodec` and `npu::constants` remain. |
| `read_f64_raw` / `read_i32_raw` now take `&WgpuDevice` instead of `(&wgpu::Device, &wgpu::Queue)` | Pass the `WgpuDevice` reference instead of raw wgpu handles |
| `sparsity_sampler` and `sparsity_sampler_gpu` require `F: Fn + Sync` | Add `+ Sync` bound to closures passed to sparsity samplers |
| `PppmGpu` no longer exposes `adapter_info()`, `device_arc()`, `queue_arc()` | Use `wgpu_device()` to access the `WgpuDevice` wrapper |
| Autotune functions use `&impl GpuDeviceForCalibration` trait instead of raw handles | Implement `GpuDeviceForCalibration` or use `WgpuDevice` directly |
| `ComputeGraph::new` requires `Arc<WgpuDevice>` parameter | Pass the device wrapper when constructing compute graphs |

### Differences from toadStool's barracuda

These are not "breaking changes" (barraCuda is a new crate) but document what
consumers migrating from toadStool should know:

| Item | toadStool | barraCuda standalone |
|------|-----------|---------------------|
| `discover_devices()` | Always available (via `toadstool-core`) | Removed — use `WgpuDevice::new()` or `Auto::new()` |
| `select_best_device()` | Always available | Removed — use `WgpuDevice::discover_best_adapter()` |
| `WgpuDevice::from_selection()` | Always available | Removed — use `WgpuDevice::new()` directly |
| `NpuMlBackend` | Always compiled (via `akida-driver`) | Removed — NPU execution is orchestrator's responsibility |
| `npu::ops::*` | Always compiled | Removed — NPU ops live in the consuming primal |
| `is_npu_available()` | Checks hardware | Use `detect_akida_boards()` (returns empty if none) |
| GPU access | Direct `device.device()` / `device.queue()` calls common | All access via synchronized `WgpuDevice` methods with atomic encoder barrier |
| Device creation | Unguarded | Serialized via global `DEVICE_CREATION_LOCK` |
| MSRV | 1.80 | 1.87 |

### Migration path for Springs

```toml
# Old (toadStool-embedded, deprecated):
barracuda = { path = "../../phase1/toadStool/crates/barracuda" }

# New (standalone barraCuda):
barracuda = { path = "../../barraCuda/crates/barracuda" }
```

No code changes needed. All `use barracuda::*` imports work identically.
hotSpring confirmed this with 716/716 tests passing after a single-line
Cargo.toml path swap.

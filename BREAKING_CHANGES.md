# Breaking Changes

Track API-breaking changes here. Each entry includes the version, what changed,
and the migration path.

## Pre-1.0 (current)

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
| GPU access | Direct `device.device()` / `device.queue()` calls common | All access via synchronized `WgpuDevice` methods |
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

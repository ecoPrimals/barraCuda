# Breaking Changes

Track API-breaking changes here. Each entry includes the version, what changed,
and the migration path.

## Pre-1.0 (current)

### 0.3.0

| Change | Migration |
|--------|-----------|
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
| `discover_devices()` | Always available | Requires `features = ["toadstool"]` |
| `select_best_device()` | Always available | Requires `features = ["toadstool"]` |
| `WgpuDevice::from_selection()` | Always available | Requires `features = ["toadstool"]` |
| `NpuMlBackend` | Always compiled | Requires `features = ["npu-akida"]` |
| `npu::ops::*` | Always compiled | Requires `features = ["npu-akida"]` |
| `is_npu_available()` | Checks hardware | Returns `false` without `npu-akida` |
| GPU access | Direct `device.device()` / `device.queue()` calls common | All access via synchronized `WgpuDevice` methods |
| Device creation | Unguarded | Serialized via global `DEVICE_CREATION_LOCK` |
| MSRV | 1.80 | 1.87 |

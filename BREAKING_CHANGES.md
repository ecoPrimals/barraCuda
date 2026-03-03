# Breaking Changes

Track API-breaking changes here. Each entry includes the version, what changed,
and the migration path.

## Pre-1.0 (current)

No breaking changes yet — API is inherited from toadStool's barracuda crate.
The first breaking change will trigger the 1.0.0 release with full SemVer
guarantees.

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
| MSRV | 1.80 | 1.87 |

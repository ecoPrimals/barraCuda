# barraCuda Showcase Collection
## Sovereign Math Engine Demonstrations

**Status**: Active | **Updated**: March 9, 2026

---

## Overview

barraCuda showcases demonstrate GPU-accelerated scientific computing via WGSL
shaders dispatched through wgpu. Demos progress from local capabilities through
IPC protocol to cross-primal compute orchestration with toadStool (hardware
discovery) and coralReef (sovereign shader compilation).

All demos are pure Rust, zero unsafe, AGPL-3.0-only.

---

## Showcases

### 0. Local Primal (`00-local-primal/`)

Standalone barraCuda capabilities — no other primals required.

- **01-device-discovery** — GPU detection, capability scoring, precision routing
- **02-precision-tiers** — F32 vs F64 vs DF64 comparison on identical math
- **03-fused-gpu-ops** — Welford variance, fused correlation, GpuView chains
- **04-science-shaders** — Molecular dynamics, spectral analysis, Hill kinetics

### 1. IPC Protocol (`01-ipc-protocol/`)

barraCuda as a primal in the ecoPrimals ecosystem.

- **01-jsonrpc-server** — Start server, query methods, exercise JSON-RPC 2.0
- **02-doctor-validate** — Health diagnostics and GPU validation canary

### 2. Cross-Primal Compute (`02-cross-primal-compute/`)

The sovereign compute triangle: toadStool + barraCuda + coralReef.

- **01-coralreef-shader-compile** — WGSL to native binary via coralReef IPC
- **02-toadstool-hw-discovery** — Hardware inventory feeding GPU selection
- **03-sovereign-pipeline** — Full pipeline: discover, route, compile, dispatch

---

## Prerequisites

- Rust toolchain (edition 2024, MSRV 1.87)
- GPU with Vulkan or Metal support (llvmpipe software rendering works for demos)
- barraCuda workspace built: `cd ../.. && cargo build --release`

For cross-primal demos:
- coralReef running for `02-cross-primal-compute/01-coralreef-shader-compile`
- toadStool running for `02-cross-primal-compute/02-toadstool-hw-discovery`
- Cross-primal demos gracefully degrade if other primals are unavailable

## Building

Each showcase is a standalone workspace excluded from the main build.
Build individually:

```bash
cd showcase/00-local-primal/01-device-discovery
cargo run --release
```

## License

AGPL-3.0-only — same as all barraCuda code.

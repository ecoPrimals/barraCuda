# Architecture Demarcation: barraCuda / toadStool / songBird

**Version**: 1.0.0
**Date**: March 2, 2026
**Status**: Active — defines ownership boundaries for Phase 1 primals
**Origin**: toadStool S89, post-barraCuda extraction

---

## Overview

The ecoPrimals compute stack is a 3-layer architecture. Each layer owns a
distinct concern. Springs (domain validation projects) consume the stack from
the top; infrastructure primals provide services from the bottom.

```
Springs (hotSpring, wetSpring, neuralSpring, airSpring, groundSpring)
    │
    ▼
barraCuda ── "WHAT to compute" ── math library + wgpu compute fabric
    │
    ▼
toadStool ── "WHERE and HOW" ── hardware orchestration + multi-framework routing
    │
    ├── songBird ── "the wire" ── network, discovery, NAT traversal
    ├── bearDog  ── "the vault" ── crypto, entropy, PKI
    └── nestGate ── "the shelf" ── storage, artifacts, data pipelines
```

Springs depend on barraCuda directly for math. They depend on toadStool
optionally when they need managed compute (multi-node, adaptive tuning,
multi-framework routing, container isolation).

---

## Layer 1: barraCuda — "WHAT to compute"

**Role**: Sovereign math library. Portable GPU-accelerated scientific computing.

barraCuda is a Rust library (`cargo add barracuda`). It owns:

### Pure Math (no GPU required)
- `linalg` — LU, QR, SVD, Cholesky, eigensolvers, sparse (CSR SpMV)
- `special` — erf, gamma, beta, Bessel, Legendre, Hermite, Laguerre
- `numerical` — gradients, integration, ODE solvers
- `spectral` — Anderson localization, Lanczos, tridiagonal eigensolve
- `stats` — bootstrap, correlation, regression, distributions
- `sample` — direct, LHS, maximin, Metropolis, Sobol, sparsity

### GPU Math (WGSL shaders, wgpu execution)
- `ops` — matmul, softmax, element-wise, reductions, FHE (NTT/INTT/pointwise)
- `tensor` — GPU tensor type, buffer management, memory layout
- `shaders` — 767 WGSL shaders (f32, f64, DF64), sovereign compiler pipeline
- `interpolate` — cubic spline, kriging
- `optimize` — L-BFGS, Brent, Nelder-Mead (GPU variants)
- `unified_math` — unified math routing across precision levels

### Compute Fabric (wgpu device layer)
- `device` — WgpuDevice, capabilities, adapter probing, registry
- `staging` — ring buffers, unidirectional pipelines (CPU→GPU→CPU math streaming)
- `pipeline` — ComputeDispatch, batched stateful, reduce, cascade pipelines
- `compute_graph` — DAG of compute operations
- `dispatch` — size-based CPU/GPU threshold routing
- `multi_gpu` — GpuPool, multi-device load balancing
- `scheduler` — hardware scoring for operation placement
- `session` — tensor session management

### Domain Models (consumers of math — candidates for Spring migration)
- `nn`, `snn`, `esn_v2`, `vision`, `timeseries` → neuralSpring
- `pde` → hotSpring / groundSpring
- `genomics` → wetSpring
- `nautilus` — stays (barraCuda's own evolutionary optimizer, CPU-only)
- `surrogate` — stays (math primitive, used by `sample` core module)

### What barraCuda does NOT own
- Multi-framework routing (CUDA, ROCm, OpenCL, Metal native APIs)
- Hardware fingerprinting and adaptive workgroup tuning
- Multi-node distribution
- Network, storage, crypto
- Container isolation, secure enclaves

### Key property
barraCuda uses wgpu as its sole GPU abstraction. It is vendor-agnostic at the
API level — same WGSL shader produces identical results on NVIDIA, AMD, Intel,
Apple, and llvmpipe. It does NOT use CUDA, ROCm, or OpenCL directly.

---

## Layer 2: toadStool — "WHERE and HOW to compute"

**Role**: Hardware orchestration service. Routes computation to the best
available substrate with adaptive tuning and multi-node distribution.

toadStool is a service/orchestrator that wraps barraCuda (and other compute
backends) with hardware intelligence. It owns:

### Multi-Framework GPU Routing (`runtime/gpu`)
- CUDA, OpenCL, Vulkan, ROCm, Metal, WebGPU, DirectCompute backends
- "This NVIDIA card supports CUDA natively → use CUDA path"
- "This AMD card → ROCm; fallback → wgpu/barraCuda"
- Self-healing, recursive GPU workloads, unified memory

### Substrate Orchestration (`runtime/orchestration`)
- Workload analysis: "this job should go to GPU, this to NPU, this to CPU cluster"
- Load balancing, fallback chains, performance feedback loops
- Policy-based routing

### Adaptive Tuning (`runtime/adaptive`)
- GPU fingerprinting per specific hardware (RTX 3090 vs Titan vs RX 7900 XT)
- Runtime profiling and workgroup size optimization for barraCuda ops
- Optimization caching across sessions

### Hardware Abstraction (`toadstool-core`)
- `HardwareDevice`, `HardwareManager`, `HardwareType`
- Direct hardware access in Rust (no scripts, no sudo)
- Device discovery across all substrates

### Universal Compute (`runtime/universal`)
- Unified CPU/GPU/NPU as parallel compute units
- Capability-based discovery, no hardcoded backends

### Distribution & Services
- `server`/`client` — HTTP/WebSocket/Unix compute service API
- `distributed` — multi-node compute via songBird
- `container` — Docker/Containerd/Podman isolation
- `secure_enclave` — zero-knowledge compute with NestGate/BearDog

### Compilation
- Compiling arbitrary math expressions to optimized models for target hardware
- Runtime selection (native process, WASM, Python, GPU, NPU)

### What toadStool does NOT own
- Math algorithms or WGSL shaders (barraCuda)
- Network transport (songBird)
- Cryptography (bearDog)
- Persistent storage (nestGate)

### Key property
toadStool knows about specific hardware. It is the reality layer — it
understands that an RTX 3090 has 10496 CUDA cores and 24GB GDDR6X, that a
Titan V has tensor cores, that an RX 7900 XTX has different optimal workgroup
sizes. barraCuda is hardware-agnostic; toadStool is hardware-aware.

---

## Layer 3: Infrastructure Primals

### songBird — "the wire"
- Network connections, NAT traversal, mDNS service discovery
- Inter-primal communication protocols
- toadStool uses songBird for multi-node compute distribution

### bearDog — "the vault"
- Cryptographic services, entropy mixing, PKI
- Could depend on barraCuda for FHE math (NTT, pointwise modular multiplication)
- Does NOT depend on toadStool for crypto operations

### nestGate — "the shelf"
- Universal storage, artifact management, data pipelines
- Distributed cache, metadata services

---

## The Streaming Demarcation

Two kinds of streaming exist in the ecosystem:

### Math streaming (barraCuda — `staging/`)
- CPU→GPU→CPU data flow through compute stages on a single device
- `unidirectional.rs` — fire-and-forget pipeline API
- `ring_buffer.rs` — lock-free staging between CPU and GPU
- `stateful.rs` — iterative simulation (MD, SCF, PDE solvers keep state on GPU)
- `pipeline.rs` — compose GPU stages into single command submissions

### Hardware streaming (toadStool — `runtime/orchestration`)
- Route workloads across heterogeneous devices and nodes
- Multiplex across CPU + GPU + NPU substrates
- Distribute across network via songBird
- Adaptive re-routing based on load and thermal conditions

**Example**:
- barraCuda: "stream this FFT through 3 GPU compute stages" (single device)
- toadStool: "stream this workload across 2 GPUs and 1 NPU on 3 nodes" (fleet)

---

## Dependency Flow

```
Springs
  └─► barraCuda (direct: cargo dep)
  └─► toadStool (optional: for managed/distributed compute)
        ├─► barraCuda (as compute backend)
        ├─► runtime/gpu (multi-framework)
        ├─► runtime/adaptive (tuning barraCuda ops)
        ├─► runtime/orchestration (substrate routing)
        ├─► songBird (network)
        ├─► bearDog (crypto)
        └─► nestGate (storage)
```

barraCuda has ZERO required dependencies on toadStool. The `toadstool` feature
flag in barraCuda enables optional integration (e.g., `DeviceSelection` from
`toadstool-core`) but is off by default.

toadStool's `runtime/adaptive` crate tunes barraCuda operations but has ZERO
code coupling to barraCuda:
- No `barracuda::` imports or Cargo dependency on the barracuda crate
- GPU fingerprinting uses `wgpu` directly (adapter probing)
- Optimization cache stores learned configs as JSON (directory: `~/.cache/barracuda/`)
- Profiler is currently simulated; future design uses wgpu micro-benchmarks
- Integration direction: barraCuda consumes adaptive's recommendations (e.g.
  `AdaptiveExecutor::optimal_workgroup()`), not the reverse

---

## Domain Model Migration Path

Nine domain modules currently live in barraCuda. They are consumers of
barraCuda math, not part of the core math library. Over time, they migrate
to their natural Spring homes:

| Module | Current Home | Future Home | Migration Trigger |
|--------|-------------|-------------|-------------------|
| `nn` | barraCuda | neuralSpring | neuralSpring matures its own inference |
| `snn` | barraCuda | neuralSpring | spiking network research stabilizes |
| `esn_v2` | barraCuda | neuralSpring | reservoir computing moves to Spring |
| `vision` | barraCuda | neuralSpring | vision pipeline matures |
| `timeseries` | barraCuda | neuralSpring | time series ops mature |
| `pde` | barraCuda | hotSpring/groundSpring | PDE solvers validated in domain |
| `genomics` | barraCuda | wetSpring | genomics pipeline matures |
| `nautilus` | barraCuda | **stays** | barraCuda's own evolutionary optimizer |
| `surrogate` | barraCuda | **stays** | math primitive (used by `sample` core) |

Feature-gate domain modules now (`domain-models` umbrella + per-module flags).
Migrate when Springs are ready. `nautilus` and `surrogate` stay as core.

---

## Summary

| Primal | Owns | Does NOT Own |
|--------|------|-------------|
| **barraCuda** | Math algorithms, WGSL shaders, wgpu device, compute fabric, staging pipelines | Multi-framework routing, hardware fingerprinting, distribution |
| **toadStool** | Multi-framework GPU, substrate orchestration, adaptive tuning, server API, distribution | Math algorithms, WGSL shaders, network transport |
| **songBird** | Network connections, discovery, NAT traversal | Compute, math, storage |
| **bearDog** | Crypto, entropy, PKI | Compute orchestration, network |
| **nestGate** | Storage, artifacts, data pipelines | Compute, network, crypto |

---

## Infrastructure Layer Audit

The 17 GPU infrastructure modules in barraCuda were compared against toadStool's
4 runtime crates. Verdict: **no functional duplication**. The modules operate at
different layers.

| barraCuda Module | toadStool Crate | Overlap | Layer |
|------------------|-----------------|---------|-------|
| `device/` | `toadstool-core` | None | barraCuda: wgpu device API. toadStool: OS-level `/sys` hardware discovery. |
| `multi_gpu/` | `runtime/gpu` | None | barraCuda: wgpu-only GpuPool in-process. toadStool: multi-framework (CUDA/ROCm/etc) orchestration. |
| `scheduler/` | `runtime/orchestration` | None | barraCuda: per-op executor scoring. toadStool: workload-level substrate selection. |
| `workload/` | `runtime/orchestration` | None | barraCuda: data/workload characteristics. toadStool: resource and performance requirements. |
| `unified_hardware/` | `runtime/universal` | Conceptual | Both do unified compute, but at different layers: execution vs substrate discovery. |
| `dispatch/` | `runtime/orchestration` | None | barraCuda: size-based CPU/GPU micro-dispatch. toadStool: macro-orchestration. |
| `cpu/gpu/npu_executor` | runtime engines | None | barraCuda: provides executors. toadStool: routes to them. |
| `staging/` | — | None | Unique to barraCuda (CPU↔GPU data movement). |
| `pipeline/` | — | None | Unique to barraCuda (ComputeDispatch builder). |
| `compute_graph/` | — | None | Unique to barraCuda (compute DAG). |
| `session/` | — | None | Unique to barraCuda (tensor session). |
| `resource_quota/` | — | None | Unique to barraCuda (per-device quotas). |
| `auto_tensor/` | — | None | Unique to barraCuda. |
| `cpu_conv_pool/` | — | None | Unique to barraCuda (CPU fallback). |

The only conceptual overlap is `unified_hardware/` vs `runtime/universal`. Both
provide unified compute abstraction, but at different granularities:
- barraCuda `unified_hardware` = execution layer (ComputeExecutor implementations)
- toadStool `runtime/universal` = substrate layer (ComputeUnit discovery + routing)

No code sharing or refactoring is needed. The layering is clean.

---

## References

- `barraCuda/specs/BARRACUDA_SPECIFICATION.md` — barraCuda crate architecture
- `toadStool/specs/BARRACUDA_PRIMAL_BUDDING.md` — extraction phases
- `toadStool/specs/SOVEREIGN_COMPUTE_EVOLUTION.md` — toadStool roadmap
- `wateringHole/handoffs/BARRACUDA_S89_EXTRACTION_COMPLETE_MAR02_2026.md`

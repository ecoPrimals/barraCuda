# barraCuda Specification

**Version**: 0.3.3
**Date**: March 4, 2026
**Status**: Active — standalone primal, fully untangled from toadStool (S89)
**Origin**: toadStool S88 budding proposal

---

## Name

**barraCuda** — *BARrier-free Rust Abstracted Cross-platform Unified Dimensional Algebra*

More concept than exact acronym. The barracuda stands still until it strikes —
fast, silent, instant math across any silicon.

barraCuda is vendor-agnostic. It runs on any GPU via WGSL/wgpu — Vulkan, Metal,
DX12, WebGPU. One source, any backend, identical results.

---

## Purpose

barraCuda is the sovereign math engine for the ecoPrimals ecosystem. It provides:

1. **Universal GPU compute** — one WGSL source runs on any GPU vendor
2. **Scientific precision** — DF64 emulation for double-precision on consumer GPUs
3. **FHE acceleration** — the only cross-vendor FHE-on-GPU implementation
4. **Domain breadth** — lattice QCD, spectral analysis, MD, hydrology, bio, ML
5. **Validation rigor** — FHE + QCD canary suite for GPU stack correctness

## Architecture

### Crate hierarchy

```
barraCuda/
├── crates/
│   ├── barracuda/            # Umbrella crate — all math, GPU ops, compute fabric
│   │   ├── src/
│   │   │   ├── linalg/       # Dense/sparse linear algebra, eigensolvers, L-BFGS
│   │   │   ├── special/      # Special functions (erf, gamma, Bessel)
│   │   │   ├── numerical/    # Gradients, integration, ODE solvers
│   │   │   ├── spectral/     # Anderson, Lanczos, eigensolve
│   │   │   ├── stats/        # Bootstrap, regression, distributions
│   │   │   ├── sample/       # LHS, Sobol, Metropolis, sparsity
│   │   │   ├── ops/          # GPU ops (matmul, softmax, FHE NTT, bio, MD)
│   │   │   ├── tensor/       # GPU tensor type, buffer management
│   │   │   ├── shaders/      # 767 WGSL shaders
│   │   │   ├── device/       # WgpuDevice, capabilities, test pool
│   │   │   ├── staging/      # Ring buffers, unidirectional pipelines
│   │   │   ├── pipeline/     # ComputeDispatch, batched pipelines
│   │   │   ├── multi_gpu/    # GpuPool, MultiDevicePool, load balancing
│   │   │   └── ...           # + nn, snn, esn, pde, genomics, vision (feature-gated)
│   │   ├── tests/            # 29 integration test suites
│   │   └── examples/         # 4 runnable examples
│   └── barracuda-core/       # Primal lifecycle wrapper
│       ├── src/
│       │   ├── lib.rs        # BarraCudaPrimal: start/stop/health, tensor store
│       │   ├── ipc/          # JSON-RPC 2.0 server + transport
│       │   ├── rpc.rs        # tarpc service (10 endpoints, full JSON-RPC parity)
│       │   └── bin/barracuda.rs  # UniBin CLI
│       └── tests/            # IPC E2E integration tests
└── specs/                    # Architecture specs
```

### Feature flags

```toml
[features]
default = ["gpu", "domain-models"]
gpu = ["wgpu", "bytemuck", "naga"]
domain-models = ["domain-nn", "domain-snn", "domain-esn", "domain-pde",
                 "domain-genomics", "domain-vision", "domain-timeseries"]
domain-nn = []
domain-snn = []
domain-esn = []
domain-pde = []
domain-genomics = []
domain-vision = []
domain-timeseries = ["domain-esn"]
parallel = ["rayon"]
serde = ["dep:serde"]
```

The `gpu` feature is default. All ops have CPU fallbacks. The crate compiles
and passes tests without GPU hardware (llvmpipe software renderer).

Three-config check (all must pass):
- `cargo check --no-default-features` — pure math, no GPU
- `cargo check --no-default-features --features gpu` — math + GPU, no domain models
- `cargo check` — everything

### Primal interface

barraCuda exposes a dual-protocol IPC interface (JSON-RPC 2.0 primary, tarpc
binary secondary). Both protocols serve the same 10 endpoints with full
parameter parity:

```
barracuda.device.list           → {devices: [{name, vendor, device_type, backend, driver}]}
barracuda.device.probe          → {available, max_buffer_size, max_storage_buffers, ...}
barracuda.health.check          → {name, version, status}
barracuda.tolerances.get        → {name} → {name, abs_tol, rel_tol}
barracuda.validate.gpu_stack    → {gpu_available, status, message}
barracuda.compute.dispatch      → {op, shape?, tensor_id?} → {status, tensor_id?, data?}
barracuda.tensor.create         → {shape, dtype?, data?} → {tensor_id, shape, elements, dtype}
barracuda.tensor.matmul         → {lhs_id, rhs_id} → {result_id, shape}
barracuda.fhe.ntt               → {modulus, degree, root_of_unity, coefficients} → {result}
barracuda.fhe.pointwise_mul     → {modulus, degree, a, b} → {result}
```

Error codes follow JSON-RPC 2.0: -32700 (parse), -32600 (invalid request),
-32601 (method not found), -32602 (invalid params), -32603 (internal).

### Shader compilation pipeline

```
WGSL source
  → op_preamble (abstract ops for F16/F32/F64/DF64)
  → naga parse → naga IR
  → df64_rewrite (infix → bridge functions, when DF64)
  → naga optimize (FMA fusion, DCE)
  → SPIR-V / Metal / DXIL / WGSL (per backend)
  → GPU driver compile
```

Sovereign compiler path. No external SDK needed for correctness.

---

## Consumers

| Consumer | Relationship | What they use |
|----------|-------------|---------------|
| **toadStool** | Orchestration primal | Device management, IPC bridge, workload routing |
| **hotSpring** | Physics validation | Lattice QCD, HMC, CG solver, MD, spectral |
| **groundSpring** | Earth science validation | Linalg, eigensolvers, optimization, hydrology |
| **neuralSpring** | ML/bio validation | Matmul, attention, ESN, HMM, activation functions |
| **wetSpring** | Marine bio validation | Statistics, diversity, genomics, spectral, FHE |
| **airSpring** | Agriculture validation | Hydrology, seasonal pipeline, Van Genuchten, kriging |
| **bearDog** | Crypto composition | FHE GPU compute (via IPC, not crate dependency) |

---

## Extraction History

barraCuda lives at `ecoPrimals/barraCuda/` as a fully standalone primal.
Extraction from toadStool (S88-S89) is **complete**. Zero cross-dependencies
remain.

### Budding Phases — ALL COMPLETE

- **Phase 0** (Decouple): `toadstool-core` dependency removed entirely
- **Phase 1** (Boundary): IPC contract defined, GPU validation binary built
- **Phase 2** (Extraction): Code in standalone repo with independent CI
- **Phase 3** (Consumer rewiring): hotSpring validated (716/716 tests, zero code changes)
- **Phase 4** (Untangle): All feature flags, integration code, and NPU coupling removed
- Springs depend on barracuda directly (cargo path dep)
- toadStool depends on barracuda as external crate (not embedded)

---

## References

- `ecoPrimals/wateringHole/handoffs/TOADSTOOL_S88_BARRACUDA_PRIMAL_BUDDING_PROPOSAL_MAR02_2026.md`
- `ecoPrimals/phase1/toadStool/specs/BARRACUDA_PRIMAL_BUDDING.md`
- `ecoPrimals/phase1/toadStool/specs/SOVEREIGN_COMPUTE_EVOLUTION.md`

# barraCuda Specification

**Version**: 0.1.0
**Date**: March 2, 2026
**Status**: RFC — pending extraction from ToadStool
**Origin**: ToadStool S88 budding proposal

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
barracuda (umbrella)
├── barracuda-core       # WgpuDevice, DevicePool, shader compilation, error types
├── barracuda-tensor     # Tensor, buffer management, memory layout
├── barracuda-ops        # ComputeDispatch, 766+ WGSL shaders, op implementations
├── barracuda-fhe        # FheNtt, FheIntt, FhePointwiseMul, modular arithmetic
├── barracuda-linalg     # Dense, sparse (CSR SpMV), eigensolvers, L-BFGS
├── barracuda-spectral   # Anderson, Lanczos, spectral analysis, localization
└── barracuda-esn        # MultiHeadEsn, ESN, reservoir computing, Nautilus bridge
```

### Feature flags

```toml
[features]
default = ["gpu"]
gpu = ["wgpu", "bytemuck", "naga"]
parallel = ["rayon"]
serde = ["dep:serde"]
benchmarks = []
```

The `gpu` feature is default. All ops have CPU fallbacks. The crate compiles
and passes tests without GPU hardware (llvmpipe software renderer).

### Primal interface

barraCuda exposes IPC endpoints via JSON-RPC 2.0:

```
barracuda.device.list           → [{adapter, vendor, features, limits}]
barracuda.device.probe          → {f64_support, max_buffers, df64_available}
barracuda.compute.dispatch      → {shader, inputs, params} → {outputs}
barracuda.fhe.ntt               → {polynomial, modulus, degree} → {transformed}
barracuda.fhe.pointwise_mul     → {a, b, modulus} → {product}
barracuda.validate.gpu_stack    → {fhe_pass, qcd_pass, df64_pass, iterations}
barracuda.tensor.create         → {shape, dtype, data} → {tensor_id}
barracuda.tensor.matmul         → {lhs_id, rhs_id} → {result_id}
barracuda.tolerances.get        → {name} → {abs_tol, rel_tol}
```

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
| **ToadStool** | Orchestration primal | Device management, IPC bridge, workload routing |
| **hotSpring** | Physics validation | Lattice QCD, HMC, CG solver, MD, spectral |
| **groundSpring** | Earth science validation | Linalg, eigensolvers, optimization, hydrology |
| **neuralSpring** | ML/bio validation | Matmul, attention, ESN, HMM, activation functions |
| **wetSpring** | Marine bio validation | Statistics, diversity, genomics, spectral, FHE |
| **airSpring** | Agriculture validation | Hydrology, seasonal pipeline, Van Genuchten, kriging |
| **BearDog** | Crypto composition | FHE GPU compute (via IPC, not crate dependency) |

---

## Extraction Plan

barraCuda currently lives at `ecoPrimals/phase1/toadStool/crates/barracuda/`.
Extraction follows the phases defined in `toadStool/specs/BARRACUDA_PRIMAL_BUDDING.md`.

### Phase 0 — Decouple (current)
- Feature-gate toadstool-core dependency
- Standalone compilation CI
- API surface audit

### Phase 1 — Boundary hardening
- Extract barracuda-types if needed
- Define IPC contract
- Build GPU validation binary

### Phase 2 — Repo extraction
- Move code to this repository
- SemVer 1.0.0
- Independent CI

### Phase 3 — Consumer rewiring
- Springs depend on barracuda directly
- ToadStool depends on barracuda as versioned dep
- Path deps for development, published for releases

---

## References

- `ecoPrimals/wateringHole/handoffs/TOADSTOOL_S88_BARRACUDA_PRIMAL_BUDDING_PROPOSAL_MAR02_2026.md`
- `ecoPrimals/phase1/toadStool/specs/BARRACUDA_PRIMAL_BUDDING.md`
- `ecoPrimals/phase1/toadStool/specs/SOVEREIGN_COMPUTE_EVOLUTION.md`

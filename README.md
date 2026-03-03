# barraCuda

**Version**: 0.2.0
**Status**: Standalone — extracted from ToadStool (S89), all tests passing
**License**: AGPL-3.0-or-later

---

## The Name

**barraCuda** — *BARrier-free Rust Abstracted Cross-platform Unified Dimensional Algebra*

More concept than exact acronym. The barracuda stands still until it strikes —
fast, silent, instant. That's the compute model: sovereign math that waits on
any GPU, then executes instantly across any vendor's silicon.

---

## What is barraCuda?

barraCuda is the **sovereign math engine** for the ecoPrimals ecosystem. It
provides GPU-accelerated scientific computing across any vendor's hardware
using WGSL shaders compiled through wgpu. One source, any GPU, identical
results.

### Key capabilities

- **766+ WGSL shaders** spanning scientific compute domains
- **DF64 emulation** — double-precision arithmetic on GPUs without native f64
- **FHE on GPU** — Number Theoretic Transform, INTT, pointwise modular
  multiplication via 32-bit emulation of 64-bit modular arithmetic. The only
  cross-vendor FHE GPU implementation in existence.
- **Lattice QCD** — SU(3) gauge theory, staggered Dirac, CG solver, HMC
- **Spectral analysis** — Anderson localization, Lanczos eigensolver
- **Molecular dynamics** — Yukawa, VV integrator, cell-list neighbor search
- **Linear algebra** — dense, sparse (CSR SpMV), eigensolvers, L-BFGS
- **Statistics** — bootstrap, jackknife, diversity indices, hydrology
- **Bioinformatics** — Smith-Waterman, HMM, phylogenetics, genomic ops
- **ML ops** — matmul, softmax, attention, ESN reservoir computing
- **Sovereign shader compilation** — naga IR optimizer, SPIR-V passthrough

### Design principles

1. **Math is universal, precision is silicon** — one WGSL source, any precision
2. **Vendor-agnostic** — same binary, identical results on any GPU
3. **Sovereign** — zero external SDK dependency for correctness or performance
4. **AGPL-3.0** — free as in freedom

---

## Structure

```
barraCuda/
├── Cargo.toml                   # Workspace manifest
├── README.md                    # You are here
├── CONVENTIONS.md               # Coding standards
├── CHANGELOG.md                 # SemVer changelog
├── crates/
│   ├── barracuda-core/          # Primal lifecycle + device management
│   ├── barracuda-tensor/        # Tensor operations, buffer management
│   ├── barracuda-ops/           # GPU ops, shaders, ComputeDispatch
│   ├── barracuda-fhe/           # FHE NTT, INTT, pointwise mod-mul
│   ├── barracuda-linalg/        # Dense + sparse linear algebra
│   ├── barracuda-spectral/      # Anderson, Lanczos, spectral analysis
│   ├── barracuda-esn/           # Multi-head ESN, reservoir computing
│   └── barracuda/               # Umbrella crate (re-exports all)
├── specs/
│   ├── BARRACUDA_SPECIFICATION.md
│   ├── SOVEREIGN_COMPUTE.md
│   └── BUDDING_PLAN.md
└── tests/
    ├── gpu_validation.rs        # FHE + QCD canary suite
    ├── cross_vendor.rs          # Multi-adapter parity
    └── integration.rs           # Full pipeline tests
```

**Note**: This scaffold defines the target structure. The actual code lives in
`ecoPrimals/phase1/toadStool/crates/barracuda/` until extraction is complete.

---

## GPU Validation Canary

barraCuda uses FHE + lattice QCD as a mathematically rigorous GPU stack
validation suite:

| Test | Pass criteria | What it validates |
|------|--------------|-------------------|
| FHE NTT round-trip | Bit-perfect: INTT(NTT(p)) == p | u32/u64 emulation, modular arithmetic |
| FHE polynomial mul | Matches symbolic reference | Buffer pipeline, dispatch, compilation |
| SU(3) unitarity | < DF64 epsilon | DF64 matrix multiply, complex arithmetic |
| Plaquette expectation | Within statistical error | Full pipeline under sustained load |
| CG convergence | Within reference ± 2 iterations | Iterative solver precision |

Any consumer runs `barracuda validate-gpu` to verify their GPU is trustworthy
for scientific compute.

---

## Relationship to ecoPrimals

barraCuda is a **NUCLEUS foundation primal**. It composes with:

- **BearDog** (crypto) — FHE key generation + barraCuda GPU compute = sovereign
  encrypted computation
- **ToadStool** (orchestration) — primal lifecycle, IPC routing, biomeOS
- **Springs** (validation) — 5 domain-specific projects that consume and validate
  barraCuda primitives across physics, biology, agriculture, ML, and earth science

barraCuda knows only itself. It discovers other primals at runtime via
capability-based IPC (JSON-RPC 2.0).

---

## Quick Start

```bash
cargo build
cargo test
```

---

**Created with SourDough. Budding from ToadStool S88.**

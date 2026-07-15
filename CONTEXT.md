# Context — barraCuda

## What This Is

barraCuda is a pure Rust GPU math engine that executes scientific computing
workloads across any vendor's hardware using WGSL shaders compiled through
wgpu. It is part of the ecoPrimals sovereign computing ecosystem — a
collection of self-contained binaries that coordinate via JSON-RPC 2.0 over
Unix sockets, with zero compile-time coupling between components.

## Role in the Ecosystem

barraCuda owns all mathematical computation. Springs (domain validation
projects) and other primals consume barraCuda as a Cargo dependency for GPU
math, or coordinate via JSON-RPC for compute dispatch. coralReef compiles
shaders to native GPU binaries; toadStool routes workloads to hardware.
barraCuda has zero dependencies on any sibling primal — lifecycle and health
traits are fully internalized.

## Technical Facts

- **Language:** 100% Rust, zero C dependencies in application code
- **Architecture:** 4-crate workspace (barracuda, barracuda-core, barracuda-spirv, barracuda-naga-exec)
- **Communication:** JSON-RPC 2.0 + tarpc over Unix socket and TCP
- **License:** AGPL-3.0-or-later (scyBorg provenance trio)
- **Tests:** 4,632 passing (3,924 barracuda + 708 barracuda-core)
- **Coverage:** 80.54% line on llvmpipe (80% CI gate, 90% target with GPU hardware)
- **MSRV:** 1.92
- **Crate count:** 4 workspace crates
- **Shaders:** 826 WGSL compute shaders with SPDX license headers
- **Rust files:** 1,169 source files, 42 integration test files
- **Unsafe code:** Zero — `#![forbid(unsafe_code)]` in barracuda and barracuda-core
- **Clippy:** Pedantic + nursery, zero warnings, `-D warnings` enforced

## Key Capabilities

- **826 WGSL shaders** spanning: linear algebra, statistics, spectral analysis,
  molecular dynamics, lattice QCD, FHE (NTT/INTT), pharmacometrics,
  bioinformatics, ML ops, health/biosignal, procedural generation
- **DF64 emulation** — double-precision arithmetic on GPUs without native f64
- **15-tier precision continuum** — Binary through DF128 with per-tier tolerances
- **NagaExecutor** — CPU interpreter for naga IR (shader-first execution without GPU)
- **Sovereign compiler** — naga IR optimizer with FMA fusion and dead expr elimination

## IPC Method Surface (98 methods)

| Domain | Methods |
|--------|---------|
| `health.*` | `liveness`, `readiness`, `check`, `version` |
| `auth.*` | `check`, `mode`, `peer_info` |
| `identity.*` | `get` |
| `primal.*` | `info`, `capabilities`, `announce` |
| `capabilities.*` | `list` |
| `device.*` | `list`, `probe` |
| `tolerances.*` | `get` |
| `validate.*` | `gpu_stack` |
| `precision.*` | `route` |
| `compute.*` | `dispatch` |
| `math.*` | `sigmoid`, `log2` |
| `activation.*` | `fitts`, `hick`, `softmax`, `gelu` |
| `stats.*` | `mean`, `std_dev`, `variance`, `correlation`, `pearson`, `spearman`, `covariance`, `weighted_mean`, `chi_squared`, `anova_oneway`, `shannon`, `entropy`, `fit_linear`, `fit_quadratic`, `fit_exponential`, `fit_logarithmic`, `empirical_spectral_density`, `simpson`, `bray_curtis`, `hill`, `rarefaction_curve`, `gamma_fit`, `gamma_cdf` |
| `signal.*` | `detect_peaks`, `bandpass`, `derivative` |
| `linalg.*` | `solve`, `eigenvalues`, `svd`, `qr`, `graph_laplacian` |
| `ode.*` | `step` |
| `graph.*` | `belief_propagation` |
| `spectral.*` | `fft`, `power_spectrum`, `stft` |
| `ml.*` | `mlp_forward`, `mlp_train`, `attention`, `esn_predict` |
| `nautilus.*` | `create`, `observe`, `train`, `predict`, `export`, `import` |
| `noise.*` | `perlin2d`, `perlin3d` |
| `rng.*` | `uniform` |
| `tensor.*` | `create`, `matmul`, `matmul_inline`, `add`, `scale`, `clamp`, `reduce`, `sigmoid`, `batch.submit` |
| `fhe.*` | `ntt`, `pointwise_mul` |
| `btsp.*` | `negotiate`, `capabilities` |

98 methods following wateringHole `{domain}.{operation}` Semantic Method Naming Standard. Wire Standard L2 compliant. Neural API announce on startup. BTSP Phase 3 encryption.

## Deployment Constraints

**musl-static (ecoBin) builds cannot use GPU directly.** wgpu requires `dlopen`
to load Vulkan/Mesa driver shared objects. musl-static binaries are fully
statically linked — no `.so` loading is possible. This is a fundamental
constraint of static linking (BC-06).

ecoBin binaries compute via two fallback paths:

1. **cpu-shader** (default-on, BC-08) — `barracuda-naga-exec` interprets WGSL
   on CPU. No GPU required. Performance is lower but correctness is identical.
2. **Sovereign IPC** (BC-07) — `Auto::new()` discovers a coralReef+toadStool
   peer via capability-based IPC and delegates GPU work over JSON-RPC. The peer
   runs on a glibc host with GPU access. This gives ecoBin binaries full GPU
   throughput without local `dlopen`.

## What This Does NOT Do

- **Does not compile shaders to native GPU binaries** — that is coralReef's domain
- **Does not manage hardware or route workloads** — that is toadStool's domain
- **Does not provide networking or distribution** — that is songBird's domain
- **Does not orchestrate multi-primal workflows** — primals coordinate via IPC at runtime
- **Does not depend on any sibling primal** — fully standalone, discovers capabilities at runtime

## Related Repositories

- [wateringHole](https://github.com/ecoPrimals/wateringHole) — ecosystem standards and primal registry
- [coralReef](https://github.com/ecoPrimals/coralReef) — sovereign GPU compiler (WGSL → native binary)
- [toadStool](https://github.com/ecoPrimals/toadStool) — hardware discovery and compute orchestration
- [sourDough](https://github.com/ecoPrimals/sourDough) — scaffold reference (no runtime dependency)
- Springs (syntheticChemistry org) — domain validation consumers of barraCuda GPU primitives

## Design Philosophy

These binaries are built using AI-assisted constrained evolution. Rust's
compiler constraints (ownership, lifetimes, type system) reshape the fitness
landscape and drive specialization. Primals are self-contained — they know
what they can do, never what others can do. Complexity emerges from runtime
coordination, not compile-time coupling. Math is universal; the execution
substrate (GPU, CPU, NPU) is a hardware implementation detail.

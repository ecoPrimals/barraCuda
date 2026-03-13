# barraCuda

**Version**: 0.3.5
**Status**: Standalone primal — zero cross-dependencies, fully concurrent, all quality gates passing
**License**: AGPL-3.0-only
**MSRV**: 1.87

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

- **806 WGSL shaders** spanning scientific compute domains (all with SPDX license headers)
- **1,088 Rust source files**, 43 integration test files, 4,011+ tests passing
- **DF64 emulation** — double-precision arithmetic on GPUs without native f64
- **FHE on GPU** — Number Theoretic Transform, INTT, pointwise modular
  multiplication via 32-bit emulation of 64-bit modular arithmetic. The only
  cross-vendor FHE GPU implementation in existence.
- **Lattice QCD** — SU(3) gauge theory, staggered Dirac, CG solver, HMC
- **Spectral analysis** — Anderson localization, Lanczos eigensolver
- **Molecular dynamics** — Yukawa, PPPM, VV integrator, cell-list neighbor search
- **Linear algebra** — dense, sparse (CSR SpMV, CG, BiCGStab), eigensolvers, L-BFGS
- **Statistics** — bootstrap, jackknife, diversity indices, hydrology
- **Pharmacometrics** — FOCE gradients, VPC Monte Carlo, population PK, dose-response
- **Health** — Michaelis-Menten PK (GPU batch), SCFA production, beat classification, EDA stress detection
- **Bioinformatics** — Smith-Waterman, HMM, phylogenetics, bipartition encoding, genomic ops
- **ML ops** — matmul, softmax, attention, ESN reservoir computing
- **Sovereign shader compilation** — naga 28 IR optimizer, SPIR-V passthrough
- **JSON-RPC 2.0 + tarpc** — dual-protocol IPC with namespace-derived method names for capability-based discovery
- **UniBin CLI** — single `barracuda` binary with `server`, `service`, `doctor`, `validate`, `version`

### Design principles

1. **Math is universal, precision is silicon** — one WGSL source, any precision
2. **Vendor-agnostic** — same binary, identical results on any GPU
3. **Sovereign** — zero external SDK dependency for correctness or performance
4. **Pure Rust** — `#![forbid(unsafe_code)]` in both crates, zero `unsafe` blocks, zero external C dependencies, zero dependencies on any other primal (lifecycle and health traits internalized from sourDough scaffold)
5. **Fully concurrent** — `GuardedDeviceHandle` + atomic encoder barrier prevents wgpu-core races without lock contention; wgpu 28 `Device`/`Queue` are `Clone` — zero `Arc` overhead for handle sharing; all tests pass at 16 threads on llvmpipe
6. **AGPL-3.0** — free as in freedom

---

## Architecture

barraCuda is a library crate (`barracuda`) wrapped by a primal lifecycle crate
(`barracuda-core`) that exposes IPC, tarpc, and the UniBin CLI. Springs and
other consumers `cargo add barracuda`. toadStool orchestrates above it;
barraCuda owns the math.

```
Your Code / Springs
    |
    v
barracuda (umbrella crate)
    |-- Pure Math: linalg, special, numerical, spectral, stats, sample, activations, rng, tolerances
    |-- GPU Math: ops (bio, pharma, fhe, qcd, ...), tensor, shaders, interpolate, optimize
    |-- Compute Fabric: device, staging, pipeline, dispatch, multi_gpu
    |-- Domain Models: nn, snn, esn, pde, genomics (feature-gated)
    |
    v
barracuda-core (primal lifecycle)
    |-- IPC: JSON-RPC 2.0 (text) + tarpc (binary)
    |-- UniBin CLI: server, doctor, validate, version
    |-- lifecycle + health: PrimalLifecycle, PrimalHealth (owned)
    |
    v
wgpu 28 (WebGPU)
    |
    +-- Vulkan (NVIDIA, AMD, Intel)
    +-- Metal (Apple)
    +-- DX12 (Windows)
    +-- Software rasterizer (CPU fallback / CI)
```

See `specs/ARCHITECTURE_DEMARCATION.md` for the full barraCuda / toadStool
boundary definition.

---

## Structure

```
barraCuda/
├── Cargo.toml                       # Workspace manifest
├── deny.toml                        # cargo-deny (license + advisory audit)
├── rustfmt.toml                     # Formatting config
├── README.md                        # You are here
├── CONTRIBUTING.md                  # How to contribute
├── CONVENTIONS.md                   # Coding standards
├── CHANGELOG.md                     # SemVer changelog
├── BREAKING_CHANGES.md              # Migration notes
├── STATUS.md                        # Quality scorecard (A+ grade)
├── WHATS_NEXT.md                    # P1–P4 roadmap
├── START_HERE.md                    # Developer quick start
├── PURE_RUST_EVOLUTION.md           # Sovereign compute evolution log
├── SPRING_ABSORPTION.md             # Cross-spring absorption tracker
├── LICENSE                          # AGPL-3.0-only
├── .github/workflows/ci.yml        # CI: fmt, clippy, deny, doc, test, coverage
├── crates/
│   ├── barracuda-core/              # Primal lifecycle wrapper
│   │   ├── src/lib.rs               # BarraCudaPrimal: start/stop/health
│   │   ├── src/ipc/                 # JSON-RPC 2.0 server + transport
│   │   ├── src/rpc.rs               # tarpc service definition (10 endpoints, parity with JSON-RPC)
│   │   └── src/bin/barracuda.rs     # UniBin CLI
│   └── barracuda/                   # Umbrella crate — all math + GPU
│       ├── src/
│       │   ├── lib.rs               # Module declarations + prelude
│       │   ├── error.rs             # BarracudaError (19 variants incl. DeviceLost)
│       │   ├── linalg/              # Dense/sparse linear algebra
│       │   ├── special/             # Special functions (erf, gamma, Bessel)
│       │   ├── numerical/           # Gradients, integration, ODE
│       │   ├── spectral/            # Anderson, Lanczos, eigensolve
│       │   ├── stats/               # Bootstrap, regression, distributions
│       │   ├── sample/              # LHS, Sobol, Metropolis, sparsity
│       │   ├── ops/                 # GPU ops (matmul, softmax, FHE, bio)
│       │   ├── tensor/              # GPU tensor type
│       │   ├── shaders/             # 806 WGSL shaders (see shaders/README.md)
│       │   ├── device/              # GpuBackend trait, WgpuDevice, CoralReefDevice, concurrency
│       │   ├── staging/             # Ring buffers, unidirectional pipelines
│       │   ├── pipeline/            # ComputeDispatch, batched pipelines
│       │   ├── dispatch/            # Size-based CPU/GPU routing
│       │   ├── multi_gpu/           # GpuPool, MultiDevicePool, load balancing
│       │   ├── unified_hardware/    # Unified CPU/GPU/NPU abstraction
│       │   └── ...                  # + nn, snn, esn, pde, genomics, vision
│       ├── examples/                # Runnable examples
│       ├── tests/                   # 42 integration test files
│       └── src/bin/                 # validate_gpu, bench_*
└── specs/
    ├── BARRACUDA_SPECIFICATION.md       # Crate architecture + IPC contract
    ├── PRECISION_TIERS_SPECIFICATION.md # 15-tier precision ladder (Binary→DF128)
    ├── REMAINING_WORK.md                # P1-P4 open work items
    └── ARCHITECTURE_DEMARCATION.md      # barraCuda vs toadStool boundaries
```

---

## Concurrency Model

barraCuda's GPU access uses a three-layer concurrency model that prevents
wgpu-core internal races without lock contention:

1. **`active_encoders: AtomicU32`** — lock-free counter incremented during
   any wgpu-core activity (buffer creation, shader compilation, command
   encoding). Multiple threads increment simultaneously with zero contention.
2. **`gpu_lock: Mutex<()>`** — serializes `queue.submit()` and `device.poll()`
   against each other. Before poll, a bounded yield loop waits for active
   encoders to reach zero (encoding is CPU-speed microsecond work).
3. **`dispatch_semaphore`** — hardware-aware cap (2 for CPU/llvmpipe, 8 for
   discrete GPU) preventing driver overload.

`GuardedEncoder` is an RAII wrapper that auto-decrements the active encoder
count on finish or drop, making the barrier leak-proof.

---

## Quality Gates

```bash
cargo fmt --all -- --check              # formatting
cargo clippy --workspace --all-targets --all-features -- -D warnings  # lints (pedantic, all clean)
cargo deny check                        # license + advisory audit
RUSTDOCFLAGS="-D warnings" cargo doc --workspace --no-deps  # documentation (zero warnings)
cargo build --workspace                 # compilation
cargo test --workspace --lib            # 4,011+ test functions
cargo llvm-cov --workspace --lib        # 90%+ line coverage target (70% on llvmpipe; GPU hardware needed for 90%)
```

All gates are enforced in `.github/workflows/ci.yml`.

---

## IPC Protocol

barraCuda exposes a dual-protocol IPC interface per wateringHole standards:

**JSON-RPC 2.0** (primary, text, newline-delimited TCP/Unix socket):

| Method | Description |
|--------|-------------|
| `barracuda.device.list` | List available compute devices |
| `barracuda.device.probe` | Probe device capabilities and limits |
| `barracuda.health.check` | Health check (name, version, status) |
| `barracuda.tolerances.get` | Numerical tolerances for a named operation |
| `barracuda.validate.gpu_stack` | GPU validation suite |
| `barracuda.compute.dispatch` | Dispatch a compute shader |
| `barracuda.tensor.create` | Create a tensor on device |
| `barracuda.tensor.matmul` | Matrix multiply two tensors |
| `barracuda.fhe.ntt` | FHE Number Theoretic Transform |
| `barracuda.fhe.pointwise_mul` | FHE pointwise polynomial multiplication |

**tarpc** (optional, binary, high-throughput primal-to-primal):

Same 10 endpoints with strongly-typed Rust signatures and full parameter
parity with the JSON-RPC handlers. Enabled via
`barracuda server --tarpc-bind 127.0.0.1:9001`.

---

## UniBin CLI

```bash
# Start IPC server (JSON-RPC on TCP)
barracuda server --bind 127.0.0.1:9000

# Start with tarpc alongside JSON-RPC
barracuda server --bind 127.0.0.1:9000 --tarpc-bind 127.0.0.1:9001

# Start with Unix socket
barracuda server --unix /tmp/barracuda.sock

# Start as systemd/init service (genomeBin mode)
barracuda service

# Health check and device diagnostics
barracuda doctor

# GPU validation suite
barracuda validate
barracuda validate --extended

# Version info
barracuda version
```

---

## Feature Flags

| Feature | Default | Description |
|---------|---------|-------------|
| `gpu` | Yes | GPU compute via wgpu/WGSL. |
| `domain-models` | Yes | All domain modules (nn, snn, esn, pde, genomics, vision, timeseries). |
| `domain-nn` | via umbrella | Neural network training API. |
| `domain-snn` | via umbrella | Spiking neural networks. |
| `domain-esn` | via umbrella | Echo state networks (reservoir computing). |
| `domain-pde` | via umbrella | PDE solvers (Richards, Crank-Nicolson). |
| `domain-genomics` | via umbrella | Bioinformatics and genomics API. |
| `domain-vision` | via umbrella | Computer vision pipelines. |
| `domain-timeseries` | via umbrella | Time series analysis (implies `domain-esn`). |
| `sovereign-dispatch` | No | Sovereign GPU dispatch via IPC to coralReef + toadStool (bypasses wgpu/Vulkan). |
| `serde` | No | Serde derive support. |
| `parallel` | No | Rayon parallelism hints. |

### Common dependency configurations

```toml
# Full (default) — everything
barracuda = { path = "../barraCuda/crates/barracuda" }

# Math + GPU only — no domain models (fastest compile)
barracuda = { path = "../barraCuda/crates/barracuda", default-features = false, features = ["gpu"] }

# Pure CPU math — no GPU at all (sub-2s compile)
barracuda = { path = "../barraCuda/crates/barracuda", default-features = false }
```

---

## Development Setup

### Prerequisites

- **Rust 1.87+** (`rustup update stable`)
- **GPU drivers** (Vulkan-capable: NVIDIA 525+, Mesa 23+, or Apple Metal)
- **llvmpipe** (optional, for headless CI — `sudo apt install mesa-vulkan-drivers`)
- **cargo-deny** (`cargo install cargo-deny`)
- **cargo-llvm-cov** (`cargo install cargo-llvm-cov`, for coverage)

### Sibling repositories

```
ecoPrimals/
├── barraCuda/          # This repo
├── coralReef/          # Shader compiler (WGSL/SPIR-V → native GPU binary)
├── sourDough/          # Scaffold reference (no runtime dependency)
├── toadStool/          # Orchestration / hardware discovery
├── wateringHole/       # Ecosystem standards and genomeBin manifest
└── ...Springs          # Consumers
```

### Build and test

```bash
cargo build --workspace
cargo test --workspace --lib
cargo run -p barracuda-core --bin barracuda -- doctor
```

---

## Relationship to ecoPrimals

barraCuda is a **NUCLEUS foundation primal** registered in `wateringHole/genomeBin/manifest.toml`.

- **toadStool** (orchestration) — routes compute to the best hardware. barraCuda
  is the execution layer; toadStool is the orchestration layer.
- **bearDog** (crypto) — FHE key generation + barraCuda GPU compute = sovereign
  encrypted computation.
- **songBird** (network) — toadStool uses songBird for multi-node distribution.
  barraCuda does not depend on songBird directly.
- **Springs** (validation) — domain-specific projects that consume barraCuda.

### Dependency direction

```
Springs ──> barraCuda (direct cargo dep)
toadStool ──> barraCuda (as compute backend)
bearDog ··> barraCuda (for FHE math)
barraCuda ──> (standalone — lifecycle/health internalized from sourDough scaffold)
```

barraCuda has ZERO dependencies on toadStool, songBird, bearDog, nestGate, or sourDough.
Lifecycle and health traits are modeled on the ecoPrimals pattern but fully owned by barraCuda.

---

## Specs

| Document | Purpose |
|----------|---------|
| `specs/BARRACUDA_SPECIFICATION.md` | Crate architecture, IPC contract, shader pipeline |
| `specs/PRECISION_TIERS_SPECIFICATION.md` | Full 15-tier precision ladder (Binary to DF128) |
| `specs/ARCHITECTURE_DEMARCATION.md` | barraCuda vs toadStool boundary definition |
| `specs/REMAINING_WORK.md` | P1-P4 open work items |
| `SOVEREIGN_PIPELINE_TRACKER.md` | Sovereign pipeline tracker (P0 blocker, libc evolution, cross-compilation) |
| `crates/barracuda/src/shaders/README.md` | Shader organization |

---

**Created with sourDough. Budded from toadStool S88-S89. Evolved to standalone primal.**

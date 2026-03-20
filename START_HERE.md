# Start Here — barraCuda Developer Guide

## What is this?

barraCuda is the sovereign math engine for the ecoPrimals ecosystem. It provides
GPU-accelerated scientific computing across any vendor's hardware using WGSL
shaders compiled through wgpu. One source, any GPU, identical results.

## Prerequisites

- **Rust 1.87+** (check with `rustc --version`)
- **GPU driver**: Vulkan-capable (NVIDIA, AMD, Intel) or software rasterizer (llvmpipe)
- **OS**: Linux, macOS, or Windows

## Quick Start

```bash
# Build everything
cargo build --workspace --all-features

# Run all tests
cargo test --workspace --all-features

# Run the doctor (device diagnostics)
cargo run --bin barracuda -- doctor

# Start the IPC server
cargo run --bin barracuda -- server
```

## Repository Layout

```
crates/
  barracuda/           Core compute library
    src/
      device/          GpuBackend trait, WgpuDevice, CoralReefDevice (IPC dispatch), driver profiles
      shaders/         806 WGSL shaders + sovereign compiler
        math/          DF64 core, transcendentals
        sovereign/     Naga-based compiler (FMA fusion, dead expr, SPIR-V emit)
        precision/     3-tier model: F32/F64/Df64 (aligned with coralReef Fp64Strategy)
      tensor/          Tensor types, GpuView persistent buffers
      ops/             Science operations (FHE, statistics, physics, bio)
      pipeline/        GPU dispatch pipeline, workgroup sizing
      error.rs         Error types

  barracuda-core/      Primal lifecycle, IPC, CLI
    src/
      ipc/             JSON-RPC 2.0 + tarpc transport
      rpc.rs           tarpc service definitions
      bin/barracuda.rs UniBin entry point

specs/                 Formal specifications
```

## Key Documents

| Document | Purpose |
|----------|---------|
| `README.md` | Project overview, architecture, usage examples |
| `CHANGELOG.md` | Detailed change history |
| `STATUS.md` | Current grade and capability status |
| `WHATS_NEXT.md` | Prioritized work items |
| `SOVEREIGN_PIPELINE_TRACKER.md` | Sovereign pipeline P0 blocker, libc evolution, cross-primal deps |
| `specs/BARRACUDA_SPECIFICATION.md` | Formal primal specification |

## Architecture

The precision pipeline is the core abstraction. Math is authored in f64 (the
canonical precision), and barraCuda handles the 3-tier hardware mapping:

```
f64 source (the "true math")
  → Precision selected by Fp64Strategy (aligned with coralReef)
    → F32:  downcast types → standard compilation
    → F64:  driver patching → ILP optimization → sovereign compiler → SPIR-V
    → Df64: naga-guided infix rewrite → DF64 library injection → compilation
```

The sovereign compiler pipeline:

```
WGSL → naga parse → FMA fusion → dead expr elimination → validate → SPIR-V emit
                                                          ↓
                                               ValidatedSpirv (type-safe)
                                                          ↓
                                          wgpu SPIRV_SHADER_PASSTHROUGH
```

## Running Specific Test Suites

```bash
# DF64 rewriter tests
cargo test -p barracuda --lib -- df64_rewrite

# Sovereign compiler tests
cargo test -p barracuda --lib -- sovereign

# IPC transport tests
cargo test -p barracuda-core -- ipc
```

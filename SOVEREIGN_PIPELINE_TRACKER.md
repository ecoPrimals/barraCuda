# Sovereign Pipeline Tracker

**Date**: March 29, 2026
**Type**: Actionable tracker (updated as work progresses)
**Scope**: All remaining work for the pure Rust sovereign GPU pipeline

---

## Pipeline Status

```
Layer 1  barraCuda    ██████████  COMPLETE       Zero unsafe, zero C deps, PrecisionBrain-coralReef routing, SiliconProfile tier routing
Layer 2  coralReef    █████████░  Phase 10 I70   f64 transcendental polyfills, structured capabilities, DRM E2E
Layer 3  Standalone   ██░░░░░░░░  Planned        Standalone coral-reef crate, multi-arch ISA
Layer 4  Sovereign HW █████████░  S163 complete  toadStool infra done; VFIO hw validation 6/7
```

### Dispatch Chain (sovereign primary — IPC-first)

```
barraCuda (WGSL math)
  → [JSON-RPC] coralReef (WGSL → native SASS/GFX binary)
    → [JSON-RPC] toadStool (dispatch binary → VFIO/DRM → GPU)
      → GPU hardware (any PCIe GPU — NVIDIA, AMD, Intel)
```

All inter-primal communication is JSON-RPC 2.0 IPC at runtime.
No compile-time coupling between primals.

### Dispatch Chain (wgpu fallback — development / non-VFIO)

```
barraCuda (WGSL math)
  → WgpuDevice (default, no feature flag)
    → wgpu → Vulkan/Metal/DX12 → Mesa/NVK → kernel driver → GPU
```

---

## P0 — SovereignDevice Backend (the bridge)

barraCuda dispatches GPU work through two backends:

1. **`SovereignDevice` (VFIO primary)** — dispatches via JSON-RPC IPC to shader.compile primal + compute.dispatch primal,
   bypassing the entire Vulkan/Mesa/kernel driver stack.
   VFIO provides exclusive device access, zero kernel driver in the data path,
   deterministic scheduling, and IOMMU hardware isolation.
2. **`WgpuDevice` (fallback)** — dispatches through wgpu → Vulkan/Metal/DX12.
   Used for development, non-VFIO environments, and platforms without IOMMU.

| Item | Detail |
|------|--------|
| Feature flag | `sovereign-dispatch` in barraCuda `Cargo.toml` |
| Primary module | `crates/barracuda/src/device/sovereign_device.rs` |
| Fallback module | `crates/barracuda/src/device/wgpu_device.rs` |
| API surface | IPC to shader.compile primal (`shader.compile.wgsl`) + compute.dispatch primal (`compute.dispatch.submit`) |
| Sovereign dispatch | Binary from coralReef IPC → toadStool IPC → GPU |
| Hardware lifecycle | toadStool owns VFIO/DRM lifecycle; barraCuda never sees hardware |
| First target | AMD RDNA2 (GFX1030 — E2E verified in coralReef) |
| Architecture | IPC-first — no compile-time deps between primals |
| NVIDIA | DRM dispatch E2E proven on Titan V + RTX 3090; VFIO 6/7 tests pass, channel init pending |

This is the critical bridge between barraCuda (Layer 1) and the sovereign
stack (Layers 2-4). Without it, the sovereign pipeline has no consumer.

---

## VFIO Primary Dispatch Path

### Strategy

VFIO (Virtual Function I/O) via toadStool is the **primary** GPU dispatch path
for the sovereign pipeline. This replaces the original architecture where wgpu
was the only dispatch backend.

### Why VFIO

| Property | VFIO | wgpu/Vulkan |
|----------|------|-------------|
| Kernel driver in data path | None | Yes (nouveau/amdgpu/NVK) |
| Device exclusivity | Hardware-enforced (IOMMU) | Shared with desktop |
| Scheduling determinism | Deterministic (user-space GPFIFO) | Non-deterministic |
| Memory isolation | IOMMU DMA remapping | Process-level only |
| Gaming compatibility | Dual-use (VFIO for compute, passback for gaming) | Always available |

### Security Enclave Stack

```
BearDog (encrypted shader transport)
  → toadStool (VFIO device lifecycle, IOMMU group management)
    → coralReef (WGSL → native compilation)
      → coral-driver (GPFIFO submission, BAR0 MMIO)
        → VFIO/IOMMU → GPU
```

Rust's memory safety guarantees (`#![forbid(unsafe_code)]` in barraCuda) plus
IOMMU hardware isolation plus encrypted shader transport produce a security
posture unmatched by any CUDA/OpenCL stack.

### Kokkos Parity Projections

With VFIO dispatch + DF64 emulation (9.9x throughput on Hybrid GPUs):
- **Target**: ~4,000 steps/s for RHMC lattice QCD
- **Kokkos baseline**: 2,630 steps/s (C++ / CUDA)
- **Projected advantage**: ~1.5x over Kokkos, achieved in pure safe Rust

### Huge Page DMA

toadStool manages huge page allocation (2 MiB / 1 GiB pages) for DMA buffers,
reducing TLB pressure for large lattice computations. barraCuda consumes these
buffers through the `GpuBackend` trait — no huge page logic lives in barraCuda.

---

## C Dependency Evolution — libc/musl → rustix

**Lead**: toadStool (proven pattern from akida-driver libc → rustix migration)

Follows the same pattern as Ring C → RustCrypto (Tower Atomic): concentrate
the C boundary, then eliminate it with pure Rust alternatives. toadStool
leads the final C elimination charge across the ecosystem — they own the
hardware layer and have already proven the libc → rustix pattern.

### Phases

| Phase | What | Owner | Status | Precedent |
|-------|------|-------|--------|-----------|
| 1 | coral-driver: `libc` → `rustix` for DRM ioctls, mmap, munmap | toadStool + coralReef | Planned | toadStool akida-driver (done) |
| 2 | Validate tokio/mio rustix backend active on our targets | toadStool | Planned | mio uses rustix on Linux by default |
| 3 | Track Rust std `linux-raw-sys` adoption | toadStool | Watching | Active work in Rust project |
| 4 | Zero-package cross-compilation (Android, ARM, RISC-V) | toadStool | Future | Requires Phase 3 for std |

### Current Transitive libc Consumers (barraCuda)

| Crate | Pulls In libc Via | Sovereign Evolution |
|-------|-------------------|---------------------|
| `wgpu-hal` | `ash` → `libvulkan.so` | Eliminated by SovereignDevice (P0) |
| `mio` | `tokio` → epoll/kqueue | Already uses rustix on Linux |
| `signal-hook-registry` | `tokio` → signal handlers | Kernel ABI — Rust std evolves |
| `getrandom` | `rand_core` → `getrandom(2)` | Has linux-raw-sys backend |
| `parking_lot_core` | `wgpu-core` → futex | Eliminated by SovereignDevice (P0) |
| `cpufeatures` | `blake3` → CPUID | Pure feature detection, minimal |
| `socket2` | `tokio` → socket ops | Kernel ABI — Rust std evolves |

### Key Insight

The musl requirement is for static linking via Rust's `std`. Once Rust std
adopts `linux-raw-sys` internally (Phase 3), musl becomes unnecessary for
Linux targets. For our own code, rustix eliminates all direct C calls today.

For Android: the kernel uses the same syscall ABI as Linux. With rustix,
OUR syscall code works on Android without Bionic/NDK. `std` still needs
a C library target until Phase 3 completes.

---

## Cross-Primal Dependency Matrix

### barraCuda needs from coralReef

| Need | Status | Notes |
|------|--------|-------|
| Shader compilation via IPC | Done | `shader.compile.wgsl` JSON-RPC endpoint |
| Compilation result format | Stable | Binary + metadata returned via IPC |
| `Fp64Strategy` in `CompileOptions` | Done | barraCuda passes via IPC `CompileWgslRequest.fp64_strategy` |
| `shader.compile.capabilities` endpoint | Done | Phase 10, with fallback for pre-Phase 10 |

### coralReef needs from barraCuda

| Need | Status | Notes |
|------|--------|-------|
| WGSL shaders parseable by naga | Done | 816 shaders, all naga-valid |
| Precision metadata in compile requests | Done | `fp64_strategy` field in IPC |
| `naga::Module` for direct consumption | Planned | Skip SPIR-V round-trip, needs Layer 3 |

### barraCuda needs from toadStool

| Need | Status | Notes |
|------|--------|-------|
| `compute.dispatch.submit` IPC endpoint | **Done** (S152) | toadStool dispatches binaries to GPU |
| VFIO device bind/unbind lifecycle management | **Done** (S151) | `bind_vfio()` / `unbind_vfio()` with DRM/IOMMU checks |
| Huge page DMA buffer descriptors | **Done** (S152) | `DmaAllocator::allocate_huge()` (2M / 1G pages) |
| Thermal safety for GPU devices | **Done** (S151) | Pre-dispatch thermal checks |
| Multi-GPU parallel init | **Done** (S152) | `compute.hardware.auto_init_all` JSON-RPC |
| Cross-gate GPU pooling | **Done** (S152) | `RemoteDispatcher`, `compute.dispatch.forward` |

### toadStool needs from coralReef

| Need | Status | Notes |
|------|--------|-------|
| Running coralReef instance for shader proxy | Done | JSON-RPC 2.0 + tarpc |
| `PrecisionRoutingAdvice` consumed by barraCuda | Done | `GpuDriverProfile` in barraCuda |

### Springs need from all three

| Need | Provider | Status |
|------|----------|--------|
| Stable IPC for shader compilation | coralReef | Done (Phase 10) |
| Capability-based hardware discovery | toadStool | Done |
| Precision strategy (15-tier: Binary→DF128) | barraCuda | Done (15-tier continuum) |
| Direct sovereign dispatch | barraCuda + coralReef + toadStool | IPC wiring in progress; DRM path E2E verified |

---

## Remaining Work by Priority

### P0 — Blocker

| Item | Owner | Depends On | Status |
|------|-------|------------|--------|
| `GpuBackend` trait + `ComputeDispatch` generic | barraCuda | — | **Done** (Mar 9) |
| `SovereignDevice` scaffold (behind `sovereign-dispatch`) | barraCuda | — | **Done** (Mar 9) |
| `dispatch_binary` + `dispatch_kernel` on `SovereignDevice` | barraCuda | — | **Done** (Mar 12) |
| Coral compiler cache → dispatch wiring | barraCuda | — | **Done** (Mar 12) |
| `SovereignDevice` IPC dispatch wiring | barraCuda | `compute.dispatch.submit` capability | **Done** (Mar 15) — discovers dispatch primal via capability scan, dispatches via JSON-RPC |
| toadStool dispatch IPC endpoint | toadStool | — | API design done (S152); integration pending |
| VFIO dispatch via toadStool IPC | toadStool + coralReef | PFIFO channel init (coralReef) | toadStool ready (S152); coralReef 6/7 VFIO tests pass |

### P1 — Immediate

| Item | Owner | Depends On | Status |
|------|-------|------------|--------|
| VFIO dispatch path documentation across all root docs | barraCuda | — | **Done** (Mar 13) |
| DF64 NVK end-to-end verification on hardware | barraCuda | NVK + NAK hardware | Planned |
| NVIDIA hardware validation (SM70 dispatch) | coralReef | USERD_TARGET fix in runlist | USERD_TARGET + INST_TARGET fix applied (Iter 50); hw revalidation on Titan V pending |
| `nak-ir-proc` unsafe → safe (array-field or bytemuck) | coralReef | — | Planned |
| Hand-written DF64 shader for weighted_dot | barraCuda | — | **Done** (Mar 12) |
| `BatchedTridiagEigh` GPU op (from groundSpring) | barraCuda | — | Planned |
| Kokkos parity validation baseline + GPU benchmarks | barraCuda | Matching hardware | Planned (unblocked by VFIO strategy) |

### P2 — Near-term

| Item | Owner | Depends On | Status |
|------|-------|------------|--------|
| coral-driver `libc` → `rustix` | toadStool + coralReef | — | In progress (toadStool S149+; extern "C" removal done S152) |
| Test coverage 80% → 90% | barraCuda | — | In progress |
| WGSL optimizer annotation coverage (`@ilp_region`) | barraCuda | — | **Done** (Mar 12) — variance_reduce_df64, weighted_dot_df64, mean_variance_df64, covariance_f64 |
| RHMC multi-shift CG absorb (from hotSpring) | barraCuda | — | **Done** (Mar 12) — rhmc.rs + rhmc_hmc.rs |

### P3 — Medium-term

| Item | Owner | Depends On | Status |
|------|-------|------------|--------|
| Standalone `coral-reef` crate (no Mesa build) | coralReef | — | Planned |
| Direct `naga::Module` → coralReef path | coralReef | Standalone crate | Planned |
| Multi-GPU dispatch (GpuView spanning devices) | barraCuda | — | Planned |
| Pipeline cache re-enable | barraCuda | wgpu safe API | Watching |
| Shader hot-reload (file watcher + recompile) | barraCuda | — | Planned |
| Zero-copy: domain_ops, LSTM, RBF allocations | barraCuda | — | Planned |

### P4 — Long-term

| Item | Owner | Depends On | Status |
|------|-------|------------|--------|
| Sovereign hardware driver (VFIO + coralDriver DMA) | toadStool | PFIFO channel init (coralReef) | toadStool S152 infra complete; coralReef VFIO 6/7 |
| AMD RDNA3/CDNA2 ISA support | coralReef | Hardware access | Planned |
| Intel Xe ISA support | coralReef | Hardware access | Future |
| WebGPU browser target (wasm-pack) | barraCuda | wgpu WebGPU backend | Future |
| Distributed compute (cross-node IPC) | barraCuda + songBird | — | Future |
| CPU shader interpreter (naga + cranelift) | barraCuda | — | Future |
| Rust std `linux-raw-sys` adoption (eliminates musl) | Rust project | — | Watching |

---

## Cross-Compilation Target Matrix

### Current (musl-based, per genomeBin standard)

| Target | Status | Package Required |
|--------|--------|-----------------|
| `x86_64-unknown-linux-musl` | Working | `musl-tools` |
| `aarch64-unknown-linux-musl` | Working | `musl-tools` + cross-compiler |
| `riscv64gc-unknown-linux-musl` | Working | `musl-tools` + cross-compiler |
| `x86_64-apple-darwin` | Working | macOS SDK (CI) |
| `aarch64-apple-darwin` | Working | macOS SDK (CI) |

### Evolution (rustix + Rust std linux-raw-sys)

| Target | Today | After Phase 1-2 | After Phase 3-4 |
|--------|-------|-----------------|-----------------|
| Linux x86_64 | musl-tools | musl-tools (our code clean) | `rustup target add` only |
| Linux aarch64 | musl + cross-cc | musl + cross-cc (our code clean) | `rustup target add` only |
| Android aarch64 | NDK required | NDK (our ioctls via rustix) | `rustup target add` only |
| RISC-V | musl + cross-cc | musl + cross-cc | `rustup target add` only |

The endgame: a Rust binary making raw syscalls without any C library,
compilable for any Linux-based architecture with just `rustup target add`.

---

## References

- `PURE_RUST_EVOLUTION.md` — layer philosophy, stack diagrams
- `WHATS_NEXT.md` — C dependency chain evolution map
- `specs/REMAINING_WORK.md` — barraCuda-specific P1-P3 work items
- `wateringHole/PURE_RUST_SOVEREIGN_STACK_GUIDANCE.md` — cross-primal contracts, Layer 2-4 guidance
- `wateringHole/GENOMEBIN_ARCHITECTURE_STANDARD.md` — musl cross-compilation standard
- `specs/ARCHITECTURE_DEMARCATION.md` — primal ownership boundaries

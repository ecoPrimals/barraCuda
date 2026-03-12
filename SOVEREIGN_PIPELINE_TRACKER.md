# Sovereign Pipeline Tracker

**Date**: March 12, 2026
**Type**: Actionable tracker (updated as work progresses)
**Scope**: All remaining work for the pure Rust sovereign GPU pipeline

---

## Pipeline Status

```
Layer 1  barraCuda    ██████████  COMPLETE   Zero unsafe, zero C deps
Layer 2  coralReef    ██████░░░░  Phase 10   9 unsafe (coral-driver RAII + nak-ir-proc)
Layer 3  Standalone   ██░░░░░░░░  Planned    Standalone coral-reef crate, multi-arch ISA
Layer 4  Sovereign HW █░░░░░░░░░  Planned    coralDriver (pure Rust DRM ioctls)
```

```
barraCuda (WGSL math)
  → coralReef (WGSL/SPIR-V → native binary)
    → coralDriver (DRM ioctls, userspace GPU dispatch)
      → Hardware (any GPU, CPU, NPU, Android ARM)
```

---

## P0 — CoralReefDevice Backend (the bridge)

barraCuda currently dispatches GPU work through wgpu (which goes through
Vulkan/Metal/DX12). The sovereign path requires a `CoralReefDevice` backend
that dispatches directly via `coral-gpu::GpuContext`, bypassing the entire
Vulkan stack.

| Item | Detail |
|------|--------|
| Feature flag | `sovereign-dispatch` in barraCuda `Cargo.toml` |
| New module | `crates/barracuda/src/device/coral_reef_device.rs` |
| API surface | Wraps `GpuContext` — `compile_wgsl`, `alloc`, `upload`, `dispatch`, `sync`, `readback` |
| Fallback | wgpu when coralReef unavailable (feature-gated, not runtime) |
| First target | AMD RDNA2 (GFX1030 — E2E verified in coralReef) |
| Blocked by | `coral-gpu` crate publishable as `cargo add` dependency |
| NVIDIA | Pending hardware validation in coralReef (SM70 codegen exists, dispatch untested) |

This is the critical bridge between barraCuda (Layer 1) and the sovereign
stack (Layers 2-4). Without it, the sovereign pipeline has no consumer.

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
| `wgpu-hal` | `ash` → `libvulkan.so` | Eliminated by CoralReefDevice (P0) |
| `mio` | `tokio` → epoll/kqueue | Already uses rustix on Linux |
| `signal-hook-registry` | `tokio` → signal handlers | Kernel ABI — Rust std evolves |
| `getrandom` | `rand_core` → `getrandom(2)` | Has linux-raw-sys backend |
| `parking_lot_core` | `wgpu-core` → futex | Eliminated by CoralReefDevice (P0) |
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
| `coral-gpu` as `cargo add` dependency | Not yet | Required for `sovereign-dispatch` feature |
| `GpuContext` API stability | Stable | `compile_wgsl`, `alloc`, `upload`, `dispatch`, `sync`, `readback` |
| `Fp64Strategy` in `CompileOptions` | Done | barraCuda passes via IPC `CompileWgslRequest.fp64_strategy` |
| `shader.compile.capabilities` endpoint | Done | Phase 10, with fallback for pre-Phase 10 |

### coralReef needs from barraCuda

| Need | Status | Notes |
|------|--------|-------|
| WGSL shaders parseable by naga | Done | 803 shaders, all naga-valid |
| Precision metadata in compile requests | Done | `fp64_strategy` field in IPC |
| `naga::Module` for direct consumption | Planned | Skip SPIR-V round-trip, needs Layer 3 |

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
| Precision strategy (F32/F64/Df64) | barraCuda | Done (3-tier model) |
| Direct sovereign dispatch | barraCuda + coralReef | Blocked on P0 |

---

## Remaining Work by Priority

### P0 — Blocker

| Item | Owner | Depends On | Status |
|------|-------|------------|--------|
| `GpuBackend` trait + `ComputeDispatch` generic | barraCuda | — | **Done** (Mar 9) |
| `CoralReefDevice` scaffold (behind `sovereign-dispatch`) | barraCuda | — | **Done** (Mar 9) |
| `dispatch_binary` + `dispatch_kernel` on `CoralReefDevice` | barraCuda | — | **Done** (Mar 12) |
| Coral compiler cache → dispatch wiring | barraCuda | — | **Done** (Mar 12) |
| `CoralReefDevice` functional implementation | barraCuda | `coral-gpu` crate publishable | Blocked |
| `coral-gpu` crate as standalone dependency | coralReef | — | In progress |

### P1 — Immediate

| Item | Owner | Depends On | Status |
|------|-------|------------|--------|
| DF64 NVK end-to-end verification on hardware | barraCuda | NVK + NAK hardware | Planned |
| NVIDIA hardware validation (SM70 dispatch) | coralReef | NVIDIA hardware access | Planned |
| `nak-ir-proc` unsafe → safe (array-field or bytemuck) | coralReef | — | Planned |
| Dedicated DF64 shaders for covariance + weighted_dot | barraCuda | — | Planned |
| `BatchedTridiagEigh` GPU op (from groundSpring) | barraCuda | — | Planned |

### P2 — Near-term

| Item | Owner | Depends On | Status |
|------|-------|------------|--------|
| coral-driver `libc` → `rustix` | toadStool + coralReef | — | Planned |
| Test coverage 80% → 90% | barraCuda | — | In progress |
| Kokkos validation baseline + GPU parity benchmarks | barraCuda | Matching hardware | Planned |
| WGSL optimizer annotation coverage (`@ilp_region`) | barraCuda | — | Planned |
| RHMC multi-shift CG absorb (from hotSpring) | barraCuda | — | Planned |

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
| Sovereign hardware driver (coralDriver DMA) | toadStool | Layer 3 | Planned |
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

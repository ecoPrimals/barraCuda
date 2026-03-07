# Spring Absorption Tracker

**Version**: 0.3.3 → 0.3.4
**Date**: March 7, 2026
**Source**: hotSpring v0.6.19, groundSpring V88, neuralSpring V128, wetSpring V97d, airSpring v0.7.0

Cross-spring evolution follows **Write → Absorb → Lean**: springs implement
domain-specific primitives, barraCuda absorbs and generalises, springs consume
the upstream version. All springs are synced to barraCuda v0.3.3 / wgpu 28
with zero local WGSL (except airSpring, 3 remaining).

---

## P0 — Cross-Spring Integration (Mar 7 2026)

| # | Item | Source | Module | Status |
|---|------|--------|--------|--------|
| A | coralReef Phase 10 `compile_wgsl` direct | coralReef Phase 10 handoff | `device::coral_compiler` | ✅ Done |
| B | coralReef `supported_archs()` query | coralReef Phase 10 handoff | `device::coral_compiler` | ✅ Done |
| C | Shader provenance registry (cross-spring evolution) | all springs | `shaders::provenance` | ✅ Done |
| D | Cross-spring shader validation harness (naga) | toadStool S128, coralReef | `device::test_harness` | ✅ Done |
| E | Cross-spring evolution benchmark suite | all springs | `tests/cross_spring_benchmark` | ✅ Done |
| F | Modern cross-spring validation (Welford, eps, Verlet, tolerance) | all springs | `tests/cross_spring_validation` | ✅ Done |

### Cross-Spring Shader Flow (from provenance registry)

```text
hotSpring → groundSpring: 3 shared shaders (df64, complex, CG solver)
hotSpring → neuralSpring: 3 shared shaders (df64, lattice CG → attention)
hotSpring → wetSpring:    3 shared shaders (df64, ESN, fused_map_reduce)
airSpring → wetSpring:    3 shared shaders (ET₀, seasonal, moving_window)
neuralSpring → hotSpring: 2 shared shaders (chi-squared, matrix_correlation)
neuralSpring → wetSpring: 2 shared shaders (KL divergence, chi-squared)
groundSpring → hotSpring: 2 shared shaders (Anderson Lyapunov, chi-squared)
```

### Benchmark Summary (CPU, 100K points)

| Primitive | Throughput | Origin |
|-----------|-----------|--------|
| Welford univariate | 97M pts/s | groundSpring V80 |
| Welford parallel merge | 97M pts/s (8 chunks) | groundSpring V80 |
| Tolerance 4-tier sweep | 56M cmp/s | groundSpring V76 |
| eps guard application | 124M ops/s | groundSpring V76 |

### Debt Evolution (Mar 7 2026)

| # | Item | Change | Status |
|---|------|--------|--------|
| G | Akida SDK paths → capability constant | Extracted `AKIDA_SDK_SYSTEM_DIRS` shared between `akida.rs` and `kernel_router.rs` | ✅ Done |
| H | Deprecated PPPM constructors removed | `new()` / `new_with_driver()` removed — zero callers, `from_device()` is the API | ✅ Done |
| I | `SeasonalGpuParams` → builder pattern | 9-arg constructor replaced with `.builder().stage().crop_coefficients().soil().build()` | ✅ Done |
| J | `HmmBatchForwardF64::dispatch` → `HmmForwardArgs` struct | 11-arg dispatch replaced with grouped buffer struct | ✅ Done |
| K | CPU reference functions: `dead_code` → `#[expect]` audit | 19 files verified — all CPU refs correctly gated with `#[expect(dead_code)]` | ✅ Done |
| L | Unused `ShaderTemplate` import cleaned | Removed from `pppm_gpu/mod.rs` after deprecated constructor removal | ✅ Done |
| M | `GpuCgSolver::solve` → `CgLatticeBuffers` + `CgSolverConfig` | 9-arg → 4-arg; 3 callers in `gpu_hmc_trajectory.rs` updated | ✅ Done |
| N | `GillespieGpu::simulate` → `GillespieModel` struct | 8-arg → 5-arg; 2 test callers updated | ✅ Done |
| O | `Rk45AdaptiveGpu::dispatch` → `Rk45DispatchArgs` | 10-arg → single struct; 0 callers | ✅ Done |
| P | `Dada2EStepGpu::dispatch` → `Dada2DispatchArgs` | 10-arg → single struct; 0 callers | ✅ Done |
| Q | `SpinOrbitGpu::compute_internal` → `SpinOrbitInputs` | 12-arg → single struct; 2 internal callers updated | ✅ Done |
| R | `GpuHmcLeapfrog::dispatch` → `LeapfrogBuffers` | 9-arg → 5-arg; 3 internal callers + omelyan updated | ✅ Done |
| S | `RBFSurrogate::from_parts` → `RbfTrainingData` + `RbfTrainedModel` | 9-arg → 3-arg; 1 caller in `adaptive/mod.rs` updated | ✅ Done |
| T | Zero `#[expect(clippy::too_many_arguments)]` remaining | All 9 instances evolved to parameter structs/builders | ✅ Done |

## P1 — High Priority

| # | Item | Source | Module | Status |
|---|------|--------|--------|--------|
| 1 | `has_reliable_f64()` device method | toadStool S97, hotSpring | `device::driver_profile` | ✅ Done |
| 2 | `eps::` guard constants in shader preamble | groundSpring V76 | `shaders::precision` | ✅ Done |
| 3 | 13-tier named tolerance system | groundSpring V76 | `numerical::tolerance` | ✅ Done |
| 4 | Verlet neighbor list (`VerletListGpu`) | hotSpring | `ops::md::neighbor` | ✅ Done |
| 5 | Dedicated DF64 shaders (covariance, weighted_dot) | hotSpring, internal | `shaders/`, ops | 🔶 Deferred (auto-rewrite works) |
| 6 | Subgroup-aware workgroup sizing | toadStool S97 | `device::driver_profile` | ✅ Done |
| 7 | Welford co-moment + covariance_gpu formalization | groundSpring V80 | `stats::welford` | ✅ Done |
| 8 | `BatchedOdeRK45F64` GPU variant | wetSpring V95 | `ops::rk45_adaptive` | 🔲 Pending |

## P2 — Medium Priority

| # | Item | Source | Module | Status |
|---|------|--------|--------|--------|
| 9 | `GpuView<T>` zero-copy expansion | hotSpring | `pipeline::gpu_view` | 🔲 Pending |
| 10 | `mean_variance_buffer()` fused GPU stats | hotSpring | `ops::stats_f64` | 🔲 Pending |
| 11 | RHMC multi-shift CG solver | hotSpring ladder L4 | `ops::lattice` | 🔲 Pending |
| 12 | Adaptive HMC dt from acceptance rate | hotSpring | `ops::lattice` | 🔲 Pending |
| 13 | Anderson Lyapunov shaders | groundSpring | `ops` | 🔲 Pending |
| 14 | airSpring local ops (Makkink, Turc, Hamon) | airSpring | `stats::hydrology` | 🔲 Pending |
| 15 | Covariance from `correlation_full` shader | groundSpring V80 | `ops` | 🔲 Pending |

## P3 — Infrastructure

| # | Item | Source | Module | Status |
|---|------|--------|--------|--------|
| 16 | `ComputeBackend` trait | hotSpring | `device` | 🔲 Pending |
| 17 | `ComputeDispatch` tarpc for NUCLEUS | wetSpring | `barracuda-core` | 🔲 Pending |
| 18 | `BandwidthTier` in device profile | wetSpring | `device::driver_profile` | 🔲 Pending |
| 19 | `domain-genomics` feature extraction | wetSpring | feature flags | 🔲 Pending |

---

## Numerical Stability Notes (from springs)

- **f32 accumulation bias**: Green-Kubo gives ~28% bias — use f64/DF64 for reductions
- **GPU transcendental precision**: one tier looser than CPU per transcendental call
- **GPU NaN source**: division-by-zero — mitigated by `eps::SAFE_DIV` (item 2)
- **NVK/NAK f64**: unreliable on Titan V, RTX 4070 — mitigated by `has_reliable_f64()` (item 1)
- **DF64 Yukawa on NVK**: 300–900 steps/s vs 29 native — requires `compile_shader_universal(Df64)`

## References

- `wateringHole/handoffs/CORALREEF_PHASE10_CROSS_SPRING_REWIRE_HANDOFF_MAR07_2026.md`
- `wateringHole/handoffs/BARRACUDA_V033_SPRING_ABSORPTION_DEEP_DEBT_HANDOFF_MAR06_2026.md`
- `wateringHole/handoffs/HOTSPRING_SCIENCE_LADDER_BARRACUDA_ABSORPTION_MAR06_2026.md`
- `wateringHole/handoffs/HOTSPRING_V0619_CROSS_SPRING_EVOLUTION_HANDOFF_MAR06_2026.md`
- `wateringHole/handoffs/GROUNDSPRING_V76_TOADSTOOL_BARRACUDA_ABSORPTION_HANDOFF_MAR05_2026.md`
- `wateringHole/handoffs/GROUNDSPRING_V80_FUSED_OPS_BARRACUDA_CATCHUP_HANDOFF_MAR05_2026.md`
- `wateringHole/handoffs/HOTSPRING_VERLET_EVOLUTION_HANDOFF_MAR05_2026.md`
- `wateringHole/handoffs/WETSPRING_V95_BARRACUDA_TOADSTOOL_EVOLUTION_HANDOFF_MAR04_2026.md`
- `wateringHole/handoffs/TOADSTOOL_S97_SPRING_ABSORPTION_HANDOFF_MAR06_2026.md`

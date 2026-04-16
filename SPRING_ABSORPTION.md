# Spring Absorption Tracker

**Version**: 0.3.12
**Date**: April 16, 2026
**Source**: hotSpring v0.6.32, groundSpring V99, neuralSpring S143, wetSpring V107, airSpring v0.7.5, healthSpring V44, ludoSpring V34, primalSpring (new), toadStool S202, coralReef Phase 10 Iter 80

## Sprint 22g — Cross-Spring Deep Absorption (Mar 29 2026)

| # | Item | Source | Module | Status |
|---|------|--------|--------|--------|
| AA | TMU Box-Muller PRNG shader | hotSpring V0632 | `shaders/lattice/su3_random_momenta_tmu_f64.wgsl` | Done |
| AB | Subgroup-accelerated reduction | hotSpring V0632 | `shaders/reduce/sum_reduce_subgroup_f64.wgsl` | Done |
| AC | ROP fermion force accumulation | hotSpring V0632 | `shaders/lattice/su3_fermion_force_accumulate_rop_f64.wgsl` | Done |
| AD | Atomic force → momentum conversion | hotSpring V0632 | `shaders/lattice/su3_force_atomic_to_momentum_f64.wgsl` | Done |
| AE | TMU lookup table builder | hotSpring V0632 | `ops/lattice/tmu_tables.rs` | Done |
| AF | ROP fixed-point accum params | hotSpring V0632 | `ops/lattice/rop_force_accum.rs` | Done |
| AG | Hill dose-response shader + dispatch | healthSpring V44 | `ops/health/hill_dose_response.rs` | Done |
| AH | Population PK Monte Carlo | healthSpring V44 | `ops/health/population_pk.rs` | Done |
| AI | Shannon/Simpson diversity | healthSpring V44 | `ops/health/diversity.rs` | Done |
| AJ | `SiliconProfile` + tier routing | hotSpring V0632 | `device/silicon_profile.rs` | Done |
| AK | `SUBGROUP` feature negotiation | hotSpring V0632 | `device/wgpu_device/creation.rs` | Done |
| AL | 6-format capability parser | ludoSpring V34 | `discovery/capabilities.rs` | Done |
| AM | Env-configurable IPC tolerances | ludoSpring V34 | `discovery/tolerances.rs` | Done |

Cross-spring evolution follows **Write → Absorb → Lean**: springs implement
domain-specific primitives, barraCuda absorbs and generalises, springs consume
the upstream version. All springs are synced to barraCuda v0.3.12 / wgpu 28
with zero local WGSL (airSpring and wetSpring fully lean).

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
hotSpring → groundSpring: 3 shaders (df64, complex, CG solver)
hotSpring → neuralSpring: 4 shaders (df64, df64_transcendentals, lattice CG, ESN)
hotSpring → wetSpring:    6 shaders (df64, df64_transcendentals, stress_virial, verlet, ESN, fused_map_reduce)
airSpring → wetSpring:    4 shaders (ET₀, seasonal, moving_window, fused_map_reduce)
airSpring → neuralSpring: 1 shader  (moving_window → streaming inference)
neuralSpring → hotSpring:  3 shaders (chi-squared, matrix_correlation, batch_ipr)
neuralSpring → wetSpring:  2 shaders (KL divergence, chi-squared)
neuralSpring → groundSpring: 2 shaders (KL divergence, matrix_correlation)
groundSpring → hotSpring:  2 shaders (Anderson Lyapunov, chi-squared)
groundSpring → ALL:        2 shaders (chi_squared universal, Welford mean+variance)
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
| K | CPU reference functions: `dead_code` lint audit | All CPU refs carry `#[allow(dead_code, reason = "...")]` where test-used, `#[expect(dead_code)]` where truly dead | ✅ Done |
| L | Unused `ShaderTemplate` import cleaned | Removed from `pppm_gpu/mod.rs` after deprecated constructor removal | ✅ Done |
| M | `GpuCgSolver::solve` → `CgLatticeBuffers` + `CgSolverConfig` | 9-arg → 4-arg; 3 callers in `gpu_hmc_trajectory.rs` updated | ✅ Done |
| N | `GillespieGpu::simulate` → `GillespieModel` struct | 8-arg → 5-arg; 2 test callers updated | ✅ Done |
| O | `Rk45AdaptiveGpu::dispatch` → `Rk45DispatchArgs` | 10-arg → single struct; 0 callers | ✅ Done |
| P | `Dada2EStepGpu::dispatch` → `Dada2DispatchArgs` | 10-arg → single struct; 0 callers | ✅ Done |
| Q | `SpinOrbitGpu::compute_internal` → `SpinOrbitInputs` | 12-arg → single struct; 2 internal callers updated | ✅ Done |
| R | `GpuHmcLeapfrog::dispatch` → `LeapfrogBuffers` | 9-arg → 5-arg; 3 internal callers + omelyan updated | ✅ Done |
| S | `RBFSurrogate::from_parts` → `RbfTrainingData` + `RbfTrainedModel` | 9-arg → 3-arg; 1 caller in `adaptive/mod.rs` updated | ✅ Done |
| T | Zero `#[expect(clippy::too_many_arguments)]` remaining | All 9 instances evolved to parameter structs/builders | ✅ Done |

### Cross-Spring Rewiring (Mar 7 2026)

| # | Item | Change | Status |
|---|------|--------|--------|
| U | Provenance evolution dates + bidirectional flows | 27 shaders with `created`/`absorbed` dates, 10 timeline events, `evolution_report()` | ✅ Done |
| V | `PrecisionRoutingAdvice` from toadStool S128 | `F64Native`/`F64NativeNoSharedMem`/`Df64Only`/`F32Only` in `GpuDriverProfile` | ✅ Done |
| W | `mean_variance_to_buffer()` GPU-resident stats | Zero-readback fused Welford — output stays as GPU buffer for chained pipelines | ✅ Done |
| X | `BatchedOdeRK45F64` integrator | Full-trajectory adaptive RK45 on GPU with step-size control (wetSpring V95) | ✅ Done |
| Y | MD + ML provenance expansion | `stress_virial`, `verlet_neighbor`, `batch_ipr`, `hmm_forward`, `hfb_gradient`, `welford` | ✅ Done |
| Z | Cross-spring evolution report generator | Dependency matrix + timeline + category report (programmatic) | ✅ Done |

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
| 8 | `BatchedOdeRK45F64` GPU variant | wetSpring V95 | `ops::rk45_adaptive` | ✅ Done |

## P2 — Medium Priority

| # | Item | Source | Module | Status |
|---|------|--------|--------|--------|
| 9 | `GpuView<T>` ops: `mean_variance`, `sum`, `correlation` | groundSpring, hotSpring | `pipeline::gpu_view` | ✅ Done |
| 10 | `mean_variance_to_buffer()` fused GPU stats | hotSpring | `ops::variance_f64_wgsl` | ✅ Done |
| 11 | RHMC multi-shift CG solver + rational approximation + RHMC HMC | hotSpring ladder L4 | `ops::lattice::rhmc`, `ops::lattice::rhmc_hmc` | ✅ Done (Mar 12) |
| 12 | Adaptive HMC dt from acceptance rate | hotSpring | `ops::lattice` | 🔲 Pending |
| 13 | Anderson Lyapunov shaders | groundSpring | `ops` | ✅ Done (shaders absorbed: `anderson_lyapunov_f64.wgsl`, `anderson_lyapunov_f32.wgsl`) |
| 14 | airSpring local ops (Makkink, Turc, Hamon) | airSpring | `stats::hydrology` | ✅ Done (already absorbed) |
| 15 | Covariance from `correlation_full` shader | groundSpring V80 | `ops` | ✅ Done (`CorrelationResult::covariance()` uses `correlation_full`) |

### Deep Debt Sprint (Mar 10 2026)

| # | Item | Source | Module | Status |
|---|------|--------|--------|--------|
| AW | **Unified GFLOPS/VRAM estimation** | groundSpring/internal | `multi_gpu::mod` | ✅ Done |
| AX | **Fp64Strategy routing fix (4 reduce ops)** | hotSpring NVVM guidance | `ops::*_reduce_f64` | ✅ Done |
| AY | **PCIe topology sysfs probing** | internal (multi-GPU evolution) | `unified_hardware::transfer` | ✅ Done |
| AZ | **VRAM quota in buffer allocation** | internal (multi-GPU evolution) | `device::wgpu_device::buffers` | ✅ Done |
| BA | **BGL builder** | wetSpring V105 | `device::compute_pipeline` | ✅ Done |
| BB | **Test pipeline optimisation** | internal audit | `nautilus::shell`, `esn_v2`, `sovereign::validation_harness` | ✅ Done |

## P3 — Infrastructure

| # | Item | Source | Module | Status |
|---|------|--------|--------|--------|
| 16 | `ComputeBackend` trait | hotSpring | `device` | ✅ Done (`GpuBackend`) |
| 17 | `ComputeDispatch` tarpc for NUCLEUS | wetSpring | `barracuda-core` | 🔲 Pending |
| 18 | `BandwidthTier` in device profile | wetSpring | `device::driver_profile` | ✅ Done (in `unified_hardware::transfer`) |
| 19 | `domain-genomics` feature extraction | wetSpring | feature flags | 🔲 Pending |

---

### Cross-Spring Absorption (Mar 8 2026)

| # | Item | Change | Status |
|---|------|--------|--------|
| AA | **P0: Fp64Strategy in `SumReduceF64`/`VarianceReduceF64`** | DF64 shader variants (`sum_reduce_df64.wgsl`, `variance_reduce_df64.wgsl`) + `Fp64Strategy` routing — Hybrid devices now use DF64 workgroup memory instead of unreliable f64 shared memory | ✅ Done |
| AB | **P1: Re-export builder types** | `HmmForwardArgs`, `Dada2DispatchArgs`, `Dada2Buffers`, `Dada2Dimensions`, `GillespieModel`, `PrecisionRoutingAdvice`, `Rk45DispatchArgs` at `barracuda::` level | ✅ Done |
| AC | **P1: `barracuda::math::{dot, l2_norm}`** | Re-exported from `stats::metrics` — 15+ wetSpring binaries can drop local implementations | ✅ Done |
| AD | **P1: `fused_ops_healthy()` canary** | `device::test_harness::fused_ops_healthy(&device)` — gates fused-reduction test suites on Hybrid devices | ✅ Done |
| AE | **P2: NVK zero-output detection** | `GpuDriverProfile::f64_zeros_risk()` — flags NVK + Full/Throttled FP64 as shared-memory-unreliable | ✅ Done |
| AF | **P2: `GpuViewF64` ops** | `mean_variance(ddof)`, `sum()`, `GpuViewF64::correlation(a, b)` — stepping-stone API for zero-readback chains | ✅ Done |
| AG | **P2: Test utilities** | `is_software_adapter(&device)`, `baseline_path(relative)` in `test_harness` + re-exported in `test_prelude` | ✅ Done |

---

### Cross-Spring Absorption Sprint 2 (Mar 9 2026)

| # | Item | Source | Module | Status |
|---|------|--------|--------|--------|
| AH | **P0: Tridiagonal QL eigensolver** | healthSpring `microbiome.rs` | `special::tridiagonal_ql` | ✅ Done |
| AI | **P0: LCG PRNG centralization** | healthSpring `rng.rs` | `rng` | ✅ Done |
| AJ | **P0: Public activations API** | neuralSpring S134 request | `activations` | ✅ Done |
| AK | **P1: Wright-Fisher population genetics** | neuralSpring `metalForge` | `ops::wright_fisher_f32` | ✅ Done |
| AL | **P1: Hill dose-response Emax** | healthSpring V13 | `ops::hill_f64` | ✅ Done |
| AM | **P1: Population PK Monte Carlo** | healthSpring V13 | `ops::population_pk_f64` | ✅ Done |
| AN | **P1: Plasma dispersion W(z)/Z(z)** | hotSpring `dielectric.rs` | `special::plasma_dispersion` | ✅ Done |
| AO | **P1: Batched f32 logsumexp** | neuralSpring confirm | `shaders/reduce/logsumexp_reduce_f32.wgsl` | ✅ Done |
| AP | **P2: healthSpring provenance domain** | healthSpring V13 | `shaders::provenance::types` | ✅ Done |
| AQ | **Cleanup: orphaned code removal** | internal audit | `ops/cyclic_reduction_wgsl.rs`, `ops/reduce/` | ✅ Removed |

---

### Cross-Spring Absorption Sprint 3 (Mar 9 2026)

| # | Item | Source | Module | Status |
|---|------|--------|--------|--------|
| AR | **P1: `Rk45Result::variable_trajectory()`** | wetSpring V102 | `numerical::rk45` | ✅ Done |
| AS | **P1: `analyze_weight_matrix()` composite** | neuralSpring S135 | `spectral::stats` | ✅ Done |
| AT | **P1: `histogram_u32_to_f64()` convenience** | wetSpring V102 | `ops::bio::kmer_histogram` | ✅ Done |
| AU | **P0: toadStool S139 discovery alignment** | toadStool S139 | `device::coral_compiler::discovery` | ✅ Done |
| AV | **Audit: confirmed existing coverage** | airSpring v0.7.5 | — | ✅ `regularized_gamma_q`, `CorrelationResult::r_squared()`, ET0 GPU shaders all already present |

---

### Ecosystem-Wide Absorption Sprint 18 (Mar 21 2026)

Full pull + review of 8 springs, 10+ primals. Key findings:

| # | Item | Source | Status |
|---|------|--------|--------|
| BC | **Remove deprecated `GpuDriverProfile` struct** | all springs (migrated Sprint 14) | ✅ Done — enums retained |
| BD | **`barracuda::cast` safe numeric cast module** | groundSpring typed-error pattern | ✅ Done — `CastOverflow`/`PrecisionLoss` error variants |
| BE | **Tolerance stability contract** | groundSpring V76 pattern | ✅ Done — documented in `tolerances.rs` |
| BF | **`ESN::wgpu_device()` + `MultiHeadEsn::wgpu_device()`** | neuralSpring S143 request | ✅ Done |
| BG | **`domain-fold` feature gate** | wetSpring structural biology | ✅ Done — gates `folding_df64` module |
| BH | **f64 shader constants exposed as public API** | all springs | ✅ Done — `WGSL_GELU_F64`, `WGSL_SOFTMAX_*_F64` |
| BI | **`cast_lossless` promoted to warn** | internal lint evolution | ✅ Done — zero violations |
| BJ | **Pairwise Hamming/Jaccard/L2** | neuralSpring request | ✅ Already existed (`ops/bio/pairwise_*`, `cdist`) |
| BK | **Chi-squared / KL divergence** | neuralSpring request | ✅ Already existed (`special/chi_squared`, `ops/kl_divergence`) |
| BL | **xoshiro128ss GPU PRNG** | wetSpring V107 | ✅ Already existed (`ops/prng_xoshiro_wgsl`) |
| BM | **HMM backward/Viterbi** | neuralSpring S143 | ✅ Already existed (`ops/bio/hmm`) |
| BN | **Health ODE systems** | healthSpring V19 | 🔲 Deferred — infrastructure ready (`ode_bio/` + `OdeSystem` trait) |
| BO | **ludoSpring / primalSpring** | new repos | ✅ Cloned — no absorption candidates yet (early) |

---

## neuralSpring V131 Shader Absorption Audit (Apr 15 2026)

Per-shader verification against `neuralSpring/docs/PRIMAL_GAPS.md` §10.
The "29 shader absorption candidates" count in primalSpring docs includes
neuralSpring-specific shaders that stay in neuralSpring (protein folding,
MSA attention). The barraCuda absorption target is **18 candidates** — all
confirmed upstream with canonical `.wgsl` files and Rust integration.

| # | neuralSpring candidate | barraCuda WGSL | Rust integration | Status |
|---|------------------------|----------------|------------------|--------|
| 1 | `softmax_f64.wgsl` | `shaders/activation/softmax_f64.wgsl` | `ops/softmax.rs` | ✅ Upstream |
| 2 | `gelu_f64.wgsl` | `shaders/activation/gelu_f64.wgsl` | `ops/gelu_wgsl.rs` | ✅ Upstream |
| 3 | `sigmoid_f64.wgsl` | `shaders/misc/sigmoid_f64.wgsl` | `ops/sigmoid.rs` | ✅ Upstream |
| 4 | `hmm_backward_log.wgsl` | `shaders/bio/hmm_backward_log_f64.wgsl` | `ops/bio/hmm.rs` | ✅ Upstream |
| 5 | `hmm_viterbi.wgsl` | `shaders/bio/hmm_viterbi_f64.wgsl` | `ops/bio/hmm.rs` | ✅ Upstream |
| 6 | `batch_ipr.wgsl` | `shaders/spectral/batch_ipr_f64.wgsl` | `spectral/batch_ipr.rs` | ✅ Upstream |
| 7 | `chi_squared_f64.wgsl` | `shaders/special/chi_squared_f64.wgsl` | `special/chi_squared.rs` | ✅ Upstream |
| 8 | `kl_divergence_f64.wgsl` | `shaders/loss/kl_divergence_f64.wgsl` | `ops/kl_divergence.rs` | ✅ Upstream |
| 9 | `linear_regression.wgsl` | `shaders/stats/linear_regression_f64.wgsl` | `ops/stats_f64.rs` | ✅ Upstream |
| 10 | `matrix_correlation.wgsl` | `shaders/stats/matrix_correlation_f64.wgsl` | `ops/stats_f64.rs` | ✅ Upstream |
| 11 | `rk4_parallel.wgsl` | `shaders/numerical/rk4_parallel_f64.wgsl` | `ops/rk_stage.rs` | ✅ Upstream |
| 12 | `rk45_adaptive.wgsl` | `shaders/numerical/rk45_adaptive_f64.wgsl` | `ops/rk45_adaptive.rs` | ✅ Upstream |
| 13 | `wright_fisher_step.wgsl` | `shaders/bio/wright_fisher_step_f64.wgsl` | `ops/bio/wright_fisher.rs` | ✅ Upstream |
| 14 | `pairwise_hamming.wgsl` | `shaders/math/pairwise_hamming_f64.wgsl` | `ops/bio/pairwise_hamming.rs` | ✅ Upstream |
| 15 | `pairwise_jaccard.wgsl` | `shaders/math/pairwise_jaccard_f64.wgsl` | `ops/bio/pairwise_jaccard.rs` | ✅ Upstream |
| 16 | `sdpa_scores_f64.wgsl` | `shaders/attention/sdpa_scores_f64.wgsl` | `ops/attention/mod.rs` | ✅ Upstream |
| 17 | `layer_norm_f64.wgsl` | `shaders/norm/layer_norm_f64.wgsl` | `ops/layer_norm_wgsl.rs` | ✅ Upstream |
| 18 | `outer_product_mean_f64.wgsl` | `shaders/attention/outer_product_mean_f64.wgsl` | `ops/alphafold2/embedding.rs` | ✅ Upstream |

**Not candidates (neuralSpring-specific — stay in neuralSpring):**
`ipa_scores_f64`, `triangle_attention_f64`, `triangle_mul_*`,
`backbone_update_f64`, `torsion_angles_f64`, `msa_*_attention_scores_f64`
— protein folding and MSA-specific shaders, not general math primitives.

**Count reconciliation:** 29 (primalSpring total) = 18 candidates + 6 neuralSpring-specific
+ wildcard expansion (triangle_mul has 3 variants, msa has 2).

---

## Numerical Stability Notes (from springs)

- **f32 accumulation bias**: Green-Kubo gives ~28% bias — use f64/DF64 for reductions
- **GPU transcendental precision**: one tier looser than CPU per transcendental call
- **GPU NaN source**: division-by-zero — mitigated by `eps::SAFE_DIV` (item 2)
- **NVK/NAK f64**: unreliable on Titan V, RTX 4070 — mitigated by `has_reliable_f64()` (item 1)
- **DF64 Yukawa on NVK**: 300–900 steps/s vs 29 native — requires DF64 compilation path
- **NVK shared-memory f64**: returns zeros for `var<workgroup>` f64 accumulators — mitigated by DF64 reduce shaders (item AA)

## References

- `wateringHole/handoffs/BARRACUDA_V034_DEEP_CLEANUP_SPRINT4_HANDOFF_MAR09_2026.md`
- `wateringHole/handoffs/BARRACUDA_V034_CROSS_SPRING_ABSORPTION_SPRINT3_HANDOFF_MAR09_2026.md`
- `wateringHole/handoffs/BARRACUDA_V033_CROSS_SPRING_ABSORPTION_SPRINT2_HANDOFF_MAR09_2026.md`
- `wateringHole/handoffs/BARRACUDA_V033_HEALTHSPRING_HOTSPRING_ABSORPTION_HANDOFF_MAR09_2026.md`
- `wateringHole/handoffs/BARRACUDA_V033_CROSS_SPRING_ABSORPTION_HANDOFF_MAR08_2026.md`
- `wateringHole/handoffs/CORALREEF_PHASE10_CROSS_SPRING_REWIRE_HANDOFF_MAR07_2026.md`
- `wateringHole/handoffs/BARRACUDA_V033_SPRING_ABSORPTION_DEEP_DEBT_HANDOFF_MAR06_2026.md`
- `wateringHole/handoffs/HOTSPRING_SCIENCE_LADDER_BARRACUDA_ABSORPTION_MAR06_2026.md`
- `wateringHole/handoffs/HOTSPRING_V0619_CROSS_SPRING_EVOLUTION_HANDOFF_MAR06_2026.md`
- `wateringHole/handoffs/GROUNDSPRING_V76_TOADSTOOL_BARRACUDA_ABSORPTION_HANDOFF_MAR05_2026.md`
- `wateringHole/handoffs/GROUNDSPRING_V80_FUSED_OPS_BARRACUDA_CATCHUP_HANDOFF_MAR05_2026.md`
- `wateringHole/handoffs/HOTSPRING_VERLET_EVOLUTION_HANDOFF_MAR05_2026.md`
- `wateringHole/handoffs/WETSPRING_V95_BARRACUDA_TOADSTOOL_EVOLUTION_HANDOFF_MAR04_2026.md`
- `wateringHole/handoffs/TOADSTOOL_S97_SPRING_ABSORPTION_HANDOFF_MAR06_2026.md`

// SPDX-License-Identifier: AGPL-3.0-or-later
//! Shader provenance registry and cross-spring dependency queries.

use std::collections::HashMap;
use std::sync::LazyLock;

use super::types::{EvolutionEvent, ShaderCategory as C, ShaderRecord, SpringDomain};

use SpringDomain as SD;

/// The canonical cross-spring shader provenance registry.
///
/// Each entry records origin, consumers, and evolution context. This enables
/// runtime queries like "which shaders did hotSpring contribute?" or "which
/// springs benefit from neuralSpring's statistics work?"
pub static REGISTRY: LazyLock<Vec<ShaderRecord>> = LazyLock::new(|| {
    vec![
        // ── Math Libraries (hotSpring → all) ────────────────────────
        ShaderRecord {
            path: "math/df64_core.wgsl",
            origin: SD::HOT_SPRING,
            consumers: &[
                SD::HOT_SPRING,
                SD::WET_SPRING,
                SD::NEURAL_SPRING,
                SD::AIR_SPRING,
                SD::GROUND_SPRING,
            ],
            category: C::MathLibrary,
            evolution_note: "FP32-pair arithmetic from nuclear physics core-streaming (S58). \
                            Unleashes FP32 cores for f64-class work on consumer GPUs.",
            created: "Feb 2026 hotSpring S58",
            absorbed: "Mar 2026 barraCuda v0.3.0",
        },
        ShaderRecord {
            path: "math/df64_transcendentals.wgsl",
            origin: SD::HOT_SPRING,
            consumers: &[
                SD::HOT_SPRING,
                SD::WET_SPRING,
                SD::NEURAL_SPRING,
                SD::GROUND_SPRING,
            ],
            category: C::MathLibrary,
            evolution_note: "DF64 exp/log/sin/cos for consumer GPUs where native f64 transcendentals fail.",
            created: "Feb 2026 hotSpring S60",
            absorbed: "Mar 2026 barraCuda v0.3.1",
        },
        ShaderRecord {
            path: "math/su3.wgsl",
            origin: SD::HOT_SPRING,
            consumers: &[SD::HOT_SPRING],
            category: C::MathLibrary,
            evolution_note: "SU(3) matrix algebra for lattice QCD.",
            created: "Feb 2026 hotSpring S46",
            absorbed: "Mar 2026 barraCuda v0.3.0",
        },
        ShaderRecord {
            path: "math/su3_df64.wgsl",
            origin: SD::HOT_SPRING,
            consumers: &[SD::HOT_SPRING],
            category: C::MathLibrary,
            evolution_note: "DF64 SU(3) for consumer GPU lattice QCD.",
            created: "Feb 2026 hotSpring S62",
            absorbed: "Mar 2026 barraCuda v0.3.1",
        },
        ShaderRecord {
            path: "math/complex_f64.wgsl",
            origin: SD::HOT_SPRING,
            consumers: &[SD::HOT_SPRING, SD::GROUND_SPRING],
            category: C::MathLibrary,
            evolution_note: "f64 complex arithmetic; shared by lattice QCD and condensed matter.",
            created: "Feb 2026 hotSpring S46",
            absorbed: "Mar 2026 barraCuda v0.3.0",
        },
        // ── Lattice QCD (hotSpring) ─────────────────────────────────
        ShaderRecord {
            path: "lattice/wilson_plaquette_df64.wgsl",
            origin: SD::HOT_SPRING,
            consumers: &[SD::HOT_SPRING],
            category: C::LatticeQcd,
            evolution_note: "DF64 Wilson plaquette from core-streaming discovery. Production wiring via compute.dispatch capability.",
            created: "Feb 2026 hotSpring S58",
            absorbed: "Mar 2026 barraCuda v0.3.0",
        },
        ShaderRecord {
            path: "lattice/su3_gauge_force_df64.wgsl",
            origin: SD::HOT_SPRING,
            consumers: &[SD::HOT_SPRING],
            category: C::LatticeQcd,
            evolution_note: "DF64 gauge force with neighbor-buffer indexing.",
            created: "Feb 2026 hotSpring S58",
            absorbed: "Mar 2026 barraCuda v0.3.0",
        },
        ShaderRecord {
            path: "lattice/cg_kernels_f64.wgsl",
            origin: SD::HOT_SPRING,
            consumers: &[SD::HOT_SPRING, SD::NEURAL_SPRING],
            category: C::LatticeQcd,
            evolution_note: "CG solver with shared memory barriers. Iterative pattern adopted by \
                            neuralSpring for attention convergence loops.",
            created: "Feb 2026 hotSpring S46",
            absorbed: "Mar 2026 barraCuda v0.3.0",
        },
        // ── Molecular Dynamics (hotSpring → wetSpring) ──────────────
        ShaderRecord {
            path: "md/stress_virial_f64.wgsl",
            origin: SD::HOT_SPRING,
            consumers: &[SD::HOT_SPRING, SD::WET_SPRING],
            category: C::MolecularDynamics,
            evolution_note: "Stress tensor via virial theorem. wetSpring uses it for \
                            bio-material mechanical property validation.",
            created: "Feb 2026 hotSpring S50",
            absorbed: "Mar 2026 barraCuda v0.3.1",
        },
        ShaderRecord {
            path: "md/verlet_neighbor_f64.wgsl",
            origin: SD::HOT_SPRING,
            consumers: &[SD::HOT_SPRING, SD::WET_SPRING],
            category: C::MolecularDynamics,
            evolution_note: "Verlet neighbor list for MD force calculations. Shared between \
                            hotSpring nuclear MD and wetSpring bio-molecular pipelines.",
            created: "Mar 2026 hotSpring V0619",
            absorbed: "Mar 2026 barraCuda v0.3.5",
        },
        // ── Statistics (neuralSpring → multiple) ────────────────────
        ShaderRecord {
            path: "stats/matrix_correlation_f64.wgsl",
            origin: SD::NEURAL_SPRING,
            consumers: &[SD::NEURAL_SPRING, SD::GROUND_SPRING, SD::HOT_SPRING],
            category: C::Statistics,
            evolution_note: "Pearson correlation matrix for multi-variate validation. \
                            hotSpring nuclear fits, groundSpring noise validation.",
            created: "Feb 2026 neuralSpring S69",
            absorbed: "Mar 2026 barraCuda v0.3.0",
        },
        ShaderRecord {
            path: "stats/linear_regression_f64.wgsl",
            origin: SD::NEURAL_SPRING,
            consumers: &[SD::NEURAL_SPRING, SD::AIR_SPRING],
            category: C::Statistics,
            evolution_note: "Batched OLS from ML; adopted by airSpring for trend analysis.",
            created: "Feb 2026 neuralSpring S69",
            absorbed: "Mar 2026 barraCuda v0.3.0",
        },
        ShaderRecord {
            path: "special/fused_kl_divergence_f64.wgsl",
            origin: SD::NEURAL_SPRING,
            consumers: &[SD::NEURAL_SPRING, SD::WET_SPRING, SD::GROUND_SPRING],
            category: C::Statistics,
            evolution_note: "KL divergence for ML validation → wetSpring cross-entropy → \
                            groundSpring fitness scoring.",
            created: "Feb 2026 neuralSpring S100",
            absorbed: "Mar 2026 barraCuda v0.3.2",
        },
        ShaderRecord {
            path: "special/fused_chi_squared_f64.wgsl",
            origin: SD::NEURAL_SPRING,
            consumers: &[SD::NEURAL_SPRING, SD::HOT_SPRING, SD::WET_SPRING],
            category: C::Statistics,
            evolution_note: "Chi-squared for ML validation → hotSpring nuclear χ² fits → \
                            wetSpring enrichment testing.",
            created: "Feb 2026 neuralSpring S100",
            absorbed: "Mar 2026 barraCuda v0.3.2",
        },
        ShaderRecord {
            path: "spectral/batch_ipr_f64.wgsl",
            origin: SD::NEURAL_SPRING,
            consumers: &[SD::NEURAL_SPRING, SD::HOT_SPRING],
            category: C::Statistics,
            evolution_note: "Inverse participation ratio for eigenstate localization. \
                            hotSpring spectral diagnostics.",
            created: "Mar 2026 neuralSpring V128",
            absorbed: "Mar 2026 barraCuda v0.3.5",
        },
        // ── Bioinformatics (wetSpring) ──────────────────────────────
        ShaderRecord {
            path: "bio/smith_waterman_banded_f64.wgsl",
            origin: SD::WET_SPRING,
            consumers: &[SD::WET_SPRING, SD::NEURAL_SPRING],
            category: C::Bioinformatics,
            evolution_note: "Banded Smith-Waterman for metagenomics; neuralSpring protein folding.",
            created: "Feb 2026 wetSpring V87",
            absorbed: "Mar 2026 barraCuda v0.3.1",
        },
        ShaderRecord {
            path: "bio/felsenstein_f64.wgsl",
            origin: SD::WET_SPRING,
            consumers: &[SD::WET_SPRING],
            category: C::Bioinformatics,
            evolution_note: "Phylogenetic likelihood from metagenomics pipeline.",
            created: "Feb 2026 wetSpring V87",
            absorbed: "Mar 2026 barraCuda v0.3.1",
        },
        ShaderRecord {
            path: "bio/gillespie_ssa_f64.wgsl",
            origin: SD::WET_SPRING,
            consumers: &[SD::WET_SPRING, SD::NEURAL_SPRING],
            category: C::Bioinformatics,
            evolution_note: "Stochastic simulation algorithm; neuralSpring evolutionary dynamics.",
            created: "Feb 2026 wetSpring V90",
            absorbed: "Mar 2026 barraCuda v0.3.2",
        },
        ShaderRecord {
            path: "bio/hmm_forward_f64.wgsl",
            origin: SD::WET_SPRING,
            consumers: &[SD::WET_SPRING, SD::NEURAL_SPRING],
            category: C::Bioinformatics,
            evolution_note: "HMM forward/backward in log-domain. neuralSpring batched inference.",
            created: "Feb 2026 wetSpring V90",
            absorbed: "Mar 2026 barraCuda v0.3.2",
        },
        ShaderRecord {
            path: "reduce/fused_map_reduce_f64.wgsl",
            origin: SD::WET_SPRING,
            consumers: &[SD::WET_SPRING, SD::AIR_SPRING, SD::HOT_SPRING],
            category: C::Primitives,
            evolution_note: "Shannon/Simpson map-reduce → airSpring batch sums → hotSpring observable stats.",
            created: "Feb 2026 wetSpring V87",
            absorbed: "Mar 2026 barraCuda v0.3.1",
        },
        // ── Hydrology (airSpring) ───────────────────────────────────
        ShaderRecord {
            path: "grid/hargreaves_et0_f64.wgsl",
            origin: SD::AIR_SPRING,
            consumers: &[SD::AIR_SPRING, SD::WET_SPRING],
            category: C::Hydrology,
            evolution_note: "Hargreaves ET₀ reference evapotranspiration for agriculture.",
            created: "Feb 2026 airSpring V043",
            absorbed: "Mar 2026 barraCuda v0.3.2",
        },
        ShaderRecord {
            path: "science/seasonal_pipeline.wgsl",
            origin: SD::AIR_SPRING,
            consumers: &[SD::AIR_SPRING, SD::WET_SPRING],
            category: C::Hydrology,
            evolution_note: "Seasonal FAO-56 pipeline → wetSpring environmental monitoring.",
            created: "Feb 2026 airSpring V043",
            absorbed: "Mar 2026 barraCuda v0.3.2",
        },
        // ── Condensed Matter (groundSpring) ─────────────────────────
        ShaderRecord {
            path: "spectral/anderson_lyapunov_f64.wgsl",
            origin: SD::GROUND_SPRING,
            consumers: &[SD::GROUND_SPRING, SD::HOT_SPRING, SD::NEURAL_SPRING],
            category: C::CondensedMatter,
            evolution_note: "Anderson localization via transfer-matrix Lyapunov exponent. \
                            hotSpring spectral diagnostics, neuralSpring disorder sweeps.",
            created: "Mar 2026 groundSpring V74",
            absorbed: "Mar 2026 barraCuda v0.3.5",
        },
        ShaderRecord {
            path: "special/chi_squared_f64.wgsl",
            origin: SD::GROUND_SPRING,
            consumers: &[
                SD::GROUND_SPRING,
                SD::HOT_SPRING,
                SD::WET_SPRING,
                SD::NEURAL_SPRING,
                SD::AIR_SPRING,
            ],
            category: C::Statistics,
            evolution_note: "Chi-squared CDF+quantile from V74. Universal statistical test for all springs.",
            created: "Mar 2026 groundSpring V74",
            absorbed: "Mar 2026 barraCuda v0.3.5",
        },
        // ── ML / ESN (hotSpring → wetSpring) ────────────────────────
        ShaderRecord {
            path: "ml/esn_readout_f64.wgsl",
            origin: SD::HOT_SPRING,
            consumers: &[SD::HOT_SPRING, SD::WET_SPRING],
            category: C::MachineLearning,
            evolution_note: "ESN readout from Stanton-Murillo transport predictions. \
                            wetSpring environmental time-series.",
            created: "Feb 2026 hotSpring S46",
            absorbed: "Mar 2026 barraCuda v0.3.0",
        },
        ShaderRecord {
            path: "stats/moving_window_f64.wgsl",
            origin: SD::AIR_SPRING,
            consumers: &[SD::AIR_SPRING, SD::WET_SPRING, SD::NEURAL_SPRING],
            category: C::Statistics,
            evolution_note: "Sliding window stats for IoT sensor streams. neuralSpring streaming inference.",
            created: "Mar 2026 airSpring V068",
            absorbed: "Mar 2026 barraCuda v0.3.5",
        },
        // ── Nuclear Physics (hotSpring) ─────────────────────────────
        ShaderRecord {
            path: "nuclear/hfb_gradient_f64.wgsl",
            origin: SD::HOT_SPRING,
            consumers: &[SD::HOT_SPRING],
            category: C::NuclearPhysics,
            evolution_note: "Hartree-Fock-Bogoliubov gradient kernel from nuclear structure ladder.",
            created: "Feb 2026 hotSpring S52",
            absorbed: "Mar 2026 barraCuda v0.3.1",
        },
        ShaderRecord {
            path: "reduce/welford_mean_variance_f64.wgsl",
            origin: SD::GROUND_SPRING,
            consumers: &[
                SD::GROUND_SPRING,
                SD::HOT_SPRING,
                SD::WET_SPRING,
                SD::NEURAL_SPRING,
                SD::AIR_SPRING,
            ],
            category: C::Primitives,
            evolution_note: "Welford single-pass fused mean+variance. Universal reduction primitive \
                            for all springs' GPU statistics.",
            created: "Mar 2026 groundSpring V80",
            absorbed: "Mar 2026 barraCuda v0.3.5",
        },
        // ── Plasma Physics (hotSpring Chuna Papers 43-45) ───────────
        ShaderRecord {
            path: "science/plasma/dielectric_mermin_f64.wgsl",
            origin: SD::HOT_SPRING,
            consumers: &[SD::HOT_SPRING],
            category: C::PlasmaPhysics,
            evolution_note: "Mermin dielectric ε(k,ω) with plasma dispersion function. \
                            Completed Mermin (momentum-conserving) variant from Chuna & Murillo (2024). \
                            Depends on complex_f64.wgsl.",
            created: "Mar 2026 hotSpring v0.6.23 (Chuna P44)",
            absorbed: "Mar 2026 barraCuda v0.3.5",
        },
        ShaderRecord {
            path: "science/plasma/dielectric_multicomponent_f64.wgsl",
            origin: SD::HOT_SPRING,
            consumers: &[SD::HOT_SPRING],
            category: C::PlasmaPhysics,
            evolution_note: "Multi-species Mermin dielectric with per-species susceptibility. \
                            Species layout: [mass, charge, density, temp, nu, v_th, k_debye] × N. \
                            Uses cscale() for correct complex scalar multiplication.",
            created: "Mar 2026 hotSpring v0.6.23 (Chuna P44)",
            absorbed: "Mar 2026 barraCuda v0.3.5",
        },
        ShaderRecord {
            path: "science/plasma/bgk_relaxation_f64.wgsl",
            origin: SD::HOT_SPRING,
            consumers: &[SD::HOT_SPRING],
            category: C::PlasmaPhysics,
            evolution_note: "Two-pass BGK relaxation for multi-species kinetic plasma. \
                            Pass 1: velocity-space moments. Pass 2: relax f toward Maxwellian. \
                            CPU reduces moments between passes (WGSL lacks f64 atomics).",
            created: "Mar 2026 hotSpring v0.6.23 (Chuna P45)",
            absorbed: "Mar 2026 barraCuda v0.3.5",
        },
        ShaderRecord {
            path: "science/plasma/euler_hll_f64.wgsl",
            origin: SD::HOT_SPRING,
            consumers: &[SD::HOT_SPRING],
            category: C::PlasmaPhysics,
            evolution_note: "1D Euler fluid with HLL approximate Riemann solver. \
                            Two-pass: compute HLL flux at interfaces, then conservative update. \
                            For kinetic-fluid coupling pipeline (Chuna P45).",
            created: "Mar 2026 hotSpring v0.6.23 (Chuna P45)",
            absorbed: "Mar 2026 barraCuda v0.3.5",
        },
    ]
});

/// The canonical cross-spring evolution timeline.
///
/// Key moments when a spring's work evolved to benefit other springs,
/// ordered chronologically. This is the programmatic fossil record of how
/// hotSpring precision shaders enabled wetSpring bio pipelines,
/// neuralSpring ML patterns enriched groundSpring validation, etc.
pub static EVOLUTION_TIMELINE: LazyLock<Vec<EvolutionEvent>> = LazyLock::new(|| {
    vec![
        EvolutionEvent {
            date: "Feb 2026 (S46-S48)",
            from: SD::HOT_SPRING,
            beneficiaries: &[
                SD::WET_SPRING,
                SD::NEURAL_SPRING,
                SD::GROUND_SPRING,
                SD::AIR_SPRING,
            ],
            description: "First f64 WGSL shaders: SU(3), CG solver, sum_reduce. \
                         Established the shader-first f64 pattern all springs now follow.",
        },
        EvolutionEvent {
            date: "Feb 2026 (S58)",
            from: SD::HOT_SPRING,
            beneficiaries: &[
                SD::WET_SPRING,
                SD::NEURAL_SPRING,
                SD::GROUND_SPRING,
                SD::AIR_SPRING,
            ],
            description: "DF64 core-streaming: FP32-pair arithmetic unleashed consumer \
                         GPUs for f64-class work. The single most impactful cross-spring \
                         contribution — every spring's GPU stats now run on any hardware.",
        },
        EvolutionEvent {
            date: "Feb 2026 (S69)",
            from: SD::NEURAL_SPRING,
            beneficiaries: &[SD::HOT_SPRING, SD::GROUND_SPRING, SD::AIR_SPRING],
            description: "matrix_correlation + linear_regression f64 shaders from ML \
                         validation: hotSpring nuclear fits, groundSpring noise validation, \
                         airSpring trend analysis.",
        },
        EvolutionEvent {
            date: "Feb 2026 (V87-V90)",
            from: SD::WET_SPRING,
            beneficiaries: &[SD::NEURAL_SPRING, SD::HOT_SPRING, SD::AIR_SPRING],
            description: "Bio shaders: Smith-Waterman, Felsenstein, Gillespie SSA, HMM, \
                         fused_map_reduce. neuralSpring adopted HMM for batched inference \
                         and Gillespie for evolutionary dynamics.",
        },
        EvolutionEvent {
            date: "Feb 2026 (S100)",
            from: SD::NEURAL_SPRING,
            beneficiaries: &[SD::HOT_SPRING, SD::WET_SPRING, SD::GROUND_SPRING],
            description: "KL divergence + chi-squared fused shaders from ML: \
                         hotSpring nuclear chi-squared fits, wetSpring enrichment testing, \
                         groundSpring fitness scoring.",
        },
        EvolutionEvent {
            date: "Mar 2026 (V043-V068)",
            from: SD::AIR_SPRING,
            beneficiaries: &[SD::WET_SPRING, SD::NEURAL_SPRING],
            description: "Hydrology shaders: Hargreaves ET₀, seasonal pipeline, \
                         moving_window_f64. wetSpring environmental monitoring, \
                         neuralSpring streaming inference windows.",
        },
        EvolutionEvent {
            date: "Mar 2026 (V74-V80)",
            from: SD::GROUND_SPRING,
            beneficiaries: &[
                SD::HOT_SPRING,
                SD::WET_SPRING,
                SD::NEURAL_SPRING,
                SD::AIR_SPRING,
            ],
            description: "Universal primitives: chi-squared CDF+quantile, Anderson \
                         Lyapunov, 13-tier tolerance, Welford fused mean+variance. \
                         The tolerance framework became the validation backbone for all springs.",
        },
        EvolutionEvent {
            date: "Mar 2026 (V128)",
            from: SD::NEURAL_SPRING,
            beneficiaries: &[SD::HOT_SPRING],
            description: "batch_ipr_f64: inverse participation ratio for eigenstate \
                         localization. hotSpring spectral diagnostics for Anderson model.",
        },
        EvolutionEvent {
            date: "Mar 2026 (V0619)",
            from: SD::HOT_SPRING,
            beneficiaries: &[SD::WET_SPRING],
            description: "Verlet neighbor list and stress virial shaders: wetSpring \
                         bio-material mechanical property validation using MD patterns.",
        },
        EvolutionEvent {
            date: "Mar 7, 2026",
            from: SpringDomain::BARRACUDA,
            beneficiaries: &[
                SD::HOT_SPRING,
                SD::WET_SPRING,
                SD::NEURAL_SPRING,
                SD::AIR_SPRING,
                SD::GROUND_SPRING,
            ],
            description: "Provenance registry formalized: all cross-spring flows now \
                         tracked programmatically with Write → Absorb → Lean lifecycle, \
                         evolution dates, and bidirectional dependency matrix.",
        },
    ]
});

/// Query all shaders from a specific spring domain.
#[must_use]
pub fn shaders_from(origin: SpringDomain) -> Vec<&'static ShaderRecord> {
    REGISTRY.iter().filter(|r| r.origin == origin).collect()
}

/// Query all shaders consumed by a specific spring domain.
#[must_use]
pub fn shaders_consumed_by(consumer: SpringDomain) -> Vec<&'static ShaderRecord> {
    REGISTRY
        .iter()
        .filter(|r| r.consumers.contains(&consumer))
        .collect()
}

/// Query cross-spring shaders — those consumed by a spring other than the origin.
#[must_use]
pub fn cross_spring_shaders() -> Vec<&'static ShaderRecord> {
    REGISTRY
        .iter()
        .filter(|r| r.consumers.iter().any(|c| *c != r.origin))
        .collect()
}

/// Build a spring-to-spring dependency map: `(from, to)` → count of shared shaders.
#[must_use]
pub fn cross_spring_matrix() -> HashMap<(SpringDomain, SpringDomain), usize> {
    let mut matrix = HashMap::new();
    for record in REGISTRY.iter() {
        for consumer in record.consumers {
            if *consumer != record.origin {
                *matrix.entry((record.origin, *consumer)).or_insert(0) += 1;
            }
        }
    }
    matrix
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_is_populated() {
        assert!(REGISTRY.len() >= 27, "expected 27+ tracked shaders");
    }

    #[test]
    fn all_records_have_dates() {
        for r in REGISTRY.iter() {
            assert!(!r.created.is_empty(), "{} missing created date", r.path);
            assert!(!r.absorbed.is_empty(), "{} missing absorbed date", r.path);
        }
    }

    #[test]
    fn hotspring_has_most_contributions() {
        let hot = shaders_from(SD::HOT_SPRING);
        assert!(hot.len() >= 8, "hotSpring should have 8+ origin shaders");
    }

    #[test]
    fn wetspring_bio_shaders_consumed_by_neuralspring() {
        let wet = shaders_from(SD::WET_SPRING);
        let consumed_by_neural: Vec<_> = wet
            .iter()
            .filter(|r| r.consumers.contains(&SD::NEURAL_SPRING))
            .collect();
        assert!(
            consumed_by_neural.len() >= 3,
            "neuralSpring should consume 3+ wetSpring bio shaders, got {}",
            consumed_by_neural.len()
        );
    }

    #[test]
    fn neuralspring_bidirectional_flow() {
        let from_neural = shaders_from(SD::NEURAL_SPRING);
        let consumed_by_neural = shaders_consumed_by(SD::NEURAL_SPRING);

        assert!(from_neural.len() >= 4, "neuralSpring writes 4+ shaders");
        assert!(
            consumed_by_neural.len() >= 8,
            "neuralSpring consumes 8+ shaders from other springs"
        );

        let external_origins: Vec<_> = consumed_by_neural
            .iter()
            .filter(|r| r.origin != SD::NEURAL_SPRING)
            .map(|r| r.origin)
            .collect();
        assert!(external_origins.contains(&SD::HOT_SPRING));
        assert!(external_origins.contains(&SD::WET_SPRING));
    }

    #[test]
    fn df64_core_consumed_by_all_springs() {
        let df64 = REGISTRY.iter().find(|r| r.path == "math/df64_core.wgsl");
        assert!(df64.is_some());
        let df64 = df64.unwrap();
        assert_eq!(df64.consumers.len(), 5);
    }

    #[test]
    fn cross_spring_shaders_exist() {
        let cross = cross_spring_shaders();
        assert!(cross.len() > 15, "expected 15+ cross-spring shaders");
    }

    #[test]
    fn cross_spring_matrix_non_empty() {
        let matrix = cross_spring_matrix();
        assert!(!matrix.is_empty());

        let hot_to_neural = matrix
            .get(&(SD::HOT_SPRING, SD::NEURAL_SPRING))
            .copied()
            .unwrap_or(0);
        assert!(
            hot_to_neural >= 2,
            "hotSpring→neuralSpring should share 2+ shaders"
        );

        let hot_to_wet = matrix
            .get(&(SD::HOT_SPRING, SD::WET_SPRING))
            .copied()
            .unwrap_or(0);
        assert!(
            hot_to_wet >= 3,
            "hotSpring→wetSpring should share 3+ shaders"
        );

        let wet_to_neural = matrix
            .get(&(SD::WET_SPRING, SD::NEURAL_SPRING))
            .copied()
            .unwrap_or(0);
        assert!(
            wet_to_neural >= 3,
            "wetSpring→neuralSpring should share 3+ shaders"
        );
    }

    #[test]
    fn kl_divergence_flows_from_neural_to_wet_and_ground() {
        let kl = REGISTRY.iter().find(|r| r.path.contains("kl_divergence"));
        assert!(kl.is_some());
        let kl = kl.unwrap();
        assert_eq!(kl.origin, SD::NEURAL_SPRING);
        assert!(kl.consumers.contains(&SD::WET_SPRING));
        assert!(kl.consumers.contains(&SD::GROUND_SPRING));
    }

    #[test]
    fn consumed_by_query_works() {
        let wet_shaders = shaders_consumed_by(SD::WET_SPRING);
        assert!(wet_shaders.len() >= 8, "wetSpring consumes 8+ shaders");
        let origins: Vec<_> = wet_shaders.iter().map(|r| r.origin).collect();
        assert!(origins.contains(&SD::HOT_SPRING));
        assert!(origins.contains(&SD::AIR_SPRING));
    }

    #[test]
    fn evolution_timeline_populated() {
        assert!(
            EVOLUTION_TIMELINE.len() >= 10,
            "expected 10+ evolution events"
        );
    }

    #[test]
    fn welford_consumed_by_all_springs() {
        let welford = REGISTRY
            .iter()
            .find(|r| r.path.contains("welford_mean_variance"));
        assert!(welford.is_some());
        let w = welford.unwrap();
        assert_eq!(
            w.consumers.len(),
            5,
            "Welford should be consumed by all 5 springs"
        );
    }
}

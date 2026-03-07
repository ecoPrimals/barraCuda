// SPDX-License-Identifier: AGPL-3.0-or-later
//! Shader provenance tracking — cross-spring evolution registry.
//!
//! Tracks which spring domain each WGSL shader originated from, which
//! springs currently consume it, and how patterns flow across domains.
//! This is the programmatic equivalent of the provenance comments in shader
//! headers, enabling runtime introspection and evolution auditing.
//!
//! # Cross-Spring Evolution (Write → Absorb → Lean)
//!
//! Shaders flow between springs following the ecosystem pattern:
//! - **Write**: a spring creates a domain-specific shader
//! - **Absorb**: barraCuda generalises it as a reusable primitive
//! - **Lean**: other springs consume the upstream version
//!
//! ## Evolution examples
//!
//! - **hotSpring precision → all springs**: `df64_core.wgsl` from nuclear
//!   physics became the universal FP32-pair arithmetic library
//! - **neuralSpring stats → wetSpring + groundSpring**: `kl_divergence_f64`
//!   for ML validation absorbed by bio-informatics and condensed matter
//! - **hotSpring MD → wetSpring**: `stress_virial_f64` used for mechanical
//!   property validation in bio-material pipelines

use std::collections::HashMap;
use std::sync::LazyLock;

/// Which spring domain originated or primarily uses a shader.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SpringDomain {
    /// Nuclear physics, lattice QCD, molecular dynamics
    HotSpring,
    /// Metagenomics, bioinformatics, phylogenetics
    WetSpring,
    /// Machine learning, attention, neuroevolution
    NeuralSpring,
    /// Agriculture, hydrology, evapotranspiration
    AirSpring,
    /// Condensed matter, Anderson localization, noise validation
    GroundSpring,
    /// Internal barraCuda primitive (no spring origin)
    BarraCuda,
}

impl std::fmt::Display for SpringDomain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::HotSpring => write!(f, "hotSpring"),
            Self::WetSpring => write!(f, "wetSpring"),
            Self::NeuralSpring => write!(f, "neuralSpring"),
            Self::AirSpring => write!(f, "airSpring"),
            Self::GroundSpring => write!(f, "groundSpring"),
            Self::BarraCuda => write!(f, "barraCuda"),
        }
    }
}

/// Provenance record for a single shader.
#[derive(Debug, Clone)]
pub struct ShaderRecord {
    /// Shader path relative to `shaders/` (e.g. `"math/df64_core.wgsl"`)
    pub path: &'static str,
    /// Spring that originally created this shader
    pub origin: SpringDomain,
    /// Springs that currently consume or reference this shader
    pub consumers: &'static [SpringDomain],
    /// Broad category for grouping
    pub category: ShaderCategory,
    /// Brief description of cross-spring evolution
    pub evolution_note: &'static str,
}

/// Shader category for grouping.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ShaderCategory {
    /// Foundational math libraries (df64, su3, complex)
    MathLibrary,
    /// Lattice QCD and gauge theory
    LatticeQcd,
    /// Molecular dynamics forces, integrators, observables
    MolecularDynamics,
    /// Nuclear physics (HFB, SEMF, deformed)
    NuclearPhysics,
    /// Statistics and correlation
    Statistics,
    /// Machine learning and neural networks
    MachineLearning,
    /// Bioinformatics and genomics
    Bioinformatics,
    /// Hydrology and earth science
    Hydrology,
    /// Condensed matter physics
    CondensedMatter,
    /// General compute primitives
    Primitives,
}

impl std::fmt::Display for ShaderCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MathLibrary => write!(f, "Math Library"),
            Self::LatticeQcd => write!(f, "Lattice QCD"),
            Self::MolecularDynamics => write!(f, "Molecular Dynamics"),
            Self::NuclearPhysics => write!(f, "Nuclear Physics"),
            Self::Statistics => write!(f, "Statistics"),
            Self::MachineLearning => write!(f, "Machine Learning"),
            Self::Bioinformatics => write!(f, "Bioinformatics"),
            Self::Hydrology => write!(f, "Hydrology"),
            Self::CondensedMatter => write!(f, "Condensed Matter"),
            Self::Primitives => write!(f, "Primitives"),
        }
    }
}

use ShaderCategory as C;
use SpringDomain::{AirSpring, GroundSpring, HotSpring, NeuralSpring, WetSpring};

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
            origin: HotSpring,
            consumers: &[HotSpring, WetSpring, NeuralSpring, AirSpring, GroundSpring],
            category: C::MathLibrary,
            evolution_note: "FP32-pair arithmetic from nuclear physics core-streaming (S58). \
                            Unleashes FP32 cores for f64-class work on consumer GPUs.",
        },
        ShaderRecord {
            path: "math/df64_transcendentals.wgsl",
            origin: HotSpring,
            consumers: &[HotSpring, WetSpring, NeuralSpring, GroundSpring],
            category: C::MathLibrary,
            evolution_note: "DF64 exp/log/sin/cos for consumer GPUs where native f64 transcendentals fail.",
        },
        ShaderRecord {
            path: "math/su3.wgsl",
            origin: HotSpring,
            consumers: &[HotSpring],
            category: C::MathLibrary,
            evolution_note: "SU(3) matrix algebra for lattice QCD.",
        },
        ShaderRecord {
            path: "math/su3_df64.wgsl",
            origin: HotSpring,
            consumers: &[HotSpring],
            category: C::MathLibrary,
            evolution_note: "DF64 SU(3) for consumer GPU lattice QCD.",
        },
        ShaderRecord {
            path: "math/complex_f64.wgsl",
            origin: HotSpring,
            consumers: &[HotSpring, GroundSpring],
            category: C::MathLibrary,
            evolution_note: "f64 complex arithmetic; shared by lattice QCD and condensed matter.",
        },
        // ── Lattice QCD (hotSpring) ─────────────────────────────────
        ShaderRecord {
            path: "lattice/wilson_plaquette_df64.wgsl",
            origin: HotSpring,
            consumers: &[HotSpring],
            category: C::LatticeQcd,
            evolution_note: "DF64 Wilson plaquette from core-streaming discovery. Production wiring via toadStool.",
        },
        ShaderRecord {
            path: "lattice/su3_gauge_force_df64.wgsl",
            origin: HotSpring,
            consumers: &[HotSpring],
            category: C::LatticeQcd,
            evolution_note: "DF64 gauge force with neighbor-buffer indexing.",
        },
        ShaderRecord {
            path: "lattice/cg_kernels_f64.wgsl",
            origin: HotSpring,
            consumers: &[HotSpring, NeuralSpring],
            category: C::LatticeQcd,
            evolution_note: "CG solver with shared memory barriers. Iterative pattern adopted by neuralSpring.",
        },
        // ── Statistics (neuralSpring → multiple) ────────────────────
        ShaderRecord {
            path: "stats/matrix_correlation_f64.wgsl",
            origin: NeuralSpring,
            consumers: &[NeuralSpring, GroundSpring, HotSpring],
            category: C::Statistics,
            evolution_note: "Pearson correlation matrix for multi-variate validation across all springs.",
        },
        ShaderRecord {
            path: "stats/linear_regression_f64.wgsl",
            origin: NeuralSpring,
            consumers: &[NeuralSpring, AirSpring],
            category: C::Statistics,
            evolution_note: "Batched OLS from ML; adopted by airSpring for trend analysis.",
        },
        ShaderRecord {
            path: "special/fused_kl_divergence_f64.wgsl",
            origin: NeuralSpring,
            consumers: &[NeuralSpring, WetSpring, GroundSpring],
            category: C::Statistics,
            evolution_note: "KL divergence for ML validation → wetSpring cross-entropy → groundSpring fitness scoring.",
        },
        ShaderRecord {
            path: "special/fused_chi_squared_f64.wgsl",
            origin: NeuralSpring,
            consumers: &[NeuralSpring, HotSpring, WetSpring],
            category: C::Statistics,
            evolution_note: "Chi-squared test for ML validation → hotSpring nuclear fits → wetSpring enrichment.",
        },
        // ── Bioinformatics (wetSpring) ──────────────────────────────
        ShaderRecord {
            path: "bio/smith_waterman_banded_f64.wgsl",
            origin: WetSpring,
            consumers: &[WetSpring, NeuralSpring],
            category: C::Bioinformatics,
            evolution_note: "Banded Smith-Waterman for metagenomics; referenced by neuralSpring protein folding.",
        },
        ShaderRecord {
            path: "bio/felsenstein_f64.wgsl",
            origin: WetSpring,
            consumers: &[WetSpring],
            category: C::Bioinformatics,
            evolution_note: "Phylogenetic likelihood computation from wetSpring metagenomics pipeline.",
        },
        ShaderRecord {
            path: "bio/gillespie_ssa_f64.wgsl",
            origin: WetSpring,
            consumers: &[WetSpring, NeuralSpring],
            category: C::Bioinformatics,
            evolution_note: "Stochastic simulation algorithm; used by neuralSpring for evolutionary dynamics.",
        },
        ShaderRecord {
            path: "reduce/fused_map_reduce_f64.wgsl",
            origin: WetSpring,
            consumers: &[WetSpring, AirSpring, HotSpring],
            category: C::Primitives,
            evolution_note: "Shannon/Simpson map-reduce pattern → airSpring batch sums → hotSpring observable stats.",
        },
        // ── Hydrology (airSpring) ───────────────────────────────────
        ShaderRecord {
            path: "grid/hargreaves_et0_f64.wgsl",
            origin: AirSpring,
            consumers: &[AirSpring, WetSpring],
            category: C::Hydrology,
            evolution_note: "Hargreaves ET₀ reference evapotranspiration for agriculture.",
        },
        ShaderRecord {
            path: "science/seasonal_pipeline.wgsl",
            origin: AirSpring,
            consumers: &[AirSpring, WetSpring],
            category: C::Hydrology,
            evolution_note: "Seasonal FAO-56 pipeline → wetSpring environmental monitoring.",
        },
        // ── Condensed Matter (groundSpring) ─────────────────────────
        ShaderRecord {
            path: "spectral/anderson_lyapunov_f64.wgsl",
            origin: GroundSpring,
            consumers: &[GroundSpring, HotSpring, NeuralSpring],
            category: C::CondensedMatter,
            evolution_note: "Anderson localization via transfer-matrix Lyapunov exponent. \
                            hotSpring spectral diagnostics, neuralSpring disorder sweeps.",
        },
        ShaderRecord {
            path: "special/chi_squared_f64.wgsl",
            origin: GroundSpring,
            consumers: &[GroundSpring, HotSpring, WetSpring, NeuralSpring, AirSpring],
            category: C::Statistics,
            evolution_note: "Chi-squared CDF+quantile from V74. Universal statistical test for all springs.",
        },
        // ── MD (hotSpring → wetSpring) ──────────────────────────────
        ShaderRecord {
            path: "ml/esn_readout_f64.wgsl",
            origin: HotSpring,
            consumers: &[HotSpring, WetSpring],
            category: C::MachineLearning,
            evolution_note: "ESN readout from Stanton-Murillo transport predictions. \
                            Adopted by wetSpring for environmental time-series.",
        },
        ShaderRecord {
            path: "stats/moving_window_f64.wgsl",
            origin: AirSpring,
            consumers: &[AirSpring, WetSpring],
            category: C::Statistics,
            evolution_note: "Sliding window stats for IoT sensor streams and environmental monitoring.",
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
        assert!(REGISTRY.len() > 20, "expected 20+ tracked shaders");
    }

    #[test]
    fn hotspring_has_most_contributions() {
        let hot = shaders_from(HotSpring);
        assert!(hot.len() >= 5, "hotSpring should have 5+ origin shaders");
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
        assert!(cross.len() > 10, "expected 10+ cross-spring shaders");
    }

    #[test]
    fn cross_spring_matrix_non_empty() {
        let matrix = cross_spring_matrix();
        assert!(!matrix.is_empty());
        let hot_to_neural = matrix.get(&(HotSpring, NeuralSpring)).copied().unwrap_or(0);
        assert!(
            hot_to_neural > 0,
            "hotSpring should share with neuralSpring"
        );
    }

    #[test]
    fn kl_divergence_flows_from_neural_to_wet_and_ground() {
        let kl = REGISTRY.iter().find(|r| r.path.contains("kl_divergence"));
        assert!(kl.is_some());
        let kl = kl.unwrap();
        assert_eq!(kl.origin, NeuralSpring);
        assert!(kl.consumers.contains(&WetSpring));
        assert!(kl.consumers.contains(&GroundSpring));
    }

    #[test]
    fn consumed_by_query_works() {
        let wet_shaders = shaders_consumed_by(WetSpring);
        assert!(wet_shaders.len() >= 5, "wetSpring consumes 5+ shaders");
        let origins: Vec<_> = wet_shaders.iter().map(|r| r.origin).collect();
        assert!(
            origins.contains(&HotSpring),
            "wetSpring should consume hotSpring shaders"
        );
    }

    #[test]
    fn display_spring_domains() {
        assert_eq!(format!("{}", HotSpring), "hotSpring");
        assert_eq!(format!("{}", WetSpring), "wetSpring");
        assert_eq!(format!("{}", NeuralSpring), "neuralSpring");
    }

    #[test]
    fn display_shader_categories() {
        assert_eq!(format!("{}", C::LatticeQcd), "Lattice QCD");
        assert_eq!(format!("{}", C::MathLibrary), "Math Library");
    }
}

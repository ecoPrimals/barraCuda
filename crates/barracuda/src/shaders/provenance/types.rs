// SPDX-License-Identifier: AGPL-3.0-or-later
//! Domain types for cross-spring shader provenance tracking.

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
    /// When this shader was first created (e.g. "Feb 2026 S58")
    pub created: &'static str,
    /// When barraCuda absorbed this shader (e.g. "Mar 2026 v0.3.3")
    pub absorbed: &'static str,
}

/// Key moment in the cross-spring evolution timeline.
#[derive(Debug, Clone)]
pub struct EvolutionEvent {
    /// Date or sprint reference (e.g. "Mar 5, 2026")
    pub date: &'static str,
    /// Which spring initiated this evolution
    pub from: SpringDomain,
    /// Which springs benefited
    pub beneficiaries: &'static [SpringDomain],
    /// What evolved and why it mattered
    pub description: &'static str,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_spring_domains() {
        use SpringDomain::{HotSpring, NeuralSpring, WetSpring};
        assert_eq!(format!("{HotSpring}"), "hotSpring");
        assert_eq!(format!("{WetSpring}"), "wetSpring");
        assert_eq!(format!("{NeuralSpring}"), "neuralSpring");
    }

    #[test]
    fn display_shader_categories() {
        assert_eq!(format!("{}", ShaderCategory::LatticeQcd), "Lattice QCD");
        assert_eq!(format!("{}", ShaderCategory::MathLibrary), "Math Library");
    }
}

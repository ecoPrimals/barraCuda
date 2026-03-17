// SPDX-License-Identifier: AGPL-3.0-only
//! Domain types for cross-spring shader provenance tracking.

/// Capability-based domain identifier for shader provenance.
///
/// String-based rather than enum-based so barraCuda holds no compile-time
/// knowledge of other primals. New domains are runtime-extensible — just
/// construct `SpringDomain("myNewDomain")`. The known constants below are
/// provided for ergonomics and backward compatibility.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SpringDomain(pub &'static str);

impl SpringDomain {
    /// Nuclear physics, lattice QCD, molecular dynamics
    pub const HOT_SPRING: Self = Self("hotSpring");
    /// Metagenomics, bioinformatics, phylogenetics
    pub const WET_SPRING: Self = Self("wetSpring");
    /// Machine learning, attention, neuroevolution
    pub const NEURAL_SPRING: Self = Self("neuralSpring");
    /// Agriculture, hydrology, evapotranspiration
    pub const AIR_SPRING: Self = Self("airSpring");
    /// Condensed matter, Anderson localization, noise validation
    pub const GROUND_SPRING: Self = Self("groundSpring");
    /// Human health, PK/PD, biosignals, microbiome
    pub const HEALTH_SPRING: Self = Self("healthSpring");
    /// Internal barraCuda primitive (no spring origin)
    pub const BARRACUDA: Self = Self("barraCuda");
}

impl std::fmt::Display for SpringDomain {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.0)
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
    /// When barraCuda absorbed this shader (e.g. "Mar 2026 v0.3.5")
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
    /// Plasma physics (dielectric, kinetic, fluid)
    PlasmaPhysics,
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
            Self::PlasmaPhysics => write!(f, "Plasma Physics"),
            Self::Primitives => write!(f, "Primitives"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_spring_domains() {
        let hot = SpringDomain::HOT_SPRING;
        let wet = SpringDomain::WET_SPRING;
        let neural = SpringDomain::NEURAL_SPRING;
        assert_eq!(format!("{hot}"), "hotSpring");
        assert_eq!(format!("{wet}"), "wetSpring");
        assert_eq!(format!("{neural}"), "neuralSpring");
    }

    #[test]
    fn runtime_extensible_domain() {
        let custom = SpringDomain("myCustomDomain");
        assert_eq!(format!("{custom}"), "myCustomDomain");
        assert_ne!(custom, SpringDomain::HOT_SPRING);
    }

    #[test]
    fn display_shader_categories() {
        let lattice = ShaderCategory::LatticeQcd;
        let math = ShaderCategory::MathLibrary;
        assert_eq!(format!("{lattice}"), "Lattice QCD");
        assert_eq!(format!("{math}"), "Math Library");
    }
}

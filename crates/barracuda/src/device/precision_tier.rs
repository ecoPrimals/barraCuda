// SPDX-License-Identifier: AGPL-3.0-or-later

//! Precision tiers and physics domains for GPU shader compilation routing.
//!
//! Absorbed from hotSpring v0.6.25 precision brain architecture. Provides:
//! - [`PrecisionTier`]: compilation-level precision selection (`F32` → `DF64` → `F64` → `F64Precise`)
//! - [`PhysicsDomain`]: physics-aware routing that matches domain precision requirements
//!   to hardware capabilities.
//!
//! # Design
//!
//! Springs express what precision their physics *needs*; the hardware calibration
//! determines what the GPU *can do*. The precision brain intersects these two axes
//! to produce safe, optimal routing decisions.

use serde::{Deserialize, Serialize};

/// Compilation-level precision tier for GPU shaders.
///
/// Ordered from lowest to highest precision. Each tier corresponds to a
/// specific compilation path on [`WgpuDevice`](super::WgpuDevice):
///
/// | Tier | Method | Mantissa bits | Throughput |
/// |------|--------|--------------|------------|
/// | `F32` | `compile_shader()` | 23 | Baseline |
/// | `DF64` | `compile_shader_df64()` | ~48 | ~1× F32 on consumer GPUs |
/// | `F64` | `compile_shader_f64()` | 52 | 1:64 consumer, 1:2 compute |
/// | `F64Precise` | `compile_shader_f64()` + FMA-free | 52 | Slowest |
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PrecisionTier {
    /// 16-bit float: ML inference, screening, tensor core fast path.
    /// Requires `SHADER_F16`; emulated via pack/unpack on older hardware.
    F16,
    /// 32-bit float: screening, preview, throughput-bound work.
    F32,
    /// Double-float emulation (f32-pair): ~14 decimal digits on FP32 cores.
    /// Up to 10× native f64 throughput on consumer GPUs (1:64 FP64:FP32).
    DF64,
    /// Native 64-bit: reference precision, validation. Full hardware f64.
    F64,
    /// Native 64-bit without FMA fusion: precision-critical domains where
    /// `a*b+c` must not be fused (dielectric, eigensolve).
    F64Precise,
}

impl PrecisionTier {
    /// Approximate mantissa bits for this tier.
    #[must_use]
    pub const fn mantissa_bits(self) -> u32 {
        match self {
            Self::F16 => 10,
            Self::F32 => 23,
            Self::DF64 => 48,
            Self::F64 | Self::F64Precise => 52,
        }
    }
}

impl std::fmt::Display for PrecisionTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::F16 => write!(f, "F16"),
            Self::F32 => write!(f, "F32"),
            Self::DF64 => write!(f, "DF64"),
            Self::F64 => write!(f, "F64"),
            Self::F64Precise => write!(f, "F64Precise"),
        }
    }
}

/// Physics domain classification for precision routing.
///
/// Each domain has known precision requirements based on its numerical
/// characteristics (cancellation sensitivity, FMA tolerance, etc.).
/// The precision brain maps domains to tiers given hardware constraints.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PhysicsDomain {
    /// Lattice QCD: gauge force, plaquette, HMC — tolerant of FMA.
    LatticeQcd,
    /// Gradient flow: energy density, scale setting — moderately sensitive.
    GradientFlow,
    /// Dielectric functions: complex arithmetic, cancellation-prone — needs precise.
    Dielectric,
    /// Kinetic-fluid: BGK relaxation, Euler HLL — tolerant.
    KineticFluid,
    /// Eigensolve: Jacobi, Lanczos — needs precise.
    Eigensolve,
    /// Molecular dynamics: forces, transport — tolerant.
    MolecularDynamics,
    /// Nuclear EOS: BCS pairing, HFB — moderate.
    NuclearEos,
    /// Population pharmacokinetics: FOCE, SAEM — moderate.
    PopulationPk,
    /// Bioinformatics: sequence alignment, diversity — tolerant.
    Bioinformatics,
    /// Hydrology: ET₀, Richards PDE, water balance — moderate.
    Hydrology,
    /// Statistics: bootstrap, jackknife, Monte Carlo — tolerant.
    Statistics,
    /// General purpose: no specific domain requirements.
    General,
}

impl PhysicsDomain {
    /// Whether this domain is cancellation-sensitive and benefits from FMA-free execution.
    #[must_use]
    pub const fn fma_sensitive(self) -> bool {
        matches!(self, Self::Dielectric | Self::Eigensolve)
    }

    /// Whether this domain is throughput-bound (can benefit from DF64 over throttled F64).
    #[must_use]
    pub const fn throughput_bound(self) -> bool {
        matches!(
            self,
            Self::LatticeQcd
                | Self::KineticFluid
                | Self::MolecularDynamics
                | Self::Bioinformatics
                | Self::Statistics
        )
    }

    /// Minimum precision tier required for correct results in this domain.
    #[must_use]
    pub const fn minimum_tier(self) -> PrecisionTier {
        match self {
            Self::Dielectric | Self::Eigensolve => PrecisionTier::F64,
            Self::GradientFlow | Self::NuclearEos | Self::PopulationPk => PrecisionTier::DF64,
            Self::LatticeQcd
            | Self::KineticFluid
            | Self::MolecularDynamics
            | Self::Bioinformatics
            | Self::Hydrology
            | Self::Statistics
            | Self::General => PrecisionTier::F32,
        }
    }
}

/// Precision routing advice for a given domain and hardware combination.
#[derive(Debug, Clone)]
pub struct PrecisionBrainAdvice {
    /// Recommended compilation tier.
    pub tier: PrecisionTier,
    /// Whether FMA fusion is safe for this domain at this tier.
    pub fma_safe: bool,
    /// Human-readable rationale for the routing decision.
    pub rationale: &'static str,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tier_display() {
        assert_eq!(PrecisionTier::F16.to_string(), "F16");
        assert_eq!(PrecisionTier::F32.to_string(), "F32");
        assert_eq!(PrecisionTier::DF64.to_string(), "DF64");
        assert_eq!(PrecisionTier::F64.to_string(), "F64");
        assert_eq!(PrecisionTier::F64Precise.to_string(), "F64Precise");
    }

    #[test]
    fn mantissa_bits_ordered() {
        assert!(PrecisionTier::F16.mantissa_bits() < PrecisionTier::F32.mantissa_bits());
        assert!(PrecisionTier::F32.mantissa_bits() < PrecisionTier::DF64.mantissa_bits());
        assert!(PrecisionTier::DF64.mantissa_bits() < PrecisionTier::F64.mantissa_bits());
        assert_eq!(
            PrecisionTier::F64.mantissa_bits(),
            PrecisionTier::F64Precise.mantissa_bits()
        );
    }

    #[test]
    fn domain_fma_sensitivity() {
        assert!(PhysicsDomain::Dielectric.fma_sensitive());
        assert!(PhysicsDomain::Eigensolve.fma_sensitive());
        assert!(!PhysicsDomain::LatticeQcd.fma_sensitive());
        assert!(!PhysicsDomain::MolecularDynamics.fma_sensitive());
    }

    #[test]
    fn domain_throughput_bound() {
        assert!(PhysicsDomain::LatticeQcd.throughput_bound());
        assert!(PhysicsDomain::Bioinformatics.throughput_bound());
        assert!(!PhysicsDomain::Dielectric.throughput_bound());
        assert!(!PhysicsDomain::PopulationPk.throughput_bound());
    }

    #[test]
    fn domain_minimum_tiers() {
        assert_eq!(PhysicsDomain::Dielectric.minimum_tier(), PrecisionTier::F64);
        assert_eq!(
            PhysicsDomain::GradientFlow.minimum_tier(),
            PrecisionTier::DF64
        );
        assert_eq!(PhysicsDomain::General.minimum_tier(), PrecisionTier::F32);
    }
}

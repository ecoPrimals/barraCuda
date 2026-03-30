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
/// 15 discrete tiers spanning the full precision-throughput continuum from
/// 1-bit binary hashing through ~104-bit mantissa extended precision.
/// barraCuda defines the mathematical contract for each tier; coralReef
/// handles compilation and toadStool routes to silicon.
///
/// Ordered from lowest to highest precision:
///
/// | Tier | Mantissa | Storage | Throughput vs F32 |
/// |------|----------|---------|-------------------|
/// | `Binary` | 1 bit | 32 per u32 | ~32× |
/// | `Int2` | 2 bits | 16 per u32 | ~16× |
/// | `Quantized4` | 4 bits | 8 per u32 | ~8× |
/// | `Quantized8` | 8 bits | 4 per u32 | ~4× |
/// | `Fp8E5M2` | 2-bit mant | 4 per u32 | ~4× |
/// | `Fp8E4M3` | 3-bit mant | 4 per u32 | ~4× |
/// | `Bf16` | 7 bits | 2 per u32 | ~2× |
/// | `F16` | 10 bits | 2 per u32 | ~2× |
/// | `Tf32` | 10 bits | 1× f32 | ~1× (tensor) |
/// | `F32` | 23 bits | 1× f32 | 1× (baseline) |
/// | `DF64` | ~48 bits | 2× f32 | ~0.4× |
/// | `F64` | 52 bits | 1× f64 | 0.03–0.5× |
/// | `F64Precise` | 52 bits | 1× f64 | 0.03–0.5× |
/// | `QF128` | ~96 bits | 4× f32 | ~0.12× |
/// | `DF128` | ~104 bits | 2× f64 | ~0.03× |
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PrecisionTier {
    /// 1-bit binary: XNOR+popcount dot products, locality-sensitive hashing.
    /// 32 values packed per u32. Universally available.
    Binary,
    /// 2-bit ternary {-1, 0, +1}: ternary neural networks.
    /// 16 values packed per u32. Universally available.
    Int2,
    /// 4-bit block quantized (`Q4_0`): LLM inference.
    /// 8 values per u32 + f16 scale per 32-value block. Universally available.
    Quantized4,
    /// 8-bit block quantized (`Q8_0`): quantized inference.
    /// 4 values per u32 + f16 scale per 32-value block. Universally available.
    Quantized8,
    /// 8-bit float E5M2: wider range, coarser mantissa — gradient communication.
    /// 4 values per u32. Emulated via u32 bit ops. Universally available.
    Fp8E5M2,
    /// 8-bit float E4M3: higher precision, smaller range — inference weights.
    /// 4 values per u32. Emulated via u32 bit ops. Universally available.
    Fp8E4M3,
    /// 16-bit bfloat (Google Brain): same exponent range as f32, 7-bit mantissa.
    /// Emulated via u32 bit manipulation. Universally available.
    Bf16,
    /// 16-bit IEEE half precision: ML inference, screening, tensor core path.
    /// Requires `SHADER_F16`; emulated via pack/unpack on older hardware.
    F16,
    /// TF32 (NVIDIA tensor core internal format): 10-bit mantissa, f32 range.
    /// Informational — cannot be expressed in WGSL directly. Tensor core
    /// accumulation format selected by toadStool when available.
    Tf32,
    /// 32-bit float: screening, preview, throughput-bound work. Universal baseline.
    F32,
    /// Double-float emulation (f32-pair): ~48-bit mantissa, ~14 decimal digits.
    /// Up to 10× native f64 throughput on consumer GPUs. Universally available.
    DF64,
    /// Native 64-bit: reference precision, validation. Requires `SHADER_F64`.
    F64,
    /// Native 64-bit without FMA fusion: precision-critical domains where
    /// `a*b+c` must not be fused (dielectric, eigensolve). Requires `SHADER_F64`.
    F64Precise,
    /// Quad-double on f32 (Bailey): ~96-bit mantissa from 4× f32 components.
    /// Works on any GPU with FP32 cores — no f64 hardware required.
    QF128,
    /// Double-double on f64 (Dekker): ~104-bit mantissa from 2× f64 components.
    /// Requires `SHADER_F64` hardware. Preferred over QF128 on compute GPUs.
    DF128,
}

impl PrecisionTier {
    /// Approximate mantissa bits for this tier.
    #[must_use]
    pub const fn mantissa_bits(self) -> u32 {
        match self {
            Self::Binary => 1,
            Self::Int2 => 2,
            Self::Quantized4 => 4,
            Self::Quantized8 | Self::Fp8E5M2 => 2,
            Self::Fp8E4M3 => 3,
            Self::Bf16 => 7,
            Self::F16 | Self::Tf32 => 10,
            Self::F32 => 23,
            Self::DF64 => 48,
            Self::F64 | Self::F64Precise => 52,
            Self::QF128 => 96,
            Self::DF128 => 104,
        }
    }

    /// Bytes of GPU storage per scalar value.
    ///
    /// For packed formats (Binary, Int2, Quantized4, FP8) this returns the
    /// per-element byte cost within a packed u32 — fractional values are
    /// rounded up to the smallest whole-byte container.
    #[must_use]
    pub const fn bytes_per_element(self) -> usize {
        match self {
            Self::Binary | Self::Int2 | Self::Quantized4 => 1,
            Self::Quantized8 | Self::Fp8E4M3 | Self::Fp8E5M2 => 1,
            Self::Bf16 | Self::F16 => 2,
            Self::Tf32 | Self::F32 => 4,
            Self::DF64 => 8,
            Self::F64 | Self::F64Precise => 8,
            Self::QF128 => 16,
            Self::DF128 => 16,
        }
    }

    /// Whether this tier requires `SHADER_F64` hardware.
    #[must_use]
    pub const fn requires_f64(self) -> bool {
        matches!(self, Self::F64 | Self::F64Precise | Self::DF128)
    }

    /// Whether this tier requires `SHADER_F16` hardware (or emulation).
    #[must_use]
    pub const fn requires_f16(self) -> bool {
        matches!(self, Self::F16)
    }

    /// Whether this tier is universally available on all GPUs with f32 cores.
    #[must_use]
    pub const fn universally_available(self) -> bool {
        !self.requires_f64() && !self.requires_f16() && !matches!(self, Self::Tf32)
    }

    /// Whether this is a scale-up tier (above F32 baseline).
    #[must_use]
    pub const fn is_scale_up(self) -> bool {
        matches!(
            self,
            Self::DF64 | Self::F64 | Self::F64Precise | Self::QF128 | Self::DF128
        )
    }

    /// Whether this is a scale-down tier (below F32 baseline).
    #[must_use]
    pub const fn is_scale_down(self) -> bool {
        matches!(
            self,
            Self::Binary
                | Self::Int2
                | Self::Quantized4
                | Self::Quantized8
                | Self::Fp8E5M2
                | Self::Fp8E4M3
                | Self::Bf16
                | Self::F16
                | Self::Tf32
        )
    }

    /// Whether this is a quantized integer format (block-quantized, not floating point).
    #[must_use]
    pub const fn is_quantized(self) -> bool {
        matches!(
            self,
            Self::Binary | Self::Int2 | Self::Quantized4 | Self::Quantized8
        )
    }
}

impl std::fmt::Display for PrecisionTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Binary => write!(f, "Binary"),
            Self::Int2 => write!(f, "Int2"),
            Self::Quantized4 => write!(f, "Q4"),
            Self::Quantized8 => write!(f, "Q8"),
            Self::Fp8E5M2 => write!(f, "FP8-E5M2"),
            Self::Fp8E4M3 => write!(f, "FP8-E4M3"),
            Self::Bf16 => write!(f, "BF16"),
            Self::F16 => write!(f, "F16"),
            Self::Tf32 => write!(f, "TF32"),
            Self::F32 => write!(f, "F32"),
            Self::DF64 => write!(f, "DF64"),
            Self::F64 => write!(f, "F64"),
            Self::F64Precise => write!(f, "F64Precise"),
            Self::QF128 => write!(f, "QF128"),
            Self::DF128 => write!(f, "DF128"),
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
    /// ML inference: quantized weight dequantization, activation compute.
    /// Routes to Q4 → Q8 → F16 → F32 depending on hardware.
    Inference,
    /// ML training: mixed-precision forward/backward passes.
    /// Routes to BF16 → F32 → DF64 depending on hardware.
    Training,
    /// Locality-sensitive hashing, binary neural networks.
    /// Routes to Binary tier (XNOR+popcount).
    Hashing,
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
                | Self::Inference
                | Self::Training
                | Self::Hashing
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
            Self::Inference => PrecisionTier::Quantized4,
            Self::Training => PrecisionTier::Bf16,
            Self::Hashing => PrecisionTier::Binary,
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
        assert_eq!(PrecisionTier::Binary.to_string(), "Binary");
        assert_eq!(PrecisionTier::Int2.to_string(), "Int2");
        assert_eq!(PrecisionTier::Quantized4.to_string(), "Q4");
        assert_eq!(PrecisionTier::Quantized8.to_string(), "Q8");
        assert_eq!(PrecisionTier::Fp8E5M2.to_string(), "FP8-E5M2");
        assert_eq!(PrecisionTier::Fp8E4M3.to_string(), "FP8-E4M3");
        assert_eq!(PrecisionTier::Bf16.to_string(), "BF16");
        assert_eq!(PrecisionTier::F16.to_string(), "F16");
        assert_eq!(PrecisionTier::Tf32.to_string(), "TF32");
        assert_eq!(PrecisionTier::F32.to_string(), "F32");
        assert_eq!(PrecisionTier::DF64.to_string(), "DF64");
        assert_eq!(PrecisionTier::F64.to_string(), "F64");
        assert_eq!(PrecisionTier::F64Precise.to_string(), "F64Precise");
        assert_eq!(PrecisionTier::QF128.to_string(), "QF128");
        assert_eq!(PrecisionTier::DF128.to_string(), "DF128");
    }

    #[test]
    fn mantissa_bits_ordered_scale_up() {
        assert!(PrecisionTier::F32.mantissa_bits() < PrecisionTier::DF64.mantissa_bits());
        assert!(PrecisionTier::DF64.mantissa_bits() < PrecisionTier::F64.mantissa_bits());
        assert_eq!(
            PrecisionTier::F64.mantissa_bits(),
            PrecisionTier::F64Precise.mantissa_bits()
        );
        assert!(PrecisionTier::F64.mantissa_bits() < PrecisionTier::QF128.mantissa_bits());
        assert!(PrecisionTier::QF128.mantissa_bits() < PrecisionTier::DF128.mantissa_bits());
    }

    #[test]
    fn mantissa_bits_scale_down() {
        assert!(PrecisionTier::Binary.mantissa_bits() < PrecisionTier::Int2.mantissa_bits());
        assert!(PrecisionTier::Fp8E4M3.mantissa_bits() < PrecisionTier::Bf16.mantissa_bits());
        assert!(PrecisionTier::Bf16.mantissa_bits() < PrecisionTier::F16.mantissa_bits());
        assert!(PrecisionTier::F16.mantissa_bits() < PrecisionTier::F32.mantissa_bits());
    }

    #[test]
    fn tier_15_count() {
        let all = [
            PrecisionTier::Binary,
            PrecisionTier::Int2,
            PrecisionTier::Quantized4,
            PrecisionTier::Quantized8,
            PrecisionTier::Fp8E5M2,
            PrecisionTier::Fp8E4M3,
            PrecisionTier::Bf16,
            PrecisionTier::F16,
            PrecisionTier::Tf32,
            PrecisionTier::F32,
            PrecisionTier::DF64,
            PrecisionTier::F64,
            PrecisionTier::F64Precise,
            PrecisionTier::QF128,
            PrecisionTier::DF128,
        ];
        assert_eq!(all.len(), 15);
    }

    #[test]
    fn f64_requirement_flags() {
        assert!(PrecisionTier::F64.requires_f64());
        assert!(PrecisionTier::F64Precise.requires_f64());
        assert!(PrecisionTier::DF128.requires_f64());
        assert!(!PrecisionTier::F32.requires_f64());
        assert!(!PrecisionTier::DF64.requires_f64());
        assert!(!PrecisionTier::QF128.requires_f64());
    }

    #[test]
    fn universal_availability() {
        assert!(PrecisionTier::F32.universally_available());
        assert!(PrecisionTier::DF64.universally_available());
        assert!(PrecisionTier::QF128.universally_available());
        assert!(PrecisionTier::Binary.universally_available());
        assert!(!PrecisionTier::F64.universally_available());
        assert!(!PrecisionTier::F16.universally_available());
        assert!(!PrecisionTier::Tf32.universally_available());
    }

    #[test]
    fn scale_up_down_classification() {
        assert!(PrecisionTier::DF128.is_scale_up());
        assert!(PrecisionTier::QF128.is_scale_up());
        assert!(PrecisionTier::F64.is_scale_up());
        assert!(!PrecisionTier::F32.is_scale_up());
        assert!(!PrecisionTier::F32.is_scale_down());
        assert!(PrecisionTier::Binary.is_scale_down());
        assert!(PrecisionTier::F16.is_scale_down());
        assert!(PrecisionTier::Bf16.is_scale_down());
    }

    #[test]
    fn quantized_classification() {
        assert!(PrecisionTier::Binary.is_quantized());
        assert!(PrecisionTier::Int2.is_quantized());
        assert!(PrecisionTier::Quantized4.is_quantized());
        assert!(PrecisionTier::Quantized8.is_quantized());
        assert!(!PrecisionTier::Fp8E4M3.is_quantized());
        assert!(!PrecisionTier::F32.is_quantized());
    }

    #[test]
    fn domain_fma_sensitivity() {
        assert!(PhysicsDomain::Dielectric.fma_sensitive());
        assert!(PhysicsDomain::Eigensolve.fma_sensitive());
        assert!(!PhysicsDomain::LatticeQcd.fma_sensitive());
        assert!(!PhysicsDomain::MolecularDynamics.fma_sensitive());
        assert!(!PhysicsDomain::Inference.fma_sensitive());
    }

    #[test]
    fn domain_throughput_bound() {
        assert!(PhysicsDomain::LatticeQcd.throughput_bound());
        assert!(PhysicsDomain::Bioinformatics.throughput_bound());
        assert!(PhysicsDomain::Inference.throughput_bound());
        assert!(PhysicsDomain::Training.throughput_bound());
        assert!(PhysicsDomain::Hashing.throughput_bound());
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
        assert_eq!(
            PhysicsDomain::Inference.minimum_tier(),
            PrecisionTier::Quantized4
        );
        assert_eq!(PhysicsDomain::Training.minimum_tier(), PrecisionTier::Bf16);
        assert_eq!(PhysicsDomain::Hashing.minimum_tier(), PrecisionTier::Binary);
    }

    #[test]
    fn bytes_per_element_consistent() {
        assert_eq!(PrecisionTier::F32.bytes_per_element(), 4);
        assert_eq!(PrecisionTier::F64.bytes_per_element(), 8);
        assert_eq!(PrecisionTier::DF64.bytes_per_element(), 8);
        assert_eq!(PrecisionTier::DF128.bytes_per_element(), 16);
        assert_eq!(PrecisionTier::QF128.bytes_per_element(), 16);
        assert_eq!(PrecisionTier::F16.bytes_per_element(), 2);
        assert_eq!(PrecisionTier::Bf16.bytes_per_element(), 2);
    }
}

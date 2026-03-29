// SPDX-License-Identifier: AGPL-3.0-or-later

//! Silicon Profile: measured personality of every functional unit on a GPU.
//!
//! Each GPU contains multiple independent silicon blocks (ALU, TMU, ROP,
//! memory controller, L2 cache, tensor cores, shared memory / LDS). Most
//! GPU codes use only one or two and leave the rest idle. barraCuda routes
//! workload phases to the cheapest silicon that can handle them, filling
//! alternative units first and reserving FP64 ALU for precision-critical work.
//!
//! Absorbed from hotSpring V0632 (March 2026) and generalised for
//! domain-agnostic workload routing.
//!
//! ## Tier routing philosophy
//!
//! ```text
//! TIER 0  TMU          lookup tables, PRNG transcendentals, stencil access
//! TIER 1  Tensor cores  matmul, preconditioner (NVIDIA only via SASS)
//! TIER 2  FP32 ALU      DF64 Dekker pairs — bulk compute
//! TIER 3  ROP / Atomics scatter-add for force accumulation
//! TIER 4  Subgroup       warp/wavefront intrinsics for reductions
//! TIER 5  Shared memory  workgroup-level communication, halo exchange
//! TIER 6  FP64 ALU      LAST — precision-critical accumulation
//! ```

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Every distinct silicon functional unit addressable on a GPU.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum SiliconUnit {
    /// Shader ALU — FP32 FMAC pipeline (bulk compute, DF64 host).
    Fp32Alu,
    /// Shader ALU — native FP64 pipeline (1:2 on HPC, 1:64 on gaming).
    Fp64Alu,
    /// Texture Mapping Unit — hardware interpolation, cache-backed lookup.
    Tmu,
    /// Render Output Pipeline — blend, atomicAdd, scatter-write.
    Rop,
    /// Tensor / Matrix Cores — FP16/TF32/FP64 DMMA tiles (NVIDIA).
    TensorCore,
    /// Memory controller — VRAM bandwidth (sequential, coalesced).
    MemoryBandwidth,
    /// L2 cache (+ Infinity Cache on AMD RDNA).
    CacheHierarchy,
    /// Workgroup shared memory / Local Data Share.
    SharedMemory,
    /// Subgroup (warp/wavefront) intrinsics — shuffle, ballot, reduce.
    SubgroupIntrinsics,
}

impl std::fmt::Display for SiliconUnit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Fp32Alu => write!(f, "FP32 ALU"),
            Self::Fp64Alu => write!(f, "FP64 ALU"),
            Self::Tmu => write!(f, "TMU"),
            Self::Rop => write!(f, "ROP"),
            Self::TensorCore => write!(f, "Tensor"),
            Self::MemoryBandwidth => write!(f, "Mem BW"),
            Self::CacheHierarchy => write!(f, "Cache"),
            Self::SharedMemory => write!(f, "LDS/Shared"),
            Self::SubgroupIntrinsics => write!(f, "Subgroup"),
        }
    }
}

/// Throughput measurement for a single silicon unit.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct UnitThroughput {
    /// Theoretical peak from vendor spec sheet (TFLOPS, GT/s, GB/s — unit-specific).
    pub theoretical_peak: f64,
    /// Measured peak from micro-benchmark saturation experiment.
    pub measured_peak: f64,
    /// Efficiency: measured / theoretical (0.0–1.0+).
    pub efficiency: f64,
    /// Human-readable unit for the throughput value (e.g. "TFLOPS", "GT/s", "GB/s").
    pub unit: String,
}

/// Composition multiplier: measured speedup when two units run simultaneously.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompositionEntry {
    /// First silicon unit.
    pub unit_a: SiliconUnit,
    /// Second silicon unit.
    pub unit_b: SiliconUnit,
    /// Time for A alone + B alone (serial estimate).
    pub serial_ms: f64,
    /// Time for A+B in same dispatch (compound).
    pub compound_ms: f64,
    /// Multiplier: serial / compound. >1.0 means they truly run in parallel.
    pub multiplier: f64,
}

/// GPU vendor classification for silicon routing heuristics.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum GpuVendorTag {
    /// NVIDIA (`GeForce`, `Quadro`, `Tesla`, etc.)
    Nvidia,
    /// AMD (Radeon, Instinct, etc.)
    Amd,
    /// Intel (Arc, UHD, etc.)
    Intel,
    /// Apple (M-series integrated GPU)
    Apple,
    /// Software rasterizer (`llvmpipe`, `SwiftShader`)
    Software,
    /// Unknown vendor.
    Unknown,
}

/// Full silicon personality for a single GPU adapter.
///
/// Captures both spec-sheet data and measured micro-benchmark results.
/// Used by the tier router to select the cheapest silicon for each
/// workload phase.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SiliconProfile {
    /// Adapter name (e.g. NVIDIA `GeForce` RTX 3090).
    pub adapter_name: String,
    /// Vendor classification.
    pub vendor: GpuVendorTag,
    /// VRAM in bytes.
    pub vram_bytes: u64,
    /// Boost clock in GHz.
    pub boost_ghz: f64,
    /// Per-unit throughput: theoretical + measured.
    pub units: BTreeMap<SiliconUnit, UnitThroughput>,
    /// Measured composition multipliers (pairs of units running simultaneously).
    pub compositions: Vec<CompositionEntry>,
    /// DF64 TFLOPS (Dekker FP32-pair arithmetic on FP32 ALU).
    pub df64_tflops: f64,
    /// L2 cache size in bytes.
    pub l2_bytes: u64,
    /// AMD Infinity Cache size in bytes (0 for non-AMD).
    pub infinity_cache_bytes: u64,
    /// TMU count (from spec sheet).
    pub tmu_count: u32,
    /// ROP count (from spec sheet).
    pub rop_count: u32,
    /// Subgroup (warp/wavefront) size, 0 if unknown.
    pub subgroup_size: u32,
    /// ISO-8601 timestamp of when this profile was last measured.
    pub measured_at: String,
}

impl SiliconProfile {
    /// Whether this GPU has measured throughput for a specific silicon unit.
    #[must_use]
    pub fn has_unit(&self, unit: SiliconUnit) -> bool {
        self.units.get(&unit).is_some_and(|t| t.measured_peak > 0.0)
    }

    /// Get the measured throughput for a silicon unit, or 0.0 if not measured.
    #[must_use]
    pub fn measured_throughput(&self, unit: SiliconUnit) -> f64 {
        self.units.get(&unit).map_or(0.0, |t| t.measured_peak)
    }

    /// The FP64:FP32 throughput ratio — key indicator of GPU class.
    ///
    /// - HPC cards (A100, MI250): ~0.5 (1:2 ratio)
    /// - Gaming cards (RTX 3090): ~0.016 (1:64 ratio)
    /// - Consumer cards without FP64: 0.0
    #[must_use]
    pub fn fp64_fp32_ratio(&self) -> f64 {
        let fp32 = self.measured_throughput(SiliconUnit::Fp32Alu);
        let fp64 = self.measured_throughput(SiliconUnit::Fp64Alu);
        if fp32 > 0.0 { fp64 / fp32 } else { 0.0 }
    }

    /// Whether this GPU has tensor cores with measured throughput.
    #[must_use]
    pub fn has_tensor_cores(&self) -> bool {
        self.has_unit(SiliconUnit::TensorCore)
    }

    /// Whether subgroup intrinsics are available (warp/wavefront level).
    #[must_use]
    pub const fn has_subgroup_intrinsics(&self) -> bool {
        self.subgroup_size > 0
    }

    /// Best composition multiplier involving a given unit.
    ///
    /// Returns the highest measured parallelism factor when this unit
    /// runs alongside another, or 1.0 (serial) if no composition data.
    #[must_use]
    pub fn best_composition(&self, unit: SiliconUnit) -> f64 {
        self.compositions
            .iter()
            .filter(|c| c.unit_a == unit || c.unit_b == unit)
            .map(|c| c.multiplier)
            .fold(1.0f64, f64::max)
    }

    /// Serialize the profile to a JSON file.
    ///
    /// # Errors
    /// Returns [`std::io::Error`] on write failure.
    pub fn save(&self, path: &std::path::Path) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(self).map_err(std::io::Error::other)?;
        std::fs::write(path, json)
    }

    /// Load a profile from a JSON file.
    ///
    /// # Errors
    /// Returns [`std::io::Error`] on read or parse failure.
    pub fn load(path: &std::path::Path) -> std::io::Result<Self> {
        let json = std::fs::read_to_string(path)?;
        serde_json::from_str(&json)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }
}

/// Select the cheapest silicon unit for a workload given a priority ordering.
///
/// Returns the first unit in `preferred` that has measured throughput > 0,
/// or falls back to `fallback`.
#[must_use]
pub fn route_workload(
    profile: &SiliconProfile,
    preferred: &[SiliconUnit],
    fallback: SiliconUnit,
) -> SiliconUnit {
    preferred
        .iter()
        .find(|&&unit| profile.has_unit(unit))
        .copied()
        .unwrap_or(fallback)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_profile() -> SiliconProfile {
        let mut units = BTreeMap::new();
        units.insert(
            SiliconUnit::Fp32Alu,
            UnitThroughput {
                theoretical_peak: 35.58,
                measured_peak: 32.0,
                efficiency: 0.9,
                unit: "TFLOPS".into(),
            },
        );
        units.insert(
            SiliconUnit::Fp64Alu,
            UnitThroughput {
                theoretical_peak: 0.556,
                measured_peak: 0.52,
                efficiency: 0.93,
                unit: "TFLOPS".into(),
            },
        );
        units.insert(
            SiliconUnit::Tmu,
            UnitThroughput {
                theoretical_peak: 556.0,
                measured_peak: 480.0,
                efficiency: 0.86,
                unit: "GT/s".into(),
            },
        );

        SiliconProfile {
            adapter_name: "NVIDIA GeForce RTX 3090".into(),
            vendor: GpuVendorTag::Nvidia,
            vram_bytes: 24 * 1024 * 1024 * 1024,
            boost_ghz: 1.695,
            units,
            compositions: vec![CompositionEntry {
                unit_a: SiliconUnit::Fp32Alu,
                unit_b: SiliconUnit::Tmu,
                serial_ms: 10.0,
                compound_ms: 6.0,
                multiplier: 1.67,
            }],
            df64_tflops: 4.2,
            l2_bytes: 6 * 1024 * 1024,
            infinity_cache_bytes: 0,
            tmu_count: 328,
            rop_count: 112,
            subgroup_size: 32,
            measured_at: "2026-03-29T12:00:00Z".into(),
        }
    }

    #[test]
    fn fp64_fp32_ratio_gaming() {
        let p = test_profile();
        let ratio = p.fp64_fp32_ratio();
        assert!(
            ratio < 0.05,
            "gaming GPU should have low FP64:FP32 ratio: {ratio}"
        );
    }

    #[test]
    fn has_units() {
        let p = test_profile();
        assert!(p.has_unit(SiliconUnit::Fp32Alu));
        assert!(p.has_unit(SiliconUnit::Tmu));
        assert!(!p.has_unit(SiliconUnit::TensorCore));
    }

    #[test]
    fn best_composition_found() {
        let p = test_profile();
        let c = p.best_composition(SiliconUnit::Tmu);
        assert!((c - 1.67).abs() < 0.01);
    }

    #[test]
    fn best_composition_missing() {
        let p = test_profile();
        let c = p.best_composition(SiliconUnit::Rop);
        assert!((c - 1.0).abs() < 0.01, "should be serial fallback");
    }

    #[test]
    fn route_workload_finds_tmu() {
        let p = test_profile();
        let unit = route_workload(
            &p,
            &[
                SiliconUnit::TensorCore,
                SiliconUnit::Tmu,
                SiliconUnit::Fp32Alu,
            ],
            SiliconUnit::Fp64Alu,
        );
        assert_eq!(unit, SiliconUnit::Tmu, "tensor cores missing → TMU first");
    }

    #[test]
    fn route_workload_fallback() {
        let p = test_profile();
        let unit = route_workload(&p, &[SiliconUnit::TensorCore], SiliconUnit::Fp32Alu);
        assert_eq!(unit, SiliconUnit::Fp32Alu, "should fallback");
    }

    #[test]
    fn serde_round_trip() {
        let p = test_profile();
        let json = serde_json::to_string(&p).unwrap();
        let restored: SiliconProfile = serde_json::from_str(&json).unwrap();
        assert_eq!(restored.adapter_name, p.adapter_name);
        assert_eq!(restored.tmu_count, p.tmu_count);
    }
}

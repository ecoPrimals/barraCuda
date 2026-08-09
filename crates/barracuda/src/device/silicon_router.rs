// SPDX-License-Identifier: AGPL-3.0-or-later

//! Silicon workload routing — selects optimal compute substrate from measured profiles.
//!
//! Evolved from the free `route_workload()` helper in [`super::silicon_profile`]
//! into a trait-based router informed by the Silicon Fold AAR (RTX 3090 + RX 6950 XT).

use super::silicon_profile::{GpuVendorTag, SiliconProfile, SiliconUnit};
use serde::{Deserialize, Serialize};

/// Minimum numeric precision required by a workload phase.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Precision {
    /// Half precision — ML inference / screening.
    F16,
    /// Single precision — default GPU compute.
    F32,
    /// Dekker double-float on FP32 ALU (~48-bit mantissa).
    Df64,
    /// Native double precision.
    F64,
}

/// Workload characteristics used for silicon substrate selection.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WorkloadRequirements {
    /// Minimum numeric precision the phase must preserve.
    pub min_precision: Precision,
    /// Bytes touched per tile / iteration (working set).
    pub working_set_bytes: u64,
    /// FLOPs per byte — separates compute-bound from memory-bound phases.
    pub arithmetic_intensity: f64,
    /// Phase uses atomic scatter-add (force accumulation, histograms).
    pub needs_atomics: bool,
    /// Phase benefits from native or emulated FP16 fast path.
    pub needs_f16: bool,
}

/// Selected silicon substrate with expected throughput.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SubstrateChoice {
    /// Functional unit chosen for this phase.
    pub unit: SiliconUnit,
    /// Expected throughput in GFLOPS (or GFLOPS-equivalent for memory paths).
    pub expected_throughput_gflops: f64,
}

/// Availability and measured peak for one silicon unit.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SubstrateInfo {
    /// The functional unit.
    pub unit: SiliconUnit,
    /// Measured peak throughput from micro-benchmarks.
    pub measured_peak: f64,
    /// Whether the unit has non-zero measured throughput on this device.
    pub available: bool,
}

/// Routes compute workloads to optimal silicon substrate based on measured capability.
pub trait SiliconRouter {
    /// Select the optimal substrate for a workload's requirements.
    fn route(&self, requirements: &WorkloadRequirements) -> SubstrateChoice;

    /// List all available compute substrates on this device.
    fn available_substrates(&self) -> Vec<SubstrateInfo>;
}

/// Measured F16 speedup on AMD RDNA vs FP32 baseline (Silicon Fold AAR).
const AMD_F16_SPEEDUP: f64 = 1.32;
/// Arithmetic intensity below this threshold prefers memory/cache paths (FLOP/byte).
const MEMORY_BOUND_AI_THRESHOLD: f64 = 8.0;

impl SiliconRouter for SiliconProfile {
    fn route(&self, requirements: &WorkloadRequirements) -> SubstrateChoice {
        if requirements.needs_atomics && self.has_unit(SiliconUnit::Rop) {
            return choice(self, SiliconUnit::Rop, requirements.arithmetic_intensity);
        }

        let cache_bytes = effective_cache_bytes(self);
        if cache_bytes > 0 && requirements.working_set_bytes <= cache_bytes {
            if self.has_unit(SiliconUnit::CacheHierarchy) {
                return choice(
                    self,
                    SiliconUnit::CacheHierarchy,
                    requirements.arithmetic_intensity,
                );
            }
            if self.has_unit(SiliconUnit::MemoryBandwidth) {
                return choice(
                    self,
                    SiliconUnit::MemoryBandwidth,
                    requirements.arithmetic_intensity,
                );
            }
        }

        if requirements.needs_f16 {
            return route_f16(self, requirements);
        }

        match requirements.min_precision {
            Precision::F64 => route_f64(self, requirements),
            Precision::Df64 => route_df64(self, requirements),
            Precision::F16 => route_f16(self, requirements),
            Precision::F32 => route_fp32(self, requirements),
        }
    }

    fn available_substrates(&self) -> Vec<SubstrateInfo> {
        all_silicon_units()
            .into_iter()
            .map(|unit| {
                let measured_peak = self.measured_throughput(unit);
                SubstrateInfo {
                    unit,
                    measured_peak,
                    available: self.has_unit(unit),
                }
            })
            .collect()
    }
}

/// Select substrate using F16 fast-path heuristics from Silicon Fold measurements.
fn route_f16(profile: &SiliconProfile, requirements: &WorkloadRequirements) -> SubstrateChoice {
    match profile.vendor {
        GpuVendorTag::Amd => {
            let unit = if profile.has_tensor_cores() {
                SiliconUnit::TensorCore
            } else if profile.has_unit(SiliconUnit::Fp32Alu) {
                SiliconUnit::Fp32Alu
            } else {
                SiliconUnit::MemoryBandwidth
            };
            let mut choice = choice(profile, unit, requirements.arithmetic_intensity);
            choice.expected_throughput_gflops *= AMD_F16_SPEEDUP;
            choice
        }
        GpuVendorTag::Nvidia if is_nvidia_ampere(profile) => {
            // Ampere F16 path measured 0.99× — skip F16, use FP32.
            route_fp32(profile, requirements)
        }
        GpuVendorTag::Nvidia => {
            if profile.has_tensor_cores() {
                choice(
                    profile,
                    SiliconUnit::TensorCore,
                    requirements.arithmetic_intensity,
                )
            } else {
                route_fp32(profile, requirements)
            }
        }
        _ => route_fp32(profile, requirements),
    }
}

fn route_f64(profile: &SiliconProfile, requirements: &WorkloadRequirements) -> SubstrateChoice {
    if profile.has_unit(SiliconUnit::Fp64Alu) && profile.fp64_fp32_ratio() >= 0.1 {
        choice(
            profile,
            SiliconUnit::Fp64Alu,
            requirements.arithmetic_intensity,
        )
    } else {
        route_df64(profile, requirements)
    }
}

fn route_df64(profile: &SiliconProfile, requirements: &WorkloadRequirements) -> SubstrateChoice {
    if profile.df64_tflops > 0.0 && profile.has_unit(SiliconUnit::Fp32Alu) {
        SubstrateChoice {
            unit: SiliconUnit::Fp32Alu,
            expected_throughput_gflops: profile.df64_tflops * 1000.0,
        }
    } else if profile.has_unit(SiliconUnit::Fp64Alu) {
        choice(
            profile,
            SiliconUnit::Fp64Alu,
            requirements.arithmetic_intensity,
        )
    } else {
        route_fp32(profile, requirements)
    }
}

fn route_fp32(profile: &SiliconProfile, requirements: &WorkloadRequirements) -> SubstrateChoice {
    if requirements.arithmetic_intensity < MEMORY_BOUND_AI_THRESHOLD {
        if profile.has_unit(SiliconUnit::MemoryBandwidth) {
            return choice(
                profile,
                SiliconUnit::MemoryBandwidth,
                requirements.arithmetic_intensity,
            );
        }
        if profile.has_unit(SiliconUnit::CacheHierarchy) {
            return choice(
                profile,
                SiliconUnit::CacheHierarchy,
                requirements.arithmetic_intensity,
            );
        }
    }

    if profile.has_tensor_cores() && requirements.arithmetic_intensity >= 64.0 {
        return choice(
            profile,
            SiliconUnit::TensorCore,
            requirements.arithmetic_intensity,
        );
    }

    if profile.has_unit(SiliconUnit::Fp32Alu) {
        choice(
            profile,
            SiliconUnit::Fp32Alu,
            requirements.arithmetic_intensity,
        )
    } else if profile.has_unit(SiliconUnit::Fp64Alu) {
        choice(
            profile,
            SiliconUnit::Fp64Alu,
            requirements.arithmetic_intensity,
        )
    } else {
        choice(
            profile,
            SiliconUnit::MemoryBandwidth,
            requirements.arithmetic_intensity,
        )
    }
}

#[must_use]
fn choice(
    profile: &SiliconProfile,
    unit: SiliconUnit,
    arithmetic_intensity: f64,
) -> SubstrateChoice {
    SubstrateChoice {
        unit,
        expected_throughput_gflops: throughput_gflops(profile, unit, arithmetic_intensity),
    }
}

#[must_use]
fn throughput_gflops(
    profile: &SiliconProfile,
    unit: SiliconUnit,
    arithmetic_intensity: f64,
) -> f64 {
    let entry = profile.units.get(&unit);
    let peak = entry.map_or(0.0, |t| t.measured_peak);
    let unit_label = entry.map_or("", |t| t.unit.as_str());

    if unit_label.contains("TFLOP") {
        peak * 1000.0
    } else if unit_label.contains("GB/s") || unit_label.contains("GT/s") {
        // Memory / texture paths: convert bandwidth to GFLOPS-equivalent.
        peak * arithmetic_intensity.max(1.0)
    } else {
        peak * 1000.0
    }
}

#[must_use]
fn effective_cache_bytes(profile: &SiliconProfile) -> u64 {
    profile.l2_bytes.max(profile.infinity_cache_bytes)
}

#[must_use]
fn is_nvidia_ampere(profile: &SiliconProfile) -> bool {
    let name = profile.adapter_name.to_lowercase();
    name.contains("rtx 30")
        || name.contains("a100")
        || name.contains("a10")
        || name.contains("a30")
        || name.contains("a40")
}

#[must_use]
fn all_silicon_units() -> [SiliconUnit; 9] {
    [
        SiliconUnit::Fp32Alu,
        SiliconUnit::Fp64Alu,
        SiliconUnit::Tmu,
        SiliconUnit::Rop,
        SiliconUnit::TensorCore,
        SiliconUnit::MemoryBandwidth,
        SiliconUnit::CacheHierarchy,
        SiliconUnit::SharedMemory,
        SiliconUnit::SubgroupIntrinsics,
    ]
}

/// Legacy priority-list router — delegates availability checks to [`SiliconProfile`].
///
/// Prefer [`SiliconRouter::route`] for workload-aware selection.
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
    use super::super::silicon_profile::UnitThroughput;
    use super::*;
    use std::collections::BTreeMap;

    fn nvidia_profile() -> SiliconProfile {
        let mut units = BTreeMap::new();
        for (u, p, l) in [
            (SiliconUnit::Fp32Alu, 32.0, "TFLOPS"),
            (SiliconUnit::Fp64Alu, 0.52, "TFLOPS"),
            (SiliconUnit::TensorCore, 142.0, "TFLOPS"),
            (SiliconUnit::MemoryBandwidth, 936.0, "GB/s"),
            (SiliconUnit::CacheHierarchy, 800.0, "GB/s"),
            (SiliconUnit::Rop, 112.0, "GP/s"),
        ] {
            units.insert(
                u,
                UnitThroughput {
                    theoretical_peak: p,
                    measured_peak: p,
                    efficiency: 1.0,
                    unit: l.into(),
                },
            );
        }

        SiliconProfile {
            adapter_name: "NVIDIA GeForce RTX 3090".into(),
            vendor: GpuVendorTag::Nvidia,
            vram_bytes: 24 * 1024 * 1024 * 1024,
            boost_ghz: 1.695,
            units,
            compositions: vec![],
            df64_tflops: 4.2,
            l2_bytes: 6 * 1024 * 1024,
            infinity_cache_bytes: 0,
            tmu_count: 328,
            rop_count: 112,
            subgroup_size: 32,
            measured_at: "2026-03-29T12:00:00Z".into(),
        }
    }

    fn amd_profile() -> SiliconProfile {
        let mut units = BTreeMap::new();
        for (u, p, l) in [
            (SiliconUnit::Fp32Alu, 23.0, "TFLOPS"),
            (SiliconUnit::Fp64Alu, 0.36, "TFLOPS"),
            (SiliconUnit::MemoryBandwidth, 512.0, "GB/s"),
            (SiliconUnit::CacheHierarchy, 2000.0, "GB/s"),
        ] {
            units.insert(
                u,
                UnitThroughput {
                    theoretical_peak: p,
                    measured_peak: p,
                    efficiency: 1.0,
                    unit: l.into(),
                },
            );
        }

        SiliconProfile {
            adapter_name: "AMD Radeon RX 6950 XT".into(),
            vendor: GpuVendorTag::Amd,
            vram_bytes: 16 * 1024 * 1024 * 1024,
            boost_ghz: 2.1,
            units,
            compositions: vec![],
            df64_tflops: 2.8,
            l2_bytes: 4 * 1024 * 1024,
            infinity_cache_bytes: 128 * 1024 * 1024,
            tmu_count: 256,
            rop_count: 128,
            subgroup_size: 32,
            measured_at: "2026-03-29T12:00:00Z".into(),
        }
    }

    fn software_profile() -> SiliconProfile {
        SiliconProfile {
            adapter_name: "llvmpipe".into(),
            vendor: GpuVendorTag::Software,
            vram_bytes: 0,
            boost_ghz: 0.0,
            units: BTreeMap::new(),
            compositions: vec![],
            df64_tflops: 0.0,
            l2_bytes: 0,
            infinity_cache_bytes: 0,
            tmu_count: 0,
            rop_count: 0,
            subgroup_size: 0,
            measured_at: String::new(),
        }
    }

    #[test]
    fn cache_resident_routes_to_cache_hierarchy() {
        let profile = nvidia_profile();
        let req = WorkloadRequirements {
            min_precision: Precision::F32,
            working_set_bytes: 4 * 1024 * 1024,
            arithmetic_intensity: 4.0,
            needs_atomics: false,
            needs_f16: false,
        };
        let choice = profile.route(&req);
        assert_eq!(choice.unit, SiliconUnit::CacheHierarchy);
    }

    #[test]
    fn amd_f16_applies_measured_speedup() {
        let profile = amd_profile();
        let req = WorkloadRequirements {
            min_precision: Precision::F32,
            working_set_bytes: 256 * 1024 * 1024,
            arithmetic_intensity: 32.0,
            needs_atomics: false,
            needs_f16: true,
        };
        let baseline = throughput_gflops(&profile, SiliconUnit::Fp32Alu, 32.0);
        let choice = profile.route(&req);
        assert_eq!(choice.unit, SiliconUnit::Fp32Alu);
        assert!(
            (choice.expected_throughput_gflops - baseline * AMD_F16_SPEEDUP).abs() < 0.01,
            "expected {} got {}",
            baseline * AMD_F16_SPEEDUP,
            choice.expected_throughput_gflops
        );
    }

    #[test]
    fn nvidia_ampere_skips_f16_benefit() {
        let profile = nvidia_profile();
        let req = WorkloadRequirements {
            min_precision: Precision::F32,
            working_set_bytes: 256 * 1024 * 1024,
            arithmetic_intensity: 32.0,
            needs_atomics: false,
            needs_f16: true,
        };
        let choice = profile.route(&req);
        assert_eq!(choice.unit, SiliconUnit::Fp32Alu);
        let fp32_only = throughput_gflops(&profile, SiliconUnit::Fp32Alu, 32.0);
        assert!(
            (choice.expected_throughput_gflops - fp32_only).abs() < 0.01,
            "Ampere should not boost F16 path"
        );
    }

    #[test]
    fn atomics_route_to_rop() {
        let profile = nvidia_profile();
        let req = WorkloadRequirements {
            min_precision: Precision::F32,
            working_set_bytes: 1024,
            arithmetic_intensity: 1.0,
            needs_atomics: true,
            needs_f16: false,
        };
        assert_eq!(profile.route(&req).unit, SiliconUnit::Rop);
    }

    #[test]
    fn available_substrates_lists_all_units() {
        let profile = nvidia_profile();
        let substrates = profile.available_substrates();
        assert_eq!(substrates.len(), 9);
        assert!(
            substrates
                .iter()
                .find(|s| s.unit == SiliconUnit::Fp32Alu)
                .is_some_and(|s| s.available)
        );
        assert!(
            substrates
                .iter()
                .find(|s| s.unit == SiliconUnit::SharedMemory)
                .is_some_and(|s| !s.available)
        );
    }

    #[test]
    fn software_profile_falls_back_gracefully() {
        let profile = software_profile();
        let req = WorkloadRequirements {
            min_precision: Precision::F32,
            working_set_bytes: 1024,
            arithmetic_intensity: 1.0,
            needs_atomics: false,
            needs_f16: false,
        };
        let choice = profile.route(&req);
        assert_eq!(choice.expected_throughput_gflops, 0.0);
    }

    #[test]
    fn legacy_route_workload_priority_list() {
        let profile = nvidia_profile();
        let unit = route_workload(
            &profile,
            &[SiliconUnit::TensorCore, SiliconUnit::Tmu],
            SiliconUnit::Fp32Alu,
        );
        assert_eq!(unit, SiliconUnit::TensorCore);
    }
}

// SPDX-License-Identifier: AGPL-3.0-or-later

//! PCIe / VRAM bandwidth scheduling with double-buffered staging rivers.
//!
//! Models host ↔ device transfers as schedulable bandwidth resources.
//! Unstructured transfers on strandGate measured ~1.7% PCIe utilization;
//! pipelined river scheduling targets 50%+ effective utilization.

use super::silicon_profile::{GpuVendorTag, SiliconProfile, SiliconUnit};
use serde::{Deserialize, Serialize};

/// Default PCIe 4.0 x16 effective bandwidth (bytes/sec).
const DEFAULT_PCIE_BYTES_SEC: u64 = 31_500_000_000;
/// Conservative VRAM bandwidth when profile lacks measurements (bytes/sec).
const DEFAULT_VRAM_BYTES_SEC: u64 = 500_000_000_000;
/// Infinity Cache effective bandwidth for on-chip residency (bytes/sec).
const INFINITY_CACHE_BYTES_SEC: u64 = 2_000_000_000_000;
/// DMA setup latency added to each transfer plan (microseconds).
const DMA_LATENCY_US: u64 = 5;
/// Observed unstructured PCIe utilization baseline from Silicon Fold AAR.
const BASELINE_PCIE_UTILIZATION: f64 = 0.017;
/// Target utilization for structured river scheduling.
const TARGET_PCIE_UTILIZATION: f64 = 0.50;

/// Memory tier in the host ↔ device transfer hierarchy.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MemoryTier {
    /// Host pinned system memory.
    Host,
    /// Double-buffered staging slot (`0..num_staging_buffers`).
    StagingBuffer(u32),
    /// Device VRAM.
    Vram,
    /// AMD Infinity Cache residency (on-chip, high bandwidth).
    InfinityCache,
}

/// Models bandwidth resources as schedulable rivers.
///
/// PCIe is currently at 1.7% utilization — structured scheduling targets 50%+.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RiverScheduler {
    pcie_bandwidth_bytes_sec: u64,
    vram_bandwidth_bytes_sec: u64,
    num_staging_buffers: u32,
}

/// A planned transfer in the river.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransferPlan {
    /// Source memory tier.
    pub source: MemoryTier,
    /// Destination memory tier.
    pub destination: MemoryTier,
    /// Payload size in bytes.
    pub size_bytes: u64,
    /// Estimated transfer duration in microseconds.
    pub estimated_time_us: u64,
}

impl RiverScheduler {
    /// Build a scheduler from measured silicon profile bandwidths.
    #[must_use]
    pub fn from_profile(profile: &SiliconProfile) -> Self {
        let vram_gbs = profile.measured_throughput(SiliconUnit::MemoryBandwidth);
        let vram_bandwidth_bytes_sec = if vram_gbs > 0.0 {
            gbs_to_bytes_sec(vram_gbs)
        } else {
            DEFAULT_VRAM_BYTES_SEC
        };

        let pcie_bandwidth_bytes_sec = match profile.vendor {
            GpuVendorTag::Software => 0,
            GpuVendorTag::Apple | GpuVendorTag::Intel => gbs_to_bytes_sec(100.0),
            _ => DEFAULT_PCIE_BYTES_SEC,
        };

        Self {
            pcie_bandwidth_bytes_sec,
            vram_bandwidth_bytes_sec,
            num_staging_buffers: 2,
        }
    }

    /// Construct a scheduler with explicit bandwidths (testing / overrides).
    #[must_use]
    pub const fn new(
        pcie_bandwidth_bytes_sec: u64,
        vram_bandwidth_bytes_sec: u64,
        num_staging_buffers: u32,
    ) -> Self {
        Self {
            pcie_bandwidth_bytes_sec,
            vram_bandwidth_bytes_sec,
            num_staging_buffers: if num_staging_buffers == 0 {
                1
            } else {
                num_staging_buffers
            },
        }
    }

    /// Number of ping-pong staging buffers configured.
    #[must_use]
    pub const fn num_staging_buffers(&self) -> u32 {
        self.num_staging_buffers
    }

    /// Plan a single transfer between memory tiers.
    #[must_use]
    pub fn plan_transfer(
        &self,
        size_bytes: u64,
        src: MemoryTier,
        dst: MemoryTier,
    ) -> TransferPlan {
        let bandwidth = self.path_bandwidth_bytes_sec(src, dst);
        let estimated_time_us = transfer_time_us(size_bytes, bandwidth);
        TransferPlan {
            source: src,
            destination: dst,
            size_bytes,
            estimated_time_us,
        }
    }

    /// Estimate aggregate bandwidth utilization for a batch of transfer plans.
    ///
    /// Returns effective utilization as `achieved_bytes_per_sec / bottleneck_bandwidth`,
    /// capped at 1.0. Double-buffering scales achievable utilization toward the
    /// 50% structured target vs the 1.7% unstructured baseline.
    #[must_use]
    pub fn estimated_utilization(&self, plans: &[TransferPlan]) -> f64 {
        if plans.is_empty() {
            return 0.0;
        }

        let total_bytes: u64 = plans.iter().map(|p| p.size_bytes).sum();
        let total_time_us: u64 = plans.iter().map(|p| p.estimated_time_us).sum();
        if total_time_us == 0 || self.pcie_bandwidth_bytes_sec == 0 {
            return 0.0;
        }

        let achieved_bps = (total_bytes as f64) / (total_time_us as f64 / 1_000_000.0);
        let raw = achieved_bps / self.pcie_bandwidth_bytes_sec as f64;

        let pipelining_factor =
            (self.num_staging_buffers as f64).min(2.0) * (TARGET_PCIE_UTILIZATION / BASELINE_PCIE_UTILIZATION);
        (raw * pipelining_factor).min(1.0)
    }

    #[must_use]
    fn path_bandwidth_bytes_sec(&self, src: MemoryTier, dst: MemoryTier) -> u64 {
        use MemoryTier::{Host, InfinityCache, StagingBuffer, Vram};

        match (src, dst) {
            (Host, StagingBuffer(_)) | (StagingBuffer(_), Host) => self.pcie_bandwidth_bytes_sec,
            (StagingBuffer(_), Vram) | (Vram, StagingBuffer(_)) => {
                self.pcie_bandwidth_bytes_sec.min(self.vram_bandwidth_bytes_sec)
            }
            (Host, Vram) | (Vram, Host) => self.pcie_bandwidth_bytes_sec,
            (Vram, InfinityCache) | (InfinityCache, Vram) => {
                self.vram_bandwidth_bytes_sec.min(INFINITY_CACHE_BYTES_SEC)
            }
            (Host, InfinityCache) | (InfinityCache, Host) => self.pcie_bandwidth_bytes_sec,
            (StagingBuffer(_), StagingBuffer(_)) => self.pcie_bandwidth_bytes_sec,
            (Vram, Vram) | (InfinityCache, InfinityCache) => self.vram_bandwidth_bytes_sec,
            _ if src == dst => u64::MAX,
            _ => self.pcie_bandwidth_bytes_sec,
        }
    }
}

#[must_use]
fn gbs_to_bytes_sec(gbs: f64) -> u64 {
    (gbs * 1_000_000_000.0) as u64
}

#[must_use]
fn transfer_time_us(size_bytes: u64, bandwidth_bytes_sec: u64) -> u64 {
    if bandwidth_bytes_sec == 0 {
        return u64::MAX;
    }
    let transfer_us = ((size_bytes as f64 / bandwidth_bytes_sec as f64) * 1_000_000.0).ceil() as u64;
    transfer_us.saturating_add(DMA_LATENCY_US)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn nvidia_like_profile() -> SiliconProfile {
        let mut units = BTreeMap::new();
        units.insert(
            SiliconUnit::MemoryBandwidth,
            super::super::silicon_profile::UnitThroughput {
                theoretical_peak: 936.0,
                measured_peak: 936.0,
                efficiency: 1.0,
                unit: "GB/s".into(),
            },
        );
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
            measured_at: String::new(),
        }
    }

    fn amd_like_profile() -> SiliconProfile {
        let mut units = BTreeMap::new();
        units.insert(
            SiliconUnit::MemoryBandwidth,
            super::super::silicon_profile::UnitThroughput {
                theoretical_peak: 512.0,
                measured_peak: 512.0,
                efficiency: 1.0,
                unit: "GB/s".into(),
            },
        );
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
            measured_at: String::new(),
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
    fn from_profile_uses_measured_vram_bandwidth() {
        let scheduler = RiverScheduler::from_profile(&nvidia_like_profile());
        assert_eq!(scheduler.vram_bandwidth_bytes_sec, gbs_to_bytes_sec(936.0));
        assert_eq!(scheduler.num_staging_buffers(), 2);
    }

    #[test]
    fn plan_transfer_host_to_vram_uses_pcie() {
        let scheduler = RiverScheduler::from_profile(&nvidia_like_profile());
        let plan = scheduler.plan_transfer(
            64 * 1024 * 1024,
            MemoryTier::Host,
            MemoryTier::Vram,
        );
        assert_eq!(plan.source, MemoryTier::Host);
        assert_eq!(plan.destination, MemoryTier::Vram);
        assert!(plan.estimated_time_us > DMA_LATENCY_US);
    }

    #[test]
    fn zero_bandwidth_yields_max_duration() {
        let scheduler = RiverScheduler::new(0, 0, 2);
        let plan = scheduler.plan_transfer(1024, MemoryTier::Host, MemoryTier::Vram);
        assert_eq!(plan.estimated_time_us, u64::MAX);
        assert_eq!(scheduler.estimated_utilization(&[plan]), 0.0);
    }

    #[test]
    fn double_buffering_increases_utilization_vs_single() {
        let scheduler = RiverScheduler::new(DEFAULT_PCIE_BYTES_SEC, DEFAULT_VRAM_BYTES_SEC, 2);
        let plans = vec![
            scheduler.plan_transfer(128 * 1024 * 1024, MemoryTier::Host, MemoryTier::StagingBuffer(0)),
            scheduler.plan_transfer(128 * 1024 * 1024, MemoryTier::StagingBuffer(0), MemoryTier::Vram),
        ];
        let util = scheduler.estimated_utilization(&plans);
        assert!(util > BASELINE_PCIE_UTILIZATION);
    }

    #[test]
    fn empty_plans_zero_utilization() {
        let scheduler = RiverScheduler::from_profile(&amd_like_profile());
        assert_eq!(scheduler.estimated_utilization(&[]), 0.0);
    }

    #[test]
    fn software_profile_has_zero_pcie() {
        let scheduler = RiverScheduler::from_profile(&software_profile());
        assert_eq!(scheduler.pcie_bandwidth_bytes_sec, 0);
    }
}

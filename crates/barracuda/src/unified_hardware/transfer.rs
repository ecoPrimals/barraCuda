// SPDX-License-Identifier: AGPL-3.0-or-later
//! Cross-device transfer cost modelling — PCIe, NVLink, shared memory.

use super::types::HardwareType;

/// Mixed dispatch substrate — cross-device routing for GPU ↔ NPU ↔ CPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MixedSubstrate {
    GpuOnly,
    CpuOnly,
    NpuOnly,
    GpuToCpu,
    CpuToGpu,
    GpuToNpu,
    NpuToGpu,
}

/// Estimated transfer cost for cross-device data movement.
#[derive(Debug, Clone, Copy)]
pub struct TransferCost {
    pub latency_us: f64,
    pub bandwidth_gbps: f64,
}

impl TransferCost {
    #[must_use]
    pub fn estimated_us(&self, bytes: usize) -> f64 {
        self.latency_us + (bytes as f64) / (self.bandwidth_gbps * 1000.0)
    }
}

/// PCIe 4.0 x16 bandwidth in GB/s.
pub const PCIE4_X16_BANDWIDTH_GBPS: f64 = 31.5;

/// Estimated latency for PCIe DMA transfer in microseconds.
pub const PCIE_DMA_LATENCY_US: f64 = 5.0;

/// Empirical GPU dispatch overhead in microseconds (queue submit + readback).
pub const GPU_DISPATCH_OVERHEAD_US: f64 = 1500.0;

/// PCIe/interconnect bandwidth tier for transfer cost estimation.
///
/// Runtime-detected from GPU adapter name heuristics. Used by
/// `dispatch_with_transfer_cost()` to factor data movement cost
/// into the CPU/GPU dispatch decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BandwidthTier {
    /// PCIe 3.0 x16 — ~15.75 GB/s (Titan V, RTX 2000 series)
    PciE3x16,
    /// PCIe 4.0 x16 — ~31.5 GB/s (RTX 3000/4000, RX 6000/7000)
    PciE4x16,
    /// PCIe 5.0 x16 — ~63 GB/s (next-gen data center)
    PciE5x16,
    /// NVLink — ~300 GB/s (A100, H100 multi-GPU)
    NvLink,
    /// Shared/unified memory — effectively infinite (Apple M-series, integrated)
    SharedMemory,
    /// Unknown — conservative fallback (PCIe 3.0 assumptions)
    Unknown,
}

impl BandwidthTier {
    #[must_use]
    pub const fn bandwidth_gbps(self) -> f64 {
        match self {
            Self::PciE3x16 => 15.75,
            Self::PciE4x16 => 31.5,
            Self::PciE5x16 => 63.0,
            Self::NvLink => 300.0,
            Self::SharedMemory => 1000.0,
            Self::Unknown => 15.75,
        }
    }

    #[must_use]
    pub const fn latency_us(self) -> f64 {
        match self {
            Self::SharedMemory => 0.1,
            Self::NvLink => 1.0,
            _ => PCIE_DMA_LATENCY_US,
        }
    }

    #[must_use]
    pub const fn transfer_cost(self) -> TransferCost {
        TransferCost {
            latency_us: self.latency_us(),
            bandwidth_gbps: self.bandwidth_gbps(),
        }
    }

    /// Detect bandwidth tier from GPU adapter name (heuristic).
    #[must_use]
    pub fn detect_from_adapter_name(name: &str) -> Self {
        let lower = name.to_lowercase();

        if lower.contains("a100") || lower.contains("h100") || lower.contains("h200") {
            return Self::NvLink;
        }
        if lower.contains("apple") || lower.contains("llvmpipe") || lower.contains("swiftshader") {
            return Self::SharedMemory;
        }
        if lower.contains("b100") || lower.contains("b200") {
            return Self::PciE5x16;
        }
        if lower.contains("rtx 30")
            || lower.contains("rtx 40")
            || lower.contains("rx 6")
            || lower.contains("rx 7")
            || lower.contains("arc")
            || lower.contains("a770")
            || lower.contains("a750")
            || lower.contains("mi2")
            || lower.contains("mi3")
        {
            return Self::PciE4x16;
        }
        if lower.contains("rtx 20")
            || lower.contains("titan v")
            || lower.contains("v100")
            || lower.contains("gv100")
        {
            return Self::PciE3x16;
        }
        Self::Unknown
    }
}

/// Select optimal mixed substrate for a workload (default PCIe 4.0).
#[must_use]
pub fn mixed_substrate(
    compute_us: f64,
    data_bytes: usize,
    source: HardwareType,
    target: HardwareType,
) -> MixedSubstrate {
    mixed_substrate_with_tier(
        compute_us,
        data_bytes,
        source,
        target,
        BandwidthTier::PciE4x16,
    )
}

/// Select optimal mixed substrate with explicit bandwidth tier.
#[must_use]
pub fn mixed_substrate_with_tier(
    compute_us: f64,
    data_bytes: usize,
    source: HardwareType,
    target: HardwareType,
    tier: BandwidthTier,
) -> MixedSubstrate {
    if source == target {
        return match source {
            HardwareType::GPU => MixedSubstrate::GpuOnly,
            HardwareType::CPU => MixedSubstrate::CpuOnly,
            HardwareType::NPU => MixedSubstrate::NpuOnly,
            _ => MixedSubstrate::CpuOnly,
        };
    }

    let cost = tier.transfer_cost();
    let transfer_us = cost.estimated_us(data_bytes) + GPU_DISPATCH_OVERHEAD_US;

    if compute_us > transfer_us {
        match (source, target) {
            (HardwareType::GPU, HardwareType::CPU) => MixedSubstrate::GpuToCpu,
            (HardwareType::CPU, HardwareType::GPU) => MixedSubstrate::CpuToGpu,
            (HardwareType::GPU, HardwareType::NPU) => MixedSubstrate::GpuToNpu,
            (HardwareType::NPU, HardwareType::GPU) => MixedSubstrate::NpuToGpu,
            _ => match source {
                HardwareType::GPU => MixedSubstrate::GpuOnly,
                HardwareType::NPU => MixedSubstrate::NpuOnly,
                _ => MixedSubstrate::CpuOnly,
            },
        }
    } else {
        match source {
            HardwareType::GPU => MixedSubstrate::GpuOnly,
            HardwareType::CPU => MixedSubstrate::CpuOnly,
            HardwareType::NPU => MixedSubstrate::NpuOnly,
            _ => MixedSubstrate::CpuOnly,
        }
    }
}

/// PCIe bridge for GPU↔NPU transfer cost estimation.
pub struct PcieBridge {
    pub p2p_available: bool,
    pub source_label: String,
    pub target_label: String,
}

impl PcieBridge {
    #[must_use]
    pub fn detect_p2p() -> Self {
        Self {
            p2p_available: false,
            source_label: "unknown".to_string(),
            target_label: "unknown".to_string(),
        }
    }

    #[must_use]
    pub fn transfer_cost(&self, _bytes: usize) -> TransferCost {
        TransferCost {
            latency_us: PCIE_DMA_LATENCY_US,
            bandwidth_gbps: PCIE4_X16_BANDWIDTH_GBPS,
        }
    }
}

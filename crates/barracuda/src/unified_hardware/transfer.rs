// SPDX-License-Identifier: AGPL-3.0-or-later
//! Cross-device transfer cost modelling — `PCIe`, `NVLink`, shared memory.

use super::types::HardwareType;

/// Mixed dispatch substrate — cross-device routing for GPU ↔ NPU ↔ CPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MixedSubstrate {
    /// Compute and data remain on GPU
    GpuOnly,
    /// Compute and data remain on CPU
    CpuOnly,
    /// Compute and data remain on NPU
    NpuOnly,
    /// Data transfer from GPU to CPU for compute
    GpuToCpu,
    /// Data transfer from CPU to GPU for compute
    CpuToGpu,
    /// Data transfer from GPU to NPU for compute
    GpuToNpu,
    /// Data transfer from NPU to GPU for compute
    NpuToGpu,
}

/// Estimated transfer cost for cross-device data movement.
#[derive(Debug, Clone, Copy)]
pub struct TransferCost {
    /// One-way latency in microseconds
    pub latency_us: f64,
    /// Bandwidth in GB/s
    pub bandwidth_gbps: f64,
}

impl TransferCost {
    /// Estimate total transfer time in microseconds for `bytes` bytes.
    #[must_use]
    pub fn estimated_us(&self, bytes: usize) -> f64 {
        self.latency_us + (bytes as f64) / (self.bandwidth_gbps * 1000.0)
    }
}

/// `PCIe` 4.0 x16 bandwidth in GB/s.
pub const PCIE4_X16_BANDWIDTH_GBPS: f64 = 31.5;

/// Estimated latency for `PCIe` DMA transfer in microseconds.
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
    /// `PCIe` 3.0 x16 — ~15.75 GB/s (Titan V, RTX 2000 series)
    PciE3x16,
    /// `PCIe` 4.0 x16 — ~31.5 GB/s (RTX 3000/4000, RX 6000/7000)
    PciE4x16,
    /// `PCIe` 5.0 x16 — ~63 GB/s (next-gen data center)
    PciE5x16,
    /// `NVLink` — ~300 GB/s (A100, H100 multi-GPU)
    NvLink,
    /// Shared/unified memory — effectively infinite (Apple M-series, integrated)
    SharedMemory,
    /// Unknown — conservative fallback (`PCIe` 3.0 assumptions)
    Unknown,
}

impl BandwidthTier {
    /// Bandwidth in GB/s for this interconnect tier.
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

    /// One-way transfer latency in microseconds.
    #[must_use]
    pub const fn latency_us(self) -> f64 {
        match self {
            Self::SharedMemory => 0.1,
            Self::NvLink => 1.0,
            _ => PCIE_DMA_LATENCY_US,
        }
    }

    /// Transfer cost (latency + bandwidth) for this tier.
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

/// Select optimal mixed substrate for a workload (default `PCIe` 4.0).
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

/// `PCIe` bridge for GPU↔NPU transfer cost estimation.
pub struct PcieBridge {
    /// Whether peer-to-peer DMA is available
    pub p2p_available: bool,
    /// Source device label for diagnostics
    pub source_label: String,
    /// Target device label for diagnostics
    pub target_label: String,
    /// Detected bandwidth tier for this bridge
    pub tier: BandwidthTier,
}

impl PcieBridge {
    /// Detect P2P availability between GPU and NPU.
    ///
    /// On Linux, probes sysfs for GPU PCI devices and checks whether they
    /// share a NUMA node (heuristic for P2P capability). Falls back to
    /// `p2p_available: false` on non-Linux or when sysfs is unavailable.
    #[must_use]
    pub fn detect_p2p() -> Self {
        let probed = PcieLinkInfo::probe_all_gpus();
        let p2p_available = Self::any_shared_numa(&probed);
        let tier = probed
            .first()
            .map_or(BandwidthTier::Unknown, PcieLinkInfo::bandwidth_tier);
        Self {
            p2p_available,
            source_label: probed
                .first()
                .map_or_else(|| "gpu".to_string(), |i| i.bdf_address.clone()),
            target_label: probed
                .get(1)
                .map_or_else(|| "npu".to_string(), |i| i.bdf_address.clone()),
            tier,
        }
    }

    /// Detect P2P with adapter-aware bandwidth tier.
    #[must_use]
    pub fn detect_with_adapter(adapter_name: &str) -> Self {
        let probed = PcieLinkInfo::probe_all_gpus();
        let tier = probed
            .iter()
            .find(|p| {
                adapter_name
                    .to_lowercase()
                    .contains(&p.bdf_address.to_lowercase())
            })
            .map_or_else(
                || BandwidthTier::detect_from_adapter_name(adapter_name),
                PcieLinkInfo::bandwidth_tier,
            );
        Self {
            p2p_available: Self::any_shared_numa(&probed),
            source_label: "gpu".to_string(),
            target_label: "npu".to_string(),
            tier,
        }
    }

    /// Return transfer cost using the detected bandwidth tier.
    #[must_use]
    pub fn transfer_cost(&self) -> TransferCost {
        self.tier.transfer_cost()
    }

    /// All probed `PCIe` GPU links from sysfs (Linux only).
    #[must_use]
    pub fn probe_gpu_links() -> Vec<PcieLinkInfo> {
        PcieLinkInfo::probe_all_gpus()
    }

    fn any_shared_numa(links: &[PcieLinkInfo]) -> bool {
        if links.len() < 2 {
            return false;
        }
        for (i, a) in links.iter().enumerate() {
            for b in &links[i + 1..] {
                if let (Some(na), Some(nb)) = (a.numa_node, b.numa_node) {
                    if na == nb {
                        return true;
                    }
                }
            }
        }
        false
    }
}

/// Runtime-probed `PCIe` link characteristics for a single device.
#[derive(Debug, Clone)]
pub struct PcieLinkInfo {
    /// Bus:Device.Function address (e.g. "0000:01:00.0").
    pub bdf_address: String,
    /// `PCIe` generation (1..=6). 0 means unknown.
    pub pcie_gen: u8,
    /// Lane width (1, 2, 4, 8, 16). 0 means unknown.
    pub lane_width: u8,
    /// NUMA node affinity, if detectable.
    pub numa_node: Option<u32>,
    /// Vendor ID (e.g. 0x10de for NVIDIA, 0x1002 for AMD).
    pub vendor_id: u32,
}

impl PcieLinkInfo {
    /// Theoretical unidirectional bandwidth in GB/s for this link.
    #[must_use]
    pub fn bandwidth_gbps(&self) -> f64 {
        let per_lane = match self.pcie_gen {
            1 => 0.25,
            2 => 0.5,
            3 => 0.985,
            4 => 1.969,
            5 => 3.938,
            6 => 7.563,
            _ => 0.985,
        };
        per_lane * f64::from(self.lane_width)
    }

    /// Map to the closest `BandwidthTier`.
    #[must_use]
    pub fn bandwidth_tier(&self) -> BandwidthTier {
        let bw = self.bandwidth_gbps();
        if bw >= 50.0 {
            BandwidthTier::PciE5x16
        } else if bw >= 25.0 {
            BandwidthTier::PciE4x16
        } else if bw >= 10.0 {
            BandwidthTier::PciE3x16
        } else {
            BandwidthTier::Unknown
        }
    }

    /// Probe all GPU-class devices from sysfs (Linux).
    ///
    /// On non-Linux platforms, returns an empty vec.
    #[must_use]
    pub fn probe_all_gpus() -> Vec<Self> {
        #[cfg(target_os = "linux")]
        {
            Self::probe_sysfs()
        }
        #[cfg(not(target_os = "linux"))]
        {
            Vec::new()
        }
    }

    #[cfg(target_os = "linux")]
    fn probe_sysfs() -> Vec<Self> {
        let pci_dir = std::path::Path::new("/sys/bus/pci/devices");
        let Ok(entries) = std::fs::read_dir(pci_dir) else {
            return Vec::new();
        };

        let mut results = Vec::new();
        for entry in entries.flatten() {
            let path = entry.path();
            let bdf = entry.file_name().to_string_lossy().to_string();

            let Some(class) = read_sysfs_hex(&path.join("class")) else {
                continue;
            };
            // PCI class 0x03xxxx = display controller (covers VGA + 3D)
            if (class >> 16) != 0x03 {
                continue;
            }

            let vendor_id = read_sysfs_hex(&path.join("vendor")).unwrap_or(0);
            let pcie_gen = parse_pcie_gen(&path.join("current_link_speed"));
            let lane_width = parse_lane_width(&path.join("current_link_width"));
            let numa_node = read_sysfs_i32(&path.join("numa_node"))
                .and_then(|n| if n >= 0 { Some(n as u32) } else { None });

            results.push(Self {
                bdf_address: bdf,
                pcie_gen,
                lane_width,
                numa_node,
                vendor_id,
            });
        }
        results
    }
}

#[cfg(target_os = "linux")]
fn read_sysfs_hex(path: &std::path::Path) -> Option<u32> {
    let text = std::fs::read_to_string(path).ok()?;
    let trimmed = text.trim().trim_start_matches("0x");
    u32::from_str_radix(trimmed, 16).ok()
}

#[cfg(target_os = "linux")]
fn read_sysfs_i32(path: &std::path::Path) -> Option<i32> {
    let text = std::fs::read_to_string(path).ok()?;
    text.trim().parse().ok()
}

#[cfg(target_os = "linux")]
fn parse_pcie_gen(path: &std::path::Path) -> u8 {
    let Ok(text) = std::fs::read_to_string(path) else {
        return 0;
    };
    let lower = text.trim().to_lowercase();
    const MARKERS: &[(&str, &str, u8)] = &[
        ("64.0", "gen6", 6),
        ("32.0", "gen5", 5),
        ("16.0", "gen4", 4),
        ("8.0", "gen3", 3),
        ("5.0", "gen2", 2),
        ("2.5", "gen1", 1),
    ];
    for &(speed, gen_label, version) in MARKERS {
        if lower.contains(speed) || lower.contains(gen_label) {
            return version;
        }
    }
    0
}

#[cfg(target_os = "linux")]
fn parse_lane_width(path: &std::path::Path) -> u8 {
    let Ok(text) = std::fs::read_to_string(path) else {
        return 0;
    };
    text.trim().parse().unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transfer_cost_estimated_us() {
        let cost = TransferCost {
            latency_us: 5.0,
            bandwidth_gbps: 31.5,
        };
        let us = cost.estimated_us(1_000_000);
        assert!(us > 5.0);
        assert!(us < 100.0);
    }

    #[test]
    fn test_bandwidth_tier_detect_additional_adapters() {
        assert_eq!(
            BandwidthTier::detect_from_adapter_name("NVIDIA H200"),
            BandwidthTier::NvLink
        );
        assert_eq!(
            BandwidthTier::detect_from_adapter_name("NVIDIA B100"),
            BandwidthTier::PciE5x16
        );
        assert_eq!(
            BandwidthTier::detect_from_adapter_name("SwiftShader"),
            BandwidthTier::SharedMemory
        );
        assert_eq!(
            BandwidthTier::detect_from_adapter_name("Intel Arc A770"),
            BandwidthTier::PciE4x16
        );
        assert_eq!(
            BandwidthTier::detect_from_adapter_name("AMD MI250"),
            BandwidthTier::PciE4x16
        );
    }

    #[test]
    fn test_bandwidth_tier_latency() {
        assert!((BandwidthTier::SharedMemory.latency_us() - 0.1).abs() < 0.01);
        assert!((BandwidthTier::NvLink.latency_us() - 1.0).abs() < 0.01);
        assert!((BandwidthTier::PciE4x16.latency_us() - PCIE_DMA_LATENCY_US).abs() < 0.01);
    }

    #[test]
    fn test_mixed_substrate_gpu_to_npu() {
        let sub = mixed_substrate_with_tier(
            100_000.0,
            1_048_576,
            HardwareType::GPU,
            HardwareType::NPU,
            BandwidthTier::PciE4x16,
        );
        assert_eq!(sub, MixedSubstrate::GpuToNpu);
    }

    #[test]
    fn test_mixed_substrate_npu_to_gpu() {
        let sub = mixed_substrate_with_tier(
            100_000.0,
            1_048_576,
            HardwareType::NPU,
            HardwareType::GPU,
            BandwidthTier::PciE4x16,
        );
        assert_eq!(sub, MixedSubstrate::NpuToGpu);
    }

    #[test]
    fn test_mixed_substrate_unknown_hardware_type() {
        let sub = mixed_substrate_with_tier(
            100.0,
            1024,
            HardwareType::Custom,
            HardwareType::Custom,
            BandwidthTier::PciE4x16,
        );
        assert_eq!(sub, MixedSubstrate::CpuOnly);
    }
}

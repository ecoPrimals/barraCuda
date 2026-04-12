// SPDX-License-Identifier: AGPL-3.0-or-later
//! Linux sysfs `PCIe` link probing for GPU devices.
//!
//! Provides [`PcieLinkInfo`] — runtime-detected PCIe generation, lane width,
//! NUMA affinity, and vendor ID for each GPU-class PCI device on the bus.
//! Used by [`super::transfer::PcieBridge`] for transfer cost estimation.

use super::transfer::BandwidthTier;

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

        entries
            .flatten()
            .filter_map(|entry| {
                let path = entry.path();
                let class = read_sysfs_hex(&path.join("class"))?;
                // PCI class 0x03xxxx = display controller (covers VGA + 3D)
                (class >> 16 == 0x03).then(|| Self {
                    bdf_address: entry.file_name().to_string_lossy().into_owned(),
                    pcie_gen: parse_pcie_gen(&path.join("current_link_speed")),
                    lane_width: parse_lane_width(&path.join("current_link_width")),
                    numa_node: read_sysfs_i32(&path.join("numa_node"))
                        .and_then(|n| u32::try_from(n).ok()),
                    vendor_id: read_sysfs_hex(&path.join("vendor")).unwrap_or(0),
                })
            })
            .collect()
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

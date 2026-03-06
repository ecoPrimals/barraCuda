// SPDX-License-Identifier: AGPL-3.0-or-later

//! `PCIe` interconnect topology for cross-substrate transfer routing.
//!
//! Models the physical interconnect between substrates so pipeline stages
//! can route data through the lowest-latency path. The key optimization
//! is NPU→GPU transfers via `PCIe` peer-to-peer (P2P), bypassing CPU host
//! memory bounce.
//!
//! # Bandwidth tiers
//!
//! | Tier | Bandwidth | Latency | Example |
//! |------|-----------|---------|---------|
//! | Local | ∞ | 0 | Same device |
//! | `NvLink` | 300 GB/s | ~1µs | Multi-GPU `NvLink` bridge |
//! | `PciePeer` | 15.8 GB/s | ~5µs | `PCIe` 4.0 x16 P2P |
//! | `PcieHost` | 15.8 GB/s | ~50µs | `PCIe` 4.0 via CPU bounce |
//! | `PcieLow` | 0.5 GB/s | ~100µs | `PCIe` 2.0 x1 (AKD1000) |
//! | Network | varies | ~1ms | LAN via NUCLEUS |
//!
//! Absorbed from groundSpring V61 `metalForge/forge/src/topology.rs`.

use crate::device::substrate::{Substrate, SubstrateType};

/// Bandwidth tier between two substrates.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BandwidthTier {
    /// Same device — no transfer needed.
    Local,
    /// NvLink/NvSwitch — GPU-to-GPU high bandwidth.
    NvLink,
    /// `PCIe` peer-to-peer — direct DMA between devices (bypasses CPU).
    PciePeer,
    /// `PCIe` via host — data bounces through CPU main memory.
    PcieHost,
    /// Low-bandwidth `PCIe` (e.g. AKD1000 at `PCIe` 2.0 x1).
    PcieLow,
    /// Network transfer via NUCLEUS LAN.
    Network,
}

impl BandwidthTier {
    /// Estimated transfer time for `bytes` at this tier, in microseconds.
    #[must_use]
    pub const fn transfer_time_us(self, bytes: u64) -> u64 {
        let (bw_mbps, latency_us): (u64, u64) = match self {
            Self::Local => return 0,
            Self::NvLink => (300_000, 1),
            Self::PciePeer => (15_800, 5),
            Self::PcieHost => (15_800, 50),
            Self::PcieLow => (500, 100),
            Self::Network => (1_000, 1_000),
        };
        let mb = bytes / (1024 * 1024);
        let transfer = if bw_mbps > 0 {
            mb * 1_000_000 / bw_mbps
        } else {
            0
        };
        latency_us + transfer
    }

    /// Whether this tier supports peer-to-peer DMA (bypasses CPU).
    #[must_use]
    pub const fn is_peer_to_peer(self) -> bool {
        matches!(self, Self::Local | Self::NvLink | Self::PciePeer)
    }

    /// Human-readable label.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Local => "local",
            Self::NvLink => "nvlink",
            Self::PciePeer => "pcie-p2p",
            Self::PcieHost => "pcie-host",
            Self::PcieLow => "pcie-low",
            Self::Network => "network",
        }
    }
}

/// A link between two substrates with transfer characteristics.
#[derive(Debug, Clone)]
pub struct Link {
    /// Source substrate index.
    pub from: usize,
    /// Destination substrate index.
    pub to: usize,
    /// Bandwidth tier for this link.
    pub tier: BandwidthTier,
}

/// Device interconnect topology graph.
#[derive(Debug, Clone)]
pub struct InterconnectTopology {
    links: Vec<Link>,
}

impl InterconnectTopology {
    /// Infer topology from an inventory of substrates.
    ///
    /// Uses substrate types and device names to determine connectivity.
    /// When PCI topology is unavailable, falls back to conservative
    /// estimates based on device types.
    #[must_use]
    pub fn infer(substrates: &[Substrate]) -> Self {
        let mut links = Vec::new();
        for (i, src) in substrates.iter().enumerate() {
            for (j, dst) in substrates.iter().enumerate() {
                if i == j {
                    continue;
                }
                let tier = infer_link_tier(src, dst);
                links.push(Link {
                    from: i,
                    to: j,
                    tier,
                });
            }
        }
        Self { links }
    }

    /// Find the best (lowest-latency) link between two substrates.
    #[must_use]
    pub fn best_link(&self, from: usize, to: usize) -> Option<&Link> {
        self.links
            .iter()
            .filter(|l| l.from == from && l.to == to)
            .min_by_key(|l| l.tier)
    }

    /// All links from a given substrate.
    #[must_use]
    pub fn links_from(&self, from: usize) -> Vec<&Link> {
        self.links.iter().filter(|l| l.from == from).collect()
    }

    /// Whether a peer-to-peer path exists between two substrates.
    #[must_use]
    pub fn has_p2p(&self, from: usize, to: usize) -> bool {
        self.best_link(from, to)
            .is_some_and(|l| l.tier.is_peer_to_peer())
    }

    /// Estimated transfer time between two substrates for `bytes` of data.
    #[must_use]
    pub fn transfer_time_us(&self, from: usize, to: usize, bytes: u64) -> u64 {
        self.best_link(from, to)
            .map_or(u64::MAX, |l| l.tier.transfer_time_us(bytes))
    }

    /// All P2P-capable pairs.
    #[must_use]
    pub fn p2p_pairs(&self) -> Vec<(usize, usize)> {
        self.links
            .iter()
            .filter(|l| l.tier.is_peer_to_peer())
            .map(|l| (l.from, l.to))
            .collect()
    }
}

fn is_gpu(st: SubstrateType) -> bool {
    matches!(
        st,
        SubstrateType::NvidiaGpu
            | SubstrateType::AmdGpu
            | SubstrateType::IntelGpu
            | SubstrateType::AppleGpu
    )
}

fn infer_link_tier(src: &Substrate, dst: &Substrate) -> BandwidthTier {
    let (s, d) = (src.substrate_type, dst.substrate_type);
    match (is_gpu(s), is_gpu(d), s, d) {
        (true, true, _, _) => {
            if is_nvlink_pair(src, dst) {
                BandwidthTier::NvLink
            } else {
                BandwidthTier::PciePeer
            }
        }
        (true, _, _, SubstrateType::Npu) | (_, true, SubstrateType::Npu, _) => {
            if is_low_bandwidth_npu(src) || is_low_bandwidth_npu(dst) {
                BandwidthTier::PcieLow
            } else {
                BandwidthTier::PciePeer
            }
        }
        _ => BandwidthTier::PcieHost,
    }
}

fn is_nvlink_pair(a: &Substrate, b: &Substrate) -> bool {
    if a.substrate_type != SubstrateType::NvidiaGpu || b.substrate_type != SubstrateType::NvidiaGpu
    {
        return false;
    }
    let a_up = a.name.to_uppercase();
    let b_up = b.name.to_uppercase();
    (a_up.contains("V100") || a_up.contains("A100"))
        && (b_up.contains("V100") || b_up.contains("A100"))
}

fn is_low_bandwidth_npu(s: &Substrate) -> bool {
    s.substrate_type == SubstrateType::Npu && s.name.to_uppercase().contains("AKD1000")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::substrate::SubstrateCapability;

    fn gpu(name: &str) -> Substrate {
        Substrate {
            substrate_type: SubstrateType::NvidiaGpu,
            name: name.to_string(),
            backend: "Vulkan".to_string(),
            index: 0,
            capabilities: vec![
                SubstrateCapability::F64Compute,
                SubstrateCapability::ShaderDispatch,
            ],
        }
    }

    fn npu() -> Substrate {
        Substrate {
            substrate_type: SubstrateType::Npu,
            name: "BrainChip AKD1000".to_string(),
            backend: "PCIe".to_string(),
            index: 0,
            capabilities: vec![SubstrateCapability::QuantizedInference { bits: 8 }],
        }
    }

    fn cpu() -> Substrate {
        Substrate {
            substrate_type: SubstrateType::Cpu,
            name: "CPU".to_string(),
            backend: "native".to_string(),
            index: 0,
            capabilities: vec![SubstrateCapability::F64Compute],
        }
    }

    #[test]
    fn gpu_to_npu_is_pcie_low_for_akd1000() {
        assert_eq!(
            infer_link_tier(&gpu("TITAN V"), &npu()),
            BandwidthTier::PcieLow
        );
    }

    #[test]
    fn gpu_to_gpu_is_pcie_peer() {
        assert_eq!(
            infer_link_tier(&gpu("TITAN V"), &gpu("RTX 4070")),
            BandwidthTier::PciePeer
        );
    }

    #[test]
    fn cpu_to_gpu_is_pcie_host() {
        assert_eq!(
            infer_link_tier(&cpu(), &gpu("RTX 4070")),
            BandwidthTier::PcieHost
        );
    }

    #[test]
    fn local_has_zero_latency() {
        assert_eq!(BandwidthTier::Local.transfer_time_us(1024), 0);
    }

    #[test]
    fn pcie_peer_is_p2p() {
        assert!(BandwidthTier::PciePeer.is_peer_to_peer());
        assert!(!BandwidthTier::PcieHost.is_peer_to_peer());
    }

    #[test]
    fn topology_infer_creates_links() {
        let subs = vec![gpu("TITAN V"), npu(), cpu()];
        let topo = InterconnectTopology::infer(&subs);
        assert_eq!(topo.links.len(), 6);
    }

    #[test]
    fn topology_has_p2p_between_gpus() {
        let subs = vec![gpu("TITAN V"), gpu("RTX 4070")];
        let topo = InterconnectTopology::infer(&subs);
        assert!(topo.has_p2p(0, 1));
    }

    #[test]
    fn topology_no_p2p_via_cpu() {
        let subs = vec![gpu("TITAN V"), cpu()];
        let topo = InterconnectTopology::infer(&subs);
        assert!(!topo.has_p2p(0, 1));
    }

    #[test]
    fn transfer_time_increases_with_data() {
        let small = BandwidthTier::PcieHost.transfer_time_us(1024);
        let large = BandwidthTier::PcieHost.transfer_time_us(1024 * 1024 * 1024);
        assert!(large > small);
    }

    #[test]
    fn bandwidth_tier_labels() {
        assert_eq!(BandwidthTier::Local.label(), "local");
        assert_eq!(BandwidthTier::PciePeer.label(), "pcie-p2p");
        assert_eq!(BandwidthTier::Network.label(), "network");
    }

    #[test]
    fn p2p_pairs_found() {
        let subs = vec![gpu("TITAN V"), gpu("RTX 4070"), cpu()];
        let topo = InterconnectTopology::infer(&subs);
        let pairs = topo.p2p_pairs();
        assert!(pairs.contains(&(0, 1)));
        assert!(pairs.contains(&(1, 0)));
        assert!(!pairs.iter().any(|&(a, b)| a == 2 || b == 2));
    }
}

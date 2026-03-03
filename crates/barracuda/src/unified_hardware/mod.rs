//! Unified Hardware Base — Universal Compute Abstraction
//!
//! Hardware executes math, math doesn't know hardware.
//!
//! - Trait-based execution (CPU, GPU, TPU, NPU implement same traits)
//! - Runtime capability discovery (query what hardware can do)
//! - Smart scheduling (match operations to best hardware)
//! - Cross-device transfer cost modelling (PCIe, NVLink, shared memory)

pub(crate) mod cpu_executor;
mod discovery;
mod scheduler;
mod traits;
mod transfer;
mod types;

// Re-export public API — all existing `use crate::unified_hardware::*` paths
// continue to work unchanged.

pub use discovery::HardwareDiscovery;
pub use scheduler::ComputeScheduler;
pub use traits::{ComputeExecutor, TensorStorage};
pub use transfer::{
    mixed_substrate, mixed_substrate_with_tier, BandwidthTier, MixedSubstrate, PcieBridge,
    TransferCost, GPU_DISPATCH_OVERHEAD_US, PCIE4_X16_BANDWIDTH_GBPS, PCIE_DMA_LATENCY_US,
};
pub use types::{
    HardwareCapabilities, HardwareType, MemoryCapabilities, OperationCapabilities,
    ParallelismCapabilities, PerformanceCapabilities, PrecisionCapabilities,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hardware_type() {
        assert_eq!(HardwareType::CPU, HardwareType::CPU);
        assert_ne!(HardwareType::CPU, HardwareType::GPU);
    }

    #[tokio::test]
    async fn test_hardware_discovery() {
        let executors = HardwareDiscovery::discover_all().await.unwrap();
        assert!(!executors.is_empty());
        assert!(executors
            .iter()
            .any(|e| e.hardware_type() == HardwareType::CPU));
    }

    #[test]
    fn test_cpu_executor() {
        let cpu = cpu_executor::CpuExecutor::new();
        assert_eq!(cpu.name(), "CPU (Native)");
        assert_eq!(cpu.hardware_type(), HardwareType::CPU);
    }

    #[test]
    fn test_mixed_substrate_same_device() {
        assert_eq!(
            mixed_substrate(100.0, 1024, HardwareType::GPU, HardwareType::GPU),
            MixedSubstrate::GpuOnly
        );
        assert_eq!(
            mixed_substrate(100.0, 1024, HardwareType::CPU, HardwareType::CPU),
            MixedSubstrate::CpuOnly
        );
        assert_eq!(
            mixed_substrate(100.0, 1024, HardwareType::NPU, HardwareType::NPU),
            MixedSubstrate::NpuOnly
        );
    }

    #[test]
    fn test_mixed_substrate_small_compute_stays_on_source() {
        let sub = mixed_substrate(100.0, 1024, HardwareType::CPU, HardwareType::GPU);
        assert_eq!(sub, MixedSubstrate::CpuOnly);
    }

    #[test]
    fn test_mixed_substrate_large_compute_transfers_to_target() {
        let sub = mixed_substrate(100_000.0, 1_048_576, HardwareType::CPU, HardwareType::GPU);
        assert_eq!(sub, MixedSubstrate::CpuToGpu);
    }

    #[test]
    fn test_mixed_substrate_gpu_to_cpu() {
        let sub = mixed_substrate(100_000.0, 1_048_576, HardwareType::GPU, HardwareType::CPU);
        assert_eq!(sub, MixedSubstrate::GpuToCpu);
    }

    #[test]
    fn test_pcie_bridge_detect_p2p_returns_false() {
        let bridge = PcieBridge::detect_p2p();
        assert!(!bridge.p2p_available);
        assert_eq!(bridge.source_label, "unknown");
        assert_eq!(bridge.target_label, "unknown");
    }

    #[test]
    fn test_pcie_bridge_transfer_cost() {
        let bridge = PcieBridge::detect_p2p();
        let cost = bridge.transfer_cost(1_048_576);
        assert!(cost.latency_us > 0.0);
        assert!(cost.bandwidth_gbps > 0.0);
        assert!(cost.estimated_us(1_048_576) > cost.latency_us);
    }

    #[test]
    fn test_bandwidth_tier_values() {
        assert!((BandwidthTier::PciE3x16.bandwidth_gbps() - 15.75).abs() < 0.01);
        assert!((BandwidthTier::PciE4x16.bandwidth_gbps() - 31.5).abs() < 0.01);
        assert!((BandwidthTier::PciE5x16.bandwidth_gbps() - 63.0).abs() < 0.01);
        assert!(BandwidthTier::NvLink.bandwidth_gbps() > 200.0);
        assert!(BandwidthTier::SharedMemory.bandwidth_gbps() > 500.0);
    }

    #[test]
    fn test_bandwidth_tier_detect_nvidia() {
        assert_eq!(
            BandwidthTier::detect_from_adapter_name("NVIDIA GeForce RTX 4070"),
            BandwidthTier::PciE4x16
        );
        assert_eq!(
            BandwidthTier::detect_from_adapter_name("NVIDIA GeForce RTX 3090"),
            BandwidthTier::PciE4x16
        );
        assert_eq!(
            BandwidthTier::detect_from_adapter_name("NVIDIA GeForce RTX 2080 Ti"),
            BandwidthTier::PciE3x16
        );
        assert_eq!(
            BandwidthTier::detect_from_adapter_name("NVIDIA TITAN V"),
            BandwidthTier::PciE3x16
        );
        assert_eq!(
            BandwidthTier::detect_from_adapter_name("NVIDIA A100"),
            BandwidthTier::NvLink
        );
        assert_eq!(
            BandwidthTier::detect_from_adapter_name("NVIDIA H100"),
            BandwidthTier::NvLink
        );
    }

    #[test]
    fn test_bandwidth_tier_detect_amd() {
        assert_eq!(
            BandwidthTier::detect_from_adapter_name("AMD Radeon RX 7900 XTX"),
            BandwidthTier::PciE4x16
        );
        assert_eq!(
            BandwidthTier::detect_from_adapter_name("AMD Radeon RX 6950 XT"),
            BandwidthTier::PciE4x16
        );
    }

    #[test]
    fn test_bandwidth_tier_detect_special() {
        assert_eq!(
            BandwidthTier::detect_from_adapter_name("Apple M2 Pro"),
            BandwidthTier::SharedMemory
        );
        assert_eq!(
            BandwidthTier::detect_from_adapter_name("llvmpipe"),
            BandwidthTier::SharedMemory
        );
        assert_eq!(
            BandwidthTier::detect_from_adapter_name("Unknown GPU XYZ"),
            BandwidthTier::Unknown
        );
    }

    #[test]
    fn test_bandwidth_tier_transfer_cost() {
        let cost = BandwidthTier::PciE4x16.transfer_cost();
        assert!((cost.bandwidth_gbps - 31.5).abs() < 0.01);
        let one_mb_us = cost.estimated_us(1_048_576);
        assert!(one_mb_us > cost.latency_us);
    }

    #[test]
    fn test_mixed_substrate_with_tier_shared_memory_transfers() {
        let sub = mixed_substrate_with_tier(
            5_000.0,
            1_048_576,
            HardwareType::CPU,
            HardwareType::GPU,
            BandwidthTier::SharedMemory,
        );
        assert_eq!(sub, MixedSubstrate::CpuToGpu);
    }

    #[test]
    fn test_mixed_substrate_with_tier_pcie3_avoids_small_transfer() {
        let sub = mixed_substrate_with_tier(
            100.0,
            1024,
            HardwareType::CPU,
            HardwareType::GPU,
            BandwidthTier::PciE3x16,
        );
        assert_eq!(sub, MixedSubstrate::CpuOnly);
    }
}

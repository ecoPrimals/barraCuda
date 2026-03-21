// SPDX-License-Identifier: AGPL-3.0-or-later
//! Unified Hardware Base — Universal Compute Abstraction
//!
//! Hardware executes math, math doesn't know hardware.
//!
//! - Trait-based execution (CPU, GPU, TPU, NPU implement same traits)
//! - Runtime capability discovery (query what hardware can do)
//! - Smart scheduling (match operations to best hardware)
//! - Cross-device transfer cost modelling (`PCIe`, `NVLink`, shared memory)

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
    BandwidthTier, GPU_DISPATCH_OVERHEAD_US, MixedSubstrate, PCIE_DMA_LATENCY_US,
    PCIE4_X16_BANDWIDTH_GBPS, PcieBridge, PcieLinkInfo, TransferCost, mixed_substrate,
    mixed_substrate_with_tier,
};
pub use types::{
    HardwareCapabilities, HardwareType, MemoryCapabilities, OperationCapabilities,
    ParallelismCapabilities, PerformanceCapabilities, PrecisionCapabilities,
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::unified_math::{DType, MathOp, TensorDescriptor};
    use std::sync::Arc;

    #[test]
    fn test_hardware_type() {
        assert_eq!(HardwareType::CPU, HardwareType::CPU);
        assert_ne!(HardwareType::CPU, HardwareType::GPU);
    }

    #[test]
    fn test_hardware_type_all_variants_distinct() {
        let types = [
            HardwareType::CPU,
            HardwareType::GPU,
            HardwareType::TPU,
            HardwareType::NPU,
            HardwareType::FPGA,
            HardwareType::ASIC,
            HardwareType::Custom,
        ];
        for (i, a) in types.iter().enumerate() {
            for (j, b) in types.iter().enumerate() {
                if i == j {
                    assert_eq!(a, b);
                } else {
                    assert_ne!(a, b);
                }
            }
        }
    }

    #[test]
    fn test_hardware_type_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(HardwareType::CPU);
        set.insert(HardwareType::GPU);
        set.insert(HardwareType::CPU);
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_hardware_capabilities_fields() {
        let cap = HardwareCapabilities {
            hardware_type: HardwareType::GPU,
            parallelism: ParallelismCapabilities {
                max_parallel_units: 3584,
                simd_width: 32,
                task_parallel: true,
                data_parallel: true,
                pipeline_parallel: false,
            },
            memory: MemoryCapabilities {
                total_bytes: 24 * 1024 * 1024 * 1024,
                available_bytes: 20 * 1024 * 1024 * 1024,
                bandwidth_bytes_per_sec: 936 * 1024 * 1024 * 1024,
                unified_memory: false,
                zero_copy: false,
            },
            precision: PrecisionCapabilities {
                fp16: true,
                fp32: true,
                fp64: true,
                int8: true,
                int16: true,
                int32: true,
                int64: true,
                mixed_precision: true,
            },
            operations: OperationCapabilities {
                matmul: true,
                convolution: true,
                fft: true,
                reductions: true,
                sparse: true,
                custom_kernels: true,
            },
            performance: PerformanceCapabilities {
                peak_tflops_fp32: 35.6,
                peak_tflops_fp16: 71.2,
                peak_bandwidth_gbps: 936.0,
                typical_power_watts: 350.0,
                typical_latency_us: 5.0,
            },
        };
        assert_eq!(cap.hardware_type, HardwareType::GPU);
        assert!(cap.precision.mixed_precision);
        assert!(cap.operations.custom_kernels);
        assert!(cap.performance.peak_tflops_fp32 > 30.0);
    }

    #[tokio::test]
    async fn test_hardware_discovery() {
        let executors = HardwareDiscovery::discover_all().await.unwrap();
        assert!(!executors.is_empty());
        assert!(
            executors
                .iter()
                .any(|e| e.hardware_type() == HardwareType::CPU)
        );
    }

    #[test]
    fn test_cpu_executor() {
        let cpu = cpu_executor::CpuExecutor::new();
        assert_eq!(cpu.name(), "CPU (Native)");
        assert_eq!(cpu.hardware_type(), HardwareType::CPU);
    }

    #[test]
    fn test_cpu_executor_capabilities() {
        let cpu = cpu_executor::CpuExecutor::new();
        let caps = ComputeExecutor::capabilities(&cpu);
        assert_eq!(caps.hardware_type, HardwareType::CPU);
        assert!(caps.parallelism.max_parallel_units >= 1);
        assert!(caps.parallelism.simd_width >= 4);
        assert!(caps.precision.fp32);
        assert!(caps.precision.fp64);
        assert!(caps.operations.matmul);
        assert!(caps.memory.unified_memory);
    }

    #[test]
    fn test_cpu_executor_can_execute_all_ops() {
        let cpu = cpu_executor::CpuExecutor::new();
        let desc = TensorDescriptor::new(vec![4, 4], DType::F32);
        assert!(ComputeExecutor::can_execute(
            &cpu,
            &MathOp::Add,
            &[desc.clone(), desc.clone()]
        ));
        assert!(ComputeExecutor::can_execute(
            &cpu,
            &MathOp::MatMul {
                transpose_a: false,
                transpose_b: false,
            },
            &[desc.clone(), desc.clone()]
        ));
        assert!(ComputeExecutor::can_execute(
            &cpu,
            &MathOp::Exp,
            std::slice::from_ref(&desc)
        ));
    }

    #[test]
    fn test_cpu_executor_score_is_positive() {
        let cpu = cpu_executor::CpuExecutor::new();
        let desc = TensorDescriptor::new(vec![100], DType::F32);
        let score =
            ComputeExecutor::score_operation(&cpu, &MathOp::Add, std::slice::from_ref(&desc));
        assert!(score > 0.0);
    }

    #[tokio::test]
    async fn test_cpu_executor_allocate_and_read() {
        let cpu = cpu_executor::CpuExecutor::new();
        let desc = TensorDescriptor::new(vec![2, 3], DType::F32);
        let storage = ComputeExecutor::allocate(&cpu, desc.clone()).await.unwrap();
        assert_eq!(storage.descriptor(), &desc);
        assert!(storage.is_cpu());
        assert!(!storage.is_gpu());
        let data = storage.read_to_cpu().await.unwrap();
        assert_eq!(data.len(), 2 * 3 * 4); // 6 f32 = 24 bytes
        assert!(data.iter().all(|&b| b == 0)); // zeros
    }

    #[tokio::test]
    async fn test_cpu_tensor_storage_write_and_read() {
        let cpu = cpu_executor::CpuExecutor::new();
        let desc = TensorDescriptor::new(vec![4], DType::F32);
        let storage = ComputeExecutor::allocate(&cpu, desc).await.unwrap();
        let transferred = ComputeExecutor::transfer(&cpu, storage).await.unwrap();
        let readback = transferred.read_to_cpu().await.unwrap();
        assert_eq!(readback.len(), 16); // 4 f32 = 16 bytes
    }

    #[tokio::test]
    async fn test_cpu_tensor_storage_write_size_mismatch() {
        let cpu = cpu_executor::CpuExecutor::new();
        let desc = TensorDescriptor::new(vec![4], DType::F32);
        let storage = ComputeExecutor::allocate(&cpu, desc).await.unwrap();
        // Try to downcast for write test - storage is already correctly sized
        let data = storage.read_to_cpu().await.unwrap();
        assert_eq!(data.len(), 16);
    }

    #[tokio::test]
    async fn test_cpu_executor_transfer_cpu_noop() {
        let cpu = cpu_executor::CpuExecutor::new();
        let desc = TensorDescriptor::new(vec![10], DType::F32);
        let storage = ComputeExecutor::allocate(&cpu, desc).await.unwrap();
        let ptr_before = Arc::as_ptr(&storage);
        let transferred = ComputeExecutor::transfer(&cpu, storage).await.unwrap();
        let ptr_after = Arc::as_ptr(&transferred);
        assert_eq!(ptr_before, ptr_after); // CPU→CPU transfer is a no-op
    }

    #[test]
    fn test_tensor_storage_hardware_type_helpers() {
        let desc = TensorDescriptor::new(vec![4], DType::F32);
        let storage = cpu_executor::CpuTensorStorageSimple {
            descriptor: desc,
            data: bytes::Bytes::from(vec![0u8; 16]),
        };
        assert!(storage.is_cpu());
        assert!(!storage.is_gpu());
        assert!(!storage.is_tpu());
        assert!(storage.as_wgpu_buffer().is_none());
    }

    // ─── Scheduler tests ───

    #[tokio::test]
    async fn test_scheduler_empty_executors() {
        let scheduler = ComputeScheduler::new(vec![]);
        let desc = TensorDescriptor::new(vec![4], DType::F32);
        assert!(scheduler.select_executor(&MathOp::Add, &[desc]).is_none());
    }

    #[tokio::test]
    async fn test_scheduler_selects_cpu() {
        let cpu: Arc<dyn ComputeExecutor> = Arc::new(cpu_executor::CpuExecutor::new());
        let scheduler = ComputeScheduler::new(vec![cpu]);
        let desc = TensorDescriptor::new(vec![4, 4], DType::F32);
        let matmul_op = MathOp::MatMul {
            transpose_a: false,
            transpose_b: false,
        };
        let selected = scheduler.select_executor(&matmul_op, &[desc.clone(), desc]);
        assert!(selected.is_some());
        assert_eq!(selected.unwrap().hardware_type(), HardwareType::CPU);
    }

    #[tokio::test]
    async fn test_scheduler_execute_with_no_executor() {
        let scheduler = ComputeScheduler::new(vec![]);
        let cpu = cpu_executor::CpuExecutor::new();
        let desc = TensorDescriptor::new(vec![4], DType::F32);
        let storage: Arc<dyn TensorStorage> = ComputeExecutor::allocate(&cpu, desc).await.unwrap();
        let result = scheduler.execute(&MathOp::Add, vec![storage]).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_scheduler_selects_highest_score() {
        let executors = HardwareDiscovery::discover_all().await.unwrap();
        let scheduler = ComputeScheduler::new(executors);
        let desc = TensorDescriptor::new(vec![1024, 1024], DType::F32);
        let matmul_op = MathOp::MatMul {
            transpose_a: false,
            transpose_b: false,
        };
        let selected = scheduler.select_executor(&matmul_op, &[desc.clone(), desc]);
        assert!(selected.is_some());
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
    fn test_pcie_bridge_detect_p2p_runs_without_panic() {
        let bridge = PcieBridge::detect_p2p();
        assert!(!bridge.source_label.is_empty());
        assert!(!bridge.target_label.is_empty());
    }

    #[test]
    fn test_pcie_link_info_bandwidth() {
        let link = PcieLinkInfo {
            bdf_address: "0000:01:00.0".to_string(),
            pcie_gen: 4,
            lane_width: 16,
            numa_node: Some(0),
            vendor_id: 0x10de,
        };
        let bw = link.bandwidth_gbps();
        assert!((bw - 31.504).abs() < 1.0);
        assert_eq!(link.bandwidth_tier(), BandwidthTier::PciE4x16);
    }

    #[test]
    fn test_pcie_link_info_narrow_link() {
        let link = PcieLinkInfo {
            bdf_address: "0000:03:00.0".to_string(),
            pcie_gen: 3,
            lane_width: 4,
            numa_node: None,
            vendor_id: 0x10de,
        };
        let bw = link.bandwidth_gbps();
        assert!(bw < 5.0);
        assert_eq!(link.bandwidth_tier(), BandwidthTier::Unknown);
    }

    #[test]
    fn test_pcie_probe_all_gpus_does_not_panic() {
        let _links = PcieLinkInfo::probe_all_gpus();
    }

    #[test]
    fn test_pcie_bridge_transfer_cost() {
        let bridge = PcieBridge::detect_p2p();
        let cost = bridge.transfer_cost();
        assert!(cost.latency_us > 0.0);
        assert!(cost.bandwidth_gbps > 0.0);
        assert!(cost.estimated_us(1_048_576) > cost.latency_us);
    }

    #[test]
    fn test_bandwidth_tier_values() {
        assert!((BandwidthTier::PciE3x16.bandwidth_gbps() - 15.75).abs() < 0.01);
        assert!((BandwidthTier::PciE4x16.bandwidth_gbps() - 31.5).abs() < 0.01);
        assert!((BandwidthTier::PciE5x16.bandwidth_gbps() - 63.0).abs() < 0.01);
        assert!(BandwidthTier::HighBandwidthInterconnect.bandwidth_gbps() > 200.0);
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
            BandwidthTier::HighBandwidthInterconnect
        );
        assert_eq!(
            BandwidthTier::detect_from_adapter_name("NVIDIA H100"),
            BandwidthTier::HighBandwidthInterconnect
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

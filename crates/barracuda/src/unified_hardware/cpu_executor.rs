// SPDX-License-Identifier: AGPL-3.0-only
//! CPU executor — always-available fallback compute device.
//!
//! Runtime CPU discovery via `std::thread::available_parallelism()` (pure Rust).
//! Memory/bandwidth are conservative estimates; higher-level orchestrators
//! with sysinfo can override via capability overrides.

use crate::error::Result;
use crate::unified_math::{MathOp, TensorDescriptor};
use std::sync::Arc;

use super::traits::{ComputeExecutor, TensorStorage};
use super::types::{
    HardwareCapabilities, HardwareType, MemoryCapabilities, OperationCapabilities,
    ParallelismCapabilities, PerformanceCapabilities, PrecisionCapabilities,
};
use bytes::Bytes;

mod defaults {
    /// Conservative fallback total system memory (16 GiB) when sysinfo unavailable.
    pub const FALLBACK_TOTAL_MEMORY_BYTES: u64 = 16 * 1024 * 1024 * 1024;
    /// Conservative fallback available memory (8 GiB) when sysinfo unavailable.
    pub const FALLBACK_AVAILABLE_MEMORY_BYTES: u64 = 8 * 1024 * 1024 * 1024;
    /// Conservative fallback memory bandwidth (50 GiB/s) when sysinfo unavailable.
    pub const FALLBACK_BANDWIDTH_BYTES_SEC: u64 = 50 * 1024 * 1024 * 1024;
    #[cfg(not(target_arch = "x86_64"))]
    pub const FALLBACK_SIMD_WIDTH: usize = 4;
}

pub(crate) struct CpuExecutor {
    capabilities: HardwareCapabilities,
}

impl CpuExecutor {
    pub(crate) fn new() -> Self {
        let cpu_cores = std::thread::available_parallelism()
            .map(std::num::NonZero::get)
            .unwrap_or(4);

        #[cfg(target_arch = "x86_64")]
        let simd_width = {
            if std::arch::is_x86_feature_detected!("avx512f") {
                16
            } else if std::arch::is_x86_feature_detected!("avx2") {
                8
            } else {
                4
            }
        };

        #[cfg(target_arch = "aarch64")]
        let simd_width = defaults::FALLBACK_SIMD_WIDTH;

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        let simd_width = defaults::FALLBACK_SIMD_WIDTH;

        Self {
            capabilities: HardwareCapabilities {
                hardware_type: HardwareType::CPU,
                parallelism: ParallelismCapabilities {
                    max_parallel_units: cpu_cores,
                    simd_width,
                    task_parallel: true,
                    data_parallel: true,
                    pipeline_parallel: true,
                },
                memory: MemoryCapabilities {
                    total_bytes: defaults::FALLBACK_TOTAL_MEMORY_BYTES,
                    available_bytes: defaults::FALLBACK_AVAILABLE_MEMORY_BYTES,
                    bandwidth_bytes_per_sec: defaults::FALLBACK_BANDWIDTH_BYTES_SEC,
                    unified_memory: true,
                    zero_copy: true,
                },
                precision: PrecisionCapabilities {
                    fp16: false,
                    fp32: true,
                    fp64: true,
                    int8: true,
                    int16: true,
                    int32: true,
                    int64: true,
                    mixed_precision: false,
                },
                operations: OperationCapabilities {
                    matmul: true,
                    convolution: true,
                    fft: true,
                    reductions: true,
                    sparse: true,
                    custom_kernels: false,
                },
                performance: PerformanceCapabilities {
                    peak_tflops_fp32: (cpu_cores as f64 * 0.1).min(2.0),
                    peak_tflops_fp16: 0.0,
                    peak_bandwidth_gbps: 50.0,
                    typical_power_watts: 65.0,
                    typical_latency_us: 10.0,
                },
            },
        }
    }

    pub(crate) fn name(&self) -> &'static str {
        "CPU (Native)"
    }

    pub(crate) fn hardware_type(&self) -> HardwareType {
        HardwareType::CPU
    }
}

impl ComputeExecutor for CpuExecutor {
    fn name(&self) -> &str {
        self.name()
    }

    fn hardware_type(&self) -> HardwareType {
        self.hardware_type()
    }

    fn capabilities(&self) -> &HardwareCapabilities {
        &self.capabilities
    }

    fn can_execute(&self, _op: &MathOp, _inputs: &[TensorDescriptor]) -> bool {
        true
    }

    fn score_operation(&self, _op: &MathOp, _inputs: &[TensorDescriptor]) -> f64 {
        0.5
    }

    fn execute(
        &self,
        op: &MathOp,
        inputs: Vec<Arc<dyn TensorStorage>>,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<Arc<dyn TensorStorage>>> + Send + '_>,
    > {
        let op = op.clone();
        let standalone = crate::cpu_executor::CpuExecutor::new();
        Box::pin(async move {
            if inputs.is_empty() {
                return Err(crate::error::BarracudaError::InvalidInput {
                    message: "No inputs provided".to_string(),
                });
            }
            standalone.execute(&op, inputs).await
        })
    }

    fn allocate(
        &self,
        descriptor: TensorDescriptor,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<Arc<dyn TensorStorage>>> + Send + '_>,
    > {
        let byte_size = descriptor.numel * descriptor.dtype.size_bytes();
        Box::pin(async move {
            Ok(Arc::new(CpuTensorStorageSimple {
                descriptor,
                data: Bytes::from(vec![0u8; byte_size]),
            }) as Arc<dyn TensorStorage>)
        })
    }

    fn transfer(
        &self,
        tensor: Arc<dyn TensorStorage>,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<Arc<dyn TensorStorage>>> + Send + '_>,
    > {
        Box::pin(async move {
            if tensor.is_cpu() {
                Ok(tensor)
            } else {
                let data = tensor.read_to_cpu().await?;
                let descriptor = tensor.descriptor().clone();
                Ok(Arc::new(CpuTensorStorageSimple { descriptor, data }) as Arc<dyn TensorStorage>)
            }
        })
    }
}

/// Simple CPU tensor storage for the scheduler path.
///
/// Uses `Bytes` for zero-copy readback — `read_to_cpu()` is a cheap
/// ref-count bump instead of cloning the entire buffer.
pub(crate) struct CpuTensorStorageSimple {
    pub(crate) descriptor: TensorDescriptor,
    pub(crate) data: Bytes,
}

impl TensorStorage for CpuTensorStorageSimple {
    fn descriptor(&self) -> &TensorDescriptor {
        &self.descriptor
    }

    fn hardware_type(&self) -> HardwareType {
        HardwareType::CPU
    }

    fn read_to_cpu(
        &self,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Bytes>> + Send + '_>> {
        let data = self.data.clone();
        Box::pin(async move { Ok(data) })
    }

    fn write_from_cpu(
        &mut self,
        data: &[u8],
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<()>> + Send + '_>> {
        let new_data = Bytes::copy_from_slice(data);
        Box::pin(async move {
            if new_data.len() != self.data.len() {
                return Err(crate::error::BarracudaError::InvalidInput {
                    message: format!(
                        "Data size mismatch: expected {}, got {}",
                        self.data.len(),
                        new_data.len()
                    ),
                });
            }
            self.data = new_data;
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::unified_math::DType;

    #[test]
    fn cpu_executor_creation() {
        let exec = CpuExecutor::new();
        assert_eq!(exec.hardware_type(), HardwareType::CPU);
        assert_eq!(exec.name(), "CPU (Native)");
        assert!(exec.capabilities.parallelism.max_parallel_units >= 1);
        assert!(exec.capabilities.memory.total_bytes > 0);
    }

    #[test]
    fn cpu_executor_can_execute_anything() {
        let exec = CpuExecutor::new();
        let desc = TensorDescriptor::new(vec![10], DType::F32);
        assert!(exec.can_execute(&MathOp::ReLU, &[desc]));
    }

    #[test]
    fn cpu_executor_score_is_middle() {
        let exec = CpuExecutor::new();
        let desc = TensorDescriptor::new(vec![1000], DType::F32);
        let score = exec.score_operation(&MathOp::Add, &[desc]);
        assert!(score > 0.0);
        assert!(score <= 1.0);
    }

    #[tokio::test]
    async fn cpu_tensor_storage_roundtrip() {
        let desc = TensorDescriptor::new(vec![4], DType::F32);
        let mut storage = CpuTensorStorageSimple {
            descriptor: desc,
            data: Bytes::from(vec![0u8; 16]),
        };
        let input = [1.0f32, 2.0, 3.0, 4.0];
        let bytes: &[u8] = bytemuck::cast_slice(&input);
        storage.write_from_cpu(bytes).await.unwrap();
        let readback = storage.read_to_cpu().await.unwrap();
        assert_eq!(readback.as_ref(), bytes);
    }

    #[tokio::test]
    async fn cpu_tensor_storage_size_mismatch() {
        let desc = TensorDescriptor::new(vec![4], DType::F32);
        let mut storage = CpuTensorStorageSimple {
            descriptor: desc,
            data: Bytes::from(vec![0u8; 16]),
        };
        let result = storage.write_from_cpu(&[1, 2, 3]).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn cpu_allocate_produces_correct_size() {
        let exec = CpuExecutor::new();
        let desc = TensorDescriptor::new(vec![8], DType::F32);
        let tensor = exec.allocate(desc.clone()).await.unwrap();
        assert_eq!(tensor.descriptor().numel, 8);
        assert_eq!(tensor.hardware_type(), HardwareType::CPU);
    }

    #[tokio::test]
    async fn cpu_transfer_already_cpu() {
        let exec = CpuExecutor::new();
        let desc = TensorDescriptor::new(vec![2], DType::F32);
        let tensor = exec.allocate(desc).await.unwrap();
        let transferred = exec.transfer(tensor).await.unwrap();
        assert!(transferred.is_cpu());
    }
}

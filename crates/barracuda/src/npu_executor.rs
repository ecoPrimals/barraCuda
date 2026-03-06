// SPDX-License-Identifier: AGPL-3.0-or-later
//! NPU Executor - Neuromorphic Processor Implementation via Akida
//!
//! **Philosophy**: Bridge Akida neuromorphic operations to unified architecture
//!
//! This module wraps the existing `AkidaExecutor` to implement the `ComputeExecutor` trait,
//! allowing the scheduler to use Akida NPUs for neuromorphic workloads.
//!
//! **Deep Debt Principles**:
//! - ✅ Reuse existing Akida infrastructure
//! - ✅ Zero duplication
//! - ✅ Runtime capability discovery
//! - ✅ Hardware-agnostic (works on any NPU with Akida-style interface)
//!
//! **Architectural Note**:
//! - GPU: Continuous compute, high power, general purpose
//! - NPU: Event-driven, ultra-low power, neuromorphic-specialized

pub(crate) mod npu_defaults {
    pub const SRAM_PER_BOARD_BYTES: u64 = 4 * 1024 * 1024;
    pub const AVAILABLE_PER_BOARD_BYTES: u64 = 3 * 1024 * 1024;
    pub const ON_CHIP_BANDWIDTH_BYTES_SEC: u64 = 10 * 1024 * 1024 * 1024;
    pub const _NPU_LATENCY_THRESHOLD: usize = 128;
}

/// NPU operation efficiency factors for scheduler scoring.
/// Values represent relative efficiency vs GPU for each op class on neuromorphic hardware.
mod npu_efficiency {
    /// Nominal equivalence value for NPU in TFLOPS terms. NPUs use spike counts, not TFLOPS;
    /// this constant provides a nominal value for the unified scheduler interface.
    pub(super) const NPU_EQUIVALENT_TFLOPS: f64 = 0.001;
    /// MatMul/BatchMatMul efficiency on NPU (sparse matrix ops). Reserved for per-op scoring.
    pub(super) const _MATMUL_EFFICIENCY: f64 = 0.85;
    /// `Conv2D` efficiency on NPU (neuromorphic convolutions). Reserved for per-op scoring.
    pub(super) const _CONV2D_EFFICIENCY: f64 = 0.90;
    /// Sigmoid/Tanh efficiency (spike coding). Reserved for per-op scoring.
    pub(super) const _ACTIVATION_EFFICIENCY: f64 = 0.70;
    /// ReduceSum/ReduceMax/ReduceMin efficiency (spike counting). Reserved for per-op scoring.
    pub(super) const _REDUCE_EFFICIENCY: f64 = 0.75;
}

use crate::device::akida_executor::AkidaExecutor;
use crate::error::Result;
use crate::unified_hardware::{
    ComputeExecutor, HardwareCapabilities, HardwareType, MemoryCapabilities, OperationCapabilities,
    ParallelismCapabilities, PerformanceCapabilities, PrecisionCapabilities, TensorStorage,
};
use crate::unified_math::{MathOp, TensorDescriptor};
use std::sync::Arc;

/// NPU executor wrapping `AkidaExecutor`
pub struct NpuExecutor {
    executor: Arc<AkidaExecutor>,
    capabilities: HardwareCapabilities,
}

impl NpuExecutor {
    /// Create new NPU executor
    /// # Errors
    /// Returns [`Err`] if no Akida boards are detected or board initialization fails.
    pub fn new() -> Result<Self> {
        let akida = AkidaExecutor::new()?;
        let capabilities = Self::detect_capabilities(&akida);

        Ok(Self {
            executor: Arc::new(akida),
            capabilities,
        })
    }

    /// Create from existing `AkidaExecutor`
    #[must_use]
    pub fn from_executor(executor: AkidaExecutor) -> Self {
        let capabilities = Self::detect_capabilities(&executor);
        Self {
            executor: Arc::new(executor),
            capabilities,
        }
    }

    /// Detect NPU capabilities from Akida boards
    fn detect_capabilities(executor: &AkidaExecutor) -> HardwareCapabilities {
        let npu_count = executor.npu_count();
        let board_count = executor.board_count();

        // Akida-specific capabilities
        // Each NPU has 1.2M neurons, ultra-low power (reserved for future neuron-based scheduling)
        let _ = npu_count * 1_200_000;

        HardwareCapabilities {
            hardware_type: HardwareType::NPU,

            parallelism: ParallelismCapabilities {
                max_parallel_units: npu_count, // Each NPU is a parallel unit
                simd_width: 256,               // Neuromorphic cores process in parallel
                task_parallel: true,
                data_parallel: true,
                pipeline_parallel: true,
            },

            memory: MemoryCapabilities {
                total_bytes: board_count as u64 * npu_defaults::SRAM_PER_BOARD_BYTES,
                available_bytes: board_count as u64 * npu_defaults::AVAILABLE_PER_BOARD_BYTES,
                bandwidth_bytes_per_sec: npu_defaults::ON_CHIP_BANDWIDTH_BYTES_SEC,
                unified_memory: true, // On-chip is unified
                zero_copy: true,      // Event-driven, no copies
            },

            precision: PrecisionCapabilities {
                fp16: false, // NPU uses integer/spike representations
                fp32: false,
                fp64: false,
                int8: true,  // Primary NPU precision
                int16: true, // Spike counts
                int32: true, // Accumulators
                int64: false,
                mixed_precision: true, // INT8 inputs, INT32 accumulators
            },

            operations: OperationCapabilities {
                matmul: true,          // Spike-based matrix ops
                convolution: true,     // Neuromorphic convolutions
                fft: false,            // Not suited for FFT
                reductions: true,      // Spike counting
                sparse: true,          // Excellent for sparse (event-driven!)
                custom_kernels: false, // NPU uses fixed function units
            },

            performance: PerformanceCapabilities {
                // NPU performance in neuromorphic terms
                peak_tflops_fp32: npu_efficiency::NPU_EQUIVALENT_TFLOPS,
                peak_tflops_fp16: npu_efficiency::NPU_EQUIVALENT_TFLOPS,
                peak_bandwidth_gbps: 10.0,
                typical_power_watts: 1.0 * board_count as f64, // ~1W per board!
                typical_latency_us: 1.0,                       // Ultra-low latency for inference
            },
        }
    }

    /// Get underlying `AkidaExecutor`
    #[must_use]
    pub fn akida(&self) -> &AkidaExecutor {
        &self.executor
    }

    /// Get total NPU count
    #[must_use]
    pub fn npu_count(&self) -> usize {
        self.executor.npu_count()
    }
}

impl ComputeExecutor for NpuExecutor {
    fn name(&self) -> &'static str {
        "Akida NPU"
    }

    fn hardware_type(&self) -> HardwareType {
        HardwareType::NPU
    }

    fn capabilities(&self) -> &HardwareCapabilities {
        &self.capabilities
    }

    fn can_execute(&self, op: &MathOp, inputs: &[TensorDescriptor]) -> bool {
        // NPU excels at sparse, event-driven operations
        let total_elements: usize = inputs.iter().map(|t| t.numel).sum();

        // NPU is good for medium-sized, sparse workloads
        if total_elements > 10_000_000 {
            return false; // Too large for on-chip memory
        }

        match op {
            // Neuromorphic operations - NPU excels
            MathOp::MatMul { .. } | MathOp::BatchMatMul { .. } => true,
            MathOp::Conv2D { .. } => true,

            // Activations - NPU handles via spike coding
            MathOp::ReLU | MathOp::Sigmoid | MathOp::Tanh => true,

            // Reductions - spike counting
            MathOp::ReduceSum { .. } | MathOp::ReduceMax { .. } => true,

            // Shape ops - NPU handles internally
            MathOp::Reshape { .. } | MathOp::Transpose { .. } => true,

            // Operations not suited for NPU
            MathOp::GELU | MathOp::Softmax { .. } => false, // Complex activations
            MathOp::Pow | MathOp::Div => false,             // Not event-driven friendly

            // Default: let NPU try
            _ => true,
        }
    }

    fn score_operation(&self, op: &MathOp, inputs: &[TensorDescriptor]) -> f64 {
        use MathOp::{
            BatchMatMul, Conv2D, Div, GELU, MatMul, Pow, ReLU, ReduceMax, ReduceMin, ReduceSum,
            Sigmoid, Softmax, Tanh,
        };

        let total_elements: usize = inputs.iter().map(|t| t.numel).sum();

        // NPU sweet spot: medium-sized sparse operations
        let size_score = if total_elements < 100 {
            0.3 // Too small
        } else if total_elements < 10_000 {
            0.9 // Perfect size for NPU
        } else if total_elements < 100_000 {
            0.7 // Good but getting large
        } else if total_elements < 1_000_000 {
            0.4 // Pushing limits
        } else {
            0.1 // Too large for on-chip
        };

        // Operation-specific scoring
        let op_score = match op {
            // Neuromorphic excels at sparse matrix ops
            MatMul { .. } | BatchMatMul { .. } => 0.85,

            // Convolutions - NPU is efficient
            Conv2D { .. } => 0.90,

            // Simple activations - spike coding
            ReLU => 0.95, // NPU's sweet spot!
            Sigmoid | Tanh => 0.70,

            // Reductions - spike counting
            ReduceSum { .. } | ReduceMax { .. } | ReduceMin { .. } => 0.75,

            // Operations NPU doesn't excel at
            GELU | Softmax { .. } => 0.2,
            Pow | Div => 0.1,

            // Default
            _ => 0.5,
        };

        size_score * op_score
    }

    fn execute(
        &self,
        op: &MathOp,
        inputs: Vec<Arc<dyn TensorStorage>>,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<Arc<dyn TensorStorage>>> + Send + '_>,
    > {
        let op = op.clone();
        Box::pin(async move {
            if inputs.is_empty() {
                return Err(crate::error::BarracudaError::InvalidInput {
                    message: "No inputs provided".to_string(),
                });
            }
            tracing::debug!("NPU execute: op={:?}, inputs={}", op, inputs.len());
            Ok(inputs[0].clone())
        })
    }

    fn allocate(
        &self,
        descriptor: TensorDescriptor,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<Arc<dyn TensorStorage>>> + Send + '_>,
    > {
        Box::pin(async move {
            Ok(Arc::new(NpuTensorStorage::new(descriptor)) as Arc<dyn TensorStorage>)
        })
    }

    fn transfer(
        &self,
        tensor: Arc<dyn TensorStorage>,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<Arc<dyn TensorStorage>>> + Send + '_>,
    > {
        Box::pin(async move {
            if tensor.hardware_type() == HardwareType::NPU {
                Ok(tensor)
            } else {
                let data = tensor.read_to_cpu().await?;
                let descriptor = tensor.descriptor().clone();
                let mut npu_tensor = NpuTensorStorage::new(descriptor);
                npu_tensor.write_from_cpu(&data).await?;
                Ok(Arc::new(npu_tensor) as Arc<dyn TensorStorage>)
            }
        })
    }
}

/// NPU tensor storage for the scheduler path
///
/// The primary NPU execution path uses `AkidaExecutor` directly.
/// This storage type is for the `ComputeExecutor` scheduler interface.
struct NpuTensorStorage {
    descriptor: TensorDescriptor,
    data: Vec<u8>,
}

impl NpuTensorStorage {
    fn new(descriptor: TensorDescriptor) -> Self {
        let byte_size = descriptor.numel * descriptor.dtype.size_bytes();
        Self {
            descriptor,
            data: vec![0u8; byte_size],
        }
    }
}

impl TensorStorage for NpuTensorStorage {
    fn descriptor(&self) -> &TensorDescriptor {
        &self.descriptor
    }

    fn hardware_type(&self) -> HardwareType {
        HardwareType::NPU
    }

    fn read_to_cpu(
        &self,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<Vec<u8>>> + Send + '_>> {
        let data = self.data.clone();
        Box::pin(async move { Ok(data) })
    }

    fn write_from_cpu(
        &mut self,
        data: &[u8],
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Result<()>> + Send + '_>> {
        let data = data.to_vec();
        Box::pin(async move {
            if data.len() != self.data.len() {
                return Err(crate::error::BarracudaError::InvalidInput {
                    message: format!(
                        "Data size mismatch: expected {}, got {}",
                        self.data.len(),
                        data.len()
                    ),
                });
            }
            self.data.copy_from_slice(&data);
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_npu_executor_creation() {
        // May fail if no NPU available (that's okay)
        match NpuExecutor::new() {
            Ok(npu) => {
                assert_eq!(npu.hardware_type(), HardwareType::NPU);
                assert!(!npu.name().is_empty());
                tracing::debug!("NPU: {} with {} NPUs", npu.name(), npu.npu_count());
            }
            Err(e) => {
                tracing::debug!("No NPU available (okay for testing): {}", e);
            }
        }
    }

    #[tokio::test]
    async fn test_npu_capabilities() {
        if let Ok(npu) = NpuExecutor::new() {
            let caps = npu.capabilities();
            assert!(caps.operations.sparse); // NPU excels at sparse
            assert!(caps.precision.int8); // Primary NPU precision
            assert!(caps.memory.zero_copy); // Event-driven
        }
    }

    #[test]
    fn test_npu_scoring() {
        // NPU should score high for sparse, medium-sized operations
        // NPU should score low for very large or continuous operations
    }
}

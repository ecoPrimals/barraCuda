// SPDX-License-Identifier: AGPL-3.0-or-later
//! CPU executor core and op dispatch

mod defaults {
    pub const FALLBACK_MEMORY_BYTES: u64 = 16 * 1024 * 1024 * 1024;
    pub const SMALL_TENSOR_THRESHOLD: usize = 1_000;
    pub const MEDIUM_TENSOR_THRESHOLD: usize = 10_000;
    pub const LARGE_TENSOR_THRESHOLD: usize = 1_000_000;
    pub const SCORE_CPU_SMALL_OPTIMAL: f64 = 0.9;
    pub const SCORE_CPU_MATMUL_LARGE: f64 = 0.2;
    pub const SCORE_CPU_MATMUL_MEDIUM: f64 = 0.5;
    pub const SCORE_CPU_ELEMENTWISE_SMALL: f64 = 0.8;
    pub const SCORE_CPU_ELEMENTWISE_MEDIUM: f64 = 0.5;
    pub const SCORE_CPU_ELEMENTWISE_LARGE: f64 = 0.3;
    pub const SCORE_CPU_REDUCE: f64 = 0.6;
    pub const SCORE_CPU_CONV_SMALL: f64 = 0.7;
    pub const SCORE_CPU_CONV_LARGE: f64 = 0.2;
    pub const SCORE_CPU_DEFAULT: f64 = 0.5;
}

use super::storage::CpuTensorStorage;
use crate::error::Result;
use crate::unified_hardware::{
    ComputeExecutor, HardwareCapabilities, HardwareType, MemoryCapabilities, OperationCapabilities,
    ParallelismCapabilities, PerformanceCapabilities, PrecisionCapabilities, TensorStorage,
};
use crate::unified_math::{MathOp, TensorDescriptor};
use rayon::prelude::*;
use std::sync::Arc;

/// CPU executor implementation
pub struct CpuExecutor {
    capabilities: HardwareCapabilities,
    /// Thread count (stored for future parallel execution tuning)
    pub(crate) _num_threads: usize,
}

impl CpuExecutor {
    /// Create new CPU executor
    pub fn new() -> Self {
        let num_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);

        Self {
            capabilities: Self::detect_capabilities(num_threads),
            _num_threads: num_threads,
        }
    }

    /// Detect CPU capabilities at runtime via OS-level probing.
    fn detect_capabilities(num_threads: usize) -> HardwareCapabilities {
        let total_memory = crate::device::capabilities::detect_system_memory_bytes()
            .unwrap_or(defaults::FALLBACK_MEMORY_BYTES);

        HardwareCapabilities {
            hardware_type: HardwareType::CPU,
            parallelism: ParallelismCapabilities {
                max_parallel_units: num_threads,
                simd_width: Self::detect_simd_width(),
                task_parallel: true,
                data_parallel: true,
                pipeline_parallel: false,
            },
            memory: MemoryCapabilities {
                total_bytes: total_memory,
                available_bytes: total_memory / 2,
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
                peak_tflops_fp32: 0.5,
                peak_tflops_fp16: 0.0,
                peak_bandwidth_gbps: 50.0,
                typical_power_watts: 65.0,
                typical_latency_us: 10.0,
            },
        }
    }

    /// Detect SIMD width at runtime
    pub(crate) fn detect_simd_width() -> usize {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                return 8;
            }
            if is_x86_feature_detected!("sse2") {
                return 4;
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            return 4;
        }
        1
    }

    pub(crate) fn read_f32(storage: &dyn TensorStorage) -> Result<Vec<f32>> {
        let rt = tokio::runtime::Handle::try_current()
            .map(|h| tokio::task::block_in_place(|| h.block_on(storage.read_to_cpu())))
            .unwrap_or_else(|_| {
                tokio::runtime::Runtime::new()
                    .map_err(|e| crate::error::BarracudaError::device(e.to_string()))
                    .and_then(|rt: tokio::runtime::Runtime| rt.block_on(storage.read_to_cpu()))
            })?;
        Ok(rt
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }

    pub(crate) fn pack_f32(data: Vec<f32>, desc: TensorDescriptor) -> Arc<dyn TensorStorage> {
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut s = CpuTensorStorage::new(desc);
        s.data = bytes;
        Arc::new(s)
    }

    pub(crate) fn execute_unary_cpu(&self, op: &MathOp, input: &[f32]) -> Result<Vec<f32>> {
        use MathOp::*;
        let output: Vec<f32> = input
            .par_iter()
            .map(|&x| match op {
                ReLU => x.max(0.0),
                Sigmoid => 1.0 / (1.0 + (-x).exp()),
                Tanh => x.tanh(),
                GELU => {
                    let sqrt_2_over_pi = 0.797_884_6;
                    0.5 * x * (1.0 + (sqrt_2_over_pi * (x + 0.044715 * x * x * x)).tanh())
                }
                Negate => -x,
                Abs => x.abs(),
                Square => x * x,
                Sqrt => x.sqrt(),
                Reciprocal => 1.0 / x,
                Exp => x.exp(),
                Log => x.ln(),
                Sin => x.sin(),
                Cos => x.cos(),
                Tan => x.tan(),
                _ => x,
            })
            .collect();
        Ok(output)
    }

    pub(crate) fn execute_binary_cpu(&self, op: &MathOp, a: &[f32], b: &[f32]) -> Result<Vec<f32>> {
        use MathOp::*;
        let output: Vec<f32> = a
            .par_iter()
            .zip(b.par_iter())
            .map(|(&x, &y)| match op {
                Add => x + y,
                Sub => x - y,
                Mul => x * y,
                Div => x / y,
                Pow => x.powf(y),
                Max => x.max(y),
                Min => x.min(y),
                _ => 0.0,
            })
            .collect();
        Ok(output)
    }

    pub(crate) fn execute_reduce_cpu(&self, op: &MathOp, input: &[f32]) -> Result<f32> {
        use MathOp::*;
        let result = match op {
            ReduceSum { .. } => input.par_iter().sum(),
            ReduceMean { .. } => input.par_iter().sum::<f32>() / input.len() as f32,
            ReduceMax { .. } => input
                .par_iter()
                .cloned()
                .fold(|| f32::NEG_INFINITY, f32::max)
                .reduce(|| f32::NEG_INFINITY, f32::max),
            ReduceMin { .. } => input
                .par_iter()
                .cloned()
                .fold(|| f32::INFINITY, f32::min)
                .reduce(|| f32::INFINITY, f32::min),
            ReduceProd { .. } => input.par_iter().product(),
            _ => 0.0,
        };
        Ok(result)
    }

    pub(crate) fn execute_matmul_cpu(
        &self,
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
    ) -> Result<Vec<f32>> {
        let mut c = vec![0.0f32; m * n];
        c.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
            for j in 0..n {
                let mut sum = 0.0;
                for p in 0..k {
                    sum += a[i * k + p] * b[p * n + j];
                }
                row[j] = sum;
            }
        });
        Ok(c)
    }
}

impl Default for CpuExecutor {
    fn default() -> Self {
        Self::new()
    }
}

impl ComputeExecutor for CpuExecutor {
    fn name(&self) -> &'static str {
        "CPU (Native Rust + SIMD)"
    }

    fn hardware_type(&self) -> HardwareType {
        HardwareType::CPU
    }

    fn capabilities(&self) -> &HardwareCapabilities {
        &self.capabilities
    }

    fn can_execute(&self, _op: &MathOp, _inputs: &[TensorDescriptor]) -> bool {
        true
    }

    fn score_operation(&self, op: &MathOp, inputs: &[TensorDescriptor]) -> f64 {
        use defaults::*;
        use MathOp::*;
        let total_elements: usize = inputs.iter().map(|t| t.numel).sum();
        match op {
            _ if total_elements < SMALL_TENSOR_THRESHOLD => SCORE_CPU_SMALL_OPTIMAL,
            MatMul { .. } | BatchMatMul { .. } if total_elements > LARGE_TENSOR_THRESHOLD => {
                SCORE_CPU_MATMUL_LARGE
            }
            MatMul { .. } | BatchMatMul { .. } => SCORE_CPU_MATMUL_MEDIUM,
            ReLU | Sigmoid | Tanh | GELU | Add | Sub | Mul | Div => {
                if total_elements < MEDIUM_TENSOR_THRESHOLD {
                    SCORE_CPU_ELEMENTWISE_SMALL
                } else if total_elements < LARGE_TENSOR_THRESHOLD {
                    SCORE_CPU_ELEMENTWISE_MEDIUM
                } else {
                    SCORE_CPU_ELEMENTWISE_LARGE
                }
            }
            ReduceSum { .. } | ReduceMean { .. } | ReduceMax { .. } | ReduceMin { .. } => {
                SCORE_CPU_REDUCE
            }
            Conv2D { .. } | MaxPool2D { .. } | AvgPool2D { .. } => {
                if total_elements < MEDIUM_TENSOR_THRESHOLD {
                    SCORE_CPU_CONV_SMALL
                } else {
                    SCORE_CPU_CONV_LARGE
                }
            }
            _ => SCORE_CPU_DEFAULT,
        }
    }

    fn execute(
        &self,
        op: &MathOp,
        inputs: Vec<Arc<dyn TensorStorage>>,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<Arc<dyn TensorStorage>>> + Send + '_>,
    > {
        let result = super::ops::dispatch(self, op, inputs);
        Box::pin(async move { result })
    }

    fn allocate(
        &self,
        descriptor: TensorDescriptor,
    ) -> std::pin::Pin<
        Box<dyn std::future::Future<Output = Result<Arc<dyn TensorStorage>>> + Send + '_>,
    > {
        Box::pin(async move {
            Ok(Arc::new(CpuTensorStorage::new(descriptor)) as Arc<dyn TensorStorage>)
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
                let mut cpu_tensor = CpuTensorStorage::new(descriptor);
                cpu_tensor.write_from_cpu(&data).await?;
                Ok(Arc::new(cpu_tensor) as Arc<dyn TensorStorage>)
            }
        })
    }
}

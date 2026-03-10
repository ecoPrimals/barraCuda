// SPDX-License-Identifier: AGPL-3.0-only
//! Hardware type classification and capability descriptors.

/// Hardware type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HardwareType {
    /// CPU (any architecture)
    CPU,
    /// GPU (via WGSL/WebGPU)
    GPU,
    /// TPU (Tensor Processing Unit)
    TPU,
    /// NPU (Neuromorphic Processing Unit)
    NPU,
    /// FPGA (Field-Programmable Gate Array)
    FPGA,
    /// ASIC (Application-Specific Integrated Circuit)
    ASIC,
    /// Custom/Unknown
    Custom,
}

/// Hardware capabilities descriptor
#[derive(Debug, Clone)]
pub struct HardwareCapabilities {
    /// Classification of the hardware (CPU, GPU, TPU, etc.).
    pub hardware_type: HardwareType,
    /// Parallelism and concurrency capabilities.
    pub parallelism: ParallelismCapabilities,
    /// Memory capacity and bandwidth.
    pub memory: MemoryCapabilities,
    /// Supported numeric precisions.
    pub precision: PrecisionCapabilities,
    /// Supported operation types.
    pub operations: OperationCapabilities,
    /// Performance characteristics.
    pub performance: PerformanceCapabilities,
}

/// Parallelism capabilities
#[derive(Debug, Clone)]
pub struct ParallelismCapabilities {
    /// Maximum number of parallel execution units (cores, SMs, etc.).
    pub max_parallel_units: usize,
    /// SIMD vector width (e.g. 4 for vec4, 8 for AVX).
    pub simd_width: usize,
    /// Supports task-level parallelism.
    pub task_parallel: bool,
    /// Supports data parallelism (same op on many elements).
    pub data_parallel: bool,
    /// Supports pipeline parallelism across stages.
    pub pipeline_parallel: bool,
}

/// Memory capabilities
#[derive(Debug, Clone)]
pub struct MemoryCapabilities {
    /// Total memory capacity in bytes.
    pub total_bytes: u64,
    /// Currently available memory in bytes.
    pub available_bytes: u64,
    /// Memory bandwidth (bytes/sec).
    pub bandwidth_bytes_per_sec: u64,
    /// Whether CPU and device share unified memory.
    pub unified_memory: bool,
    /// Whether zero-copy access is supported.
    pub zero_copy: bool,
}

/// Precision capabilities
#[derive(Debug, Clone)]
pub struct PrecisionCapabilities {
    /// 16-bit floating point support.
    pub fp16: bool,
    /// 32-bit floating point support.
    pub fp32: bool,
    /// 64-bit floating point support.
    pub fp64: bool,
    /// 8-bit integer support.
    pub int8: bool,
    /// 16-bit integer support.
    pub int16: bool,
    /// 32-bit integer support.
    pub int32: bool,
    /// 64-bit integer support.
    pub int64: bool,
    /// Mixed-precision (e.g. Tensor Core) support.
    pub mixed_precision: bool,
}

/// Operation capabilities
#[derive(Debug, Clone)]
pub struct OperationCapabilities {
    /// Matrix multiplication support.
    pub matmul: bool,
    /// Convolution support.
    pub convolution: bool,
    /// FFT support.
    pub fft: bool,
    /// Reduction (sum, max, etc.) support.
    pub reductions: bool,
    /// Sparse linear algebra support.
    pub sparse: bool,
    /// Custom kernel / shader dispatch support.
    pub custom_kernels: bool,
}

/// Performance characteristics
#[derive(Debug, Clone)]
pub struct PerformanceCapabilities {
    /// Peak FP32 throughput (teraflops).
    pub peak_tflops_fp32: f64,
    /// Peak FP16 throughput (teraflops).
    pub peak_tflops_fp16: f64,
    /// Peak memory bandwidth (GB/s).
    pub peak_bandwidth_gbps: f64,
    /// Typical power draw (watts).
    pub typical_power_watts: f64,
    /// Typical kernel launch latency (microseconds).
    pub typical_latency_us: f64,
}

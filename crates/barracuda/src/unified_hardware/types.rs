// SPDX-License-Identifier: AGPL-3.0-or-later
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
    pub hardware_type: HardwareType,
    pub parallelism: ParallelismCapabilities,
    pub memory: MemoryCapabilities,
    pub precision: PrecisionCapabilities,
    pub operations: OperationCapabilities,
    pub performance: PerformanceCapabilities,
}

/// Parallelism capabilities
#[derive(Debug, Clone)]
pub struct ParallelismCapabilities {
    pub max_parallel_units: usize,
    pub simd_width: usize,
    pub task_parallel: bool,
    pub data_parallel: bool,
    pub pipeline_parallel: bool,
}

/// Memory capabilities
#[derive(Debug, Clone)]
pub struct MemoryCapabilities {
    pub total_bytes: u64,
    pub available_bytes: u64,
    pub bandwidth_bytes_per_sec: u64,
    pub unified_memory: bool,
    pub zero_copy: bool,
}

/// Precision capabilities
#[derive(Debug, Clone)]
pub struct PrecisionCapabilities {
    pub fp16: bool,
    pub fp32: bool,
    pub fp64: bool,
    pub int8: bool,
    pub int16: bool,
    pub int32: bool,
    pub int64: bool,
    pub mixed_precision: bool,
}

/// Operation capabilities
#[derive(Debug, Clone)]
pub struct OperationCapabilities {
    pub matmul: bool,
    pub convolution: bool,
    pub fft: bool,
    pub reductions: bool,
    pub sparse: bool,
    pub custom_kernels: bool,
}

/// Performance characteristics
#[derive(Debug, Clone)]
pub struct PerformanceCapabilities {
    pub peak_tflops_fp32: f64,
    pub peak_tflops_fp16: f64,
    pub peak_bandwidth_gbps: f64,
    pub typical_power_watts: f64,
    pub typical_latency_us: f64,
}

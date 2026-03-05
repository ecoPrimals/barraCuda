// SPDX-License-Identifier: AGPL-3.0-or-later
//! Device topology detection: vendor, driver, and capability classification.
//!
//! Extracted from multi_gpu.rs for separation of concerns.

/// GPU vendor classification for capability-based routing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GpuVendor {
    /// NVIDIA GPUs (GeForce, RTX, Quadro, Tesla, etc.)
    Nvidia,
    /// AMD GPUs (Radeon, RDNA, CDNA)
    Amd,
    /// Intel GPUs (Iris, Arc)
    Intel,
    /// Software rasterizer (llvmpipe, SwiftShader, CPU)
    Software,
    /// Unknown or unrecognized vendor
    Unknown,
}

// PCI Vendor IDs for capability-based detection (no string matching)
const VENDOR_ID_NVIDIA: u32 = 0x10DE;
const VENDOR_ID_AMD: u32 = 0x1002;
const VENDOR_ID_INTEL: u32 = 0x8086;

impl GpuVendor {
    /// Detect vendor from PCI vendor ID (preferred - no string matching)
    pub fn from_vendor_id(vendor_id: u32) -> Self {
        match vendor_id {
            VENDOR_ID_NVIDIA => Self::Nvidia,
            VENDOR_ID_AMD => Self::Amd,
            VENDOR_ID_INTEL => Self::Intel,
            0 => Self::Software,
            _ => Self::Unknown,
        }
    }

    /// Detect vendor from device name string
    pub fn from_name(name: &str) -> Self {
        let lower = name.to_lowercase();
        if lower.contains("nvidia")
            || lower.contains("geforce")
            || lower.contains("rtx")
            || lower.contains("gtx")
        {
            return Self::Nvidia;
        }
        if lower.contains("amd") || lower.contains("radeon") || lower.contains("radv") {
            return Self::Amd;
        }
        if lower.contains("intel") || lower.contains("iris") {
            return Self::Intel;
        }
        if lower.contains("llvmpipe")
            || lower.contains("software")
            || lower.contains("swiftshader")
            || lower.contains("cpu")
            || lower.contains("sse2")
            || lower.contains("sse4")
            || lower.contains("avx")
        {
            return Self::Software;
        }
        Self::Unknown
    }
}

/// GPU driver type (affects f64 builtin support).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GpuDriver {
    /// NVIDIA proprietary driver (NVVM/PTXAS)
    NvidiaProprietary,
    /// Mesa NVK (open-source NVIDIA Vulkan)
    Nvk,
    /// Mesa RADV (AMD Vulkan)
    Radv,
    /// Intel ANV driver
    Intel,
    /// Software rasterizer
    Software,
    /// Unknown driver
    Unknown,
}

impl GpuDriver {
    /// Detect driver type from adapter info strings
    pub fn from_adapter_info(name: &str, driver: &str, driver_info: &str) -> Self {
        let name_lower = name.to_lowercase();
        let driver_lower = driver.to_lowercase();
        let info_lower = driver_info.to_lowercase();

        if driver_lower.contains("nvk")
            || driver_lower.contains("nouveau")
            || info_lower.contains("nvk")
            || info_lower.contains("nouveau")
        {
            return Self::Nvk;
        }
        if driver_lower.contains("radv") || info_lower.contains("radv") {
            return Self::Radv;
        }
        if (name_lower.contains("nvidia")
            || name_lower.contains("geforce")
            || name_lower.contains("rtx")
            || name_lower.contains("gtx"))
            && !driver_lower.contains("mesa")
        {
            return Self::NvidiaProprietary;
        }
        if name_lower.contains("intel") || name_lower.contains("iris") {
            return Self::Intel;
        }
        if name_lower.contains("llvmpipe")
            || name_lower.contains("swiftshader")
            || name_lower.contains("software")
        {
            return Self::Software;
        }
        Self::Unknown
    }
}

/// Workload classification for intelligent GPU routing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WorkloadType {
    /// High-throughput streaming workloads (matmul, conv)
    Streaming,
    /// Iterative workloads (CG, eigensolve) with convergence checks
    Iterative,
    /// Workloads requiring native f64 builtins (exp, log, pow)
    F64Builtins,
}

/// GPU information for load balancing and workload routing.
#[derive(Debug, Clone)]
pub struct GpuInfo {
    /// Device index in the adapter list
    pub index: usize,
    /// Human-readable device name
    pub name: String,
    /// GPU vendor classification
    pub vendor: GpuVendor,
    /// Driver type (affects f64 support)
    pub driver: GpuDriver,
    /// Estimated peak GFLOPS for capacity weighting
    pub gflops: f64,
    /// Whether the GPU is currently busy
    pub busy: bool,
}

impl GpuInfo {
    /// Returns true if this GPU supports native f64 builtins (exp, log, pow).
    pub fn supports_f64_builtins(&self) -> bool {
        !matches!(self.driver, GpuDriver::Nvk | GpuDriver::Software)
    }

    /// Returns true if this GPU is suitable for compute workloads (≥500 GFLOPS).
    pub fn is_compute_capable(&self) -> bool {
        matches!(
            self.vendor,
            GpuVendor::Nvidia | GpuVendor::Amd | GpuVendor::Intel
        ) && self.gflops >= 500.0
    }
}

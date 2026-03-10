// SPDX-License-Identifier: AGPL-3.0-only
//! Device topology detection: vendor, driver, and capability classification.
//!
//! Extracted from `multi_gpu.rs` for separation of concerns.

/// GPU vendor classification for capability-based routing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum GpuVendor {
    /// NVIDIA GPUs (`GeForce`, RTX, Quadro, Tesla, etc.)
    Nvidia,
    /// AMD GPUs (Radeon, RDNA, CDNA)
    Amd,
    /// Intel GPUs (Iris, Arc)
    Intel,
    /// Software rasterizer (llvmpipe, `SwiftShader`, CPU)
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
    #[must_use]
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
    #[must_use]
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
    #[must_use]
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
    #[must_use]
    pub fn supports_f64_builtins(&self) -> bool {
        !matches!(self.driver, GpuDriver::Nvk | GpuDriver::Software)
    }

    /// Returns true if this GPU is suitable for compute workloads (≥500 GFLOPS).
    #[must_use]
    pub fn is_compute_capable(&self) -> bool {
        matches!(
            self.vendor,
            GpuVendor::Nvidia | GpuVendor::Amd | GpuVendor::Intel
        ) && self.gflops >= 500.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vendor_from_pci_id() {
        assert_eq!(GpuVendor::from_vendor_id(0x10DE), GpuVendor::Nvidia);
        assert_eq!(GpuVendor::from_vendor_id(0x1002), GpuVendor::Amd);
        assert_eq!(GpuVendor::from_vendor_id(0x8086), GpuVendor::Intel);
        assert_eq!(GpuVendor::from_vendor_id(0), GpuVendor::Software);
        assert_eq!(GpuVendor::from_vendor_id(0xFFFF), GpuVendor::Unknown);
    }

    #[test]
    fn vendor_from_device_name() {
        assert_eq!(
            GpuVendor::from_name("NVIDIA GeForce RTX 4090"),
            GpuVendor::Nvidia
        );
        assert_eq!(GpuVendor::from_name("RTX 3090"), GpuVendor::Nvidia);
        assert_eq!(GpuVendor::from_name("AMD Radeon RX 7900"), GpuVendor::Amd);
        assert_eq!(GpuVendor::from_name("RADV NAVI10"), GpuVendor::Amd);
        assert_eq!(GpuVendor::from_name("Intel Iris Xe"), GpuVendor::Intel);
        assert_eq!(
            GpuVendor::from_name("llvmpipe (LLVM 17.0.6)"),
            GpuVendor::Software
        );
        assert_eq!(GpuVendor::from_name("SwiftShader"), GpuVendor::Software);
        assert_eq!(GpuVendor::from_name("unknown device"), GpuVendor::Unknown);
    }

    #[test]
    fn driver_detection_nvk() {
        assert_eq!(
            GpuDriver::from_adapter_info("NVIDIA", "NVK", "Mesa 24.0"),
            GpuDriver::Nvk
        );
        assert_eq!(
            GpuDriver::from_adapter_info("NVIDIA", "mesa", "nouveau/nvk"),
            GpuDriver::Nvk
        );
    }

    #[test]
    fn driver_detection_radv() {
        assert_eq!(
            GpuDriver::from_adapter_info("AMD RADV", "radv", "Mesa 24.0"),
            GpuDriver::Radv
        );
    }

    #[test]
    fn driver_detection_nvidia_proprietary() {
        assert_eq!(
            GpuDriver::from_adapter_info("NVIDIA GeForce RTX 4090", "nvidia", "550.0"),
            GpuDriver::NvidiaProprietary
        );
    }

    #[test]
    fn driver_detection_software() {
        assert_eq!(
            GpuDriver::from_adapter_info("llvmpipe", "", ""),
            GpuDriver::Software
        );
    }

    #[test]
    fn gpu_info_f64_builtins() {
        let nvidia_prop = GpuInfo {
            index: 0,
            name: "RTX 4090".into(),
            vendor: GpuVendor::Nvidia,
            driver: GpuDriver::NvidiaProprietary,
            gflops: 10_000.0,
            busy: false,
        };
        assert!(nvidia_prop.supports_f64_builtins());

        let nvk = GpuInfo {
            index: 0,
            name: "RTX 4090 (NVK)".into(),
            vendor: GpuVendor::Nvidia,
            driver: GpuDriver::Nvk,
            gflops: 10_000.0,
            busy: false,
        };
        assert!(!nvk.supports_f64_builtins());
    }

    #[test]
    fn gpu_info_compute_capable() {
        let discrete = GpuInfo {
            index: 0,
            name: "RTX 4090".into(),
            vendor: GpuVendor::Nvidia,
            driver: GpuDriver::NvidiaProprietary,
            gflops: 10_000.0,
            busy: false,
        };
        assert!(discrete.is_compute_capable());

        let software = GpuInfo {
            index: 0,
            name: "llvmpipe".into(),
            vendor: GpuVendor::Software,
            driver: GpuDriver::Software,
            gflops: 50.0,
            busy: false,
        };
        assert!(!software.is_compute_capable());

        let weak_gpu = GpuInfo {
            index: 0,
            name: "Intel Iris".into(),
            vendor: GpuVendor::Intel,
            driver: GpuDriver::Intel,
            gflops: 200.0,
            busy: false,
        };
        assert!(!weak_gpu.is_compute_capable());
    }
}

// SPDX-License-Identifier: AGPL-3.0-or-later
//! Device topology detection: device class and capability classification.
//!
//! Vendor-agnostic: classifies devices by form factor and capabilities,
//! not by manufacturer or driver name.

/// Device class for capability-based routing (vendor-agnostic).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceClass {
    /// Discrete GPU (PCIe-attached, any vendor)
    DiscreteGpu,
    /// Integrated GPU (unified memory, any vendor)
    IntegratedGpu,
    /// Software rasterizer (llvmpipe, `SwiftShader`, CPU)
    Software,
    /// Unknown or unrecognized device
    Unknown,
}

impl DeviceClass {
    /// Detect device class from wgpu `DeviceType` and adapter name.
    #[must_use]
    pub fn from_device_type(device_type: wgpu::DeviceType, name: &str) -> Self {
        match device_type {
            wgpu::DeviceType::DiscreteGpu => Self::DiscreteGpu,
            wgpu::DeviceType::IntegratedGpu => Self::IntegratedGpu,
            wgpu::DeviceType::Cpu => Self::Software,
            _ => {
                let lower = name.to_lowercase();
                if lower.contains("llvmpipe")
                    || lower.contains("software")
                    || lower.contains("swiftshader")
                {
                    Self::Software
                } else {
                    Self::Unknown
                }
            }
        }
    }
}

/// Workload classification for intelligent GPU routing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
    /// Device class (discrete/integrated/software)
    pub device_class: DeviceClass,
    /// Estimated peak GFLOPS for capacity weighting
    pub gflops: f64,
    /// Whether the GPU is currently busy
    pub busy: bool,
    /// Whether native f64 builtins (exp, log, sin, cos) work on this device
    pub f64_builtins_available: bool,
}

impl GpuInfo {
    /// Returns true if this GPU supports native f64 builtins (exp, log, pow).
    #[must_use]
    pub fn supports_f64_builtins(&self) -> bool {
        self.f64_builtins_available
    }

    /// Returns true if this GPU is suitable for compute workloads (>=500 GFLOPS).
    #[must_use]
    pub fn is_compute_capable(&self) -> bool {
        matches!(
            self.device_class,
            DeviceClass::DiscreteGpu | DeviceClass::IntegratedGpu
        ) && self.gflops >= 500.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn device_class_from_discrete() {
        assert_eq!(
            DeviceClass::from_device_type(wgpu::DeviceType::DiscreteGpu, "RTX 4090"),
            DeviceClass::DiscreteGpu
        );
    }

    #[test]
    fn device_class_from_integrated() {
        assert_eq!(
            DeviceClass::from_device_type(wgpu::DeviceType::IntegratedGpu, "Intel Iris Xe"),
            DeviceClass::IntegratedGpu
        );
    }

    #[test]
    fn device_class_from_cpu() {
        assert_eq!(
            DeviceClass::from_device_type(wgpu::DeviceType::Cpu, "llvmpipe"),
            DeviceClass::Software
        );
    }

    #[test]
    fn device_class_from_name_fallback() {
        assert_eq!(
            DeviceClass::from_device_type(wgpu::DeviceType::Other, "SwiftShader"),
            DeviceClass::Software
        );
        assert_eq!(
            DeviceClass::from_device_type(wgpu::DeviceType::Other, "unknown device"),
            DeviceClass::Unknown
        );
    }

    #[test]
    fn gpu_info_f64_builtins() {
        let with_builtins = GpuInfo {
            index: 0,
            name: "RTX 4090".into(),
            device_class: DeviceClass::DiscreteGpu,
            gflops: 10_000.0,
            busy: false,
            f64_builtins_available: true,
        };
        assert!(with_builtins.supports_f64_builtins());

        let without_builtins = GpuInfo {
            index: 0,
            name: "RTX 4090 (NVK)".into(),
            device_class: DeviceClass::DiscreteGpu,
            gflops: 10_000.0,
            busy: false,
            f64_builtins_available: false,
        };
        assert!(!without_builtins.supports_f64_builtins());
    }

    #[test]
    fn gpu_info_compute_capable() {
        let discrete = GpuInfo {
            index: 0,
            name: "RTX 4090".into(),
            device_class: DeviceClass::DiscreteGpu,
            gflops: 10_000.0,
            busy: false,
            f64_builtins_available: true,
        };
        assert!(discrete.is_compute_capable());

        let software = GpuInfo {
            index: 0,
            name: "llvmpipe".into(),
            device_class: DeviceClass::Software,
            gflops: 50.0,
            busy: false,
            f64_builtins_available: false,
        };
        assert!(!software.is_compute_capable());

        let weak_gpu = GpuInfo {
            index: 0,
            name: "Intel Iris".into(),
            device_class: DeviceClass::IntegratedGpu,
            gflops: 200.0,
            busy: false,
            f64_builtins_available: false,
        };
        assert!(!weak_gpu.is_compute_capable());
    }
}

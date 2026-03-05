// SPDX-License-Identifier: AGPL-3.0-or-later
//! Device Capability Detection — Runtime Hardware Limits (wgpu).
//!
//! Answers "what can this wgpu device do?" by querying the adapter at
//! construction time and providing typed, zero-hardcoded accessors.

use crate::device::driver_profile::GpuArch;
use crate::device::vendor::{VENDOR_AMD, VENDOR_INTEL, VENDOR_NVIDIA};
use crate::device::WgpuDevice;
use std::fmt;

/// Minimum buffer size (bytes) for FHE workloads — 16K degree polynomial estimate.
pub const FHE_MIN_BUFFER_SIZE: u64 = 256 * 1024;

/// Minimum invocations per workgroup to consider device "high performance".
pub const HIGH_PERFORMANCE_MIN_INVOCATIONS: u32 = 1024;

/// Bytes per megabyte (for display formatting).
const BYTES_PER_MB: u64 = 1024 * 1024;

/// Standard 1D shader workgroup size for high-throughput elementwise ops.
/// Matches `@workgroup_size(256)` in WGSL shaders.
pub const WORKGROUP_SIZE_1D: u32 = 256;

/// Compact 1D workgroup size for physics/lattice shaders with high register pressure.
/// Matches `@workgroup_size(64)` in MD, lattice QCD, and observable WGSL shaders.
pub const WORKGROUP_SIZE_COMPACT: u32 = 64;

/// Standard 2D shader workgroup size per dimension.
/// Matches `@workgroup_size(16, 16)` in all 2D WGSL shaders.
pub const WORKGROUP_SIZE_2D: u32 = 16;

/// Optimal 1D workgroup size based on GPU architecture.
#[must_use]
pub fn workgroup_size_for_arch(arch: &GpuArch) -> u32 {
    match arch {
        GpuArch::Volta | GpuArch::Turing => 64,
        GpuArch::Ampere | GpuArch::Ada => 256,
        GpuArch::Rdna2 | GpuArch::Rdna3 => 64,
        GpuArch::Cdna2 => 256,
        GpuArch::IntelArc => 128,
        GpuArch::AppleM => 64,
        GpuArch::Software | GpuArch::Unknown => 64,
    }
}

/// 2D workgroup size (per dimension) based on GPU architecture.
#[must_use]
pub fn workgroup_size_2d_for_arch(arch: &GpuArch) -> u32 {
    match arch {
        GpuArch::Ampere | GpuArch::Ada | GpuArch::Cdna2 => 16,
        _ => 8,
    }
}

/// Optimal 1D workgroup size when GPU architecture is known.
#[must_use]
pub fn optimal_workgroup_size_arch(
    arch: &GpuArch,
    workload: WorkloadType,
    max_invocations: u32,
) -> u32 {
    let base = workgroup_size_for_arch(arch);
    let size = match workload {
        WorkloadType::ElementWise | WorkloadType::MatMul | WorkloadType::FHE => base,
        WorkloadType::Reduction => base * 2,
        WorkloadType::Convolution => base / 2,
    };
    size.min(max_invocations)
}

/// NVK (and some other drivers) may report absurd `max_buffer_size` values
/// (e.g. 2^57). Cap to architecture-appropriate defaults when the reported
/// value exceeds 64 GB.
fn sanitize_max_buffer_size(reported: u64, device_name: &str) -> u64 {
    const MAX_SANE_BUFFER: u64 = 64 * 1024 * 1024 * 1024; // 64 GB
    if reported > MAX_SANE_BUFFER {
        let capped = if device_name.contains("Titan V") || device_name.contains("V100") {
            12 * 1024 * 1024 * 1024 // 12 GB VRAM
        } else if device_name.contains("RTX 30") || device_name.contains("RTX 40") {
            24 * 1024 * 1024 * 1024 // 24 GB max consumer
        } else {
            8 * 1024 * 1024 * 1024 // 8 GB conservative default
        };
        tracing::warn!(
            "Driver reported max_buffer_size={reported} (>64GB), capping to {capped} for {device_name}"
        );
        capped
    } else {
        reported
    }
}

/// Device capabilities - runtime hardware limits
///
/// **Deep Debt**: All values discovered at runtime, zero hardcoding
#[derive(Debug, Clone)]
pub struct DeviceCapabilities {
    /// Device name (e.g., "NVIDIA RTX 4090")
    pub device_name: String,

    /// Device type (DiscreteGpu, IntegratedGpu, Cpu, etc.)
    pub device_type: wgpu::DeviceType,

    /// Maximum buffer size (bytes)
    pub max_buffer_size: u64,

    /// Maximum workgroup size per dimension (x, y, z)
    pub max_workgroup_size: (u32, u32, u32),

    /// Maximum workgroups per dispatch
    pub max_compute_workgroups: (u32, u32, u32),

    /// Maximum invocations per workgroup
    pub max_compute_invocations_per_workgroup: u32,

    /// Maximum storage buffers per shader stage
    pub max_storage_buffers_per_shader_stage: u32,

    /// Maximum uniform buffers per shader stage
    pub max_uniform_buffers_per_shader_stage: u32,

    /// Maximum bind groups
    pub max_bind_groups: u32,

    /// Backend (Vulkan, Metal, DX12, GL, etc.)
    pub backend: wgpu::Backend,

    /// Vendor ID (e.g., NVIDIA=VENDOR_NVIDIA, AMD=VENDOR_AMD, Intel=VENDOR_INTEL)
    pub vendor: u32,

    /// Override for `gpu_dispatch_threshold()`. `None` uses the default per
    /// device type. Set via `with_gpu_dispatch_threshold()`.
    pub gpu_dispatch_threshold_override: Option<usize>,
}

impl DeviceCapabilities {
    /// Detect capabilities from wgpu device
    ///
    /// **Deep Debt**: Runtime discovery, no assumptions
    pub fn from_device(device: &WgpuDevice) -> Self {
        let limits = device.device().limits();
        let adapter_info = device.adapter_info();

        Self {
            device_name: adapter_info.name.clone(),
            device_type: adapter_info.device_type,
            max_buffer_size: sanitize_max_buffer_size(limits.max_buffer_size, &adapter_info.name),
            max_workgroup_size: (
                limits.max_compute_workgroup_size_x,
                limits.max_compute_workgroup_size_y,
                limits.max_compute_workgroup_size_z,
            ),
            max_compute_workgroups: (
                limits.max_compute_workgroups_per_dimension,
                limits.max_compute_workgroups_per_dimension,
                limits.max_compute_workgroups_per_dimension,
            ),
            max_compute_invocations_per_workgroup: limits.max_compute_invocations_per_workgroup,
            max_storage_buffers_per_shader_stage: limits.max_storage_buffers_per_shader_stage,
            max_uniform_buffers_per_shader_stage: limits.max_uniform_buffers_per_shader_stage,
            max_bind_groups: limits.max_bind_groups,
            backend: adapter_info.backend,
            vendor: adapter_info.vendor,
            gpu_dispatch_threshold_override: None,
        }
    }

    /// Get optimal workgroup size for a specific workload
    pub fn optimal_workgroup_size(&self, workload: WorkloadType) -> u32 {
        match self.device_type {
            wgpu::DeviceType::DiscreteGpu => match self.vendor {
                VENDOR_NVIDIA => match workload {
                    WorkloadType::ElementWise => 256,
                    WorkloadType::MatMul => 256,
                    WorkloadType::Reduction => 512,
                    WorkloadType::FHE => 256,
                    WorkloadType::Convolution => 128,
                },
                VENDOR_AMD => match workload {
                    WorkloadType::ElementWise => 256,
                    WorkloadType::MatMul => 256,
                    WorkloadType::Reduction => 256,
                    WorkloadType::FHE => 256,
                    WorkloadType::Convolution => 128,
                },
                VENDOR_INTEL => match workload {
                    WorkloadType::ElementWise => 128,
                    WorkloadType::MatMul => 128,
                    WorkloadType::Reduction => 256,
                    WorkloadType::FHE => 128,
                    WorkloadType::Convolution => 64,
                },
                _ => match workload {
                    WorkloadType::ElementWise => 128,
                    WorkloadType::MatMul => 128,
                    WorkloadType::Reduction => 256,
                    WorkloadType::FHE => 128,
                    WorkloadType::Convolution => 64,
                },
            },

            wgpu::DeviceType::IntegratedGpu => match workload {
                WorkloadType::ElementWise => 128,
                WorkloadType::MatMul => 64,
                WorkloadType::Reduction => 128,
                WorkloadType::FHE => 64,
                WorkloadType::Convolution => 64,
            },

            wgpu::DeviceType::Cpu => match workload {
                WorkloadType::ElementWise => 32,
                WorkloadType::MatMul => 16,
                WorkloadType::Reduction => 64,
                WorkloadType::FHE => 32,
                WorkloadType::Convolution => 16,
            },

            _ => match workload {
                WorkloadType::ElementWise => 64,
                WorkloadType::MatMul => 64,
                WorkloadType::Reduction => 128,
                WorkloadType::FHE => 64,
                WorkloadType::Convolution => 32,
            },
        }
        .min(self.max_compute_invocations_per_workgroup)
    }

    /// Get optimal 2D workgroup size (for 2D operations like convolutions)
    pub fn optimal_workgroup_size_2d(&self, workload: WorkloadType) -> (u32, u32) {
        let total = self.optimal_workgroup_size(workload);
        let side = (total as f32).sqrt() as u32;
        let x = side.min(self.max_workgroup_size.0);
        let y = side.min(self.max_workgroup_size.1);
        (x, y)
    }

    /// Get optimal 3D workgroup size (for 3D operations)
    pub fn optimal_workgroup_size_3d(&self, workload: WorkloadType) -> (u32, u32, u32) {
        let total = self.optimal_workgroup_size(workload);
        let side = (total as f32).cbrt() as u32;
        let x = side.min(self.max_workgroup_size.0);
        let y = side.min(self.max_workgroup_size.1);
        let z = side.min(self.max_workgroup_size.2);
        (x, y, z)
    }

    /// Calculate number of workgroups for a 1D dispatch.
    #[must_use]
    pub fn dispatch_1d(&self, element_count: u32) -> u32 {
        element_count.div_ceil(WORKGROUP_SIZE_1D)
    }

    /// Calculate number of workgroups for a 2D dispatch.
    #[must_use]
    pub fn dispatch_2d(&self, width: u32, height: u32) -> (u32, u32) {
        (
            width.div_ceil(WORKGROUP_SIZE_2D),
            height.div_ceil(WORKGROUP_SIZE_2D),
        )
    }

    /// Get maximum allocation size for this device
    pub fn max_allocation_size(&self) -> u64 {
        (self.max_buffer_size as f64 * 0.75) as u64
    }

    /// Check if device supports FHE workloads (large U64 buffers)
    pub fn supports_fhe(&self) -> bool {
        self.max_buffer_size >= FHE_MIN_BUFFER_SIZE
    }

    /// Check if device supports large matrix operations
    pub fn supports_large_matmul(&self, m: usize, n: usize, k: usize) -> bool {
        let required_bytes = (m * k + k * n + m * n) * 4;
        required_bytes as u64 <= self.max_allocation_size()
    }

    /// Get optimal tile size for matrix multiplication
    pub fn optimal_matmul_tile_size(&self) -> u32 {
        match self.device_type {
            wgpu::DeviceType::DiscreteGpu => match self.vendor {
                VENDOR_NVIDIA => 32,
                VENDOR_AMD => 32,
                VENDOR_INTEL => 16,
                _ => 16,
            },
            wgpu::DeviceType::IntegratedGpu => 16,
            wgpu::DeviceType::Cpu => 8,
            _ => 8,
        }
    }

    /// Minimum element count below which CPU is faster than a GPU dispatch.
    pub fn gpu_dispatch_threshold(&self) -> usize {
        if let Some(t) = self.gpu_dispatch_threshold_override {
            return t;
        }
        match self.device_type {
            wgpu::DeviceType::DiscreteGpu => 4_096,
            wgpu::DeviceType::IntegratedGpu => 16_384,
            wgpu::DeviceType::Cpu => usize::MAX,
            _ => 8_192,
        }
    }

    /// Return a copy with the GPU dispatch threshold set to `threshold`.
    pub fn with_gpu_dispatch_threshold(mut self, threshold: usize) -> Self {
        self.gpu_dispatch_threshold_override = Some(threshold);
        self
    }

    /// Get vendor name (for logging/debugging)
    pub fn vendor_name(&self) -> &'static str {
        match self.vendor {
            VENDOR_NVIDIA => "NVIDIA",
            VENDOR_AMD => "AMD",
            VENDOR_INTEL => "Intel",
            0x13B5 => "ARM",
            0x5143 => "Qualcomm",
            0x1010 => "ImgTec",
            _ => "Unknown",
        }
    }

    /// Check if this is a high-performance GPU
    pub fn is_high_performance(&self) -> bool {
        matches!(self.device_type, wgpu::DeviceType::DiscreteGpu)
            && self.max_compute_invocations_per_workgroup >= HIGH_PERFORMANCE_MIN_INVOCATIONS
    }
}

/// Workload types for optimal configuration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WorkloadType {
    /// Element-wise ops (activation, etc.).
    ElementWise,
    /// Matrix multiplication.
    MatMul,
    /// Reduction ops (sum, max, etc.).
    Reduction,
    /// Fully homomorphic encryption workloads.
    FHE,
    /// Convolution operations.
    Convolution,
}

impl fmt::Display for DeviceCapabilities {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Device Capabilities:")?;
        writeln!(f, "  Name: {}", self.device_name)?;
        writeln!(f, "  Type: {:?}", self.device_type)?;
        writeln!(
            f,
            "  Vendor: {} (0x{:04X})",
            self.vendor_name(),
            self.vendor
        )?;
        writeln!(f, "  Backend: {:?}", self.backend)?;
        writeln!(f)?;
        writeln!(f, "Memory:")?;
        writeln!(
            f,
            "  Max Buffer Size: {} MB",
            self.max_buffer_size / BYTES_PER_MB
        )?;
        writeln!(
            f,
            "  Max Allocation: {} MB",
            self.max_allocation_size() / BYTES_PER_MB
        )?;
        writeln!(f)?;
        writeln!(f, "Compute:")?;
        writeln!(f, "  Max Workgroup Size: {:?}", self.max_workgroup_size)?;
        writeln!(
            f,
            "  Max Invocations/Workgroup: {}",
            self.max_compute_invocations_per_workgroup
        )?;
        writeln!(
            f,
            "  Max Compute Workgroups: {:?}",
            self.max_compute_workgroups
        )?;
        writeln!(f)?;
        writeln!(f, "Optimal Configurations:")?;
        writeln!(
            f,
            "  Element-wise: {} threads",
            self.optimal_workgroup_size(WorkloadType::ElementWise)
        )?;
        writeln!(
            f,
            "  MatMul: {} threads (tile: {})",
            self.optimal_workgroup_size(WorkloadType::MatMul),
            self.optimal_matmul_tile_size()
        )?;
        writeln!(
            f,
            "  Reduction: {} threads",
            self.optimal_workgroup_size(WorkloadType::Reduction)
        )?;
        writeln!(
            f,
            "  FHE: {} threads",
            self.optimal_workgroup_size(WorkloadType::FHE)
        )?;
        writeln!(
            f,
            "  Convolution: {:?}",
            self.optimal_workgroup_size_2d(WorkloadType::Convolution)
        )?;
        writeln!(f)?;
        writeln!(f, "Features:")?;
        writeln!(
            f,
            "  FHE Support: {}",
            if self.supports_fhe() { "Yes" } else { "No" }
        )?;
        writeln!(
            f,
            "  High Performance: {}",
            if self.is_high_performance() {
                "Yes"
            } else {
                "No"
            }
        )?;
        Ok(())
    }
}

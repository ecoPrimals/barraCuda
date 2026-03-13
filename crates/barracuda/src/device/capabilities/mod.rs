// SPDX-License-Identifier: AGPL-3.0-only
//! Device Capability Detection — Runtime Hardware Limits.
//!
//! Answers the question **"what can this wgpu device do?"** by querying the
//! adapter at construction time and providing typed, zero-hardcoded accessors.
//!
//! For driver/compiler identity and shader strategy (the "who is driving?"
//! question), see [`crate::device::driver_profile`].
//!
//! # Philosophy
//!
//! - ✅ **Query, don't hardcode**: ask the device for limits
//! - ✅ **Adapt to hardware**: different optimal configs per vendor
//! - ✅ **Portability**: works on any WebGPU device

mod device_info;
mod wgpu_caps;

// Re-export driver/compiler types so callers that previously imported them
// from `capabilities` continue to compile without path changes.
pub use crate::device::driver_profile::{
    CompilerKind, DriverKind, EigensolveStrategy, Fp64Rate, Fp64Strategy, GpuArch,
    GpuDriverProfile, Workaround,
};

pub use device_info::{Capability, DeviceInfo};
pub use device_info::{
    VfioGpuInfo, build_device_info, detect_system_memory_bytes, discover_vfio_gpus,
    estimate_system_memory, is_gpu_available, is_npu_available, is_vfio_gpu_available,
};
pub use wgpu_caps::{
    DeviceCapabilities, FHE_MIN_BUFFER_SIZE, WORKGROUP_SIZE_1D, WORKGROUP_SIZE_2D,
    WORKGROUP_SIZE_COMPACT, WorkloadType, optimal_workgroup_size_arch, workgroup_size_2d_for_arch,
    workgroup_size_for_arch,
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::vendor::{VENDOR_INTEL, VENDOR_NVIDIA};

    const MOCK_DEVICE_MAX_BUFFER_SIZE: u64 = 1024 * 1024 * 1024;
    const MOCK_DEVICE_MAX_BUFFER_SIZE_LIMITED: u64 = 128 * 1024;

    #[test]
    fn test_workgroup_sizes_within_limits() {
        let caps = DeviceCapabilities {
            device_name: "Test GPU".to_string(),
            device_type: wgpu::DeviceType::DiscreteGpu,
            max_buffer_size: MOCK_DEVICE_MAX_BUFFER_SIZE,
            max_workgroup_size: (256, 256, 64),
            max_compute_workgroups: (65_535, 65_535, 65_535),
            max_compute_invocations_per_workgroup: 256,
            max_storage_buffers_per_shader_stage: 8,
            max_uniform_buffers_per_shader_stage: 12,
            max_bind_groups: 4,
            backend: wgpu::Backend::Vulkan,
            vendor: VENDOR_NVIDIA,
            gpu_dispatch_threshold_override: None,
            subgroup_min_size: 32,
            subgroup_max_size: 32,
            f64_shaders: true,
            f64_shared_memory: false,
        };

        let workloads = vec![
            WorkloadType::ElementWise,
            WorkloadType::MatMul,
            WorkloadType::Reduction,
            WorkloadType::FHE,
            WorkloadType::Convolution,
        ];

        for workload in workloads {
            let size = caps.optimal_workgroup_size(workload);
            assert!(
                size <= caps.max_compute_invocations_per_workgroup,
                "Workgroup size {} exceeds max {} for workload {:?}",
                size,
                caps.max_compute_invocations_per_workgroup,
                workload
            );

            let (x, y) = caps.optimal_workgroup_size_2d(workload);
            assert!(x <= caps.max_workgroup_size.0);
            assert!(y <= caps.max_workgroup_size.1);
            assert!(x * y <= caps.max_compute_invocations_per_workgroup);

            let (x, y, z) = caps.optimal_workgroup_size_3d(workload);
            assert!(x <= caps.max_workgroup_size.0);
            assert!(y <= caps.max_workgroup_size.1);
            assert!(z <= caps.max_workgroup_size.2);
            assert!(x * y * z <= caps.max_compute_invocations_per_workgroup);
        }
    }

    #[test]
    fn test_fhe_support_detection() {
        let caps_supported = DeviceCapabilities {
            device_name: "Large GPU".to_string(),
            device_type: wgpu::DeviceType::DiscreteGpu,
            max_buffer_size: MOCK_DEVICE_MAX_BUFFER_SIZE,
            max_workgroup_size: (256, 256, 64),
            max_compute_workgroups: (65_535, 65_535, 65_535),
            max_compute_invocations_per_workgroup: 1024,
            max_storage_buffers_per_shader_stage: 8,
            max_uniform_buffers_per_shader_stage: 12,
            max_bind_groups: 4,
            backend: wgpu::Backend::Vulkan,
            vendor: VENDOR_NVIDIA,
            gpu_dispatch_threshold_override: None,
            subgroup_min_size: 32,
            subgroup_max_size: 32,
            f64_shaders: true,
            f64_shared_memory: false,
        };

        assert!(caps_supported.supports_fhe());

        let caps_limited = DeviceCapabilities {
            device_name: "Small GPU".to_string(),
            device_type: wgpu::DeviceType::IntegratedGpu,
            max_buffer_size: MOCK_DEVICE_MAX_BUFFER_SIZE_LIMITED,
            max_workgroup_size: (128, 128, 32),
            max_compute_workgroups: (65_535, 65_535, 65_535),
            max_compute_invocations_per_workgroup: 256,
            max_storage_buffers_per_shader_stage: 8,
            max_uniform_buffers_per_shader_stage: 12,
            max_bind_groups: 4,
            backend: wgpu::Backend::Vulkan,
            vendor: VENDOR_INTEL,
            gpu_dispatch_threshold_override: None,
            subgroup_min_size: 8,
            subgroup_max_size: 32,
            f64_shaders: true,
            f64_shared_memory: false,
        };

        assert!(!caps_limited.supports_fhe());
    }

    #[test]
    fn test_workgroup_size_volta() {
        assert_eq!(workgroup_size_for_arch(&GpuArch::Volta), 64);
    }

    #[test]
    fn test_workgroup_size_ada() {
        assert_eq!(workgroup_size_for_arch(&GpuArch::Ada), 256);
    }

    #[test]
    fn test_workgroup_size_rdna2() {
        assert_eq!(workgroup_size_for_arch(&GpuArch::Rdna2), 64);
    }

    #[test]
    fn test_workgroup_size_unknown() {
        assert_eq!(workgroup_size_for_arch(&GpuArch::Unknown), 64);
    }

    #[test]
    fn test_workgroup_2d_ampere() {
        assert_eq!(workgroup_size_2d_for_arch(&GpuArch::Ampere), 16);
    }

    #[test]
    fn test_workgroup_2d_volta() {
        assert_eq!(workgroup_size_2d_for_arch(&GpuArch::Volta), 8);
    }
}

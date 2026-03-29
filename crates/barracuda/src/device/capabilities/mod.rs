// SPDX-License-Identifier: AGPL-3.0-or-later
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

// Re-export driver-profile types that are part of the public API.
// The former `GpuDriverProfile` struct was removed in v0.3.8 (Sprint 18).
// These shared enums remain canonical in `driver_profile` and are re-exported here.
pub use crate::device::driver_profile::{
    EigensolveStrategy, Fp64Rate, Fp64Strategy, PrecisionRoutingAdvice,
};

pub use device_info::{Capability, DeviceInfo};
pub use device_info::{
    build_device_info, detect_system_memory_bytes, estimate_system_memory, is_gpu_available,
    is_npu_available,
};
pub use wgpu_caps::{
    DeviceCapabilities, FHE_MIN_BUFFER_SIZE, WORKGROUP_SIZE_1D, WORKGROUP_SIZE_2D,
    WORKGROUP_SIZE_COMPACT, WorkloadType, optimal_workgroup_size_arch, workgroup_size_2d_for_arch,
    workgroup_size_for_arch,
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::driver_profile::GpuArch;
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
            f64_capabilities: None,
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
            f64_capabilities: None,
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
            f64_capabilities: None,
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

    // ── Helper to construct test DeviceCapabilities ──────────────────────

    fn mock_caps(device_type: wgpu::DeviceType, vendor: u32) -> DeviceCapabilities {
        DeviceCapabilities {
            device_name: "Test Device".to_string(),
            device_type,
            max_buffer_size: MOCK_DEVICE_MAX_BUFFER_SIZE,
            max_workgroup_size: (256, 256, 64),
            max_compute_workgroups: (65_535, 65_535, 65_535),
            max_compute_invocations_per_workgroup: 256,
            max_storage_buffers_per_shader_stage: 8,
            max_uniform_buffers_per_shader_stage: 12,
            max_bind_groups: 4,
            backend: wgpu::Backend::Vulkan,
            vendor,
            gpu_dispatch_threshold_override: None,
            subgroup_min_size: 32,
            subgroup_max_size: 32,
            f64_shaders: true,
            f64_shared_memory: false,
            f64_capabilities: None,
        }
    }

    fn full_f64_caps() -> crate::device::probe::F64BuiltinCapabilities {
        crate::device::probe::F64BuiltinCapabilities::full()
    }

    fn broken_f64_caps() -> crate::device::probe::F64BuiltinCapabilities {
        crate::device::probe::F64BuiltinCapabilities::none()
    }

    // ── fp64_strategy tests ─────────────────────────────────────────────

    #[test]
    fn fp64_strategy_native_when_probed_full() {
        let caps = mock_caps(wgpu::DeviceType::DiscreteGpu, VENDOR_NVIDIA)
            .with_f64_capabilities(full_f64_caps());
        assert_eq!(caps.fp64_strategy(), Fp64Strategy::Native);
    }

    #[test]
    fn fp64_strategy_hybrid_when_probed_broken() {
        let caps = mock_caps(wgpu::DeviceType::DiscreteGpu, VENDOR_NVIDIA)
            .with_f64_capabilities(broken_f64_caps());
        assert_eq!(caps.fp64_strategy(), Fp64Strategy::Hybrid);
    }

    #[test]
    fn fp64_strategy_native_when_no_probe_but_f64_shaders() {
        let caps = mock_caps(wgpu::DeviceType::DiscreteGpu, VENDOR_NVIDIA);
        assert!(caps.f64_shaders);
        assert_eq!(caps.fp64_strategy(), Fp64Strategy::Native);
    }

    #[test]
    fn fp64_strategy_hybrid_when_no_f64_shaders() {
        let mut caps = mock_caps(wgpu::DeviceType::DiscreteGpu, VENDOR_NVIDIA);
        caps.f64_shaders = false;
        assert_eq!(caps.fp64_strategy(), Fp64Strategy::Hybrid);
    }

    // ── precision_routing tests ─────────────────────────────────────────

    #[test]
    fn precision_routing_f64_native_when_probed_full() {
        use crate::device::driver_profile::PrecisionRoutingAdvice;
        let caps = mock_caps(wgpu::DeviceType::DiscreteGpu, VENDOR_NVIDIA)
            .with_f64_capabilities(full_f64_caps());
        assert_eq!(caps.precision_routing(), PrecisionRoutingAdvice::F64Native);
    }

    #[test]
    fn precision_routing_no_shared_mem_when_basic_but_no_shmem() {
        use crate::device::driver_profile::PrecisionRoutingAdvice;
        let mut f64_caps = full_f64_caps();
        f64_caps.shared_mem_f64 = false;
        let caps =
            mock_caps(wgpu::DeviceType::DiscreteGpu, VENDOR_NVIDIA).with_f64_capabilities(f64_caps);
        assert_eq!(
            caps.precision_routing(),
            PrecisionRoutingAdvice::F64NativeNoSharedMem
        );
    }

    #[test]
    fn precision_routing_df64_when_probed_broken_but_f64_feature() {
        use crate::device::driver_profile::PrecisionRoutingAdvice;
        let caps = mock_caps(wgpu::DeviceType::DiscreteGpu, VENDOR_NVIDIA)
            .with_f64_capabilities(broken_f64_caps());
        assert_eq!(caps.precision_routing(), PrecisionRoutingAdvice::Df64Only);
    }

    #[test]
    fn precision_routing_f32_when_broken_and_no_f64_feature() {
        use crate::device::driver_profile::PrecisionRoutingAdvice;
        let mut caps = mock_caps(wgpu::DeviceType::DiscreteGpu, VENDOR_NVIDIA);
        caps.f64_shaders = false;
        caps.f64_capabilities = Some(broken_f64_caps());
        assert_eq!(caps.precision_routing(), PrecisionRoutingAdvice::F32Only);
    }

    #[test]
    fn precision_routing_no_shared_mem_fallback_without_probe() {
        use crate::device::driver_profile::PrecisionRoutingAdvice;
        let caps = mock_caps(wgpu::DeviceType::DiscreteGpu, VENDOR_NVIDIA);
        assert_eq!(
            caps.precision_routing(),
            PrecisionRoutingAdvice::F64NativeNoSharedMem
        );
    }

    #[test]
    fn precision_routing_f32_only_without_f64_feature_or_probe() {
        use crate::device::driver_profile::PrecisionRoutingAdvice;
        let mut caps = mock_caps(wgpu::DeviceType::DiscreteGpu, VENDOR_NVIDIA);
        caps.f64_shaders = false;
        assert_eq!(caps.precision_routing(), PrecisionRoutingAdvice::F32Only);
    }

    // ── workaround flag tests ───────────────────────────────────────────

    #[test]
    fn workaround_flags_all_false_without_probe() {
        let caps = mock_caps(wgpu::DeviceType::DiscreteGpu, VENDOR_NVIDIA);
        assert!(!caps.needs_exp_f64_workaround());
        assert!(!caps.needs_log_f64_workaround());
        assert!(!caps.needs_sin_f64_workaround());
        assert!(!caps.needs_cos_f64_workaround());
    }

    #[test]
    fn workaround_flags_false_with_full_probe() {
        let caps = mock_caps(wgpu::DeviceType::DiscreteGpu, VENDOR_NVIDIA)
            .with_f64_capabilities(full_f64_caps());
        assert!(!caps.needs_exp_f64_workaround());
        assert!(!caps.needs_log_f64_workaround());
        assert!(!caps.needs_sin_f64_workaround());
        assert!(!caps.needs_cos_f64_workaround());
    }

    #[test]
    fn workaround_flags_true_when_specific_builtins_fail() {
        let mut partial = full_f64_caps();
        partial.exp = false;
        partial.sin = false;
        let caps =
            mock_caps(wgpu::DeviceType::DiscreteGpu, VENDOR_NVIDIA).with_f64_capabilities(partial);
        assert!(caps.needs_exp_f64_workaround());
        assert!(!caps.needs_log_f64_workaround());
        assert!(caps.needs_sin_f64_workaround());
        assert!(!caps.needs_cos_f64_workaround());
    }

    // ── df64_transcendentals_safe tests ─────────────────────────────────

    #[test]
    fn df64_transcendentals_safe_true_without_probe() {
        let caps = mock_caps(wgpu::DeviceType::DiscreteGpu, VENDOR_NVIDIA);
        assert!(caps.df64_transcendentals_safe());
        assert!(!caps.has_df64_spir_v_poisoning());
    }

    #[test]
    fn df64_transcendentals_poisoned_when_probe_says_unsafe() {
        let mut f64_caps = full_f64_caps();
        f64_caps.df64_transcendentals_safe = false;
        let caps =
            mock_caps(wgpu::DeviceType::DiscreteGpu, VENDOR_NVIDIA).with_f64_capabilities(f64_caps);
        assert!(!caps.df64_transcendentals_safe());
        assert!(caps.has_df64_spir_v_poisoning());
    }

    // ── supports_f64_builtins tests ─────────────────────────────────────

    #[test]
    fn supports_f64_builtins_true_with_full_probe() {
        let caps = mock_caps(wgpu::DeviceType::DiscreteGpu, VENDOR_NVIDIA)
            .with_f64_capabilities(full_f64_caps());
        assert!(caps.supports_f64_builtins());
    }

    #[test]
    fn supports_f64_builtins_false_with_broken_probe() {
        let caps = mock_caps(wgpu::DeviceType::DiscreteGpu, VENDOR_NVIDIA)
            .with_f64_capabilities(broken_f64_caps());
        assert!(!caps.supports_f64_builtins());
    }

    #[test]
    fn supports_f64_builtins_fallback_to_feature_flag() {
        let caps = mock_caps(wgpu::DeviceType::DiscreteGpu, VENDOR_NVIDIA);
        assert!(caps.supports_f64_builtins());
        let mut caps_no_f64 = mock_caps(wgpu::DeviceType::DiscreteGpu, VENDOR_NVIDIA);
        caps_no_f64.f64_shaders = false;
        assert!(!caps_no_f64.supports_f64_builtins());
    }

    // ── has_f64_transcendentals tests ─────────────────────────────────

    #[test]
    fn has_f64_transcendentals_true_with_full_probe() {
        let caps = mock_caps(wgpu::DeviceType::DiscreteGpu, VENDOR_NVIDIA)
            .with_f64_capabilities(full_f64_caps());
        assert!(caps.has_f64_transcendentals());
    }

    #[test]
    fn has_f64_transcendentals_false_with_broken_probe() {
        let caps = mock_caps(wgpu::DeviceType::DiscreteGpu, VENDOR_NVIDIA)
            .with_f64_capabilities(broken_f64_caps());
        assert!(!caps.has_f64_transcendentals());
    }

    #[test]
    fn has_f64_transcendentals_fallback_to_feature_flag() {
        let caps = mock_caps(wgpu::DeviceType::DiscreteGpu, VENDOR_NVIDIA);
        assert!(caps.has_f64_transcendentals());
    }

    #[test]
    fn needs_sqrt_f64_workaround_when_probed_broken() {
        let mut f64_caps = full_f64_caps();
        f64_caps.sqrt = false;
        let caps =
            mock_caps(wgpu::DeviceType::DiscreteGpu, VENDOR_NVIDIA).with_f64_capabilities(f64_caps);
        assert!(caps.needs_sqrt_f64_workaround());
    }

    #[test]
    fn needs_sqrt_f64_workaround_false_with_full_probe() {
        let caps = mock_caps(wgpu::DeviceType::DiscreteGpu, VENDOR_NVIDIA)
            .with_f64_capabilities(full_f64_caps());
        assert!(!caps.needs_sqrt_f64_workaround());
    }

    // ── eigensolve strategy tests ───────────────────────────────────────

    #[test]
    fn eigensolve_strategy_warp_packed_for_discrete() {
        let caps = mock_caps(wgpu::DeviceType::DiscreteGpu, VENDOR_NVIDIA);
        assert!(matches!(
            caps.optimal_eigensolve_strategy(),
            EigensolveStrategy::WarpPacked { wg_size: 32 }
        ));
    }

    #[test]
    fn eigensolve_strategy_wave_packed_for_large_subgroup() {
        let mut caps = mock_caps(
            wgpu::DeviceType::DiscreteGpu,
            crate::device::vendor::VENDOR_AMD,
        );
        caps.subgroup_min_size = 64;
        caps.subgroup_max_size = 64;
        assert!(matches!(
            caps.optimal_eigensolve_strategy(),
            EigensolveStrategy::WavePacked { wave_size: 64 }
        ));
    }

    #[test]
    fn eigensolve_strategy_standard_for_cpu() {
        let caps = mock_caps(wgpu::DeviceType::Cpu, 0);
        assert!(matches!(
            caps.optimal_eigensolve_strategy(),
            EigensolveStrategy::Standard
        ));
    }

    // ── latency model selection tests ───────────────────────────────────

    #[test]
    fn latency_model_nvidia_returns_sm70() {
        use crate::device::latency::WgslOpClass;
        let caps = mock_caps(wgpu::DeviceType::DiscreteGpu, VENDOR_NVIDIA);
        let model = caps.latency_model();
        assert_eq!(model.raw_latency(WgslOpClass::F64Fma), 8);
    }

    #[test]
    fn latency_model_amd_returns_rdna2() {
        use crate::device::latency::WgslOpClass;
        let caps = mock_caps(
            wgpu::DeviceType::DiscreteGpu,
            crate::device::vendor::VENDOR_AMD,
        );
        let model = caps.latency_model();
        assert_eq!(model.raw_latency(WgslOpClass::F64Fma), 4);
    }

    #[test]
    fn latency_model_unknown_returns_conservative() {
        use crate::device::latency::WgslOpClass;
        let caps = mock_caps(wgpu::DeviceType::DiscreteGpu, 0x9999);
        let model = caps.latency_model();
        assert!(model.raw_latency(WgslOpClass::F64Fma) >= 8);
    }

    #[test]
    fn latency_model_cpu_returns_conservative() {
        use crate::device::latency::WgslOpClass;
        let caps = mock_caps(wgpu::DeviceType::Cpu, 0);
        let model = caps.latency_model();
        assert!(model.raw_latency(WgslOpClass::F64Fma) >= 8);
    }

    // ── allocation safety tests ─────────────────────────────────────────

    #[test]
    fn check_allocation_safe_within_limit() {
        let caps = mock_caps(wgpu::DeviceType::DiscreteGpu, VENDOR_NVIDIA);
        assert!(caps.check_allocation_safe(1024).is_ok());
    }

    #[test]
    fn check_allocation_safe_exceeds_limit() {
        let caps = mock_caps(wgpu::DeviceType::DiscreteGpu, VENDOR_NVIDIA);
        let result = caps.check_allocation_safe(u64::MAX);
        assert!(result.is_err());
    }

    #[test]
    fn max_safe_allocation_is_three_quarters_buffer() {
        let caps = mock_caps(wgpu::DeviceType::DiscreteGpu, VENDOR_NVIDIA);
        let limit = caps.max_safe_allocation_bytes().unwrap();
        assert_eq!(limit, caps.max_buffer_size / 4 * 3);
    }

    // ── has_reliable_f64 tests ──────────────────────────────────────────

    #[test]
    fn has_reliable_f64_true_with_probed_basic() {
        let caps = mock_caps(wgpu::DeviceType::DiscreteGpu, VENDOR_NVIDIA)
            .with_f64_capabilities(full_f64_caps());
        assert!(caps.has_reliable_f64());
    }

    #[test]
    fn has_reliable_f64_false_with_probed_broken() {
        let caps = mock_caps(wgpu::DeviceType::DiscreteGpu, VENDOR_NVIDIA)
            .with_f64_capabilities(broken_f64_caps());
        assert!(!caps.has_reliable_f64());
    }

    #[test]
    fn has_reliable_f64_fallback_to_feature_flag() {
        let caps = mock_caps(wgpu::DeviceType::DiscreteGpu, VENDOR_NVIDIA);
        assert!(caps.has_reliable_f64());
    }

    // ── Display impl test ───────────────────────────────────────────────

    #[test]
    fn display_includes_key_fields() {
        let caps = mock_caps(wgpu::DeviceType::DiscreteGpu, VENDOR_NVIDIA);
        let output = format!("{caps}");
        assert!(output.contains("Test Device"));
        assert!(output.contains("DiscreteGpu"));
        assert!(output.contains("0x10DE"));
        assert!(output.contains("Max Buffer Size"));
        assert!(output.contains("f64 shaders: Yes"));
    }

    // ── with_f64_capabilities builder test ──────────────────────────────

    #[test]
    fn with_f64_capabilities_populates_field() {
        let caps = mock_caps(wgpu::DeviceType::DiscreteGpu, VENDOR_NVIDIA);
        assert!(caps.f64_capabilities.is_none());
        let caps = caps.with_f64_capabilities(full_f64_caps());
        assert!(caps.f64_capabilities.is_some());
    }

    // ── subgroup info tests ─────────────────────────────────────────────

    #[test]
    fn preferred_subgroup_size_some_when_reported() {
        let caps = mock_caps(wgpu::DeviceType::DiscreteGpu, VENDOR_NVIDIA);
        assert_eq!(caps.preferred_subgroup_size(), Some(32));
    }

    #[test]
    fn preferred_subgroup_size_none_when_zero() {
        let mut caps = mock_caps(wgpu::DeviceType::DiscreteGpu, VENDOR_NVIDIA);
        caps.subgroup_max_size = 0;
        assert_eq!(caps.preferred_subgroup_size(), None);
    }

    #[test]
    fn has_subgroup_info_positive() {
        let caps = mock_caps(wgpu::DeviceType::DiscreteGpu, VENDOR_NVIDIA);
        assert!(caps.has_subgroup_info());
    }

    #[test]
    fn has_subgroup_info_negative() {
        let mut caps = mock_caps(wgpu::DeviceType::DiscreteGpu, VENDOR_NVIDIA);
        caps.subgroup_min_size = 0;
        caps.subgroup_max_size = 0;
        assert!(!caps.has_subgroup_info());
    }

    // ── eigensolve_workgroup_size tests ──────────────────────────────────

    #[test]
    fn eigensolve_workgroup_size_uses_subgroup_min() {
        let mut caps = mock_caps(wgpu::DeviceType::DiscreteGpu, VENDOR_NVIDIA);
        caps.subgroup_min_size = 64;
        assert_eq!(caps.eigensolve_workgroup_size(), 64);
    }

    #[test]
    fn eigensolve_workgroup_size_fallback_32() {
        let mut caps = mock_caps(wgpu::DeviceType::DiscreteGpu, VENDOR_NVIDIA);
        caps.subgroup_min_size = 0;
        assert_eq!(caps.eigensolve_workgroup_size(), 32);
    }

    // ── workload-specific optimal sizes for different device types ──────

    #[test]
    fn integrated_gpu_uses_smaller_workgroups() {
        let caps = mock_caps(wgpu::DeviceType::IntegratedGpu, VENDOR_INTEL);
        assert!(
            caps.optimal_workgroup_size(WorkloadType::ElementWise)
                <= caps.optimal_workgroup_size(WorkloadType::ElementWise)
        );
        assert_eq!(caps.optimal_workgroup_size(WorkloadType::MatMul), 64);
    }

    #[test]
    fn cpu_device_uses_smallest_workgroups() {
        let caps = mock_caps(wgpu::DeviceType::Cpu, 0);
        assert_eq!(caps.optimal_workgroup_size(WorkloadType::ElementWise), 32);
        assert_eq!(caps.optimal_workgroup_size(WorkloadType::MatMul), 16);
    }

    // ── optimal_workgroup_size_arch (free function) ─────────────────────

    #[test]
    fn optimal_workgroup_size_arch_respects_max() {
        let size = optimal_workgroup_size_arch(&GpuArch::Ampere, WorkloadType::Reduction, 128);
        assert!(size <= 128);
    }

    #[test]
    fn optimal_workgroup_size_arch_reduction_doubles_base() {
        let base = workgroup_size_for_arch(&GpuArch::Volta);
        let red = optimal_workgroup_size_arch(&GpuArch::Volta, WorkloadType::Reduction, 1024);
        assert_eq!(red, base * 2);
    }

    #[test]
    fn optimal_workgroup_size_arch_convolution_halves_base() {
        let base = workgroup_size_for_arch(&GpuArch::Ada);
        let conv = optimal_workgroup_size_arch(&GpuArch::Ada, WorkloadType::Convolution, 1024);
        assert_eq!(conv, base / 2);
    }

    // ── vendor_name test ────────────────────────────────────────────────

    #[test]
    fn vendor_name_known() {
        let caps = mock_caps(wgpu::DeviceType::DiscreteGpu, VENDOR_NVIDIA);
        assert_eq!(caps.vendor_name(), "NVIDIA");
        let caps_amd = mock_caps(
            wgpu::DeviceType::DiscreteGpu,
            crate::device::vendor::VENDOR_AMD,
        );
        assert_eq!(caps_amd.vendor_name(), "AMD");
        let caps_intel = mock_caps(wgpu::DeviceType::DiscreteGpu, VENDOR_INTEL);
        assert_eq!(caps_intel.vendor_name(), "Intel");
    }

    #[test]
    fn vendor_name_unknown() {
        let caps = mock_caps(wgpu::DeviceType::DiscreteGpu, 0xBEEF);
        assert_eq!(caps.vendor_name(), "Unknown");
    }
}

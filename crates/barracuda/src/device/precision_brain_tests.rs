// SPDX-License-Identifier: AGPL-3.0-or-later
#![expect(
    clippy::unwrap_used,
    reason = "test assertions: unwrap is idiomatic for test code"
)]

use super::*;
use crate::device::capabilities::DeviceCapabilities;
use crate::device::hardware_calibration::TierCapability;
use crate::device::vendor::VENDOR_NVIDIA;

#[expect(
    clippy::fn_params_excessive_bools,
    reason = "test helper — bool flags are clear and concise here"
)]
fn make_cal(f32_ok: bool, df64_ok: bool, f64_ok: bool, precise_ok: bool) -> HardwareCalibration {
    let mk = |tier, ok: bool| TierCapability {
        tier,
        compiles: ok,
        dispatches: ok,
        transcendentals_safe: ok,
    };
    let universal = |tier| mk(tier, true);
    HardwareCalibration {
        adapter_name: "Test GPU".into(),
        tiers: vec![
            universal(PrecisionTier::Binary),
            universal(PrecisionTier::Int2),
            universal(PrecisionTier::Quantized4),
            universal(PrecisionTier::Quantized8),
            universal(PrecisionTier::Fp8E5M2),
            universal(PrecisionTier::Fp8E4M3),
            universal(PrecisionTier::Bf16),
            mk(PrecisionTier::F16, false),
            mk(PrecisionTier::Tf32, false),
            mk(PrecisionTier::F32, f32_ok),
            mk(PrecisionTier::DF64, df64_ok),
            mk(PrecisionTier::F64, f64_ok),
            mk(PrecisionTier::F64Precise, precise_ok),
            universal(PrecisionTier::QF128),
            mk(PrecisionTier::DF128, f64_ok),
        ],
        has_any_f64: f64_ok || precise_ok,
        df64_safe: df64_ok,
        nvvm_transcendental_risk: false,
    }
}

#[test]
fn full_hw_routes_dielectric_to_precise() {
    let cal = make_cal(true, true, true, true);
    let tier = route_domain(PhysicsDomain::Dielectric, &cal, true, false);
    assert_eq!(tier, PrecisionTier::F64Precise);
}

#[test]
fn no_precise_routes_dielectric_to_f64() {
    let cal = make_cal(true, true, true, false);
    let tier = route_domain(PhysicsDomain::Dielectric, &cal, true, false);
    assert_eq!(tier, PrecisionTier::F64);
}

#[test]
fn no_f64_routes_to_df64() {
    let cal = make_cal(true, true, false, false);
    let tier = route_domain(PhysicsDomain::Dielectric, &cal, false, false);
    assert_eq!(tier, PrecisionTier::DF64);
}

#[test]
fn nothing_works_falls_to_f32() {
    let cal = make_cal(true, false, false, false);
    let tier = route_domain(PhysicsDomain::LatticeQcd, &cal, false, false);
    assert_eq!(tier, PrecisionTier::F32);
}

#[test]
fn throughput_domain_prefers_f64() {
    let cal = make_cal(true, true, true, true);
    let tier = route_domain(PhysicsDomain::MolecularDynamics, &cal, true, false);
    assert_eq!(tier, PrecisionTier::F64);
}

#[test]
fn population_pk_routes_moderate() {
    let cal = make_cal(true, true, true, false);
    let tier = route_domain(PhysicsDomain::PopulationPk, &cal, true, false);
    assert_eq!(tier, PrecisionTier::F64);
}

#[test]
fn hydrology_fallback_to_df64() {
    let cal = make_cal(true, true, false, false);
    let tier = route_domain(PhysicsDomain::Hydrology, &cal, false, false);
    assert_eq!(tier, PrecisionTier::DF64);
}

#[test]
fn domain_index_roundtrip() {
    for (i, &domain) in ALL_DOMAINS.iter().enumerate() {
        assert_eq!(domain_index(domain), i);
    }
}

#[test]
fn advice_fma_sensitive() {
    let (fma_safe, _) = domain_requirements(PhysicsDomain::Dielectric, PrecisionTier::F64Precise);
    assert!(!fma_safe);
}

#[test]
fn advice_throughput_bound() {
    let (fma_safe, _) = domain_requirements(PhysicsDomain::LatticeQcd, PrecisionTier::DF64);
    assert!(fma_safe);
}

fn test_caps_volta_full() -> DeviceCapabilities {
    DeviceCapabilities {
        device_name: "Test GPU".into(),
        device_type: wgpu::DeviceType::DiscreteGpu,
        max_buffer_size: 1024 * 1024 * 1024,
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
        has_subgroups: false,
        f64_shaders: true,
        f64_shared_memory: false,
        f64_capabilities: None,
    }
}

#[test]
fn domain_requirements_moderate_domains_native_f64() {
    let (fma, rationale) = domain_requirements(PhysicsDomain::GradientFlow, PrecisionTier::F64);
    assert!(fma);
    assert!(rationale.contains("moderate"));
}

#[test]
fn route_advice_dielectric_includes_fma_flag() {
    let cal = make_cal(true, true, true, true);
    let brain = PrecisionBrain::from_capabilities(cal, &test_caps_volta_full());
    let adv = brain.route_advice(PhysicsDomain::Dielectric);
    assert_eq!(adv.tier, PrecisionTier::F64Precise);
    assert!(!adv.fma_safe);
    assert!(!adv.rationale.is_empty());
}

#[test]
fn precision_brain_display_covers_adapter_name() {
    let cal = make_cal(true, true, true, true);
    let brain = PrecisionBrain::from_capabilities(cal, &test_caps_volta_full());
    let s = brain.to_string();
    assert!(s.contains("Test GPU"));
    assert!(s.contains("LatticeQcd"));
}

#[test]
fn coral_aware_routes_f64_when_hw_broken_but_coral_available() {
    let cal = make_cal(true, false, false, false);
    let brain = PrecisionBrain::from_capabilities_with_coral(cal, &test_caps_volta_full(), true);
    let tier = brain.route(PhysicsDomain::LatticeQcd);
    assert_eq!(
        tier,
        PrecisionTier::F64,
        "with coral lowering, F64 should be routed even without hw native"
    );
}

#[test]
fn coral_aware_routes_precise_for_dielectric() {
    let cal = make_cal(true, true, true, true);
    let brain = PrecisionBrain::from_capabilities_with_coral(cal, &test_caps_volta_full(), true);
    let tier = brain.route(PhysicsDomain::Dielectric);
    assert_eq!(tier, PrecisionTier::F64Precise);
}

fn test_caps_no_f64() -> DeviceCapabilities {
    DeviceCapabilities {
        f64_shaders: false,
        ..test_caps_volta_full()
    }
}

#[test]
fn needs_sovereign_compile_true_when_coral_and_no_hw() {
    let cal = make_cal(true, false, false, false);
    let brain = PrecisionBrain::from_capabilities_with_coral(cal, &test_caps_no_f64(), true);
    assert!(brain.needs_sovereign_compile(PhysicsDomain::LatticeQcd));
    assert!(brain.has_coral_f64_lowering());
}

#[test]
fn needs_sovereign_compile_false_when_hw_native() {
    let cal = make_cal(true, true, true, true);
    let brain = PrecisionBrain::from_capabilities_with_coral(cal, &test_caps_volta_full(), true);
    assert!(
        !brain.needs_sovereign_compile(PhysicsDomain::LatticeQcd),
        "hw native means wgpu path works, no sovereign needed"
    );
}

#[test]
fn needs_sovereign_compile_false_without_coral() {
    let cal = make_cal(true, false, false, false);
    let brain = PrecisionBrain::from_capabilities_with_coral(cal, &test_caps_volta_full(), false);
    assert!(!brain.needs_sovereign_compile(PhysicsDomain::LatticeQcd));
    assert!(!brain.has_coral_f64_lowering());
}

#[test]
fn has_native_f64_true_when_profile_reports_native_paths() {
    let cal = make_cal(true, true, true, true);
    let brain = PrecisionBrain::from_capabilities(cal, &test_caps_volta_full());
    assert!(brain.has_native_f64());
}

#[test]
fn adapter_name_accessor() {
    let cal = make_cal(true, true, true, true);
    let brain = PrecisionBrain::from_capabilities(cal, &test_caps_volta_full());
    assert_eq!(brain.adapter_name(), "Test GPU");
}

#[test]
fn inference_domain_routes_to_q4() {
    let cal = make_cal(true, true, true, true);
    let tier = route_domain(PhysicsDomain::Inference, &cal, true, false);
    assert_eq!(tier, PrecisionTier::Quantized4);
}

#[test]
fn training_domain_routes_to_bf16() {
    let cal = make_cal(true, true, true, true);
    let tier = route_domain(PhysicsDomain::Training, &cal, true, false);
    assert_eq!(tier, PrecisionTier::Bf16);
}

#[test]
fn hashing_domain_routes_to_binary() {
    let cal = make_cal(true, true, true, true);
    let tier = route_domain(PhysicsDomain::Hashing, &cal, true, false);
    assert_eq!(tier, PrecisionTier::Binary);
}

#[test]
fn domain_index_roundtrip_all_15() {
    for (i, &domain) in ALL_DOMAINS.iter().enumerate() {
        assert_eq!(
            domain_index(domain),
            i,
            "domain_index mismatch for {domain:?}"
        );
    }
}

#[test]
fn inference_requirements_throughput() {
    let (fma_safe, _) = domain_requirements(PhysicsDomain::Inference, PrecisionTier::Quantized4);
    assert!(fma_safe);
}

#[test]
fn hashing_requirements_binary() {
    let (fma_safe, rationale) = domain_requirements(PhysicsDomain::Hashing, PrecisionTier::Binary);
    assert!(fma_safe);
    assert!(rationale.contains("XNOR") || rationale.contains("maximum"));
}

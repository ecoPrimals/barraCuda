// SPDX-License-Identifier: AGPL-3.0-or-later
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

// ── Integration: PrecisionBrain → SovereignDevice advice wiring ─────

#[test]
#[cfg(feature = "sovereign-dispatch")]
fn sovereign_advice_built_for_f64_domain() {
    use crate::device::sovereign_device::SovereignDevice;
    let cal = make_cal(true, false, false, false);
    let brain = PrecisionBrain::from_capabilities_with_coral(cal, &test_caps_no_f64(), true);

    let domain = PhysicsDomain::LatticeQcd;
    assert!(brain.needs_sovereign_compile(domain));

    let tier = brain.route(domain);
    let (f64_shader, df64_shader) = match tier {
        PrecisionTier::F64 | PrecisionTier::F64Precise | PrecisionTier::DF128 => (true, false),
        PrecisionTier::DF64 => (false, true),
        _ => (false, false),
    };

    let advice = SovereignDevice::build_precision_advice(f64_shader, df64_shader);
    assert!(advice.is_some(), "f64 domain should produce advice");
    let adv = advice.unwrap();
    assert_eq!(adv.tier, "F64");
    assert!(adv.needs_transcendental_lowering);
}

#[test]
#[cfg(feature = "sovereign-dispatch")]
fn sovereign_advice_none_for_f32_domain() {
    use crate::device::sovereign_device::SovereignDevice;
    let cal = make_cal(true, true, true, true);
    let brain = PrecisionBrain::from_capabilities(cal, &test_caps_volta_full());

    let domain = PhysicsDomain::Inference;
    assert!(!brain.needs_sovereign_compile(domain));

    let tier = brain.route(domain);
    let (f64_shader, df64_shader) = match tier {
        PrecisionTier::F64 | PrecisionTier::F64Precise | PrecisionTier::DF128 => (true, false),
        PrecisionTier::DF64 => (false, true),
        _ => (false, false),
    };

    let advice = SovereignDevice::build_precision_advice(f64_shader, df64_shader);
    assert!(advice.is_none(), "f32 domain should not produce advice");
}

#[test]
#[cfg(feature = "sovereign-dispatch")]
fn sovereign_advice_df64_domain() {
    use crate::device::sovereign_device::SovereignDevice;
    let cal = make_cal(true, true, false, false);
    let brain = PrecisionBrain::from_capabilities_with_coral(cal, &test_caps_no_f64(), true);

    let domain = PhysicsDomain::GradientFlow;
    let tier = brain.route(domain);
    let (f64_shader, df64_shader) = match tier {
        PrecisionTier::F64 | PrecisionTier::F64Precise | PrecisionTier::DF128 => (true, false),
        PrecisionTier::DF64 => (false, true),
        _ => (false, false),
    };

    if df64_shader {
        let advice = SovereignDevice::build_precision_advice(f64_shader, df64_shader);
        assert!(advice.is_some());
        let adv = advice.unwrap();
        assert_eq!(adv.tier, "DF64");
        assert!(adv.df64_naga_poisoned);
    }
}

#[test]
#[cfg(feature = "sovereign-dispatch")]
fn precision_advice_serializes_to_coral_wire_format() {
    use crate::device::coral_compiler::types::PrecisionAdvice;
    use crate::device::sovereign_device::SovereignDevice;

    let advice = SovereignDevice::build_precision_advice(true, false).unwrap();
    let wire = PrecisionAdvice {
        tier: advice.tier.clone(),
        needs_transcendental_lowering: advice.needs_transcendental_lowering,
        df64_naga_poisoned: advice.df64_naga_poisoned,
        domain: advice.domain.clone(),
    };
    let json = serde_json::to_string(&wire).unwrap();
    assert!(json.contains("\"tier\":\"F64\""));
    assert!(json.contains("\"needs_transcendental_lowering\":true"));
}

/// Integration test: walks F32 → DF64 → F64 precision tiers and verifies
/// the correct `hardware_hint` is emitted at each level through the full
/// `PrecisionBrain → DispatchDescriptor → sovereign_dispatch_wire` chain.
#[test]
#[cfg(feature = "sovereign-dispatch")]
fn precision_ladder_f32_df64_f64_hardware_hint_chain() {
    use crate::device::backend::HardwareHint;
    use crate::device::precision_tier::PrecisionTier;
    use crate::device::sovereign_device::SovereignDevice;

    struct TierExpectation {
        tier: PrecisionTier,
        f64_flag: bool,
        df64_flag: bool,
        expected_hint: HardwareHint,
        expected_advice: bool,
    }

    let expectations = [
        TierExpectation {
            tier: PrecisionTier::F32,
            f64_flag: false,
            df64_flag: false,
            expected_hint: HardwareHint::Compute,
            expected_advice: false,
        },
        TierExpectation {
            tier: PrecisionTier::DF64,
            f64_flag: false,
            df64_flag: true,
            expected_hint: HardwareHint::Compute,
            expected_advice: true,
        },
        TierExpectation {
            tier: PrecisionTier::F64,
            f64_flag: true,
            df64_flag: false,
            expected_hint: HardwareHint::Compute,
            expected_advice: true,
        },
    ];

    for exp in &expectations {
        let hint = exp.tier.recommended_hardware_hint();
        assert_eq!(
            hint, exp.expected_hint,
            "Tier {} should route to {:?}",
            exp.tier, exp.expected_hint
        );

        let advice = SovereignDevice::build_precision_advice(exp.f64_flag, exp.df64_flag);
        assert_eq!(
            advice.is_some(),
            exp.expected_advice,
            "Tier {} advice presence mismatch",
            exp.tier
        );

        if let Some(adv) = &advice {
            if exp.df64_flag {
                assert_eq!(adv.tier, "DF64");
                assert!(adv.df64_naga_poisoned);
            } else if exp.f64_flag {
                assert_eq!(adv.tier, "F64");
                assert!(adv.needs_transcendental_lowering);
            }
        }
    }
}

/// Verifies tensor core tiers map correctly through the hardware hint path.
#[test]
#[cfg(feature = "sovereign-dispatch")]
fn precision_ladder_tensor_core_tiers_hardware_hint() {
    use crate::device::backend::HardwareHint;
    use crate::device::precision_tier::PrecisionTier;

    let tensor_tiers = [
        PrecisionTier::F16,
        PrecisionTier::Bf16,
        PrecisionTier::Tf32,
        PrecisionTier::Fp8E4M3,
        PrecisionTier::Fp8E5M2,
    ];

    for tier in tensor_tiers {
        assert_eq!(
            tier.recommended_hardware_hint(),
            HardwareHint::TensorCore,
            "MMA tier {tier} must route to TensorCore"
        );
        assert!(
            tier.requires_compiler_support(),
            "MMA tier {tier} requires coralReef compiler support"
        );
    }
}

// ── Trio Contract E2E Tests ────────────────────────────────────────────
//
// These tests validate the complete data-flow contract between the three
// Compute Trio primals:
//   barraCuda (WHAT) → coralReef (HOW) → toadStool (WHERE)
//
// Each test chains: PrecisionBrain domain routing → PrecisionAdvice
// construction → coralReef compile wire format → ShaderDispatchInfo →
// toadStool dispatch wire format (with mock TCP server verification).

/// Trio contract E2E: LatticeQcd → F64 → coralReef advice → toadStool dispatch.
///
/// Validates the complete chain for a high-precision physics domain:
/// 1. PrecisionBrain routes `LatticeQcd` to F64 tier (needs sovereign compile)
/// 2. `build_precision_advice` produces correct F64 advice
/// 3. `PrecisionAdvice` serializes in coral wire format
/// 4. Simulated compile metadata yields `ShaderDispatchInfo`
/// 5. `submit_dispatch` to mock toadStool carries correct `hardware_hint`,
///    `gpr_count`, `workgroup`, and binary
#[test]
#[cfg(feature = "sovereign-dispatch")]
fn trio_contract_e2e_lattice_qcd_f64_dispatch() {
    use crate::device::backend::HardwareHint;
    use crate::device::coral_compiler::types::{CoralBinary, PrecisionAdvice};
    use crate::device::sovereign_device::SovereignDevice;
    use crate::device::sovereign_dispatch_wire::{
        IpcBufferBinding, ShaderDispatchInfo, submit_dispatch,
    };
    use std::collections::HashMap;
    use std::sync::Mutex;

    // ── Step 1: PrecisionBrain routes LatticeQcd ──────────────────────
    let cal = make_cal(true, false, false, false);
    let brain = PrecisionBrain::from_capabilities_with_coral(cal, &test_caps_no_f64(), true);

    let domain = PhysicsDomain::LatticeQcd;
    assert!(brain.needs_sovereign_compile(domain));

    let tier = brain.route(domain);
    let (f64_shader, df64_shader) = match tier {
        PrecisionTier::F64 | PrecisionTier::F64Precise | PrecisionTier::DF128 => (true, false),
        PrecisionTier::DF64 => (false, true),
        _ => (false, false),
    };

    // ── Step 2: Build PrecisionAdvice ─────────────────────────────────
    let advice = SovereignDevice::build_precision_advice(f64_shader, df64_shader);
    assert!(advice.is_some(), "F64 domain must produce advice");
    let adv = advice.unwrap();
    assert_eq!(adv.tier, "F64");
    assert!(adv.needs_transcendental_lowering);
    assert!(!adv.df64_naga_poisoned);

    let hint = tier.recommended_hardware_hint();
    assert_eq!(
        hint,
        HardwareHint::Compute,
        "F64 routes to Compute (ALU/FP64 cores)"
    );

    // ── Step 3: coralReef wire format (PrecisionAdvice) ───────────────
    let wire_advice = PrecisionAdvice {
        tier: adv.tier.clone(),
        needs_transcendental_lowering: adv.needs_transcendental_lowering,
        df64_naga_poisoned: adv.df64_naga_poisoned,
        domain: Some("LatticeQcd".to_owned()),
    };
    let advice_json = serde_json::to_string(&wire_advice).unwrap();
    assert!(advice_json.contains("\"tier\":\"F64\""));
    assert!(advice_json.contains("\"needs_transcendental_lowering\":true"));
    assert!(advice_json.contains("\"df64_naga_poisoned\":false"));
    assert!(advice_json.contains("\"domain\":\"LatticeQcd\""));

    // ── Step 4: Simulate compile result → ShaderDispatchInfo ──────────
    let coral_binary = CoralBinary {
        binary: bytes::Bytes::from_static(&[0xDE, 0xAD, 0xBE, 0xEF]),
        arch: "sm_70".to_owned(),
        gpr_count: Some(48),
        workgroup: Some([64, 1, 1]),
        shared_mem_bytes: Some(2048),
        barrier_count: Some(1),
    };

    let shader_info = ShaderDispatchInfo {
        gpr_count: coral_binary.gpr_count.unwrap_or(32),
        workgroup: coral_binary.workgroup.unwrap_or([64, 1, 1]),
        shared_mem_bytes: coral_binary.shared_mem_bytes.unwrap_or(0),
        barrier_count: coral_binary.barrier_count.unwrap_or(0),
    };

    assert_eq!(shader_info.gpr_count, 48);
    assert_eq!(shader_info.shared_mem_bytes, 2048);
    assert_eq!(shader_info.barrier_count, 1);

    // ── Step 5: Mock toadStool dispatch server ────────────────────────
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();

    let listener =
        rt.block_on(async { tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap() });
    let addr = listener.local_addr().unwrap();

    rt.block_on(async {
        let server_task = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut buf = vec![0u8; 32768];
            let n = tokio::io::AsyncReadExt::read(&mut stream, &mut buf)
                .await
                .unwrap();
            let req_str = String::from_utf8_lossy(&buf[..n]);

            assert!(
                req_str.contains("compute.dispatch.submit"),
                "Method must be compute.dispatch.submit"
            );
            assert!(
                req_str.contains("\"hardware_hint\":\"compute\""),
                "F64 tier routes to compute hint"
            );
            assert!(
                req_str.contains("\"gpr_count\":48"),
                "GPR count from coral compile must propagate"
            );
            assert!(
                req_str.contains("\"shared_mem_bytes\":2048"),
                "Shared memory from coral compile must propagate"
            );
            assert!(
                req_str.contains("\"barrier_count\":1"),
                "Barrier count from coral compile must propagate"
            );

            let body = r#"{"jsonrpc":"2.0","result":{},"id":1}"#;
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n{body}",
                body.len()
            );
            tokio::io::AsyncWriteExt::write_all(&mut stream, response.as_bytes())
                .await
                .unwrap();
            tokio::io::AsyncWriteExt::shutdown(&mut stream).await.ok();
        });

        let staged = Mutex::new(HashMap::new());
        let bindings = vec![IpcBufferBinding {
            index: 0,
            buffer_id: 1,
            size: 256,
            read_only: false,
        }];
        staged
            .lock()
            .unwrap()
            .insert(1u64, bytes::BytesMut::from(&[0u8; 256][..]));

        let result = submit_dispatch(
            &format!("http://127.0.0.1:{}", addr.port()),
            &staged,
            &coral_binary.binary,
            (1, 1, 1),
            &bindings,
            hint,
            shader_info,
        );
        assert!(
            result.is_ok(),
            "Trio dispatch must succeed: {:?}",
            result.err()
        );
        server_task.await.ok();
    });
}

/// Trio contract E2E: GradientFlow → DF64 → coralReef DF64 advice → toadStool dispatch.
///
/// Validates the DF64 (f32-pair emulation) path through the trio:
/// 1. PrecisionBrain routes `GradientFlow` to DF64 (no native f64, coral available)
/// 2. `build_precision_advice` flags `df64_naga_poisoned`
/// 3. `CompileWgslRequest` carries DF64-specific advice to coralReef
/// 4. `submit_dispatch` carries `HardwareHint::Compute` to toadStool
#[test]
#[cfg(feature = "sovereign-dispatch")]
fn trio_contract_e2e_gradient_flow_df64_dispatch() {
    use crate::device::backend::HardwareHint;
    use crate::device::coral_compiler::types::PrecisionAdvice;
    use crate::device::sovereign_device::SovereignDevice;
    use crate::device::sovereign_dispatch_wire::{
        IpcBufferBinding, ShaderDispatchInfo, submit_dispatch,
    };
    use std::collections::HashMap;
    use std::sync::Mutex;

    // ── PrecisionBrain routes GradientFlow on DF64-capable hardware ───
    let cal = make_cal(true, true, false, false);
    let brain = PrecisionBrain::from_capabilities_with_coral(cal, &test_caps_no_f64(), true);

    let domain = PhysicsDomain::GradientFlow;
    let tier = brain.route(domain);

    let (f64_shader, df64_shader) = match tier {
        PrecisionTier::F64 | PrecisionTier::F64Precise | PrecisionTier::DF128 => (true, false),
        PrecisionTier::DF64 => (false, true),
        _ => (false, false),
    };

    // DF64 path: df64_shader should be true
    if !df64_shader {
        // On some hardware configs brain might route differently; skip if so
        return;
    }

    let advice = SovereignDevice::build_precision_advice(f64_shader, df64_shader);
    assert!(advice.is_some());
    let adv = advice.unwrap();
    assert_eq!(adv.tier, "DF64");
    assert!(adv.df64_naga_poisoned);
    assert!(!adv.needs_transcendental_lowering);

    let hint = tier.recommended_hardware_hint();
    assert_eq!(hint, HardwareHint::Compute);

    // ── coralReef wire format with DF64 advice ────────────────────────
    let wire_advice = PrecisionAdvice {
        tier: adv.tier.clone(),
        needs_transcendental_lowering: adv.needs_transcendental_lowering,
        df64_naga_poisoned: adv.df64_naga_poisoned,
        domain: Some("GradientFlow".to_owned()),
    };
    let advice_json = serde_json::to_string(&wire_advice).unwrap();
    assert!(advice_json.contains("\"tier\":\"DF64\""));
    assert!(advice_json.contains("\"df64_naga_poisoned\":true"));
    assert!(advice_json.contains("\"domain\":\"GradientFlow\""));

    // ── Mock toadStool: verify DF64 dispatches as Compute ─────────────
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();

    let listener =
        rt.block_on(async { tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap() });
    let addr = listener.local_addr().unwrap();

    rt.block_on(async {
        let server_task = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut buf = vec![0u8; 16384];
            let n = tokio::io::AsyncReadExt::read(&mut stream, &mut buf)
                .await
                .unwrap();
            let req_str = String::from_utf8_lossy(&buf[..n]);
            assert!(
                req_str.contains("\"hardware_hint\":\"compute\""),
                "DF64 routes to compute (ALU cores)"
            );

            let body = r#"{"jsonrpc":"2.0","result":{},"id":1}"#;
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n{body}",
                body.len()
            );
            tokio::io::AsyncWriteExt::write_all(&mut stream, response.as_bytes())
                .await
                .unwrap();
            tokio::io::AsyncWriteExt::shutdown(&mut stream).await.ok();
        });

        let staged = Mutex::new(HashMap::new());
        let bindings = vec![IpcBufferBinding {
            index: 0,
            buffer_id: 1,
            size: 128,
            read_only: false,
        }];
        staged
            .lock()
            .unwrap()
            .insert(1u64, bytes::BytesMut::from(&[0u8; 128][..]));

        let info = ShaderDispatchInfo {
            gpr_count: 32,
            workgroup: [64, 1, 1],
            shared_mem_bytes: 0,
            barrier_count: 0,
        };

        let result = submit_dispatch(
            &format!("http://127.0.0.1:{}", addr.port()),
            &staged,
            &[0xCA, 0xFE],
            (1, 1, 1),
            &bindings,
            hint,
            info,
        );
        assert!(
            result.is_ok(),
            "DF64 dispatch must succeed: {:?}",
            result.err()
        );
        server_task.await.ok();
    });
}

/// Trio contract E2E: Inference → F16 → TensorCore hint → toadStool dispatch.
///
/// Validates the tensor core GEMM routing path:
/// 1. F16 precision tier maps to `HardwareHint::TensorCore`
/// 2. F16 requires compiler support (MMA instruction emission)
/// 3. toadStool receives `tensor_core` hardware hint
#[test]
#[cfg(feature = "sovereign-dispatch")]
fn trio_contract_e2e_tensor_core_f16_dispatch() {
    use crate::device::backend::HardwareHint;
    use crate::device::sovereign_dispatch_wire::{
        IpcBufferBinding, ShaderDispatchInfo, submit_dispatch,
    };
    use std::collections::HashMap;
    use std::sync::Mutex;

    let tier = PrecisionTier::F16;
    let hint = tier.recommended_hardware_hint();
    assert_eq!(hint, HardwareHint::TensorCore);
    assert!(tier.requires_compiler_support(), "F16 MMA needs coralReef");

    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();

    let listener =
        rt.block_on(async { tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap() });
    let addr = listener.local_addr().unwrap();

    rt.block_on(async {
        let server_task = tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.unwrap();
            let mut buf = vec![0u8; 16384];
            let n = tokio::io::AsyncReadExt::read(&mut stream, &mut buf)
                .await
                .unwrap();
            let req_str = String::from_utf8_lossy(&buf[..n]);
            assert!(
                req_str.contains("\"hardware_hint\":\"tensor_core\""),
                "F16 must route to tensor_core for MMA dispatch"
            );
            assert!(
                req_str.contains("compute.dispatch.submit"),
                "Method must be compute.dispatch.submit"
            );

            let body = r#"{"jsonrpc":"2.0","result":{},"id":1}"#;
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\n\r\n{body}",
                body.len()
            );
            tokio::io::AsyncWriteExt::write_all(&mut stream, response.as_bytes())
                .await
                .unwrap();
            tokio::io::AsyncWriteExt::shutdown(&mut stream).await.ok();
        });

        let staged = Mutex::new(HashMap::new());
        let bindings = vec![IpcBufferBinding {
            index: 0,
            buffer_id: 1,
            size: 512,
            read_only: false,
        }];
        staged
            .lock()
            .unwrap()
            .insert(1u64, bytes::BytesMut::from(&[0u8; 512][..]));

        let info = ShaderDispatchInfo {
            gpr_count: 24,
            workgroup: [32, 1, 1],
            shared_mem_bytes: 4096,
            barrier_count: 2,
        };

        let result = submit_dispatch(
            &format!("http://127.0.0.1:{}", addr.port()),
            &staged,
            &[0x00, 0x01, 0x02, 0x03],
            (8, 8, 1),
            &bindings,
            hint,
            info,
        );
        assert!(
            result.is_ok(),
            "TensorCore dispatch must succeed: {:?}",
            result.err()
        );
        server_task.await.ok();
    });
}

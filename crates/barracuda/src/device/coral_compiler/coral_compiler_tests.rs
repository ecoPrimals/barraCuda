// SPDX-License-Identifier: AGPL-3.0-or-later
use super::*;

#[test]
fn test_coral_compiler_new() {
    let cc = CoralCompiler::new();
    assert!(format!("{cc:?}").contains("CoralCompiler"));
}

#[test]
fn test_coral_compiler_default() {
    let cc = CoralCompiler::default();
    assert!(format!("{cc:?}").contains("CoralCompiler"));
}

#[test]
fn test_coral_binary_debug() {
    let binary = CoralBinary {
        binary: bytes::Bytes::from_static(&[0xDE, 0xAD]),
        arch: "sm_70".to_owned(),
    };
    assert!(format!("{binary:?}").contains("sm_70"));
}

#[test]
fn test_wgsl_to_spirv_valid() {
    let wgsl = r"
        @compute @workgroup_size(64)
        fn main() {}
    ";
    let words = wgsl_to_spirv(wgsl);
    assert!(words.is_some(), "valid WGSL should produce SPIR-V");
    let words = words.unwrap();
    assert!(words.len() > 10, "SPIR-V should have non-trivial length");
    assert_eq!(words[0], 0x0723_0203, "SPIR-V magic number");
}

#[test]
fn test_wgsl_to_spirv_invalid() {
    let invalid = "fn main() { let x = ; }";
    assert!(wgsl_to_spirv(invalid).is_none());
}

#[tokio::test]
async fn test_discovery_graceful_without_shader_compiler() {
    let addr = discover_shader_compiler().await;
    if let Some(ref a) = addr {
        assert!(!a.is_empty(), "discovered address must be non-empty");
    }
}

#[tokio::test]
async fn test_compile_wgsl_graceful_without_coralreef() {
    let cc = CoralCompiler::new();
    let result = cc
        .compile_wgsl("@compute @workgroup_size(64) fn main() {}", "sm_70", true)
        .await;
    if let Some(ref bin) = result {
        assert!(!bin.binary.is_empty(), "compiled binary must be non-empty");
        assert_eq!(bin.arch, "sm_70");
    }
}

#[tokio::test]
async fn test_compile_wgsl_direct_graceful_without_coralreef() {
    let cc = CoralCompiler::new();
    let result = cc
        .compile_wgsl_direct("@compute @workgroup_size(64) fn main() {}", "sm_70", false)
        .await;
    if let Some(ref bin) = result {
        assert!(!bin.binary.is_empty(), "compiled binary must be non-empty");
        assert_eq!(bin.arch, "sm_70");
    }
}

#[tokio::test]
async fn test_supported_archs_graceful_without_coralreef() {
    let cc = CoralCompiler::new();
    let archs = cc.supported_archs().await;
    if let Some(ref list) = archs {
        assert!(
            !list.is_empty(),
            "arch list must be non-empty when available"
        );
        for arch in list {
            assert!(!arch.is_empty(), "each arch string must be non-empty");
        }
    }
}

#[tokio::test]
async fn test_reset_allows_rediscovery() {
    let cc = CoralCompiler::new();
    let health = cc.health().await;
    if let Some(ref h) = health {
        assert!(
            h.name == "coralReef" || h.name == "coralreef-core",
            "unexpected primal name: {}",
            h.name
        );
        assert!(!h.version.is_empty());
    }
    cc.reset().await;
    let state = cc.state.read().await;
    assert!(matches!(&*state, ConnectionState::Uninit));
}

#[tokio::test]
async fn test_capabilities_graceful_without_coralreef() {
    let cc = CoralCompiler::new();
    let caps = cc.capabilities().await;
    if let Some(ref list) = caps {
        assert!(!list.is_empty(), "capabilities list must be non-empty");
    }
}

#[tokio::test]
async fn test_connection_state_transitions() {
    let cc = CoralCompiler::new();
    {
        let state = cc.state.read().await;
        assert!(matches!(&*state, ConnectionState::Uninit));
    }
    let _ = cc.health().await;
    {
        let state = cc.state.read().await;
        assert!(
            matches!(&*state, ConnectionState::Connected { .. })
                || matches!(&*state, ConnectionState::Unavailable)
        );
    }
}

#[test]
fn test_coral_f64_capabilities_full() {
    let caps = types::CoralF64Capabilities {
        sin: true,
        cos: true,
        sqrt: true,
        exp2: true,
        log2: true,
        rcp: true,
        exp: true,
        log: true,
        composite_lowering: true,
    };
    assert!(caps.has_full_lowering());
}

#[test]
fn test_coral_f64_capabilities_partial() {
    let mut caps = types::CoralF64Capabilities {
        sin: true,
        cos: true,
        sqrt: true,
        exp2: true,
        log2: true,
        rcp: true,
        exp: true,
        log: true,
        composite_lowering: false,
    };
    assert!(
        !caps.has_full_lowering(),
        "composite_lowering=false should fail"
    );
    caps.composite_lowering = true;
    caps.sin = false;
    assert!(!caps.has_full_lowering(), "sin=false should fail");
}

#[test]
fn test_coral_f64_capabilities_default_empty() {
    let caps = types::CoralF64Capabilities::default();
    assert!(!caps.has_full_lowering());
}

#[test]
fn test_coral_capabilities_response_json_roundtrip() {
    let resp = types::CoralCapabilitiesResponse {
        supported_archs: vec!["sm_70".to_owned(), "gfx1030".to_owned()],
        cpu_archs: vec!["x86_64".to_owned()],
        supports_cpu_execution: true,
        supports_validation: true,
        f64_transcendental_capabilities: types::CoralF64Capabilities {
            sin: true,
            cos: true,
            sqrt: true,
            exp2: true,
            log2: true,
            rcp: true,
            exp: true,
            log: true,
            composite_lowering: true,
        },
    };
    let json = serde_json::to_string(&resp).unwrap();
    let back: types::CoralCapabilitiesResponse = serde_json::from_str(&json).unwrap();
    assert_eq!(back.supported_archs.len(), 2);
    assert!(back.f64_transcendental_capabilities.has_full_lowering());
}

#[test]
fn test_precision_advice_json_roundtrip() {
    let advice = types::PrecisionAdvice {
        tier: "F64".to_owned(),
        needs_transcendental_lowering: true,
        df64_naga_poisoned: true,
        domain: Some("LatticeQcd".to_owned()),
    };
    let json = serde_json::to_string(&advice).unwrap();
    let back: types::PrecisionAdvice = serde_json::from_str(&json).unwrap();
    assert_eq!(back.tier, "F64");
    assert!(back.needs_transcendental_lowering);
    assert!(back.df64_naga_poisoned);
    assert_eq!(back.domain.as_deref(), Some("LatticeQcd"));
}

#[tokio::test]
async fn test_capabilities_structured_graceful_without_coralreef() {
    let cc = CoralCompiler::new();
    let caps = cc.capabilities_structured().await;
    if let Some(ref c) = caps {
        assert!(!c.supported_archs.is_empty());
    }
}

#[tokio::test]
async fn test_has_f64_lowering_graceful_without_coralreef() {
    let cc = CoralCompiler::new();
    let _ = cc.has_f64_lowering().await;
}

#[test]
fn test_adapter_descriptor_cache_key() {
    let desc = types::AdapterDescriptor {
        vendor_id: 0x10DE,
        device_name: "NVIDIA GeForce RTX 3090".to_owned(),
        device_type: "DiscreteGpu".to_owned(),
    };
    let key = desc.cache_key();
    assert!(key.starts_with("adapter:10de:"));
    assert!(key.contains("RTX 3090"));
}

#[test]
fn test_best_target_for_adapter_nvidia() {
    let archs = vec!["sm_70".to_owned(), "sm_80".to_owned(), "gfx1030".to_owned()];
    let nvidia_adapter = types::AdapterDescriptor {
        vendor_id: 0x10DE,
        device_name: "NVIDIA GPU".to_owned(),
        device_type: "DiscreteGpu".to_owned(),
    };
    let target = best_target_for_adapter(&archs, &nvidia_adapter);
    assert!(target.is_some());
    assert!(target.unwrap().starts_with("sm_"));
}

#[test]
fn test_best_target_for_adapter_amd() {
    let archs = vec![
        "sm_70".to_owned(),
        "gfx1030".to_owned(),
        "gfx1100".to_owned(),
    ];
    let amd_adapter = types::AdapterDescriptor {
        vendor_id: 0x1002,
        device_name: "AMD Radeon".to_owned(),
        device_type: "DiscreteGpu".to_owned(),
    };
    let target = best_target_for_adapter(&archs, &amd_adapter);
    assert!(target.is_some());
    assert!(target.unwrap().starts_with("gfx"));
}

#[test]
fn test_best_target_for_adapter_unsupported() {
    let archs = vec!["sm_70".to_owned()];
    let intel_adapter = types::AdapterDescriptor {
        vendor_id: 0x8086,
        device_name: "Intel Arc".to_owned(),
        device_type: "DiscreteGpu".to_owned(),
    };
    assert!(best_target_for_adapter(&archs, &intel_adapter).is_none());
}

// ── cache tests ─────────────────────────────────────────────────────

#[test]
fn cache_insert_and_lookup() {
    let hash = cache::shader_hash("test_shader_source_1234");
    let binary = types::CoralBinary {
        binary: bytes::Bytes::from_static(&[0xCA, 0xFE]),
        arch: "sm_70".to_owned(),
    };
    cache::cache_native_binary(&hash, "sm_70", binary);
    let found = cache::cached_native_binary(&hash, "sm_70");
    assert!(found.is_some());
    assert_eq!(found.unwrap().arch, "sm_70");
}

#[test]
fn cache_miss_returns_none() {
    assert!(cache::cached_native_binary("nonexistent_hash_xyz", "sm_70").is_none());
}

#[test]
fn cache_any_arch_finds_first_match() {
    let hash = cache::shader_hash("any_arch_test_source_5678");
    cache::cache_native_binary(
        &hash,
        "gfx1030",
        types::CoralBinary {
            binary: bytes::Bytes::from_static(&[0xAA]),
            arch: "gfx1030".to_owned(),
        },
    );
    let found = cache::cached_native_binary_any_arch(&hash);
    assert!(found.is_some());
    assert_eq!(found.unwrap().arch, "gfx1030");
}

#[test]
fn cache_any_arch_miss() {
    assert!(cache::cached_native_binary_any_arch("completely_missing_hash").is_none());
}

#[test]
fn shader_hash_deterministic() {
    let h1 = cache::shader_hash("hello world");
    let h2 = cache::shader_hash("hello world");
    assert_eq!(h1, h2);
    let h3 = cache::shader_hash("hello world!");
    assert_ne!(h1, h3);
}

#[test]
fn shader_hash_is_hex() {
    let h = cache::shader_hash("some shader code");
    assert!(h.chars().all(|c| c.is_ascii_hexdigit()));
    assert_eq!(h.len(), 64);
}

// ── types serialization tests ───────────────────────────────────────

#[test]
fn adapter_descriptor_json_roundtrip() {
    let desc = types::AdapterDescriptor {
        vendor_id: 0x10DE,
        device_name: "Test GPU".to_owned(),
        device_type: "DiscreteGpu".to_owned(),
    };
    let json = serde_json::to_string(&desc).unwrap();
    let back: types::AdapterDescriptor = serde_json::from_str(&json).unwrap();
    assert_eq!(back.vendor_id, 0x10DE);
    assert_eq!(back.device_name, "Test GPU");
    assert_eq!(back.device_type, "DiscreteGpu");
}

#[test]
fn health_response_deserialize() {
    let json = r#"{"name":"coralReef","version":"0.3.0","status":"healthy","supported_archs":["sm_70","sm_80"]}"#;
    let resp: types::HealthResponse = serde_json::from_str(json).unwrap();
    assert_eq!(resp.name, "coralReef");
    assert_eq!(resp.supported_archs.len(), 2);
}

#[test]
fn precision_to_coral_strategy_all_variants() {
    use crate::shaders::precision::Precision;
    assert_eq!(
        types::precision_to_coral_strategy(&Precision::Binary),
        "binary"
    );
    assert_eq!(types::precision_to_coral_strategy(&Precision::Int2), "int2");
    assert_eq!(
        types::precision_to_coral_strategy(&Precision::Q4),
        "q4_block"
    );
    assert_eq!(
        types::precision_to_coral_strategy(&Precision::Q8),
        "q8_block"
    );
    assert_eq!(
        types::precision_to_coral_strategy(&Precision::Fp8E5M2),
        "fp8_e5m2"
    );
    assert_eq!(
        types::precision_to_coral_strategy(&Precision::Fp8E4M3),
        "fp8_e4m3"
    );
    assert_eq!(
        types::precision_to_coral_strategy(&Precision::Bf16),
        "bf16_emulated"
    );
    assert_eq!(
        types::precision_to_coral_strategy(&Precision::F16),
        "f16_fast"
    );
    assert_eq!(
        types::precision_to_coral_strategy(&Precision::F32),
        "f32_only"
    );
    assert_eq!(
        types::precision_to_coral_strategy(&Precision::F64),
        "native"
    );
    assert_eq!(
        types::precision_to_coral_strategy(&Precision::Df64),
        "double_float"
    );
    assert_eq!(
        types::precision_to_coral_strategy(&Precision::Qf128),
        "quad_float"
    );
    assert_eq!(
        types::precision_to_coral_strategy(&Precision::Df128),
        "double_double_f64"
    );
}

// SPDX-License-Identifier: AGPL-3.0-only
//! Precision validation tests — edge cases, E2E, and fault recovery.

use super::*;

#[test]
fn test_downcast_df64_only_maps_existing_transcendentals() {
    let source = "let a = exp_f64(x); let b = tan_f64(y); let c = sqrt_f64(z);";
    let result = downcast_f64_to_df64(source);
    assert!(result.contains("exp_df64("), "exp should map");
    assert!(result.contains("sqrt_df64("), "sqrt should map");
    assert!(
        result.contains("tan_f64("),
        "tan_f64 should stay unmapped (no df64 impl)"
    );
}

#[test]
fn test_downcast_f32_mixed_u32_f64() {
    let source = "let n: u32 = 100u;\nlet x: f64 = f64(1.5);\nlet y: f64 = input[n];";
    let result = downcast_f64_to_f32(source);
    assert!(result.contains("n: u32"), "u32 type preserved");
    assert!(result.contains("x: f32"), "f64 downcasts to f32");
    assert!(result.contains("f32(1.5)"), "constructor downcasts");
}

#[test]
fn test_clamp_f64_range_handles_all_patterns() {
    let patterns = vec![
        ("-1.7976931348623157e+308", "-3.4028235e+38"),
        ("1.7976931348623157e+308", "3.4028235e+38"),
        ("-1e308", "-3.4028235e+38"),
        ("1e308", "3.4028235e+38"),
        ("-1e300", "-3.4028235e+38"),
        ("1e300", "3.4028235e+38"),
    ];
    for (input, expected) in &patterns {
        let result = downcast_f64_to_f32(input);
        assert!(
            result.contains(expected),
            "pattern {input} should become {expected}, got {result}"
        );
    }
}

/// Universal shader with comparison ops validates at all 3 tiers.
const UNIVERSAL_COMPARISON: &str = r"
@group(0) @binding(0) var<storage, read> a: array<Scalar>;
@group(0) @binding(1) var<storage, read> b: array<Scalar>;
@group(0) @binding(2) var<storage, read_write> output: array<Scalar>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&output)) { return; }
    if (op_gt(a[idx], b[idx])) {
        output[idx] = a[idx];
    } else {
        output[idx] = b[idx];
    }
}
";

#[cfg(feature = "gpu")]
#[test]
fn test_e2e_comparison_shader_all_precisions() {
    for prec in [Precision::F32, Precision::F64] {
        let preamble = prec.op_preamble();
        let source = format!("{preamble}\n{UNIVERSAL_COMPARISON}");
        let module = naga::front::wgsl::parse_str(&source)
            .unwrap_or_else(|e| panic!("{prec:?} comparison parse: {e}"));
        let mut v = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        v.validate(&module)
            .unwrap_or_else(|e| panic!("{prec:?} comparison validate: {e}"));
    }
    const DF64_CORE: &str = include_str!("../../shaders/math/df64_core.wgsl");
    const DF64_TRANS: &str = include_str!("../../shaders/math/df64_transcendentals.wgsl");
    let preamble = Precision::Df64.op_preamble();
    let source = format!("{DF64_CORE}\n{DF64_TRANS}\n{preamble}\n{UNIVERSAL_COMPARISON}");
    let module = naga::front::wgsl::parse_str(&source)
        .unwrap_or_else(|e| panic!("DF64 comparison parse: {e}"));
    let mut v = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );
    v.validate(&module)
        .unwrap_or_else(|e| panic!("DF64 comparison validate: {e}"));
}

/// Universal shader with pack/unpack validates at DF64.
const UNIVERSAL_PACK_UNPACK: &str = r"
@group(0) @binding(0) var<storage, read> input: array<StorageType>;
@group(0) @binding(1) var<storage, read_write> output: array<StorageType>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&output)) { return; }
    let val = op_unpack(input[idx]);
    let result = op_add(val, op_one());
    output[idx] = op_pack(result);
}
";

#[cfg(feature = "gpu")]
#[test]
fn test_e2e_pack_unpack_df64() {
    const DF64_CORE: &str = include_str!("../../shaders/math/df64_core.wgsl");
    const DF64_TRANS: &str = include_str!("../../shaders/math/df64_transcendentals.wgsl");
    let preamble = Precision::Df64.op_preamble();
    let source = format!("{DF64_CORE}\n{DF64_TRANS}\n{preamble}\n{UNIVERSAL_PACK_UNPACK}");
    let module = naga::front::wgsl::parse_str(&source)
        .unwrap_or_else(|e| panic!("DF64 pack/unpack parse: {e}"));
    let mut v = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );
    v.validate(&module)
        .unwrap_or_else(|e| panic!("DF64 pack/unpack validate: {e}"));
}

/// E2E: f64 canonical shader with transcendentals downcasts to f32 correctly.
#[test]
fn test_e2e_transcendental_downcast_f32() {
    let f64_shader = r"
@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;

fn activate(x: f64) -> f64 {
    return tanh_f64(x);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&output)) { return; }
    output[idx] = activate(input[idx]);
}
";
    let f32_source = downcast_f64_to_f32_with_transcendentals(f64_shader);
    assert!(f32_source.contains("array<f32>"), "storage downcast");
    assert!(f32_source.contains("tanh("), "tanh_f64 → tanh");
    assert!(!f32_source.contains("f64"), "no f64 should remain");
    let module = naga::front::wgsl::parse_str(&f32_source)
        .unwrap_or_else(|e| panic!("f32 downcast parse: {e}\n{f32_source}"));
    let mut v = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );
    v.validate(&module)
        .unwrap_or_else(|e| panic!("f32 downcast validate: {e}"));
}

#[test]
fn test_fault_preamble_consistency_under_concatenation() {
    for prec in [Precision::F32, Precision::F64, Precision::Df64] {
        let p = prec.op_preamble();
        let opens = p.matches('{').count();
        let closes = p.matches('}').count();
        assert_eq!(opens, closes, "{prec:?} preamble has unbalanced braces");
    }
}

#[test]
fn test_fault_downcast_idempotent_f32() {
    let source = "let x: f64 = f64(1.0);";
    let once = downcast_f64_to_f32(source);
    let twice = downcast_f64_to_f32(&once);
    assert_eq!(once, twice, "double downcast should be idempotent");
}

#[test]
fn test_fault_precision_bytes_consistent() {
    assert_eq!(Precision::F32.bytes_per_element(), 4);
    assert_eq!(Precision::F64.bytes_per_element(), 8);
    assert_eq!(Precision::Df64.bytes_per_element(), 8);
}

#[test]
fn test_fault_precision_is_f64_class() {
    assert!(!Precision::F32.is_f64_class());
    assert!(Precision::F64.is_f64_class());
    assert!(Precision::Df64.is_f64_class());
}

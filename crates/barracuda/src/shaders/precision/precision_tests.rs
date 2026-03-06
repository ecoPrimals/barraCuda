// SPDX-License-Identifier: AGPL-3.0-or-later
use super::*;

mod precision_chaos_tests;

#[test]
fn test_precision_types() {
    assert_eq!(Precision::F32.scalar(), "f32");
    assert_eq!(Precision::F64.scalar(), "f64");
    assert_eq!(Precision::F16.scalar(), "f16");
    assert!(Precision::F32.has_vec4());
    assert!(!Precision::F64.has_vec4());
}

#[test]
fn test_shader_generation() {
    let f32_shader = ShaderTemplate::elementwise_add(Precision::F32);
    assert!(f32_shader.contains("array<f32>"));
    assert!(f32_shader.contains("vec4<f32>"));
    let f64_shader = ShaderTemplate::elementwise_add(Precision::F64);
    assert!(f64_shader.contains("array<f64>"));
    assert!(!f64_shader.contains("vec4"));
}

#[test]
fn test_cpu_matches_description() {
    let a = vec![1.0_f64, 2.0, 3.0];
    let b = vec![4.0_f64, 5.0, 6.0];
    let mut out = vec![0.0_f64; 3];
    cpu::elementwise_add(&a, &b, &mut out);
    assert_eq!(out, vec![5.0, 7.0, 9.0]);
}

#[test]
fn test_math_f64_subset() {
    let subset = ShaderTemplate::math_f64_subset(&["sqrt_f64"]);
    // sqrt_f64 is fossil — still extractable from the library for reference
    assert!(subset.contains("fn sqrt_f64"));
    assert!(subset.contains("fn f64_const"));
    assert!(!subset.contains("fn exp_f64"));
    assert!(!subset.contains("fn sin_f64"));
    let pow_subset = ShaderTemplate::math_f64_subset(&["pow_f64"]);
    assert!(pow_subset.contains("fn pow_f64"));
    assert!(pow_subset.contains("fn exp_f64"));
    assert!(pow_subset.contains("fn log_f64"));
    // abs_f64 is no longer a dep of pow_f64 — pow body uses native abs() directly
    assert!(!pow_subset.contains("fn abs_f64"));
}

#[test]
fn test_math_f64_auto_detection() {
    let shader = r"
            @compute @workgroup_size(256)
            fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                let a = sqrt_f64(input[id.x]);
                let b = exp_f64(input[id.x]);
                output[id.x] = a + b;
            }
        ";
    let full_shader = ShaderTemplate::with_math_f64_auto(shader);
    // sqrt_f64 is fossil but still in the library — with_math_f64_auto includes it
    assert!(full_shader.contains("fn sqrt_f64"));
    assert!(full_shader.contains("fn exp_f64"));
    // abs_f64 and round_f64 are no longer deps of exp_f64 — body uses native builtins
    assert!(!full_shader.contains("fn abs_f64"));
    assert!(!full_shader.contains("fn round_f64"));
    assert!(!full_shader.contains("fn sin_f64"));
    assert!(!full_shader.contains("fn gamma_f64"));
}

#[test]
fn test_math_f64_auto_no_functions() {
    let shader = r"
            @compute @workgroup_size(256)
            fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                output[id.x] = input[id.x] * 2.0;
            }
        ";
    let full_shader = ShaderTemplate::with_math_f64_auto(shader);
    assert!(full_shader.contains("output[id.x] = input[id.x] * 2.0"));
    assert!(!full_shader.contains("fn sqrt_f64"));
}

#[test]
fn test_shader_defines_function() {
    let shader = r"
            fn f64_const(x: f64, c: f32) -> f64 {
                return x - x + f64(c);
            }
        ";
    assert!(ShaderTemplate::shader_defines_function(shader, "f64_const"));
    assert!(!ShaderTemplate::shader_defines_function(shader, "sqrt_f64"));
    let shader_space = r"fn sqrt_f64 (x: f64) -> f64 { return x; }";
    assert!(ShaderTemplate::shader_defines_function(
        shader_space,
        "sqrt_f64"
    ));
}

#[test]
fn test_shader_defines_module_var() {
    let shader_module_scope = r"
let zero = 0.0;
fn main() { }
";
    assert!(ShaderTemplate::shader_defines_module_var(
        shader_module_scope,
        "zero"
    ));
    let shader_local = r"
fn main() {
    let zero = x - x;
}
";
    assert!(!ShaderTemplate::shader_defines_module_var(
        shader_local,
        "zero"
    ));
    let shader_const = r"
const EPSILON: f64 = 1e-15;
";
    assert!(ShaderTemplate::shader_defines_module_var(
        shader_const,
        "EPSILON"
    ));
}

#[test]
fn test_safe_injects_only_called_functions() {
    // Fossil functions (sqrt_f64) are NOT injected — native sqrt() handles them.
    // Active fallbacks (cbrt_f64) ARE injected when called.
    let fossil_shader = r"
            @compute @workgroup_size(256)
            fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                output[id.x] = sqrt_f64(input[id.x]);
            }
        ";
    let fossil_result = ShaderTemplate::with_math_f64_safe(fossil_shader);
    assert!(
        !fossil_result.contains("fn sqrt_f64"),
        "fossil must not be injected"
    );
    assert!(
        !fossil_result.contains("fn f64_const"),
        "no injection means no preamble"
    );

    let active_shader = r"
            @compute @workgroup_size(256)
            fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                output[id.x] = cbrt_f64(input[id.x]);
            }
        ";
    let active_result = ShaderTemplate::with_math_f64_safe(active_shader);
    assert!(
        active_result.contains("fn cbrt_f64"),
        "active fallback must be injected"
    );
    assert!(active_result.contains("fn f64_const"));
    assert!(!active_result.contains("fn exp_f64"));
}

#[test]
fn test_safe_no_injection_for_native_calls() {
    let shader = r"
            @compute @workgroup_size(256)
            fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                output[id.x] = sqrt(input[id.x]);
            }
        ";
    let result = ShaderTemplate::with_math_f64_safe(shader);
    assert!(!result.contains("fn f64_const"));
}

#[test]
fn test_substitute_fossil_f64() {
    let legacy = "let y = sqrt_f64(x); let z = abs_f64(y); let w = min_f64(y, z);";
    let upgraded = ShaderTemplate::substitute_fossil_f64(legacy);
    assert!(upgraded.contains("sqrt(x)"));
    assert!(upgraded.contains("abs(y)"));
    assert!(upgraded.contains("min(y, z)"));
    assert!(!upgraded.contains("sqrt_f64("));
    assert!(!upgraded.contains("abs_f64("));
    assert!(!upgraded.contains("min_f64("));
    // Active fallbacks must NOT be touched
    let with_active = "let e = exp_f64(x); let c = cbrt_f64(x);";
    let result = ShaderTemplate::substitute_fossil_f64(with_active);
    assert!(result.contains("exp_f64("));
    assert!(result.contains("cbrt_f64("));
}

#[test]
fn test_sin_cos_taylor_workaround_asin_acos_protected() {
    use crate::device::capabilities::{
        CompilerKind, DriverKind, Fp64Rate, GpuArch, GpuDriverProfile, Workaround,
    };

    // NVK profile with sin/cos workaround
    let nvk_profile = GpuDriverProfile {
        driver: DriverKind::Nvk,
        compiler: CompilerKind::Nak,
        arch: GpuArch::Volta,
        fp64_rate: Fp64Rate::Full,
        workarounds: vec![
            Workaround::NvkExpF64Crash,
            Workaround::NvkLogF64Crash,
            Workaround::NvkSinCosF64Imprecise,
        ],
        adapter_key: String::new(),
    };
    let shader = "let a = sin(x); let b = cos(y); let c = asin(z); let d = acos(w);";
    let result = ShaderTemplate::for_driver_profile(shader, true, &nvk_profile);
    assert!(
        result.contains("sin_f64_safe("),
        "sin must become sin_f64_safe"
    );
    assert!(
        result.contains("cos_f64_safe("),
        "cos must become cos_f64_safe"
    );
    assert!(
        result.contains("asin_f64("),
        "asin must become asin_f64, not asin_f64_safe"
    );
    assert!(
        result.contains("acos_f64("),
        "acos must become acos_f64, not acos_f64_safe"
    );
    assert!(
        result.contains("fn sin_f64_safe"),
        "Taylor preamble must be injected"
    );
}

#[test]
fn test_for_driver_auto_applies_fossil_substitution() {
    // for_driver_auto should substitute fossils AND apply exp/log workaround
    let legacy_shader = r"
            @compute @workgroup_size(256)
            fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                let s = sqrt_f64(input[id.x]);
                let e = exp(s);
                output[id.x] = s + e;
            }
        ";
    let result = ShaderTemplate::for_driver_auto(legacy_shader, true);
    // sqrt_f64 → sqrt (fossil substitution)
    assert!(
        result.contains("sqrt("),
        "fossil sqrt_f64 must become native sqrt"
    );
    assert!(!result.contains("sqrt_f64("), "fossil name must be gone");
    // exp → exp_f64 (workaround)
    assert!(result.contains("exp_f64("));
    assert!(
        result.contains("fn exp_f64"),
        "exp fallback must be injected"
    );
}

#[test]
fn test_safe_partial_definitions_respected() {
    let shader = r"
            fn f64_const(x: f64, c: f32) -> f64 {
                return x - x + f64(c);
            }

            @compute @workgroup_size(256)
            fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                output[id.x] = exp_f64(f64_const(input[id.x], 1.0));
            }
        ";
    let result = ShaderTemplate::with_math_f64_safe(shader);
    assert_eq!(result.matches("fn f64_const").count(), 1);
    assert!(result.contains("fn exp_f64"));
}

#[test]
fn test_safe_all_defined_no_injection() {
    let shader = r"
            fn f64_const(x: f64, c: f32) -> f64 {
                return x - x + f64(c);
            }

            @compute @workgroup_size(256)
            fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                output[id.x] = f64_const(input[id.x], 1.0);
            }
        ";
    let result = ShaderTemplate::with_math_f64_safe(shader);
    assert_eq!(result.matches("fn f64_const").count(), 1);
}

#[test]
fn test_driver_workaround_with_partial_definitions() {
    let shader = r"
            fn f64_const(x: f64, c: f32) -> f64 {
                return x - x + f64(c);
            }
            fn erfc_f64(x: f64) -> f64 {
                return f64_const(x, 1.0) - erf_f64(x);
            }
            @compute @workgroup_size(256)
            fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                let v = exp(-input[id.x]);
                output[id.x] = erfc_f64(v);
            }
        ";
    let result = ShaderTemplate::for_driver_auto(shader, true);
    assert!(result.contains("exp_f64("));
    assert!(result.contains("fn exp_f64"));
    assert!(result.contains("fn erf_f64"));
    assert_eq!(result.matches("fn f64_const").count(), 1);
}

#[test]
fn test_driver_workaround_disabled() {
    let shader = r"
            @compute @workgroup_size(256)
            fn main(@builtin(global_invocation_id) id: vec3<u32>) {
                output[id.x] = exp(input[id.x]);
            }
        ";
    let result = ShaderTemplate::for_driver_auto(shader, false);
    assert!(result.contains("exp("));
    assert!(!result.contains("exp_f64("));
}

#[test]
fn test_precision_df64() {
    assert_eq!(Precision::Df64.scalar(), "vec2<f32>");
    assert_eq!(Precision::Df64.bytes_per_element(), 8);
    assert!(!Precision::Df64.has_vec4());
    assert!(Precision::Df64.required_feature().is_none());
    assert!(Precision::Df64.is_f64_class());
    assert!(Precision::F64.is_f64_class());
    assert!(!Precision::F32.is_f64_class());
}

#[test]
fn test_downcast_f64_to_f32_elementwise() {
    let f64_source = r"
@group(0) @binding(0) var<storage, read> a: array<f64>;
@group(0) @binding(1) var<storage, read> b: array<f64>;
@group(0) @binding(2) var<storage, read_write> output: array<f64>;

var<workgroup> shared: array<f64, 256>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&output)) { return; }
    output[idx] = a[idx] + b[idx];
}
";
    let f32_source = downcast_f64_to_f32(f64_source);
    assert!(f32_source.contains("array<f32>"));
    assert!(!f32_source.contains("array<f64>"));
    assert!(f32_source.contains("array<f32, 256>"));
    assert!(f32_source.contains("a[idx] + b[idx]"));
}

#[test]
fn test_downcast_f64_to_f32_with_transcendentals() {
    let f64_source = r"
@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    output[gid.x] = exp_f64(input[gid.x]) + sin_f64(input[gid.x]);
}
";
    let f32_source = downcast_f64_to_f32_with_transcendentals(f64_source);
    assert!(f32_source.contains("array<f32>"));
    assert!(f32_source.contains("exp(input"));
    assert!(f32_source.contains("sin(input"));
    assert!(!f32_source.contains("exp_f64"));
    assert!(!f32_source.contains("sin_f64"));
}

#[test]
fn test_downcast_preserves_u32_and_structure() {
    let f64_source = r"
struct Params { size: u32, _pad1: u32, _pad2: u32, _pad3: u32, }
@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<uniform> params: Params;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.size) { return; }
    let val: f64 = input[gid.x];
}
";
    let f32_source = downcast_f64_to_f32(f64_source);
    assert!(f32_source.contains("size: u32"));
    assert!(f32_source.contains("vec3<u32>"));
    assert!(f32_source.contains("let val: f32"));
    assert!(f32_source.contains("array<f32>"));
}

#[test]
fn test_template_renders_df64() {
    let shader = ShaderTemplate::elementwise_add(Precision::Df64);
    assert!(shader.contains("array<vec2<f32>>"));
    assert!(!shader.contains("vec4"));
}

#[test]
fn test_downcast_clamps_f64_range_sentinels() {
    let f64_source = r"
var max_val: f64 = -1e308;
var min_val: f64 = 1e308;
const DBL_MAX: f64 = 1.7976931348623157e+308;
const NEG_DBL_MAX: f64 = -1.7976931348623157e+308;
var approx: f64 = -1e300;
";
    let f32_source = downcast_f64_to_f32(f64_source);
    assert!(!f32_source.contains("1e308"), "f64 sentinel survived");
    assert!(!f32_source.contains("1e300"), "f64 sentinel survived");
    assert!(f32_source.contains("-3.4028235e+38"));
    assert!(f32_source.contains("3.4028235e+38"));
}

#[test]
fn test_downcast_f64_to_df64_types() {
    let f64_source = r"
@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;

fn process(val: f64) -> f64 {
    return val;
}
";
    let df64 = downcast_f64_to_df64(f64_source);
    assert!(
        df64.contains("array<vec2<f32>>"),
        "storage should be vec2<f32>"
    );
    assert!(df64.contains("val: Df64"), "param type should be Df64");
    assert!(df64.contains("-> Df64"), "return type should be Df64");
    assert!(
        !df64.contains("array<f64>"),
        "no raw f64 should remain in storage"
    );
}

#[test]
fn test_downcast_f64_to_df64_constructors() {
    let f64_source = r"
let zero: f64 = f64(0.0);
let one: f64 = f64(1.0);
let half: f64 = f64(0.5);
";
    let df64 = downcast_f64_to_df64(f64_source);
    assert!(
        df64.contains("df64_from_f32(0.0)"),
        "f64(0.0) → df64_from_f32(0.0)"
    );
    assert!(
        df64.contains("df64_from_f32(1.0)"),
        "f64(1.0) → df64_from_f32(1.0)"
    );
    assert!(
        df64.contains("df64_from_f32(0.5)"),
        "f64(0.5) → df64_from_f32(0.5)"
    );
    assert!(
        !df64.contains("f64("),
        "no raw f64 constructors should remain"
    );
}

#[test]
fn test_downcast_f64_to_df64_transcendentals() {
    let f64_source = r"
let y = exp_f64(x);
let z = sin_f64(x);
let w = sqrt_f64(x);
let v = tanh_f64(x);
let a = abs_f64(x);
";
    let df64 = downcast_f64_to_df64(f64_source);
    assert!(df64.contains("exp_df64("), "exp_f64 → exp_df64");
    assert!(df64.contains("sin_df64("), "sin_f64 → sin_df64");
    assert!(df64.contains("sqrt_df64("), "sqrt_f64 → sqrt_df64");
    assert!(df64.contains("tanh_df64("), "tanh_f64 → tanh_df64");
    assert!(df64.contains("df64_abs("), "abs_f64 → df64_abs");
    assert!(!df64.contains("exp_f64"), "no raw f64 polyfill calls");
}

#[test]
fn test_op_preamble_f32_has_all_ops() {
    let p = Precision::F32.op_preamble();
    assert!(p.contains("fn op_add("));
    assert!(p.contains("fn op_sub("));
    assert!(p.contains("fn op_mul("));
    assert!(p.contains("fn op_div("));
    assert!(p.contains("fn op_neg("));
    assert!(p.contains("fn op_abs("));
    assert!(p.contains("fn op_max("));
    assert!(p.contains("fn op_min("));
    assert!(p.contains("fn op_gt("));
    assert!(p.contains("fn op_lt("));
    assert!(p.contains("fn op_from_f32("));
    assert!(p.contains("fn op_zero("));
    assert!(p.contains("fn op_one("));
    assert!(p.contains("alias Scalar = f32"));
}

#[test]
fn test_op_preamble_df64_routes_to_library() {
    let p = Precision::Df64.op_preamble();
    assert!(
        p.contains("df64_add(a, b)"),
        "op_add should route to df64_add"
    );
    assert!(
        p.contains("df64_mul(a, b)"),
        "op_mul should route to df64_mul"
    );
    assert!(
        p.contains("df64_div(a, b)"),
        "op_div should route to df64_div"
    );
    assert!(
        p.contains("df64_sub(a, b)"),
        "op_sub should route to df64_sub"
    );
    assert!(p.contains("df64_neg(a)"), "op_neg should route to df64_neg");
    assert!(p.contains("fn op_pack("), "DF64 needs pack for storage");
    assert!(p.contains("fn op_unpack("), "DF64 needs unpack for storage");
    assert!(p.contains("alias Scalar = Df64"));
    assert!(p.contains("alias StorageType = vec2<f32>"));
}

#[test]
fn test_op_preamble_all_precisions_consistent() {
    for prec in [
        Precision::F16,
        Precision::F32,
        Precision::F64,
        Precision::Df64,
    ] {
        let p = prec.op_preamble();
        assert!(p.contains("fn op_add("), "{prec:?} missing op_add");
        assert!(p.contains("fn op_mul("), "{prec:?} missing op_mul");
        assert!(p.contains("fn op_zero("), "{prec:?} missing op_zero");
        assert!(p.contains("alias Scalar"), "{prec:?} missing Scalar alias");
    }
}

#[test]
fn test_downcast_f64_to_df64_preserves_u32() {
    let f64_source = r"
struct Params { size: u32, }
@group(0) @binding(0) var<storage, read> input: array<f64>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.size) { return; }
}
";
    let df64 = downcast_f64_to_df64(f64_source);
    assert!(df64.contains("size: u32"), "u32 fields preserved");
    assert!(df64.contains("vec3<u32>"), "u32 builtins preserved");
    assert!(df64.contains("array<vec2<f32>>"), "f64 storage → vec2<f32>");
}

/// A universal shader written with op_* functions — works at ALL precisions.
const UNIVERSAL_ELEMENTWISE_ADD: &str = r"
@group(0) @binding(0) var<storage, read> a: array<Scalar>;
@group(0) @binding(1) var<storage, read> b: array<Scalar>;
@group(0) @binding(2) var<storage, read_write> output: array<Scalar>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= arrayLength(&output)) { return; }
    output[idx] = op_add(a[idx], b[idx]);
}
";

/// Prove the universal shader is valid WGSL at f32 precision via naga parse.
#[cfg(feature = "gpu")]
#[test]
fn test_universal_shader_validates_f32() {
    let preamble = Precision::F32.op_preamble();
    let source = format!("{preamble}\n{UNIVERSAL_ELEMENTWISE_ADD}");
    let module = naga::front::wgsl::parse_str(&source).expect("f32 universal shader should parse");
    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );
    validator
        .validate(&module)
        .expect("f32 universal shader should validate");
}

/// Prove the universal shader is valid WGSL at f64 precision via naga parse.
#[cfg(feature = "gpu")]
#[test]
fn test_universal_shader_validates_f64() {
    let preamble = Precision::F64.op_preamble();
    let source = format!("{preamble}\n{UNIVERSAL_ELEMENTWISE_ADD}");
    let module = naga::front::wgsl::parse_str(&source).expect("f64 universal shader should parse");
    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );
    validator
        .validate(&module)
        .expect("f64 universal shader should validate");
}

/// Prove the universal shader is valid WGSL at DF64 precision via naga parse.
/// The DF64 path: `df64_core` + `df64_transcendentals` + `op_preamble` + shader.
#[cfg(feature = "gpu")]
#[test]
fn test_universal_shader_validates_df64() {
    const DF64_CORE: &str = include_str!("../../shaders/math/df64_core.wgsl");
    const DF64_TRANSCENDENTALS: &str = include_str!("../../shaders/math/df64_transcendentals.wgsl");
    let preamble = Precision::Df64.op_preamble();
    let source =
        format!("{DF64_CORE}\n{DF64_TRANSCENDENTALS}\n{preamble}\n{UNIVERSAL_ELEMENTWISE_ADD}");
    let module = naga::front::wgsl::parse_str(&source).expect("DF64 universal shader should parse");
    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );
    validator
        .validate(&module)
        .expect("DF64 universal shader should validate");
}

/// Universal shader with more complex math — reduction with `op_add`.
const UNIVERSAL_REDUCE_SUM: &str = r"
var<workgroup> wg_buf: array<Scalar, 256>;

@group(0) @binding(0) var<storage, read> input: array<Scalar>;
@group(0) @binding(1) var<storage, read_write> output: array<Scalar>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let n = arrayLength(&input);
    if (gid.x < n) {
        wg_buf[lid.x] = input[gid.x];
    } else {
        wg_buf[lid.x] = op_zero();
    }
    workgroupBarrier();

    for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
        if (lid.x < stride) {
            wg_buf[lid.x] = op_add(wg_buf[lid.x], wg_buf[lid.x + stride]);
        }
        workgroupBarrier();
    }

    if (lid.x == 0u) {
        output[wid.x] = wg_buf[0];
    }
}
";

/// Prove the universal reduce shader validates at all 3 main precisions.
#[cfg(feature = "gpu")]
#[test]
fn test_universal_reduce_validates_all_precisions() {
    for prec in [Precision::F32, Precision::F64] {
        let preamble = prec.op_preamble();
        let source = format!("{preamble}\n{UNIVERSAL_REDUCE_SUM}");
        let module = naga::front::wgsl::parse_str(&source)
            .unwrap_or_else(|e| panic!("{prec:?} reduce parse failed: {e}"));
        let mut v = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        v.validate(&module)
            .unwrap_or_else(|e| panic!("{prec:?} reduce validation failed: {e}"));
    }

    // DF64 needs the core library prepended
    const DF64_CORE: &str = include_str!("../../shaders/math/df64_core.wgsl");
    const DF64_TRANSCENDENTALS: &str = include_str!("../../shaders/math/df64_transcendentals.wgsl");
    let preamble = Precision::Df64.op_preamble();
    let source = format!("{DF64_CORE}\n{DF64_TRANSCENDENTALS}\n{preamble}\n{UNIVERSAL_REDUCE_SUM}");
    let module = naga::front::wgsl::parse_str(&source)
        .unwrap_or_else(|e| panic!("DF64 reduce parse failed: {e}"));
    let mut v = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );
    v.validate(&module)
        .unwrap_or_else(|e| panic!("DF64 reduce validation failed: {e}"));
}

// ══════════════════════════════════════════════════════════════════════
// Unit Tests — edge cases and uncovered paths
// ══════════════════════════════════════════════════════════════════════

#[test]
fn test_downcast_f64_to_f16_sentinel_protection() {
    let source = "let x = exp_f64(input[i]);\nlet y: f64 = f64(1.0);";
    let result = downcast_f64_to_f16(source);
    assert!(
        result.contains("exp_f64("),
        "polyfill name must survive sentinel"
    );
    assert!(result.contains(": f16"), "type should downcast to f16");
    assert!(result.contains("f16(1.0)"), "constructor should downcast");
    assert!(
        !result.contains("exp_f16("),
        "exp_f64 must NOT become exp_f16"
    );
}

#[test]
fn test_downcast_f64_to_f16_clamps_f64_range_literals() {
    let source = "let x: f64 = f64(-1e308);";
    let result = downcast_f64_to_f16(source);
    assert!(
        result.contains("-65504.0"),
        "f64 sentinel should clamp to f16 max"
    );
    assert!(!result.contains("1e308"), "f64-range literal must be gone");
}

#[test]
fn test_downcast_f64_to_f16_clamps_f32_range_to_f16() {
    let source = "let x = 3.4028235e+38;";
    let result = downcast_f64_to_f16(source);
    assert!(
        result.contains("65504.0"),
        "f32-range literal should clamp to f16"
    );
}

#[test]
fn test_op_preamble_pack_unpack_all_precisions() {
    for prec in [
        Precision::F16,
        Precision::F32,
        Precision::F64,
        Precision::Df64,
    ] {
        let p = prec.op_preamble();
        assert!(p.contains("fn op_pack("), "{prec:?} missing op_pack");
        assert!(p.contains("fn op_unpack("), "{prec:?} missing op_unpack");
    }
}

#[test]
fn test_downcast_df64_only_maps_existing_transcendentals() {
    let source = "let a = exp_f64(x); let b = tan_f64(y); let c = sqrt_f64(z);";
    let result = downcast_f64_to_df64(source);
    assert!(result.contains("exp_df64("), "exp should map");
    assert!(result.contains("sqrt_df64("), "sqrt should map");
    // tan_f64 should NOT be mapped since tan_df64 doesn't exist
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

// ══════════════════════════════════════════════════════════════════════
// End-to-End Tests — real shader patterns through full pipeline
// ══════════════════════════════════════════════════════════════════════

/// Universal shader with comparison ops validates at all precisions.
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
    // DF64
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
    // Parse with naga
    let module = naga::front::wgsl::parse_str(&f32_source)
        .unwrap_or_else(|e| panic!("f32 downcast parse: {e}\n{f32_source}"));
    let mut v = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );
    v.validate(&module)
        .unwrap_or_else(|e| panic!("f32 downcast validate: {e}"));
}

// ══════════════════════════════════════════════════════════════════════
// Fault Tests — graceful degradation, fallback paths, error recovery
// ══════════════════════════════════════════════════════════════════════

#[test]
fn test_fault_preamble_consistency_under_concatenation() {
    for prec in [
        Precision::F16,
        Precision::F32,
        Precision::F64,
        Precision::Df64,
    ] {
        let p = prec.op_preamble();
        // Every preamble must have matching open/close braces
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
fn test_fault_downcast_idempotent_f16() {
    let source = "let x: f64 = f64(1.0);";
    let once = downcast_f64_to_f16(source);
    let twice = downcast_f64_to_f16(&once);
    assert_eq!(once, twice, "double f16 downcast should be idempotent");
}

#[test]
fn test_fault_precision_bytes_consistent() {
    assert_eq!(Precision::F16.bytes_per_element(), 2);
    assert_eq!(Precision::F32.bytes_per_element(), 4);
    assert_eq!(Precision::F64.bytes_per_element(), 8);
    assert_eq!(Precision::Df64.bytes_per_element(), 8);
}

#[test]
fn test_fault_precision_is_f64_class() {
    assert!(!Precision::F16.is_f64_class());
    assert!(!Precision::F32.is_f64_class());
    assert!(Precision::F64.is_f64_class());
    assert!(Precision::Df64.is_f64_class());
}

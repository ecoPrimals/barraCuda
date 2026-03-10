// SPDX-License-Identifier: AGPL-3.0-only
use super::*;

fn rewrite_and_resolve(wgsl: &str) -> String {
    let rewritten = rewrite_f64_infix_to_df64(wgsl).expect("should parse");

    resolve_spans(&rewritten, wgsl)
}

#[test]
fn test_count_f64_infix_simple_add() {
    let wgsl = r"
@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
let i = gid.x;
let a = input[i];
let b = input[i + 1u];
output[i] = a + b;
}
";
    let count = count_f64_infix_ops(wgsl).expect("should parse");
    assert!(count >= 1, "expected at least 1 f64 infix op, got {count}");
}

#[test]
fn test_count_f64_infix_no_f64_ops() {
    let wgsl = r"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
let i = gid.x;
output[i] = input[i] + 1.0;
}
";
    let count = count_f64_infix_ops(wgsl).expect("should parse");
    assert_eq!(count, 0, "f32 ops should not be counted");
}

#[test]
fn test_rewrite_simple_add() {
    let wgsl = r"
@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
let i = gid.x;
let a = input[i];
let b = input[i + 1u];
output[i] = a + b;
}
";
    let result = rewrite_and_resolve(wgsl);
    assert!(
        result.contains("_df64_add_f64("),
        "should contain bridge call, got:\n{result}"
    );
}

#[test]
fn test_rewrite_nested_ops() {
    let wgsl = r"
@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
let i = gid.x;
let a = input[i];
let b = input[i + 1u];
let c = input[i + 2u];
output[i] = (a + b) * c;
}
";
    let result = rewrite_and_resolve(wgsl);
    assert!(
        result.contains("_df64_mul_f64("),
        "should contain mul bridge, got:\n{result}"
    );
    assert!(
        result.contains("_df64_add_f64("),
        "should contain add bridge for nested op, got:\n{result}"
    );
}

#[test]
fn test_rewrite_preserves_u32_ops() {
    let wgsl = r"
@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
let i = gid.x;
let j = i + 1u;
let a = input[i];
let b = input[j];
output[i] = a * b;
}
";
    let result = rewrite_and_resolve(wgsl);
    assert!(
        result.contains("_df64_mul_f64("),
        "f64 mul should be rewritten, got:\n{result}"
    );
}

#[test]
fn test_rewrite_negation() {
    let wgsl = r"
@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
let i = gid.x;
let a = input[i];
output[i] = -a;
}
";
    let result = rewrite_and_resolve(wgsl);
    assert!(
        result.contains("_df64_neg_f64("),
        "negation should be rewritten, got:\n{result}"
    );
}

/// End-to-end: full pipeline produces valid WGSL when compiled with df64 core.
#[test]
fn test_full_pipeline_validates() {
    let f64_source = r"
@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
let i = gid.x;
let a = input[i];
let b = input[i + 1u];
let c = input[i + 2u];
let sum = a + b;
let product = sum * c;
let result = product - a;
output[i] = result;
}
";
    let rewritten = rewrite_f64_infix_full(f64_source).expect("rewrite");

    const DF64_CORE: &str = include_str!("../../../shaders/math/df64_core.wgsl");
    const DF64_TRANS: &str = include_str!("../../../shaders/math/df64_transcendentals.wgsl");
    let full = format!("{DF64_CORE}\n{DF64_TRANS}\n{rewritten}");

    let module = naga::front::wgsl::parse_str(&full)
        .unwrap_or_else(|e| panic!("should parse: {e}\n\nSource:\n{full}"));
    let mut v = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );
    v.validate(&module)
        .unwrap_or_else(|e| panic!("should validate: {e}\n\nSource:\n{full}"));
}

#[test]
fn test_fma_pattern() {
    let wgsl = r"
@group(0) @binding(0) var<storage, read> a_buf: array<f64>;
@group(0) @binding(1) var<storage, read> b_buf: array<f64>;
@group(0) @binding(2) var<storage, read> c_buf: array<f64>;
@group(0) @binding(3) var<storage, read_write> out: array<f64>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
let i = gid.x;
let a = a_buf[i];
let b = b_buf[i];
let c = c_buf[i];
out[i] = a * b + c;
}
";
    let result = rewrite_and_resolve(wgsl);
    assert!(
        result.contains("_df64_add_f64(") && result.contains("_df64_mul_f64("),
        "FMA pattern should produce both, got:\n{result}"
    );
}

// ══════════════════════════════════════════════════════════════
// Chaos tests for the naga rewriter
// ══════════════════════════════════════════════════════════════

#[test]
fn test_chaos_naga_f32_only_shader() {
    let wgsl = r"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
output[gid.x] = input[gid.x] + 1.0;
}
";
    let result = rewrite_f64_infix_to_df64(wgsl).expect("should not fail on f32 shader");
    assert_eq!(result, wgsl, "f32-only shader should be returned unchanged");
}

#[test]
fn test_chaos_naga_invalid_wgsl() {
    let result = rewrite_f64_infix_to_df64("this is not valid wgsl");
    assert!(result.is_err(), "invalid WGSL should return Err");
}

#[test]
fn test_chaos_naga_empty_shader() {
    // Empty source may or may not parse/validate in naga.
    // Either way, the function should not panic.
    let result = rewrite_f64_infix_to_df64("");
    // If naga accepts it, the result is the source unchanged.
    // If naga rejects it, we get an Err. Both are acceptable.
    if let Ok(s) = result {
        assert_eq!(s, "", "empty in, empty out");
    }
}

#[test]
fn test_chaos_count_mixed_f32_f64_ops() {
    let wgsl = r"
@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
let i = gid.x;
let j = i + 1u;
let k = j * 2u;
let a = input[i];
let b = input[k];
output[i] = a + b;
}
";
    let count = count_f64_infix_ops(wgsl).expect("should parse");
    assert!(
        count >= 1,
        "should count f64 add but not u32 ops, got {count}"
    );
}

#[test]
fn test_chaos_naga_subtraction_chain() {
    let wgsl = r"
@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
let i = gid.x;
let a = input[i];
let b = input[i + 1u];
let c = input[i + 2u];
let d = input[i + 3u];
output[i] = a - b - c - d;
}
";
    let result = rewrite_and_resolve(wgsl);
    assert!(
        result.contains("_df64_sub_f64("),
        "subtraction chain should use sub bridges, got:\n{result}"
    );
}

#[test]
fn test_chaos_naga_division() {
    let wgsl = r"
@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
let i = gid.x;
output[i] = input[i] / input[i + 1u];
}
";
    let result = rewrite_and_resolve(wgsl);
    assert!(
        result.contains("_df64_div_f64("),
        "division should be rewritten, got:\n{result}"
    );
}

// ══════════════════════════════════════════════════════════════
// Fault tests for the naga rewriter
// ══════════════════════════════════════════════════════════════

#[test]
fn test_fault_resolve_spans_empty() {
    let result = resolve_spans("no spans here", "original");
    assert_eq!(result, "no spans here");
}

#[test]
fn test_fault_resolve_spans_valid() {
    let original = "hello world";
    let rewritten = "prefix __SPAN__0__5 suffix";
    let result = resolve_spans(rewritten, original);
    assert_eq!(result, "prefix hello suffix");
}

#[test]
fn test_fault_resolve_spans_out_of_bounds() {
    let original = "short";
    let rewritten = "__SPAN__0__999";
    let result = resolve_spans(rewritten, original);
    // Should not panic, marker stays as-is
    assert!(
        result.contains("__SPAN__"),
        "out of bounds should leave marker"
    );
}

#[test]
fn test_fault_resolve_spans_inverted_range() {
    let original = "hello world";
    let rewritten = "__SPAN__5__0";
    let result = resolve_spans(rewritten, original);
    assert!(
        result.contains("__SPAN__"),
        "inverted range should leave marker"
    );
}

#[test]
fn test_fault_bridge_functions_defined() {
    let bf = bridge_functions();
    assert!(bf.contains("_df64_add_f64"), "add bridge");
    assert!(bf.contains("_df64_sub_f64"), "sub bridge");
    assert!(bf.contains("_df64_mul_f64"), "mul bridge");
    assert!(bf.contains("_df64_div_f64"), "div bridge");
    assert!(bf.contains("_df64_neg_f64"), "neg bridge");
    assert!(bf.contains("_df64_gt_f64"), "gt bridge");
    assert!(bf.contains("_df64_lt_f64"), "lt bridge");
    assert!(bf.contains("_df64_gte_f64"), "gte bridge");
    assert!(bf.contains("_df64_lte_f64"), "lte bridge");
}

#[test]
fn test_fault_full_pipeline_no_f64_ops_passthrough() {
    let wgsl = r"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
output[gid.x] = input[gid.x] * 2.0;
}
";
    let result = rewrite_f64_infix_full(wgsl).expect("should succeed");
    assert_eq!(result, wgsl, "no f64 ops = passthrough unchanged");
}

#[test]
fn test_fault_dedup_preserves_non_overlapping() {
    let mut replacements = vec![
        Replacement {
            span_start: 100,
            span_end: 110,
            text: "A".into(),
        },
        Replacement {
            span_start: 50,
            span_end: 60,
            text: "B".into(),
        },
        Replacement {
            span_start: 10,
            span_end: 20,
            text: "C".into(),
        },
    ];
    dedup_overlapping(&mut replacements);
    assert_eq!(replacements.len(), 3, "non-overlapping should all survive");
}

#[test]
fn test_fault_dedup_removes_nested() {
    let mut replacements = vec![
        Replacement {
            span_start: 15,
            span_end: 25,
            text: "inner".into(),
        },
        Replacement {
            span_start: 10,
            span_end: 30,
            text: "outer".into(),
        },
    ];
    // sorted by span_start descending
    replacements.sort_by(|a, b| b.span_start.cmp(&a.span_start));
    dedup_overlapping(&mut replacements);
    assert_eq!(replacements.len(), 1, "nested should be deduped");
    assert_eq!(replacements[0].text, "outer", "outermost should survive");
}

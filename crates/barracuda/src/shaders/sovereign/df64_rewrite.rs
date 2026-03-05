// SPDX-License-Identifier: AGPL-3.0-or-later
//! DF64 Infix Rewrite Pass — Sovereign Compiler Phase 5.
//!
//! Uses naga for type analysis and source-span-guided text replacement
//! to transform f64 infix arithmetic into df64 function calls.
//!
//! ## Architecture ("both" approach)
//!
//! Layer 1 (source level): Operation preamble (`op_add`, `op_mul`, etc.)
//!   — new shaders use abstract ops that work at all precisions immediately.
//!
//! Layer 2 (compiler level, this module): Naga-guided source rewrite
//!   — existing f64 shaders with infix operators get transformed automatically.
//!   The naga frontend parses the f64 WGSL into typed IR, we walk the IR to
//!   find f64 Binary/Unary expressions, extract their source spans, and
//!   replace the source text with bridge function calls that keep the f64
//!   type system intact while routing computation through DF64.
//!
//! Together: op_preamble makes new code portable by design, naga makes
//! everything portable by force.
//!
//! ## Algorithm
//!
//! 1. Parse f64-canonical WGSL with `naga::front::wgsl`
//! 2. Validate with `naga::valid::Validator` to get expression types
//! 3. Walk expressions: find `Binary{+,-,*,/}` and `Unary{-}` on f64 types
//! 4. Replace each with bridge function calls (`_df64_add_f64(a, b)` etc.)
//!    that accept f64, compute in Df64, return f64 — no type mismatches
//! 5. Prepend bridge function definitions
//!
//! The bridge approach means every f64 infix op routes through DF64 cores
//! transparently. Storage stays `array<f64>`, variables stay `f64`, the
//! type system is untouched. Only the arithmetic is redirected.

use naga::proc::TypeResolution;
use naga::valid::FunctionInfo;
use naga::{Arena, BinaryOperator, Expression, Handle, TypeInner, UnaryOperator};
use std::collections::HashSet;

/// Bridge functions that accept f64, compute in Df64, return f64.
/// These get prepended to the shader source after the df64 core library.
/// The GPU compiler inlines these — zero overhead.
const DF64_BRIDGE_FUNCTIONS: &str = r#"
fn _df64_add_f64(a: f64, b: f64) -> f64 { return df64_to_f64(df64_add(df64_from_f64(a), df64_from_f64(b))); }
fn _df64_sub_f64(a: f64, b: f64) -> f64 { return df64_to_f64(df64_sub(df64_from_f64(a), df64_from_f64(b))); }
fn _df64_mul_f64(a: f64, b: f64) -> f64 { return df64_to_f64(df64_mul(df64_from_f64(a), df64_from_f64(b))); }
fn _df64_div_f64(a: f64, b: f64) -> f64 { return df64_to_f64(df64_div(df64_from_f64(a), df64_from_f64(b))); }
fn _df64_neg_f64(a: f64) -> f64 { return df64_to_f64(df64_neg(df64_from_f64(a))); }
fn _df64_gt_f64(a: f64, b: f64) -> bool { return df64_gt(df64_from_f64(a), df64_from_f64(b)); }
fn _df64_lt_f64(a: f64, b: f64) -> bool { return df64_lt(df64_from_f64(a), df64_from_f64(b)); }
fn _df64_gte_f64(a: f64, b: f64) -> bool { let da = df64_from_f64(a); let db = df64_from_f64(b); return df64_gt(da, db) || (da.hi == db.hi && da.lo == db.lo); }
fn _df64_lte_f64(a: f64, b: f64) -> bool { let da = df64_from_f64(a); let db = df64_from_f64(b); return df64_lt(da, db) || (da.hi == db.hi && da.lo == db.lo); }
"#;

struct Replacement {
    span_start: usize,
    span_end: usize,
    text: String,
}

/// Rewrite f64 infix arithmetic to route through DF64 computation.
///
/// Takes f64-canonical WGSL source, uses naga for type-aware analysis,
/// and produces WGSL with all f64 infix operators replaced by bridge
/// functions that compute in Df64 while keeping the f64 type system.
///
/// The returned source still uses f64 types throughout — bridge functions
/// handle the f64 → Df64 → f64 conversion transparently.
///
/// Returns `Err` if the source fails to parse or validate as f64 WGSL.
pub fn rewrite_f64_infix_to_df64(f64_source: &str) -> Result<String, String> {
    let module =
        naga::front::wgsl::parse_str(f64_source).map_err(|e| format!("naga parse: {e}"))?;

    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );
    let info = validator
        .validate(&module)
        .map_err(|e| format!("naga validate: {e}"))?;

    let mut replacements: Vec<Replacement> = Vec::new();

    for (ep_idx, ep) in module.entry_points.iter().enumerate() {
        let fi = info.get_entry_point(ep_idx);
        collect_f64_infix_ops(&ep.function.expressions, fi, &module, &mut replacements);
    }

    for (fh, func) in module.functions.iter() {
        let fi = &info[fh];
        collect_f64_infix_ops(&func.expressions, fi, &module, &mut replacements);
    }

    if replacements.is_empty() {
        return Ok(f64_source.to_string());
    }

    // Sort back-to-front so byte offsets remain valid during replacement.
    // For overlapping spans (nested ops), take the outermost only.
    replacements.sort_by(|a, b| b.span_start.cmp(&a.span_start));
    dedup_overlapping(&mut replacements);

    let mut result = f64_source.to_string();
    for r in &replacements {
        if r.span_start <= r.span_end && r.span_end <= result.len() {
            result.replace_range(r.span_start..r.span_end, &r.text);
        }
    }

    Ok(result)
}

/// Remove overlapping replacements, keeping the outermost (widest span).
/// Input must be sorted by span_start descending.
fn dedup_overlapping(replacements: &mut Vec<Replacement>) {
    let mut keep = vec![true; replacements.len()];
    for i in 0..replacements.len() {
        if !keep[i] {
            continue;
        }
        for j in (i + 1)..replacements.len() {
            if !keep[j] {
                continue;
            }
            // j has earlier (or equal) span_start since sorted descending
            if replacements[j].span_start <= replacements[i].span_start
                && replacements[j].span_end >= replacements[i].span_end
            {
                // j contains i — keep j (outermost), drop i
                keep[i] = false;
                break;
            }
            if replacements[i].span_start <= replacements[j].span_start
                && replacements[i].span_end >= replacements[j].span_end
            {
                // i contains j — keep i, drop j
                keep[j] = false;
            }
        }
    }
    let mut idx = 0;
    replacements.retain(|_| {
        let k = keep[idx];
        idx += 1;
        k
    });
}

/// Count how many f64 infix operations exist in a shader (for audit/reporting).
pub fn count_f64_infix_ops(f64_source: &str) -> Result<usize, String> {
    let module =
        naga::front::wgsl::parse_str(f64_source).map_err(|e| format!("naga parse: {e}"))?;

    let mut validator = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    );
    let info = validator
        .validate(&module)
        .map_err(|e| format!("naga validate: {e}"))?;

    let mut count = 0usize;

    for (ep_idx, ep) in module.entry_points.iter().enumerate() {
        let fi = info.get_entry_point(ep_idx);
        count += count_f64_ops_in(&ep.function.expressions, fi, &module);
    }
    for (fh, func) in module.functions.iter() {
        let fi = &info[fh];
        count += count_f64_ops_in(&func.expressions, fi, &module);
    }

    Ok(count)
}

/// Returns the bridge function definitions that must be prepended when
/// `rewrite_f64_infix_to_df64` has produced replacements.
pub fn bridge_functions() -> &'static str {
    DF64_BRIDGE_FUNCTIONS
}

fn count_f64_ops_in(
    expressions: &Arena<Expression>,
    fi: &FunctionInfo,
    module: &naga::Module,
) -> usize {
    let mut count = 0;
    for (_handle, expr) in expressions.iter() {
        match *expr {
            Expression::Binary { op, left, .. }
                if is_rewritable_op(op) && is_f64_expr(left, fi, module) =>
            {
                count += 1;
            }
            Expression::Unary {
                op: UnaryOperator::Negate,
                expr: inner,
            } if is_f64_expr(inner, fi, module) => {
                count += 1;
            }
            _ => {}
        }
    }
    count
}

/// Collect ALL f64 infix operations and build their replacements.
/// Each infix op gets replaced with a bridge function call that keeps
/// the f64 type system while routing through DF64 computation.
fn collect_f64_infix_ops(
    expressions: &Arena<Expression>,
    fi: &FunctionInfo,
    module: &naga::Module,
    out: &mut Vec<Replacement>,
) {
    // Find which handles are consumed as operands of another f64 infix op.
    // We need this to decide whether to use recursive building for nested ops.
    let mut consumed_by_f64_op: HashSet<Handle<Expression>> = HashSet::new();
    for (_handle, expr) in expressions.iter() {
        match *expr {
            Expression::Binary { op, left, right }
                if is_rewritable_op(op) && is_f64_expr(left, fi, module) =>
            {
                consumed_by_f64_op.insert(left);
                consumed_by_f64_op.insert(right);
            }
            Expression::Unary {
                op: UnaryOperator::Negate,
                expr: inner,
            } if is_f64_expr(inner, fi, module) => {
                consumed_by_f64_op.insert(inner);
            }
            _ => {}
        }
    }

    // Collect root f64 ops (not consumed by another f64 op).
    // Use recursive builder to handle nested expressions correctly.
    for (handle, expr) in expressions.iter() {
        if consumed_by_f64_op.contains(&handle) {
            continue;
        }

        let is_f64_op = match *expr {
            Expression::Binary { op, left, .. } => {
                is_rewritable_op(op) && is_f64_expr(left, fi, module)
            }
            Expression::Unary {
                op: UnaryOperator::Negate,
                expr: inner,
            } => is_f64_expr(inner, fi, module),
            _ => false,
        };

        if is_f64_op {
            if let Some(span_range) = expressions.get_span(handle).to_range() {
                let text = build_bridge_text(handle, expressions, fi, module, &consumed_by_f64_op);
                out.push(Replacement {
                    span_start: span_range.start,
                    span_end: span_range.end,
                    text,
                });
            }
        }
    }

    // Also collect non-root f64 ops whose span doesn't overlap with any root.
    // These are in separate statements (e.g. `let x = a + b; let y = x * c;`).
    for (handle, expr) in expressions.iter() {
        if !consumed_by_f64_op.contains(&handle) {
            continue;
        }

        let is_f64_op = match *expr {
            Expression::Binary { op, left, .. } => {
                is_rewritable_op(op) && is_f64_expr(left, fi, module)
            }
            Expression::Unary {
                op: UnaryOperator::Negate,
                expr: inner,
            } => is_f64_expr(inner, fi, module),
            _ => false,
        };

        if is_f64_op {
            if let Some(span_range) = expressions.get_span(handle).to_range() {
                // Check if this span overlaps with any existing replacement
                let overlaps = out
                    .iter()
                    .any(|r| span_range.start < r.span_end && span_range.end > r.span_start);
                if !overlaps {
                    let text =
                        build_bridge_text(handle, expressions, fi, module, &consumed_by_f64_op);
                    out.push(Replacement {
                        span_start: span_range.start,
                        span_end: span_range.end,
                        text,
                    });
                }
            }
        }
    }
}

/// Build bridge function call text for an f64 infix expression.
///
/// For nested f64 ops within the same expression tree (e.g. `(a+b)*c`),
/// recursively builds: `_df64_mul_f64(_df64_add_f64(a, b), c)`.
///
/// For leaves (non-f64-op operands), uses the original source text.
/// Since bridge functions accept f64 and return f64, the type system is preserved.
fn build_bridge_text(
    handle: Handle<Expression>,
    expressions: &Arena<Expression>,
    fi: &FunctionInfo,
    module: &naga::Module,
    consumed: &HashSet<Handle<Expression>>,
) -> String {
    match expressions[handle] {
        Expression::Binary { op, left, right }
            if is_rewritable_op(op) && is_f64_expr(left, fi, module) =>
        {
            let op_name = match op {
                BinaryOperator::Add => "_df64_add_f64",
                BinaryOperator::Subtract => "_df64_sub_f64",
                BinaryOperator::Multiply => "_df64_mul_f64",
                BinaryOperator::Divide => "_df64_div_f64",
                BinaryOperator::Greater => "_df64_gt_f64",
                BinaryOperator::Less => "_df64_lt_f64",
                BinaryOperator::GreaterEqual => "_df64_gte_f64",
                BinaryOperator::LessEqual => "_df64_lte_f64",
                _ => unreachable!("is_rewritable_op checked"),
            };
            let left_text = if consumed.contains(&left) {
                build_bridge_text(left, expressions, fi, module, consumed)
            } else {
                leaf_text(left, expressions)
            };
            let right_text = if consumed.contains(&right) {
                build_bridge_text(right, expressions, fi, module, consumed)
            } else {
                leaf_text(right, expressions)
            };
            format!("{op_name}({left_text}, {right_text})")
        }
        Expression::Unary {
            op: UnaryOperator::Negate,
            expr: inner,
        } if is_f64_expr(inner, fi, module) => {
            let inner_text = if consumed.contains(&inner) {
                build_bridge_text(inner, expressions, fi, module, consumed)
            } else {
                leaf_text(inner, expressions)
            };
            format!("_df64_neg_f64({inner_text})")
        }
        _ => leaf_text(handle, expressions),
    }
}

/// Extract source text for a leaf expression via its naga span.
/// Encodes the span as a `__SPAN__start__end` marker that `resolve_spans`
/// later replaces with actual source text.
fn leaf_text(handle: Handle<Expression>, expressions: &Arena<Expression>) -> String {
    if let Some(span_range) = expressions.get_span(handle).to_range() {
        if span_range.start <= span_range.end {
            return format!("__SPAN__{}__{}", span_range.start, span_range.end);
        }
    }
    // Undefined or invalid span — emit a safe f64 zero that won't break compilation.
    // This is a fallback; real shaders should always have valid spans.
    String::from("f64(0.0)")
}

/// Whether this binary operator should be rewritten for DF64.
fn is_rewritable_op(op: BinaryOperator) -> bool {
    matches!(
        op,
        BinaryOperator::Add
            | BinaryOperator::Subtract
            | BinaryOperator::Multiply
            | BinaryOperator::Divide
            | BinaryOperator::Greater
            | BinaryOperator::Less
            | BinaryOperator::GreaterEqual
            | BinaryOperator::LessEqual
    )
}

/// Check if an expression resolves to f64 type.
fn is_f64_expr(handle: Handle<Expression>, fi: &FunctionInfo, module: &naga::Module) -> bool {
    let ei = &fi[handle];
    match &ei.ty {
        TypeResolution::Value(TypeInner::Scalar(scalar)) => {
            scalar.kind == naga::ScalarKind::Float && scalar.width == 8
        }
        TypeResolution::Handle(th) => {
            matches!(
                module.types[*th].inner,
                TypeInner::Scalar(s) if s.kind == naga::ScalarKind::Float && s.width == 8
            )
        }
        _ => false,
    }
}

/// Post-process the rewritten source: resolve `__SPAN__` markers to actual
/// source text from the original f64 source.
pub(crate) fn resolve_spans(rewritten: &str, original: &str) -> String {
    let mut result = rewritten.to_string();
    // Find all __SPAN__start__end markers and replace with original source text
    while let Some(pos) = result.find("__SPAN__") {
        let rest = &result[pos + 8..];
        if let Some(mid) = rest.find("__") {
            let start_str = &rest[..mid];
            let after_mid = &rest[mid + 2..];
            // Find the end of the end number
            let end_len = after_mid
                .find(|c: char| !c.is_ascii_digit())
                .unwrap_or(after_mid.len());
            let end_str = &after_mid[..end_len];

            if let (Ok(start), Ok(end)) = (start_str.parse::<usize>(), end_str.parse::<usize>()) {
                if start <= end && end <= original.len() {
                    let span_text = &original[start..end];
                    let marker_end = pos + 8 + mid + 2 + end_len;
                    result.replace_range(pos..marker_end, span_text);
                    continue;
                }
            }
        }
        break; // malformed marker, stop
    }
    result
}

/// Full pipeline: parse, rewrite infix ops, resolve spans, prepend bridge functions.
pub fn rewrite_f64_infix_full(f64_source: &str) -> Result<String, String> {
    let rewritten = rewrite_f64_infix_to_df64(f64_source)?;
    if rewritten == *f64_source {
        return Ok(f64_source.to_string());
    }
    let resolved = resolve_spans(&rewritten, f64_source);
    Ok(format!("{DF64_BRIDGE_FUNCTIONS}\n{resolved}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rewrite_and_resolve(wgsl: &str) -> String {
        let rewritten = rewrite_f64_infix_to_df64(wgsl).expect("should parse");

        resolve_spans(&rewritten, wgsl)
    }

    #[test]
    fn test_count_f64_infix_simple_add() {
        let wgsl = r#"
@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let a = input[i];
    let b = input[i + 1u];
    output[i] = a + b;
}
"#;
        let count = count_f64_infix_ops(wgsl).expect("should parse");
        assert!(count >= 1, "expected at least 1 f64 infix op, got {count}");
    }

    #[test]
    fn test_count_f64_infix_no_f64_ops() {
        let wgsl = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    output[i] = input[i] + 1.0;
}
"#;
        let count = count_f64_infix_ops(wgsl).expect("should parse");
        assert_eq!(count, 0, "f32 ops should not be counted");
    }

    #[test]
    fn test_rewrite_simple_add() {
        let wgsl = r#"
@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let a = input[i];
    let b = input[i + 1u];
    output[i] = a + b;
}
"#;
        let result = rewrite_and_resolve(wgsl);
        assert!(
            result.contains("_df64_add_f64("),
            "should contain bridge call, got:\n{result}"
        );
    }

    #[test]
    fn test_rewrite_nested_ops() {
        let wgsl = r#"
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
"#;
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
        let wgsl = r#"
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
"#;
        let result = rewrite_and_resolve(wgsl);
        assert!(
            result.contains("_df64_mul_f64("),
            "f64 mul should be rewritten, got:\n{result}"
        );
    }

    #[test]
    fn test_rewrite_negation() {
        let wgsl = r#"
@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let a = input[i];
    output[i] = -a;
}
"#;
        let result = rewrite_and_resolve(wgsl);
        assert!(
            result.contains("_df64_neg_f64("),
            "negation should be rewritten, got:\n{result}"
        );
    }

    /// End-to-end: full pipeline produces valid WGSL when compiled with df64 core.
    #[test]
    fn test_full_pipeline_validates() {
        let f64_source = r#"
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
"#;
        let rewritten = rewrite_f64_infix_full(f64_source).expect("rewrite");

        const DF64_CORE: &str = include_str!("../../shaders/math/df64_core.wgsl");
        const DF64_TRANS: &str = include_str!("../../shaders/math/df64_transcendentals.wgsl");
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
        let wgsl = r#"
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
"#;
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
        let wgsl = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    output[gid.x] = input[gid.x] + 1.0;
}
"#;
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
        let wgsl = r#"
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
"#;
        let count = count_f64_infix_ops(wgsl).expect("should parse");
        assert!(
            count >= 1,
            "should count f64 add but not u32 ops, got {count}"
        );
    }

    #[test]
    fn test_chaos_naga_subtraction_chain() {
        let wgsl = r#"
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
"#;
        let result = rewrite_and_resolve(wgsl);
        assert!(
            result.contains("_df64_sub_f64("),
            "subtraction chain should use sub bridges, got:\n{result}"
        );
    }

    #[test]
    fn test_chaos_naga_division() {
        let wgsl = r#"
@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    output[i] = input[i] / input[i + 1u];
}
"#;
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
        let wgsl = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    output[gid.x] = input[gid.x] * 2.0;
}
"#;
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
}

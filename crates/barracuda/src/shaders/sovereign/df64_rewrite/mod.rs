// SPDX-License-Identifier: AGPL-3.0-only
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
//! Together: `op_preamble` makes new code portable by design, naga makes
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
use naga::{
    Arena, BinaryOperator, Expression, Handle, LocalVariable, Statement, TypeInner, UnaryOperator,
};
use std::collections::{HashMap, HashSet};

/// Bridge functions that accept f64, compute in Df64, return f64.
/// These get prepended to the shader source after the df64 core library.
/// The GPU compiler inlines these — zero overhead.
const DF64_BRIDGE_FUNCTIONS: &str = r"
fn _df64_add_f64(a: f64, b: f64) -> f64 { return df64_to_f64(df64_add(df64_from_f64(a), df64_from_f64(b))); }
fn _df64_sub_f64(a: f64, b: f64) -> f64 { return df64_to_f64(df64_sub(df64_from_f64(a), df64_from_f64(b))); }
fn _df64_mul_f64(a: f64, b: f64) -> f64 { return df64_to_f64(df64_mul(df64_from_f64(a), df64_from_f64(b))); }
fn _df64_div_f64(a: f64, b: f64) -> f64 { return df64_to_f64(df64_div(df64_from_f64(a), df64_from_f64(b))); }
fn _df64_neg_f64(a: f64) -> f64 { return df64_to_f64(df64_neg(df64_from_f64(a))); }
fn _df64_gt_f64(a: f64, b: f64) -> bool { return df64_gt(df64_from_f64(a), df64_from_f64(b)); }
fn _df64_lt_f64(a: f64, b: f64) -> bool { return df64_lt(df64_from_f64(a), df64_from_f64(b)); }
fn _df64_gte_f64(a: f64, b: f64) -> bool { let da = df64_from_f64(a); let db = df64_from_f64(b); return df64_gt(da, db) || (da.hi == db.hi && da.lo == db.lo); }
fn _df64_lte_f64(a: f64, b: f64) -> bool { let da = df64_from_f64(a); let db = df64_from_f64(b); return df64_lt(da, db) || (da.hi == db.hi && da.lo == db.lo); }
";

struct Replacement {
    span_start: usize,
    span_end: usize,
    text: String,
}

/// Per-function context for the rewriter. Carries named expressions,
/// local variable names, and compound assignment detection results
/// so that `build_bridge_text` can emit correct references.
struct RewriteCtx<'a> {
    expressions: &'a Arena<Expression>,
    fi: &'a FunctionInfo,
    module: &'a naga::Module,
    consumed_by_f64_op: HashSet<Handle<Expression>>,
    /// Expression handle → name from `let` bindings (naga `named_expressions`)
    expr_names: HashMap<Handle<Expression>, String>,
    /// Local variable handle → name for `var` bindings
    local_var_names: HashMap<Handle<LocalVariable>, String>,
    /// Binary expression handles from compound assignments → target variable name.
    /// When `fx += expr` desugars to `Store(ptr, Binary(Add, Load(ptr), expr))`,
    /// the Binary handle maps to `"fx"` so we can emit `fx = bridge(fx, rhs)`.
    compound_targets: HashMap<Handle<Expression>, String>,
}

impl<'a> RewriteCtx<'a> {
    fn from_function(
        func: &'a naga::Function,
        fi: &'a FunctionInfo,
        module: &'a naga::Module,
    ) -> Self {
        let expressions = &func.expressions;

        let expr_names: HashMap<_, _> = func
            .named_expressions
            .iter()
            .map(|(h, name)| (*h, name.clone()))
            .collect();

        let local_var_names: HashMap<_, _> = func
            .local_variables
            .iter()
            .filter_map(|(h, lv)| lv.name.as_ref().map(|n| (h, n.clone())))
            .collect();

        let mut consumed_by_f64_op = HashSet::new();
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

        let mut compound_targets = HashMap::new();
        find_compound_assignments(
            &func.body,
            expressions,
            &local_var_names,
            &mut compound_targets,
        );

        Self {
            expressions,
            fi,
            module,
            consumed_by_f64_op,
            expr_names,
            local_var_names,
            compound_targets,
        }
    }
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
/// # Errors
///
/// Returns [`Err`] if the source fails to parse or validate as f64 WGSL.
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
        let ctx = RewriteCtx::from_function(&ep.function, fi, &module);
        collect_f64_infix_ops(&ctx, &mut replacements);
    }

    for (fh, func) in module.functions.iter() {
        let fi = &info[fh];
        let ctx = RewriteCtx::from_function(func, fi, &module);
        collect_f64_infix_ops(&ctx, &mut replacements);
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
/// Input must be sorted by `span_start` descending.
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
///
/// # Errors
///
/// Returns [`Err`] if the source fails to parse or validate as f64 WGSL.
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

/// Walk a naga [`Block`] to find compound assignments (`x += expr`,
/// `x -= expr`, etc.) and record the Binary expression handle →
/// target variable name mapping.
///
/// Compound assignments desugar in naga as:
///   `Store { pointer, value: Binary(op, Load { pointer }, rhs) }`
/// The Binary's left operand is a `Load` from the same pointer as the
/// Store target. The variable name comes from the `LocalVariable`.
fn find_compound_assignments(
    block: &naga::Block,
    expressions: &Arena<Expression>,
    local_var_names: &HashMap<Handle<LocalVariable>, String>,
    out: &mut HashMap<Handle<Expression>, String>,
) {
    for stmt in block {
        match *stmt {
            Statement::Store { pointer, value } => {
                if let Expression::Binary { left, .. } = expressions[value] {
                    if let Expression::Load {
                        pointer: load_ptr, ..
                    } = expressions[left]
                    {
                        if load_ptr == pointer {
                            if let Expression::LocalVariable(lv) = expressions[pointer] {
                                if let Some(name) = local_var_names.get(&lv) {
                                    out.insert(value, name.clone());
                                }
                            }
                        }
                    }
                }
            }
            Statement::Block(ref inner) => {
                find_compound_assignments(inner, expressions, local_var_names, out);
            }
            Statement::If {
                ref accept,
                ref reject,
                ..
            } => {
                find_compound_assignments(accept, expressions, local_var_names, out);
                find_compound_assignments(reject, expressions, local_var_names, out);
            }
            Statement::Loop {
                ref body,
                ref continuing,
                ..
            } => {
                find_compound_assignments(body, expressions, local_var_names, out);
                find_compound_assignments(continuing, expressions, local_var_names, out);
            }
            Statement::Switch { ref cases, .. } => {
                for case in cases {
                    find_compound_assignments(&case.body, expressions, local_var_names, out);
                }
            }
            _ => {}
        }
    }
}

/// Returns the bridge function definitions that must be prepended when
/// `rewrite_f64_infix_to_df64` has produced replacements.
#[must_use]
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
fn collect_f64_infix_ops(ctx: &RewriteCtx, out: &mut Vec<Replacement>) {
    // Collect root f64 ops (not consumed by another f64 op).
    for (handle, expr) in ctx.expressions.iter() {
        if ctx.consumed_by_f64_op.contains(&handle) {
            continue;
        }

        if !is_f64_op_expr(expr, ctx.fi, ctx.module) {
            continue;
        }

        if let Some(span_range) = ctx.expressions.get_span(handle).to_range() {
            let text = build_bridge_text(handle, ctx);
            out.push(Replacement {
                span_start: span_range.start,
                span_end: span_range.end,
                text,
            });
        }
    }

    // Also collect consumed f64 ops whose span doesn't overlap with any root.
    // These are in separate statements (e.g. `let x = a + b; let y = x * c;`).
    for (handle, expr) in ctx.expressions.iter() {
        if !ctx.consumed_by_f64_op.contains(&handle) {
            continue;
        }

        if !is_f64_op_expr(expr, ctx.fi, ctx.module) {
            continue;
        }

        if let Some(span_range) = ctx.expressions.get_span(handle).to_range() {
            let overlaps = out
                .iter()
                .any(|r| span_range.start < r.span_end && span_range.end > r.span_start);
            if !overlaps {
                let text = build_bridge_text(handle, ctx);
                out.push(Replacement {
                    span_start: span_range.start,
                    span_end: span_range.end,
                    text,
                });
            }
        }
    }
}

fn is_f64_op_expr(expr: &Expression, fi: &FunctionInfo, module: &naga::Module) -> bool {
    match *expr {
        Expression::Binary { op, left, .. } => {
            is_rewritable_op(op) && is_f64_expr(left, fi, module)
        }
        Expression::Unary {
            op: UnaryOperator::Negate,
            expr: inner,
        } => is_f64_expr(inner, fi, module),
        _ => false,
    }
}

/// Build bridge function call text for an f64 infix expression.
///
/// For nested f64 ops within the same expression tree (e.g. `(a+b)*c`),
/// recursively builds: `_df64_mul_f64(_df64_add_f64(a, b), c)`.
///
/// Uses naga `named_expressions` to reference `let` bindings by name
/// instead of expanding their full expression trees. This prevents
/// cross-statement inlining and ensures variable names stay intact.
///
/// For compound assignments (`+=`, `-=`), prefixes the output with
/// `var = ` to preserve the assignment semantics that naga's desugaring
/// would otherwise destroy.
fn build_bridge_text(handle: Handle<Expression>, ctx: &RewriteCtx) -> String {
    match ctx.expressions[handle] {
        Expression::Binary { op, left, right }
            if is_rewritable_op(op) && is_f64_expr(left, ctx.fi, ctx.module) =>
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
                _ => {
                    debug_assert!(false, "is_rewritable_op should prevent reaching here");
                    return format!("/* unhandled f64 op: {op:?} */");
                }
            };
            let left_text = resolve_operand(left, ctx);
            let right_text = resolve_operand(right, ctx);
            let bridge_call = format!("{op_name}({left_text}, {right_text})");

            if let Some(var_name) = ctx.compound_targets.get(&handle) {
                format!("{var_name} = {bridge_call}")
            } else {
                bridge_call
            }
        }
        Expression::Unary {
            op: UnaryOperator::Negate,
            expr: inner,
        } if is_f64_expr(inner, ctx.fi, ctx.module) => {
            let inner_text = resolve_operand(inner, ctx);
            format!("_df64_neg_f64({inner_text})")
        }
        _ => leaf_text(handle, ctx),
    }
}

/// Resolve an operand to its text representation.
///
/// Priority chain:
/// 1. Named expression (`let` binding) → use the name directly
/// 2. Consumed f64 op → recursively build bridge text
/// 3. Leaf → span text or `LocalVariable` name fallback
fn resolve_operand(handle: Handle<Expression>, ctx: &RewriteCtx) -> String {
    if let Some(name) = ctx.expr_names.get(&handle) {
        return name.clone();
    }

    if ctx.consumed_by_f64_op.contains(&handle) {
        return build_bridge_text(handle, ctx);
    }

    leaf_text(handle, ctx)
}

/// Extract source text for a leaf expression.
///
/// Priority chain:
/// 1. Valid naga span → `__SPAN__start__end` marker (resolved later)
/// 2. `Load` of a `LocalVariable` → variable name (handles compound assignments
///    where the implicit Load has no source span)
/// 3. Fallback → `f64(0.0)` (safe default that won't break compilation)
fn leaf_text(handle: Handle<Expression>, ctx: &RewriteCtx) -> String {
    if let Some(span_range) = ctx.expressions.get_span(handle).to_range() {
        if span_range.start < span_range.end {
            return format!("__SPAN__{}__{}", span_range.start, span_range.end);
        }
    }

    if let Expression::Load { pointer, .. } = ctx.expressions[handle] {
        if let Expression::LocalVariable(lv) = ctx.expressions[pointer] {
            if let Some(name) = ctx.local_var_names.get(&lv) {
                return name.clone();
            }
        }
    }

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
///
/// # Errors
///
/// Returns [`Err`] if the source fails to parse or validate as f64 WGSL.
pub fn rewrite_f64_infix_full(f64_source: &str) -> Result<String, String> {
    let rewritten = rewrite_f64_infix_to_df64(f64_source)?;
    if rewritten == *f64_source {
        return Ok(f64_source.to_string());
    }
    let resolved = resolve_spans(&rewritten, f64_source);
    Ok(format!("{DF64_BRIDGE_FUNCTIONS}\n{resolved}"))
}

#[cfg(test)]
mod tests;
#[cfg(test)]
mod tests_nak;

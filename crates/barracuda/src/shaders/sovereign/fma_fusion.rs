// SPDX-License-Identifier: AGPL-3.0-or-later
//! FMA Fusion Pass — Sovereign Compiler Phase 4.
//!
//! Walks a `naga::Arena<Expression>` and detects `Mul(a,b)` whose result
//! feeds into exactly one `Add(mul_result, c)` or `Sub(mul_result, c)`.
//! Replaces the `Add`/`Sub` with `Math { fun: Fma, arg: a, arg1: b, arg2: c }`
//! (or the negated variant for subtraction).
//!
//! ## Why at IR level
//!
//! NAK Deficiency 4 documents that NAK emits `DMUL + DADD` instead of a
//! single `DFMA` for patterns like `a * b + c`. By fusing at the naga IR
//! level, we guarantee `OpFma` appears in SPIR-V regardless of which
//! backend compiles it.
//!
//! ## Precision note
//!
//! IEEE 754: `fma(a, b, c)` has ONE rounding step; `a*b + c` has TWO.
//! This pass only fuses patterns where the intermediate multiply result
//! has no other consumers — semantically equivalent to what a hardware
//! FMA unit would compute, and strictly more precise than the two-op
//! version.

use naga::{Arena, BinaryOperator, Expression, Handle, MathFunction};

/// Build a flat boolean map marking expressions whose resolved type is f64.
///
/// WGSL `fma()` is defined for f32/f16/AbstractFloat only — NOT f64.
/// Fusing f64 multiply-add patterns into `Fma` produces WGSL that silently
/// returns 0 on all GPU backends. This map lets the fusion pass skip them.
///
/// The naga arena is topologically sorted (expressions only reference earlier
/// handles), so a single forward pass propagates f64-ness correctly.
fn mark_f64_expressions(expressions: &Arena<Expression>) -> Vec<bool> {
    let mut is_f64 = vec![false; expressions.len()];
    for (handle, expr) in expressions.iter() {
        is_f64[handle.index()] = match *expr {
            Expression::Literal(naga::Literal::F64(_)) => true,
            Expression::As {
                kind: naga::ScalarKind::Float,
                convert: Some(8),
                ..
            } => true,
            Expression::Binary { left, right, .. } => is_f64[left.index()] || is_f64[right.index()],
            Expression::Math { arg, .. } => is_f64[arg.index()],
            Expression::Unary { expr, .. } => is_f64[expr.index()],
            Expression::Select { accept, reject, .. } => {
                is_f64[accept.index()] || is_f64[reject.index()]
            }
            _ => false,
        };
    }
    is_f64
}

/// Fuse `Mul + Add/Sub` patterns into `Fma` in the given expression arena.
///
/// Skips f64 expressions: WGSL `fma()` is f32/f16 only.
///
/// Returns the number of fusions performed.
pub fn fuse_multiply_add(expressions: &mut Arena<Expression>) -> usize {
    let len = expressions.len();
    if len < 2 {
        return 0;
    }

    // Phase 0: Mark f64 expressions so we can skip them during fusion.
    let is_f64 = mark_f64_expressions(expressions);

    // Phase 1: Count how many times each expression handle is referenced
    // as an operand. We only fuse Mul results consumed exactly once.
    let mut ref_counts = vec![0u32; len];
    for (_handle, expr) in expressions.iter() {
        visit_operands(expr, |h| {
            ref_counts[h.index()] += 1;
        });
    }

    // Phase 2: Scan for Add/Sub whose left or right operand is a Mul
    // with ref_count == 1. Collect replacements.
    let replacements: Vec<(Handle<Expression>, Expression)> = expressions
        .iter()
        .filter_map(|(handle, expr)| {
            match *expr {
                // Pattern: Mul(a,b) + c  or  c + Mul(a,b)
                Expression::Binary {
                    op: BinaryOperator::Add,
                    left,
                    right,
                } => try_fuse_add(expressions, left, right, &ref_counts, &is_f64)
                    .or_else(|| try_fuse_add(expressions, right, left, &ref_counts, &is_f64))
                    .map(|fma| (handle, fma)),

                // Pattern: Mul(a,b) - c  →  fma(a, b, -c)
                // We express this as fma(a, b, c) and negate c via Unary(Negate).
                // However, introducing a new expression into the arena at the
                // wrong position would violate naga's ordering invariant.
                // We handle the subtracted-Mul case: c - Mul(a,b) → fma(-a, b, c)
                // requires a negate node. Currently supported pattern:
                //   Mul(a,b) - c  → fma(a, b, -c) when -c already exists as
                //   Unary(Negate, c) in the expression tree. The symmetric
                //   case (c - Mul) is a P3 naga IR optimization.
                //
                // For the common Jacobi pattern `c * akp - s * akq`:
                //   Binary(Sub, Binary(Mul, c, akp), Binary(Mul, s, akq))
                // Neither side's Mul has ref_count 1 if both are used elsewhere,
                // so this correctly skips non-fusible patterns.
                Expression::Binary {
                    op: BinaryOperator::Subtract,
                    left: mul_candidate,
                    right: addend,
                } => try_fuse_sub_left(expressions, mul_candidate, addend, &ref_counts, &is_f64)
                    .map(|fma| (handle, fma)),

                _ => None,
            }
        })
        .collect();

    let count = replacements.len();
    for (handle, replacement) in replacements {
        expressions[handle] = replacement;
    }
    count
}

/// Try to fuse `mul_candidate + addend` where `mul_candidate` is `Mul(a,b)`.
fn try_fuse_add(
    expressions: &Arena<Expression>,
    mul_candidate: Handle<Expression>,
    addend: Handle<Expression>,
    ref_counts: &[u32],
    is_f64: &[bool],
) -> Option<Expression> {
    if ref_counts[mul_candidate.index()] != 1 {
        return None;
    }
    if is_f64[mul_candidate.index()] {
        return None;
    }
    match expressions[mul_candidate] {
        Expression::Binary {
            op: BinaryOperator::Multiply,
            left: a,
            right: b,
        } => Some(Expression::Math {
            fun: MathFunction::Fma,
            arg: a,
            arg1: Some(b),
            arg2: Some(addend),
            arg3: None,
        }),
        _ => None,
    }
}

/// Try to fuse `Mul(a,b) - c` → `fma(a, b, -c)`.
///
/// We can only do this if `-c` already exists as a `Unary(Negate, c)` node
/// earlier in the arena (we cannot insert new nodes without violating ordering).
/// If it doesn't exist, we skip the fusion.
fn try_fuse_sub_left(
    expressions: &Arena<Expression>,
    mul_candidate: Handle<Expression>,
    subtrahend: Handle<Expression>,
    ref_counts: &[u32],
    is_f64: &[bool],
) -> Option<Expression> {
    if ref_counts[mul_candidate.index()] != 1 {
        return None;
    }
    if is_f64[mul_candidate.index()] {
        return None;
    }
    match expressions[mul_candidate] {
        Expression::Binary {
            op: BinaryOperator::Multiply,
            left: a,
            right: b,
        } => {
            // Look for an existing Unary(Negate, subtrahend) in the arena.
            let neg_handle = find_negate(expressions, subtrahend)?;
            Some(Expression::Math {
                fun: MathFunction::Fma,
                arg: a,
                arg1: Some(b),
                arg2: Some(neg_handle),
                arg3: None,
            })
        }
        _ => None,
    }
}

/// Find an existing `Unary(Negate, target)` expression in the arena.
fn find_negate(
    expressions: &Arena<Expression>,
    target: Handle<Expression>,
) -> Option<Handle<Expression>> {
    for (handle, expr) in expressions.iter() {
        if let Expression::Unary {
            op: naga::UnaryOperator::Negate,
            expr: inner,
        } = *expr
        {
            if inner == target {
                return Some(handle);
            }
        }
    }
    None
}

/// Visit all `Handle<Expression>` operands of an expression.
pub(crate) fn visit_operands<F: FnMut(Handle<Expression>)>(expr: &Expression, mut f: F) {
    match *expr {
        Expression::Access { base, index } => {
            f(base);
            f(index);
        }
        Expression::AccessIndex { base, .. } => f(base),
        Expression::Splat { value, .. } => f(value),
        Expression::Swizzle { vector, .. } => f(vector),
        Expression::Compose { ref components, .. } => {
            for &c in components {
                f(c);
            }
        }
        Expression::Load { pointer } => f(pointer),
        Expression::Unary { expr, .. } => f(expr),
        Expression::Binary { left, right, .. } => {
            f(left);
            f(right);
        }
        Expression::Select {
            condition,
            accept,
            reject,
        } => {
            f(condition);
            f(accept);
            f(reject);
        }
        Expression::Derivative { expr, .. } => f(expr),
        Expression::Relational { argument, .. } => f(argument),
        Expression::Math {
            arg,
            arg1,
            arg2,
            arg3,
            ..
        } => {
            f(arg);
            if let Some(a) = arg1 {
                f(a);
            }
            if let Some(a) = arg2 {
                f(a);
            }
            if let Some(a) = arg3 {
                f(a);
            }
        }
        Expression::As { expr, .. } => f(expr),
        Expression::ArrayLength(expr) => f(expr),
        Expression::ImageSample {
            image,
            sampler,
            coordinate,
            array_index,
            offset,
            depth_ref,
            ..
        } => {
            f(image);
            f(sampler);
            f(coordinate);
            if let Some(a) = array_index {
                f(a);
            }
            if let Some(o) = offset {
                f(o);
            }
            if let Some(d) = depth_ref {
                f(d);
            }
        }
        Expression::ImageLoad {
            image,
            coordinate,
            array_index,
            sample,
            level,
        } => {
            f(image);
            f(coordinate);
            if let Some(a) = array_index {
                f(a);
            }
            if let Some(s) = sample {
                f(s);
            }
            if let Some(l) = level {
                f(l);
            }
        }
        Expression::ImageQuery { image, .. } => f(image),
        Expression::RayQueryGetIntersection { query, .. } => f(query),
        Expression::RayQueryVertexPositions { .. } => {}
        // Leaf expressions — no operands
        Expression::Literal(_)
        | Expression::Constant(_)
        | Expression::Override(_)
        | Expression::ZeroValue(_)
        | Expression::FunctionArgument(_)
        | Expression::GlobalVariable(_)
        | Expression::LocalVariable(_)
        | Expression::CallResult(_)
        | Expression::AtomicResult { .. }
        | Expression::WorkGroupUniformLoadResult { .. }
        | Expression::RayQueryProceedResult
        | Expression::SubgroupBallotResult
        | Expression::SubgroupOperationResult { .. } => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fma_fusion_on_add_pattern() {
        let wgsl = r"
@group(0) @binding(0) var<storage, read> a_buf: array<f32>;
@group(0) @binding(1) var<storage, read> b_buf: array<f32>;
@group(0) @binding(2) var<storage, read> c_buf: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let a = a_buf[i];
    let b = b_buf[i];
    let c = c_buf[i];
    let product = a * b;
    let result = product + c;
    out[i] = result;
}
";
        let mut module = naga::front::wgsl::parse_str(wgsl).expect("parse");
        let mut total_fusions = 0usize;
        for (_h, func) in module.functions.iter_mut() {
            total_fusions += fuse_multiply_add(&mut func.expressions);
        }
        for ep in &mut module.entry_points {
            total_fusions += fuse_multiply_add(&mut ep.function.expressions);
        }
        assert!(
            total_fusions >= 1,
            "expected at least 1 FMA fusion, got {total_fusions}"
        );

        // Validate that the fused module is still valid
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        validator
            .validate(&module)
            .expect("fused module should validate");
    }

    #[test]
    fn test_no_fusion_when_mul_has_multiple_consumers() {
        let wgsl = r"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let a = input[i];
    let b = input[i + 1u];
    let c = input[i + 2u];
    let product = a * b;
    let r1 = product + c;
    let r2 = product + 1.0;
    out[i] = r1 + r2;
}
";
        let mut module = naga::front::wgsl::parse_str(wgsl).expect("parse");
        let mut total_fusions = 0usize;
        for (_h, func) in module.functions.iter_mut() {
            total_fusions += fuse_multiply_add(&mut func.expressions);
        }
        for ep in &mut module.entry_points {
            total_fusions += fuse_multiply_add(&mut ep.function.expressions);
        }
        // product has 2 consumers, so no fusion should happen
        assert_eq!(
            total_fusions, 0,
            "should not fuse when mul has multiple consumers"
        );
    }

    #[test]
    fn test_no_fusion_on_f64_multiply_add() {
        let wgsl = r"
@group(0) @binding(0) var<storage, read_write> out: array<f64>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x: f64 = f64(3.0);
    let y: f64 = x * f64(2.0) + f64(1.0);
    out[gid.x] = y;
}
";
        let mut module = naga::front::wgsl::parse_str(wgsl).expect("parse");
        let mut total_fusions = 0usize;
        for (_h, func) in module.functions.iter_mut() {
            total_fusions += fuse_multiply_add(&mut func.expressions);
        }
        for ep in &mut module.entry_points {
            total_fusions += fuse_multiply_add(&mut ep.function.expressions);
        }
        assert_eq!(
            total_fusions, 0,
            "must not fuse f64 multiply-add: WGSL fma() is f32/f16 only"
        );
    }
}

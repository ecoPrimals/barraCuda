// SPDX-License-Identifier: AGPL-3.0-or-later
//! Dead Expression Elimination — Sovereign Compiler Phase 4.
//!
//! After FMA fusion, some intermediate `Mul` expressions may become unused
//! (their only consumer was the `Add` that got fused into `Fma`).
//!
//! naga's SPIR-V backend naturally skips un-emitted expressions, but leaving
//! dead nodes in the arena can inflate register pressure estimates and slow
//! validation. This pass zeroes out dead expressions by replacing them with
//! `Literal(I32(0))` — a minimal leaf node that satisfies arena ordering
//! invariants.
//!
//! ## Approach
//!
//! 1. Walk all statements in the function body to find expression handles
//!    that are "roots" (directly used by statements like `Store`, `Call`, etc.)
//! 2. From those roots, transitively mark all referenced expressions as live.
//! 3. Replace un-marked expressions with a zero literal.

use naga::{Arena, Expression, Handle, Literal, Statement};

/// Eliminate dead expressions in the arena.
///
/// Returns the number of expressions eliminated.
pub fn eliminate(expressions: &mut Arena<Expression>, body: &naga::Block) -> usize {
    let len = expressions.len();
    if len == 0 {
        return 0;
    }

    let mut live = vec![false; len];

    // Mark roots from statements.
    mark_roots_from_block(body, &mut live);

    // Transitively mark all expressions reachable from roots.
    // Process in reverse order (high handles first) to propagate through
    // the DAG in a single pass — naga guarantees that every expression
    // only references handles with lower indices.
    let handles: Vec<Handle<Expression>> = expressions.iter().map(|(h, _)| h).collect();
    for &handle in handles.iter().rev() {
        let idx = handle.index();
        if !live[idx] {
            continue;
        }
        let expr = &expressions[handle];
        super::fma_fusion::visit_operands(expr, |dep| {
            live[dep.index()] = true;
        });
    }

    // Also keep expressions that need_pre_emit (naga requirement).
    for (handle, expr) in expressions.iter() {
        if expr.needs_pre_emit() {
            live[handle.index()] = true;
        }
    }

    // Replace dead expressions with zero literals.
    let mut eliminated = 0usize;
    for handle in &handles {
        let idx = handle.index();
        if live[idx] {
            continue;
        }
        let expr = &expressions[*handle];
        if matches!(expr, Expression::Literal(Literal::I32(0))) {
            continue;
        }
        expressions[*handle] = Expression::Literal(Literal::I32(0));
        eliminated += 1;
    }

    eliminated
}

/// Recursively mark expression handles referenced by statements as live.
fn mark_roots_from_block(block: &naga::Block, live: &mut [bool]) {
    for stmt in block.iter() {
        mark_roots_from_statement(stmt, live);
    }
}

fn mark_roots_from_statement(stmt: &Statement, live: &mut [bool]) {
    match *stmt {
        Statement::Emit(ref range) => {
            for handle in range.clone() {
                live[handle.index()] = true;
            }
        }
        Statement::Block(ref block) => mark_roots_from_block(block, live),
        Statement::If {
            condition,
            ref accept,
            ref reject,
        } => {
            live[condition.index()] = true;
            mark_roots_from_block(accept, live);
            mark_roots_from_block(reject, live);
        }
        Statement::Switch {
            selector,
            ref cases,
        } => {
            live[selector.index()] = true;
            for case in cases {
                mark_roots_from_block(&case.body, live);
            }
        }
        Statement::Loop {
            ref body,
            ref continuing,
            break_if,
        } => {
            mark_roots_from_block(body, live);
            mark_roots_from_block(continuing, live);
            if let Some(cond) = break_if {
                live[cond.index()] = true;
            }
        }
        Statement::Return { value } => {
            if let Some(v) = value {
                live[v.index()] = true;
            }
        }
        Statement::Store { pointer, value } => {
            live[pointer.index()] = true;
            live[value.index()] = true;
        }
        Statement::ImageStore {
            image,
            coordinate,
            array_index,
            value,
        } => {
            live[image.index()] = true;
            live[coordinate.index()] = true;
            if let Some(ai) = array_index {
                live[ai.index()] = true;
            }
            live[value.index()] = true;
        }
        Statement::Atomic {
            pointer,
            ref fun,
            value,
            result,
        } => {
            live[pointer.index()] = true;
            live[value.index()] = true;
            if let Some(r) = result {
                live[r.index()] = true;
            }
            if let naga::AtomicFunction::Exchange { compare: Some(c) } = *fun {
                live[c.index()] = true;
            }
        }
        Statement::WorkGroupUniformLoad { pointer, result } => {
            live[pointer.index()] = true;
            live[result.index()] = true;
        }
        Statement::Call {
            ref arguments,
            result,
            ..
        } => {
            for &arg in arguments {
                live[arg.index()] = true;
            }
            if let Some(r) = result {
                live[r.index()] = true;
            }
        }
        Statement::RayQuery { query, ref fun } => {
            live[query.index()] = true;
            match *fun {
                naga::RayQueryFunction::Initialize {
                    acceleration_structure,
                    descriptor,
                } => {
                    live[acceleration_structure.index()] = true;
                    live[descriptor.index()] = true;
                }
                naga::RayQueryFunction::Proceed { result } => {
                    live[result.index()] = true;
                }
                naga::RayQueryFunction::Terminate => {}
            }
        }
        Statement::SubgroupBallot { result, predicate } => {
            if let Some(p) = predicate {
                live[p.index()] = true;
            }
            live[result.index()] = true;
        }
        Statement::SubgroupCollectiveOperation {
            argument, result, ..
        } => {
            live[argument.index()] = true;
            live[result.index()] = true;
        }
        Statement::SubgroupGather {
            argument,
            result,
            ref mode,
            ..
        } => {
            live[argument.index()] = true;
            live[result.index()] = true;
            match *mode {
                naga::GatherMode::BroadcastFirst => {}
                naga::GatherMode::Broadcast(h)
                | naga::GatherMode::Shuffle(h)
                | naga::GatherMode::ShuffleDown(h)
                | naga::GatherMode::ShuffleUp(h)
                | naga::GatherMode::ShuffleXor(h) => {
                    live[h.index()] = true;
                }
            }
        }
        Statement::Break | Statement::Continue | Statement::Kill | Statement::Barrier(_) => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dead_expr_elimination_preserves_valid_module() {
        let wgsl = r#"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let a = input[idx];
    let b = a * 2.0;
    output[idx] = b;
}
"#;
        let mut module = naga::front::wgsl::parse_str(wgsl).expect("parse");
        for ep in &mut module.entry_points {
            let _ = eliminate(&mut ep.function.expressions, &ep.function.body);
        }
        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        );
        validator
            .validate(&module)
            .expect("module should remain valid after dead expr elimination");
    }
}

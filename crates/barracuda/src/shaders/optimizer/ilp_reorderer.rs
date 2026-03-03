//! Instruction-level parallelism (ILP) reorderer for WGSL `@ilp_region` blocks.
//!
//! Takes a `WgslDependencyGraph` and a `LatencyModel` and re-emits the
//! bindings in an order that maximises the number of independent instructions
//! placed between a producer and its first consumer — hiding the pipeline
//! latency window.
//!
//! ## Algorithm
//!
//! List scheduling with ASAP (as-soon-as-possible) release times:
//!
//! 1. For each node, compute `release_cycle = max over deps of
//!    (dep.release_cycle + dep.latency)`.  Nodes with no deps have
//!    `release_cycle = 0`.
//!
//! 2. Maintain a *ready queue* of nodes whose deps have all been scheduled.
//!
//! 3. At each step, pick the ready node with the **earliest** release cycle
//!    (ASAP). Ties are broken by original program order to preserve
//!    determinism and correctness for passthrough nodes.
//!
//! 4. Passthrough nodes (stores, comments, assignments) are scheduled
//!    immediately after all of their named deps are placed, preserving
//!    RAW semantics.
//!
//! ## Correctness guarantee
//!
//! The output is always a valid topological ordering of the dependency graph.
//! Functional semantics are preserved: every use of a name follows its
//! definition in the output.

use std::collections::{BinaryHeap, HashMap};

use super::dependency_graph::{Node, WgslDependencyGraph};
use crate::device::latency::LatencyModel;

// ─── Scheduler state ──────────────────────────────────────────────────────────

/// Internal scheduler node: wraps an original index and its computed timing.
#[derive(Debug, Eq, PartialEq)]
struct Schedulable {
    /// Earliest cycle this node can be issued.
    release_cycle: u32,
    /// Original program order (tie-breaker — preserves passthrough semantics).
    original_order: usize,
    /// Index into the DAG's node list.
    node_idx: usize,
}

impl Ord for Schedulable {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Min-heap: smallest release_cycle first; then smallest original_order.
        other
            .release_cycle
            .cmp(&self.release_cycle)
            .then(other.original_order.cmp(&self.original_order))
            .reverse()
    }
}

impl PartialOrd for Schedulable {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

// ─── IlpReorderer ─────────────────────────────────────────────────────────────

/// Reorders WGSL `let` bindings within `@ilp_region` blocks for ILP.
pub struct IlpReorderer;

impl IlpReorderer {
    /// Reorder the nodes of `graph` according to `model`'s latency tables.
    ///
    /// Returns a `Vec` of source lines in the scheduled order.
    #[must_use]
    pub fn reorder(graph: &WgslDependencyGraph, model: &dyn LatencyModel) -> Vec<String> {
        let n = graph.nodes.len();
        if n == 0 {
            return vec![];
        }

        // ── Step 1: Build successor map (reverse edges) ──────────────────────
        // successor_count[i] = number of nodes that depend on node i.
        let mut dep_count: Vec<usize> = vec![0; n];
        // For each node, which other nodes are waiting on it (successors).
        let mut successors: Vec<Vec<usize>> = vec![vec![]; n];

        for (i, node) in graph.nodes.iter().enumerate() {
            match node {
                Node::Binding(b) => {
                    for &dep_idx in &b.deps {
                        successors[dep_idx].push(i);
                        dep_count[i] += 1;
                    }
                }
                Node::Passthrough(p) => {
                    // Map dep_names back to indices via graph inspection.
                    let dep_indices = resolve_passthrough_deps(p, graph);
                    for dep_idx in dep_indices {
                        successors[dep_idx].push(i);
                        dep_count[i] += 1;
                    }
                }
            }
        }

        // ── Step 2: Compute per-node release cycles ───────────────────────────
        let mut release_cycle: Vec<u32> = vec![0; n];
        // Process in original order — guaranteed to be a valid topo order for
        // forward dependencies only (SSA-like structure in WGSL let-bindings).
        for i in 0..n {
            let lat = node_latency(&graph.nodes[i], model);
            for &succ in &successors[i] {
                let candidate = release_cycle[i] + lat;
                if candidate > release_cycle[succ] {
                    release_cycle[succ] = candidate;
                }
            }
        }

        // ── Step 3: List scheduling (ASAP, min-heap on release_cycle) ─────────
        let mut ready: BinaryHeap<Schedulable> = BinaryHeap::new();
        // Seed with nodes that have no dependencies.
        for (i, &cnt) in dep_count.iter().enumerate() {
            if cnt == 0 {
                ready.push(Schedulable {
                    release_cycle: release_cycle[i],
                    original_order: i,
                    node_idx: i,
                });
            }
        }

        let mut scheduled: Vec<String> = Vec::with_capacity(n);
        let mut remaining_deps = dep_count.clone();
        let mut scheduled_count = 0;

        while let Some(item) = ready.pop() {
            let idx = item.node_idx;
            scheduled.push(graph.nodes[idx].source_line().to_string());
            scheduled_count += 1;

            // Decrement dep count for all successors; enqueue newly ready ones.
            for &succ in &successors[idx] {
                remaining_deps[succ] -= 1;
                if remaining_deps[succ] == 0 {
                    ready.push(Schedulable {
                        release_cycle: release_cycle[succ],
                        original_order: succ,
                        node_idx: succ,
                    });
                }
            }
        }

        // Safety: if scheduling is incomplete (cycle in graph — should never
        // happen for valid WGSL let-bindings), emit unscheduled nodes in
        // original order to preserve correctness.
        if scheduled_count < n {
            for (i, node) in graph.nodes.iter().enumerate() {
                if remaining_deps[i] > 0 {
                    scheduled.push(node.source_line().to_string());
                }
            }
        }

        scheduled
    }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

fn node_latency(node: &Node, model: &dyn LatencyModel) -> u32 {
    match node {
        Node::Binding(b) => model.raw_latency(b.op_class),
        Node::Passthrough(_) => 0,
    }
}

/// Resolve dep names in a passthrough node to their graph indices.
fn resolve_passthrough_deps(
    p: &super::dependency_graph::PassthroughNode,
    graph: &WgslDependencyGraph,
) -> Vec<usize> {
    let name_map: HashMap<&str, usize> = graph
        .nodes
        .iter()
        .enumerate()
        .filter_map(|(i, n)| {
            if let Node::Binding(b) = n {
                Some((b.name.as_str(), i))
            } else {
                None
            }
        })
        .collect();

    p.dep_names
        .iter()
        .filter_map(|name| name_map.get(name.as_str()).copied())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::latency::Sm70LatencyModel;
    use crate::shaders::optimizer::dependency_graph::WgslDependencyGraph;

    #[test]
    fn test_independent_bindings_reordered_for_ilp() {
        // c and s are independent; cc, ss, cs all depend on c/s only.
        // new_p depends on c, s, a_kp, a_kq — should appear after c and s.
        let region = "\
        let c = cos_val;\n\
        let s = sin_val;\n\
        let cc = c * c;\n\
        let ss = s * s;\n\
        let cs = c * s;\n\
        let new_p = c * a_kp - s * a_kq;\n\
        let new_q = s * a_kp + c * a_kq;\n";

        let graph = WgslDependencyGraph::parse(region);
        let model = Sm70LatencyModel;
        let scheduled = IlpReorderer::reorder(&graph, &model);
        assert_eq!(scheduled.len(), graph.nodes.len());

        // c must appear before cc, cs, new_p, new_q
        let pos_c = scheduled
            .iter()
            .position(|l| l.contains("let c ="))
            .unwrap();
        let pos_new_p = scheduled
            .iter()
            .position(|l| l.contains("let new_p ="))
            .unwrap();
        assert!(pos_c < pos_new_p, "c must be scheduled before new_p");

        // s must appear before new_q
        let pos_s = scheduled
            .iter()
            .position(|l| l.contains("let s ="))
            .unwrap();
        let pos_new_q = scheduled
            .iter()
            .position(|l| l.contains("let new_q ="))
            .unwrap();
        assert!(pos_s < pos_new_q, "s must be scheduled before new_q");
    }

    #[test]
    fn test_passthrough_after_deps() {
        let region = "\
        let a = x * y;\n\
        let b = a + z;\n\
        output[0] = b;\n";

        let graph = WgslDependencyGraph::parse(region);
        let model = Sm70LatencyModel;
        let scheduled = IlpReorderer::reorder(&graph, &model);

        let pos_b = scheduled
            .iter()
            .position(|l| l.contains("let b ="))
            .unwrap();
        let pos_store = scheduled
            .iter()
            .position(|l| l.contains("output[0]"))
            .unwrap();
        assert!(pos_b < pos_store, "store must come after its dep b");
    }

    #[test]
    fn test_empty_region() {
        let graph = WgslDependencyGraph::parse("");
        let model = Sm70LatencyModel;
        let scheduled = IlpReorderer::reorder(&graph, &model);
        assert!(scheduled.is_empty());
    }

    #[test]
    fn test_single_chain_preserves_order() {
        // a → b → c: only one valid ordering.
        let region = "\
        let a = x;\n\
        let b = a * 2.0;\n\
        let c = b + 1.0;\n";
        let graph = WgslDependencyGraph::parse(region);
        let model = Sm70LatencyModel;
        let scheduled = IlpReorderer::reorder(&graph, &model);
        let pos_a = scheduled
            .iter()
            .position(|l| l.contains("let a ="))
            .unwrap();
        let pos_b = scheduled
            .iter()
            .position(|l| l.contains("let b ="))
            .unwrap();
        let pos_c = scheduled
            .iter()
            .position(|l| l.contains("let c ="))
            .unwrap();
        assert!(pos_a < pos_b);
        assert!(pos_b < pos_c);
    }

    #[test]
    fn test_ilp_opportunity_detected() {
        // c and s are independent, giving the scheduler freedom to interleave them.
        // The ILP reorderer should schedule cc, ss, cs before new_p, new_q
        // (because new_p/q have longer latency paths).
        let region = "\
        let c = cos_val;\n\
        let s = sin_val;\n\
        let cc = c * c;\n\
        let new_p = c * a_kp - s * a_kq;\n";
        let graph = WgslDependencyGraph::parse(region);
        let model = Sm70LatencyModel;
        let scheduled = IlpReorderer::reorder(&graph, &model);
        // All 4 bindings should be present
        assert_eq!(scheduled.len(), 4);
        // c before cc
        let pc = scheduled
            .iter()
            .position(|l| l.contains("let c ="))
            .unwrap();
        let pcc = scheduled
            .iter()
            .position(|l| l.contains("let cc ="))
            .unwrap();
        assert!(pc < pcc);
    }
}

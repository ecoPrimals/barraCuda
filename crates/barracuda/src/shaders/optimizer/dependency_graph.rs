// SPDX-License-Identifier: AGPL-3.0-only
//! WGSL `let`-binding dependency graph for ILP analysis.
//!
//! Parses the straight-line `let name = expr;` sequence inside an
//! `// @ilp_region begin … // @ilp_region end` annotated WGSL block and
//! builds a directed acyclic graph (DAG) of data dependencies.
//!
//! ## Scope
//!
//! This is **not** a general WGSL parser. It handles the 80% case:
//! sequences of `let` bindings in straight-line code. Branches, loops, and
//! assignments are passed through unmodified (outside `@ilp_region`).
//!
//! ## Grammar subset handled
//!
//! ```wgsl
//! // @ilp_region begin
//! let foo = bar * baz;
//! let qux = foo + quux;
//! // @ilp_region end
//! ```
//!
//! All other statement forms (var, assignments, function calls not in a let)
//! are treated as "passthrough" nodes with no outgoing dependencies.

use crate::device::latency::{LatencyModel, WgslOpClass};

// ─── Op classification ────────────────────────────────────────────────────────

/// Classify the dominant operation in a `let` binding RHS expression.
///
/// This is a lightweight heuristic, not a full type-aware analysis.
/// It returns the op class of the *slowest* (highest latency) operation
/// visible in the expression string.
#[must_use]
pub fn classify_op(expr: &str) -> WgslOpClass {
    // Global memory loads are always the most expensive — detect pointer dereferences
    // and array indexing that references storage buffers.
    if expr.contains("_batch[") || expr.contains("_buffer[") || expr.contains("global[") {
        return WgslOpClass::GlobalMem;
    }
    // f64 transcendentals in BarraCuda are recognised by their _f64 suffix
    if expr.contains("sqrt(")
        || expr.contains("exp_f64(")
        || expr.contains("log_f64(")
        || expr.contains("cos_f64(")
        || expr.contains("sin_f64(")
    {
        return WgslOpClass::F64Transcend;
    }
    // FP64 FMA: any expression that involves f64 arithmetic operators with
    // f64-typed operands. Heuristic: contains "f64(" cast or involves a
    // previously named f64 binding. For ordering purposes, treat any
    // multiply-add pattern as F64Fma.
    if expr.contains("f64(")
        || (expr.contains(" * ") && (expr.contains(" + ") || expr.contains(" - ")))
    {
        return WgslOpClass::F64Fma;
    }
    if expr.contains(" * ") || expr.contains(" / ") {
        return WgslOpClass::F64MulAdd;
    }
    // Integer / pointer arithmetic for index expressions
    if expr.contains("idx2d(")
        || expr.contains("batch_offset(")
        || expr.contains(" * n")
        || expr.contains("+ idx")
    {
        return WgslOpClass::I32Arith;
    }
    WgslOpClass::F32Fma
}

// ─── Binding node ─────────────────────────────────────────────────────────────

/// A single `let` binding node in the dependency graph.
#[derive(Debug, Clone)]
pub struct BindingNode {
    /// The binding name (left-hand side of `let name = expr;`).
    pub name: String,
    /// The full RHS expression.
    pub expr: String,
    /// The raw source line (including indentation and semicolon).
    pub source_line: String,
    /// Op class of the dominant operation — drives latency lookup.
    pub op_class: WgslOpClass,
    /// Indices into `WgslDependencyGraph::nodes` of direct data dependencies.
    pub deps: Vec<usize>,
}

impl BindingNode {
    /// Number of cycles this node takes before its result is available.
    pub fn latency(&self, model: &dyn LatencyModel) -> u32 {
        model.raw_latency(self.op_class)
    }
}

// ─── Passthrough node ─────────────────────────────────────────────────────────

/// Non-`let` statement that must be preserved in order (stores, assignments,
/// control flow, comments, blank lines).
#[derive(Debug, Clone)]
pub struct PassthroughNode {
    /// The raw source line.
    pub source_line: String,
    /// If this store references a named binding, record the dep so the
    /// scheduler knows the store must come after the producer.
    pub dep_names: Vec<String>,
}

// ─── Node kind ────────────────────────────────────────────────────────────────

/// Node in the dependency graph: either a let binding or passthrough statement.
#[derive(Debug, Clone)]
pub enum Node {
    /// A `let name = expr;` binding.
    Binding(BindingNode),
    /// A non-let statement (store, assignment, control flow).
    Passthrough(PassthroughNode),
}

impl Node {
    /// Return the raw source line for this node.
    #[must_use]
    pub fn source_line(&self) -> &str {
        match self {
            Node::Binding(b) => &b.source_line,
            Node::Passthrough(p) => &p.source_line,
        }
    }
}

// ─── Dependency graph ─────────────────────────────────────────────────────────

/// Directed acyclic graph of `let` bindings and passthrough statements within
/// an `@ilp_region` block.
///
/// Edges represent data dependencies: a binding node points to all other nodes
/// whose results it consumes in its RHS expression.
#[derive(Debug)]
pub struct WgslDependencyGraph {
    /// All nodes (bindings and passthroughs) in parse order.
    pub nodes: Vec<Node>,
}

impl WgslDependencyGraph {
    /// Parse a WGSL `@ilp_region` block into a dependency graph.
    ///
    /// `region` should be the *interior* of the block — i.e. the lines between
    /// `// @ilp_region begin` and `// @ilp_region end`, exclusive.
    #[must_use]
    pub fn parse(region: &str) -> Self {
        let mut nodes: Vec<Node> = Vec::new();
        // Map of binding name → index in `nodes` for dep resolution.
        let mut name_to_idx: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();

        for line in region.lines() {
            let trimmed = line.trim();

            // Skip blank lines and pure comment lines — passthrough, no deps.
            if trimmed.is_empty() || trimmed.starts_with("//") {
                nodes.push(Node::Passthrough(PassthroughNode {
                    source_line: line.to_string(),
                    dep_names: vec![],
                }));
                continue;
            }

            // Try to parse as `let name = expr;`
            if let Some(binding) = parse_let_binding(line, trimmed, &name_to_idx, &nodes) {
                let idx = nodes.len();
                name_to_idx.insert(binding.name.clone(), idx);
                nodes.push(Node::Binding(binding));
            } else {
                // Non-let statement — passthrough; track which bindings it references
                // so the scheduler respects RAW order.
                let dep_names = find_referenced_names(trimmed, &name_to_idx);
                nodes.push(Node::Passthrough(PassthroughNode {
                    source_line: line.to_string(),
                    dep_names,
                }));
            }
        }

        Self { nodes }
    }

    /// Return the number of nodes (bindings + passthroughs).
    #[must_use]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Return true if the graph has no nodes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
}

// ─── Parsing helpers ──────────────────────────────────────────────────────────

/// Parse a line as `let name = expr;` — returns `None` for non-let lines.
fn parse_let_binding(
    raw_line: &str,
    trimmed: &str,
    name_to_idx: &std::collections::HashMap<String, usize>,
    nodes: &[Node],
) -> Option<BindingNode> {
    // Match `let <ident> = <expr>;` — WGSL identifier is [A-Za-z_][A-Za-z0-9_]*
    let without_let = trimmed.strip_prefix("let ")?;
    let eq_pos = without_let.find('=')?;
    let name = without_let[..eq_pos].trim().to_string();
    // Validate identifier (simple check — alphanumeric + underscore)
    if name.is_empty() || !name.chars().all(|c| c.is_alphanumeric() || c == '_') {
        return None;
    }
    let after_eq = without_let[eq_pos + 1..].trim();
    let expr = after_eq.trim_end_matches(';').trim().to_string();

    // Find dependencies: any previously-defined binding name that appears as
    // a whole word in the expression.
    let deps = find_dep_indices(&expr, name_to_idx, nodes);
    let op_class = classify_op(&expr);

    Some(BindingNode {
        name,
        expr,
        source_line: raw_line.to_string(),
        op_class,
        deps,
    })
}

/// Return indices of bindings referenced in `expr`.
fn find_dep_indices(
    expr: &str,
    name_to_idx: &std::collections::HashMap<String, usize>,
    _nodes: &[Node],
) -> Vec<usize> {
    let mut deps = Vec::new();
    for (name, &idx) in name_to_idx {
        if contains_identifier(expr, name) {
            deps.push(idx);
        }
    }
    deps.sort_unstable();
    deps
}

/// Return names from `name_to_idx` that appear as whole identifiers in `text`.
fn find_referenced_names(
    text: &str,
    name_to_idx: &std::collections::HashMap<String, usize>,
) -> Vec<String> {
    name_to_idx
        .keys()
        .filter(|name| contains_identifier(text, name))
        .cloned()
        .collect()
}

/// Check whether `text` contains `name` as a whole identifier (not as part of
/// a longer identifier).
fn contains_identifier(text: &str, name: &str) -> bool {
    let name_bytes = name.as_bytes();
    let text_bytes = text.as_bytes();
    if name_bytes.is_empty() || text_bytes.len() < name_bytes.len() {
        return false;
    }
    let mut start = 0;
    while start + name_bytes.len() <= text_bytes.len() {
        if let Some(pos) = text[start..].find(name) {
            let abs_pos = start + pos;
            let end_pos = abs_pos + name_bytes.len();
            // Check word boundaries
            let before_ok = abs_pos == 0
                || !text_bytes[abs_pos - 1].is_ascii_alphanumeric()
                    && text_bytes[abs_pos - 1] != b'_';
            let after_ok = end_pos >= text_bytes.len()
                || !text_bytes[end_pos].is_ascii_alphanumeric() && text_bytes[end_pos] != b'_';
            if before_ok && after_ok {
                return true;
            }
            start = abs_pos + 1;
        } else {
            break;
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_region() {
        let region = r"
        let c = cos_val;
        let s = sin_val;
        let cc = c * c;
        let ss = s * s;
        let new_p = c * a_kp - s * a_kq;
        ";
        let graph = WgslDependencyGraph::parse(region);
        // Count let bindings
        let binding_count = graph
            .nodes
            .iter()
            .filter(|n| matches!(n, Node::Binding(_)))
            .count();
        assert_eq!(binding_count, 5);
    }

    #[test]
    fn test_dependency_detection() {
        let region = "        let a = x * y;\n        let b = a + z;\n";
        let graph = WgslDependencyGraph::parse(region);
        // Find binding 'b'
        let b = graph.nodes.iter().find_map(|n| {
            if let Node::Binding(b) = n {
                if b.name == "b" { Some(b) } else { None }
            } else {
                None
            }
        });
        assert!(b.is_some());
        let b = b.unwrap();
        assert!(!b.deps.is_empty(), "b should depend on a");
    }

    #[test]
    fn test_classify_op_f64_fma() {
        assert_eq!(classify_op("c * akp - s * akq"), WgslOpClass::F64Fma);
    }

    #[test]
    fn test_classify_op_global_mem() {
        assert_eq!(
            classify_op("A_batch[base + idx2d(k, p, n)]"),
            WgslOpClass::GlobalMem
        );
    }

    #[test]
    fn test_classify_op_transcend() {
        assert_eq!(
            classify_op("sqrt(f64(1.0) + t * t)"),
            WgslOpClass::F64Transcend
        );
    }

    #[test]
    fn test_contains_identifier() {
        assert!(contains_identifier("let x = foo + bar;", "foo"));
        assert!(contains_identifier("let x = foo + bar;", "bar"));
        assert!(!contains_identifier("let x = foobar;", "foo"));
        assert!(!contains_identifier("let x = 1.0;", "x1"));
    }

    #[test]
    fn test_passthrough_preserved() {
        let region = "        let a = 1.0;\n        A_batch[0] = a;\n";
        let graph = WgslDependencyGraph::parse(region);
        assert_eq!(graph.nodes.len(), 2);
        assert!(matches!(graph.nodes[0], Node::Binding(_)));
        assert!(matches!(graph.nodes[1], Node::Passthrough(_)));
    }
}

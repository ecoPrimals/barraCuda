//! WGSL ILP Optimizer — Phase 3 of the Sovereign Compute Evolution.
//!
//! Processes `@ilp_region` and `@unroll_hint` annotations in WGSL shaders
//! to pre-schedule instructions for GPU hardware scoreboards.
//!
//! ## Background
//!
//! GPU shader compilers (NAK, ACO, PTXAS) perform instruction scheduling.
//! When they lack accurate per-architecture latency data (e.g. NAK on SM70),
//! scoreboard stalls leak into the emitted code. The WGSL optimizer bypasses
//! this by emitting pre-scheduled WGSL: the compiler receives an already-ILP-
//! optimised instruction stream and translates it 1:1 to machine code.
//!
//! This delivers near-PTXAS performance on any backend (NVK, RADV, PTXAS,
//! Metal) without waiting for compiler improvements — sovereign compute.
//!
//! ## What this optimizer does
//!
//! 1. **`@ilp_region` blocks** — rewrites `let`-binding sequences using ASAP
//!    list scheduling guided by a `LatencyModel`, placing independent
//!    instructions in the latency window of high-latency ops (FP64 FMA, etc.)
//!
//! 2. **`@unroll_hint N` loops** — unrolls annotated bounded `for` loops
//!    (trip count ≤ 32), exposing inter-iteration ILP to the reorderer.
//!
//! 3. Everything **outside** annotations is passed through **unchanged**.
//!
//! ## Usage
//!
//! ```rust,ignore
//! use barracuda::shaders::optimizer::WgslOptimizer;
//! use barracuda::device::latency::Sm70LatencyModel;
//!
//! let optimizer = WgslOptimizer::new(Box::new(Sm70LatencyModel));
//! let optimized = optimizer.optimize(my_shader_source);
//! ```
//!
//! Or use `ShaderTemplate::for_driver_auto` which picks the model automatically
//! from the detected `GpuDriverProfile`.
//!
//! ## Annotation syntax
//!
//! ```wgsl
//! // @ilp_region begin
//! let c  = cos_val;               // FP64 FMA — 8cy latency on SM70
//! let s  = sin_val;               // independent: scheduler may reorder
//! let cc = c * c;                 // dep on c — starts after c resolves
//! let ss = s * s;                 // dep on s — independent of cc
//! let new_p = c * a_kp - s * a_kq;  // dep on c, s — scheduled last
//! // @ilp_region end
//!
//! // @unroll_hint 8
//! for (var k = 0u; k < 8u; k = k + 1u) {
//!     // ... body with k ...
//! }
//! ```
//!
//! ## Reference
//!
//! `SOVEREIGN_COMPUTE_EVOLUTION.md` Phase 3
//! `NAK_CONTRIBUTION_PLAN_FEB18_2026.md` — context for upstream contribution

pub mod dependency_graph;
pub mod ilp_reorderer;
pub mod loop_unroller;

use crate::device::latency::{model_for_arch, ConservativeModel, LatencyModel};
use dependency_graph::WgslDependencyGraph;
use ilp_reorderer::IlpReorderer;
pub use loop_unroller::WgslLoopUnroller;

// ── Annotation markers ─────────────────────────────────────────────────────────
const ILP_BEGIN: &str = "// @ilp_region begin";
const ILP_END: &str = "// @ilp_region end";

// ─── WgslOptimizer ────────────────────────────────────────────────────────────

/// Top-level WGSL optimizer.
///
/// Instantiate with the target GPU's `LatencyModel`, then call `optimize()`.
pub struct WgslOptimizer {
    model: Box<dyn LatencyModel>,
}

impl WgslOptimizer {
    /// Create an optimizer using a specific `LatencyModel`.
    #[must_use]
    pub fn new(model: Box<dyn LatencyModel>) -> Self {
        Self { model }
    }

    /// Create an optimizer from a `GpuArch`.
    ///
    /// Convenience wrapper around `model_for_arch`.
    #[must_use]
    pub fn for_arch(arch: crate::device::capabilities::GpuArch) -> Self {
        Self::new(model_for_arch(arch))
    }

    /// Optimize a WGSL shader source string.
    ///
    /// Processes `@unroll_hint` annotations first (loop unrolling exposes more
    /// `let`-binding sequences to the ILP reorderer), then processes all
    /// `@ilp_region begin … end` blocks.
    #[must_use]
    pub fn optimize(&self, shader: &str) -> String {
        // Step 1: Unroll annotated bounded loops.
        let unrolled = WgslLoopUnroller::unroll(shader);
        // Step 2: Reorder @ilp_region blocks.
        self.reorder_ilp_regions(&unrolled)
    }

    /// Process all `@ilp_region begin … end` blocks in `source`.
    fn reorder_ilp_regions(&self, source: &str) -> String {
        let mut output = String::with_capacity(source.len());
        let mut rest: &str = source;

        loop {
            match rest.find(ILP_BEGIN) {
                None => {
                    // No more regions — append the remainder unchanged.
                    output.push_str(rest);
                    break;
                }
                Some(begin_pos) => {
                    // Emit everything before the region marker.
                    output.push_str(&rest[..begin_pos]);

                    // Include the marker line itself.
                    let after_begin = &rest[begin_pos..];
                    let marker_end = after_begin
                        .find('\n')
                        .map(|p| p + 1)
                        .unwrap_or(after_begin.len());
                    output.push_str(&after_begin[..marker_end]);

                    // Find the end marker.
                    let region_start = marker_end;
                    let region_src = &after_begin[region_start..];
                    match region_src.find(ILP_END) {
                        None => {
                            // Unterminated region — pass through unchanged.
                            output.push_str(region_src);
                            break;
                        }
                        Some(end_pos) => {
                            let region_body = &region_src[..end_pos];
                            let after_end = &region_src[end_pos..];

                            // Reorder the region body.
                            let scheduled = self.schedule_region(region_body);
                            output.push_str(&scheduled);

                            // Emit the end marker.
                            let end_marker_end = after_end
                                .find('\n')
                                .map(|p| p + 1)
                                .unwrap_or(after_end.len());
                            output.push_str(&after_end[..end_marker_end]);

                            // Advance past the end marker.
                            rest = &after_begin[region_start + end_pos + end_marker_end..];
                        }
                    }
                }
            }
        }

        output
    }

    /// Schedule one `@ilp_region` body.
    fn schedule_region(&self, region_body: &str) -> String {
        let graph = WgslDependencyGraph::parse(region_body);
        if graph.is_empty() {
            return region_body.to_string();
        }
        let scheduled_lines = IlpReorderer::reorder(&graph, self.model.as_ref());
        let mut out = String::new();
        for line in &scheduled_lines {
            out.push_str(line);
            if !line.ends_with('\n') {
                out.push('\n');
            }
        }
        out
    }
}

impl Default for WgslOptimizer {
    /// Default optimizer uses `ConservativeModel` — safe on any hardware.
    fn default() -> Self {
        Self::new(Box::new(ConservativeModel))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::latency::Sm70LatencyModel;

    fn make_optimizer() -> WgslOptimizer {
        WgslOptimizer::new(Box::new(Sm70LatencyModel))
    }

    #[test]
    fn test_optimizer_passthrough_no_annotations() {
        let shader = "fn main() {\n    let x = 1.0;\n    let y = x + 2.0;\n}\n";
        let opt = make_optimizer();
        assert_eq!(opt.optimize(shader), shader);
    }

    #[test]
    fn test_optimizer_rewrites_ilp_region() {
        let shader = "\
fn jacobi() {\n\
    // @ilp_region begin\n\
    let c = cos_val;\n\
    let s = sin_val;\n\
    let cc = c * c;\n\
    let ss = s * s;\n\
    let new_p = c * a_kp - s * a_kq;\n\
    // @ilp_region end\n\
}\n";
        let opt = make_optimizer();
        let result = opt.optimize(shader);
        // Both markers preserved
        assert!(result.contains(ILP_BEGIN));
        assert!(result.contains(ILP_END));
        // All bindings still present
        assert!(result.contains("let c ="));
        assert!(result.contains("let s ="));
        assert!(result.contains("let cc ="));
        assert!(result.contains("let ss ="));
        assert!(result.contains("let new_p ="));
        // c must precede new_p (dependency preserved)
        let pos_c = result.find("let c =").unwrap();
        let pos_new_p = result.find("let new_p =").unwrap();
        assert!(pos_c < pos_new_p);
    }

    #[test]
    fn test_optimizer_unrolls_loop() {
        let shader = "\
    // @unroll_hint 3\n\
    for (var k = 0u; k < 3u; k = k + 1u) {\n\
        let v = k;\n\
    }\n";
        let opt = make_optimizer();
        let result = opt.optimize(shader);
        assert!(!result.contains("for (var k"));
        assert!(result.contains("let k = 0u;"));
        assert!(result.contains("let k = 1u;"));
        assert!(result.contains("let k = 2u;"));
    }

    #[test]
    fn test_multiple_ilp_regions() {
        let shader = "\
    // @ilp_region begin\n\
    let a = x;\n\
    // @ilp_region end\n\
    // middle code\n\
    // @ilp_region begin\n\
    let b = y;\n\
    // @ilp_region end\n";
        let opt = make_optimizer();
        let result = opt.optimize(shader);
        assert!(result.contains("let a ="));
        assert!(result.contains("let b ="));
        assert!(result.contains("// middle code"));
        // Both begin/end markers present (2 each)
        assert_eq!(result.matches(ILP_BEGIN).count(), 2);
        assert_eq!(result.matches(ILP_END).count(), 2);
    }

    #[test]
    fn test_default_optimizer_conservative() {
        let opt = WgslOptimizer::default();
        // Should not panic on empty input
        assert_eq!(opt.optimize(""), "");
    }

    #[test]
    fn test_for_arch_constructor() {
        use crate::device::capabilities::GpuArch;
        let opt = WgslOptimizer::for_arch(GpuArch::Volta);
        let result = opt.optimize("// @ilp_region begin\nlet x = 1.0;\n// @ilp_region end\n");
        assert!(result.contains("let x ="));
    }

    /// Validates the Phase 1 SOVEREIGN optimization: the Jacobi k-loop with a variable
    /// bound `n` and `@unroll_hint 32` produces 32 guarded copies.
    /// This mirrors the pattern in `batched_eigh_single_dispatch_f64.wgsl`.
    #[test]
    fn test_jacobi_kloop_variable_bound_unroll() {
        let shader = concat!(
            "                // @unroll_hint 32\n",
            "                for (var k = 0u; k < n; k = k + 1u) {\n",
            "                    if (k != p && k != q) {\n",
            "                        let akp = A_batch[k * p];\n",
            "                        let akq = A_batch[k * q];\n",
            "                        let new_akp = c * akp - s * akq;\n",
            "                        let new_akq = s * akp + c * akq;\n",
            "                        A_batch[k * p] = new_akp;\n",
            "                        A_batch[k * q] = new_akq;\n",
            "                    }\n",
            "                }\n",
        );
        let opt = WgslOptimizer::default();
        let result = opt.optimize(shader);

        // Original for-loop replaced
        assert!(
            !result.contains("for (var k"),
            "for-loop should be unrolled"
        );

        // 32 guarded iterations, 0..31
        assert_eq!(result.matches("let k = 0u;").count(), 1);
        assert_eq!(result.matches("let k = 31u;").count(), 1);
        assert_eq!(result.matches("let k = 32u;").count(), 0);

        // Each guarded with `if (<iter>u < n)`
        assert!(result.contains("if (0u < n)"), "iteration 0 needs guard");
        assert!(result.contains("if (31u < n)"), "iteration 31 needs guard");

        // Body expressions still present (substituted k → literals with u32 suffix)
        assert!(result.contains("A_batch[0u * p]"), "k=0 substitution");
        assert!(result.contains("A_batch[31u * p]"), "k=31 substitution");
    }

    /// Verifies that an `@ilp_region` block followed immediately by a `@unroll_hint`
    /// loop are both handled correctly without interference.
    #[test]
    fn test_ilp_region_then_unroll_coexist() {
        let shader = concat!(
            "    // @ilp_region begin\n",
            "    let cc = c * c;\n",
            "    let ss = s * s;\n",
            "    // @ilp_region end\n",
            "    // @unroll_hint 2\n",
            "    for (var k = 0u; k < 2u; k = k + 1u) {\n",
            "        let x = k;\n",
            "    }\n",
        );
        let opt = WgslOptimizer::default();
        let result = opt.optimize(shader);

        // ILP region preserved
        assert!(result.contains(ILP_BEGIN));
        assert!(result.contains("let cc ="));
        assert!(result.contains("let ss ="));
        assert!(result.contains(ILP_END));

        // Loop unrolled
        assert!(!result.contains("for (var k"));
        assert!(result.contains("let k = 0u;"));
        assert!(result.contains("let k = 1u;"));
    }
}

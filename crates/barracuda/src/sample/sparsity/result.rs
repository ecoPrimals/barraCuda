// SPDX-License-Identifier: AGPL-3.0-only
//! `SparsitySampler` result types.

use crate::optimize::eval_record::EvaluationCache;
use crate::surrogate::RBFSurrogate;

/// Diagnostics for a single `SparsitySampler` iteration.
#[derive(Debug, Clone)]
pub struct IterationResult {
    /// Iteration number (0-indexed)
    pub iteration: usize,
    /// Best f found by NM solvers in this iteration
    pub best_f: f64,
    /// Number of new evaluations in this iteration
    pub n_new_evals: usize,
    /// Total evaluations accumulated
    pub total_evals: usize,
    /// Surrogate training error (leave-one-out or None if not computed)
    pub surrogate_error: Option<f64>,
    /// Whether GPU was used for surrogate training in this iteration
    pub used_gpu: bool,
}

/// Result of `SparsitySampler` optimization.
#[derive(Debug)]
pub struct SparsitySamplerResult {
    /// Best point found
    pub x_best: Vec<f64>,
    /// Best function value
    pub f_best: f64,
    /// All evaluations (for surrogate training)
    pub cache: EvaluationCache,
    /// Final trained surrogate (if training succeeded)
    pub surrogate: Option<RBFSurrogate>,
    /// Results per iteration
    pub iteration_results: Vec<IterationResult>,
}

impl SparsitySamplerResult {
    /// Extract top-k points as warm-start seeds for `DirectSampler` or subsequent optimization.
    #[must_use]
    pub fn top_k_seeds(&self, k: usize) -> Vec<Vec<f64>> {
        let mut records: Vec<_> = self.cache.records().to_vec();
        records.sort_by(|a, b| a.f.partial_cmp(&b.f).unwrap_or(std::cmp::Ordering::Equal));
        records.into_iter().take(k).map(|r| r.x).collect()
    }

    /// Get total number of true objective evaluations.
    #[must_use]
    pub fn total_evals(&self) -> usize {
        self.cache.len()
    }

    /// Get evaluations per iteration.
    #[must_use]
    pub fn evals_per_iteration(&self) -> Vec<usize> {
        self.iteration_results
            .iter()
            .map(|r| r.n_new_evals)
            .collect()
    }
}

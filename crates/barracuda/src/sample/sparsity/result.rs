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

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, reason = "test code")]

    use super::*;

    fn make_result_with_cache(records: Vec<(Vec<f64>, f64)>) -> SparsitySamplerResult {
        let mut cache = EvaluationCache::new();
        for (x, f) in records {
            cache.record(x, f);
        }
        SparsitySamplerResult {
            x_best: cache.best_x().unwrap().to_vec(),
            f_best: cache.best_f().unwrap(),
            cache,
            surrogate: None,
            iteration_results: vec![],
        }
    }

    #[test]
    fn total_evals_matches_cache_len() {
        let result = make_result_with_cache(vec![
            (vec![1.0, 2.0], 5.0),
            (vec![0.5, 1.5], 3.0),
            (vec![0.0, 0.0], 0.0),
        ]);
        assert_eq!(result.total_evals(), 3);
    }

    #[test]
    fn top_k_seeds_returns_best_k_by_f() {
        let result = make_result_with_cache(vec![
            (vec![1.0, 2.0], 5.0),
            (vec![0.5, 1.5], 3.0),
            (vec![0.0, 0.0], 0.0),
            (vec![2.0, 1.0], 4.0),
        ]);
        let seeds = result.top_k_seeds(2);
        assert_eq!(seeds.len(), 2);
        assert_eq!(seeds[0], vec![0.0, 0.0]);
        assert_eq!(seeds[1], vec![0.5, 1.5]);
    }

    #[test]
    fn top_k_seeds_k_zero_returns_empty() {
        let result = make_result_with_cache(vec![(vec![1.0], 1.0)]);
        let seeds = result.top_k_seeds(0);
        assert!(seeds.is_empty());
    }

    #[test]
    fn top_k_seeds_k_exceeds_len_returns_all() {
        let result = make_result_with_cache(vec![(vec![1.0], 1.0), (vec![2.0], 2.0)]);
        let seeds = result.top_k_seeds(10);
        assert_eq!(seeds.len(), 2);
    }

    #[test]
    fn top_k_seeds_empty_cache_returns_empty() {
        let result = SparsitySamplerResult {
            x_best: vec![],
            f_best: 0.0,
            cache: EvaluationCache::new(),
            surrogate: None,
            iteration_results: vec![],
        };
        let seeds = result.top_k_seeds(5);
        assert!(seeds.is_empty());
    }

    #[test]
    fn evals_per_iteration() {
        let mut cache = EvaluationCache::new();
        cache.record(vec![1.0], 1.0);
        cache.record(vec![2.0], 2.0);
        let result = SparsitySamplerResult {
            x_best: vec![1.0],
            f_best: 1.0,
            cache,
            surrogate: None,
            iteration_results: vec![
                IterationResult {
                    iteration: 0,
                    best_f: 1.5,
                    n_new_evals: 5,
                    total_evals: 5,
                    surrogate_error: None,
                    used_gpu: false,
                },
                IterationResult {
                    iteration: 1,
                    best_f: 1.0,
                    n_new_evals: 3,
                    total_evals: 8,
                    surrogate_error: Some(0.01),
                    used_gpu: true,
                },
            ],
        };
        assert_eq!(result.evals_per_iteration(), vec![5, 3]);
    }
}

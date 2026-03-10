// SPDX-License-Identifier: AGPL-3.0-only
//! Cascade filtering pipeline
//!
//! A cascade is a series of increasingly expensive filters that progressively
//! reduce the candidate set. Early stages are cheap and filter aggressively,
//! later stages are expensive but only see high-quality candidates.
//!
//! # hotSpring Validated Pattern
//!
//! ```text
//! Input: 6000 candidates
//!     ↓
//! Stage 1: NMP constraints (1μs/item, 79% reject)
//!     → 1260 candidates
//!     ↓
//! Stage 2: SEMF proxy (0.1ms/item, 13% reject)
//!     → 540 candidates
//!     ↓
//! Stage 3: Classifier (10μs/item, optional)
//!     → ~540 candidates
//!     ↓
//! Stage 4: Full HFB (0.2s/item)
//!     → 488 evaluated with results
//!
//! Savings: 91.9% of expensive evaluations avoided
//! ```

use crate::error::Result;
use std::time::{Duration, Instant};

/// Type alias for filter predicates
type FilterPredicate<T> = Box<dyn Fn(&T) -> bool + Send + Sync>;

/// Type alias for transformation functions
type TransformFn<T, U> = Box<dyn Fn(&T) -> Result<U> + Send + Sync>;

/// Result of filtering at a cascade stage
#[derive(Debug, Clone)]
pub struct FilterResult<T> {
    /// Items that passed the filter
    pub passed: Vec<T>,
    /// Number of items filtered out
    pub rejected: usize,
    /// Number of items that caused errors
    pub errors: usize,
    /// Filter execution time
    pub duration: Duration,
}

impl<T> FilterResult<T> {
    /// Get rejection rate
    pub fn rejection_rate(&self) -> f64 {
        let total = self.passed.len() + self.rejected + self.errors;
        if total == 0 {
            0.0
        } else {
            (self.rejected + self.errors) as f64 / total as f64
        }
    }
}

/// Result of running a cascade
#[derive(Debug, Clone)]
pub struct CascadeResult<T, U> {
    /// Final results (input, output) pairs
    pub results: Vec<(T, U)>,
    /// Per-stage statistics
    pub stages: Vec<CascadeStageStats>,
    /// Total candidates at input
    pub total_input: usize,
    /// Total candidates evaluated at final stage
    pub total_evaluated: usize,
    /// Total execution time
    pub total_duration: Duration,
}

impl<T, U> CascadeResult<T, U> {
    /// Get overall rejection rate
    pub fn overall_rejection_rate(&self) -> f64 {
        if self.total_input == 0 {
            0.0
        } else {
            1.0 - (self.total_evaluated as f64 / self.total_input as f64)
        }
    }

    /// Get speedup factor (how many expensive evals were avoided)
    pub fn speedup_factor(&self) -> f64 {
        if self.total_evaluated == 0 {
            f64::INFINITY
        } else {
            self.total_input as f64 / self.total_evaluated as f64
        }
    }

    /// Human-readable summary
    pub fn summary(&self) -> String {
        let mut lines = Vec::new();

        lines.push("╔═══════════════════════════════════════════════════════════════╗".to_string());
        lines.push("║                    CASCADE PIPELINE RESULTS                   ║".to_string());
        lines.push("╚═══════════════════════════════════════════════════════════════╝".to_string());
        lines.push(format!(
            "Input: {} → Evaluated: {} → Savings: {:.1}%",
            self.total_input,
            self.total_evaluated,
            self.overall_rejection_rate() * 100.0
        ));
        lines.push(format!(
            "Total time: {:.3}s",
            self.total_duration.as_secs_f64()
        ));
        lines.push(String::new());

        for (i, stage) in self.stages.iter().enumerate() {
            lines.push(format!(
                "Stage {}: {} → {} ({:.1}% rejected, {:.3}s)",
                i + 1,
                stage.name,
                stage.output_count,
                stage.rejection_rate * 100.0,
                stage.duration.as_secs_f64()
            ));
        }

        lines.join("\n")
    }
}

/// Statistics for a single cascade stage
#[derive(Debug, Clone)]
pub struct CascadeStageStats {
    /// Stage name
    pub name: String,
    /// Number of items input to this stage
    pub input_count: usize,
    /// Number of items output from this stage
    pub output_count: usize,
    /// Rejection rate (1 - output/input)
    pub rejection_rate: f64,
    /// Execution time
    pub duration: Duration,
}

/// Builder for cascade pipelines
pub struct CascadeBuilder<T, U> {
    /// Filter stages (applied before final transform)
    filters: Vec<FilterPredicate<T>>,
    /// Filter names
    filter_names: Vec<String>,
    /// Final transform function
    transform: Option<TransformFn<T, U>>,
    /// Transform name
    transform_name: String,
}

impl<T, U> CascadeBuilder<T, U>
where
    T: Clone + Send + Sync,
    U: Clone + Send + Sync,
{
    /// Create a new cascade builder
    pub fn new() -> Self {
        Self {
            filters: Vec::new(),
            filter_names: Vec::new(),
            transform: None,
            transform_name: "transform".to_string(),
        }
    }

    /// Add a filter stage
    pub fn filter<F>(mut self, name: impl Into<String>, f: F) -> Self
    where
        F: Fn(&T) -> bool + Send + Sync + 'static,
    {
        self.filter_names.push(name.into());
        self.filters.push(Box::new(f));
        self
    }

    /// Set the final transform function
    pub fn transform<F>(mut self, name: impl Into<String>, f: F) -> Self
    where
        F: Fn(&T) -> Result<U> + Send + Sync + 'static,
    {
        self.transform_name = name.into();
        self.transform = Some(Box::new(f));
        self
    }

    /// Build the cascade
    pub fn build(self) -> Cascade<T, U> {
        Cascade {
            filters: self.filters,
            filter_names: self.filter_names,
            transform: self.transform,
            transform_name: self.transform_name,
        }
    }
}

impl<T, U> Default for CascadeBuilder<T, U>
where
    T: Clone + Send + Sync,
    U: Clone + Send + Sync,
{
    fn default() -> Self {
        Self::new()
    }
}

/// A cascade filtering pipeline
pub struct Cascade<T, U> {
    /// Filter stages
    filters: Vec<FilterPredicate<T>>,
    /// Filter names
    filter_names: Vec<String>,
    /// Final transform
    transform: Option<TransformFn<T, U>>,
    /// Transform name
    transform_name: String,
}

impl<T, U> Cascade<T, U>
where
    T: Clone + Send + Sync,
    U: Clone + Send + Sync,
{
    /// Create a cascade builder
    pub fn builder() -> CascadeBuilder<T, U> {
        CascadeBuilder::new()
    }

    /// Run the cascade on a set of candidates
    pub fn run(&self, candidates: &[T]) -> CascadeResult<T, U> {
        let start = Instant::now();
        let total_input = candidates.len();
        let mut current: Vec<T> = candidates.to_vec();
        let mut stages = Vec::new();

        // Run filter stages
        for (i, filter) in self.filters.iter().enumerate() {
            let stage_start = Instant::now();
            let input_count = current.len();

            let passed: Vec<T> = current.iter().filter(|x| filter(x)).cloned().collect();

            let output_count = passed.len();
            let rejection_rate = if input_count == 0 {
                0.0
            } else {
                1.0 - (output_count as f64 / input_count as f64)
            };

            stages.push(CascadeStageStats {
                name: self
                    .filter_names
                    .get(i)
                    .cloned()
                    .unwrap_or_else(|| format!("filter_{}", i)),
                input_count,
                output_count,
                rejection_rate,
                duration: stage_start.elapsed(),
            });

            current = passed;
        }

        // Run transform stage
        let transform_start = Instant::now();
        let transform_input = current.len();
        let mut results = Vec::new();
        let mut errors = 0;

        if let Some(ref transform) = self.transform {
            for item in &current {
                match transform(item) {
                    Ok(output) => results.push((item.clone(), output)),
                    Err(_) => errors += 1,
                }
            }
        }

        stages.push(CascadeStageStats {
            name: self.transform_name.clone(),
            input_count: transform_input,
            output_count: results.len(),
            rejection_rate: if transform_input == 0 {
                0.0
            } else {
                errors as f64 / transform_input as f64
            },
            duration: transform_start.elapsed(),
        });

        CascadeResult {
            results,
            stages,
            total_input,
            total_evaluated: transform_input,
            total_duration: start.elapsed(),
        }
    }

    /// Run cascade in parallel (requires rayon feature)
    #[cfg(feature = "parallel")]
    pub fn run_parallel(&self, candidates: &[T]) -> CascadeResult<T, U>
    where
        T: Send + Sync,
        U: Send + Sync,
    {
        use rayon::prelude::*;

        let start = Instant::now();
        let total_input = candidates.len();
        let mut current: Vec<T> = candidates.to_vec();
        let mut stages = Vec::new();

        // Run filter stages in parallel
        for (i, filter) in self.filters.iter().enumerate() {
            let stage_start = Instant::now();
            let input_count = current.len();

            let passed: Vec<T> = current.par_iter().filter(|x| filter(x)).cloned().collect();

            let output_count = passed.len();
            let rejection_rate = if input_count == 0 {
                0.0
            } else {
                1.0 - (output_count as f64 / input_count as f64)
            };

            stages.push(CascadeStageStats {
                name: self
                    .filter_names
                    .get(i)
                    .cloned()
                    .unwrap_or_else(|| format!("filter_{}", i)),
                input_count,
                output_count,
                rejection_rate,
                duration: stage_start.elapsed(),
            });

            current = passed;
        }

        // Run transform stage in parallel
        let transform_start = Instant::now();
        let transform_input = current.len();

        let results: Vec<(T, U)> = if let Some(ref transform) = self.transform {
            current
                .par_iter()
                .filter_map(|item| transform(item).ok().map(|output| (item.clone(), output)))
                .collect()
        } else {
            Vec::new()
        };

        let errors = transform_input - results.len();

        stages.push(CascadeStageStats {
            name: self.transform_name.clone(),
            input_count: transform_input,
            output_count: results.len(),
            rejection_rate: if transform_input == 0 {
                0.0
            } else {
                errors as f64 / transform_input as f64
            },
            duration: transform_start.elapsed(),
        });

        CascadeResult {
            results,
            stages,
            total_input,
            total_evaluated: transform_input,
            total_duration: start.elapsed(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cascade_builder() {
        let cascade = Cascade::<i32, i32>::builder()
            .filter("positive", |x| *x > 0)
            .filter("even", |x| x % 2 == 0)
            .transform("square", |x| Ok(x * x))
            .build();

        let candidates = vec![-2, -1, 0, 1, 2, 3, 4, 5, 6];
        let result = cascade.run(&candidates);

        // Stage 1: positive → -2, -1, 0 rejected → 1, 2, 3, 4, 5, 6 pass (6)
        // Stage 2: even → 1, 3, 5 rejected → 2, 4, 6 pass (3)
        // Transform: 2, 4, 6 → 4, 16, 36

        assert_eq!(result.results.len(), 3);
        assert_eq!(result.results[0], (2, 4));
        assert_eq!(result.results[1], (4, 16));
        assert_eq!(result.results[2], (6, 36));
        assert_eq!(result.total_input, 9);
        assert_eq!(result.total_evaluated, 3);
    }

    #[test]
    fn test_cascade_rejection_rate() {
        let cascade = Cascade::<i32, i32>::builder()
            .filter("half", |x| x % 2 == 0)
            .transform("identity", |x| Ok(*x))
            .build();

        let candidates: Vec<i32> = (0..100).collect();
        let result = cascade.run(&candidates);

        assert_eq!(result.total_input, 100);
        assert_eq!(result.total_evaluated, 50);
        assert!((result.overall_rejection_rate() - 0.5).abs() < 0.01);
        assert!((result.speedup_factor() - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_cascade_stats() {
        let cascade = Cascade::<i32, i32>::builder()
            .filter("gt_10", |x| *x > 10)
            .filter("lt_20", |x| *x < 20)
            .transform("double", |x| Ok(x * 2))
            .build();

        let candidates: Vec<i32> = (0..30).collect();
        let result = cascade.run(&candidates);

        // Stage 1: gt_10 → 0-10 rejected (11), 11-29 pass (19)
        // Stage 2: lt_20 → 20-29 rejected (10), 11-19 pass (9)
        // Transform: 11-19 → doubled

        assert_eq!(result.stages.len(), 3);
        assert_eq!(result.stages[0].name, "gt_10");
        assert_eq!(result.stages[0].input_count, 30);
        assert_eq!(result.stages[0].output_count, 19);

        assert_eq!(result.stages[1].name, "lt_20");
        assert_eq!(result.stages[1].input_count, 19);
        assert_eq!(result.stages[1].output_count, 9);

        assert_eq!(result.results.len(), 9);
    }

    #[test]
    fn test_cascade_summary() {
        let cascade = Cascade::<i32, i32>::builder()
            .filter("positive", |x| *x > 0)
            .transform("square", |x| Ok(x * x))
            .build();

        let candidates: Vec<i32> = (-5..6).collect();
        let result = cascade.run(&candidates);

        let summary = result.summary();
        assert!(summary.contains("CASCADE"));
        assert!(summary.contains("positive"));
    }

    #[test]
    fn test_filter_result() {
        let result = FilterResult {
            passed: vec![1, 2, 3],
            rejected: 5,
            errors: 2,
            duration: Duration::from_millis(10),
        };

        assert!((result.rejection_rate() - 0.7).abs() < 0.01); // (5+2)/10
    }
}

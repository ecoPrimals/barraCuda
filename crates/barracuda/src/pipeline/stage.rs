// SPDX-License-Identifier: AGPL-3.0-or-later
//! Pipeline stage definitions
//!
//! A stage represents a single step in a compute pipeline with:
//! - Optional input filtering (reject candidates early)
//! - Transform function (compute on valid candidates)
//! - Target device preference (CPU, GPU, NPU)
//! - Metrics collection

use crate::error::Result;
use std::time::{Duration, Instant};

/// Type alias for stage filter predicates
type StageFilter<T> = Box<dyn Fn(&T) -> bool + Send + Sync>;

/// Type alias for stage transformation functions
type StageTransform<T, U> = Box<dyn Fn(&T) -> Result<U> + Send + Sync>;

/// Compute target for a stage
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Target {
    /// CPU (single-threaded, f64 precision)
    #[default]
    Cpu,
    /// CPU (multi-threaded via rayon)
    CpuParallel,
    /// GPU (f32 precision, high throughput)
    Gpu,
    /// NPU (inference, classification)
    Npu,
    /// Auto-select based on workload size
    Auto,
}

impl Target {
    /// Check if this is a CPU target
    pub fn is_cpu(self) -> bool {
        matches!(self, Target::Cpu | Target::CpuParallel)
    }

    /// Check if this is a GPU target
    pub fn is_gpu(self) -> bool {
        matches!(self, Target::Gpu)
    }

    /// Check if this is an NPU target
    pub fn is_npu(self) -> bool {
        matches!(self, Target::Npu)
    }
}

/// Configuration for a pipeline stage
#[derive(Debug, Clone)]
pub struct StageConfig {
    /// Stage name (for logging/metrics)
    pub name: String,
    /// Target compute device
    pub target: Target,
    /// Whether this stage can be skipped on error
    pub optional: bool,
    /// Timeout for this stage (None = no timeout)
    pub timeout: Option<Duration>,
    /// Expected cost per item (for scheduling)
    pub cost_estimate: f64,
}

impl Default for StageConfig {
    fn default() -> Self {
        Self {
            name: "unnamed".to_string(),
            target: Target::Cpu,
            optional: false,
            timeout: None,
            cost_estimate: 1.0,
        }
    }
}

impl StageConfig {
    /// Create a new stage config with the given name
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ..Default::default()
        }
    }

    /// Set target device
    #[must_use]
    pub fn with_target(mut self, target: Target) -> Self {
        self.target = target;
        self
    }

    /// Mark stage as optional (can be skipped on error)
    pub fn optional(mut self) -> Self {
        self.optional = true;
        self
    }

    /// Set timeout
    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = Some(timeout);
        self
    }

    /// Set cost estimate
    #[must_use]
    pub fn with_cost(mut self, cost: f64) -> Self {
        self.cost_estimate = cost;
        self
    }
}

/// Result of running a stage
#[derive(Debug, Clone)]
pub struct StageResult<T> {
    /// Stage name
    pub name: String,
    /// Items that passed the stage
    pub passed: Vec<T>,
    /// Items that were filtered out
    pub filtered: usize,
    /// Items that caused errors
    pub errors: usize,
    /// Execution time
    pub duration: Duration,
    /// Target device used
    pub target: Target,
}

impl<T> StageResult<T> {
    /// Get pass rate (passed / total)
    pub fn pass_rate(&self) -> f64 {
        let total = self.passed.len() + self.filtered + self.errors;
        if total == 0 {
            0.0
        } else {
            self.passed.len() as f64 / total as f64
        }
    }

    /// Get throughput (items / second)
    pub fn throughput(&self) -> f64 {
        let total = self.passed.len() + self.filtered + self.errors;
        total as f64 / self.duration.as_secs_f64().max(1e-9)
    }
}

/// A pipeline stage that filters and/or transforms data
pub struct Stage<T, U> {
    /// Stage configuration
    config: StageConfig,
    /// Optional filter function (returns true to pass)
    filter: Option<StageFilter<T>>,
    /// Optional transform function
    transform: Option<StageTransform<T, U>>,
}

impl<T, U> Stage<T, U>
where
    T: Clone + Send + Sync,
    U: Clone + Send + Sync,
{
    /// Create a new stage with the given name
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            config: StageConfig::new(name),
            filter: None,
            transform: None,
        }
    }

    /// Set the target device
    pub fn target(mut self, target: Target) -> Self {
        self.config.target = target;
        self
    }

    /// Set a filter function (returns true to pass through)
    pub fn filter<F>(mut self, f: F) -> Self
    where
        F: Fn(&T) -> bool + Send + Sync + 'static,
    {
        self.filter = Some(Box::new(f));
        self
    }

    /// Set a transform function
    pub fn transform<F>(mut self, f: F) -> Self
    where
        F: Fn(&T) -> Result<U> + Send + Sync + 'static,
    {
        self.transform = Some(Box::new(f));
        self
    }

    /// Mark stage as optional
    pub fn optional(mut self) -> Self {
        self.config.optional = true;
        self
    }

    /// Set cost estimate
    pub fn cost(mut self, cost: f64) -> Self {
        self.config.cost_estimate = cost;
        self
    }

    /// Get stage name
    pub fn name(&self) -> &str {
        &self.config.name
    }

    /// Get stage config
    pub fn config(&self) -> &StageConfig {
        &self.config
    }
}

impl<T, U> Stage<T, U>
where
    T: Clone + Send + Sync,
    U: Clone + Send + Sync,
{
    /// Run the stage as a filter (ignoring transform)
    pub fn run_filter(&self, items: &[T]) -> StageResult<T> {
        let start = Instant::now();
        let mut passed = Vec::new();
        let mut filtered = 0;

        for item in items {
            let pass = self.filter.as_ref().map(|f| f(item)).unwrap_or(true);

            if pass {
                passed.push(item.clone());
            } else {
                filtered += 1;
            }
        }

        StageResult {
            name: self.config.name.clone(),
            passed,
            filtered,
            errors: 0,
            duration: start.elapsed(),
            target: self.config.target,
        }
    }

    /// Run the stage with transform
    pub fn run(&self, items: &[T]) -> StageResult<(T, U)> {
        let start = Instant::now();
        let mut passed = Vec::new();
        let mut filtered = 0;
        let mut errors = 0;

        for item in items {
            // Apply filter first
            let pass_filter = self.filter.as_ref().map(|f| f(item)).unwrap_or(true);

            if !pass_filter {
                filtered += 1;
                continue;
            }

            // Apply transform
            if let Some(ref transform) = self.transform {
                match transform(item) {
                    Ok(output) => {
                        passed.push((item.clone(), output));
                    }
                    Err(_) => {
                        errors += 1;
                    }
                }
            } else {
                // No transform - skip this item
                errors += 1;
            }
        }

        StageResult {
            name: self.config.name.clone(),
            passed,
            filtered,
            errors,
            duration: start.elapsed(),
            target: self.config.target,
        }
    }
}

/// Convenience type alias for filter-only stages
pub type FilterStage<T> = Stage<T, T>;

/// Convenience type alias for transform stages
pub type TransformStage<T, U> = Stage<T, U>;

#[expect(clippy::unwrap_used, reason = "tests")]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_target() {
        assert!(Target::Cpu.is_cpu());
        assert!(Target::CpuParallel.is_cpu());
        assert!(Target::Gpu.is_gpu());
        assert!(Target::Npu.is_npu());
        assert!(!Target::Cpu.is_gpu());
    }

    #[test]
    fn test_stage_config() {
        let config = StageConfig::new("test")
            .with_target(Target::Gpu)
            .with_cost(2.5)
            .optional();

        assert_eq!(config.name, "test");
        assert_eq!(config.target, Target::Gpu);
        assert!(config.optional);
        assert!((config.cost_estimate - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_filter_stage() {
        let stage = Stage::<i32, i32>::new("even_filter").filter(|x: &i32| x % 2 == 0);

        let items = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let result = stage.run_filter(&items);

        assert_eq!(result.passed.len(), 5);
        assert_eq!(result.filtered, 5);
        assert_eq!(result.errors, 0);
        assert!((result.pass_rate() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_transform_stage() {
        let stage = Stage::<i32, i32>::new("square").transform(|x: &i32| Ok(x * x));

        let items = vec![1, 2, 3, 4, 5];
        let result = stage.run(&items);

        assert_eq!(result.passed.len(), 5);
        assert_eq!(result.passed[0], (1, 1));
        assert_eq!(result.passed[1], (2, 4));
        assert_eq!(result.passed[4], (5, 25));
    }

    #[test]
    fn test_stage_with_filter_and_transform() {
        let stage = Stage::<i32, i32>::new("filter_and_square")
            .filter(|x| *x > 2)
            .transform(|x| Ok(x * x));

        let items = vec![1, 2, 3, 4, 5];
        let result = stage.run(&items);

        assert_eq!(result.passed.len(), 3); // 3, 4, 5 pass filter
        assert_eq!(result.filtered, 2); // 1, 2 filtered
        assert_eq!(result.passed[0], (3, 9));
        assert_eq!(result.passed[1], (4, 16));
        assert_eq!(result.passed[2], (5, 25));
    }

    #[test]
    fn test_stage_result_metrics() {
        let result = StageResult {
            name: "test".to_string(),
            passed: vec![1, 2, 3],
            filtered: 5,
            errors: 2,
            duration: Duration::from_millis(100),
            target: Target::Cpu,
        };

        assert!((result.pass_rate() - 0.3).abs() < 1e-10); // 3/10
        assert!(result.throughput() > 0.0);
    }
}

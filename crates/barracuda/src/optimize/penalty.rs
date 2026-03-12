// SPDX-License-Identifier: AGPL-3.0-only
//! Adaptive penalty functions for constrained optimization
//!
//! Provides data-driven penalty scaling that adapts to the objective landscape.
//! This prevents common issues:
//! - Penalty too low: optimizer converges TO penalty boundary
//! - Penalty too high: gradients dominated by penalty, missing feasible optima
//!
//! # Algorithm
//!
//! Given a set of feasible objective values, compute a penalty that is
//! guaranteed to exceed the worst feasible value by a safe margin.
//!
//! # Reference
//!
//! hotSpring validation: `surrogate.rs::adaptive_penalty()`

use crate::error::{BarracudaError, Result};

/// Adaptive penalty configuration.
#[derive(Debug, Clone, Copy)]
pub struct PenaltyConfig {
    /// Minimum penalty value (default: 1e6)
    pub min_penalty: f64,
    /// Maximum penalty value (default: 1e12)
    pub max_penalty: f64,
    /// Safety margin multiplier (default: 10.0)
    pub safety_margin: f64,
    /// Use log-transform for penalties (default: true)
    pub use_log: bool,
}

impl Default for PenaltyConfig {
    fn default() -> Self {
        Self {
            min_penalty: 1e6,
            max_penalty: 1e12,
            safety_margin: 10.0,
            use_log: true,
        }
    }
}

impl PenaltyConfig {
    /// Set custom minimum penalty value.
    #[must_use]
    pub fn with_min(mut self, min: f64) -> Self {
        self.min_penalty = min;
        self
    }

    /// Set custom safety margin multiplier.
    #[must_use]
    pub fn with_margin(mut self, margin: f64) -> Self {
        self.safety_margin = margin;
        self
    }
}

/// Result of adaptive penalty computation.
#[derive(Debug, Clone)]
pub struct AdaptivePenalty {
    /// Computed raw penalty value
    pub raw_penalty: f64,
    /// Transformed penalty (log if `use_log`)
    pub penalty: f64,
    /// Maximum objective value among feasible points
    pub feasible_max: f64,
    /// Minimum objective value among feasible points
    pub feasible_min: f64,
    /// Mean objective value among feasible points
    pub feasible_mean: f64,
    /// Number of feasible points used
    pub n_feasible: usize,
}

impl AdaptivePenalty {
    /// Apply penalty to an infeasible point.
    ///
    /// Returns a value guaranteed to be worse than any feasible point.
    #[must_use]
    pub fn apply(&self, constraint_violation: f64) -> f64 {
        self.penalty * (1.0 + constraint_violation.abs())
    }
}

/// Compute adaptive penalty from feasible objective values.
///
/// The penalty is set such that `log(1 + penalty)` exceeds the maximum
/// feasible value by a safety margin, preventing the optimizer from
/// converging to the penalty boundary.
///
/// # Arguments
///
/// * `feasible_values` - Objective values from feasible (constraint-satisfying) points
/// * `config` - Penalty configuration
///
/// # Returns
///
/// [`AdaptivePenalty`] with computed penalty and statistics.
///
/// # Example
///
/// ```
/// use barracuda::optimize::{adaptive_penalty, PenaltyConfig};
///
/// // Observed feasible objective values
/// let feasible = vec![5.2, 8.1, 3.4, 12.3, 7.8];
///
/// let penalty = adaptive_penalty(&feasible, PenaltyConfig::default()).unwrap();
///
/// // Penalty is much larger than max feasible (log-transformed)
/// assert!(penalty.penalty > 12.3);
///
/// // Apply to infeasible point
/// let penalized = penalty.apply(0.5);  // 50% constraint violation
/// assert!(penalized > penalty.penalty);  // Violation increases penalty
/// ```
///
/// # Reference
///
/// hotSpring validation: `surrogate.rs::adaptive_penalty()`
///
/// # Errors
///
/// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
/// readback fails (e.g. device lost or out of memory).
pub fn adaptive_penalty(feasible_values: &[f64], config: PenaltyConfig) -> Result<AdaptivePenalty> {
    if feasible_values.is_empty() {
        // No feasible points - use minimum penalty
        let raw_penalty = config.min_penalty;
        let penalty = if config.use_log {
            raw_penalty.ln_1p()
        } else {
            raw_penalty
        };

        return Ok(AdaptivePenalty {
            raw_penalty,
            penalty,
            feasible_max: f64::NEG_INFINITY,
            feasible_min: f64::INFINITY,
            feasible_mean: 0.0,
            n_feasible: 0,
        });
    }

    // Filter out non-finite values
    let valid: Vec<f64> = feasible_values
        .iter()
        .filter(|v| v.is_finite())
        .copied()
        .collect();

    if valid.is_empty() {
        return Err(BarracudaError::InvalidInput {
            message: "No finite feasible values provided".to_string(),
        });
    }

    // Compute statistics
    let n = valid.len();
    let feasible_max = valid.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let feasible_min = valid.iter().copied().fold(f64::INFINITY, f64::min);
    let feasible_mean = valid.iter().sum::<f64>() / n as f64;

    // Compute raw penalty: safety_margin × max feasible (or min_penalty if max is small)
    let raw_penalty = (feasible_max.abs() * config.safety_margin)
        .max(config.min_penalty)
        .min(config.max_penalty);

    // Transform if using log
    let penalty = if config.use_log {
        // Penalty such that log(1 + raw_penalty) exceeds max feasible
        let target = feasible_max * config.safety_margin;
        raw_penalty.max(target.exp_m1()).ln_1p()
    } else {
        raw_penalty
    };

    Ok(AdaptivePenalty {
        raw_penalty,
        penalty,
        feasible_max,
        feasible_min,
        feasible_mean,
        n_feasible: n,
    })
}

/// Compute penalty from observed values using robust statistics (MAD).
///
/// Uses median absolute deviation for outlier-resistant penalty estimation.
///
/// # Arguments
///
/// * `values` - All observed objective values (feasible and infeasible mixed)
/// * `config` - Penalty configuration
/// * `mad_multiplier` - Number of MADs above median to set penalty (default: 5.0)
///
/// # Example
///
/// ```
/// use barracuda::optimize::{adaptive_penalty_mad, PenaltyConfig};
///
/// // Mixed feasible and penalty values
/// let values = vec![5.0, 8.0, 3.0, 1e10, 7.0, 1e10];
///
/// let penalty = adaptive_penalty_mad(&values, PenaltyConfig::default(), 5.0).unwrap();
/// ```
///
/// # Errors
///
/// Returns [`Err`] if values is empty or contains no finite values.
pub fn adaptive_penalty_mad(
    values: &[f64],
    config: PenaltyConfig,
    mad_multiplier: f64,
) -> Result<AdaptivePenalty> {
    if values.is_empty() {
        return Err(BarracudaError::InvalidInput {
            message: "values cannot be empty".to_string(),
        });
    }

    // Filter finite values
    let mut valid: Vec<f64> = values.iter().filter(|v| v.is_finite()).copied().collect();

    if valid.is_empty() {
        return Err(BarracudaError::InvalidInput {
            message: "No finite values provided".to_string(),
        });
    }

    // Compute median
    valid.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = valid.len();
    let median = if n.is_multiple_of(2) {
        f64::midpoint(valid[n / 2 - 1], valid[n / 2])
    } else {
        valid[n / 2]
    };

    // Compute MAD
    let mut deviations: Vec<f64> = valid.iter().map(|v| (v - median).abs()).collect();
    deviations.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mad = if n.is_multiple_of(2) {
        f64::midpoint(deviations[n / 2 - 1], deviations[n / 2])
    } else {
        deviations[n / 2]
    };

    // Penalty threshold: median + k × MAD
    let threshold = median + mad_multiplier * mad;

    // Filter feasible values (below threshold)
    let feasible: Vec<f64> = valid.iter().filter(|&&v| v <= threshold).copied().collect();

    adaptive_penalty(&feasible, config)
}

/// Create a penalized objective function.
///
/// Wraps an objective function with constraint checking and automatic penalty.
///
/// # Arguments
///
/// * `objective` - Original objective function
/// * `constraints` - Constraint function returning violation amount (0 = feasible, >0 = infeasible)
/// * `penalty` - Adaptive penalty to apply
///
/// # Example
///
/// ```ignore
/// use barracuda::optimize::{adaptive_penalty, penalized_objective, PenaltyConfig};
///
/// let feasible_vals = vec![5.0, 8.0, 3.0];
/// let penalty = adaptive_penalty(&feasible_vals, PenaltyConfig::default()).unwrap();
///
/// let objective = |x: &[f64]| x.iter().map(|v| v * v).sum();
/// let constraint = |x: &[f64]| {
///     let sum: f64 = x.iter().sum();
///     if sum > 1.0 { sum - 1.0 } else { 0.0 }  // sum(x) <= 1 constraint
/// };
///
/// let penalized = penalized_objective(objective, constraint, &penalty);
/// let value = penalized(&[0.3, 0.3, 0.5]);  // Violates constraint
/// ```
pub fn penalized_objective<F, G>(
    objective: F,
    constraints: G,
    penalty: &AdaptivePenalty,
) -> impl Fn(&[f64]) -> f64
where
    F: Fn(&[f64]) -> f64,
    G: Fn(&[f64]) -> f64,
{
    let penalty_val = penalty.penalty;
    move |x: &[f64]| {
        let violation = constraints(x);
        if violation <= 0.0 {
            objective(x)
        } else {
            penalty_val * (1.0 + violation)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adaptive_penalty_basic() {
        let feasible = vec![5.0, 10.0, 3.0, 8.0];
        let result = adaptive_penalty(&feasible, PenaltyConfig::default()).unwrap();

        assert_eq!(result.n_feasible, 4);
        assert!((result.feasible_max - 10.0).abs() < 1e-10);
        assert!((result.feasible_min - 3.0).abs() < 1e-10);
        assert!(result.penalty > result.feasible_max);
    }

    #[test]
    fn test_adaptive_penalty_empty() {
        let result = adaptive_penalty(&[], PenaltyConfig::default()).unwrap();

        assert_eq!(result.n_feasible, 0);
        assert!(result.raw_penalty >= 1e6);
    }

    #[test]
    fn test_penalty_apply() {
        let feasible = vec![5.0, 10.0];
        let result = adaptive_penalty(&feasible, PenaltyConfig::default()).unwrap();

        // Higher violation = higher penalty
        let p1 = result.apply(0.1);
        let p2 = result.apply(0.5);
        assert!(p2 > p1);
    }

    #[test]
    fn test_penalty_exceeds_feasible() {
        let feasible = vec![5.0, 10.0, 15.0];
        let result = adaptive_penalty(&feasible, PenaltyConfig::default()).unwrap();

        // Penalty should exceed max feasible
        let penalized = result.apply(0.0);
        assert!(penalized > result.feasible_max);
    }

    #[test]
    fn test_adaptive_penalty_mad() {
        // Mix of feasible and outliers
        let values = vec![5.0, 8.0, 3.0, 1e10, 7.0, 1e10, 6.0];
        let result = adaptive_penalty_mad(&values, PenaltyConfig::default(), 5.0).unwrap();

        // Should identify ~5 feasible points (excluding 1e10 outliers)
        assert!(result.n_feasible <= 5);
        assert!(result.feasible_max < 1e10);
    }

    #[test]
    fn test_penalty_config() {
        let config = PenaltyConfig::default().with_min(1e4).with_margin(5.0);

        assert!((config.min_penalty - 1e4).abs() < 1e-10);
        assert!((config.safety_margin - 5.0).abs() < 1e-10);
    }
}

// SPDX-License-Identifier: AGPL-3.0-or-later
//! Optimization convergence diagnostics
//!
//! Provides tools for detecting convergence, stagnation, and oscillation
//! in optimization trajectories. Useful for:
//! - Early stopping decisions
//! - Detecting when to restart with different parameters
//! - Quality assessment of optimization runs
//!
//! # Reference
//!
//! hotSpring validation: `stats.rs::convergence_diagnostics()`

use crate::error::{BarracudaError, Result};

/// Convergence state of an optimization run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConvergenceState {
    /// Making progress (f values decreasing)
    Improving,
    /// No significant change (stagnation)
    Stagnant,
    /// Values oscillating without trend
    Oscillating,
    /// Diverging (f values increasing)
    Diverging,
    /// Insufficient data to assess
    Unknown,
}

/// Convergence diagnostics result.
#[derive(Debug, Clone)]
pub struct ConvergenceDiagnostics {
    /// Current convergence state
    pub state: ConvergenceState,
    /// Recent improvement rate (negative = improving)
    pub improvement_rate: f64,
    /// Number of consecutive non-improving iterations
    pub stagnant_count: usize,
    /// Best value seen
    pub best_f: f64,
    /// Current value
    pub current_f: f64,
    /// Relative change from best
    pub relative_gap: f64,
    /// Recommendation
    pub should_continue: bool,
}

impl ConvergenceDiagnostics {
    /// Get a human-readable summary.
    pub fn summary(&self) -> String {
        let state_str = match self.state {
            ConvergenceState::Improving => "IMPROVING ↓",
            ConvergenceState::Stagnant => "STAGNANT ●",
            ConvergenceState::Oscillating => "OSCILLATING ↔",
            ConvergenceState::Diverging => "DIVERGING ↑",
            ConvergenceState::Unknown => "UNKNOWN ?",
        };

        format!(
            "Convergence: {}\n\
             Best: {:.6}, Current: {:.6}\n\
             Improvement rate: {:.2e}\n\
             Stagnant iterations: {}\n\
             Continue: {}",
            state_str,
            self.best_f,
            self.current_f,
            self.improvement_rate,
            self.stagnant_count,
            if self.should_continue { "yes" } else { "no" }
        )
    }
}

/// Analyze convergence from a history of objective values.
///
/// # Arguments
///
/// * `history` - Sequence of objective values (oldest to newest)
/// * `window` - Number of recent values to consider for trend (default: 5)
/// * `improvement_threshold` - Minimum improvement to count as progress
/// * `patience` - Number of stagnant iterations before recommending stop
///
/// # Returns
///
/// [`ConvergenceDiagnostics`] with state assessment and recommendation.
///
/// # Example
///
/// ```
/// use barracuda::optimize::convergence_diagnostics;
///
/// let history = vec![100.0, 50.0, 25.0, 24.0, 23.9, 23.85];
///
/// let diag = convergence_diagnostics(&history, 5, 0.01, 3).unwrap();
///
/// println!("{}", diag.summary());
/// ```
///
/// # Reference
///
/// hotSpring validation: `stats.rs::convergence_diagnostics()`
pub fn convergence_diagnostics(
    history: &[f64],
    window: usize,
    improvement_threshold: f64,
    patience: usize,
) -> Result<ConvergenceDiagnostics> {
    let n = history.len();
    if n == 0 {
        return Err(BarracudaError::InvalidInput {
            message: "history cannot be empty".to_string(),
        });
    }

    // Find best value
    let best_f = history.iter().cloned().fold(f64::INFINITY, f64::min);
    let current_f = history[n - 1];

    // Relative gap from best
    let relative_gap = if best_f.abs() > 1e-14 {
        (current_f - best_f) / best_f.abs()
    } else {
        current_f - best_f
    };

    // Not enough data
    if n < 2 {
        return Ok(ConvergenceDiagnostics {
            state: ConvergenceState::Unknown,
            improvement_rate: 0.0,
            stagnant_count: 0,
            best_f,
            current_f,
            relative_gap,
            should_continue: true,
        });
    }

    // Analyze recent window
    let window = window.min(n);
    let recent = &history[n - window..];

    // Compute improvement rate (linear regression slope)
    let improvement_rate = linear_slope(recent);

    // Count stagnant iterations
    let mut stagnant_count = 0;
    let mut prev = history[0];
    for &val in history.iter().skip(1) {
        let improvement = prev - val;
        if improvement.abs() < improvement_threshold {
            stagnant_count += 1;
        } else {
            stagnant_count = 0;
        }
        prev = val;
    }

    // Detect oscillation (variance much larger than trend)
    let mean: f64 = recent.iter().sum::<f64>() / window as f64;
    let variance: f64 = recent.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / window as f64;
    let std_dev = variance.sqrt();

    // Determine state
    let state = if improvement_rate < -improvement_threshold {
        ConvergenceState::Improving
    } else if improvement_rate > improvement_threshold {
        ConvergenceState::Diverging
    } else if std_dev > improvement_threshold * 10.0
        && improvement_rate.abs() < improvement_threshold
    {
        ConvergenceState::Oscillating
    } else if stagnant_count >= patience {
        ConvergenceState::Stagnant
    } else {
        ConvergenceState::Stagnant
    };

    // Recommendation
    let should_continue = match state {
        ConvergenceState::Improving => true,
        ConvergenceState::Stagnant => stagnant_count < patience,
        ConvergenceState::Oscillating => stagnant_count < patience * 2,
        ConvergenceState::Diverging => false,
        ConvergenceState::Unknown => true,
    };

    Ok(ConvergenceDiagnostics {
        state,
        improvement_rate,
        stagnant_count,
        best_f,
        current_f,
        relative_gap,
        should_continue,
    })
}

/// Compute linear regression slope for a sequence.
fn linear_slope(values: &[f64]) -> f64 {
    let n = values.len() as f64;
    if n < 2.0 {
        return 0.0;
    }

    // Simple linear regression: y = a + b*x
    // b = Σ(x - x̄)(y - ȳ) / Σ(x - x̄)²

    let x_mean = (n - 1.0) / 2.0;
    let y_mean: f64 = values.iter().sum::<f64>() / n;

    let mut numerator = 0.0;
    let mut denominator = 0.0;

    for (i, &y) in values.iter().enumerate() {
        let x = i as f64;
        numerator += (x - x_mean) * (y - y_mean);
        denominator += (x - x_mean).powi(2);
    }

    if denominator.abs() < 1e-14 {
        0.0
    } else {
        numerator / denominator
    }
}

/// Check if optimization should terminate early.
///
/// Convenience function that returns true if the optimizer should stop.
///
/// # Example
///
/// ```
/// use barracuda::optimize::should_stop_early;
///
/// // Clearly stagnant - no improvement
/// let stagnant = vec![50.0, 50.0, 50.0, 50.0, 50.0];
/// assert!(should_stop_early(&stagnant, 0.1, 3));
///
/// // Clearly improving - don't stop
/// let improving = vec![100.0, 50.0, 25.0, 12.0];
/// assert!(!should_stop_early(&improving, 0.1, 3));
/// ```
pub fn should_stop_early(history: &[f64], improvement_threshold: f64, patience: usize) -> bool {
    if history.len() < patience + 1 {
        return false;
    }

    match convergence_diagnostics(history, patience + 1, improvement_threshold, patience) {
        Ok(diag) => !diag.should_continue,
        Err(_) => false,
    }
}

#[expect(clippy::unwrap_used, reason = "tests")]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_improving() {
        let history = vec![100.0, 50.0, 25.0, 12.0, 6.0, 3.0];
        let diag = convergence_diagnostics(&history, 5, 0.1, 3).unwrap();

        assert_eq!(diag.state, ConvergenceState::Improving);
        assert!(diag.improvement_rate < 0.0);
        assert!(diag.should_continue);
    }

    #[test]
    fn test_stagnant() {
        let history = vec![100.0, 50.0, 50.001, 50.0, 49.999, 50.0];
        let diag = convergence_diagnostics(&history, 5, 0.1, 3).unwrap();

        assert_eq!(diag.state, ConvergenceState::Stagnant);
        assert!(!diag.should_continue || diag.stagnant_count < 3);
    }

    #[test]
    fn test_diverging() {
        let history = vec![1.0, 2.0, 4.0, 8.0, 16.0];
        let diag = convergence_diagnostics(&history, 5, 0.1, 3).unwrap();

        assert_eq!(diag.state, ConvergenceState::Diverging);
        assert!(!diag.should_continue);
    }

    #[test]
    fn test_should_stop_early() {
        // Clear improvement - don't stop
        let improving = vec![100.0, 50.0, 25.0, 12.0];
        assert!(!should_stop_early(&improving, 0.1, 3));

        // Stagnant - stop
        let stagnant = vec![50.0, 50.0, 50.0, 50.0, 50.0];
        assert!(should_stop_early(&stagnant, 0.1, 3));
    }

    #[test]
    fn test_linear_slope() {
        // Constant - slope 0
        let constant = vec![5.0, 5.0, 5.0];
        assert!((linear_slope(&constant) - 0.0).abs() < 1e-10);

        // Linear increasing - positive slope
        let increasing = vec![0.0, 1.0, 2.0, 3.0];
        assert!((linear_slope(&increasing) - 1.0).abs() < 1e-10);

        // Linear decreasing - negative slope
        let decreasing = vec![3.0, 2.0, 1.0, 0.0];
        assert!((linear_slope(&decreasing) - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_summary() {
        let history = vec![100.0, 50.0, 25.0];
        let diag = convergence_diagnostics(&history, 3, 0.1, 3).unwrap();
        let summary = diag.summary();

        assert!(summary.contains("Best:"));
        assert!(summary.contains("Current:"));
    }
}

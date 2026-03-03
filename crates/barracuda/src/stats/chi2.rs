// SPDX-License-Identifier: AGPL-3.0-or-later
//! Chi-squared decomposition and analysis
//!
//! Provides detailed per-datum chi-squared analysis beyond the simple
//! `chi_squared_test()` function in `special::chi_squared`.
//!
//! # Features
//!
//! - Per-datum residuals and contributions
//! - Pull values (standardized residuals)
//! - Worst-N identification
//! - Human-readable summary
//!
//! # Reference
//!
//! hotSpring validation: `stats.rs::chi2_decomposed()`

use crate::error::{BarracudaError, Result};
use crate::special::chi_squared_sf;

/// Result of chi-squared decomposition.
#[derive(Debug, Clone)]
pub struct Chi2Decomposed {
    /// Total chi-squared statistic
    pub chi2_total: f64,
    /// Chi-squared per datum (chi2_total / n)
    pub chi2_per_datum: f64,
    /// Chi-squared per degree of freedom (chi2_total / dof)
    pub chi2_per_dof: f64,
    /// Degrees of freedom
    pub dof: usize,
    /// Number of data points
    pub n_data: usize,
    /// Per-datum contributions (chi2_i = (O_i - E_i)² / E_i)
    pub contributions: Vec<f64>,
    /// Residuals (O_i - E_i)
    pub residuals: Vec<f64>,
    /// Pull values ((O_i - E_i) / σ_i where σ_i = √E_i)
    pub pulls: Vec<f64>,
    /// P-value (probability of observing chi2 >= chi2_total under null hypothesis)
    pub p_value: f64,
}

impl Chi2Decomposed {
    /// Get indices of N worst-fitting points (highest contributions).
    pub fn worst_n(&self, n: usize) -> Vec<usize> {
        let mut indexed: Vec<(usize, f64)> = self
            .contributions
            .iter()
            .enumerate()
            .map(|(i, &c)| (i, c))
            .collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed.into_iter().take(n).map(|(i, _)| i).collect()
    }

    /// Get a human-readable summary.
    pub fn summary(&self) -> String {
        let verdict = if self.p_value > 0.05 {
            "ACCEPTABLE"
        } else if self.p_value > 0.01 {
            "MARGINAL"
        } else {
            "POOR"
        };

        format!(
            "Chi² Analysis:\n\
             ─────────────────────────────────\n\
             χ²/datum:    {:.4}\n\
             χ²/dof:      {:.4}  (ideal ≈ 1.0)\n\
             χ² total:    {:.4}\n\
             dof:         {}\n\
             p-value:     {:.4}  → {}\n\
             n_data:      {}\n\
             worst pull:  {:.2}σ",
            self.chi2_per_datum,
            self.chi2_per_dof,
            self.chi2_total,
            self.dof,
            self.p_value,
            verdict,
            self.n_data,
            self.pulls.iter().map(|p| p.abs()).fold(0.0, f64::max)
        )
    }
}

/// Compute decomposed chi-squared statistics.
///
/// Provides detailed per-datum analysis including residuals, pulls (standardized
/// residuals), and individual contributions to the total chi-squared.
///
/// # Arguments
///
/// * `observed` - Observed values
/// * `expected` - Expected values (model predictions)
/// * `n_params` - Number of fitted parameters (for dof calculation)
///
/// # Returns
///
/// [`Chi2Decomposed`] with total, per-datum, residuals, pulls, and p-value.
///
/// # Example
///
/// ```
/// use barracuda::stats::chi2_decomposed;
///
/// let observed = vec![10.0, 15.0, 12.0, 8.0];
/// let expected = vec![11.0, 14.0, 13.0, 9.0];
///
/// let result = chi2_decomposed(&observed, &expected, 1).unwrap();
///
/// println!("{}", result.summary());
/// println!("Worst point: index {}", result.worst_n(1)[0]);
/// ```
///
/// # Reference
///
/// hotSpring validation: `stats.rs::chi2_decomposed()`
pub fn chi2_decomposed(
    observed: &[f64],
    expected: &[f64],
    n_params: usize,
) -> Result<Chi2Decomposed> {
    let n = observed.len();
    if n != expected.len() {
        return Err(BarracudaError::InvalidInput {
            message: format!(
                "observed and expected must have same length: {} vs {}",
                n,
                expected.len()
            ),
        });
    }
    if n == 0 {
        return Err(BarracudaError::InvalidInput {
            message: "observed and expected cannot be empty".to_string(),
        });
    }

    let dof = n.saturating_sub(n_params);
    if dof == 0 {
        return Err(BarracudaError::InvalidInput {
            message: "degrees of freedom must be > 0".to_string(),
        });
    }

    let mut chi2_total = 0.0;
    let mut contributions = Vec::with_capacity(n);
    let mut residuals = Vec::with_capacity(n);
    let mut pulls = Vec::with_capacity(n);

    for i in 0..n {
        let o = observed[i];
        let e = expected[i];

        if e <= 0.0 {
            return Err(BarracudaError::InvalidInput {
                message: format!("expected[{i}] = {e} must be > 0"),
            });
        }

        let residual = o - e;
        let sigma = e.sqrt();
        let pull = residual / sigma;
        let contribution = (residual * residual) / e;

        residuals.push(residual);
        pulls.push(pull);
        contributions.push(contribution);
        chi2_total += contribution;
    }

    // P-value: probability of chi2 >= chi2_total under null hypothesis
    // P(χ² ≥ x | k) = 1 - CDF(x, k) = survival function = Q(k/2, x/2)
    let p_value = chi_squared_sf(chi2_total, dof as f64)?;

    Ok(Chi2Decomposed {
        chi2_total,
        chi2_per_datum: chi2_total / n as f64,
        chi2_per_dof: chi2_total / dof as f64,
        dof,
        n_data: n,
        contributions,
        residuals,
        pulls,
        p_value,
    })
}

/// Compute chi-squared with uncertainties (weighted least squares).
///
/// When uncertainties σ_i are known for each observation, use weighted chi-squared:
/// χ²_i = (O_i - E_i)² / σ_i²
///
/// # Arguments
///
/// * `observed` - Observed values
/// * `expected` - Expected values (model predictions)
/// * `uncertainties` - Standard errors for each observation
/// * `n_params` - Number of fitted parameters
///
/// # Example
///
/// ```
/// use barracuda::stats::chi2_decomposed_weighted;
///
/// let observed = vec![10.0, 15.0, 12.0];
/// let expected = vec![11.0, 14.0, 13.0];
/// let errors = vec![1.0, 0.5, 0.8];  // Known uncertainties
///
/// let result = chi2_decomposed_weighted(&observed, &expected, &errors, 1).unwrap();
/// println!("χ²/dof = {:.2}", result.chi2_per_dof);
/// ```
pub fn chi2_decomposed_weighted(
    observed: &[f64],
    expected: &[f64],
    uncertainties: &[f64],
    n_params: usize,
) -> Result<Chi2Decomposed> {
    let n = observed.len();
    if n != expected.len() || n != uncertainties.len() {
        return Err(BarracudaError::InvalidInput {
            message: "observed, expected, and uncertainties must have same length".to_string(),
        });
    }
    if n == 0 {
        return Err(BarracudaError::InvalidInput {
            message: "arrays cannot be empty".to_string(),
        });
    }

    let dof = n.saturating_sub(n_params);
    if dof == 0 {
        return Err(BarracudaError::InvalidInput {
            message: "degrees of freedom must be > 0".to_string(),
        });
    }

    let mut chi2_total = 0.0;
    let mut contributions = Vec::with_capacity(n);
    let mut residuals = Vec::with_capacity(n);
    let mut pulls = Vec::with_capacity(n);

    for i in 0..n {
        let o = observed[i];
        let e = expected[i];
        let sigma = uncertainties[i];

        if sigma <= 0.0 {
            return Err(BarracudaError::InvalidInput {
                message: format!("uncertainties[{i}] = {sigma} must be > 0"),
            });
        }

        let residual = o - e;
        let pull = residual / sigma;
        let contribution = pull * pull;

        residuals.push(residual);
        pulls.push(pull);
        contributions.push(contribution);
        chi2_total += contribution;
    }

    let p_value = chi_squared_sf(chi2_total, dof as f64)?;

    Ok(Chi2Decomposed {
        chi2_total,
        chi2_per_datum: chi2_total / n as f64,
        chi2_per_dof: chi2_total / dof as f64,
        dof,
        n_data: n,
        contributions,
        residuals,
        pulls,
        p_value,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chi2_decomposed_basic() {
        let observed = vec![10.0, 20.0, 30.0, 40.0];
        let expected = vec![11.0, 19.0, 31.0, 39.0];

        let result = chi2_decomposed(&observed, &expected, 0).unwrap();

        assert_eq!(result.n_data, 4);
        assert_eq!(result.dof, 4);
        assert_eq!(result.residuals.len(), 4);
        assert_eq!(result.pulls.len(), 4);
        assert_eq!(result.contributions.len(), 4);

        // Verify chi2_total is sum of contributions
        let sum: f64 = result.contributions.iter().sum();
        assert!((result.chi2_total - sum).abs() < 1e-10);

        // Verify chi2_per_datum
        assert!((result.chi2_per_datum - result.chi2_total / 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_chi2_decomposed_perfect_fit() {
        let data = vec![10.0, 20.0, 30.0];
        let result = chi2_decomposed(&data, &data, 0).unwrap();

        assert!((result.chi2_total - 0.0).abs() < 1e-10);
        assert!(result.p_value > 0.99); // Perfect fit = high p-value
    }

    #[test]
    fn test_chi2_decomposed_with_params() {
        let observed = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let expected = vec![11.0, 19.0, 31.0, 39.0, 51.0];

        // With 2 fitted parameters
        let result = chi2_decomposed(&observed, &expected, 2).unwrap();

        assert_eq!(result.dof, 3); // 5 - 2 = 3
    }

    #[test]
    fn test_chi2_worst_n() {
        let observed = vec![10.0, 20.0, 100.0, 40.0]; // 100 is outlier
        let expected = vec![11.0, 19.0, 30.0, 39.0];

        let result = chi2_decomposed(&observed, &expected, 0).unwrap();
        let worst = result.worst_n(1);

        assert_eq!(worst[0], 2); // Index 2 has the outlier
    }

    #[test]
    fn test_chi2_summary() {
        let observed = vec![10.0, 20.0, 30.0];
        let expected = vec![11.0, 19.0, 31.0];

        let result = chi2_decomposed(&observed, &expected, 0).unwrap();
        let summary = result.summary();

        assert!(summary.contains("χ²/datum"));
        assert!(summary.contains("p-value"));
    }

    #[test]
    fn test_chi2_weighted() {
        let observed = vec![10.0, 20.0, 30.0];
        let expected = vec![11.0, 19.0, 31.0];
        let errors = vec![1.0, 0.5, 2.0];

        let result = chi2_decomposed_weighted(&observed, &expected, &errors, 0).unwrap();

        // Pull = residual / sigma
        assert!((result.pulls[0] - (-1.0 / 1.0)).abs() < 1e-10);
        assert!((result.pulls[1] - (1.0 / 0.5)).abs() < 1e-10);
    }

    #[test]
    fn test_chi2_errors() {
        // Mismatched lengths
        assert!(chi2_decomposed(&[1.0, 2.0], &[1.0], 0).is_err());

        // Empty arrays
        assert!(chi2_decomposed(&[], &[], 0).is_err());

        // Zero expected
        assert!(chi2_decomposed(&[1.0], &[0.0], 0).is_err());

        // Zero dof
        assert!(chi2_decomposed(&[1.0], &[1.0], 1).is_err());
    }
}

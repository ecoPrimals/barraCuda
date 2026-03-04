// SPDX-License-Identifier: AGPL-3.0-or-later

//! Ridge regression via Cholesky decomposition.
//!
//! Solves `(XᵀX + λI)w = Xᵀy` for each output column of `y`.
//!
//! Uses the existing [`CholeskyDecomposition`] for the normal equations.
//! Falls back to diagonal solve when the Gram matrix is not positive definite
//! (degenerate data or extreme regularization).
//!
//! # Provenance
//! Absorbed from wetSpring ESN readout training (`bio/esn.rs`, Feb 2026).
//!
//! [`CholeskyDecomposition`]: super::cholesky::CholeskyDecomposition

use crate::error::{BarracudaError, Result};

/// Result of ridge regression fitting.
#[derive(Debug, Clone)]
pub struct RidgeResult {
    /// Weight matrix (n_outputs × n_features, row-major).
    pub weights: Vec<f64>,
    /// Number of features (columns of X).
    pub n_features: usize,
    /// Number of output dimensions.
    pub n_outputs: usize,
}

impl RidgeResult {
    /// Predict outputs for new data points.
    ///
    /// `x` is `n_samples × n_features` (row-major).
    /// Returns `n_samples × n_outputs` (row-major).
    pub fn predict(&self, x: &[f64], n_samples: usize) -> Result<Vec<f64>> {
        if x.len() != n_samples * self.n_features {
            return Err(BarracudaError::InvalidInput {
                message: format!(
                    "x has {} elements, expected {} × {} = {}",
                    x.len(),
                    n_samples,
                    self.n_features,
                    n_samples * self.n_features
                ),
            });
        }
        let mut out = vec![0.0; n_samples * self.n_outputs];
        for s in 0..n_samples {
            let row = &x[s * self.n_features..(s + 1) * self.n_features];
            for o in 0..self.n_outputs {
                let w = &self.weights[o * self.n_features..(o + 1) * self.n_features];
                out[s * self.n_outputs + o] = row.iter().zip(w).map(|(a, b)| a * b).sum();
            }
        }
        Ok(out)
    }
}

/// Solve ridge regression: `(XᵀX + λI)w = Xᵀy`.
///
/// - `x`: design matrix, `n_samples × n_features` (row-major)
/// - `y`: target matrix, `n_samples × n_outputs` (row-major)
/// - `regularization`: Tikhonov parameter λ ≥ 0
///
/// Returns a [`RidgeResult`] containing the fitted weight matrix.
pub fn ridge_regression(
    x: &[f64],
    y: &[f64],
    n_samples: usize,
    n_features: usize,
    n_outputs: usize,
    regularization: f64,
) -> Result<RidgeResult> {
    if x.len() != n_samples * n_features {
        return Err(BarracudaError::InvalidInput {
            message: format!(
                "x has {} elements, expected {} × {} = {}",
                x.len(),
                n_samples,
                n_features,
                n_samples * n_features
            ),
        });
    }
    if y.len() != n_samples * n_outputs {
        return Err(BarracudaError::InvalidInput {
            message: format!(
                "y has {} elements, expected {} × {} = {}",
                y.len(),
                n_samples,
                n_outputs,
                n_samples * n_outputs
            ),
        });
    }
    if regularization < 0.0 {
        return Err(BarracudaError::InvalidInput {
            message: format!("regularization must be ≥ 0, got {regularization}"),
        });
    }

    // Gram matrix: XᵀX + λI  (n_features × n_features)
    let nf = n_features;
    let mut gram = vec![0.0_f64; nf * nf];
    for s in 0..n_samples {
        let row = &x[s * nf..(s + 1) * nf];
        for i in 0..nf {
            for j in i..nf {
                let v = row[i] * row[j];
                gram[i * nf + j] += v;
                if i != j {
                    gram[j * nf + i] += v;
                }
            }
        }
    }
    for i in 0..nf {
        gram[i * nf + i] += regularization;
    }

    let chol = cholesky_factor(&gram, nf);

    let mut weights = vec![0.0; n_outputs * nf];

    for o in 0..n_outputs {
        // Xᵀy_o
        let mut xty = vec![0.0_f64; nf];
        for s in 0..n_samples {
            let y_val = y[s * n_outputs + o];
            let row = &x[s * nf..(s + 1) * nf];
            for r in 0..nf {
                xty[r] += row[r] * y_val;
            }
        }

        match &chol {
            Some(l) => {
                // Forward: Lz = Xᵀy
                let mut z = vec![0.0; nf];
                for i in 0..nf {
                    let mut sum = 0.0;
                    for j in 0..i {
                        sum += l[i * nf + j] * z[j];
                    }
                    z[i] = (xty[i] - sum) / l[i * nf + i];
                }
                // Backward: Lᵀw = z
                for i in (0..nf).rev() {
                    let mut sum = 0.0;
                    for j in (i + 1)..nf {
                        sum += l[j * nf + i] * weights[o * nf + j];
                    }
                    weights[o * nf + i] = (z[i] - sum) / l[i * nf + i];
                }
            }
            None => {
                for r in 0..nf {
                    let diag = gram[r * nf + r];
                    weights[o * nf + r] = if diag.abs() > 1e-15 {
                        xty[r] / diag
                    } else {
                        0.0
                    };
                }
            }
        }
    }

    Ok(RidgeResult {
        weights,
        n_features: nf,
        n_outputs,
    })
}

/// In-place Cholesky factorization. Returns lower triangular L where A = LLᵀ,
/// or `None` if a leading minor is non-positive.
fn cholesky_factor(a: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut l = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = 0.0;
            for k in 0..j {
                sum += l[i * n + k] * l[j * n + k];
            }
            if i == j {
                let diag = a[i * n + i] - sum;
                if diag <= 0.0 {
                    return None;
                }
                l[i * n + j] = diag.sqrt();
            } else {
                l[i * n + j] = (a[i * n + j] - sum) / l[j * n + j];
            }
        }
    }
    Some(l)
}

#[expect(clippy::unwrap_used, reason = "tests")]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ridge_identity_regression() {
        // y = x (identity mapping), should recover w ≈ I
        let x = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let y = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let result = ridge_regression(&x, &y, 3, 2, 2, 1e-6).unwrap();
        assert_eq!(result.n_features, 2);
        assert_eq!(result.n_outputs, 2);
        // w should be close to identity
        assert!((result.weights[0] - 1.0).abs() < 0.01);
        assert!(result.weights[1].abs() < 0.01);
        assert!(result.weights[2].abs() < 0.01);
        assert!((result.weights[3] - 1.0).abs() < 0.01);
    }

    #[test]
    fn ridge_simple_linear() {
        // y = 2*x1 + 3*x2
        let x = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0];
        let y = vec![2.0, 3.0, 5.0, 7.0];
        let result = ridge_regression(&x, &y, 4, 2, 1, 1e-8).unwrap();
        assert!((result.weights[0] - 2.0).abs() < 0.01);
        assert!((result.weights[1] - 3.0).abs() < 0.01);
    }

    #[test]
    fn ridge_regularization_shrinks_weights() {
        let x = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let y = vec![100.0, 100.0, 200.0];
        let r_small = ridge_regression(&x, &y, 3, 2, 1, 0.001).unwrap();
        let r_large = ridge_regression(&x, &y, 3, 2, 1, 100.0).unwrap();
        let norm_small: f64 = r_small.weights.iter().map(|w| w * w).sum();
        let norm_large: f64 = r_large.weights.iter().map(|w| w * w).sum();
        assert!(
            norm_large < norm_small,
            "larger λ should shrink weights: small={norm_small:.2}, large={norm_large:.2}"
        );
    }

    #[test]
    fn ridge_predict() {
        let x = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let y = vec![2.0, 3.0, 5.0];
        let result = ridge_regression(&x, &y, 3, 2, 1, 1e-8).unwrap();
        let pred = result.predict(&[2.0, 3.0], 1).unwrap();
        // y ≈ 2*x1 + 3*x2 → pred(2,3) ≈ 13
        assert!((pred[0] - 13.0).abs() < 0.1);
    }

    #[test]
    fn ridge_multi_output() {
        // y1 = x, y2 = 2*x
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let y = vec![1.0, 2.0, 2.0, 4.0, 3.0, 6.0, 4.0, 8.0];
        let result = ridge_regression(&x, &y, 4, 1, 2, 1e-8).unwrap();
        assert!((result.weights[0] - 1.0).abs() < 0.01);
        assert!((result.weights[1] - 2.0).abs() < 0.01);
    }

    #[test]
    fn ridge_invalid_dimensions_rejected() {
        let x = vec![1.0, 2.0];
        let y = vec![1.0, 2.0, 3.0];
        let result = ridge_regression(&x, &y, 2, 1, 1, 0.01);
        assert!(result.is_err());
    }
}

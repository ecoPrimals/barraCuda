// SPDX-License-Identifier: AGPL-3.0-or-later
//! Ridge regression readout for reservoir outputs.

#[cfg(feature = "gpu")]
use crate::linalg::solve_f64_cpu;
use serde::{Deserialize, Serialize};

use crate::error::Result;

/// Linear readout with ridge regression.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearReadout {
    /// Trained weights (None until trained).
    pub weights: Option<Vec<f64>>,
    /// Input feature dimension.
    pub input_dim: usize,
    /// Output dimension.
    pub output_dim: usize,
    /// Ridge regularization parameter.
    pub lambda: f64,
}

impl LinearReadout {
    /// Create a new readout. Lambda defaults to 1e-3.
    #[must_use]
    pub fn new(input_dim: usize, output_dim: usize, lambda: f64) -> Self {
        Self {
            weights: None,
            input_dim,
            output_dim,
            lambda: if lambda > 0.0 { lambda } else { 1e-3 },
        }
    }

    /// Train via ridge regression: (X^T X + λI)^-1 X^T y. Returns MSE.
    /// # Errors
    /// Returns [`Err`] if the linear solve fails (e.g. singular matrix).
    #[cfg(feature = "gpu")]
    pub fn train(&mut self, responses: &[Vec<f64>], targets: &[Vec<f64>]) -> Result<f64> {
        let n = responses.len();
        if n == 0 || n != targets.len() {
            return Ok(f64::MAX);
        }
        let nf = self.input_dim;
        let no = self.output_dim;

        let mut x = vec![0.0; n * nf];
        for (i, r) in responses.iter().enumerate() {
            let len = r.len().min(nf);
            x[i * nf..i * nf + len].copy_from_slice(&r[..len]);
        }

        let mut gram = vec![0.0; nf * nf];
        for s in 0..n {
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
            gram[i * nf + i] += self.lambda;
        }

        let mut w = vec![0.0; no * nf];
        for o in 0..no {
            let mut xty = vec![0.0; nf];
            for s in 0..n {
                let y_val = targets[s].get(o).copied().unwrap_or(0.0);
                let row = &x[s * nf..(s + 1) * nf];
                for r in 0..nf {
                    xty[r] += row[r] * y_val;
                }
            }
            let col = solve_f64_cpu(&gram, &xty, nf)?;
            w[o * nf..(o + 1) * nf].copy_from_slice(&col);
        }

        self.weights = Some(w);

        let mut mse = 0.0;
        for (i, r) in responses.iter().enumerate() {
            if let Some(pred) = self.predict(r) {
                for (j, &t) in targets[i].iter().enumerate().take(no) {
                    let p = pred.get(j).copied().unwrap_or(0.0);
                    mse += (p - t) * (p - t);
                }
            }
        }
        Ok(if n > 0 { mse / (n * no) as f64 } else { 0.0 })
    }

    /// Train via ridge regression on CPU using [`crate::linalg::ridge_regression`]
    /// (the `gpu` build uses a Gram-matrix + [`solve_f64_cpu`](crate::linalg::solve_f64_cpu) path instead).
    /// Returns mean squared error over the training set.
    /// # Errors
    /// Returns [`Err`] if ridge regression fails (e.g. degenerate data).
    #[cfg(not(feature = "gpu"))]
    pub fn train(&mut self, responses: &[Vec<f64>], targets: &[Vec<f64>]) -> Result<f64> {
        use crate::linalg::ridge_regression;

        let n = responses.len();
        if n == 0 || n != targets.len() {
            return Ok(f64::MAX);
        }
        let nf = self.input_dim;
        let no = self.output_dim;

        let mut x = vec![0.0; n * nf];
        for (i, r) in responses.iter().enumerate() {
            let len = r.len().min(nf);
            x[i * nf..i * nf + len].copy_from_slice(&r[..len]);
        }

        let mut y = vec![0.0; n * no];
        for (i, t) in targets.iter().enumerate() {
            for (j, &v) in t.iter().take(no).enumerate() {
                y[i * no + j] = v;
            }
        }

        let result = ridge_regression(&x, &y, n, nf, no, self.lambda)?;
        self.weights = Some(result.weights);

        let mut mse = 0.0;
        for (i, r) in responses.iter().enumerate() {
            if let Some(pred) = self.predict(r) {
                for (j, &t) in targets[i].iter().enumerate().take(no) {
                    let p = pred.get(j).copied().unwrap_or(0.0);
                    mse += (p - t) * (p - t);
                }
            }
        }
        Ok(if n > 0 { mse / (n * no) as f64 } else { 0.0 })
    }

    /// Predict: W·x.
    #[must_use]
    pub fn predict(&self, response: &[f64]) -> Option<Vec<f64>> {
        let w = self.weights.as_ref()?;
        let nf = self.input_dim;
        let no = self.output_dim;
        let mut out = vec![0.0; no];
        for o in 0..no {
            for r in 0..nf.min(response.len()) {
                out[o] += w[o * nf + r] * response[r];
            }
        }
        Some(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_defaults() {
        let r = LinearReadout::new(4, 2, 1e-3);
        assert_eq!(r.input_dim, 4);
        assert_eq!(r.output_dim, 2);
        assert!(r.weights.is_none());
        assert!((r.lambda - 1e-3).abs() < 1e-15);
    }

    #[test]
    fn negative_lambda_clamps() {
        let r = LinearReadout::new(4, 2, -1.0);
        assert!((r.lambda - 1e-3).abs() < 1e-15);
    }

    #[test]
    fn predict_untrained_returns_none() {
        let r = LinearReadout::new(3, 1, 1e-3);
        assert!(r.predict(&[1.0, 2.0, 3.0]).is_none());
    }

    #[test]
    fn predict_with_known_weights() {
        let mut r = LinearReadout::new(3, 2, 1e-3);
        r.weights = Some(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
        let pred = r.predict(&[3.0, 7.0, 11.0]).unwrap();
        assert!((pred[0] - 3.0).abs() < 1e-12);
        assert!((pred[1] - 7.0).abs() < 1e-12);
    }

    #[test]
    fn train_identity_mapping() {
        let mut r = LinearReadout::new(2, 2, 1e-6);
        let responses = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
        let targets = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
        let mse = r.train(&responses, &targets).unwrap();
        assert!(mse < 0.1, "MSE should be small for identity mapping: {mse}");
        assert!(r.weights.is_some());
    }

    #[test]
    fn train_empty_returns_max() {
        let mut r = LinearReadout::new(2, 1, 1e-3);
        let mse = r.train(&[], &[]).unwrap();
        assert_eq!(mse, f64::MAX);
    }

    #[test]
    fn predict_shorter_input_pads_zeros() {
        let mut r = LinearReadout::new(4, 1, 1e-3);
        r.weights = Some(vec![1.0, 2.0, 3.0, 4.0]);
        let pred = r.predict(&[1.0, 1.0]).unwrap();
        assert!((pred[0] - 3.0).abs() < 1e-12);
    }
}

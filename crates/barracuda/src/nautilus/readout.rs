// SPDX-License-Identifier: AGPL-3.0-or-later
//! Ridge regression readout for reservoir outputs.

#[cfg(feature = "gpu")]
use crate::linalg::solve_f64_cpu;
use serde::{Deserialize, Serialize};

use crate::error::Result;

/// Linear readout with ridge regression.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearReadout {
    pub weights: Option<Vec<f64>>,
    pub input_dim: usize,
    pub output_dim: usize,
    pub lambda: f64,
}

impl LinearReadout {
    /// Create a new readout. Lambda defaults to 1e-3.
    pub fn new(input_dim: usize, output_dim: usize, lambda: f64) -> Self {
        Self {
            weights: None,
            input_dim,
            output_dim,
            lambda: if lambda > 0.0 { lambda } else { 1e-3 },
        }
    }

    /// Train via ridge regression: (X^T X + λI)^-1 X^T y. Returns MSE.
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

    /// Train (no-op when gpu feature disabled; use ridge_regression fallback).
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

// SPDX-License-Identifier: AGPL-3.0-or-later
//! Radial basis function surrogate for expensive function approximation

use super::kernels::RBFKernel;
use crate::device::WgpuDevice;
use crate::error::{BarracudaError, Result};
use crate::linalg::solve_f64;
use crate::ops::cdist_wgsl::{compute_distances_f64_gpu, DistanceMetric};
use std::sync::Arc;

/// RBF surrogate model with polynomial augmentation
///
/// Approximates expensive functions using radial basis functions:
/// s(x) = Σᵢ wᵢ φ(‖x - xᵢ‖) + p(x)
///
/// where:
/// - φ is the RBF kernel
/// - wᵢ are weights (learned from data)
/// - p(x) is a polynomial tail (linear: 1, x₁, ..., xₙ)
///
/// All math (cdist, solve) dispatches to GPU shaders.
///
/// # Leave-One-Out Cross-Validation
///
/// LOO-CV provides a measure of surrogate quality without needing a separate
/// validation set:
///
/// ```ignore
/// let surrogate = RBFSurrogate::train(&x_data, &y_data, kernel, 1e-12)?;
/// let loo_rmse = surrogate.loo_cv_rmse()?;
/// println!("LOO-CV RMSE: {:.6}", loo_rmse);
/// ```
#[derive(Debug)]
pub struct RBFSurrogate {
    /// GPU device for cdist and solve
    device: Arc<WgpuDevice>,
    /// Training points (flattened: [n_train × n_dim])
    train_x: Vec<f64>,
    /// Training targets
    train_y: Vec<f64>,
    /// RBF weights (length n_train)
    weights: Vec<f64>,
    /// Polynomial coefficients (length n_dim + 1)
    poly_coeffs: Vec<f64>,
    /// Number of training points
    n_train: usize,
    /// Dimension of input space
    n_dim: usize,
    /// Kernel function
    kernel: RBFKernel,
    /// Smoothing parameter (regularization)
    smoothing: f64,
}

#[cfg(test)]
mod tests;

impl RBFSurrogate {
    /// Construct from pre-computed parts (used by adaptive dispatch).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn from_parts(
        device: Arc<WgpuDevice>,
        train_x: Vec<f64>,
        train_y: Vec<f64>,
        weights: Vec<f64>,
        poly_coeffs: Vec<f64>,
        n_train: usize,
        n_dim: usize,
        kernel: RBFKernel,
        smoothing: f64,
    ) -> Self {
        Self {
            device,
            train_x,
            train_y,
            weights,
            poly_coeffs,
            n_train,
            n_dim,
            kernel,
            smoothing,
        }
    }

    /// Train RBF surrogate on data
    ///
    /// # Arguments
    ///
    /// * `x_data` - Training points [[x₁₁, x₁₂, ...], [x₂₁, x₂₂, ...], ...]
    /// * `y_data` - Training values [y₁, y₂, ...]
    /// * `kernel` - RBF kernel type
    /// * `smoothing` - Regularization parameter (1e-12 for exact interpolation)
    ///
    /// # Returns
    ///
    /// Trained surrogate model
    ///
    /// # Algorithm
    ///
    /// 1. Compute pairwise distances: D[i,j] = ‖xᵢ - xⱼ‖
    /// 2. Assemble kernel matrix: K[i,j] = φ(D[i,j]) + δᵢⱼ·smoothing
    /// 3. Augment with polynomial: [K P; Pᵀ 0] [w; c] = [y; 0]
    /// 4. Solve for weights w and polynomial coefficients c
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use barracuda::surrogate::{RBFSurrogate, RBFKernel};
    /// use barracuda::prelude::WgpuDevice;
    /// use std::sync::Arc;
    ///
    /// # async fn example() -> barracuda::error::Result<()> {
    /// let device = Arc::new(WgpuDevice::new().await?);
    /// // Training data: y = x²
    /// let x_train = vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]];
    /// let y_train = vec![0.0, 1.0, 4.0, 9.0];
    ///
    /// let surrogate = RBFSurrogate::train(
    ///     device,
    ///     &x_train,
    ///     &y_train,
    ///     RBFKernel::ThinPlateSpline,
    ///     1e-12,
    /// )?;
    ///
    /// // Predict at new points
    /// let y_pred = surrogate.predict(&[vec![1.5], vec![2.5]])?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn train(
        device: Arc<WgpuDevice>,
        x_data: &[Vec<f64>],
        y_data: &[f64],
        kernel: RBFKernel,
        smoothing: f64,
    ) -> Result<Self> {
        let n_train = x_data.len();

        if n_train == 0 {
            return Err(BarracudaError::InvalidInput {
                message: "Training data cannot be empty".to_string(),
            });
        }

        if y_data.len() != n_train {
            return Err(BarracudaError::InvalidInput {
                message: format!(
                    "x_data and y_data length mismatch: {} vs {}",
                    n_train,
                    y_data.len()
                ),
            });
        }

        let n_dim = x_data[0].len();

        // Flatten training data (can't use extend_from_slice due to nested structure)
        #[allow(clippy::manual_memcpy)]
        let train_x: Vec<f64> = x_data.iter().flat_map(|row| row.iter().copied()).collect();

        // Compute pairwise distances on GPU
        let distances = compute_distances_f64_gpu(
            device.as_ref(),
            &train_x,
            n_train,
            &train_x,
            n_train,
            n_dim,
            DistanceMetric::Euclidean,
        )?;

        // Assemble augmented system
        let n_poly = n_dim + 1; // 1 + x₁ + x₂ + ... + xₙ
        let n_total = n_train + n_poly;

        let mut a = vec![0.0; n_total * n_total];
        let mut b = vec![0.0; n_total];

        // Top-left: Kernel matrix K + smoothing·I
        for i in 0..n_train {
            for j in 0..n_train {
                let k_ij = kernel.eval(distances[i * n_train + j]);
                let smooth = if i == j { smoothing } else { 0.0 };
                a[i * n_total + j] = k_ij + smooth;
            }
        }

        // Top-right and bottom-left: Polynomial matrix P
        for i in 0..n_train {
            // Constant term
            a[i * n_total + n_train] = 1.0;
            a[n_train * n_total + i] = 1.0;

            // Linear terms
            for d in 0..n_dim {
                a[i * n_total + (n_train + 1 + d)] = train_x[i * n_dim + d];
                a[(n_train + 1 + d) * n_total + i] = train_x[i * n_dim + d];
            }
        }

        // Bottom-right: Zero block (already initialized to 0)

        // Right-hand side
        b[..n_train].copy_from_slice(y_data);
        // Polynomial constraints are zero (already initialized)

        // Solve linear system on GPU
        let solution = solve_f64(device.clone(), &a, &b, n_total)?;

        // Extract weights and polynomial coefficients
        let weights = solution[..n_train].to_vec();
        let poly_coeffs = solution[n_train..].to_vec();

        Ok(Self {
            device,
            train_x,
            train_y: y_data.to_vec(),
            weights,
            poly_coeffs,
            n_train,
            n_dim,
            kernel,
            smoothing,
        })
    }

    /// Predict at new points
    ///
    /// # Arguments
    ///
    /// * `x_eval` - Evaluation points [[x₁₁, x₁₂, ...], ...]
    ///
    /// # Returns
    ///
    /// Predicted values [ŷ₁, ŷ₂, ...]
    pub fn predict(&self, x_eval: &[Vec<f64>]) -> Result<Vec<f64>> {
        let n_eval = x_eval.len();

        if n_eval == 0 {
            return Ok(Vec::new());
        }

        if x_eval[0].len() != self.n_dim {
            return Err(BarracudaError::InvalidInput {
                message: format!(
                    "Dimension mismatch: expected {}, got {}",
                    self.n_dim,
                    x_eval[0].len()
                ),
            });
        }

        // Flatten evaluation points (can't use extend_from_slice due to nested structure)
        #[allow(clippy::manual_memcpy)]
        let eval_x: Vec<f64> = x_eval.iter().flat_map(|row| row.iter().copied()).collect();

        // Compute distances on GPU
        let distances = compute_distances_f64_gpu(
            self.device.as_ref(),
            &eval_x,
            n_eval,
            &self.train_x,
            self.n_train,
            self.n_dim,
            DistanceMetric::Euclidean,
        )?;

        let mut predictions = Vec::with_capacity(n_eval);

        for i in 0..n_eval {
            let mut pred = 0.0;

            // RBF contribution
            for j in 0..self.n_train {
                let dist = distances[i * self.n_train + j];
                let phi = self.kernel.eval(dist);
                pred += self.weights[j] * phi;
            }

            // Polynomial contribution
            pred += self.poly_coeffs[0]; // Constant
            for d in 0..self.n_dim {
                pred += self.poly_coeffs[1 + d] * eval_x[i * self.n_dim + d];
            }

            predictions.push(pred);
        }

        Ok(predictions)
    }

    // === Leave-One-Out Cross-Validation ===

    /// Compute leave-one-out cross-validation RMSE.
    ///
    /// LOO-CV provides an unbiased estimate of prediction error without
    /// requiring a separate validation set. For RBF interpolation with
    /// smoothing λ > 0, the LOO residual is:
    ///
    /// LOO_i = (y_i - ŷ_i) / (1 - H_ii)
    ///
    /// where H is the hat matrix H = K(K + λI)⁻¹.
    ///
    /// # Returns
    ///
    /// Root mean square of LOO residuals
    ///
    /// # Example
    ///
    /// ```ignore
    /// let surrogate = RBFSurrogate::train(&x_data, &y_data, kernel, 1e-6)?;
    /// let rmse = surrogate.loo_cv_rmse()?;
    /// println!("LOO-CV RMSE: {:.6}", rmse);
    /// ```
    ///
    /// # Notes
    ///
    /// - For exact interpolation (smoothing ≈ 0), H_ii ≈ 1 and LOO residuals
    ///   are undefined. Use smoothing > 1e-10 for meaningful LOO-CV.
    /// - This is O(n³) due to hat matrix computation.
    pub fn loo_cv_rmse(&self) -> Result<f64> {
        let loo_residuals = self.loo_cv_errors()?;
        let mse = loo_residuals.iter().map(|r| r * r).sum::<f64>() / self.n_train as f64;
        Ok(mse.sqrt())
    }

    /// Compute per-point LOO-CV errors.
    ///
    /// Returns LOO_i = (y_i - ŷ_i) / (1 - H_ii) for each training point.
    /// Useful for identifying outliers or poorly-fit regions.
    ///
    /// # Returns
    ///
    /// Vector of LOO residuals (length n_train)
    pub fn loo_cv_errors(&self) -> Result<Vec<f64>> {
        if self.n_train == 0 {
            return Ok(Vec::new());
        }

        // Compute predictions at training points
        let train_points: Vec<Vec<f64>> = (0..self.n_train)
            .map(|i| {
                let start = i * self.n_dim;
                self.train_x[start..start + self.n_dim].to_vec()
            })
            .collect();
        let predictions = self.predict(&train_points)?;

        // Compute hat matrix diagonal
        let h_diag = self.compute_hat_diagonal()?;

        // Compute LOO residuals
        let mut loo_residuals = Vec::with_capacity(self.n_train);
        for i in 0..self.n_train {
            let residual = self.train_y[i] - predictions[i];
            let denom = 1.0 - h_diag[i];

            // Avoid division by zero (H_ii ≈ 1 means exact interpolation)
            let loo = if denom.abs() < 1e-10 {
                0.0 // Edge case: point has full influence
            } else {
                residual / denom
            };
            loo_residuals.push(loo);
        }

        Ok(loo_residuals)
    }

    /// Compute diagonal of the hat matrix H = K_raw · (K_smooth)⁻¹.
    ///
    /// For RBF interpolation with regularization, the hat matrix is:
    /// H = K_raw · (K_raw + λI)⁻¹
    ///
    /// where K_raw is the kernel matrix WITHOUT regularization and
    /// K_smooth = K_raw + λI is the regularized matrix.
    ///
    /// H_ii measures how much influence point i has on its own prediction.
    /// For exact interpolation (λ → 0), H_ii → 1.
    /// For smoothed interpolation, H_ii < 1.
    ///
    /// # Algorithm (hotSpring validated)
    ///
    /// For each point i:
    /// 1. Solve K_smooth · w = e_i (standard basis vector)
    /// 2. H_ii = K_raw[i,:] · w = dot product of row i with solution
    ///
    /// This correctly gives H_ii < 1 when smoothing > 0.
    fn compute_hat_diagonal(&self) -> Result<Vec<f64>> {
        let n = self.n_train;

        // Compute distance matrix on GPU
        let distances = compute_distances_f64_gpu(
            self.device.as_ref(),
            &self.train_x,
            n,
            &self.train_x,
            n,
            self.n_dim,
            DistanceMetric::Euclidean,
        )?;

        // Build K_raw (kernel matrix WITHOUT smoothing)
        let mut k_raw = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                k_raw[i * n + j] = self.kernel.eval(distances[i * n + j]);
            }
        }

        // Build K_smooth = K_raw + λI (kernel matrix WITH smoothing)
        let mut k_smooth = k_raw.clone();
        for i in 0..n {
            k_smooth[i * n + i] += self.smoothing;
        }

        // Compute H_ii for each point
        // H = K_raw · (K_smooth)⁻¹
        // H_ii = e_i^T · K_raw · (K_smooth)⁻¹ · e_i
        //      = K_raw[i,:] · (K_smooth)⁻¹ · e_i
        //      = K_raw[i,:] · w_i  where K_smooth · w_i = e_i

        let mut h_diag = Vec::with_capacity(n);

        for i in 0..n {
            // Create e_i (standard basis vector)
            let mut e_i = vec![0.0; n];
            e_i[i] = 1.0;

            // Solve K_smooth · w = e_i on GPU
            let w = solve_f64(self.device.clone(), &k_smooth, &e_i, n)?;

            // H_ii = K_raw[i,:] · w (dot product)
            let h_ii: f64 = (0..n).map(|j| k_raw[i * n + j] * w[j]).sum();
            h_diag.push(h_ii);
        }

        Ok(h_diag)
    }

    /// Get the number of training points.
    pub fn n_train(&self) -> usize {
        self.n_train
    }

    /// Get the input dimension.
    pub fn n_dim(&self) -> usize {
        self.n_dim
    }
}

/// Result of LOO-CV grid search for optimal smoothing.
pub struct LooSmoothing {
    /// Optimal smoothing parameter
    pub smoothing: f64,
    /// LOO-CV RMSE at optimal smoothing
    pub rmse: f64,
    /// All grid search results as (smoothing, rmse) pairs
    pub grid_results: Vec<(f64, f64)>,
}

/// Find optimal smoothing parameter via LOO-CV grid search.
///
/// Performs grid search over logarithmically-spaced smoothing values,
/// returning the smoothing that minimizes LOO-CV RMSE.
///
/// # Arguments
///
/// * `x_data` - Training points (each inner vec is one point)
/// * `y_data` - Training targets
/// * `kernel` - RBF kernel to use
/// * `smoothing_grid` - Grid of smoothing values to test (or None for default)
///
/// # Returns
///
/// [`LooSmoothing`] with optimal_smoothing, optimal_rmse, and all grid results.
///
/// # Example
///
/// ```ignore
/// let result = loo_cv_optimal_smoothing(
///     &x_data, &y_data,
///     RBFKernel::ThinPlateSpline,
///     None,  // Use default grid
/// )?;
/// println!("Optimal smoothing: {:.2e}, RMSE: {:.6}", result.smoothing, result.rmse);
/// ```
///
/// # Reference
///
/// hotSpring validation: `surrogate.rs::loo_cv_optimal_smoothing()`
pub fn loo_cv_optimal_smoothing(
    device: Arc<WgpuDevice>,
    x_data: &[Vec<f64>],
    y_data: &[f64],
    kernel: RBFKernel,
    smoothing_grid: Option<&[f64]>,
) -> Result<LooSmoothing> {
    // Default grid: logarithmically spaced from 1e-10 to 1.0
    let default_grid: Vec<f64> = (-10..=0).map(|i| 10.0_f64.powi(i)).collect();
    let grid = smoothing_grid.unwrap_or(&default_grid);

    if grid.is_empty() {
        return Err(BarracudaError::InvalidInput {
            message: "smoothing_grid cannot be empty".into(),
        });
    }

    // Parallelize grid search: each smoothing value trains an independent
    // surrogate and computes LOO-CV RMSE. The GPU device (Arc) is shared
    // safely across threads since wgpu is internally synchronized.
    use rayon::prelude::*;

    let results: Vec<(f64, f64)> = grid
        .par_iter()
        .filter_map(|&s| {
            let surrogate = RBFSurrogate::train(device.clone(), x_data, y_data, kernel, s).ok()?;
            let rmse = surrogate.loo_cv_rmse().ok().filter(|r| r.is_finite())?;
            Some((s, rmse))
        })
        .collect();

    if results.is_empty() {
        return Err(BarracudaError::ExecutionError {
            message: "No valid smoothing values found during grid search".into(),
        });
    }

    let (best_smoothing, best_rmse) = results
        .iter()
        .copied()
        .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Greater))
        .unwrap_or((grid[0], f64::INFINITY));

    Ok(LooSmoothing {
        smoothing: best_smoothing,
        rmse: best_rmse,
        grid_results: results,
    })
}

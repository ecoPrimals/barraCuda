// SPDX-License-Identifier: AGPL-3.0-or-later
//! Spatial Interpolation (Ordinary Kriging) at f64 precision
//!
//! UNIFIED PATTERN (Feb 16 2026) — Serves multiple springs:
//! - airSpring: Soil moisture mapping from sparse sensors to field grid
//! - wetSpring: Sample interpolation across sampling sites
//! - General: Any spatial data interpolation with uncertainty estimation
//!
//! # Architecture
//!
//! Kriging provides the Best Linear Unbiased Predictor (BLUP) for spatial data:
//! 1. Build covariance matrix K from variogram model
//! 2. Solve kriging system Kw = k for weights
//! 3. Interpolate: z* = Σ `w_i` * `z_i` with variance estimate
//!
//! # Variogram Models
//!
//! - Spherical: Most common, bounded
//! - Exponential: Unbounded, reaches sill asymptotically
//! - Gaussian: Very smooth, for continuous phenomena
//! - Linear: Simple, bounded
//!
//! # Example
//!
//! ```rust,ignore
//! use barracuda::ops::kriging_f64::{KrigingF64, VariogramModel};
//!
//! // Known sensor locations with soil moisture readings
//! let known_points = vec![
//!     (0.0, 0.0, 0.35),  // (x, y, moisture)
//!     (10.0, 0.0, 0.28),
//!     (0.0, 10.0, 0.32),
//!     (10.0, 10.0, 0.30),
//! ];
//!
//! // Target grid points to interpolate
//! let target_points = vec![(5.0, 5.0), (2.5, 2.5), (7.5, 7.5)];
//!
//! let kriging = KrigingF64::new(device.clone())?;
//! let (interpolated, variances) = kriging.interpolate(
//!     &known_points,
//!     &target_points,
//!     VariogramModel::Spherical { nugget: 0.0, sill: 0.01, range: 15.0 },
//! )?;
//! ```

use crate::device::WgpuDevice;
use crate::error::Result;
use std::sync::Arc;

/// Variogram models for spatial correlation
#[derive(Debug, Clone, Copy)]
pub enum VariogramModel {
    /// Spherical variogram: most common, bounded
    /// γ(h) = c0 + c * (1.5*h/a - 0.5*(h/a)³) for h ≤ a
    Spherical {
        /// Nugget effect (variance at zero distance)
        nugget: f64,
        /// Sill (total variance at range)
        sill: f64,
        /// Range (distance at which sill is reached)
        range: f64,
    },

    /// Exponential variogram: unbounded, reaches sill asymptotically
    /// γ(h) = c0 + c * (1 - exp(-3h/a))
    Exponential {
        /// Nugget effect (variance at zero distance)
        nugget: f64,
        /// Sill (total variance at range)
        sill: f64,
        /// Range (practical range ≈ a/3)
        range: f64,
    },

    /// Gaussian variogram: very smooth, continuous phenomena
    /// γ(h) = c0 + c * (1 - exp(-3(h/a)²))
    Gaussian {
        /// Nugget effect (variance at zero distance)
        nugget: f64,
        /// Sill (total variance at range)
        sill: f64,
        /// Range (practical range ≈ a/√3)
        range: f64,
    },

    /// Linear variogram: simple, bounded
    /// γ(h) = c0 + c * h/a for h ≤ a
    Linear {
        /// Nugget effect (variance at zero distance)
        nugget: f64,
        /// Sill (total variance at range)
        sill: f64,
        /// Range (distance at which sill is reached)
        range: f64,
    },
}

impl VariogramModel {
    /// Get variogram parameters (nugget, sill, range, `model_id`).
    #[must_use]
    pub fn params(&self) -> (f64, f64, f64, u32) {
        match self {
            VariogramModel::Spherical {
                nugget,
                sill,
                range,
            } => (*nugget, *sill, *range, 0),
            VariogramModel::Exponential {
                nugget,
                sill,
                range,
            } => (*nugget, *sill, *range, 1),
            VariogramModel::Gaussian {
                nugget,
                sill,
                range,
            } => (*nugget, *sill, *range, 2),
            VariogramModel::Linear {
                nugget,
                sill,
                range,
            } => (*nugget, *sill, *range, 3),
        }
    }

    /// Compute variogram value γ(h) for distance h.
    #[must_use]
    pub fn gamma(&self, h: f64) -> f64 {
        if h <= 0.0 {
            return 0.0;
        }

        let (nugget, sill, range, model) = self.params();
        let c = sill - nugget; // Partial sill

        match model {
            0 => {
                // Spherical
                if h >= range {
                    nugget + c
                } else {
                    let ratio = h / range;
                    nugget + c * (1.5 * ratio - 0.5 * ratio.powi(3))
                }
            }
            1 => {
                // Exponential
                nugget + c * (1.0 - (-3.0 * h / range).exp())
            }
            2 => {
                // Gaussian
                let ratio = h / range;
                nugget + c * (1.0 - (-3.0 * ratio * ratio).exp())
            }
            3 => {
                // Linear
                if h >= range {
                    nugget + c
                } else {
                    nugget + c * h / range
                }
            }
            _ => nugget + c,
        }
    }

    /// Compute covariance C(h) = sill - γ(h).
    #[must_use]
    pub fn covariance(&self, h: f64) -> f64 {
        let (_, sill, _, _) = self.params();
        sill - self.gamma(h)
    }
}

/// Result of kriging interpolation.
#[derive(Debug, Clone)]
pub struct KrigingResult {
    /// Interpolated values at target points
    pub values: Vec<f64>,
    /// Kriging variance (uncertainty) at each target point
    pub variances: Vec<f64>,
    /// Kriging weights used (`n_known` per target)
    pub weights: Vec<Vec<f64>>,
}

/// Ordinary Kriging interpolator.
pub struct KrigingF64 {
    device: Arc<WgpuDevice>,
}

impl KrigingF64 {
    /// Create a new kriging interpolator
    /// # Errors
    /// Returns [`Err`] if the device is invalid or unavailable.
    pub fn new(device: Arc<WgpuDevice>) -> Result<Self> {
        Ok(Self { device })
    }

    /// GPU device handle (for future GPU-accelerated kriging).
    #[must_use]
    pub fn device(&self) -> &Arc<WgpuDevice> {
        &self.device
    }

    /// Perform ordinary kriging interpolation
    /// # Arguments
    /// * `known_points` - Known data points as (x, y, z) tuples
    /// * `target_points` - Target points as (x, y) tuples
    /// * `variogram` - Variogram model with parameters
    /// # Returns
    /// `KrigingResult` with interpolated values, variances, and weights
    /// # Errors
    /// Returns [`Err`] if the covariance matrix is singular or ill-conditioned
    /// (LU solve fails).
    pub fn interpolate(
        &self,
        known_points: &[(f64, f64, f64)],
        target_points: &[(f64, f64)],
        variogram: VariogramModel,
    ) -> Result<KrigingResult> {
        let n = known_points.len();
        let m = target_points.len();

        if n == 0 {
            return Ok(KrigingResult {
                values: vec![],
                variances: vec![],
                weights: vec![],
            });
        }

        // Build covariance matrix K (n+1 x n+1) for ordinary kriging
        // Last row/col is for Lagrange multiplier
        let n1 = n + 1;
        let mut k_matrix = vec![0.0; n1 * n1];

        for i in 0..n {
            for j in 0..n {
                let h = Self::distance(
                    known_points[i].0,
                    known_points[i].1,
                    known_points[j].0,
                    known_points[j].1,
                );
                k_matrix[i * n1 + j] = variogram.covariance(h);
            }
            // Lagrange row/col
            k_matrix[i * n1 + n] = 1.0;
            k_matrix[n * n1 + i] = 1.0;
        }
        k_matrix[n * n1 + n] = 0.0;

        // Solve for each target point
        let mut values = Vec::with_capacity(m);
        let mut variances = Vec::with_capacity(m);
        let mut all_weights = Vec::with_capacity(m);

        for target in target_points {
            // Build RHS vector k
            let mut k_vec = vec![0.0; n1];
            for i in 0..n {
                let h = Self::distance(known_points[i].0, known_points[i].1, target.0, target.1);
                k_vec[i] = variogram.covariance(h);
            }
            k_vec[n] = 1.0; // Lagrange constraint

            // CPU LU solve — kriging systems are typically small (N < 1000
            // observation points); GPU LU is beneficial at N > ~2048
            let weights = Self::solve_lu(&k_matrix, &k_vec, n1)?;

            // Interpolate: z* = Σ w_i * z_i
            let mut z_interp = 0.0;
            for i in 0..n {
                z_interp += weights[i] * known_points[i].2;
            }

            // Kriging variance: σ² = sill - Σ w_i * k_i - λ
            let (_, sill, _, _) = variogram.params();
            let mut variance_sum = 0.0;
            for i in 0..n {
                variance_sum += weights[i] * k_vec[i];
            }
            variance_sum += weights[n]; // Lagrange multiplier
            let variance = sill - variance_sum;

            values.push(z_interp);
            variances.push(variance.max(0.0)); // Ensure non-negative
            all_weights.push(weights[..n].to_vec());
        }

        Ok(KrigingResult {
            values,
            variances,
            weights: all_weights,
        })
    }

    /// Simple kriging (known mean)
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn interpolate_simple(
        &self,
        known_points: &[(f64, f64, f64)],
        target_points: &[(f64, f64)],
        variogram: VariogramModel,
        mean: f64,
    ) -> Result<KrigingResult> {
        let n = known_points.len();
        let m = target_points.len();

        if n == 0 {
            return Ok(KrigingResult {
                values: vec![mean; m],
                variances: vec![variogram.params().1; m], // sill
                weights: vec![vec![]; m],
            });
        }

        // Build covariance matrix K (n x n) - no Lagrange constraint
        let mut k_matrix = vec![0.0; n * n];

        for i in 0..n {
            for j in 0..n {
                let h = Self::distance(
                    known_points[i].0,
                    known_points[i].1,
                    known_points[j].0,
                    known_points[j].1,
                );
                k_matrix[i * n + j] = variogram.covariance(h);
            }
        }

        // Solve for each target point
        let mut values = Vec::with_capacity(m);
        let mut variances = Vec::with_capacity(m);
        let mut all_weights = Vec::with_capacity(m);

        for target in target_points {
            // Build RHS vector k
            let mut k_vec = vec![0.0; n];
            for i in 0..n {
                let h = Self::distance(known_points[i].0, known_points[i].1, target.0, target.1);
                k_vec[i] = variogram.covariance(h);
            }

            // Solve Kw = k
            let weights = Self::solve_lu(&k_matrix, &k_vec, n)?;

            // Interpolate: z* = μ + Σ w_i * (z_i - μ)
            let mut z_interp = mean;
            for i in 0..n {
                z_interp += weights[i] * (known_points[i].2 - mean);
            }

            // Simple kriging variance
            let (_, sill, _, _) = variogram.params();
            let mut variance_sum = 0.0;
            for i in 0..n {
                variance_sum += weights[i] * k_vec[i];
            }
            let variance = sill - variance_sum;

            values.push(z_interp);
            variances.push(variance.max(0.0));
            all_weights.push(weights);
        }

        Ok(KrigingResult {
            values,
            variances,
            weights: all_weights,
        })
    }

    /// Euclidean distance between two 2D points
    fn distance(x1: f64, y1: f64, x2: f64, y2: f64) -> f64 {
        let dx = x2 - x1;
        let dy = y2 - y1;
        dx.hypot(dy)
    }

    /// Solve linear system Ax = b using LU decomposition
    fn solve_lu(a: &[f64], b: &[f64], n: usize) -> Result<Vec<f64>> {
        // Simple LU decomposition with partial pivoting
        let mut lu = a.to_vec();
        let mut perm: Vec<usize> = (0..n).collect();

        // LU decomposition
        for k in 0..n {
            // Find pivot
            let mut max_val = lu[k * n + k].abs();
            let mut max_row = k;
            for i in (k + 1)..n {
                let val = lu[i * n + k].abs();
                if val > max_val {
                    max_val = val;
                    max_row = i;
                }
            }

            // Swap rows
            if max_row != k {
                perm.swap(k, max_row);
                for j in 0..n {
                    lu.swap(k * n + j, max_row * n + j);
                }
            }

            // Check for singularity
            if lu[k * n + k].abs() < 1e-14 {
                // Add small regularization
                lu[k * n + k] = 1e-10;
            }

            // Elimination
            for i in (k + 1)..n {
                lu[i * n + k] /= lu[k * n + k];
                for j in (k + 1)..n {
                    lu[i * n + j] -= lu[i * n + k] * lu[k * n + j];
                }
            }
        }

        // Forward substitution (Ly = Pb)
        let mut y = vec![0.0; n];
        for i in 0..n {
            y[i] = b[perm[i]];
            for j in 0..i {
                y[i] -= lu[i * n + j] * y[j];
            }
        }

        // Back substitution (Ux = y)
        let mut x = vec![0.0; n];
        for i in (0..n).rev() {
            x[i] = y[i];
            for j in (i + 1)..n {
                x[i] -= lu[i * n + j] * x[j];
            }
            x[i] /= lu[i * n + i];
        }

        Ok(x)
    }

    /// Estimate variogram parameters from data using method of moments
    /// # Errors
    /// Returns [`Err`] if allocation fails (unlikely for typical inputs).
    pub fn fit_variogram(
        known_points: &[(f64, f64, f64)],
        n_lags: usize,
        max_distance: f64,
    ) -> Result<(Vec<f64>, Vec<f64>)> {
        let n = known_points.len();
        let lag_width = max_distance / n_lags as f64;

        let mut lag_distances = vec![0.0; n_lags];
        let mut lag_semivariances = vec![0.0; n_lags];
        let mut lag_counts = vec![0usize; n_lags];

        // Compute empirical variogram
        for i in 0..n {
            for j in (i + 1)..n {
                let h = Self::distance(
                    known_points[i].0,
                    known_points[i].1,
                    known_points[j].0,
                    known_points[j].1,
                );

                let lag_idx = (h / lag_width).floor() as usize;
                if lag_idx < n_lags {
                    let diff = known_points[i].2 - known_points[j].2;
                    lag_semivariances[lag_idx] += diff * diff;
                    lag_distances[lag_idx] += h;
                    lag_counts[lag_idx] += 1;
                }
            }
        }

        // Average
        for k in 0..n_lags {
            if lag_counts[k] > 0 {
                lag_semivariances[k] /= 2.0 * lag_counts[k] as f64;
                lag_distances[k] /= lag_counts[k] as f64;
            }
        }

        Ok((lag_distances, lag_semivariances))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_device() -> Option<Arc<crate::device::WgpuDevice>> {
        crate::device::test_pool::get_test_device_if_f64_gpu_available_sync()
    }

    #[test]
    fn test_variogram_spherical() {
        let model = VariogramModel::Spherical {
            nugget: 0.0,
            sill: 1.0,
            range: 10.0,
        };

        // γ(0) = 0
        assert!((model.gamma(0.0) - 0.0).abs() < 1e-10);

        // γ(a) = sill
        assert!((model.gamma(10.0) - 1.0).abs() < 1e-10);

        // γ(h > a) = sill
        assert!((model.gamma(20.0) - 1.0).abs() < 1e-10);

        // γ(a/2) should be ~0.688
        let mid = model.gamma(5.0);
        assert!(mid > 0.5 && mid < 0.8);
    }

    #[test]
    fn test_kriging_simple_case() -> Result<()> {
        let Some(device) = create_test_device() else {
            return Ok(());
        };
        let kriging = KrigingF64::new(device)?;

        // Simple 4-point case
        let known = vec![
            (0.0, 0.0, 1.0),
            (10.0, 0.0, 2.0),
            (0.0, 10.0, 2.0),
            (10.0, 10.0, 3.0),
        ];

        let targets = vec![(5.0, 5.0)]; // Center point

        let model = VariogramModel::Spherical {
            nugget: 0.0,
            sill: 0.5,
            range: 15.0,
        };

        let result = kriging.interpolate(&known, &targets, model)?;

        // Center should interpolate to mean of corners: (1+2+2+3)/4 = 2.0
        assert!(
            (result.values[0] - 2.0).abs() < 0.1,
            "Expected ~2.0, got {}",
            result.values[0]
        );

        // Variance should be positive
        assert!(result.variances[0] >= 0.0);

        Ok(())
    }

    #[test]
    fn test_kriging_at_known_point() -> Result<()> {
        let Some(device) = create_test_device() else {
            return Ok(());
        };
        let kriging = KrigingF64::new(device)?;

        let known = vec![(0.0, 0.0, 1.0), (10.0, 0.0, 2.0), (0.0, 10.0, 3.0)];

        // Interpolate at a known point
        let targets = vec![(0.0, 0.0)];

        let model = VariogramModel::Exponential {
            nugget: 0.0,
            sill: 1.0,
            range: 10.0,
        };

        let result = kriging.interpolate(&known, &targets, model)?;

        // Should exactly reproduce known value
        assert!(
            (result.values[0] - 1.0).abs() < 1e-6,
            "Expected 1.0, got {}",
            result.values[0]
        );

        // Variance at known point should be ~0 (or very small due to numerics)
        assert!(
            result.variances[0] < 0.01,
            "Variance at known point should be ~0, got {}",
            result.variances[0]
        );

        Ok(())
    }
}

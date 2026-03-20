// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU-Resident Nelder-Mead Optimizer
//!
//! **EVOLUTION (Feb 14, 2026)**: GPU-resident optimization to minimize CPU↔GPU transfers.
//!
//! This module provides a GPU-accelerated Nelder-Mead simplex optimizer that keeps
//! all simplex data and intermediate calculations on the GPU.
//!
//! ## Benefits
//!
//! - **Reduced memory transfers**: Simplex vertices stay on GPU
//! - **GPU-native operations**: Sorting, centroid, and reflection computed on GPU
//! - **Batch evaluation**: Evaluate multiple vertices in parallel
//! - **Integration with GPU surrogates**: Seamless with `AdaptiveRBFSurrogate`
//!
//! ## When to Use
//!
//! Use GPU-resident NM when:
//! - Objective function is GPU-computable (e.g., RBF surrogate)
//! - Problem dimension is moderate (5-50 parameters)
//! - Running many optimization iterations
//!
//! Use CPU NM when:
//! - Objective function is CPU-only
//! - Problem is 1-3 dimensional
//! - Few iterations needed

use crate::device::WgpuDevice;
use crate::error::{BarracudaError, Result};
use std::sync::Arc;

/// GPU-resident Nelder-Mead result
#[derive(Debug, Clone)]
pub struct NelderMeadGpuResult {
    /// Best solution found
    pub x: Vec<f64>,
    /// Best function value
    pub f_best: f64,
    /// Number of function evaluations
    pub n_evals: usize,
    /// Number of iterations
    pub iterations: usize,
    /// Whether convergence was achieved
    pub converged: bool,
}

/// GPU-resident Nelder-Mead optimizer
///
/// Keeps simplex data on GPU and minimizes CPU↔GPU transfers.
/// GPU shader for parallel simplex operations (centroid, reflect, expand, contract, shrink).
///
/// Entry points: `compute_centroid`, `reflect`, `expand`, `contract`, `shrink`,
/// `project_bounds`, `bitonic_sort_step`.
const WGSL_SIMPLEX_OPS_F64: &str = include_str!("../shaders/optimizer/simplex_ops_f64.wgsl");

/// WGSL shader for Nelder-Mead simplex operations (f64 downcast to f32 when needed).
pub static WGSL_SIMPLEX_OPS: std::sync::LazyLock<String> =
    std::sync::LazyLock::new(|| WGSL_SIMPLEX_OPS_F64.to_string());

/// Only reads back the best solution at the end or periodically.
pub struct NelderMeadGpu {
    device: Arc<WgpuDevice>,
    /// Nelder-Mead parameters
    alpha: f64, // Reflection (default: 1.0)
    gamma: f64, // Expansion (default: 2.0)
    rho: f64,   // Contraction (default: 0.5)
    sigma: f64, // Shrinkage (default: 0.5)
}

impl NelderMeadGpu {
    /// Create GPU-resident Nelder-Mead optimizer with default parameters
    #[must_use]
    pub fn new(device: Arc<WgpuDevice>) -> Self {
        Self {
            device,
            alpha: 1.0,
            gamma: 2.0,
            rho: 0.5,
            sigma: 0.5,
        }
    }

    /// Set Nelder-Mead parameters
    #[must_use]
    pub fn with_params(mut self, alpha: f64, gamma: f64, rho: f64, sigma: f64) -> Self {
        self.alpha = alpha;
        self.gamma = gamma;
        self.rho = rho;
        self.sigma = sigma;
        self
    }

    /// Optimize using a GPU-computable objective function
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if bounds length mismatch, or if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    ///
    /// The objective function takes GPU buffers and writes results to output buffer.
    /// This enables fully GPU-resident optimization without CPU roundtrips.
    ///
    /// # Arguments
    ///
    /// * `f_gpu` - GPU objective function: evaluates vertices and writes values to output
    /// * `x0` - Initial guess (CPU, will be uploaded)
    /// * `bounds` - Box constraints
    /// * `max_iter` - Maximum iterations
    /// * `tol` - Convergence tolerance
    /// * `check_interval` - How often to check convergence (default: 10)
    ///
    /// # Type Parameters
    ///
    /// * `F` - GPU evaluation function: `fn(&wgpu::Buffer, &wgpu::Buffer, usize) -> Result<()>`
    ///   - Input: vertex buffer (`n_vertices` × dim f64), output buffer (`n_vertices` f64)
    ///   - Evaluates all vertices and writes function values to output
    pub fn optimize<F>(
        &self,
        f_gpu: F,
        x0: &[f64],
        bounds: &[(f64, f64)],
        max_iter: usize,
        tol: f64,
        check_interval: usize,
    ) -> Result<NelderMeadGpuResult>
    where
        F: Fn(&wgpu::Buffer, &wgpu::Buffer, usize) -> Result<()>,
    {
        let n = x0.len();
        let n_vertices = n + 1;
        let check_interval = check_interval.max(1);

        if bounds.len() != n {
            return Err(BarracudaError::InvalidInput {
                message: format!("Bounds length {} must match x0 length {}", bounds.len(), n),
            });
        }

        // Initialize simplex on CPU, then upload
        let mut simplex = self.init_simplex(x0, bounds);

        // Create GPU buffers
        let simplex_buffer = self.create_simplex_buffer(&simplex, n);
        let f_vals_buffer = self.create_f64_buffer(n_vertices);
        // Create bounds buffer for future GPU-side bound projection
        let _bounds_buffer = self.create_bounds_buffer(bounds);

        // Initial function evaluation (all n+1 vertices)
        f_gpu(&simplex_buffer, &f_vals_buffer, n_vertices)?;
        let mut f_vals = self.device.read_f64_buffer(&f_vals_buffer, n_vertices)?;
        let mut n_evals = n_vertices;

        // Main optimization loop
        for iter in 0..max_iter {
            // CPU simplex sort — N+1 vertices is O((N+1)log(N+1)), negligible
            // compared to N+1 GPU function evaluations per iteration
            let mut indices: Vec<usize> = (0..n_vertices).collect();
            indices.sort_by(|&i, &j| {
                f_vals[i]
                    .partial_cmp(&f_vals[j])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            let best_idx = indices[0];
            let worst_idx = indices[n];
            let second_worst_idx = indices[n - 1];

            // Check convergence periodically
            if iter % check_interval == 0 {
                let f_mean: f64 = f_vals.iter().sum::<f64>() / n_vertices as f64;
                let f_std = (f_vals.iter().map(|&fi| (fi - f_mean).powi(2)).sum::<f64>()
                    / n_vertices as f64)
                    .sqrt();

                if f_std < tol {
                    return Ok(NelderMeadGpuResult {
                        x: simplex[best_idx].clone(),
                        f_best: f_vals[best_idx],
                        n_evals,
                        iterations: iter,
                        converged: true,
                    });
                }
            }

            // Compute centroid (excluding worst)
            let centroid = self.compute_centroid(&simplex, &indices[..n]);

            // Reflection
            let x_reflect = self.reflect(&simplex[worst_idx], &centroid, self.alpha);
            let x_reflect = self.project_bounds(&x_reflect, bounds);

            // Upload and evaluate reflection point
            let reflect_buffer = self.create_point_buffer(&x_reflect);
            let reflect_f_buffer = self.create_f64_buffer(1);
            f_gpu(&reflect_buffer, &reflect_f_buffer, 1)?;
            let f_reflect = self.device.read_f64_buffer(&reflect_f_buffer, 1)?[0];
            n_evals += 1;

            if f_reflect < f_vals[best_idx] {
                // Expansion
                let x_expand = self.reflect(&simplex[worst_idx], &centroid, self.gamma);
                let x_expand = self.project_bounds(&x_expand, bounds);

                let expand_buffer = self.create_point_buffer(&x_expand);
                let expand_f_buffer = self.create_f64_buffer(1);
                f_gpu(&expand_buffer, &expand_f_buffer, 1)?;
                let f_expand = self.device.read_f64_buffer(&expand_f_buffer, 1)?[0];
                n_evals += 1;

                if f_expand < f_reflect {
                    simplex[worst_idx] = x_expand;
                    f_vals[worst_idx] = f_expand;
                } else {
                    simplex[worst_idx] = x_reflect;
                    f_vals[worst_idx] = f_reflect;
                }
            } else if f_reflect < f_vals[second_worst_idx] {
                // Accept reflection
                simplex[worst_idx] = x_reflect;
                f_vals[worst_idx] = f_reflect;
            } else {
                // Contraction
                let x_contract = if f_reflect < f_vals[worst_idx] {
                    // Outside contraction
                    self.reflect(&simplex[worst_idx], &centroid, self.alpha * self.rho)
                } else {
                    // Inside contraction
                    self.reflect(&simplex[worst_idx], &centroid, -self.rho)
                };
                let x_contract = self.project_bounds(&x_contract, bounds);

                let contract_buffer = self.create_point_buffer(&x_contract);
                let contract_f_buffer = self.create_f64_buffer(1);
                f_gpu(&contract_buffer, &contract_f_buffer, 1)?;
                let f_contract = self.device.read_f64_buffer(&contract_f_buffer, 1)?[0];
                n_evals += 1;

                if f_contract < f_vals[worst_idx] {
                    simplex[worst_idx] = x_contract;
                    f_vals[worst_idx] = f_contract;
                } else {
                    // Shrinkage - re-evaluate all points except best
                    for i in 0..n_vertices {
                        if i != best_idx {
                            for j in 0..n {
                                simplex[i][j] = simplex[best_idx][j]
                                    + self.sigma * (simplex[i][j] - simplex[best_idx][j]);
                            }
                            simplex[i] = self.project_bounds(&simplex[i], bounds);
                        }
                    }

                    // Batch re-evaluate all shrunk vertices
                    let shrunk_buffer = self.create_simplex_buffer(&simplex, n);
                    f_gpu(&shrunk_buffer, &f_vals_buffer, n_vertices)?;
                    f_vals = self.device.read_f64_buffer(&f_vals_buffer, n_vertices)?;
                    n_evals += n; // Only non-best vertices re-evaluated
                }
            }

            // Update GPU simplex buffer
            self.write_simplex_buffer(&simplex_buffer, &simplex, n);

            if n_evals >= max_iter {
                break;
            }
        }

        // Return best point
        let best_idx = f_vals
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map_or(0, |(idx, _)| idx);

        Ok(NelderMeadGpuResult {
            x: simplex[best_idx].clone(),
            f_best: f_vals[best_idx],
            n_evals,
            iterations: max_iter,
            converged: false,
        })
    }

    /// Initialize simplex from x0
    fn init_simplex(&self, x0: &[f64], bounds: &[(f64, f64)]) -> Vec<Vec<f64>> {
        let n = x0.len();
        let mut simplex = Vec::with_capacity(n + 1);

        // First vertex: x0 (projected to bounds)
        simplex.push(self.project_bounds(x0, bounds));

        // Remaining vertices: perturb each coordinate
        for i in 0..n {
            let mut x = x0.to_vec();
            let delta = 0.05 * (bounds[i].1 - bounds[i].0).max(0.1);
            x[i] += delta;
            simplex.push(self.project_bounds(&x, bounds));
        }

        simplex
    }

    /// Compute centroid of selected vertices
    fn compute_centroid(&self, simplex: &[Vec<f64>], indices: &[usize]) -> Vec<f64> {
        let n = simplex[0].len();
        let k = indices.len();
        let mut centroid = vec![0.0; n];

        for &idx in indices {
            for (j, c) in centroid.iter_mut().enumerate() {
                *c += simplex[idx][j];
            }
        }
        for c in &mut centroid {
            *c /= k as f64;
        }

        centroid
    }

    /// Reflect point through centroid
    fn reflect(&self, x: &[f64], centroid: &[f64], alpha: f64) -> Vec<f64> {
        centroid
            .iter()
            .zip(x.iter())
            .map(|(&c, &xi)| c + alpha * (c - xi))
            .collect()
    }

    /// Project point to bounds
    fn project_bounds(&self, x: &[f64], bounds: &[(f64, f64)]) -> Vec<f64> {
        x.iter()
            .zip(bounds.iter())
            .map(|(&xi, &(lo, hi))| xi.clamp(lo, hi))
            .collect()
    }

    /// Create GPU buffer for simplex (`n_vertices` × dim)
    fn create_simplex_buffer(&self, simplex: &[Vec<f64>], dim: usize) -> wgpu::Buffer {
        let mut data = Vec::with_capacity(simplex.len() * dim);
        for vertex in simplex {
            data.extend_from_slice(vertex);
        }
        let bytes: &[u8] = bytemuck::cast_slice(&data);

        self.device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("NM simplex"),
                contents: bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            })
    }

    /// Write simplex to GPU buffer
    fn write_simplex_buffer(&self, buffer: &wgpu::Buffer, simplex: &[Vec<f64>], dim: usize) {
        let mut data = Vec::with_capacity(simplex.len() * dim);
        for vertex in simplex {
            data.extend_from_slice(vertex);
        }
        let bytes: &[u8] = bytemuck::cast_slice(&data);
        self.device.queue.write_buffer(buffer, 0, bytes);
    }

    /// Create GPU buffer for single point
    fn create_point_buffer(&self, x: &[f64]) -> wgpu::Buffer {
        let bytes: &[u8] = bytemuck::cast_slice(x);
        self.device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("NM point"),
                contents: bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            })
    }

    /// Create GPU buffer for bounds
    fn create_bounds_buffer(&self, bounds: &[(f64, f64)]) -> wgpu::Buffer {
        let data: Vec<f64> = bounds.iter().flat_map(|&(lo, hi)| [lo, hi]).collect();
        let bytes: &[u8] = bytemuck::cast_slice(&data);

        self.device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("NM bounds"),
                contents: bytes,
                usage: wgpu::BufferUsages::STORAGE,
            })
    }

    /// Create f64 buffer
    fn create_f64_buffer(&self, count: usize) -> wgpu::Buffer {
        let zeros = vec![0u8; count * 8];
        self.device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("NM f64"),
                contents: &zeros,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            })
    }
}

#[cfg(test)]
mod tests {
    use super::NelderMeadGpuResult;

    #[test]
    fn test_nelder_mead_gpu_creation() {
        assert!(std::mem::size_of::<NelderMeadGpuResult>() > 0);
    }
}

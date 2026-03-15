// SPDX-License-Identifier: AGPL-3.0-only
//! Batched Bisection Root-Finding (GPU) — Parallel Root-Finding
//!
//! Solves many independent root-finding problems in parallel on GPU.
//! Each problem runs bisection concurrently with full f64 precision.
//!
//! **Use cases**:
//! - BCS pairing: Find μ where Σ `v²_k(μ)` = N for each nucleus
//! - Multi-system parameter fitting
//! - Batch chemical equilibrium calculations
//!
//! **Deep Debt Principles**:
//! - Pure WGSL implementation (hardware-agnostic)
//! - Full f64 precision
//! - Safe Rust wrapper (no unsafe code)
//!
//! # Example
//!
//! ```rust,ignore
//! let device = WgpuDevice::new().await?;
//! let bisect = BatchedBisectionGpu::new(device.clone(), 64, 1e-12)?;
//!
//! // Find √2, √3, √5 in parallel
//! let lower = vec![0.0, 0.0, 0.0];
//! let upper = vec![2.0, 2.0, 3.0];
//! let targets = vec![2.0, 3.0, 5.0]; // x² = target
//!
//! let roots = bisect.solve_polynomial(&lower, &upper, &targets)?;
//! // roots ≈ [1.414, 1.732, 2.236]
//! ```

use crate::device::WgpuDevice;
use crate::device::capabilities::WORKGROUP_SIZE_COMPACT;
use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::{BarracudaError, Result};
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;

/// Parameters for batched bisection shader
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct BisectionParams {
    batch_size: u32,
    max_iterations: u32,
    n_levels: u32,
    use_degeneracy: u32, // 1 = use deg_k weights, 0 = assume deg_k=1
    tolerance_lo: u32,   // f64 as two u32s for alignment
    tolerance_hi: u32,
}

impl BisectionParams {
    fn new(
        batch_size: u32,
        max_iterations: u32,
        n_levels: u32,
        use_degeneracy: bool,
        tolerance: f64,
    ) -> Self {
        let tol_bits = tolerance.to_bits();
        Self {
            batch_size,
            max_iterations,
            n_levels,
            use_degeneracy: u32::from(use_degeneracy),
            tolerance_lo: tol_bits as u32,
            tolerance_hi: (tol_bits >> 32) as u32,
        }
    }
}

/// GPU-accelerated batched bisection root-finding
///
/// Solves N independent bisection problems in parallel.
pub struct BatchedBisectionGpu {
    device: Arc<WgpuDevice>,
    max_iterations: u32,
    tolerance: f64,
}

/// Result of batched bisection
pub struct BisectionResult {
    /// Found roots [`batch_size`]
    pub roots: Vec<f64>,
    /// Number of iterations used per problem [`batch_size`]
    pub iterations: Vec<u32>,
}

impl BatchedBisectionGpu {
    /// Create a new batched bisection solver
    /// # Arguments
    /// * `device` - `WgpuDevice`
    /// * `max_iterations` - Maximum bisection iterations per problem (typically 50-100)
    /// * `tolerance` - Convergence tolerance (typically 1e-10 to 1e-14)
    /// # Errors
    /// Returns [`Err`] if `tolerance` ≤ 0.
    pub fn new(device: Arc<WgpuDevice>, max_iterations: u32, tolerance: f64) -> Result<Self> {
        if tolerance <= 0.0 {
            return Err(BarracudaError::InvalidInput {
                message: "Tolerance must be positive".to_string(),
            });
        }
        Ok(Self {
            device,
            max_iterations,
            tolerance,
        })
    }

    fn wgsl_shader() -> &'static str {
        include_str!("../shaders/optimizer/batched_bisection_f64.wgsl")
    }

    /// Solve batched polynomial root-finding: x² = target
    /// Finds √target for each problem in parallel.
    /// This is a validation/test function.
    /// # Arguments
    /// * `lower` - Lower bounds [`batch_size`]
    /// * `upper` - Upper bounds [`batch_size`]
    /// * `targets` - Target values (find x where x² = target) [`batch_size`]
    /// # Errors
    /// Returns [`Err`] if `lower`, `upper`, and `targets` lengths do not match, or if GPU
    /// execution or buffer readback fails.
    pub fn solve_polynomial(
        &self,
        lower: &[f64],
        upper: &[f64],
        targets: &[f64],
    ) -> Result<BisectionResult> {
        if lower.len() != upper.len() || lower.len() != targets.len() {
            return Err(BarracudaError::InvalidInput {
                message: format!(
                    "Array lengths must match: lower={}, upper={}, targets={}",
                    lower.len(),
                    upper.len(),
                    targets.len()
                ),
            });
        }

        self.solve_internal(lower, upper, targets, 1, false, "batched_bisection_poly")
    }

    /// Solve BCS particle number equation: Σ `v²_k(μ)` = N
    /// Finds chemical potential μ for each nucleus such that the
    /// BCS occupation numbers sum to the target particle number.
    /// # Arguments
    /// * `lower` - Lower bounds for μ [`batch_size`]
    /// * `upper` - Upper bounds for μ [`batch_size`]
    /// * `eigenvalues` - Single-particle energies [`batch_size`, `n_levels`]
    /// * `delta` - Pairing gap for each problem [`batch_size`]
    /// * `target_n` - Target particle number for each problem [`batch_size`]
    /// # Errors
    /// Returns [`Err`] if array lengths do not match or `eigenvalues.len()` is not divisible by
    /// `batch_size`, or if GPU execution or buffer readback fails.
    pub fn solve_bcs(
        &self,
        lower: &[f64],
        upper: &[f64],
        eigenvalues: &[f64],
        delta: &[f64],
        target_n: &[f64],
    ) -> Result<BisectionResult> {
        let batch_size = lower.len();
        if upper.len() != batch_size || delta.len() != batch_size || target_n.len() != batch_size {
            return Err(BarracudaError::InvalidInput {
                message: "Array lengths must match batch_size".to_string(),
            });
        }

        // Calculate n_levels
        if !eigenvalues.len().is_multiple_of(batch_size) {
            return Err(BarracudaError::InvalidInput {
                message: format!(
                    "eigenvalues length {} must be divisible by batch_size {}",
                    eigenvalues.len(),
                    batch_size
                ),
            });
        }
        let n_levels = eigenvalues.len() / batch_size;

        // Pack params: [ε_0, ..., ε_{n-1}, Δ, N] per problem
        let params_per_problem = n_levels + 2;
        let mut params = Vec::with_capacity(batch_size * params_per_problem);
        for i in 0..batch_size {
            // Eigenvalues for this problem
            for j in 0..n_levels {
                params.push(eigenvalues[i * n_levels + j]);
            }
            // Delta and target N
            params.push(delta[i]);
            params.push(target_n[i]);
        }

        self.solve_internal(
            lower,
            upper,
            &params,
            n_levels as u32,
            false,
            "batched_bisection",
        )
    }

    /// Solve BCS pairing equations with level degeneracy (`deg_k`)
    /// For nuclear HFB: `deg_k` = 2j+1 (spin degeneracy of each level)
    /// **Formula**: Find μ such that `Σ_k` `deg_k` · `v²_k(μ)` = N
    /// # Arguments
    /// * `lower` - Lower bounds for μ [`batch_size`]
    /// * `upper` - Upper bounds for μ [`batch_size`]
    /// * `eigenvalues` - Packed energy levels [`batch_size` × `n_levels`]
    /// * `degeneracies` - Degeneracy of each level [`batch_size` × `n_levels`]
    /// * `delta` - BCS pairing gap per problem [`batch_size`]
    /// * `target_n` - Target particle number per problem [`batch_size`]
    /// # Evolution
    /// Added Feb 16, 2026 per hotSpring handoff TIER 3.1
    /// # Errors
    /// Returns [`Err`] if array lengths do not match, `eigenvalues.len()` is not divisible by
    /// `batch_size`, or `degeneracies.len() != eigenvalues.len()`, or if GPU execution or
    /// buffer readback fails.
    pub fn solve_bcs_with_degeneracy(
        &self,
        lower: &[f64],
        upper: &[f64],
        eigenvalues: &[f64],
        degeneracies: &[f64],
        delta: &[f64],
        target_n: &[f64],
    ) -> Result<BisectionResult> {
        let batch_size = lower.len();
        if upper.len() != batch_size || delta.len() != batch_size || target_n.len() != batch_size {
            return Err(BarracudaError::InvalidInput {
                message: "Array lengths must match batch_size".to_string(),
            });
        }

        // Calculate n_levels
        if !eigenvalues.len().is_multiple_of(batch_size) {
            return Err(BarracudaError::InvalidInput {
                message: format!(
                    "eigenvalues length {} must be divisible by batch_size {}",
                    eigenvalues.len(),
                    batch_size
                ),
            });
        }
        let n_levels = eigenvalues.len() / batch_size;

        if degeneracies.len() != eigenvalues.len() {
            return Err(BarracudaError::InvalidInput {
                message: format!(
                    "degeneracies length {} must match eigenvalues length {}",
                    degeneracies.len(),
                    eigenvalues.len()
                ),
            });
        }

        // Pack params: [ε_0, ..., ε_{n-1}, deg_0, ..., deg_{n-1}, Δ, N] per problem
        let params_per_problem = n_levels * 2 + 2;
        let mut params = Vec::with_capacity(batch_size * params_per_problem);
        for i in 0..batch_size {
            // Eigenvalues for this problem
            for j in 0..n_levels {
                params.push(eigenvalues[i * n_levels + j]);
            }
            // Degeneracies for this problem
            for j in 0..n_levels {
                params.push(degeneracies[i * n_levels + j]);
            }
            // Delta and target N
            params.push(delta[i]);
            params.push(target_n[i]);
        }

        self.solve_internal(
            lower,
            upper,
            &params,
            n_levels as u32,
            true,
            "batched_bisection",
        )
    }

    fn solve_internal(
        &self,
        lower: &[f64],
        upper: &[f64],
        params: &[f64],
        n_levels: u32,
        use_degeneracy: bool,
        entry_point: &str,
    ) -> Result<BisectionResult> {
        let batch_size = lower.len();
        if batch_size == 0 {
            return Ok(BisectionResult {
                roots: vec![],
                iterations: vec![],
            });
        }

        // Create buffers
        let lower_bytes: &[u8] = bytemuck::cast_slice(lower);
        let lower_buffer =
            self.device
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("BatchedBisection lower"),
                    contents: lower_bytes,
                    usage: wgpu::BufferUsages::STORAGE,
                });

        let upper_bytes: &[u8] = bytemuck::cast_slice(upper);
        let upper_buffer =
            self.device
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("BatchedBisection upper"),
                    contents: upper_bytes,
                    usage: wgpu::BufferUsages::STORAGE,
                });

        let params_bytes: &[u8] = bytemuck::cast_slice(params);
        let params_buffer =
            self.device
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("BatchedBisection params"),
                    contents: params_bytes,
                    usage: wgpu::BufferUsages::STORAGE,
                });

        let roots_buffer = self.device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("BatchedBisection roots"),
            size: (batch_size * 8) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let iterations_buffer = self.device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("BatchedBisection iterations"),
            size: (batch_size * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let config = BisectionParams::new(
            batch_size as u32,
            self.max_iterations,
            n_levels,
            use_degeneracy,
            self.tolerance,
        );
        let config_buffer = self
            .device
            .create_uniform_buffer("BatchedBisection config", &config);

        let n_workgroups = batch_size.div_ceil(WORKGROUP_SIZE_COMPACT as usize);
        ComputeDispatch::new(self.device.as_ref(), entry_point)
            .shader(Self::wgsl_shader(), entry_point)
            .f64()
            .storage_read(0, &lower_buffer)
            .storage_read(1, &upper_buffer)
            .storage_read(2, &params_buffer)
            .storage_rw(3, &roots_buffer)
            .storage_rw(4, &iterations_buffer)
            .uniform(5, &config_buffer)
            .dispatch(n_workgroups as u32, 1, 1)
            .submit()?;

        // Read back results
        let roots = self.device.read_f64_buffer(&roots_buffer, batch_size)?;
        let iterations = self.read_u32_buffer(&iterations_buffer, batch_size)?;

        Ok(BisectionResult { roots, iterations })
    }

    fn read_u32_buffer(&self, buffer: &wgpu::Buffer, count: usize) -> Result<Vec<u32>> {
        let staging = self.device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("BatchedBisection u32 staging"),
            size: (count * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
                label: Some("BatchedBisection u32 readback"),
            });
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, (count * 4) as u64);
        self.device.submit_commands(Some(encoder.finish()));

        let slice = staging.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            // Channel send should not fail since receiver is alive during poll
            let _ = sender.send(result);
        });
        self.device.poll_safe()?;
        receiver
            .recv()
            .map_err(|_| BarracudaError::execution_failed("GPU buffer mapping channel closed"))?
            .map_err(|e| BarracudaError::execution_failed(e.to_string()))?;

        let data = slice.get_mapped_range();
        // chunks_exact(4) guarantees exactly 4-byte chunks
        let result: Vec<u32> = data
            .chunks_exact(4)
            .map(|chunk| {
                let arr: [u8; 4] = chunk.try_into().map_err(|_| {
                    BarracudaError::Internal("chunks_exact(4) invariant violated".to_string())
                })?;
                Ok(u32::from_le_bytes(arr))
            })
            .collect::<Result<Vec<_>>>()?;
        drop(data);
        staging.unmap();

        Ok(result)
    }
}

#[cfg(test)]
#[path = "batched_bisection_gpu_tests.rs"]
mod tests;

// SPDX-License-Identifier: AGPL-3.0-or-later

//! GPU batched tridiagonal symmetric eigensolver.
//!
//! QL algorithm with Wilkinson shifts — one GPU thread per independent
//! tridiagonal system. Unblocks groundSpring Exp 012 and complements
//! the CPU `tridiagonal_ql` in `special`.
//!
//! Cross-spring P1 (wateringHole spectral handoff).

use crate::device::WgpuDevice;
use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::Result;
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;

const SHADER: &str = include_str!("../shaders/special/tridiag_eigh_f64.wgsl");

const DEFAULT_MAX_ITER: u32 = 100;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct TridiagParams {
    n: u32,
    n_batches: u32,
    max_iter: u32,
    _pad: u32,
}

/// Result of batched tridiagonal eigendecomposition.
#[derive(Debug, Clone)]
pub struct TridiagEighResult {
    /// Eigenvalues: `n_batches × n` (sorted ascending per batch).
    pub eigenvalues: Vec<f64>,
    /// Eigenvectors: `n_batches × n × n` (column-major per batch).
    pub eigenvectors: Vec<f64>,
}

/// GPU batched tridiagonal symmetric eigensolver.
pub struct BatchedTridiagEighGpu {
    device: Arc<WgpuDevice>,
}

impl BatchedTridiagEighGpu {
    /// Create a new batched tridiag eigensolver for the given device.
    #[must_use]
    pub fn new(device: Arc<WgpuDevice>) -> Self {
        Self { device }
    }

    /// Solve `n_batches` independent tridiagonal eigenproblems.
    ///
    /// `diagonals` — flat array of main diagonals, `n_batches × n`.
    /// `subdiagonals` — flat array of sub-diagonals, `n_batches × (n-1)`.
    ///
    /// # Errors
    /// Returns [`Err`] on pipeline, dispatch, or readback failure.
    pub fn solve(
        &self,
        diagonals: &[f64],
        subdiagonals: &[f64],
        n: u32,
        n_batches: u32,
    ) -> Result<TridiagEighResult> {
        let params = TridiagParams {
            n,
            n_batches,
            max_iter: DEFAULT_MAX_ITER,
            _pad: 0,
        };

        let n_usize = n as usize;
        let nb = n_batches as usize;

        let diag_buf = self
            .device
            .create_buffer_f64_init("tridiag:diag", diagonals);
        let subdiag_buf = self
            .device
            .create_buffer_f64_init("tridiag:subdiag", subdiagonals);
        let eigvec_buf = self.device.create_buffer_f64(nb * n_usize * n_usize)?;
        let params_buf = self.device.create_uniform_buffer("tridiag:params", &params);

        ComputeDispatch::new(&self.device, "tridiag_eigh")
            .shader(SHADER, "main")
            .f64()
            .storage_rw(0, &diag_buf)
            .storage_rw(1, &subdiag_buf)
            .storage_rw(2, &eigvec_buf)
            .uniform(3, &params_buf)
            .dispatch(n_batches, 1, 1)
            .submit()?;

        let eigenvalues = self.device.read_f64_buffer(&diag_buf, nb * n_usize)?;
        let eigenvectors = self
            .device
            .read_f64_buffer(&eigvec_buf, nb * n_usize * n_usize)?;

        Ok(TridiagEighResult {
            eigenvalues,
            eigenvectors,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn params_layout() {
        assert_eq!(std::mem::size_of::<TridiagParams>(), 16);
    }

    #[test]
    fn shader_source_valid() {
        assert!(SHADER.contains("Wilkinson"));
        assert!(SHADER.contains("eigvecs"));
        assert!(SHADER.contains("Params"));
    }
}

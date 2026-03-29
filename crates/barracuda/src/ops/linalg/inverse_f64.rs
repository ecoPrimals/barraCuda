// SPDX-License-Identifier: AGPL-3.0-or-later
//! Matrix Inverse (f64) — Gauss-Jordan GPU elimination
//!
//! Computes A⁻¹ via [A | I] → [I | A⁻¹] with partial pivoting.
//! Full f64 precision for ill-conditioned matrices (κ(A) >> 1).
//!
//! Optimized for small–medium matrices (N ≤ 32) in a single workgroup.

use crate::device::WgpuDevice;
use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::{BarracudaError, Result};
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;

const SHADER: &str = include_str!("../../shaders/linalg/inverse_f64.wgsl");

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Params {
    n: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

/// GPU-accelerated matrix inverse with f64 precision.
pub struct InverseF64 {
    device: Arc<WgpuDevice>,
}

impl InverseF64 {
    /// Creates a new f64 matrix inverse calculator for the given WGPU device.
    #[must_use]
    pub fn new(device: Arc<WgpuDevice>) -> Self {
        Self { device }
    }

    /// Compute the inverse of an n×n matrix.
    /// Returns the inverse as a flat `Vec<f64>` (n×n, row-major).
    /// Returns zeros if the matrix is singular.
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn compute(&self, matrix_data: &[f64], n: usize) -> Result<Vec<f64>> {
        if matrix_data.len() != n * n {
            return Err(BarracudaError::InvalidShape {
                expected: vec![n, n],
                actual: vec![matrix_data.len()],
            });
        }

        let params = Params {
            n: n as u32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };

        let aug_size = n * 2 * n;
        let input_buf = self
            .device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("InvF64 input"),
                contents: bytemuck::cast_slice(matrix_data),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let work_buf = self.device.create_buffer_f64(aug_size)?;
        let output_buf = self.device.create_buffer_f64(n * n)?;
        let params_buf = self
            .device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("InvF64 params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        ComputeDispatch::new(self.device.as_ref(), "InvF64")
            .shader(SHADER, "main")
            .f64()
            .storage_read(0, &input_buf)
            .storage_rw(1, &work_buf)
            .storage_rw(2, &output_buf)
            .uniform(3, &params_buf)
            .dispatch(1, 1, 1)
            .submit()?;
        self.device.read_buffer_f64(&output_buf, n * n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::test_prelude::test_f64_device;

    #[tokio::test]
    async fn test_inverse_f64_identity() {
        let Some(device) = test_f64_device().await else {
            return;
        };
        let inv = InverseF64::new(device);
        let a = vec![1.0, 0.0, 0.0, 1.0];
        let result = inv.compute(&a, 2).unwrap();
        assert!((result[0] - 1.0).abs() < 1e-12);
        assert!(result[1].abs() < 1e-12);
        assert!(result[2].abs() < 1e-12);
        assert!((result[3] - 1.0).abs() < 1e-12);
    }

    #[tokio::test]
    async fn test_inverse_f64_2x2() {
        let Some(device) = test_f64_device().await else {
            return;
        };
        let inv = InverseF64::new(device);
        // [[2, 1], [1, 2]] → inverse [[2/3, -1/3], [-1/3, 2/3]]
        let a = vec![2.0, 1.0, 1.0, 2.0];
        let result = inv.compute(&a, 2).unwrap();
        assert!((result[0] - 2.0 / 3.0).abs() < 1e-12);
        assert!((result[1] + 1.0 / 3.0).abs() < 1e-12);
        assert!((result[2] + 1.0 / 3.0).abs() < 1e-12);
        assert!((result[3] - 2.0 / 3.0).abs() < 1e-12);
    }

    #[test]
    fn inverse_f64_params_layout() {
        assert_eq!(std::mem::size_of::<Params>(), 16);
    }

    #[test]
    fn inverse_f64_shader_source_valid() {
        assert!(!SHADER.is_empty());
        assert!(SHADER.contains("fn main") || SHADER.contains("@compute"));
    }
}

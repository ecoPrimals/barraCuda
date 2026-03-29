// SPDX-License-Identifier: AGPL-3.0-or-later
//! Linear System Solve (f64) — Full double-precision GPU Gaussian elimination
//!
//! **Deep Debt Principles**:
//! - Pure WGSL implementation (GPU-optimized)
//! - Safe Rust wrapper (no unsafe code)
//! - Hardware-agnostic via WebGPU
//! - Full f64 precision for ill-conditioned systems
//!
//! Solves A·x = b using Gaussian elimination with partial pivoting.
//! For systems where f32 precision is insufficient (condition number > 10⁶).

use crate::device::WgpuDevice;
use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::{BarracudaError, Result};
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;

const SHADER: &str = include_str!("../../shaders/linalg/linsolve_f64.wgsl");

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct Params {
    n: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

/// GPU-accelerated linear system solve with f64 precision.
pub struct LinSolveF64 {
    device: Arc<WgpuDevice>,
}

impl LinSolveF64 {
    /// Creates a new f64 linear system solver for the given WGPU device.
    #[must_use]
    pub fn new(device: Arc<WgpuDevice>) -> Self {
        Self { device }
    }

    /// Solve A·x = b where A is n×n and b is length n.
    /// Returns solution vector x (length n) as `Vec<f64>`.
    /// Returns zeros if the matrix is singular.
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn solve(&self, matrix_data: &[f64], rhs_data: &[f64], n: usize) -> Result<Vec<f64>> {
        if matrix_data.len() != n * n {
            return Err(BarracudaError::InvalidShape {
                expected: vec![n, n],
                actual: vec![matrix_data.len()],
            });
        }
        if rhs_data.len() != n {
            return Err(BarracudaError::InvalidShape {
                expected: vec![n],
                actual: vec![rhs_data.len()],
            });
        }

        let output_size = n * n + n;
        let params = Params {
            n: n as u32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };

        let matrix_buf = self
            .device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("LinSolveF64 A"),
                contents: bytemuck::cast_slice(matrix_data),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let rhs_buf = self
            .device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("LinSolveF64 b"),
                contents: bytemuck::cast_slice(rhs_data),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let output_buf = self.device.create_buffer_f64(output_size)?;
        let params_buf = self
            .device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("LinSolveF64 params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        ComputeDispatch::new(self.device.as_ref(), "LinSolveF64")
            .shader(SHADER, "main")
            .f64()
            .storage_read(0, &matrix_buf)
            .storage_read(1, &rhs_buf)
            .storage_rw(2, &output_buf)
            .uniform(3, &params_buf)
            .dispatch(1, 1, 1)
            .submit()?;

        let full = self.device.read_buffer_f64(&output_buf, output_size)?;
        Ok(full[n * n..].to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::test_prelude::test_f64_device;

    #[test]
    fn linsolve_f64_params_layout() {
        assert_eq!(std::mem::size_of::<Params>(), 16);
    }

    #[test]
    fn linsolve_f64_shader_source_valid() {
        assert!(!SHADER.is_empty());
        assert!(SHADER.contains("fn main") || SHADER.contains("@compute"));
    }

    #[tokio::test]
    async fn test_linsolve_f64_identity() {
        let Some(device) = test_f64_device().await else {
            return;
        };
        let solver = LinSolveF64::new(device);
        let a = vec![1.0, 0.0, 0.0, 1.0];
        let b = vec![3.0, 7.0];
        let x = solver.solve(&a, &b, 2).unwrap();
        assert!((x[0] - 3.0).abs() < 1e-12);
        assert!((x[1] - 7.0).abs() < 1e-12);
    }

    #[tokio::test]
    async fn test_linsolve_f64_2x2() {
        let Some(device) = test_f64_device().await else {
            return;
        };
        let solver = LinSolveF64::new(device);
        let a = vec![2.0, 1.0, 1.0, 2.0];
        let b = vec![5.0, 4.0];
        let x = solver.solve(&a, &b, 2).unwrap();
        assert!((x[0] - 2.0).abs() < 1e-12, "x[0]={}", x[0]);
        assert!((x[1] - 1.0).abs() < 1e-12, "x[1]={}", x[1]);
    }
}

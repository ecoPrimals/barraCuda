// SPDX-License-Identifier: AGPL-3.0-or-later
//! Spectral Normalization
//!
//! **Pure WGSL**: Single implementation via WebGPU shader
//! Normalizes weights by their spectral norm (largest singular value)

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct SpectralNormParams {
    rows: u32,
    cols: u32,
    num_iterations: u32,
    _padding: u32,
}

/// Spectral normalization: normalizes weights by their spectral norm (largest singular value).
pub struct SpectralNorm {
    weight: Tensor,
    u: Tensor,
    v: Tensor,
    num_iterations: u32,
}

impl SpectralNorm {
    /// Create `SpectralNorm` operation
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn new(weight: Tensor, u: Tensor, v: Tensor, num_iterations: u32) -> Result<Self> {
        if num_iterations == 0 {
            return Err(BarracudaError::invalid_op(
                "SpectralNorm",
                "num_iterations must be > 0",
            ));
        }

        Ok(Self {
            weight,
            u,
            v,
            num_iterations,
        })
    }

    /// WGSL shader source (embedded at compile time)
    fn wgsl_shader() -> &'static str {
        {
            const SHADER: &str = include_str!("../shaders/norm/spectral_norm_f64.wgsl");
            SHADER
        }
    }

    /// Execute `SpectralNorm` on tensor (modifies weight in-place)
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.weight.device();
        let weight_shape = self.weight.shape();

        if weight_shape.len() != 2 {
            return Err(BarracudaError::invalid_op(
                "SpectralNorm",
                format!("weight must be 2D [rows, cols], got shape {weight_shape:?}"),
            ));
        }

        let rows = weight_shape[0];
        let cols = weight_shape[1];

        // Validate u and v shapes
        if self.u.shape() != [rows] {
            return Err(BarracudaError::invalid_op(
                "SpectralNorm",
                format!("u must be 1D [rows], got shape {:?}", self.u.shape()),
            ));
        }

        if self.v.shape() != [cols] {
            return Err(BarracudaError::invalid_op(
                "SpectralNorm",
                format!("v must be 1D [cols], got shape {:?}", self.v.shape()),
            ));
        }

        let params = SpectralNormParams {
            rows: rows as u32,
            cols: cols as u32,
            num_iterations: self.num_iterations,
            _padding: 0,
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("SpectralNorm Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        ComputeDispatch::new(device, "SpectralNorm")
            .shader(Self::wgsl_shader(), "main")
            .storage_rw(0, self.weight.buffer())
            .storage_rw(1, self.u.buffer())
            .storage_rw(2, self.v.buffer())
            .uniform(3, &params_buffer)
            .dispatch_1d(rows.max(cols) as u32)
            .submit()?;

        // Return normalized weight (modified in-place)
        Ok(self.weight)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_spectral_norm_basic() {
        let device = crate::device::test_pool::get_test_device().await;
        let rows = 10;
        let cols = 8;

        let weight = Tensor::from_vec_on(vec![0.1; rows * cols], vec![rows, cols], device.clone())
            .await
            .unwrap();

        let u = Tensor::from_vec_on(vec![1.0; rows], vec![rows], device.clone())
            .await
            .unwrap();

        let v = Tensor::from_vec_on(vec![1.0; cols], vec![cols], device.clone())
            .await
            .unwrap();

        let result = SpectralNorm::new(weight, u, v, 1)
            .unwrap()
            .execute()
            .unwrap();

        assert_eq!(result.shape(), &[rows, cols]);
    }
}

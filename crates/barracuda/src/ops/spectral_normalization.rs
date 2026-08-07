// SPDX-License-Identifier: AGPL-3.0-or-later
//! Spectral Normalization - Pure WGSL
//!
//! Deep Debt Principles:
//! - Self-knowledge: Operation knows its computation
//! - Zero hardcoding: Hardware-agnostic implementation
//! - Modern idiomatic Rust: Safe, zero unsafe code
//! - Complete implementation: Production-ready, no mocks
//! - Hardware-agnostic: Pure WGSL for universal compute
//!
//! Stabilizes GAN training by constraining Lipschitz constant.
//! Used in SNGAN, `BigGAN`.
//!
//! Normalizes weights by their spectral norm (largest singular value).

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

/// Spectral Normalization operation
pub struct SpectralNormalization {
    weight: Tensor,
    u: Tensor,
    v: Tensor,
    num_iterations: u32,
}

impl SpectralNormalization {
    /// Create a new spectral normalization operation
    /// # Arguments
    /// * `weight` - Weight matrix [rows, cols]
    /// * `u` - Left singular vector [rows]
    /// * `v` - Right singular vector [cols]
    /// * `num_iterations` - Number of power iteration steps (typically 1)
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn new(weight: Tensor, u: Tensor, v: Tensor, num_iterations: u32) -> Result<Self> {
        if num_iterations == 0 {
            return Err(BarracudaError::invalid_op(
                "SpectralNormalization",
                "num_iterations must be > 0",
            ));
        }

        // Validate weight shape
        let weight_shape = weight.shape();
        if weight_shape.len() != 2 {
            return Err(BarracudaError::invalid_op(
                "SpectralNormalization",
                format!("weight must be 2D [rows, cols], got shape {weight_shape:?}"),
            ));
        }

        let rows = weight_shape[0];
        let cols = weight_shape[1];

        // Validate u and v shapes
        if u.shape() != [rows] {
            return Err(BarracudaError::invalid_op(
                "SpectralNormalization",
                format!("u must be 1D [rows], got shape {:?}", u.shape()),
            ));
        }

        if v.shape() != [cols] {
            return Err(BarracudaError::invalid_op(
                "SpectralNormalization",
                format!("v must be 1D [cols], got shape {:?}", v.shape()),
            ));
        }

        Ok(Self {
            weight,
            u,
            v,
            num_iterations,
        })
    }

    /// Get the WGSL shader source
    fn wgsl_shader() -> &'static str {
        {
            const SHADER: &str = include_str!("../shaders/norm/spectral_norm_f64.wgsl");
            SHADER
        }
    }

    /// Execute the spectral normalization operation (modifies weight in-place)
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.weight.device();
        let weight_shape = self.weight.shape();
        let rows = weight_shape[0];
        let cols = weight_shape[1];

        // Create uniform buffer for parameters
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            rows: u32,
            cols: u32,
            num_iterations: u32,
            _padding: u32,
        }

        let params = Params {
            rows: rows as u32,
            cols: cols as u32,
            num_iterations: self.num_iterations,
            _padding: 0,
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("SpectralNormalization Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        ComputeDispatch::new(device, "SpectralNormalization")
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
    async fn test_spectral_normalization() {
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

        let result = SpectralNormalization::new(weight, u, v, 1)
            .unwrap()
            .execute()
            .unwrap();

        assert_eq!(result.shape(), &[rows, cols]);
    }
}

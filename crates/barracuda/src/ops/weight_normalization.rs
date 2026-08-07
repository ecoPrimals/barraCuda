// SPDX-License-Identifier: AGPL-3.0-or-later
//! Weight Normalization - Pure WGSL
//!
//! Deep Debt Principles:
//! - Self-knowledge: Operation knows its computation
//! - Zero hardcoding: Hardware-agnostic implementation
//! - Modern idiomatic Rust: Safe, zero unsafe code
//! - Complete implementation: Production-ready, no mocks
//! - Hardware-agnostic: Pure WGSL for universal compute
//!
//! Reparameterizes weights as: w = g * (v / ||v||)
//! Decouples magnitude and direction of weight vectors.
//! Speeds up training convergence.
//!
//! Reference: "Weight Normalization" by Salimans & Kingma (2016)

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

/// Weight Normalization operation
pub struct WeightNormalization {
    v: Tensor,
    g: Tensor,
    dim: u32,
}

impl WeightNormalization {
    /// Create a new weight normalization operation
    /// # Arguments
    /// * `v` - Direction vectors (weights to normalize)
    /// * `g` - Magnitude scalars
    /// * `dim` - Dimension to normalize over (0 = all)
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn new(v: Tensor, g: Tensor, dim: u32) -> Result<Self> {
        let v_shape = v.shape();

        if v_shape.is_empty() {
            return Err(BarracudaError::invalid_op(
                "WeightNormalization",
                format!("v must have at least 1 dimension, got shape {v_shape:?}"),
            ));
        }

        Ok(Self { v, g, dim })
    }

    /// Get the WGSL shader source
    fn wgsl_shader() -> &'static str {
        {
            const SHADER: &str = include_str!("../shaders/norm/weight_norm_f64.wgsl");
            SHADER
        }
    }

    /// Execute the weight normalization operation
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.v.device();
        let v_shape = self.v.shape();
        let num_weights = self.v.len();

        // Create output buffer: same shape as v
        let output_buffer = device.create_buffer_f32(num_weights)?;

        // Create uniform buffer for parameters
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            num_weights: u32,
            dim: u32,
            _padding: [u32; 2],
        }

        let params = Params {
            num_weights: num_weights as u32,
            dim: self.dim,
            _padding: [0; 2],
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("WeightNormalization Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        ComputeDispatch::new(device, "WeightNormalization")
            .shader(Self::wgsl_shader(), "main")
            .storage_read(0, self.v.buffer())
            .storage_read(1, self.g.buffer())
            .storage_rw(2, &output_buffer)
            .uniform(3, &params_buffer)
            .dispatch_1d(num_weights as u32)
            .submit()?;

        // Create output tensor
        Ok(Tensor::from_buffer(
            output_buffer,
            v_shape.to_vec(),
            device.clone(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_weight_normalization() {
        let device = crate::device::test_pool::get_test_device().await;
        // 2 filters, 3 weights each
        let v = Tensor::from_vec_on(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],
            device.clone(),
        )
        .await
        .unwrap();

        let g = Tensor::from_vec_on(vec![2.0], vec![1], device.clone())
            .await
            .unwrap();

        let result = WeightNormalization::new(v, g, 0)
            .unwrap()
            .execute()
            .unwrap();

        assert_eq!(result.shape(), &[2, 3]);
        let data = result.to_vec().unwrap();
        assert!(data.iter().all(|&x| x.is_finite()));
    }
}

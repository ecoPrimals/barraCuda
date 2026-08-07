// SPDX-License-Identifier: AGPL-3.0-or-later
//! Weight Normalization
//!
//! **Pure WGSL**: Single implementation via WebGPU shader
//! Reparameterizes weights as: w = g * (v / ||v||)

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct WeightNormParams {
    num_weights: u32,
    dim: u32,
    _padding: [u32; 2],
}

/// Weight normalization: reparameterizes weights as w = g * (v / ||v||).
pub struct WeightNorm {
    v: Tensor,
    g: Tensor,
    dim: u32,
}

impl WeightNorm {
    /// Create `WeightNorm` operation
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn new(v: Tensor, g: Tensor, dim: u32) -> Result<Self> {
        Ok(Self { v, g, dim })
    }

    /// WGSL shader source (embedded at compile time)
    fn wgsl_shader() -> &'static str {
        {
            const SHADER: &str = include_str!("../shaders/norm/weight_norm_f64.wgsl");
            SHADER
        }
    }

    /// Execute `WeightNorm` on tensor
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.v.device();
        let v_shape = self.v.shape();

        if v_shape.is_empty() {
            return Err(BarracudaError::invalid_op(
                "WeightNorm",
                format!("v must have at least 1 dimension, got shape {v_shape:?}"),
            ));
        }

        let num_weights = self.v.len();

        // Create output buffer: same shape as v
        let output_buffer = device.create_buffer_f32(num_weights)?;

        let params = WeightNormParams {
            num_weights: num_weights as u32,
            dim: self.dim,
            _padding: [0; 2],
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("WeightNorm Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        ComputeDispatch::new(device, "WeightNorm")
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
    async fn test_weight_norm_basic() {
        let device = crate::device::test_pool::get_test_device().await;
        let num_weights = 20;

        let v = Tensor::from_vec_on(vec![1.0; num_weights], vec![num_weights], device.clone())
            .await
            .unwrap();

        let g = Tensor::from_vec_on(vec![2.0], vec![1], device.clone())
            .await
            .unwrap();

        let result = WeightNorm::new(v, g, 0).unwrap().execute().unwrap();

        assert_eq!(result.shape(), &[num_weights]);
    }
}

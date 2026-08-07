// SPDX-License-Identifier: AGPL-3.0-or-later
//! `LayerScale` - Per-layer learnable scaling
//!
//! **Canonical `BarraCuda` Pattern**: Struct with new/execute
//!
//! Used in vision transformers (`CaiT`, `LeViT`) to stabilize training.
//!
//! ## Algorithm
//!
//! ```text
//! LayerScale(x) = gamma ⊙ x
//! ```
//!
//! Where gamma is a learnable per-channel parameter.

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

/// `LayerScale` operation
pub struct LayerScale {
    input: Tensor,
    gamma: Tensor,
}

impl LayerScale {
    /// Create a new layer scale operation
    /// # Errors
    /// Returns [`Err`] if input and gamma shapes do not match.
    pub fn new(input: Tensor, gamma: Tensor) -> Result<Self> {
        // Validate shapes match
        if input.shape() != gamma.shape() {
            return Err(BarracudaError::shape_mismatch(
                input.shape().to_vec(),
                gamma.shape().to_vec(),
            ));
        }

        Ok(Self { input, gamma })
    }

    /// Execute the layer scale operation
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let size = self.input.len();

        // Create output buffer
        let output_buffer = device.create_buffer_f32(size)?;

        // Create uniform buffer for parameters
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            size: u32,
            _padding: [u32; 3],
        }

        let params = Params {
            size: size as u32,
            _padding: [0; 3],
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("LayerScale Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        ComputeDispatch::new(device, "LayerScale")
            .shader(include_str!("../shaders/misc/layer_scale_f64.wgsl"), "main")
            .storage_read(0, self.input.buffer())
            .storage_read(1, self.gamma.buffer())
            .storage_rw(2, &output_buffer)
            .uniform(3, &params_buffer)
            .dispatch_1d(size as u32)
            .submit()?;

        // Return tensor with same shape as input
        Ok(Tensor::from_buffer(
            output_buffer,
            self.input.shape().to_vec(),
            device.clone(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_layer_scale_basic() {
        let device = crate::device::test_pool::get_test_device().await;
        let input = Tensor::from_vec_on(vec![1.0, 2.0, 3.0], vec![3], device.clone())
            .await
            .unwrap();
        let gamma = Tensor::from_vec_on(vec![0.1, 0.2, 0.3], vec![3], device.clone())
            .await
            .unwrap();
        let output = LayerScale::new(input, gamma).unwrap().execute().unwrap();
        let result = output.to_vec().unwrap();
        assert_eq!(result.len(), 3);
        assert!((result[0] - 0.1).abs() < 1e-5);
        assert!((result[1] - 0.4).abs() < 1e-5);
        assert!((result[2] - 0.9).abs() < 1e-5);
    }

    #[tokio::test]
    async fn test_layer_scale_edge_cases() {
        let device = crate::device::test_pool::get_test_device().await;
        // Single element
        let input = Tensor::from_vec_on(vec![5.0], vec![1], device.clone())
            .await
            .unwrap();
        let gamma = Tensor::from_vec_on(vec![0.5], vec![1], device.clone())
            .await
            .unwrap();
        let output = LayerScale::new(input, gamma).unwrap().execute().unwrap();
        let result = output.to_vec().unwrap();
        assert_eq!(result.len(), 1);
        assert!((result[0] - 2.5).abs() < 1e-5);

        // All zeros
        let input = Tensor::from_vec_on(vec![0.0, 0.0, 0.0], vec![3], device.clone())
            .await
            .unwrap();
        let gamma = Tensor::from_vec_on(vec![1.0, 2.0, 3.0], vec![3], device.clone())
            .await
            .unwrap();
        let output = LayerScale::new(input, gamma).unwrap().execute().unwrap();
        let result = output.to_vec().unwrap();
        assert!(result.iter().all(|&x| x.abs() < 1e-5));
    }

    #[tokio::test]
    async fn test_layer_scale_boundary() {
        let device = crate::device::test_pool::get_test_device().await;
        // Gamma = 0 (complete suppression)
        let input = Tensor::from_vec_on(vec![1.0, 2.0, 3.0], vec![3], device.clone())
            .await
            .unwrap();
        let gamma = Tensor::from_vec_on(vec![0.0, 0.0, 0.0], vec![3], device.clone())
            .await
            .unwrap();
        let output = LayerScale::new(input, gamma).unwrap().execute().unwrap();
        let result = output.to_vec().unwrap();
        assert!(result.iter().all(|&x| x.abs() < 1e-5));

        // Gamma = 1 (identity)
        let input = Tensor::from_vec_on(vec![1.0, 2.0, 3.0], vec![3], device.clone())
            .await
            .unwrap();
        let gamma = Tensor::from_vec_on(vec![1.0, 1.0, 1.0], vec![3], device.clone())
            .await
            .unwrap();
        let output = LayerScale::new(input.clone(), gamma)
            .unwrap()
            .execute()
            .unwrap();
        let result = output.to_vec().unwrap();
        let input_vec = input.to_vec().unwrap();
        for (r, i) in result.iter().zip(input_vec.iter()) {
            assert!((r - i).abs() < 1e-5);
        }
    }
}

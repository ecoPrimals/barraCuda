// SPDX-License-Identifier: AGPL-3.0-or-later
//! `SpatialDropout` - Spatial Dropout (Channel-wise dropout)
//!
//! **Deep Debt Principles**:
//! - ✅ Pure WGSL implementation
//! - ✅ Safe Rust wrapper (no unsafe code)
//! - ✅ Hardware-agnostic via WebGPU
//! - ✅ Complete implementation (production-ready)
//! - ✅ Modern idiomatic Rust (no traits, direct impl)
//!
//! Drops entire feature maps (channels) instead of individual elements
//! More effective for convolutional networks
//!
//! Reference: "Efficient Object Localization Using Convolutional Networks" by Tompson et al.

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

/// f64 is the canonical source — f32 derived via `downcast_f64_to_f32` when needed.
const SHADER_F64: &str = include_str!("../shaders/dropout/spatial_dropout_f64.wgsl");

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct SpatialDropoutParams {
    batch_size: u32,
    channels: u32,
    height: u32,
    width: u32,
    drop_prob: f32,
    training: u32,
    _padding: u32,
    _padding2: u32,
}

/// Channel-wise spatial dropout for convolutional networks.
pub struct SpatialDropout {
    input: Tensor,
    mask: Tensor,
    drop_prob: f32,
    training: bool,
}

impl SpatialDropout {
    /// Creates a new spatial dropout operation. Mask shape must be [B, C].
    /// # Errors
    /// Returns [`Err`] if input is not 4D, mask shape does not match [B, C], or `drop_prob` is not in [0, 1).
    pub fn new(input: Tensor, mask: Tensor, drop_prob: f32, training: bool) -> Result<Self> {
        // Validate input shape: must be 4D [B, C, H, W]
        let input_shape = input.shape();
        if input_shape.len() != 4 {
            return Err(BarracudaError::invalid_op(
                "spatial_dropout",
                "input must be 4D tensor [B, C, H, W]",
            ));
        }

        // Validate mask shape: must be [B, C]
        let mask_shape = mask.shape();
        if mask_shape.len() != 2
            || mask_shape[0] != input_shape[0]
            || mask_shape[1] != input_shape[1]
        {
            return Err(BarracudaError::invalid_op(
                "spatial_dropout",
                "mask must be 2D tensor [B, C] matching input batch and channels",
            ));
        }

        if !(0.0..1.0).contains(&drop_prob) {
            return Err(BarracudaError::invalid_op(
                "spatial_dropout",
                "drop_prob must be in range [0.0, 1.0)",
            ));
        }

        Ok(Self {
            input,
            mask,
            drop_prob,
            training,
        })
    }

    /// Executes spatial dropout and returns the output tensor.
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or device submission fails (e.g. device lost).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let shape = self.input.shape();
        let batch_size = shape[0];
        let channels = shape[1];
        let height = shape[2];
        let width = shape[3];

        let output_size = batch_size * channels * height * width;
        let output_buffer = device.create_buffer_f32(output_size)?;

        let params = SpatialDropoutParams {
            batch_size: batch_size as u32,
            channels: channels as u32,
            height: height as u32,
            width: width as u32,
            drop_prob: self.drop_prob,
            training: u32::from(self.training),
            _padding: 0,
            _padding2: 0,
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("spatial_dropout_params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let workgroups_x = (width as u32).div_ceil(8);
        let workgroups_y = (height as u32).div_ceil(8);
        let workgroups_z = (batch_size * channels) as u32;

        ComputeDispatch::new(device, "SpatialDropout")
            .shader(SHADER_F64, "main")
            .storage_read(0, self.input.buffer())
            .storage_read(1, self.mask.buffer())
            .storage_rw(2, &output_buffer)
            .uniform(3, &params_buffer)
            .dispatch(workgroups_x, workgroups_y, workgroups_z)
            .submit()?;

        Ok(Tensor::from_buffer(
            output_buffer,
            vec![batch_size, channels, height, width],
            device.clone(),
        ))
    }
}

impl Tensor {
    /// Apply spatial dropout (channel-wise dropout)
    /// # Arguments
    /// - `mask`: Channel mask tensor [B, C] (random values on CPU, passed in)
    /// - `drop_prob`: Dropout probability
    /// - `training`: Whether in training mode
    /// # Errors
    /// Returns [`Err`] if validation fails or buffer allocation/GPU dispatch fails (e.g. device lost).
    pub fn spatial_dropout(self, mask: Self, drop_prob: f32, training: bool) -> Result<Self> {
        SpatialDropout::new(self, mask, drop_prob, training)?.execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_spatial_dropout_basic() {
        let device = crate::device::test_pool::get_test_device().await;
        let input = Tensor::from_vec_on(vec![1.0; 3 * 4 * 4], vec![1, 3, 4, 4], device.clone())
            .await
            .unwrap();
        let mask = Tensor::from_vec_on(vec![1.0; 3], vec![1, 3], device.clone())
            .await
            .unwrap();

        let output = input.spatial_dropout(mask, 0.5, true).unwrap();
        let result = output.to_vec().unwrap();

        assert_eq!(output.shape(), &[1, 3, 4, 4]);
        assert_eq!(result.len(), 48);
        assert!(result.iter().all(|&x| x.is_finite()));
    }
}

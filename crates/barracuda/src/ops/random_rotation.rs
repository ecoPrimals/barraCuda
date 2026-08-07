// SPDX-License-Identifier: AGPL-3.0-or-later
//! Random rotation augmentation
//!
//! **Pure WGSL**: Single implementation via WebGPU shader
//! Rotates images by random angles
//! Shader: f64 canonical (downcast to f32 at compile)

const SHADER_F64: &str = include_str!("../shaders/augmentation/random_rotation_f64.wgsl");

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct RandomRotationParams {
    batch_size: u32,
    channels: u32,
    height: u32,
    width: u32,
    fill_value: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
    _pad4: u32,
    _pad5: u32,
    _pad6: u32,
}

/// Random rotation augmentation: rotates images by per-batch rotation matrices.
pub struct RandomRotation {
    input: Tensor,
    rotation_matrices: Tensor,
    fill_value: f32,
}

impl RandomRotation {
    /// Create `RandomRotation` operation
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn new(input: Tensor, rotation_matrices: Tensor, fill_value: f32) -> Result<Self> {
        // Validate rotation_matrices shape: [batch_size, 4] (cos, -sin, sin, cos)
        let rot_shape = rotation_matrices.shape();
        if rot_shape.len() != 2 || rot_shape[1] != 4 {
            return Err(BarracudaError::invalid_op(
                "RandomRotation",
                format!("rotation_matrices must be 2D [batch_size, 4], got shape {rot_shape:?}"),
            ));
        }

        Ok(Self {
            input,
            rotation_matrices,
            fill_value,
        })
    }

    /// WGSL shader source (f64 canonical, downcast to f32 at compile)
    fn wgsl_shader() -> &'static str {
        SHADER_F64
    }

    /// Execute `RandomRotation` on tensor
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let input_shape = self.input.shape();

        if input_shape.len() != 4 {
            return Err(BarracudaError::invalid_op(
                "RandomRotation",
                format!(
                    "input must be 4D [batch, channels, height, width], got shape {input_shape:?}"
                ),
            ));
        }

        let batch_size = input_shape[0];
        let channels = input_shape[1];
        let height = input_shape[2];
        let width = input_shape[3];

        if self.rotation_matrices.shape()[0] != batch_size {
            return Err(BarracudaError::invalid_op(
                "RandomRotation",
                format!(
                    "rotation_matrices batch size {} must match input batch size {}",
                    self.rotation_matrices.shape()[0],
                    batch_size
                ),
            ));
        }

        // Create output buffer: [batch, channels, height, width]
        let output_size = batch_size * channels * height * width;
        let output_buffer = device.create_buffer_f32(output_size)?;

        let params = RandomRotationParams {
            batch_size: batch_size as u32,
            channels: channels as u32,
            height: height as u32,
            width: width as u32,
            fill_value: self.fill_value,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
            _pad3: 0,
            _pad4: 0,
            _pad5: 0,
            _pad6: 0,
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("RandomRotation Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let workgroups_x = (width as u32).div_ceil(8);
        let workgroups_y = (height as u32).div_ceil(8);
        let workgroups_z = (batch_size * channels) as u32;

        ComputeDispatch::new(device, "RandomRotation")
            .shader(Self::wgsl_shader(), "main")
            .storage_read(0, self.input.buffer())
            .storage_read(1, self.rotation_matrices.buffer())
            .storage_rw(2, &output_buffer)
            .uniform(3, &params_buffer)
            .dispatch(workgroups_x, workgroups_y, workgroups_z)
            .submit()?;

        // Create output tensor
        Ok(Tensor::from_buffer(
            output_buffer,
            input_shape.to_vec(),
            device.clone(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_random_rotation_basic() {
        let device = crate::device::test_pool::get_test_device().await;
        let batch_size = 2;
        let channels = 3;
        let height = 32;
        let width = 32;

        let input = Tensor::from_vec_on(
            vec![1.0; batch_size * channels * height * width],
            vec![batch_size, channels, height, width],
            device.clone(),
        )
        .await
        .unwrap();

        // Rotation matrices: [cos, -sin, sin, cos] for each batch item
        let rotation_matrices = Tensor::from_vec_on(
            vec![1.0, 0.0, 0.0, 1.0, 0.707, -0.707, 0.707, 0.707], // Identity and 45° rotation
            vec![batch_size, 4],
            device.clone(),
        )
        .await
        .unwrap();

        let result = RandomRotation::new(input, rotation_matrices, 0.0)
            .unwrap()
            .execute()
            .unwrap();

        assert_eq!(result.shape(), &[batch_size, channels, height, width]);
    }
}

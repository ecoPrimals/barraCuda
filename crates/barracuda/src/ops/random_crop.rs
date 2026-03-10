// SPDX-License-Identifier: AGPL-3.0-only
//! Random crop augmentation
//!
//! **Pure WGSL**: Single implementation via WebGPU shader
//! Randomly crops images to specified size

use crate::device::compute_pipeline::ComputeDispatch;
use crate::device::{DeviceCapabilities, WorkloadType};
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

const SHADER_F64: &str = include_str!("../shaders/augmentation/random_crop_f64.wgsl");
static SHADER_F32: std::sync::LazyLock<String> =
    std::sync::LazyLock::new(|| SHADER_F64.to_string());

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct RandomCropParams {
    batch_size: u32,
    channels: u32,
    in_height: u32,
    in_width: u32,
    out_height: u32,
    out_width: u32,
    _padding: [u32; 2],
}

/// Random crop augmentation: crops images to specified size at per-batch positions.
pub struct RandomCrop {
    input: Tensor,
    crop_positions: Tensor,
    out_height: usize,
    out_width: usize,
}

impl RandomCrop {
    /// Create `RandomCrop` operation
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if `crop_positions` is not 2D [`batch_size`, 2].
    pub fn new(
        input: Tensor,
        crop_positions: Tensor,
        out_height: usize,
        out_width: usize,
    ) -> Result<Self> {
        // Validate crop_positions shape: [batch_size, 2] (top, left)
        let crop_shape = crop_positions.shape();
        if crop_shape.len() != 2 || crop_shape[1] != 2 {
            return Err(BarracudaError::invalid_op(
                "RandomCrop",
                format!("crop_positions must be 2D [batch_size, 2], got shape {crop_shape:?}"),
            ));
        }

        Ok(Self {
            input,
            crop_positions,
            out_height,
            out_width,
        })
    }

    /// WGSL shader source (embedded at compile time)
    fn wgsl_shader() -> &'static str {
        &SHADER_F32
    }

    /// Execute `RandomCrop` on tensor
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
                "RandomCrop",
                format!(
                    "input must be 4D [batch, channels, height, width], got shape {input_shape:?}"
                ),
            ));
        }

        let batch_size = input_shape[0];
        let channels = input_shape[1];
        let in_height = input_shape[2];
        let in_width = input_shape[3];

        if self.crop_positions.shape()[0] != batch_size {
            return Err(BarracudaError::invalid_op(
                "RandomCrop",
                format!(
                    "crop_positions batch size {} must match input batch size {}",
                    self.crop_positions.shape()[0],
                    batch_size
                ),
            ));
        }

        // Create output buffer: [batch, channels, out_height, out_width]
        let output_size = batch_size * channels * self.out_height * self.out_width;
        let output_buffer = device.create_buffer_f32(output_size)?;

        let params = RandomCropParams {
            batch_size: batch_size as u32,
            channels: channels as u32,
            in_height: in_height as u32,
            in_width: in_width as u32,
            out_height: self.out_height as u32,
            out_width: self.out_width as u32,
            _padding: [0; 2],
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("RandomCrop Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let caps = DeviceCapabilities::from_device(device.as_ref());
        let (wg_x, wg_y, wg_z) = caps.optimal_workgroup_size_3d(WorkloadType::Convolution);
        let workgroups_x = (self.out_width as u32).div_ceil(wg_x);
        let workgroups_y = (self.out_height as u32).div_ceil(wg_y);
        let workgroups_z = ((batch_size * channels) as u32).div_ceil(wg_z);

        ComputeDispatch::new(device.as_ref(), "RandomCrop")
            .shader(Self::wgsl_shader(), "main")
            .storage_read(0, self.input.buffer())
            .storage_read(1, self.crop_positions.buffer())
            .storage_rw(2, &output_buffer)
            .uniform(3, &params_buffer)
            .dispatch(workgroups_x, workgroups_y, workgroups_z)
            .submit()?;

        // Create output tensor
        Ok(Tensor::from_buffer(
            output_buffer,
            vec![batch_size, channels, self.out_height, self.out_width],
            device.clone(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device_if_gpu_available;

    #[tokio::test]
    async fn test_random_crop_basic() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let batch_size = 2;
        let channels = 3;
        let in_height = 32;
        let in_width = 32;
        let out_height = 16;
        let out_width = 16;

        let input = Tensor::from_vec_on(
            vec![1.0; batch_size * channels * in_height * in_width],
            vec![batch_size, channels, in_height, in_width],
            device.clone(),
        )
        .await
        .unwrap();

        let crop_positions = Tensor::from_vec_on(
            vec![5.0, 5.0, 10.0, 10.0], // [batch, 2] - (top, left)
            vec![batch_size, 2],
            device.clone(),
        )
        .await
        .unwrap();

        let result = RandomCrop::new(input, crop_positions, out_height, out_width)
            .unwrap()
            .execute()
            .unwrap();

        assert_eq!(
            result.shape(),
            &[batch_size, channels, out_height, out_width]
        );
    }
}

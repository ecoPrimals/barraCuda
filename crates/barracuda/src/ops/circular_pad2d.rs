// SPDX-License-Identifier: AGPL-3.0-or-later
//! Circular pad 2D operation - Circular/wrap padding for 2D tensors
//!
//! Wraps edges around (toroidal topology)
//! Useful for periodic boundary conditions

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;
use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct CircularPad2dParams {
    batch_size: u32,
    channels: u32,
    in_height: u32,
    in_width: u32,
    out_height: u32,
    out_width: u32,
    pad_top: u32,
    pad_left: u32,
}

/// Circular pad 2D operation
pub struct CircularPad2d {
    input: Tensor,
    pad_top: u32,
    pad_bottom: u32,
    pad_left: u32,
    pad_right: u32,
}

impl CircularPad2d {
    /// Create circular pad 2D operation
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn new(
        input: Tensor,
        pad_top: u32,
        pad_bottom: u32,
        pad_left: u32,
        pad_right: u32,
    ) -> Result<Self> {
        let shape = input.shape();
        if shape.len() != 4 {
            return Err(BarracudaError::invalid_op(
                "circular_pad2d",
                format!("input must be 4D [B, C, H, W], got shape {shape:?}"),
            ));
        }

        Ok(Self {
            input,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
        })
    }

    /// Execute circular pad 2D on tensor
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let shape = self.input.shape();
        let batch_size = shape[0];
        let channels = shape[1];
        let in_height = shape[2];
        let in_width = shape[3];

        let out_height = in_height + self.pad_top as usize + self.pad_bottom as usize;
        let out_width = in_width + self.pad_left as usize + self.pad_right as usize;
        let output_size = batch_size * channels * out_height * out_width;

        // Create output buffer
        let output_buffer = device.create_buffer_f32(output_size)?;

        // Create params
        let params = CircularPad2dParams {
            batch_size: batch_size as u32,
            channels: channels as u32,
            in_height: in_height as u32,
            in_width: in_width as u32,
            out_height: out_height as u32,
            out_width: out_width as u32,
            pad_top: self.pad_top,
            pad_left: self.pad_left,
        };

        let params_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("CircularPad2d Params"),
            size: std::mem::size_of::<CircularPad2dParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        device
            .queue
            .write_buffer(&params_buffer, 0, bytemuck::bytes_of(&params));

        let workgroups_x = (out_width as u32).div_ceil(8);
        let workgroups_y = (out_height as u32).div_ceil(8);
        let workgroups_z = (batch_size * channels) as u32;

        ComputeDispatch::new(device, "CircularPad2d")
            .shader(
                include_str!("../shaders/tensor/circular_pad2d_f64.wgsl"),
                "main",
            )
            .storage_read(0, self.input.buffer())
            .storage_rw(1, &output_buffer)
            .uniform(2, &params_buffer)
            .dispatch(workgroups_x, workgroups_y, workgroups_z)
            .submit()?;

        // Create output tensor
        Ok(Tensor::from_buffer(
            output_buffer,
            vec![batch_size, channels, out_height, out_width],
            device.clone(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_circular_pad2d_basic() {
        let device = crate::device::test_pool::get_test_device().await;
        let input = Tensor::from_vec_on(vec![1.0; 2 * 3 * 4 * 4], vec![2, 3, 4, 4], device)
            .await
            .unwrap();

        let output = CircularPad2d::new(input, 1, 1, 1, 1)
            .unwrap()
            .execute()
            .unwrap();
        let result = output.to_vec().unwrap();

        // Output should be [2, 3, 6, 6]
        assert_eq!(result.len(), 2 * 3 * 6 * 6);
    }
}

// SPDX-License-Identifier: AGPL-3.0-or-later
//! Fold - Pure WGSL
//!
//! Deep Debt Principles:
//! - Self-knowledge: Operation knows its computation
//! - Zero hardcoding: Hardware-agnostic implementation
//! - Modern idiomatic Rust: Safe, zero unsafe code
//! - Complete implementation: Production-ready, no mocks
//! - Hardware-agnostic: Pure WGSL for universal compute

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::Result;
use crate::tensor::Tensor;

/// Fold operation (col2im - inverse of unfold)
pub struct Fold {
    input: Tensor,
    output_size: (usize, usize),
    kernel_size: (usize, usize),
    stride: usize,
    padding: usize,
    dilation: usize,
}

impl Fold {
    /// Create a new fold operation
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn new(
        input: Tensor,
        output_size: (usize, usize),
        kernel_size: (usize, usize),
        stride: usize,
        padding: usize,
        dilation: usize,
    ) -> Result<Self> {
        let shape = input.shape();
        if shape.len() != 3 {
            return Err(crate::error::BarracudaError::invalid_input(format!(
                "Fold expects 3D tensor [B, C*K*K, L], got shape {shape:?}"
            )));
        }

        Ok(Self {
            input,
            output_size,
            kernel_size,
            stride,
            padding,
            dilation,
        })
    }

    /// Execute the fold operation
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let shape = self.input.shape();
        let batch_size = shape[0];
        let channels_times_kernel = shape[1];

        // Infer channels from input shape
        // channels_times_kernel = channels * kernel_height * kernel_width
        let kernel_elements = self.kernel_size.0 * self.kernel_size.1;
        let channels = channels_times_kernel / kernel_elements;

        if !channels_times_kernel.is_multiple_of(kernel_elements) {
            return Err(crate::error::BarracudaError::invalid_input(format!(
                "Input channels*kernel ({channels_times_kernel}) must be divisible by kernel elements ({kernel_elements})"
            )));
        }

        let out_height = self.output_size.0;
        let out_width = self.output_size.1;
        let output_size = batch_size * channels * out_height * out_width;

        // Compute number of blocks
        let num_blocks_h =
            ((out_height + 2 * self.padding - self.dilation * (self.kernel_size.0 - 1) - 1)
                / self.stride)
                + 1;
        let num_blocks_w =
            ((out_width + 2 * self.padding - self.dilation * (self.kernel_size.1 - 1) - 1)
                / self.stride)
                + 1;

        // Access input buffer directly (zero-copy)
        let input_buffer = self.input.buffer();

        // Create output buffer
        let output_buffer = device.create_buffer_f32(output_size)?;

        // Create uniform buffer for parameters
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            batch_size: u32,
            channels: u32,
            out_height: u32,
            out_width: u32,
            kernel_height: u32,
            kernel_width: u32,
            stride: u32,
            padding: u32,
            dilation: u32,
            num_blocks_h: u32,
            num_blocks_w: u32,
            _pad1: u32,
        }

        let params = Params {
            batch_size: batch_size as u32,
            channels: channels as u32,
            out_height: out_height as u32,
            out_width: out_width as u32,
            kernel_height: self.kernel_size.0 as u32,
            kernel_width: self.kernel_size.1 as u32,
            stride: self.stride as u32,
            padding: self.padding as u32,
            dilation: self.dilation as u32,
            num_blocks_h: num_blocks_h as u32,
            num_blocks_w: num_blocks_w as u32,
            _pad1: 0,
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Fold Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let workgroups_x = (out_width as u32).div_ceil(8);
        let workgroups_y = (out_height as u32).div_ceil(8);
        let workgroups_z = (batch_size * channels) as u32;

        ComputeDispatch::new(device, "Fold")
            .shader(include_str!("../shaders/tensor/fold_f64.wgsl"), "main")
            .uniform(0, &params_buffer)
            .storage_read(1, input_buffer)
            .storage_rw(2, &output_buffer)
            .dispatch(workgroups_x, workgroups_y, workgroups_z)
            .submit()?;

        // Return tensor without reading back (zero-copy)
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
    async fn test_fold_basic() {
        let device = crate::device::test_pool::get_test_device().await;
        // Input shape: [B, C*K*K, L] where K=3, so C*9
        let data: Vec<f32> = (0..324).map(|i| i as f32).collect();
        let input = Tensor::from_data(
            &data,
            vec![1, 9, 36], // 1 channel * 9 kernel elements, 36 blocks
            device,
        )
        .unwrap();

        let folded = Fold::new(input, (6, 6), (3, 3), 1, 0, 1)
            .unwrap()
            .execute()
            .unwrap();
        assert_eq!(folded.shape(), &vec![1, 1, 6, 6]);
    }

    #[tokio::test]
    async fn test_fold_invalid_shape() {
        let device = crate::device::test_pool::get_test_device().await;
        let input = Tensor::from_data(&[1.0, 2.0, 3.0], vec![3], device).unwrap();

        assert!(Fold::new(input, (4, 4), (3, 3), 1, 0, 1).is_err());
    }

    #[tokio::test]
    async fn test_fold_with_padding() {
        let device = crate::device::test_pool::get_test_device().await;
        let data: Vec<f32> = (0..576).map(|i| i as f32).collect();
        let input = Tensor::from_data(&data, vec![1, 9, 64], device).unwrap();

        let folded = Fold::new(input, (8, 8), (3, 3), 1, 1, 1)
            .unwrap()
            .execute()
            .unwrap();
        assert_eq!(folded.shape(), &vec![1, 1, 8, 8]);
    }
}

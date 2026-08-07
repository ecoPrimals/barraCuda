// SPDX-License-Identifier: AGPL-3.0-or-later
//! `MaxPool3D` - 3D Max Pooling
//!
//! **Deep Debt Principles**:
//! - ✅ Pure WGSL implementation
//! - ✅ Safe Rust wrapper (no unsafe code)
//! - ✅ Hardware-agnostic via WebGPU
//! - ✅ Complete implementation (production-ready)
//! - ✅ Modern idiomatic Rust (no traits, direct impl)
//!
//! Max pooling for 3D data (video, volumetric medical imaging)
//! Commonly used in 3D CNNs for action recognition and medical imaging

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct MaxPool3DParams {
    batch_size: u32,
    channels: u32,
    in_depth: u32,
    in_height: u32,
    in_width: u32,
    out_depth: u32,
    out_height: u32,
    out_width: u32,
    kernel_d: u32,
    kernel_h: u32,
    kernel_w: u32,
    stride_d: u32,
    stride_h: u32,
    stride_w: u32,
    pad_d: u32,
    pad_h: u32,
    pad_w: u32,
    _padding: u32,
}

/// 3D max pooling for volumetric data (video, medical imaging).
pub struct MaxPool3D {
    input: Tensor,
    kernel_size: (usize, usize, usize),
    stride: (usize, usize, usize),
    padding: (usize, usize, usize),
}

impl MaxPool3D {
    /// Creates a new `MaxPool3D`. Input must be 5D [B, C, D, H, W].
    /// # Errors
    /// Returns [`Err`] if input is not 5D, kernel sizes are zero, or strides are zero.
    pub fn new(
        input: Tensor,
        kernel_size: (usize, usize, usize),
        stride: (usize, usize, usize),
        padding: (usize, usize, usize),
    ) -> Result<Self> {
        // Validate input shape: must be 5D [B, C, D, H, W]
        let shape = input.shape();
        if shape.len() != 5 {
            return Err(BarracudaError::invalid_op(
                "maxpool3d",
                "input must be 5D tensor [B, C, D, H, W]",
            ));
        }

        // Validate kernel sizes
        if kernel_size.0 == 0 || kernel_size.1 == 0 || kernel_size.2 == 0 {
            return Err(BarracudaError::invalid_op(
                "maxpool3d",
                "kernel_size must be positive",
            ));
        }

        // Validate strides
        if stride.0 == 0 || stride.1 == 0 || stride.2 == 0 {
            return Err(BarracudaError::invalid_op(
                "maxpool3d",
                "stride must be positive",
            ));
        }

        Ok(Self {
            input,
            kernel_size,
            stride,
            padding,
        })
    }

    /// Executes 3D max pooling and returns the output tensor.
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or device submission fails (e.g. device lost).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let shape = self.input.shape();
        let batch_size = shape[0];
        let channels = shape[1];
        let in_depth = shape[2];
        let in_height = shape[3];
        let in_width = shape[4];

        // Calculate output dimensions
        let out_depth = ((in_depth + 2 * self.padding.0 - self.kernel_size.0) / self.stride.0) + 1;
        let out_height =
            ((in_height + 2 * self.padding.1 - self.kernel_size.1) / self.stride.1) + 1;
        let out_width = ((in_width + 2 * self.padding.2 - self.kernel_size.2) / self.stride.2) + 1;

        let output_size = batch_size * channels * out_depth * out_height * out_width;
        let output_buffer = device.create_buffer_f32(output_size)?;

        let params = MaxPool3DParams {
            batch_size: batch_size as u32,
            channels: channels as u32,
            in_depth: in_depth as u32,
            in_height: in_height as u32,
            in_width: in_width as u32,
            out_depth: out_depth as u32,
            out_height: out_height as u32,
            out_width: out_width as u32,
            kernel_d: self.kernel_size.0 as u32,
            kernel_h: self.kernel_size.1 as u32,
            kernel_w: self.kernel_size.2 as u32,
            stride_d: self.stride.0 as u32,
            stride_h: self.stride.1 as u32,
            stride_w: self.stride.2 as u32,
            pad_d: self.padding.0 as u32,
            pad_h: self.padding.1 as u32,
            pad_w: self.padding.2 as u32,
            _padding: 0,
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("maxpool3d_params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let workgroups_x = (out_width as u32).div_ceil(8);
        let workgroups_y = (out_height as u32).div_ceil(8);
        let workgroups_z = (out_depth as u32).div_ceil(8);

        ComputeDispatch::new(device, "MaxPool3D")
            .shader(include_str!("../shaders/pooling/maxpool3d_f64.wgsl"), "main")
            .storage_read(0, self.input.buffer())
            .storage_rw(1, &output_buffer)
            .uniform(2, &params_buffer)
            .dispatch(workgroups_x, workgroups_y, workgroups_z)
            .submit()?;

        Ok(Tensor::from_buffer(
            output_buffer,
            vec![batch_size, channels, out_depth, out_height, out_width],
            device.clone(),
        ))
    }
}

impl Tensor {
    /// Apply 3D max pooling
    /// # Arguments
    /// - `kernel_size`: (depth, height, width) kernel dimensions
    /// - `stride`: (depth, height, width) stride dimensions
    /// - `padding`: (depth, height, width) padding dimensions
    /// # Errors
    /// Returns [`Err`] if input is not 5D, kernel/stride are invalid, or buffer allocation/GPU dispatch fails (e.g. device lost).
    pub fn maxpool3d(
        self,
        kernel_size: (usize, usize, usize),
        stride: (usize, usize, usize),
        padding: (usize, usize, usize),
    ) -> Result<Self> {
        MaxPool3D::new(self, kernel_size, stride, padding)?.execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_maxpool3d_basic() {
        let device = crate::device::test_pool::get_test_device().await;
        let input_data = vec![1.0; 2 * 4 * 4 * 4];
        let input = Tensor::from_vec_on(input_data, vec![1, 2, 4, 4, 4], device.clone())
            .await
            .unwrap();

        let output = input.maxpool3d((2, 2, 2), (2, 2, 2), (0, 0, 0)).unwrap();
        let result = output.to_vec().unwrap();

        assert_eq!(output.shape(), &[1, 2, 2, 2, 2]);
        assert_eq!(result.len(), 16);
        assert!(result.iter().all(|&x| x.is_finite()));
    }

    #[tokio::test]
    async fn test_maxpool3d_validation() {
        let device = crate::device::test_pool::get_test_device().await;
        // Invalid shape (not 5D)
        let input = Tensor::from_vec_on(vec![1.0; 16], vec![4, 4], device.clone())
            .await
            .unwrap();
        assert!(input.maxpool3d((2, 2, 2), (2, 2, 2), (0, 0, 0)).is_err());
    }
}

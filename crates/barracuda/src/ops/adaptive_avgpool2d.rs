// SPDX-License-Identifier: AGPL-3.0-or-later
//! Adaptive Average Pooling 2D - Output-size-driven pooling
//!
//! **Deep Debt Evolution**: Modernized from trait-based to direct `impl Tensor`
//!
//! ## Deep Debt Principles
//!
//! - ✅ Modern idiomatic Rust (direct `impl Tensor`, not trait extension)
//! - ✅ Universal compute (WGSL shader for all substrates)
//! - ✅ Safe Rust (no unsafe blocks)
//! - ✅ Flexible (specify output size, not kernel size)
//!
//! ## Evolution History
//!
//! **Before** (Phase 3): `AdaptiveAvgPool2DExt` trait extension  
//! **After** (Phase 6): Direct `impl Tensor` method
//!
//! ## Usage
//!
//! ```ignore
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! # use barracuda::tensor::Tensor;
//! # use barracuda::device::test_pool;
//! # let device = test_pool::tokio_block_on(test_pool::get_test_device_if_gpu_available()).unwrap();
//! # let input = Tensor::from_data(&[1.0f32; 196], vec![1, 1, 14, 14], device).unwrap();
//! // Input: [batch, channels, 14, 14]
//! // Output: [batch, channels, 7, 7] (adaptive to target size)
//! let _pooled = input.adaptive_avgpool2d((7, 7))?;
//! # Ok(())
//! # }
//! ```

use crate::device::{ComputeDispatch, DeviceCapabilities};
use crate::error::Result;
use crate::tensor::Tensor;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct AdaptiveAvgPool2DParams {
    batch: u32,
    channels: u32,
    in_height: u32,
    in_width: u32,
    out_height: u32,
    out_width: u32,
}

/// Adaptive average pooling 2D with output-size-driven kernel.
pub struct AdaptiveAvgPool2D {
    input: Tensor,
    output_size: (usize, usize),
}

impl AdaptiveAvgPool2D {
    fn wgsl_shader() -> &'static str {
        {
            static SHADER: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
                crate::shaders::precision::downcast_f64_to_f32(include_str!(
                    "../shaders/pooling/adaptive_avgpool2d_f64.wgsl"
                ))
            });
            std::sync::LazyLock::force(&SHADER).as_str()
        }
    }

    /// Execute adaptive average pooling and return the result tensor.
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let shape = self.input.shape();

        if shape.len() != 4 {
            return Err(crate::error::BarracudaError::invalid_op("Shape Error", 
                format!("AdaptiveAvgPool2D expects 4D input [batch, channels, height, width], got shape {shape:?}")
            ));
        }

        let batch = shape[0];
        let channels = shape[1];
        let in_height = shape[2];
        let in_width = shape[3];
        let (out_height, out_width) = self.output_size;

        let params = AdaptiveAvgPool2DParams {
            batch: batch as u32,
            channels: channels as u32,
            in_height: in_height as u32,
            in_width: in_width as u32,
            out_height: out_height as u32,
            out_width: out_width as u32,
        };

        let output_shape = &[batch, channels, out_height, out_width];
        let output_size = output_shape.iter().product::<usize>() * std::mem::size_of::<f32>();

        let output_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("adaptive_avgpool2d_output"),
            size: output_size as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("adaptive_avgpool2d_params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let caps = DeviceCapabilities::from_device(device);
        let (workgroups_x, workgroups_y) = caps.dispatch_2d(out_width as u32, out_height as u32);
        let workgroups_z = (batch * channels) as u32;

        ComputeDispatch::new(device, "adaptive_avgpool2d")
            .shader(Self::wgsl_shader(), "main")
            .storage_read(0, self.input.buffer())
            .storage_rw(1, &output_buffer)
            .uniform(2, &params_buffer)
            .dispatch(workgroups_x, workgroups_y, workgroups_z)
            .submit();

        let output_elem_count = output_shape.iter().product::<usize>();
        let output_data = crate::utils::read_buffer(device, &output_buffer, output_elem_count)?;
        Ok(Tensor::new(
            output_data,
            output_shape.to_vec(),
            device.clone(),
        ))
    }
}

// ============================================================================
// Modern API: Direct impl Tensor (Phase 6 Evolution)
// ============================================================================

impl Tensor {
    /// Apply adaptive average pooling with target output size
    ///
    /// Automatically calculates kernel/stride to achieve desired output dimensions
    ///
    /// **Deep Debt**: Modern direct method, no trait extension needed
    ///
    /// ## Arguments
    ///
    /// * `output_size` - Target (height, width) for output
    ///
    /// ## Input/Output Shapes
    ///
    /// - Input: `[batch, channels, height_in, width_in]`
    /// - Output: `[batch, channels, height_out, width_out]`
    ///
    /// ## Example
    ///
    /// ```ignore
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # use barracuda::tensor::Tensor;
    /// # use barracuda::device::test_pool;
    /// # let device = test_pool::tokio_block_on(test_pool::get_test_device_if_gpu_available()).unwrap();
    /// # let input = Tensor::from_data(&[1.0f32; 49], vec![1, 1, 7, 7], device).unwrap();
    /// // Adaptive pool to 7x7 (regardless of input size)
    /// let _pooled = input.adaptive_avgpool2d((7, 7))?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn adaptive_avgpool2d(self, output_size: (usize, usize)) -> Result<Self> {
        let op = AdaptiveAvgPool2D {
            input: self,
            output_size,
        };
        op.execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_adaptive_avgpool2d() {
        let Some(device) = crate::device::test_pool::get_test_device_if_gpu_available().await
        else {
            return;
        };

        // Test case: 4x4 input -> 2x2 output
        let input = Tensor::from_data(
            &vec![
                // Batch 0, Channel 0
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0,
            ],
            vec![1, 1, 4, 4],
            device.clone(),
        )
        .unwrap();

        let result = input.adaptive_avgpool2d((2, 2)).unwrap();
        let output = result.to_vec().unwrap();

        // Each 2x2 region should be averaged
        assert_eq!(result.shape(), &[1, 1, 2, 2]);
        assert_eq!(output.len(), 4);

        // Top-left: avg(1,2,5,6) = 3.5
        assert!((output[0] - 3.5).abs() < 1e-5);
        // Top-right: avg(3,4,7,8) = 5.5
        assert!((output[1] - 5.5).abs() < 1e-5);
        // Bottom-left: avg(9,10,13,14) = 11.5
        assert!((output[2] - 11.5).abs() < 1e-5);
        // Bottom-right: avg(11,12,15,16) = 13.5
        assert!((output[3] - 13.5).abs() < 1e-5);
    }

    #[tokio::test]
    async fn test_adaptive_avgpool2d_1x1_output() {
        let Some(device) = crate::device::test_pool::get_test_device_if_gpu_available().await
        else {
            return;
        };

        // Test global average pooling (adaptive pool to 1x1)
        let input =
            Tensor::from_data(&[1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2], device.clone()).unwrap();

        let result = input.adaptive_avgpool2d((1, 1)).unwrap();
        let output = result.to_vec().unwrap();

        assert_eq!(result.shape(), &[1, 1, 1, 1]);
        assert_eq!(output.len(), 1);
        assert!((output[0] - 2.5).abs() < 1e-5); // avg(1,2,3,4) = 2.5
    }
}

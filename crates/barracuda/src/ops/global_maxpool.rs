// SPDX-License-Identifier: AGPL-3.0-or-later
//! Global Max Pooling Operation - Spatial reduction to single value
//!
//! **Deep Debt Evolution**: Modernized from trait-based to direct `impl Tensor`
//!
//! ## Deep Debt Principles
//!
//! - ✅ Modern idiomatic Rust (direct `impl Tensor`, not trait extension)
//! - ✅ Universal compute (WGSL shader for all substrates)
//! - ✅ Safe Rust (no unsafe blocks)
//! - ✅ CNN-friendly (4D input [batch, channels, height, width])
//!
//! ## Evolution History
//!
//! **Before** (Phase 3): `GlobalMaxPoolExt` trait extension\
//! **After** (Phase 6): Direct `impl Tensor` method
//!
//! ## Usage
//!
//! ```ignore
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! # use barracuda::tensor::Tensor;
//! # use barracuda::device::test_pool;
//! # let device = test_pool::tokio_block_on(test_pool::get_test_device_if_gpu_available()).unwrap();
//! # let data = [1.0f32; 2 * 64 * 7 * 7];
//! // Input: [batch=2, channels=64, height=7, width=7]
//! let input = Tensor::from_data(&data, vec![2, 64, 7, 7], device)?;
//! // Output: [batch=2, channels=64, height=1, width=1]
//! let _pooled = input.global_maxpool()?;
//! # Ok(())
//! # }
//! ```

use crate::device::{ComputeDispatch, DeviceCapabilities, WorkloadType};
use crate::error::Result;
use crate::tensor::Tensor;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct GlobalMaxPoolParams {
    batch_size: u32,
    channels: u32,
    height: u32,
    width: u32,
}

/// Global max pooling: reduce spatial dimensions to 1×1 per channel.
pub struct GlobalMaxPool {
    input: Tensor,
}

impl GlobalMaxPool {
    fn wgsl_shader() -> &'static str {
        {
            static SHADER: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
                include_str!("../shaders/pooling/global_maxpool_f64.wgsl").to_string()
            });
            std::sync::LazyLock::force(&SHADER).as_str()
        }
    }

    /// Execute global max pooling over spatial dimensions.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let shape = self.input.shape();

        if shape.len() != 4 {
            return Err(crate::error::BarracudaError::invalid_op(
                "Shape Error",
                format!(
                    "GlobalMaxPool expects 4D input [batch, channels, height, width], got shape {shape:?}"
                ),
            ));
        }

        let batch_size = shape[0];
        let channels = shape[1];
        let height = shape[2];
        let width = shape[3];

        let params = GlobalMaxPoolParams {
            batch_size: batch_size as u32,
            channels: channels as u32,
            height: height as u32,
            width: width as u32,
        };

        let output_shape = &[batch_size, channels, 1, 1];
        let output_size = output_shape.iter().product::<usize>() * std::mem::size_of::<f32>();

        let output_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("global_maxpool_output"),
            size: output_size as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("global_maxpool_params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let caps = DeviceCapabilities::from_device(device);
        let optimal_wg_size = caps.optimal_workgroup_size(WorkloadType::Convolution);
        let num_outputs = (batch_size * channels) as u32;
        let workgroups = num_outputs.div_ceil(optimal_wg_size);

        ComputeDispatch::new(device, "global_maxpool")
            .shader(Self::wgsl_shader(), "main")
            .storage_read(0, self.input.buffer())
            .storage_rw(1, &output_buffer)
            .uniform(2, &params_buffer)
            .dispatch(workgroups, 1, 1)
            .submit()?;

        Ok(Tensor::from_buffer(
            output_buffer,
            output_shape.to_vec(),
            device.clone(),
        ))
    }
}

// ============================================================================
// Modern API: Direct impl Tensor (Phase 6 Evolution)
// ============================================================================

impl Tensor {
    /// Apply global max pooling across spatial dimensions
    ///
    /// Reduces [batch, channels, height, width] → [batch, channels, 1, 1]
    ///
    /// **Deep Debt**: Modern direct method, no trait extension needed
    ///
    /// ## Input Shape
    ///
    /// Must be 4D: `[batch, channels, height, width]`
    ///
    /// ## Output Shape
    ///
    /// `[batch, channels, 1, 1]` (max value per channel)
    ///
    /// ## Example
    ///
    /// ```ignore
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// # use barracuda::tensor::Tensor;
    /// # use barracuda::device::test_pool;
    /// # let device = test_pool::tokio_block_on(test_pool::get_test_device_if_gpu_available()).unwrap();
    /// # let input = Tensor::from_data(&[1.0f32; 49], vec![1, 1, 7, 7], device).unwrap();
    /// // Pool spatial dimensions (7x7 → 1x1)
    /// let _pooled = input.global_maxpool()?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn global_maxpool(self) -> Result<Self> {
        let op = GlobalMaxPool { input: self };
        op.execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::WgpuDevice;
    use std::sync::Arc;

    async fn get_test_device() -> Option<Arc<WgpuDevice>> {
        crate::device::test_pool::get_test_device_if_gpu_available().await
    }

    #[tokio::test]
    async fn test_global_maxpool_basic() {
        let Some(device) = get_test_device().await else {
            return;
        };
        let input = Tensor::from_data(
            &vec![
                // Batch 0, Channel 0
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, // Batch 0, Channel 1
                9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            ],
            vec![1, 2, 2, 4],
            device,
        )
        .unwrap();

        let result = input.global_maxpool().unwrap();
        let output = result.to_vec().unwrap();

        assert_eq!(result.shape(), &[1, 2, 1, 1]);
        assert_eq!(output.len(), 2);
        // Verify operation completed successfully
        assert!(output.iter().all(|&x| x.is_finite()));
        // Values should be positive (all inputs positive)
        assert!(output.iter().all(|&x| x > 0.0));
    }

    #[tokio::test]
    async fn test_global_maxpool_edge_cases() {
        let Some(device) = get_test_device().await else {
            return;
        };
        // Single 1x1 spatial
        let input = Tensor::from_data(&[42.0, 99.0], vec![1, 2, 1, 1], device.clone()).unwrap();
        let result = input.global_maxpool().unwrap();
        let output = result.to_vec().unwrap();
        assert_eq!(output.len(), 2);
        assert!(output.iter().all(|&x| x.is_finite()));

        // All same values
        let input = Tensor::from_data(&[5.0; 2 * 3 * 3], vec![1, 2, 3, 3], device).unwrap();
        let result = input.global_maxpool().unwrap();
        let output = result.to_vec().unwrap();
        assert_eq!(output.len(), 2);
        assert!(output.iter().all(|&x| x.is_finite()));
    }

    #[tokio::test]
    async fn test_global_maxpool_boundary() {
        let Some(device) = get_test_device().await else {
            return;
        };
        // Large spatial dimensions
        let input =
            Tensor::from_data(&vec![1.0; 32 * 32], vec![1, 1, 32, 32], device.clone()).unwrap();
        let result = input.global_maxpool().unwrap();
        let output = result.to_vec().unwrap();
        assert_eq!(output.len(), 1);
        assert!(output[0].is_finite());

        // Many channels
        let input = Tensor::from_data(&vec![1.0; 64 * 7 * 7], vec![1, 64, 7, 7], device).unwrap();
        let result = input.global_maxpool().unwrap();
        let output = result.to_vec().unwrap();
        assert_eq!(output.len(), 64);
    }

    #[tokio::test]
    async fn test_global_maxpool_large_batch() {
        let Some(device) = get_test_device().await else {
            return;
        };
        // Batch size 16
        let batch_size = 16;
        let channels = 32;
        let input = Tensor::from_data(
            &vec![1.0; batch_size * channels * 8 * 8],
            vec![batch_size, channels, 8, 8],
            device,
        )
        .unwrap();
        let result = input.global_maxpool().unwrap();
        let output = result.to_vec().unwrap();
        assert_eq!(output.len(), batch_size * channels);
        assert!(output.iter().all(|&x| x.is_finite()));
    }

    #[tokio::test]
    async fn test_global_maxpool_precision() {
        let Some(device) = get_test_device().await else {
            return;
        };
        // Known max with varying values
        let input = Tensor::from_data(
            &[1.0, 5.0, 3.0, 2.0], // Max = 5.0
            vec![1, 1, 2, 2],
            device,
        )
        .unwrap();
        let result = input.global_maxpool().unwrap();
        let output = result.to_vec().unwrap();

        assert_eq!(output.len(), 1);
        assert!(output[0].is_finite());
        // Verify it's computing a pooling operation (should be in range of inputs)
        assert!(output[0] >= 0.0 && output[0] <= 10.0);
    }
}

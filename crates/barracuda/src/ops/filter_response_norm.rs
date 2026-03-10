// SPDX-License-Identifier: AGPL-3.0-only
//! Filter Response Normalization (FRN) - Normalization without batch dependency
//!
//! **Canonical `BarraCuda` Pattern**: Struct with new/execute
//!
//! Normalizes activations per filter, not per batch.
//! Enables single-sample inference.

use crate::device::DeviceCapabilities;
use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

/// f64 is the canonical source — math is universal, precision is silicon.
const SHADER_F64: &str = include_str!("../shaders/norm/filter_response_norm_f64.wgsl");
static SHADER_F32: std::sync::LazyLock<String> =
    std::sync::LazyLock::new(|| SHADER_F64.to_string());

/// Filter Response Normalization operation
pub struct FilterResponseNorm {
    input: Tensor,
    gamma: Tensor,
    beta: Tensor,
    batch_size: usize,
    channels: usize,
    height: usize,
    width: usize,
    epsilon: f32,
}

impl FilterResponseNorm {
    /// Create a new filter response normalization operation
    /// # Errors
    /// Returns [`Err`] if input size does not match batch×channels×height×width,
    /// or gamma/beta shapes are not [channels].
    pub fn new(
        input: Tensor,
        gamma: Tensor,
        beta: Tensor,
        batch_size: usize,
        channels: usize,
        height: usize,
        width: usize,
        epsilon: f32,
    ) -> Result<Self> {
        // Validate input shape
        let input_shape = input.shape();
        let expected_size = batch_size * channels * height * width;
        if input_shape.iter().product::<usize>() != expected_size {
            return Err(BarracudaError::InvalidShape {
                expected: vec![batch_size, channels, height, width],
                actual: input_shape.to_vec(),
            });
        }

        // Validate gamma and beta shapes
        if gamma.shape() != [channels] {
            return Err(BarracudaError::InvalidShape {
                expected: vec![channels],
                actual: gamma.shape().to_vec(),
            });
        }

        if beta.shape() != [channels] {
            return Err(BarracudaError::InvalidShape {
                expected: vec![channels],
                actual: beta.shape().to_vec(),
            });
        }

        Ok(Self {
            input,
            gamma,
            beta,
            batch_size,
            channels,
            height,
            width,
            epsilon,
        })
    }

    /// Get the WGSL shader source
    fn wgsl_shader() -> &'static str {
        &SHADER_F32
    }

    /// Execute the filter response normalization operation
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let spatial_size = self.height * self.width;
        let total_elements = self.batch_size * self.channels * spatial_size;

        // Create reduction buffer for sum of squares
        let sum_sq_buffer = device.create_buffer_f32(self.batch_size * self.channels)?;

        // Create output buffer
        let output_buffer = device.create_buffer_f32(total_elements)?;

        // Create uniform buffer for parameters
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            batch_size: u32,
            channels: u32,
            height: u32,
            width: u32,
            spatial_size: u32,
            epsilon: f32,
            _pad1: u32,
        }

        let params = Params {
            batch_size: self.batch_size as u32,
            channels: self.channels as u32,
            height: self.height as u32,
            width: self.width as u32,
            spatial_size: spatial_size as u32,
            epsilon: self.epsilon,
            _pad1: 0,
        };
        let params_buffer = device.create_uniform_buffer("FRN Params", &params);

        let pass1_workgroups = self.batch_size * self.channels;
        let caps = DeviceCapabilities::from_device(device);
        let workgroups_pass2 = caps.dispatch_1d(total_elements as u32);

        ComputeDispatch::new(device, "frn_sum_sq")
            .shader(Self::wgsl_shader(), "compute_sum_sq")
            .storage_read(0, self.input.buffer())
            .storage_read(1, self.gamma.buffer())
            .storage_read(2, self.beta.buffer())
            .storage_rw(3, &sum_sq_buffer)
            .storage_rw(4, &output_buffer)
            .uniform(5, &params_buffer)
            .dispatch(pass1_workgroups as u32, 1, 1)
            .submit()?;

        ComputeDispatch::new(device, "frn_normalize")
            .shader(Self::wgsl_shader(), "normalize_and_scale")
            .storage_read(0, self.input.buffer())
            .storage_read(1, self.gamma.buffer())
            .storage_read(2, self.beta.buffer())
            .storage_rw(3, &sum_sq_buffer)
            .storage_rw(4, &output_buffer)
            .uniform(5, &params_buffer)
            .dispatch(workgroups_pass2, 1, 1)
            .submit()?;

        let output_shape = self.input.shape().to_vec();
        let output_data = crate::utils::read_buffer(device, &output_buffer, total_elements)?;
        Ok(Tensor::new(output_data, output_shape, device.clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device_if_gpu_available;

    #[tokio::test]
    async fn test_filter_response_norm_basic() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let input = Tensor::from_vec_on(vec![1.0; 3 * 4 * 4], vec![1, 3, 4, 4], device.clone())
            .await
            .unwrap();
        let gamma = Tensor::from_vec_on(vec![1.0; 3], vec![3], device.clone())
            .await
            .unwrap();
        let beta = Tensor::from_vec_on(vec![0.0; 3], vec![3], device.clone())
            .await
            .unwrap();
        let output = FilterResponseNorm::new(input, gamma, beta, 1, 3, 4, 4, 1e-5)
            .unwrap()
            .execute()
            .unwrap();
        assert_eq!(output.shape(), &[1, 3, 4, 4]);
        let result = output.to_vec().unwrap();
        assert_eq!(result.len(), 48);
        assert!(result.iter().all(|&x| x.is_finite()));
    }
}

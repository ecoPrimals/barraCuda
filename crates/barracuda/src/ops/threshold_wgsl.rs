// SPDX-License-Identifier: AGPL-3.0-or-later
//! Threshold - Threshold activation function - Pure WGSL
//!
//! f64 canonical — f32 derived via `downcast_f64_to_f32` when needed.
//!
//! Deep Debt Principles:
//! - Self-knowledge: Operation knows its threshold and value
//! - Zero hardcoding: All parameters passed at runtime
//! - Modern idiomatic Rust: Safe, zero unsafe code
//! - Complete implementation: Production-ready, no mocks
//! - Hardware-agnostic: Pure WGSL for universal compute

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::Result;
use crate::tensor::Tensor;

/// Threshold operation
pub struct Threshold {
    input: Tensor,
    threshold: f32,
    value: f32,
}

impl Threshold {
    /// Create a new threshold operation
    #[must_use]
    pub fn new(input: Tensor, threshold: f32, value: f32) -> Self {
        Self {
            input,
            threshold,
            value,
        }
    }

    /// Execute the threshold operation
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let size: usize = self.input.shape().iter().product();

        // Create buffers
        // Access input buffer directly (zero-copy)
        let input_buffer = self.input.buffer();

        let output_buffer = device.create_buffer_f32(size)?;

        // Create uniform buffer for parameters
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            size: u32,
            threshold: f32,
            value: f32,
        }

        let params = Params {
            size: size as u32,
            threshold: self.threshold,
            value: self.value,
        };

        let params_buffer = device.create_uniform_buffer("Threshold Params", &params);

        ComputeDispatch::new(device, "Threshold")
            .shader(
                include_str!("../shaders/activation/threshold_f64.wgsl"),
                "main",
            )
            .storage_read(0, input_buffer)
            .storage_rw(1, &output_buffer)
            .uniform(2, &params_buffer)
            .dispatch_1d(size as u32)
            .submit()?;

        // Read back results
        let output_data = crate::utils::read_buffer(device, &output_buffer, size)?;

        Ok(Tensor::new(
            output_data,
            self.input.shape().to_vec(),
            device.clone(),
        ))
    }
}

impl Tensor {
    /// Apply threshold activation
    /// # Arguments
    /// * `threshold` - Threshold value
    /// * `value` - Replacement value for elements <= threshold
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn threshold_wgsl(self, threshold: f32, value: f32) -> Result<Self> {
        Threshold::new(self, threshold, value).execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_threshold() {
        let device = crate::device::test_pool::get_test_device().await;
        let data = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let input = Tensor::new(data, vec![5], device);

        let output = input.threshold_wgsl(0.0, -10.0).unwrap();

        let result = output.to_vec().unwrap();
        assert_eq!(result[0], -10.0); // -2 <= 0, replaced
        assert_eq!(result[1], -10.0); // -1 <= 0, replaced
        assert_eq!(result[2], -10.0); // 0 <= 0, replaced
        assert_eq!(result[3], 1.0); // 1 > 0, unchanged
        assert_eq!(result[4], 2.0); // 2 > 0, unchanged
    }

    #[tokio::test]
    async fn test_threshold_custom() {
        let device = crate::device::test_pool::get_test_device().await;
        let data = vec![0.5, 1.0, 1.5, 2.0];
        let input = Tensor::new(data, vec![4], device);

        let output = input.threshold_wgsl(1.0, 0.0).unwrap();

        let result = output.to_vec().unwrap();
        assert_eq!(result[0], 0.0); // 0.5 <= 1.0
        assert_eq!(result[1], 0.0); // 1.0 <= 1.0
        assert_eq!(result[2], 1.5); // 1.5 > 1.0
        assert_eq!(result[3], 2.0); // 2.0 > 1.0
    }
}

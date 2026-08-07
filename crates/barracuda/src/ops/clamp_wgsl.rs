// SPDX-License-Identifier: AGPL-3.0-or-later
//! Clamp - Clamp values between min and max - Pure WGSL
//!
//! Deep Debt Principles:
//! - Self-knowledge: Operation knows its min/max bounds
//! - Zero hardcoding: All parameters passed at runtime
//! - Modern idiomatic Rust: Safe, zero unsafe code
//! - Complete implementation: Production-ready, no mocks
//! - Hardware-agnostic: Pure WGSL for universal compute

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::Result;
use crate::tensor::Tensor;

/// Simple clamp variant (f64 canonical).
const WGSL_CLAMP_SIMPLE_F64: &str = include_str!("../shaders/math/clamp_simple_f64.wgsl");

/// Simple clamp variant (f32 derived from f64).
pub const WGSL_CLAMP_SIMPLE: &str = WGSL_CLAMP_SIMPLE_F64;

/// Clamp operation - Clamp values between min and max
pub struct Clamp {
    input: Tensor,
    min_val: f32,
    max_val: f32,
}

impl Clamp {
    /// Create a new clamp operation
    #[must_use]
    pub fn new(input: Tensor, min_val: f32, max_val: f32) -> Self {
        Self {
            input,
            min_val,
            max_val,
        }
    }

    /// Execute the clamp operation
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let size: usize = self.input.shape().iter().product();

        // Access input buffer directly (zero-copy)
        let input_buffer = self.input.buffer();

        // Create output buffer
        let output_buffer = device.create_buffer_f32(size)?;

        // Create uniform buffer for parameters
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            size: u32,
            min_val: f32,
            max_val: f32,
        }

        let params = Params {
            size: size as u32,
            min_val: self.min_val,
            max_val: self.max_val,
        };

        let params_buffer = device.create_uniform_buffer("Clamp Params", &params);

        ComputeDispatch::new(device, "Clamp")
            .shader(include_str!("../shaders/math/clamp_f64.wgsl"), "main")
            .storage_read(0, &input_buffer)
            .storage_rw(1, &output_buffer)
            .uniform(2, &params_buffer)
            .dispatch_1d(size as u32)
            .submit()?;

        // Return tensor without reading back (zero-copy)
        Ok(Tensor::from_buffer(
            output_buffer,
            self.input.shape().to_vec(),
            device.clone(),
        ))
    }
}

impl Tensor {
    /// Clamp values between min and max
    ///
    /// # Arguments
    ///
    /// * `min_val` - Minimum value
    /// * `max_val` - Maximum value
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn clamp_wgsl(self, min_val: f32, max_val: f32) -> Result<Self> {
        Clamp::new(self, min_val, max_val).execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_clamp() {
        let device = crate::device::test_pool::get_test_device().await;
        let data = vec![-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0];
        let input = Tensor::new(data, vec![7], device);

        let output = input.clamp_wgsl(0.0, 2.0).unwrap();

        let result = output.to_vec().unwrap();
        assert_eq!(result[0], 0.0);
        assert_eq!(result[1], 0.0);
        assert_eq!(result[2], 0.0);
        assert_eq!(result[3], 1.0);
        assert_eq!(result[4], 2.0);
        assert_eq!(result[5], 2.0);
        assert_eq!(result[6], 2.0);
    }

    #[tokio::test]
    async fn test_clamp_no_effect() {
        let device = crate::device::test_pool::get_test_device().await;
        let data = vec![0.5, 1.0, 1.5];
        let input = Tensor::new(data, vec![3], device);

        let output = input.clamp_wgsl(0.0, 2.0).unwrap();

        let result = output.to_vec().unwrap();
        assert_eq!(result[0], 0.5);
        assert_eq!(result[1], 1.0);
        assert_eq!(result[2], 1.5);
    }
}

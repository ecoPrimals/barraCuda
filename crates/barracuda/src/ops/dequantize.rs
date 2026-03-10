// SPDX-License-Identifier: AGPL-3.0-only
//! Dequantize operation - Convert quantized integers to floating point
//!
//! Dequantization: Convert low-precision integers back to FP32
//! Used for inference with quantized models

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;
use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct DequantizeParams {
    size: u32,
    scale: f32,
    zero_point: f32,
    _padding: u32,
}

/// Dequantize operation
pub struct Dequantize {
    input: Tensor,
    scale: f32,
    zero_point: f32,
}

impl Dequantize {
    /// Create dequantize operation
    /// Note: input tensor should contain i32 values (quantized integers)
    /// This will be converted to f32 output
    /// # Errors
    /// Returns [`Err`] if scale is not positive.
    pub fn new(input: Tensor, scale: f32, zero_point: f32) -> Result<Self> {
        if scale <= 0.0 {
            return Err(BarracudaError::invalid_op(
                "dequantize",
                format!("scale must be positive, got {scale}"),
            ));
        }

        Ok(Self {
            input,
            scale,
            zero_point,
        })
    }

    /// WGSL shader source (embedded at compile time)
    fn wgsl_shader() -> &'static str {
        static SHADER: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
            include_str!("../shaders/misc/dequantize_f64.wgsl").to_string()
        });
        &SHADER
    }

    /// Execute dequantize on tensor
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let size = self.input.len();

        // Create output buffer (f32)
        let output_buffer = device.create_buffer_f32(size)?;

        // Use tensor buffer directly (zero-copy)
        // The shader will handle f32->i32 conversion
        let input_buffer = self.input.buffer();

        // Create params
        let params = DequantizeParams {
            size: size as u32,
            scale: self.scale,
            zero_point: self.zero_point,
            _padding: 0,
        };

        let params_buffer = device.create_uniform_buffer("Dequantize Params", &params);

        ComputeDispatch::new(device, "Dequantize")
            .shader(Self::wgsl_shader(), "main")
            .storage_read(0, input_buffer)
            .storage_rw(1, &output_buffer)
            .uniform(2, &params_buffer)
            .dispatch_1d(size as u32)
            .submit()?;

        // Create output tensor
        Ok(Tensor::from_buffer(
            output_buffer,
            self.input.shape().to_vec(),
            device.clone(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device_if_gpu_available;

    #[tokio::test]
    async fn test_dequantize_basic() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        // Simulate quantized values (as f32, will be cast to i32)
        let input = Tensor::from_vec_on(vec![100.0, 150.0, 200.0, 250.0], vec![4], device)
            .await
            .unwrap();

        let output = Dequantize::new(input, 0.1, 128.0)
            .unwrap()
            .execute()
            .unwrap();
        let result = output.to_vec().unwrap();

        assert_eq!(result.len(), 4);
        // Dequantized: (quantized - zero_point) * scale
        // (100 - 128) * 0.1 = -2.8
        assert!((result[0] - (-2.8)).abs() < 0.1);
    }
}

// SPDX-License-Identifier: AGPL-3.0-or-later
//! Fake quantization operation - Simulate quantization for training
//!
//! Fake quantization simulates the effect of quantization during training
//! by quantizing values to N bits but keeping them in floating point format.

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;
use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct FakeQuantizeParams {
    size: u32,
    num_bits: u32,
    scale: f32,
    zero_point: f32,
}

/// Fake quantization operation
pub struct FakeQuantize {
    input: Tensor,
    num_bits: u32,
    scale: f32,
    zero_point: f32,
}

impl FakeQuantize {
    /// Create fake quantization operation
    /// # Errors
    /// Returns [`Err`] if `num_bits` is 0 or > 32, or scale is not positive.
    pub fn new(input: Tensor, num_bits: u32, scale: f32, zero_point: f32) -> Result<Self> {
        if num_bits == 0 || num_bits > 32 {
            return Err(BarracudaError::invalid_op(
                "fake_quantize",
                format!("num_bits must be between 1 and 32, got {num_bits}"),
            ));
        }
        if scale <= 0.0 {
            return Err(BarracudaError::invalid_op(
                "fake_quantize",
                format!("scale must be positive, got {scale}"),
            ));
        }
        Ok(Self {
            input,
            num_bits,
            scale,
            zero_point,
        })
    }

    /// Execute fake quantization on tensor
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let size = self.input.len();

        // Create output buffer
        let output_buffer = device.create_buffer_f32(size)?;

        // Create params
        let params = FakeQuantizeParams {
            size: size as u32,
            num_bits: self.num_bits,
            scale: self.scale,
            zero_point: self.zero_point,
        };

        let params_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FakeQuantize Params"),
            size: std::mem::size_of::<FakeQuantizeParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        device
            .queue
            .write_buffer(&params_buffer, 0, bytemuck::bytes_of(&params));

        ComputeDispatch::new(device, "FakeQuantize")
            .shader(
                include_str!("../shaders/misc/fake_quantize_f64.wgsl"),
                "main",
            )
            .storage_read(0, self.input.buffer())
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

    #[tokio::test]
    async fn test_fake_quantize_basic() {
        let device = crate::device::test_pool::get_test_device().await;
        let input = Tensor::from_vec_on(vec![-2.0, -1.0, 0.0, 1.0, 2.0], vec![5], device)
            .await
            .unwrap();

        let output = FakeQuantize::new(input, 8, 1.0, 0.0)
            .unwrap()
            .execute()
            .unwrap();
        let result = output.to_vec().unwrap();

        // Should quantize and dequantize values
        assert_eq!(result.len(), 5);
    }
}

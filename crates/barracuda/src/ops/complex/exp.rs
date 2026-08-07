// SPDX-License-Identifier: AGPL-3.0-or-later
//! Complex Exponential Operation
//!
//! **Operation**: exp(a + bi) = exp(a)[cos(b) + i·sin(b)] (Euler's formula)
//! **Complexity**: O(1) - 1 exp + 2 trig functions
//! **CRITICAL**: This is THE operation for FFT twiddle factors `W_N^k` = exp(-2πik/N)

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

/// Complex exponential: exp(z) = exp(re)·[cos(im) + i·sin(im)] (Euler's formula).
pub struct ComplexExp {
    input: Tensor,
}

impl ComplexExp {
    /// Create complex exp operation. Input must have last dim = 2 (re, im).
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn new(input: Tensor) -> Result<Self> {
        let shape = input.shape();
        if shape.last() != Some(&2) {
            return Err(BarracudaError::Device(
                "Complex tensor must have last dimension = 2".to_string(),
            ));
        }

        Ok(Self { input })
    }

    /// Execute complex exponential on GPU.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let num_elements = self.input.len();
        let num_complex = num_elements / 2;

        let output_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Complex Exp Output"),
            size: (num_elements * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params = [num_complex as u32];
        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Complex Exp Params"),
                contents: bytemuck::cast_slice(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        ComputeDispatch::new(device, "ComplexExp")
            .shader(include_str!("exp.wgsl"), "main")
            .storage_read(0, self.input.buffer())
            .storage_rw(1, &output_buffer)
            .uniform(2, &params_buffer)
            .dispatch_1d(num_complex as u32)
            .submit()?;

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
    async fn test_complex_exp_euler() {
        let device = crate::device::test_pool::get_test_device().await;

        // exp(iπ) = cos(π) + i·sin(π) = -1 + 0i
        // (Euler's identity: exp(iπ) + 1 = 0)
        let pi = std::f32::consts::PI;
        let data = vec![0.0f32, pi]; // 0 + πi
        let tensor = Tensor::from_data(&data, vec![1, 2], device).unwrap();

        let op = ComplexExp::new(tensor).unwrap();
        let result = op.execute().unwrap();

        let result_data = result.to_vec().unwrap();
        assert!((result_data[0] - (-1.0)).abs() < 1e-5); // Real ≈ -1
        assert!((result_data[1] - 0.0).abs() < 1e-5); // Imag ≈ 0
    }

    #[tokio::test]
    async fn test_complex_exp_zero() {
        let device = crate::device::test_pool::get_test_device().await;
        // exp(0) = 1+0i
        let data = vec![0.0f32, 0.0];
        let tensor = Tensor::from_data(&data, vec![1, 2], device).unwrap();
        let result = ComplexExp::new(tensor)
            .unwrap()
            .execute()
            .unwrap()
            .to_vec()
            .unwrap();
        assert!((result[0] - 1.0).abs() < 1e-6);
        assert!((result[1] - 0.0).abs() < 1e-6);
    }
}

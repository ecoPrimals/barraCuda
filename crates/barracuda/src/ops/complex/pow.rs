// SPDX-License-Identifier: AGPL-3.0-or-later
//! Complex Power z^n

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

/// Complex power z^n via polar form.
pub struct ComplexPow {
    input: Tensor,
    exponent: f32,
}

impl ComplexPow {
    /// Create complex power operation. Input must have last dim = 2 (re, im).
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn new(input: Tensor, exponent: f32) -> Result<Self> {
        if input.shape().last() != Some(&2) {
            return Err(BarracudaError::Device(
                "Must have last dimension = 2".to_string(),
            ));
        }
        Ok(Self { input, exponent })
    }

    /// Execute complex power on GPU.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let n = self.input.len();
        let num_complex = n / 2;

        let output_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Out"),
            size: (n * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            num_complex: u32,
            exponent: f32,
        }
        let params = Params {
            num_complex: num_complex as u32,
            exponent: self.exponent,
        };
        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("P"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        ComputeDispatch::new(device, "ComplexPow")
            .shader(include_str!("pow.wgsl"), "main")
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
    async fn test_complex_pow_square() {
        let device = crate::device::test_pool::get_test_device().await;

        // Test (2+0i)^2 = 4+0i
        let data = vec![2.0f32, 0.0];
        let tensor = Tensor::from_data(&data, vec![1, 2], device).unwrap();

        let pow_op = ComplexPow::new(tensor, 2.0).unwrap();
        let result = pow_op.execute().unwrap();
        let result_data = result.to_vec().unwrap();

        assert!(
            (result_data[0] - 4.0).abs() < 1e-4,
            "2^2 real part should be 4"
        );
        assert!(
            (result_data[1] - 0.0).abs() < 1e-4,
            "2^2 imag part should be 0"
        );
    }

    #[tokio::test]
    async fn test_complex_pow_identity() {
        let device = crate::device::test_pool::get_test_device().await;

        // Test z^1 = z
        let data = vec![3.0f32, 4.0];
        let tensor = Tensor::from_data(&data, vec![1, 2], device).unwrap();

        let pow_op = ComplexPow::new(tensor, 1.0).unwrap();
        let result = pow_op.execute().unwrap();
        let result_data = result.to_vec().unwrap();

        assert!(
            (result_data[0] - 3.0).abs() < 1e-4,
            "z^1 should preserve real part"
        );
        assert!(
            (result_data[1] - 4.0).abs() < 1e-4,
            "z^1 should preserve imag part"
        );
    }

    #[tokio::test]
    async fn test_complex_pow_zero_exponent() {
        let device = crate::device::test_pool::get_test_device().await;

        // Test z^0 = 1+0i
        let data = vec![5.0f32, 12.0];
        let tensor = Tensor::from_data(&data, vec![1, 2], device).unwrap();

        let pow_op = ComplexPow::new(tensor, 0.0).unwrap();
        let result = pow_op.execute().unwrap();
        let result_data = result.to_vec().unwrap();

        assert!(
            (result_data[0] - 1.0).abs() < 1e-4,
            "z^0 real part should be 1"
        );
        assert!(
            (result_data[1] - 0.0).abs() < 1e-4,
            "z^0 imag part should be 0"
        );
    }
}

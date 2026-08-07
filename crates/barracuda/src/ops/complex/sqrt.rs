// SPDX-License-Identifier: AGPL-3.0-or-later
//! Complex Square Root via polar form

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

/// Complex square root via polar form: √z = √|z| · exp(i·arg(z)/2).
pub struct ComplexSqrt {
    input: Tensor,
}

impl ComplexSqrt {
    /// Create complex square root operation. Input must have last dim = 2 (re, im).
    /// # Errors
    /// Returns [`Err`] if input last dimension is not 2.
    pub fn new(input: Tensor) -> Result<Self> {
        if input.shape().last() != Some(&2) {
            return Err(BarracudaError::Device(
                "Must have last dimension = 2".to_string(),
            ));
        }
        Ok(Self { input })
    }

    /// Execute complex square root on GPU.
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let num_elements = self.input.len();
        let num_complex = num_elements / 2;

        let output_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Complex Sqrt Output"),
            size: (num_elements * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let params = [num_complex as u32];
        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Params"),
                contents: bytemuck::cast_slice(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        ComputeDispatch::new(device, "ComplexSqrt")
            .shader(include_str!("sqrt.wgsl"), "main")
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
    async fn test_complex_sqrt_basic() {
        let device = crate::device::test_pool::get_test_device().await;

        // Test sqrt(4+0i) = 2+0i
        let data = vec![4.0f32, 0.0];
        let tensor = Tensor::from_data(&data, vec![1, 2], device).unwrap();

        let sqrt_op = ComplexSqrt::new(tensor).unwrap();
        let result = sqrt_op.execute().unwrap();
        let result_data = result.to_vec().unwrap();

        assert!(
            (result_data[0] - 2.0).abs() < 1e-5,
            "Real part should be ~2.0"
        );
        assert!(
            (result_data[1] - 0.0).abs() < 1e-5,
            "Imag part should be ~0.0"
        );
    }

    #[tokio::test]
    async fn test_complex_sqrt_identity() {
        let device = crate::device::test_pool::get_test_device().await;

        // Test sqrt(z)^2 = z for z = 3+4i
        let data = vec![3.0f32, 4.0];
        let tensor = Tensor::from_data(&data, vec![1, 2], device).unwrap();

        let sqrt_op = ComplexSqrt::new(tensor).unwrap();
        let sqrt_result = sqrt_op.execute().unwrap();

        // Square the result (re^2 - im^2, 2*re*im)
        let sqrt_data = sqrt_result.to_vec().unwrap();
        let re = sqrt_data[0];
        let im = sqrt_data[1];
        let squared_re = re * re - im * im;
        let squared_im = 2.0 * re * im;

        assert!((squared_re - 3.0).abs() < 1e-4, "Should recover real part");
        assert!((squared_im - 4.0).abs() < 1e-4, "Should recover imag part");
    }
}

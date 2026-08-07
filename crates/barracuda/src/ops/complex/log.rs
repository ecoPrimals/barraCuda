// SPDX-License-Identifier: AGPL-3.0-or-later
//! Complex Logarithm

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

/// Complex natural logarithm: log(z) = log|z| + i·arg(z).
pub struct ComplexLog {
    input: Tensor,
}

impl ComplexLog {
    /// Create complex log operation. Input must have last dim = 2 (re, im).
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn new(input: Tensor) -> Result<Self> {
        if input.shape().last() != Some(&2) {
            return Err(BarracudaError::Device(
                "Must have last dimension = 2".to_string(),
            ));
        }
        Ok(Self { input })
    }

    /// Execute complex logarithm on GPU.
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
        let params = [num_complex as u32];
        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("P"),
                contents: bytemuck::cast_slice(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        ComputeDispatch::new(device, "ComplexLog")
            .shader(include_str!("log.wgsl"), "main")
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
    async fn test_complex_log_one() {
        let device = crate::device::test_pool::get_test_device().await;

        // Test log(1+0i) = 0+0i
        let data = vec![1.0f32, 0.0];
        let tensor = Tensor::from_data(&data, vec![1, 2], device).unwrap();

        let log_op = ComplexLog::new(tensor).unwrap();
        let result = log_op.execute().unwrap();
        let result_data = result.to_vec().unwrap();

        assert!(
            (result_data[0] - 0.0).abs() < 1e-5,
            "log(1) real part should be 0"
        );
        assert!(
            (result_data[1] - 0.0).abs() < 1e-5,
            "log(1) imag part should be 0"
        );
    }

    #[tokio::test]
    async fn test_complex_log_euler_base() {
        let device = crate::device::test_pool::get_test_device().await;

        // Test log(e+0i) = 1+0i (approximately)
        let e = std::f32::consts::E;
        let data = vec![e, 0.0];
        let tensor = Tensor::from_data(&data, vec![1, 2], device).unwrap();

        let log_op = ComplexLog::new(tensor).unwrap();
        let result = log_op.execute().unwrap();
        let result_data = result.to_vec().unwrap();

        assert!(
            (result_data[0] - 1.0).abs() < 1e-5,
            "log(e) real part should be 1"
        );
        assert!(
            (result_data[1] - 0.0).abs() < 1e-5,
            "log(e) imag part should be 0"
        );
    }
}

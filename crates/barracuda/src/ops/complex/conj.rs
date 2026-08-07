// SPDX-License-Identifier: AGPL-3.0-or-later
//! Complex Conjugate Operation
//!
//! **Operation**: conj(a + bi) = a - bi
//! **Complexity**: O(1) - trivial (one negation)
//! **CRITICAL**: Required for FFT normalization and inverse operations

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

/// Complex conjugate: conj(a+bi) = a-bi.
pub struct ComplexConj {
    input: Tensor,
}

impl ComplexConj {
    /// Create complex conjugate operation. Input must have last dim = 2 (re, im).
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

    /// Execute complex conjugate on GPU.
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
            label: Some("Complex Conj Output"),
            size: (num_elements * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params = [num_complex as u32];
        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Complex Conj Params"),
                contents: bytemuck::cast_slice(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        ComputeDispatch::new(device, "ComplexConj")
            .shader(include_str!("conj.wgsl"), "main")
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
    async fn test_complex_conj() {
        let device = crate::device::test_pool::get_test_device().await;

        // conj(3+4i) = 3-4i
        let data = vec![3.0f32, 4.0];
        let tensor = Tensor::from_data(&data, vec![1, 2], device).unwrap();

        let op = ComplexConj::new(tensor).unwrap();
        let result = op.execute().unwrap();

        let result_data = result.to_vec().unwrap();
        assert!((result_data[0] - 3.0).abs() < 1e-6);
        assert!((result_data[1] - (-4.0)).abs() < 1e-6);
    }

    #[tokio::test]
    async fn test_complex_conj_twice() {
        let device = crate::device::test_pool::get_test_device().await;
        // conj(conj(z)) = z
        let data = vec![3.0f32, 4.0];
        let tensor = Tensor::from_data(&data, vec![1, 2], device).unwrap();
        let conj1 = ComplexConj::new(tensor).unwrap().execute().unwrap();
        let conj2 = ComplexConj::new(conj1).unwrap().execute().unwrap();
        let result = conj2.to_vec().unwrap();
        assert!((result[0] - 3.0).abs() < 1e-6);
        assert!((result[1] - 4.0).abs() < 1e-6);
    }
}

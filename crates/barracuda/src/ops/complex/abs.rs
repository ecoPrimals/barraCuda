// SPDX-License-Identifier: AGPL-3.0-or-later
//! Complex Absolute Value (Magnitude) Operation
//!
//! **Operation**: |a + bi| = sqrt(a² + b²)
//! **Complexity**: O(1) - native WGSL `length()` function
//! **Use Case**: Power spectra, structure factors S(q)

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

/// Complex absolute value (magnitude): |a + bi| = sqrt(a² + b²).
pub struct ComplexAbs {
    input: Tensor,
}

impl ComplexAbs {
    /// Creates a new complex abs. Input last dimension must be 2 (real, imag).
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

    /// Executes complex abs and returns real-valued magnitudes.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let num_elements = self.input.len();
        let num_complex = num_elements / 2;

        // Output is real-valued (one f32 per complex number)
        let output_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Complex Abs Output"),
            size: (num_complex * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params = [num_complex as u32];
        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Complex Abs Params"),
                contents: bytemuck::cast_slice(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        ComputeDispatch::new(device, "ComplexAbs")
            .shader(include_str!("abs.wgsl"), "main")
            .storage_read(0, self.input.buffer())
            .storage_rw(1, &output_buffer)
            .uniform(2, &params_buffer)
            .dispatch_1d(num_complex as u32)
            .submit()?;

        // Output shape is [batch..., 1] (real magnitudes)
        let mut output_shape = self.input.shape().to_vec();
        *output_shape
            .last_mut()
            .ok_or_else(|| BarracudaError::execution_failed("output_shape empty"))? = 1;

        Ok(Tensor::from_buffer(
            output_buffer,
            output_shape,
            device.clone(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_complex_abs() {
        let device = crate::device::test_pool::get_test_device().await;

        // |3+4i| = 5
        let data = vec![3.0f32, 4.0];
        let tensor = Tensor::from_data(&data, vec![1, 2], device).unwrap();

        let op = ComplexAbs::new(tensor).unwrap();
        let result = op.execute().unwrap();

        let result_data = result.to_vec().unwrap();
        assert!((result_data[0] - 5.0).abs() < 1e-6);
    }
}

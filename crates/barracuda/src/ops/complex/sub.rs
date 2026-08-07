// SPDX-License-Identifier: AGPL-3.0-or-later
//! Complex Subtraction Operation
//!
//! **Operation**: (a + bi) - (c + di) = (a-c) + (b-d)i
//! **Complexity**: O(1) - trivial (native vec2 subtraction)
//! **Performance**: 1 SIMD operation

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

/// Complex subtraction: (a+bi) - (c+di) = (a-c) + (b-d)i.
pub struct ComplexSub {
    input_a: Tensor,
    input_b: Tensor,
}

impl ComplexSub {
    /// Creates a new complex sub. Both inputs must have last dim = 2 and same shape.
    /// # Errors
    /// Returns [`Err`] if shapes differ, last dimension is not 2, or tensors
    /// are on different devices.
    pub fn new(input_a: Tensor, input_b: Tensor) -> Result<Self> {
        if input_a.shape() != input_b.shape() {
            return Err(BarracudaError::Device(
                "Complex tensors must have same shape".to_string(),
            ));
        }

        let shape = input_a.shape();
        if shape.last() != Some(&2) {
            return Err(BarracudaError::Device(
                "Complex tensors must have last dimension = 2".to_string(),
            ));
        }

        if !std::ptr::eq(input_a.device().as_ref(), input_b.device().as_ref()) {
            return Err(BarracudaError::Device(
                "Tensors must be on the same device".to_string(),
            ));
        }

        Ok(Self { input_a, input_b })
    }

    /// Executes complex subtraction and returns the result.
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input_a.device();
        let num_elements = self.input_a.len();

        let output_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Complex Sub Output"),
            size: (num_elements * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params = [num_elements as u32 / 2];
        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Complex Sub Params"),
                contents: bytemuck::cast_slice(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        ComputeDispatch::new(device, "Complex Sub")
            .shader(include_str!("sub.wgsl"), "main")
            .storage_read(0, self.input_a.buffer())
            .storage_read(1, self.input_b.buffer())
            .storage_rw(2, &output_buffer)
            .uniform(3, &params_buffer)
            .dispatch_1d((num_elements / 2) as u32)
            .submit()?;

        Ok(Tensor::from_buffer(
            output_buffer,
            self.input_a.shape().to_vec(),
            device.clone(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_complex_sub_simple() {
        let device = crate::device::test_pool::get_test_device().await;

        let data_a = vec![5.0f32, 7.0];
        let data_b = vec![2.0f32, 3.0];

        let tensor_a = Tensor::from_data(&data_a, vec![1, 2], device.clone()).unwrap();
        let tensor_b = Tensor::from_data(&data_b, vec![1, 2], device).unwrap();

        let op = ComplexSub::new(tensor_a, tensor_b).unwrap();
        let result = op.execute().unwrap();

        let result_data = result.to_vec().unwrap();
        assert!((result_data[0] - 3.0).abs() < 1e-6);
        assert!((result_data[1] - 4.0).abs() < 1e-6);
    }

    #[tokio::test]
    async fn test_euler_identity_via_add_sub() {
        let device = crate::device::test_pool::get_test_device().await;
        let data = vec![3.0f32, 4.0];
        let tensor = Tensor::from_data(&data, vec![1, 2], device).unwrap();
        let op = ComplexSub::new(tensor.clone(), tensor).unwrap();
        let result = op.execute().unwrap().to_vec().unwrap();
        assert!((result[0] - 0.0).abs() < 1e-6);
        assert!((result[1] - 0.0).abs() < 1e-6);
    }
}

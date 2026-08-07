// SPDX-License-Identifier: AGPL-3.0-or-later
//! Stack - Pure WGSL
//!
//! Deep Debt Principles:
//! - Self-knowledge: Operation knows its computation
//! - Zero hardcoding: Hardware-agnostic implementation
//! - Modern idiomatic Rust: Safe, zero unsafe code
//! - Complete implementation: Production-ready, no mocks
//! - Hardware-agnostic: Pure WGSL for universal compute

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::Result;
use crate::tensor::Tensor;

/// Stack operation
pub struct Stack {
    tensors: Vec<Tensor>,
    dim: usize,
}

impl Stack {
    /// Create a new stack operation
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn new(tensors: Vec<Tensor>, dim: usize) -> Result<Self> {
        if tensors.is_empty() {
            return Err(crate::error::BarracudaError::invalid_input(
                "Cannot stack empty tensor list",
            ));
        }

        // Validate all tensors have same shape
        let first_shape = tensors[0].shape();
        for (i, tensor) in tensors.iter().enumerate().skip(1) {
            if tensor.shape() != first_shape {
                return Err(crate::error::BarracudaError::invalid_input(format!(
                    "All tensors must have same shape. Tensor 0: {:?}, Tensor {}: {:?}",
                    first_shape,
                    i,
                    tensor.shape()
                )));
            }
        }

        if dim > first_shape.len() {
            return Err(crate::error::BarracudaError::invalid_input(format!(
                "dim {} exceeds tensor rank {}",
                dim,
                first_shape.len()
            )));
        }

        Ok(Self { tensors, dim })
    }

    /// Execute the stack operation
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.tensors[0].device();
        let num_tensors = self.tensors.len();
        let tensor_size: usize = self.tensors[0].shape().iter().product();

        // Compute output shape
        let mut output_shape = self.tensors[0].shape().to_vec();
        output_shape.insert(self.dim, num_tensors);
        let output_size: usize = output_shape.iter().product();

        // Create concatenated input buffer using direct buffer-to-buffer copies
        let input_size = num_tensors * tensor_size;
        let input_buffer = device.create_buffer_f32(input_size)?;

        // Copy each tensor buffer directly to the concatenated buffer
        let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("Stack Copy Encoder"),
        });

        for (i, tensor) in self.tensors.iter().enumerate() {
            let offset = i * tensor_size * std::mem::size_of::<f32>();
            encoder.copy_buffer_to_buffer(
                tensor.buffer(),
                0,
                &input_buffer,
                offset as u64,
                (tensor_size * std::mem::size_of::<f32>()) as u64,
            );
        }

        device.submit_commands(Some(encoder.finish()));

        // Create output buffer
        let output_buffer = device.create_buffer_f32(output_size)?;

        // Create parameters
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct StackParams {
            num_tensors: u32,
            tensor_size: u32,
            output_size: u32,
            stack_dim: u32,
        }

        let params = StackParams {
            num_tensors: num_tensors as u32,
            tensor_size: tensor_size as u32,
            output_size: output_size as u32,
            stack_dim: self.dim as u32,
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Stack Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        ComputeDispatch::new(device, "Stack")
            .shader(include_str!("../shaders/tensor/stack_f64.wgsl"), "main")
            .uniform(0, &params_buffer)
            .storage_read(1, &input_buffer)
            .storage_rw(2, &output_buffer)
            .dispatch_1d(output_size as u32)
            .submit()?;

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
    async fn test_stack_basic() {
        let device = crate::device::test_pool::get_test_device().await;
        let t1 = Tensor::from_data(&[1.0, 2.0], vec![2], device.clone()).unwrap();
        let t2 = Tensor::from_data(&[3.0, 4.0], vec![2], device).unwrap();

        let stacked = Stack::new(vec![t1, t2], 0).unwrap().execute().unwrap();
        assert_eq!(stacked.shape(), &vec![2, 2]);
    }

    #[tokio::test]
    async fn test_stack_multiple() {
        let device = crate::device::test_pool::get_test_device().await;
        let tensors: Vec<Tensor> = (0..5)
            .map(|i| Tensor::from_data(&[i as f32; 4], vec![2, 2], device.clone()).unwrap())
            .collect();

        let stacked = Stack::new(tensors, 0).unwrap().execute().unwrap();
        assert_eq!(stacked.shape(), &vec![5, 2, 2]);
    }

    #[tokio::test]
    async fn test_stack_empty() {
        let _device = crate::device::test_pool::get_test_device().await;
        assert!(Stack::new(vec![], 0).is_err());
    }

    #[tokio::test]
    async fn test_stack_shape_mismatch() {
        let device = crate::device::test_pool::get_test_device().await;
        let t1 = Tensor::from_data(&[1.0, 2.0], vec![2], device.clone()).unwrap();
        let t2 = Tensor::from_data(&[3.0, 4.0, 5.0], vec![3], device).unwrap();

        assert!(Stack::new(vec![t1, t2], 0).is_err());
    }

    #[tokio::test]
    async fn test_stack_dim_invalid() {
        let device = crate::device::test_pool::get_test_device().await;
        let t1 = Tensor::from_data(&[1.0, 2.0], vec![2], device.clone()).unwrap();
        let t2 = Tensor::from_data(&[3.0, 4.0], vec![2], device).unwrap();

        assert!(Stack::new(vec![t1, t2], 10).is_err());
    }
}

// SPDX-License-Identifier: AGPL-3.0-or-later
//! Permute - Pure WGSL
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

/// Permute operation (reorder dimensions)
pub struct Permute {
    input: Tensor,
    permutation: Vec<usize>,
}

impl Permute {
    /// Create a new permute operation
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn new(input: Tensor, permutation: Vec<usize>) -> Result<Self> {
        let num_dims = input.shape().len();
        if permutation.len() != num_dims {
            return Err(crate::error::BarracudaError::InvalidInput {
                message: format!(
                    "Permutation length {} doesn't match tensor rank {}",
                    permutation.len(),
                    num_dims
                ),
            });
        }

        // Validate permutation is valid (contains all indices 0..num_dims-1)
        let mut seen = vec![false; num_dims];
        for &idx in &permutation {
            if idx >= num_dims {
                return Err(crate::error::BarracudaError::InvalidInput {
                    message: format!("Invalid permutation index {idx} for rank {num_dims}"),
                });
            }
            if seen[idx] {
                return Err(crate::error::BarracudaError::InvalidInput {
                    message: format!("Duplicate index {idx} in permutation"),
                });
            }
            seen[idx] = true;
        }

        Ok(Self { input, permutation })
    }

    /// Get the WGSL shader source
    fn wgsl_shader() -> &'static str {
        static S: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
            include_str!("../shaders/tensor/permute_f64.wgsl").to_string()
        });
        &S
    }

    /// Execute the permute operation
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let input_shape = self.input.shape();
        let num_dims = input_shape.len();
        let total_size: usize = input_shape.iter().product();

        // Compute output shape
        let output_shape: Vec<usize> = self
            .permutation
            .iter()
            .map(|&idx| input_shape[idx])
            .collect();

        // Compute input strides
        let mut input_strides = vec![1; num_dims];
        for i in (0..num_dims - 1).rev() {
            input_strides[i] = input_strides[i + 1] * input_shape[i + 1];
        }

        // Access input buffer directly (zero-copy)
        let input_buffer = self.input.buffer();

        // Create buffers for shape and stride data
        let input_shape_u32: Vec<u32> = input_shape.iter().map(|&x| x as u32).collect();
        let output_shape_u32: Vec<u32> = output_shape.iter().map(|&x| x as u32).collect();
        let permutation_u32: Vec<u32> = self.permutation.iter().map(|&x| x as u32).collect();
        let input_strides_u32: Vec<u32> = input_strides.iter().map(|&x| x as u32).collect();

        let input_shape_buffer =
            device.create_buffer_u32_init("Permute Input Shape", &input_shape_u32);
        let output_shape_buffer =
            device.create_buffer_u32_init("Permute Output Shape", &output_shape_u32);
        let permutation_buffer =
            device.create_buffer_u32_init("Permute Permutation", &permutation_u32);
        let input_strides_buffer =
            device.create_buffer_u32_init("Permute Input Strides", &input_strides_u32);

        // Create output buffer
        let output_buffer = device.create_buffer_f32(total_size)?;

        // Create uniform buffer for parameters
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            total_size: u32,
            num_dims: u32,
            _pad1: u32,
            _pad2: u32,
        }

        let params = Params {
            total_size: total_size as u32,
            num_dims: num_dims as u32,
            _pad1: 0,
            _pad2: 0,
        };

        let params_buffer = device.create_uniform_buffer("Permute Params", &params);

        ComputeDispatch::new(device, "Permute")
            .shader(Self::wgsl_shader(), "main")
            .uniform(0, &params_buffer)
            .storage_read(1, input_buffer)
            .storage_read(2, &input_shape_buffer)
            .storage_read(3, &output_shape_buffer)
            .storage_read(4, &permutation_buffer)
            .storage_read(5, &input_strides_buffer)
            .storage_rw(6, &output_buffer)
            .dispatch_1d(total_size as u32)
            .submit()?;

        // Return tensor without reading back (zero-copy)
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
    async fn test_permute_basic() {
        let device = crate::device::test_pool::get_test_device().await;
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let input = Tensor::from_data(&data, vec![2, 3, 4], device).unwrap();

        let permuted = Permute::new(input, vec![0, 2, 1])
            .unwrap()
            .execute()
            .unwrap();
        assert_eq!(permuted.shape(), &vec![2, 4, 3]);
    }

    #[tokio::test]
    async fn test_permute_identity() {
        let device = crate::device::test_pool::get_test_device().await;
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let input = Tensor::from_data(&data, vec![2, 3, 2], device).unwrap();

        let permuted = Permute::new(input, vec![0, 1, 2])
            .unwrap()
            .execute()
            .unwrap();
        assert_eq!(permuted.shape(), &vec![2, 3, 2]);
    }

    #[tokio::test]
    async fn test_permute_invalid_length() {
        let device = crate::device::test_pool::get_test_device().await;
        let input = Tensor::from_data(&[1.0, 2.0, 3.0], vec![3], device).unwrap();

        assert!(Permute::new(input, vec![0, 1, 2, 3]).is_err());
    }

    #[tokio::test]
    async fn test_permute_invalid_index() {
        let device = crate::device::test_pool::get_test_device().await;
        let input = Tensor::from_data(&[1.0, 2.0, 3.0], vec![3], device).unwrap();

        assert!(Permute::new(input, vec![5]).is_err());
    }

    #[tokio::test]
    async fn test_permute_duplicate() {
        let device = crate::device::test_pool::get_test_device().await;
        let input = Tensor::from_data(&[1.0, 2.0, 3.0, 4.0], vec![2, 2], device).unwrap();

        assert!(Permute::new(input, vec![0, 0]).is_err());
    }
}

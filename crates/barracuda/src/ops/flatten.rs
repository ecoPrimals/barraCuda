// SPDX-License-Identifier: AGPL-3.0-or-later
//! Flatten - Pure WGSL
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

/// Flatten operation
pub struct Flatten {
    input: Tensor,
    start_dim: usize,
    end_dim: usize,
}

impl Flatten {
    /// Create a new flatten operation
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if `start_dim` >= `end_dim`, or dimensions are out of range.
    pub fn new(input: Tensor, start_dim: usize, end_dim: usize) -> Result<Self> {
        let shape = input.shape();
        if start_dim >= shape.len() || end_dim >= shape.len() || start_dim > end_dim {
            return Err(crate::error::BarracudaError::InvalidInput {
                message: format!(
                    "Invalid flatten dimensions: start_dim={start_dim}, end_dim={end_dim}, shape={shape:?}"
                ),
            });
        }
        Ok(Self {
            input,
            start_dim,
            end_dim,
        })
    }

    /// Get the WGSL shader source
    fn wgsl_shader() -> &'static str {
        {
            static S: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
                include_str!("../shaders/tensor/flatten_f64.wgsl").to_string()
            });
            &S
        }
    }

    /// Execute the flatten operation
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let size: usize = self.input.shape().iter().product();

        // Compute output shape
        let input_shape = self.input.shape();
        let mut output_shape = input_shape[..self.start_dim].to_vec();
        let flattened_size: usize = input_shape[self.start_dim..=self.end_dim].iter().product();
        output_shape.push(flattened_size);
        if self.end_dim + 1 < input_shape.len() {
            output_shape.extend_from_slice(&input_shape[self.end_dim + 1..]);
        }

        // Access input buffer directly (zero-copy)
        let input_buffer = self.input.buffer();

        // Create output buffer
        let output_buffer = device.create_buffer_f32(size)?;

        // Create uniform buffer for parameters
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            size: u32,
            _pad1: u32,
            _pad2: u32,
            _pad3: u32,
        }

        let params = Params {
            size: size as u32,
            _pad1: 0,
            _pad2: 0,
            _pad3: 0,
        };

        let params_buffer = device.create_uniform_buffer("Flatten Params", &params);

        ComputeDispatch::new(device, "Flatten")
            .shader(Self::wgsl_shader(), "main")
            .uniform(0, &params_buffer)
            .storage_read(1, input_buffer)
            .storage_rw(2, &output_buffer)
            .dispatch_1d(size as u32)
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
    use crate::device::WgpuDevice;
    use std::sync::Arc;

    async fn get_test_device() -> Option<Arc<WgpuDevice>> {
        crate::device::test_pool::get_test_device_if_gpu_available().await
    }

    #[tokio::test]
    async fn test_flatten_basic() {
        let Some(device) = get_test_device().await else {
            return;
        };
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let input = Tensor::from_data(&data, vec![2, 3, 4], device).unwrap();

        let flattened = Flatten::new(input, 1, 2).unwrap().execute().unwrap();
        assert_eq!(flattened.shape(), &vec![2, 12]);
    }

    #[tokio::test]
    async fn test_flatten_all() {
        let Some(device) = get_test_device().await else {
            return;
        };
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let input = Tensor::from_data(&data, vec![2, 3, 2], device).unwrap();

        let flattened = Flatten::new(input, 0, 2).unwrap().execute().unwrap();
        assert_eq!(flattened.shape(), &vec![12]);
    }

    #[tokio::test]
    async fn test_flatten_partial() {
        let Some(device) = get_test_device().await else {
            return;
        };
        let data: Vec<f32> = (0..60).map(|i| i as f32).collect();
        let input = Tensor::from_data(&data, vec![2, 3, 5, 2], device).unwrap();

        let flattened = Flatten::new(input, 1, 2).unwrap().execute().unwrap();
        assert_eq!(flattened.shape(), &vec![2, 15, 2]);
    }

    #[tokio::test]
    async fn test_flatten_single_dim() {
        let Some(device) = get_test_device().await else {
            return;
        };
        let data: Vec<f32> = (0..20).map(|i| i as f32).collect();
        let input = Tensor::from_data(&data, vec![4, 5], device).unwrap();

        let flattened = Flatten::new(input, 0, 0).unwrap().execute().unwrap();
        assert_eq!(flattened.shape(), &vec![4, 5]);
    }

    #[tokio::test]
    async fn test_flatten_invalid() {
        let Some(device) = get_test_device().await else {
            return;
        };
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let input = Tensor::from_data(&data, vec![2, 3, 2], device).unwrap();

        assert!(Flatten::new(input.clone(), 3, 2).is_err());
        assert!(Flatten::new(input, 1, 0).is_err());
    }
}

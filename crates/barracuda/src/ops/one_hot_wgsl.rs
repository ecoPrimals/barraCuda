// SPDX-License-Identifier: AGPL-3.0-or-later
//! One-hot - Convert indices to one-hot vectors - Pure WGSL
//!
//! Deep Debt Principles:
//! - Self-knowledge: Operation knows number of classes
//! - Zero hardcoding: All parameters passed at runtime
//! - Modern idiomatic Rust: Safe, zero unsafe code
//! - Complete implementation: Production-ready, no mocks
//! - Hardware-agnostic: Pure WGSL for universal compute

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::Result;
use crate::tensor::Tensor;

/// One-hot operation - Convert indices to one-hot encoded vectors
pub struct OneHot {
    indices: Vec<usize>,
    num_classes: usize,
}

impl OneHot {
    /// Create a new one-hot operation
    #[must_use]
    pub fn new(indices: Vec<usize>, num_classes: usize) -> Self {
        Self {
            indices,
            num_classes,
        }
    }

    /// Execute the one-hot operation
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self, device: &std::sync::Arc<crate::device::WgpuDevice>) -> Result<Tensor> {
        let num_indices = self.indices.len();
        let output_size = num_indices * self.num_classes;

        // Create buffers
        let indices_u32: Vec<u32> = self.indices.iter().map(|&x| x as u32).collect();
        let indices_buffer = device.create_buffer_u32_init("OneHot Indices", &indices_u32);

        let output_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("OneHot Output"),
            size: (output_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create uniform buffer for parameters
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            num_indices: u32,
            num_classes: u32,
        }

        let params = Params {
            num_indices: num_indices as u32,
            num_classes: self.num_classes as u32,
        };

        let params_buffer = device.create_uniform_buffer("OneHot Params", &params);

        ComputeDispatch::new(device, "OneHot")
            .shader(include_str!("../shaders/misc/one_hot_f64.wgsl"), "main")
            .storage_read(0, &indices_buffer)
            .storage_rw(1, &output_buffer)
            .uniform(2, &params_buffer)
            .dispatch_1d(output_size as u32)
            .submit()?;

        // Read back results
        let output_data = crate::utils::read_buffer(device, &output_buffer, output_size)?;

        Ok(Tensor::new(
            output_data,
            vec![num_indices, self.num_classes],
            device.clone(),
        ))
    }
}

impl Tensor {
    /// Convert indices to one-hot encoded vectors
    ///
    /// # Arguments
    ///
    /// * `num_classes` - Number of classes for one-hot encoding
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn one_hot_wgsl(self, num_classes: usize) -> Result<Self> {
        // Extract indices from tensor
        let data = self.to_vec()?;
        let indices: Vec<usize> = data.iter().map(|&x| x as usize).collect();
        OneHot::new(indices, num_classes).execute(self.device())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_one_hot_basic() {
        let device = crate::device::test_pool::get_test_device().await;
        let indices = vec![0, 1, 2];
        let output = OneHot::new(indices, 4).execute(&device).unwrap();

        assert_eq!(output.shape(), &[3, 4]);
        let result = output.to_vec().unwrap();

        // First row: class 0
        assert_eq!(result[0], 1.0);
        assert_eq!(result[1], 0.0);
        assert_eq!(result[2], 0.0);
        assert_eq!(result[3], 0.0);

        // Second row: class 1
        assert_eq!(result[4], 0.0);
        assert_eq!(result[5], 1.0);
        assert_eq!(result[6], 0.0);
        assert_eq!(result[7], 0.0);

        // Third row: class 2
        assert_eq!(result[8], 0.0);
        assert_eq!(result[9], 0.0);
        assert_eq!(result[10], 1.0);
        assert_eq!(result[11], 0.0);
    }
}

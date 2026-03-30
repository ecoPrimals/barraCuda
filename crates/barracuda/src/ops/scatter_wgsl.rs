// SPDX-License-Identifier: AGPL-3.0-or-later
//! Scatter - Write values to specific indices - Pure WGSL
//!
//! Deep Debt Principles:
//! - Self-knowledge: Operation knows its dimension and indices
//! - Zero hardcoding: All parameters passed at runtime
//! - Modern idiomatic Rust: Safe, zero unsafe code
//! - Complete implementation: Production-ready, no mocks
//! - Hardware-agnostic: Pure WGSL for universal compute

use crate::device::DeviceCapabilities;
use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::Result;
use crate::tensor::Tensor;

/// Scatter operation - Write values to specific indices
pub struct Scatter {
    input: Tensor,
    dim: usize,
    indices: Vec<u32>,
    values: Tensor,
}

impl Scatter {
    /// Create a new scatter operation
    #[must_use]
    pub fn new(input: Tensor, dim: usize, indices: Vec<u32>, values: Tensor) -> Self {
        Self {
            input,
            dim,
            indices,
            values,
        }
    }

    /// Get the WGSL shader source
    fn wgsl_shader() -> &'static str {
        {
            static S: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
                include_str!("../shaders/tensor/scatter_f64.wgsl").to_string()
            });
            &S
        }
    }

    /// Execute the scatter operation
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let shape = self.input.shape();
        let size: usize = shape.iter().product();

        // Calculate dimension parameters
        let dim_size = shape[self.dim];
        let outer_size: usize = shape[..self.dim].iter().product();
        let inner_size: usize = shape[self.dim + 1..].iter().product();
        let scatter_size = self.indices.len();

        let values_size = outer_size * scatter_size * inner_size;

        // Create buffers
        // Access input buffer directly (zero-copy)
        let input_buffer = self.input.buffer();

        let indices_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Scatter Indices"),
                contents: bytemuck::cast_slice(&self.indices),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

        let values_buffer = self.values.buffer();

        let output_buffer = device.create_buffer_f32(size)?;

        // Create uniform buffer for parameters
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            size: u32,
            dim_size: u32,
            outer_size: u32,
            inner_size: u32,
            scatter_size: u32,
        }

        let params = Params {
            size: size as u32,
            dim_size: dim_size as u32,
            outer_size: outer_size as u32,
            inner_size: inner_size as u32,
            scatter_size: scatter_size as u32,
        };

        let params_buffer = device.create_uniform_buffer("Scatter Params", &params);

        let caps = DeviceCapabilities::from_device(device);
        let workgroups_copy = caps.dispatch_1d(size as u32);
        let workgroups_scatter = caps.dispatch_1d(values_size as u32);

        // First pass: copy input to output
        ComputeDispatch::new(device, "scatter_copy")
            .shader(Self::wgsl_shader(), "main")
            .storage_read(0, input_buffer)
            .storage_read(1, &indices_buffer)
            .storage_read(2, values_buffer)
            .storage_rw(3, &output_buffer)
            .uniform(4, &params_buffer)
            .dispatch(workgroups_copy.max(1), 1, 1)
            .submit()?;

        // Second pass: scatter values
        ComputeDispatch::new(device, "scatter")
            .shader(Self::wgsl_shader(), "scatter")
            .storage_read(0, input_buffer)
            .storage_read(1, &indices_buffer)
            .storage_read(2, values_buffer)
            .storage_rw(3, &output_buffer)
            .uniform(4, &params_buffer)
            .dispatch(workgroups_scatter.max(1), 1, 1)
            .submit()?;

        // Read back results
        let output_data = crate::utils::read_buffer(device, &output_buffer, size)?;

        Ok(Tensor::new(output_data, shape.to_vec(), device.clone()))
    }
}

impl Tensor {
    /// Scatter values to specific indices along a dimension
    /// # Arguments
    /// * `dim` - Dimension to scatter to
    /// * `indices` - Indices to scatter to
    /// * `values` - Values to scatter
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn scatter_wgsl(self, dim: usize, indices: Vec<u32>, values: Self) -> Result<Self> {
        Scatter::new(self, dim, indices, values).execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_scatter_1d() {
        let device = crate::device::test_pool::get_test_device().await;
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let input = Tensor::new(data, vec![5], device.clone());

        let values = Tensor::new(vec![10.0, 20.0], vec![2], device);
        let output = input.scatter_wgsl(0, vec![1, 3], values).unwrap();

        assert_eq!(output.shape(), &[5]);
        let result = output.to_vec().unwrap();
        assert_eq!(result[0], 1.0);
        assert_eq!(result[1], 10.0);
        assert_eq!(result[2], 3.0);
        assert_eq!(result[3], 20.0);
        assert_eq!(result[4], 5.0);
    }

    #[tokio::test]
    async fn test_scatter_2d() {
        let device = crate::device::test_pool::get_test_device().await;
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let input = Tensor::new(data, vec![3, 2], device.clone());

        let values = Tensor::new(vec![10.0, 20.0], vec![1, 2], device);
        let output = input.scatter_wgsl(0, vec![1], values).unwrap();

        assert_eq!(output.shape(), &[3, 2]);
        let result = output.to_vec().unwrap();
        // Original: [[1,2], [3,4], [5,6]]
        // Scatter [10, 20] to index 1: [[1,2], [10,20], [5,6]]
        assert_eq!(result[0], 1.0);
        assert_eq!(result[1], 2.0);
        assert_eq!(result[2], 10.0);
        assert_eq!(result[3], 20.0);
        assert_eq!(result[4], 5.0);
        assert_eq!(result[5], 6.0);
    }
}

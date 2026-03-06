// SPDX-License-Identifier: AGPL-3.0-or-later
//! Index Add - Add values at specific indices - Pure WGSL
//!
//! Deep Debt Principles:
//! - Self-knowledge: Operation knows its indices and values
//! - Zero hardcoding: All parameters passed at runtime
//! - Modern idiomatic Rust: Safe, zero unsafe code
//! - Complete implementation: Production-ready, no mocks
//! - Hardware-agnostic: Pure WGSL for universal compute

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

/// Index Add operation - Add values at specific indices (scatter-add)
pub struct IndexAdd {
    input: Tensor,
    dim: usize,
    indices: Vec<u32>,
    values: Tensor,
}

impl IndexAdd {
    /// Create a new index add operation
    /// # Errors
    /// Returns [`Err`] if dimension is out of bounds, values size does not match expected shape,
    /// or any index is out of bounds for the dimension.
    pub fn new(input: Tensor, dim: usize, indices: Vec<u32>, values: Tensor) -> Result<Self> {
        let input_shape = input.shape();

        // Validate dimension
        if dim >= input_shape.len() {
            return Err(BarracudaError::invalid_op(
                "IndexAdd",
                format!(
                    "Dimension {} out of bounds for rank {}",
                    dim,
                    input_shape.len()
                ),
            ));
        }

        // Calculate dimension parameters
        let dim_size = input_shape[dim];
        let outer_size: usize = input_shape[..dim].iter().product();
        let inner_size: usize = input_shape[dim + 1..].iter().product();
        let values_size = values.shape().iter().product::<usize>();
        let expected_values_size = outer_size * indices.len() * inner_size;

        if values_size != expected_values_size {
            return Err(BarracudaError::invalid_op(
                "IndexAdd",
                format!(
                    "Values size {values_size} doesn't match expected size {expected_values_size}"
                ),
            ));
        }

        // Validate indices are in bounds
        for &idx in &indices {
            if idx as usize >= dim_size {
                return Err(BarracudaError::invalid_op(
                    "IndexAdd",
                    format!("Index {idx} out of bounds for dimension size {dim_size}"),
                ));
            }
        }

        Ok(Self {
            input,
            dim,
            indices,
            values,
        })
    }

    /// Get the WGSL shader source
    fn wgsl_shader() -> &'static str {
        static SHADER: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
            crate::shaders::precision::downcast_f64_to_f32_with_transcendentals(include_str!(
                "../shaders/math/index_add_f64.wgsl"
            ))
        });
        &SHADER
    }

    /// Execute the index add operation (modifies input in-place)
    /// # Errors
    /// Returns [`Err`] if buffer allocation fails, GPU dispatch fails, or the device is lost.
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

        // Access buffers directly (zero-copy)
        let input_buffer = self.input.buffer();
        let values_buffer = self.values.buffer();

        // Create indices buffer
        let indices_buffer = device.create_buffer_u32_init("IndexAdd Indices", &self.indices);

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

        let params_buffer = device.create_uniform_buffer("IndexAdd Params", &params);

        ComputeDispatch::new(device, "IndexAdd")
            .shader(Self::wgsl_shader(), "main")
            .uniform(0, &params_buffer)
            .storage_read(1, values_buffer)
            .storage_read(2, &indices_buffer)
            .storage_rw(3, input_buffer)
            .dispatch_1d(values_size as u32)
            .submit()?;

        // Return the input tensor (modified in-place)
        Ok(self.input)
    }
}

impl Tensor {
    /// Add values at specific indices along a dimension (scatter-add)
    /// # Arguments
    /// * `dim` - Dimension to add along
    /// * `indices` - Indices to add at
    /// * `values` - Values to add
    /// # Errors
    /// Returns [`Err`] if dimension is out of bounds, values shape does not match, indices are
    /// out of bounds, buffer allocation fails, GPU dispatch fails, or the device is lost.
    pub fn index_add(self, dim: usize, indices: Vec<u32>, values: Tensor) -> Result<Self> {
        IndexAdd::new(self, dim, indices, values)?.execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn get_test_device() -> Option<std::sync::Arc<crate::device::WgpuDevice>> {
        crate::device::test_pool::get_test_device_if_gpu_available().await
    }

    #[tokio::test]
    async fn test_index_add_1d() {
        let Some(device) = get_test_device().await else {
            return;
        };
        let input = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![5], device.clone());
        let values = Tensor::new(vec![10.0, 20.0], vec![2], device.clone());

        let result = input.index_add(0, vec![1, 3], values).unwrap();
        let output_data = result.to_vec().unwrap();

        // Expected: [1, 12, 3, 24, 5]
        assert_eq!(output_data[0], 1.0);
        assert_eq!(output_data[1], 12.0);
        assert_eq!(output_data[2], 3.0);
        assert_eq!(output_data[3], 24.0);
        assert_eq!(output_data[4], 5.0);
    }
}

// SPDX-License-Identifier: AGPL-3.0-only
//! `MoveDim` - Complete dimension reordering
//!
//! **Deep Debt Principles**:
//! - Complete implementation: Full dimension reordering, not simplified copy
//! - Zero hardcoding: All parameters configurable
//! - Self-knowledge: Validates input shapes and dimension indices
//! - Modern idiomatic Rust: Result<T, E>, pattern matching
//! - Pure GPU: No CPU fallbacks

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct MoveDimParams {
    total_size: u32,
    num_dims: u32,
    source_dim: u32,
    dest_dim: u32,
    _padding: [u32; 4],
}

/// Dimension reordering operation (move source dimension to destination).
pub struct MoveDim {
    input: Tensor,
    source_dim: usize,
    dest_dim: usize,
}

impl MoveDim {
    /// Creates a new movedim operation. Moves `source_dim` to `dest_dim`.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn new(input: Tensor, source_dim: usize, dest_dim: usize) -> Result<Self> {
        let shape = input.shape();
        let num_dims = shape.len();

        if num_dims == 0 {
            return Err(BarracudaError::invalid_op(
                "movedim",
                "Cannot move dimension of scalar tensor",
            ));
        }

        if source_dim >= num_dims {
            return Err(BarracudaError::invalid_op(
                "movedim",
                format!("source_dim {source_dim} exceeds tensor rank {num_dims}"),
            ));
        }

        if dest_dim >= num_dims {
            return Err(BarracudaError::invalid_op(
                "movedim",
                format!("dest_dim {dest_dim} exceeds tensor rank {num_dims}"),
            ));
        }

        Ok(Self {
            input,
            source_dim,
            dest_dim,
        })
    }

    fn wgsl_shader() -> &'static str {
        {
            static S: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
                include_str!("../shaders/tensor/movedim_f64.wgsl").to_string()
            });
            &S
        }
    }

    /// Executes dimension reordering and returns the result tensor.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let shape = self.input.shape();
        let num_dims = shape.len();
        let total_size: usize = shape.iter().product();

        // Compute output shape (reordered)
        let mut output_shape = shape.to_vec();
        let dim_value = output_shape.remove(self.source_dim);
        output_shape.insert(self.dest_dim.min(num_dims - 1), dim_value);

        // Compute strides for input and output
        let mut input_strides = vec![1u32; num_dims];
        let mut output_strides = vec![1u32; num_dims];

        for i in (0..num_dims - 1).rev() {
            input_strides[i] = input_strides[i + 1] * shape[i + 1] as u32;
        }

        for i in (0..num_dims - 1).rev() {
            output_strides[i] = output_strides[i + 1] * output_shape[i + 1] as u32;
        }

        // Create dimension mapping
        let mut input_dims: Vec<usize> = (0..num_dims).collect();
        let moved_dim = input_dims.remove(self.source_dim);
        input_dims.insert(self.dest_dim.min(num_dims - 1), moved_dim);

        // Simplified: direct mapping
        let mut dim_mapping = vec![0u32; num_dims];
        for i in 0..num_dims {
            dim_mapping[i] = input_dims[i] as u32;
        }

        // Create buffers
        let output_buffer = device.create_buffer_f32(total_size)?;

        let input_shape_u32: Vec<u32> = shape.iter().map(|&s| s as u32).collect();
        let output_shape_u32: Vec<u32> = output_shape.iter().map(|&s| s as u32).collect();

        let input_shape_buffer =
            device.create_buffer_u32_init("MoveDim Input Shape", &input_shape_u32);
        let output_shape_buffer =
            device.create_buffer_u32_init("MoveDim Output Shape", &output_shape_u32);
        let input_strides_buffer =
            device.create_buffer_u32_init("MoveDim Input Strides", &input_strides);
        let output_strides_buffer =
            device.create_buffer_u32_init("MoveDim Output Strides", &output_strides);
        let dim_mapping_buffer = device.create_buffer_u32_init("MoveDim Dim Mapping", &dim_mapping);

        let params = MoveDimParams {
            total_size: total_size as u32,
            num_dims: num_dims as u32,
            source_dim: self.source_dim as u32,
            dest_dim: self.dest_dim as u32,
            _padding: [0; 4],
        };

        let params_buffer = device.create_uniform_buffer("MoveDim Params", &params);

        ComputeDispatch::new(device, "MoveDim")
            .shader(Self::wgsl_shader(), "main")
            .uniform(0, &params_buffer)
            .storage_read(1, self.input.buffer())
            .storage_read(2, &input_shape_buffer)
            .storage_read(3, &output_shape_buffer)
            .storage_read(4, &input_strides_buffer)
            .storage_read(5, &output_strides_buffer)
            .storage_read(6, &dim_mapping_buffer)
            .storage_rw(7, &output_buffer)
            .dispatch_1d(total_size as u32)
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
    use crate::device::test_pool::get_test_device_if_gpu_available;

    #[tokio::test]
    async fn test_movedim_basic() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let input = Tensor::from_vec_on(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],
            device.clone(),
        )
        .await
        .unwrap();

        let result = MoveDim::new(input, 0, 1).unwrap().execute().unwrap();
        assert_eq!(result.shape(), &[3, 2]);
    }

    #[tokio::test]
    async fn test_movedim_3d() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let input = Tensor::from_vec_on(
            (0..24).map(|i| i as f32).collect(),
            vec![2, 3, 4],
            device.clone(),
        )
        .await
        .unwrap();

        let result = MoveDim::new(input, 1, 2).unwrap().execute().unwrap();
        assert_eq!(result.shape(), &[2, 4, 3]);
    }

    #[tokio::test]
    async fn test_movedim_same_dim() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let input = Tensor::from_vec_on(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], device.clone())
            .await
            .unwrap();

        let result = MoveDim::new(input, 0, 0).unwrap().execute().unwrap();
        assert_eq!(result.shape(), &[2, 2]);
    }

    #[tokio::test]
    async fn test_movedim_invalid_dim() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let input = Tensor::from_vec_on(vec![1.0, 2.0, 3.0], vec![3], device.clone())
            .await
            .unwrap();

        assert!(MoveDim::new(input.clone(), 5, 0).is_err());
        assert!(MoveDim::new(input, 0, 5).is_err());
    }

    #[tokio::test]
    async fn test_movedim_4d() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let input = Tensor::from_vec_on(
            (0..120).map(|i| i as f32).collect(),
            vec![2, 3, 4, 5],
            device.clone(),
        )
        .await
        .unwrap();

        let result = MoveDim::new(input, 0, 3).unwrap().execute().unwrap();
        assert_eq!(result.shape(), &[3, 4, 5, 2]);
    }
}

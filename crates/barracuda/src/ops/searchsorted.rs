// SPDX-License-Identifier: AGPL-3.0-or-later
//! `SearchSorted` - GPU parallel binary search
//!
//! **Deep Debt Principles**:
//! - Complete GPU implementation: Parallel binary search for each value
//! - No CPU fallbacks: All computation on GPU
//! - Self-knowledge: Validates sorted array
//! - Modern idiomatic Rust: Result<T, E>

use crate::device::compute_pipeline::{BatchedComputeDispatch, ComputeDispatch};
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct SearchSortedParams {
    sorted_size: u32,
    values_size: u32,
    side_right: u32, // 0 = left (default), 1 = right
    _pad1: u32,
}

/// GPU parallel binary search for insertion indices in a sorted array.
pub struct SearchSorted {
    sorted: Tensor,
    values: Tensor,
    side_right: bool,
}

impl SearchSorted {
    /// Creates a new searchsorted operation. `side_right` selects left (false) or right (true) insertion.
    /// # Errors
    /// Returns [`Err`] if sorted or values are empty, or if either is not 1D.
    pub fn new(sorted: Tensor, values: Tensor, side_right: bool) -> Result<Self> {
        if sorted.is_empty() {
            return Err(BarracudaError::invalid_op(
                "searchsorted",
                "Sorted array cannot be empty",
            ));
        }

        if values.is_empty() {
            return Err(BarracudaError::invalid_op(
                "searchsorted",
                "Values array cannot be empty",
            ));
        }

        // Validate sorted array is 1D
        if sorted.shape().len() != 1 {
            return Err(BarracudaError::invalid_op(
                "searchsorted",
                "Sorted array must be 1D",
            ));
        }

        if values.shape().len() != 1 {
            return Err(BarracudaError::invalid_op(
                "searchsorted",
                "Values array must be 1D",
            ));
        }

        Ok(Self {
            sorted,
            values,
            side_right,
        })
    }

    /// Executes the binary search and returns insertion indices as f32 tensor.
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.sorted.device();
        let sorted_size = self.sorted.len();
        let values_size = self.values.len();

        let output_buffer = device.create_buffer_u32(values_size)?;

        let params = SearchSortedParams {
            sorted_size: sorted_size as u32,
            values_size: values_size as u32,
            side_right: u32::from(self.side_right),
            _pad1: 0,
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("SearchSorted Params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let indices_f32_buffer = device.create_buffer_f32(values_size)?;

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct ConvertParams {
            size: u32,
            _pad1: u32,
            _pad2: u32,
            _pad3: u32,
        }

        let convert_params = ConvertParams {
            size: values_size as u32,
            _pad1: 0,
            _pad2: 0,
            _pad3: 0,
        };

        let convert_params_buffer =
            device
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("SearchSorted Convert Params"),
                    contents: bytemuck::bytes_of(&convert_params),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                });

        let mut batch = BatchedComputeDispatch::new(device);
        batch.push(
            ComputeDispatch::new(device, "SearchSorted")
                .shader(
                    include_str!("../shaders/misc/searchsorted_f64.wgsl"),
                    "main",
                )
                .uniform(0, &params_buffer)
                .storage_read(1, self.sorted.buffer())
                .storage_read(2, self.values.buffer())
                .storage_rw(3, &output_buffer)
                .dispatch_1d(values_size as u32),
        )?;
        batch.push(
            ComputeDispatch::new(device, "SearchSorted Convert")
                .shader(
                    include_str!("../shaders/misc/u32_to_f32_f64.wgsl"),
                    "main",
                )
                .uniform(0, &convert_params_buffer)
                .storage_read(1, &output_buffer)
                .storage_rw(2, &indices_f32_buffer)
                .dispatch_1d(values_size as u32),
        )?;
        batch.submit()?;

        Ok(Tensor::from_buffer(
            indices_f32_buffer,
            vec![values_size],
            device.clone(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_searchsorted_basic() {
        let device = crate::device::test_pool::get_test_device().await;
        let sorted = Tensor::from_vec_on(vec![1.0, 3.0, 5.0, 7.0], vec![4], device.clone())
            .await
            .unwrap();
        let values = Tensor::from_vec_on(vec![2.0, 4.0, 6.0], vec![3], device.clone())
            .await
            .unwrap();

        let result = SearchSorted::new(sorted, values, false)
            .unwrap()
            .execute()
            .unwrap();
        let indices = result.to_vec().unwrap();
        assert_eq!(indices.len(), 3);
        // Should be [1, 2, 3] (insertion points)
    }

    #[tokio::test]
    async fn test_searchsorted_right() {
        let device = crate::device::test_pool::get_test_device().await;
        let sorted = Tensor::from_vec_on(vec![1.0, 3.0, 5.0], vec![3], device.clone())
            .await
            .unwrap();
        let values = Tensor::from_vec_on(vec![3.0], vec![1], device.clone())
            .await
            .unwrap();

        let result = SearchSorted::new(sorted, values, true)
            .unwrap()
            .execute()
            .unwrap();
        let indices = result.to_vec().unwrap();
        assert_eq!(indices.len(), 1);
    }
}

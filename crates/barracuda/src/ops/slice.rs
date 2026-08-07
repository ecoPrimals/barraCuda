// SPDX-License-Identifier: AGPL-3.0-or-later
//! Slice operation - Pure WGSL

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::Result;
use crate::tensor::Tensor;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct SliceParams {
    start: u32,
    length: u32,
    _padding: [u32; 2],
}

/// Slice operation — extracts a contiguous subregion from a 1D tensor.
pub struct Slice {
    input: Tensor,
    start: usize,
    length: usize,
}

impl Slice {
    /// Create a slice operation.
    /// # Arguments
    /// * `input` - Input tensor (1D)
    /// * `start` - Start index (inclusive)
    /// * `length` - Number of elements to extract
    #[must_use]
    pub fn new(input: Tensor, start: usize, length: usize) -> Self {
        Self {
            input,
            start,
            length,
        }
    }

    /// Execute slice operation (extract contiguous region from input).
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let output_buffer = device.create_buffer_f32(self.length)?;

        let params = SliceParams {
            start: self.start as u32,
            length: self.length as u32,
            _padding: [0; 2],
        };

        let params_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Slice Params"),
            size: std::mem::size_of::<SliceParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        device
            .queue
            .write_buffer(&params_buffer, 0, bytemuck::bytes_of(&params));

        ComputeDispatch::new(device, "Slice")
            .shader(include_str!("../shaders/tensor/slice_f64.wgsl"), "main")
            .storage_read(0, self.input.buffer())
            .storage_rw(1, &output_buffer)
            .uniform(2, &params_buffer)
            .dispatch_1d(self.length as u32)
            .submit()?;

        Ok(Tensor::from_buffer(
            output_buffer,
            vec![self.length],
            device.clone(),
        ))
    }
}

impl Tensor {
    /// Extract a contiguous subregion [start..start+length] from this 1D tensor.
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn slice(self, start: usize, length: usize) -> Result<Self> {
        Slice::new(self, start, length).execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_slice_basic() {
        let device = crate::device::test_pool::get_test_device().await;
        let input = Tensor::from_vec_on(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![5], device)
            .await
            .unwrap();
        let result = input.slice(1, 3).unwrap().to_vec().unwrap();

        assert_eq!(result.len(), 3);
        assert!(result.iter().all(|&x| x.is_finite()));
    }

    #[tokio::test]
    async fn test_slice_edge_cases() {
        let device = crate::device::test_pool::get_test_device().await;
        // Slice from start
        let input = Tensor::from_vec_on(vec![1.0, 2.0, 3.0], vec![3], device.clone())
            .await
            .unwrap();
        let result = input.slice(0, 2).unwrap().to_vec().unwrap();
        assert_eq!(result.len(), 2);

        // Single element
        let input = Tensor::from_vec_on(vec![1.0, 2.0, 3.0], vec![3], device.clone())
            .await
            .unwrap();
        let result = input.slice(1, 1).unwrap().to_vec().unwrap();
        assert_eq!(result.len(), 1);

        // Full slice
        let input = Tensor::from_vec_on(vec![1.0, 2.0], vec![2], device.clone())
            .await
            .unwrap();
        let result = input.slice(0, 2).unwrap().to_vec().unwrap();
        assert_eq!(result.len(), 2);
    }

    #[tokio::test]
    async fn test_slice_boundary() {
        let device = crate::device::test_pool::get_test_device().await;
        // Slice to end
        let input = Tensor::from_vec_on(vec![1.0, 2.0, 3.0, 4.0], vec![4], device.clone())
            .await
            .unwrap();
        let result = input.slice(2, 2).unwrap().to_vec().unwrap();
        assert_eq!(result.len(), 2);

        // Large slice
        let input_data: Vec<f32> = (0..100).map(|i| i as f32).collect();
        let input = Tensor::from_vec_on(input_data, vec![100], device.clone())
            .await
            .unwrap();
        let result = input.slice(10, 50).unwrap().to_vec().unwrap();
        assert_eq!(result.len(), 50);
    }

    #[tokio::test]
    async fn test_slice_large_batch() {
        let device = crate::device::test_pool::get_test_device().await;
        // 1000 elements
        let input_data: Vec<f32> = (0..1000).map(|i| i as f32).collect();
        let input = Tensor::from_vec_on(input_data, vec![1000], device)
            .await
            .unwrap();
        let result = input.slice(100, 500).unwrap().to_vec().unwrap();
        assert_eq!(result.len(), 500);
    }

    #[tokio::test]
    async fn test_slice_precision() {
        let device = crate::device::test_pool::get_test_device().await;
        // Verify exact values
        let input = Tensor::from_vec_on(vec![10.0, 20.0, 30.0, 40.0, 50.0], vec![5], device)
            .await
            .unwrap();
        let result = input.slice(2, 2).unwrap().to_vec().unwrap();

        assert_eq!(result.len(), 2);
        assert!(result.iter().all(|&x| x.is_finite()));
    }
}

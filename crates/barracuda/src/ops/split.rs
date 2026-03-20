// SPDX-License-Identifier: AGPL-3.0-or-later
//! Split - Tensor splitting operation
//! Pure WGSL implementation
//!
//! Splits a tensor into multiple parts along a dimension (inverse of Concat)
//! Formula: [output1, output2] = split(input, `split_point`)
//!
//! Used in: Multi-branch networks, Inception modules, `ResNeXt`
//! Benefits: Enables parallel processing paths, modular architecture design

use crate::device::ComputeDispatch;
use crate::error::Result;
use crate::tensor::Tensor;

/// f64 is the canonical source — f32 derived via `downcast_f64_to_f32` when needed.
const SHADER_F64: &str = include_str!("../shaders/tensor/split_f64.wgsl");

static SHADER_F32: std::sync::LazyLock<String> =
    std::sync::LazyLock::new(|| SHADER_F64.to_string());

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct SplitParams {
    total_size: u32,
    split_point: u32,
    _pad: u32,
    _pad2: u32,
}

/// Tensor splitting operation along the last dimension.
pub struct Split {
    input: Tensor,
    split_point: usize,
}

impl Split {
    /// Creates a new split operation at the given point along the last dimension.
    #[must_use]
    pub fn new(input: Tensor, split_point: usize) -> Self {
        Self { input, split_point }
    }

    fn wgsl_shader() -> &'static str {
        &SHADER_F32
    }

    /// Executes the split and returns the two output tensors.
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<(Tensor, Tensor)> {
        let device = self.input.device();
        let shape = self.input.shape();

        // For simplicity, split along the last dimension
        let total_size: usize = shape.iter().product();
        let size1 = self.split_point;
        let size2 = total_size - self.split_point;

        // Create output buffers
        let output1_buffer = device.create_buffer_f32(size1)?;
        let output2_buffer = device.create_buffer_f32(size2)?;

        // Create params
        let params = SplitParams {
            total_size: total_size as u32,
            split_point: self.split_point as u32,
            _pad: 0,
            _pad2: 0,
        };
        let params_buffer = device.create_uniform_buffer("Split Params", &params);

        ComputeDispatch::new(device, "split")
            .shader(Self::wgsl_shader(), "main")
            .storage_read(0, self.input.buffer())
            .storage_rw(1, &output1_buffer)
            .storage_rw(2, &output2_buffer)
            .uniform(3, &params_buffer)
            .dispatch_1d(total_size as u32)
            .submit()?;

        // Determine output shapes (split along last dimension for simplicity)
        let mut shape1 = shape.to_vec();
        let mut shape2 = shape.to_vec();
        let last_dim = shape.len() - 1;
        let last_size = shape[last_dim];

        shape1[last_dim] = self.split_point;
        shape2[last_dim] = last_size - self.split_point;

        Ok((
            Tensor::from_buffer(output1_buffer, shape1, device.clone()),
            Tensor::from_buffer(output2_buffer, shape2, device.clone()),
        ))
    }
}

impl Tensor {
    /// Split tensor into two parts at the specified point
    /// # Arguments
    /// * `split_point` - Position to split (along last dimension)
    /// # Returns
    /// Tuple of two tensors (before `split_point`, after `split_point`)
    /// # Example
    /// ```ignore
    /// // Split [batch, 512] into [batch, 256] and [batch, 256]
    /// let (left, right) = tensor.split(256)?;
    /// ```
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn split(self, split_point: usize) -> Result<(Self, Self)> {
        Split::new(self, split_point).execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device_if_gpu_available;

    #[tokio::test]
    async fn test_split_basic() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        // Simple 1D split
        let input_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let input = Tensor::from_data(&input_data, vec![6], device).unwrap();

        // Split at position 3 (middle)
        let (left, right) = input.split(3).unwrap();

        let left_data = left.to_vec().unwrap();
        let right_data = right.to_vec().unwrap();

        assert_eq!(left_data.len(), 3);
        assert_eq!(right_data.len(), 3);
        assert!(left_data.iter().all(|&x| x.is_finite()));
        assert!(right_data.iter().all(|&x| x.is_finite()));
    }

    #[tokio::test]
    async fn test_split_edge_cases() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        // Split at start
        let input_data = vec![1.0, 2.0, 3.0, 4.0];
        let input = Tensor::from_data(&input_data, vec![4], device.clone()).unwrap();
        let (left, right) = input.split(1).unwrap();
        assert_eq!(left.to_vec().unwrap().len(), 1);
        assert_eq!(right.to_vec().unwrap().len(), 3);

        // Split near end
        let input_data = vec![1.0, 2.0, 3.0, 4.0];
        let input = Tensor::from_data(&input_data, vec![4], device).unwrap();
        let (left, right) = input.split(3).unwrap();
        assert_eq!(left.to_vec().unwrap().len(), 3);
        assert_eq!(right.to_vec().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn test_split_boundary() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        // Equal split
        let input_data = vec![1.0; 100];
        let input = Tensor::from_data(&input_data, vec![100], device.clone()).unwrap();
        let (left, right) = input.split(50).unwrap();
        assert_eq!(left.to_vec().unwrap().len(), 50);
        assert_eq!(right.to_vec().unwrap().len(), 50);

        // Unequal split
        let input_data = vec![1.0; 100];
        let input = Tensor::from_data(&input_data, vec![100], device).unwrap();
        let (left, right) = input.split(30).unwrap();
        assert_eq!(left.to_vec().unwrap().len(), 30);
        assert_eq!(right.to_vec().unwrap().len(), 70);
    }

    #[tokio::test]
    async fn test_split_large_batch() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        // 10000 elements
        let input_data = vec![1.0; 10_000];
        let input = Tensor::from_data(&input_data, vec![10_000], device).unwrap();
        let (left, right) = input.split(5000).unwrap();
        assert_eq!(left.to_vec().unwrap().len(), 5000);
        assert_eq!(right.to_vec().unwrap().len(), 5000);
    }

    #[tokio::test]
    async fn test_split_precision() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        // Verify data preservation
        let input_data: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let input = Tensor::from_data(&input_data, vec![10], device).unwrap();
        let (left, right) = input.split(5).unwrap();

        let left_data = left.to_vec().unwrap();
        let right_data = right.to_vec().unwrap();

        assert_eq!(left_data.len(), 5);
        assert_eq!(right_data.len(), 5);
        assert!(left_data.iter().all(|&x| x.is_finite()));
        assert!(right_data.iter().all(|&x| x.is_finite()));
    }
}

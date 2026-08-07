// SPDX-License-Identifier: AGPL-3.0-or-later
//! Bucketize - Assign values to bins - Pure WGSL
//!
//! **Deep Debt Principles**:
//! - ✅ Pure WGSL implementation (GPU-optimized)
//! - ✅ Safe Rust wrapper (no unsafe code)
//! - ✅ Hardware-agnostic via WebGPU
//! - ✅ Complete implementation (production-ready)
//!
//! ## Algorithm
//!
//! Maps each value to a bucket index based on boundaries:
//! ```text
//! Input:  [0.5, 1.5, 2.5, 3.5]
//! Boundaries: [1.0, 2.0, 3.0]
//! Output: [0, 1, 2, 3]  (bucket indices)
//! ```

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::Result;
use crate::tensor::Tensor;

/// Assign values to buckets based on boundary thresholds.
pub struct Bucketize {
    input: Tensor,
    boundaries: Vec<f32>,
}

impl Bucketize {
    /// Create a bucketize operation. Boundaries must be sorted.
    #[must_use]
    pub fn new(input: Tensor, boundaries: Vec<f32>) -> Self {
        Self { input, boundaries }
    }

    /// Execute bucketize on GPU. Returns bucket indices as f32.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let input_size = self.input.len();
        let num_boundaries = self.boundaries.len();

        // Create output buffer (u32 for bucket indices)
        let output_buffer = device.create_buffer_u32(input_size)?;

        // Create boundaries buffer
        let boundaries_buffer =
            device.create_storage_buffer("Data", bytemuck::cast_slice(&self.boundaries));

        // Create params buffer
        let params_data = [input_size as u32, num_boundaries as u32];
        let params_buffer = device.create_uniform_buffer("Params", &params_data);

        ComputeDispatch::new(device, "Bucketize")
            .shader(include_str!("../shaders/misc/bucketize_f64.wgsl"), "main")
            .storage_read(0, self.input.buffer())
            .storage_read(1, &boundaries_buffer)
            .storage_rw(2, &output_buffer)
            .uniform(3, &params_buffer)
            .dispatch_1d(input_size as u32)
            .submit()?;

        // Read u32 buffer and convert to f32 for Tensor compatibility
        let u32_data = crate::utils::read_buffer_u32(device, &output_buffer, input_size)?;
        let f32_data: Vec<f32> = u32_data.iter().map(|&x| x as f32).collect();

        Ok(Tensor::new(
            f32_data,
            self.input.shape().to_vec(),
            device.clone(),
        ))
    }
}

impl Tensor {
    /// Map values to bucket indices based on boundaries.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn bucketize_wgsl(self, boundaries: Vec<f32>) -> Result<Self> {
        Bucketize::new(self, boundaries).execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_bucketize_basic() {
        let device = crate::device::test_pool::get_test_device().await;
        let input_data = vec![0.5, 1.5, 2.5, 3.5];
        let input = Tensor::from_vec_on(input_data, vec![4], device)
            .await
            .unwrap();

        let boundaries = vec![1.0, 2.0, 3.0];
        let result = input.bucketize_wgsl(boundaries).unwrap();
        let output_f32 = result.to_vec().unwrap();
        let output: Vec<u32> = output_f32.iter().map(|&x| x as u32).collect();

        // 0.5 < 1.0 → bucket 0
        // 1.0 <= 1.5 < 2.0 → bucket 1
        // 2.0 <= 2.5 < 3.0 → bucket 2
        // 3.5 >= 3.0 → bucket 3
        assert_eq!(output, vec![0, 1, 2, 3]);
    }

    #[tokio::test]
    async fn test_bucketize_edge_cases() {
        let device = crate::device::test_pool::get_test_device().await;
        let input_data = vec![0.0, 1.0, 2.0, 10.0];
        let input = Tensor::from_vec_on(input_data, vec![4], device)
            .await
            .unwrap();

        let boundaries = vec![1.0, 2.0];
        let result = input.bucketize_wgsl(boundaries).unwrap();
        let output_f32 = result.to_vec().unwrap();
        let output: Vec<u32> = output_f32.iter().map(|&x| x as u32).collect();

        // 0.0 < 1.0 → bucket 0
        // 1.0 (boundary) → bucket 1
        // 2.0 (boundary) → bucket 2
        // 10.0 >= 2.0 → bucket 2
        assert_eq!(output, vec![0, 1, 2, 2]);
    }
}

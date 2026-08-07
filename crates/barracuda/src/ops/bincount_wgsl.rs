// SPDX-License-Identifier: AGPL-3.0-or-later
//! Bincount - Count occurrences of values - Pure WGSL
//!
//! **Deep Debt Principles**:
//! - ✅ Pure WGSL implementation (GPU-optimized with atomics)
//! - ✅ Safe Rust wrapper (no unsafe code)
//! - ✅ Hardware-agnostic via WebGPU
//! - ✅ Complete implementation (production-ready)
//!
//! ## Algorithm
//!
//! Counts occurrences of each non-negative integer value:
//! ```text
//! Input:  [0, 1, 1, 2, 2, 2]
//! Output: [1, 2, 3]  (counts for values 0, 1, 2)
//! ```
//! Uses atomic operations for thread-safe GPU counting.

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::Result;
use crate::tensor::Tensor;

/// Count occurrences of non-negative integer values.
pub struct Bincount {
    input: Tensor,
    num_bins: Option<usize>,
}

impl Bincount {
    /// Create a bincount operation. `num_bins` defaults to 256 if None.
    #[must_use]
    pub fn new(input: Tensor, num_bins: Option<usize>) -> Self {
        Self { input, num_bins }
    }

    /// Execute bincount on GPU. Returns counts as f32.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let input_size = self.input.len();

        const DEFAULT_NUM_BINS: usize = 256;
        let num_bins = self.num_bins.unwrap_or(DEFAULT_NUM_BINS);

        // Create output buffer initialized to zeros
        let output_buffer = device.create_buffer_u32_zeros(num_bins)?;

        // Create params buffer
        let params_data = [input_size as u32, num_bins as u32];
        let params_buffer = device.create_uniform_buffer("Params", &params_data);
        let input_buffer = self.input.buffer();

        ComputeDispatch::new(device, "Bincount")
            .shader(include_str!("../shaders/misc/bincount_f64.wgsl"), "main")
            .storage_read(0, input_buffer)
            .storage_rw(1, &output_buffer)
            .uniform(2, &params_buffer)
            .dispatch_1d(input_size as u32)
            .submit()?;

        // Read u32 buffer and convert to f32 for Tensor compatibility
        let u32_data = crate::utils::read_buffer_u32(device, &output_buffer, num_bins)?;
        let f32_data: Vec<f32> = u32_data.iter().map(|&x| x as f32).collect();

        Ok(Tensor::new(f32_data, vec![num_bins], device.clone()))
    }
}

impl Tensor {
    /// Count value occurrences into bins. Input values as f32 (cast from u32).
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn bincount_wgsl(self, num_bins: Option<usize>) -> Result<Self> {
        Bincount::new(self, num_bins).execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_bincount_basic() {
        let device = crate::device::test_pool::get_test_device().await;
        let input_data = [0u32, 1, 1, 2, 2, 2];
        // Convert u32 to f32 for Tensor
        let input_f32: Vec<f32> = input_data.iter().map(|&x| x as f32).collect();
        let input = Tensor::from_vec_on(input_f32, vec![6], device.clone())
            .await
            .unwrap();

        let result = input.bincount_wgsl(Some(3)).unwrap();
        let output_f32 = result.to_vec().unwrap();
        let output: Vec<u32> = output_f32.iter().map(|&x| x as u32).collect();

        // Value 0 appears 1 time
        // Value 1 appears 2 times
        // Value 2 appears 3 times
        assert_eq!(output[0], 1);
        assert_eq!(output[1], 2);
        assert_eq!(output[2], 3);
    }

    #[tokio::test]
    async fn test_bincount_sparse() {
        let device = crate::device::test_pool::get_test_device().await;
        let input_data = [0u32, 0, 5, 5, 5];
        // Convert u32 to f32 for Tensor
        let input_f32: Vec<f32> = input_data.iter().map(|&x| x as f32).collect();
        let input = Tensor::from_vec_on(input_f32, vec![5], device.clone())
            .await
            .unwrap();

        let result = input.bincount_wgsl(Some(10)).unwrap();
        let output_f32 = result.to_vec().unwrap();
        let output: Vec<u32> = output_f32.iter().map(|&x| x as u32).collect();

        assert_eq!(output[0], 2); // 0 appears twice
        assert_eq!(output[5], 3); // 5 appears three times
        assert_eq!(output[1], 0); // 1 never appears
    }
}

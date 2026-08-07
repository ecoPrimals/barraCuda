// SPDX-License-Identifier: AGPL-3.0-or-later
//! GLU - Gated Linear Unit - Pure WGSL
//!
//! **Deep Debt Principles**:
//! - ✅ Pure WGSL implementation (GPU-optimized)
//! - ✅ Safe Rust wrapper (no unsafe code)
//! - ✅ Hardware-agnostic via WebGPU
//! - ✅ Complete implementation (production-ready)
//!
//! ## Algorithm
//!
//! Gated Linear Unit:
//! ```text
//! Input x split into two halves: [a, b]
//! glu(x) = a ⊙ sigmoid(b)
//!
//! Output size is half of input size
//! Used in language models and transformers
//! ```

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::Result;
use crate::tensor::Tensor;

/// Gated Linear Unit: output = a ⊙ sigmoid(b) where input = [a, b].
pub struct GLU {
    input: Tensor,
}

impl GLU {
    /// Create a GLU operation. Input size must be even.
    #[must_use]
    pub fn new(input: Tensor) -> Self {
        Self { input }
    }

    /// Execute GLU on GPU. Output is half the input size.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let size = self.input.len();

        // Input must be even-sized for splitting
        if !size.is_multiple_of(2) {
            return Err(crate::error::BarracudaError::InvalidShape {
                expected: vec![size / 2 * 2],
                actual: vec![size],
            });
        }

        let half_size = size / 2;
        let output_buffer = device.create_buffer_f32(half_size)?;

        let params_data = [
            size as u32,
            0u32, // split_dim: last dim (standard GLU behavior)
        ];
        let params_buffer = device.create_uniform_buffer("Params", &params_data);
        let input_buffer = self.input.buffer();

        ComputeDispatch::new(device, "GLU")
            .shader(include_str!("../shaders/activation/glu_f64.wgsl"), "main")
            .storage_read(0, &input_buffer)
            .storage_rw(1, &output_buffer)
            .uniform(2, &params_buffer)
            .dispatch_1d(half_size as u32)
            .submit()?;

        // Output shape is half the input size
        let mut output_shape = self.input.shape().to_vec();
        if let Some(last) = output_shape.last_mut() {
            *last = half_size;
        }

        Ok(Tensor::from_buffer(
            output_buffer,
            output_shape,
            device.clone(),
        ))
    }
}

impl Tensor {
    /// Apply Gated Linear Unit. Splits input in half; output = `first_half` ⊙ `sigmoid(second_half)`.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn glu_wgsl(self) -> Result<Self> {
        GLU::new(self).execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_glu() {
        let device = crate::device::test_pool::get_test_device().await;
        // 6 elements: first 3 for a, last 3 for b
        let input_data = vec![1.0, 2.0, 3.0, 0.0, 0.0, 0.0];
        let input = Tensor::from_vec_on(input_data, vec![6], device)
            .await
            .unwrap();

        let result = input.glu_wgsl().unwrap();
        let output = result.to_vec().unwrap();

        // Output should be 3 elements (half of input)
        assert_eq!(output.len(), 3);
        // glu([a, b]) = a * sigmoid(b)
        // sigmoid(0) = 0.5, so output ≈ [0.5, 1.0, 1.5]
        assert!((output[0] - 0.5).abs() < 0.1);
    }
}

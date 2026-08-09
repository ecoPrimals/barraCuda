// SPDX-License-Identifier: AGPL-3.0-or-later
//! BETA — Beta function B(a,b) — Pure WGSL via ComputeDispatch builder.

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::Result;
use crate::tensor::Tensor;

/// Beta function B(a,b) = Γ(a)Γ(b)/Γ(a+b)
pub struct Beta {
    input: Tensor, // Interleaved pairs [a₀, b₀, a₁, b₁, ...]
}

impl Beta {
    /// Create new Beta function operation
    /// Input tensor must have even length: [a₀, b₀, a₁, b₁, ...]
    #[must_use]
    pub fn new(input: Tensor) -> Self {
        Self { input }
    }

    /// Execute Beta function B(a,b) = Γ(a)Γ(b)/Γ(a+b) on input pairs.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let input_size: usize = self.input.shape().iter().product();
        let output_size = input_size / 2;
        let input_buffer = self.input.buffer();
        let output_buffer = device.create_buffer_f32(output_size)?;

        let params = [output_size as u32];
        let params_buffer = device.create_uniform_buffer("Beta Params", &params);

        ComputeDispatch::new(device, "beta")
            .shader(include_str!("../shaders/special/beta.wgsl"), "main")
            .storage_read(0, input_buffer)
            .storage_rw(1, &output_buffer)
            .uniform(2, &params_buffer)
            .dispatch_1d(output_size as u32)
            .submit()?;

        Ok(Tensor::from_buffer(
            output_buffer,
            vec![output_size],
            device.clone(),
        ))
    }
}

impl Tensor {
    /// Compute Beta function B(a,b) for interleaved pairs
    /// Input: [a₀, b₀, a₁, b₁, ...]
    /// Output: [B(a₀,b₀), B(a₁,b₁), ...]
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn beta(self) -> Result<Self> {
        Beta::new(self).execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_beta_1_1() {
        let device = crate::device::test_pool::get_test_device().await;
        let input = Tensor::new(vec![1.0, 1.0], vec![2], device);
        let output = input.beta().unwrap();
        let result = output.to_vec().unwrap();
        assert!(
            (result[0] - 1.0).abs() < 0.01,
            "B(1,1) = {}, expected 1",
            result[0]
        );
    }

    #[tokio::test]
    async fn test_beta_2_2() {
        let device = crate::device::test_pool::get_test_device().await;
        let input = Tensor::new(vec![2.0, 2.0], vec![2], device);
        let output = input.beta().unwrap();
        let result = output.to_vec().unwrap();
        let expected = 1.0 / 6.0;
        assert!(
            (result[0] - expected).abs() < 0.01,
            "B(2,2) = {}, expected {}",
            result[0],
            expected
        );
    }

    #[tokio::test]
    async fn test_beta_multiple() {
        let device = crate::device::test_pool::get_test_device().await;
        let input = Tensor::new(vec![1.0, 2.0, 2.0, 3.0, 3.0, 4.0], vec![6], device);
        let output = input.beta().unwrap();
        let result = output.to_vec().unwrap();

        assert!(
            (result[0] - 0.5).abs() < 0.01,
            "B(1,2) = {}, expected 0.5",
            result[0]
        );
        assert!(
            (result[1] - 1.0 / 12.0).abs() < 0.01,
            "B(2,3) = {}, expected {}",
            result[1],
            1.0 / 12.0
        );
        assert!(
            (result[2] - 1.0 / 60.0).abs() < 0.01,
            "B(3,4) = {}, expected {}",
            result[2],
            1.0 / 60.0
        );
    }
}

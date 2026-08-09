// SPDX-License-Identifier: AGPL-3.0-or-later
//! Unsqueeze operation - Add dimensions of size 1
//! Pure WGSL implementation

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::Result;
use crate::tensor::Tensor;

/// Add a dimension of size 1 at the given axis.
pub struct Unsqueeze {
    input: Tensor,
    axis: usize,
}

impl Unsqueeze {
    /// Create an unsqueeze operation inserting a dimension at `axis`.
    #[must_use]
    pub fn new(input: Tensor, axis: usize) -> Self {
        Self { input, axis }
    }

    /// Execute the unsqueeze operation on GPU.
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let size = self.input.len();
        let output_buffer = device.create_buffer_f32(size)?;

        ComputeDispatch::new(device, "Unsqueeze")
            .shader(include_str!("../shaders/tensor/unsqueeze_f64.wgsl"), "main")
            .storage_read(0, self.input.buffer())
            .storage_rw(1, &output_buffer)
            .dispatch_1d(size as u32)
            .submit()?;

        // Compute new shape by inserting dimension of size 1 at axis
        let mut new_shape = self.input.shape().to_vec();
        new_shape.insert(self.axis, 1);

        Ok(Tensor::from_buffer(
            output_buffer,
            new_shape,
            device.clone(),
        ))
    }
}

impl Tensor {
    /// Add a dimension of size 1 at the given axis.
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn unsqueeze(self, axis: usize) -> Result<Self> {
        Unsqueeze::new(self, axis).execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_unsqueeze_basic() {
        let device = crate::device::test_pool::get_test_device().await;

        // Shape [3] should become [1, 3] when unsqueeze at axis 0
        let input = Tensor::from_vec_on(vec![1.0, 2.0, 3.0], vec![3], device)
            .await
            .unwrap();
        let result = input.unsqueeze(0).unwrap();

        assert_eq!(result.shape(), &[1, 3]);
        let data = result.to_vec().unwrap();
        assert!((data[0] - 1.0).abs() < 1e-5);
        assert!((data[1] - 2.0).abs() < 1e-5);
        assert!((data[2] - 3.0).abs() < 1e-5);
    }
}

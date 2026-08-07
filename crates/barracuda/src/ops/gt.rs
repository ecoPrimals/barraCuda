// SPDX-License-Identifier: AGPL-3.0-or-later
//! Gt operation - Greater than comparison
//! Pure WGSL implementation

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::Result;
use crate::tensor::Tensor;

/// Element-wise greater-than comparison: output = lhs > rhs.
pub struct Gt {
    lhs: Tensor,
    rhs: Tensor,
}

impl Gt {
    /// Create a greater-than comparison operation.
    #[must_use]
    pub fn new(lhs: Tensor, rhs: Tensor) -> Self {
        Self { lhs, rhs }
    }
    /// Execute greater-than comparison on GPU.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.lhs.device();
        let size = self.lhs.len();
        let output_buffer = device.create_buffer_f32(size)?;

        ComputeDispatch::new(device, "Gt")
            .shader(include_str!("../shaders/misc/gt_f64.wgsl"), "main")
            .storage_read(0, self.lhs.buffer())
            .storage_read(1, self.rhs.buffer())
            .storage_rw(2, &output_buffer)
            .dispatch_1d(size as u32)
            .submit()?;

        Ok(Tensor::from_buffer(
            output_buffer,
            self.lhs.shape().to_vec(),
            device.clone(),
        ))
    }
}

impl Tensor {
    /// Element-wise greater-than comparison. Returns 1.0 where true, 0.0 where false.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn gt(self, other: &Self) -> Result<Self> {
        Gt::new(self, other.clone()).execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_gt_basic() {
        let device = crate::device::test_pool::get_test_device().await;

        let a = Tensor::from_vec_on(vec![1.0, 3.0, 2.0], vec![3], device.clone())
            .await
            .unwrap();
        let b = Tensor::from_vec_on(vec![2.0, 2.0, 2.0], vec![3], device)
            .await
            .unwrap();
        let result = a.gt(&b).unwrap().to_vec().unwrap();
        assert!((result[0] - 0.0).abs() < 1e-5); // 1 > 2? no
        assert!((result[1] - 1.0).abs() < 1e-5); // 3 > 2? yes
        assert!((result[2] - 0.0).abs() < 1e-5); // 2 > 2? no
    }
}

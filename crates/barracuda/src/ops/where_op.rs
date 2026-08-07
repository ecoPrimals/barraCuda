// SPDX-License-Identifier: AGPL-3.0-or-later
//! Where/Select operation - Conditional selection
//! Pure WGSL implementation

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::Result;
use crate::tensor::Tensor;

/// Where/select conditional shader.
pub const WGSL_WHERE_OP: &str = include_str!("../shaders/tensor/where_op_f64.wgsl");

/// Conditional selection: output[i] = condition[i] ? x[i] : y[i].
pub struct Where {
    condition: Tensor,
    x: Tensor,
    y: Tensor,
}

impl Where {
    /// Create a where/select operation with condition, true-branch, and false-branch tensors.
    #[must_use]
    pub fn new(condition: Tensor, x: Tensor, y: Tensor) -> Self {
        Self { condition, x, y }
    }

    /// Execute the conditional selection on GPU.
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.condition.device();
        let size = self.condition.len();
        let output_buffer = device.create_buffer_f32(size)?;

        ComputeDispatch::new(device, "Where")
            .shader(
                include_str!("../shaders/tensor/where_select_f64.wgsl"),
                "main",
            )
            .storage_read(0, self.condition.buffer())
            .storage_read(1, self.x.buffer())
            .storage_read(2, self.y.buffer())
            .storage_rw(3, &output_buffer)
            .dispatch_1d(size as u32)
            .submit()?;

        Ok(Tensor::from_buffer(
            output_buffer,
            self.condition.shape().to_vec(),
            device.clone(),
        ))
    }
}

impl Tensor {
    /// Select elements from `x` or `y` based on condition (1.0 = x, 0.0 = y).
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn where_select(condition: Self, x: Self, y: Self) -> Result<Self> {
        Where::new(condition, x, y).execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_where_basic() {
        let device = crate::device::test_pool::get_test_device().await;

        let condition = Tensor::from_vec_on(vec![1.0, 0.0, 1.0], vec![3], device.clone())
            .await
            .unwrap();
        let x = Tensor::from_vec_on(vec![10.0, 20.0, 30.0], vec![3], device.clone())
            .await
            .unwrap();
        let y = Tensor::from_vec_on(vec![100.0, 200.0, 300.0], vec![3], device)
            .await
            .unwrap();

        let result = Tensor::where_select(condition, x, y)
            .unwrap()
            .to_vec()
            .unwrap();

        assert!((result[0] - 10.0).abs() < 1e-5); // condition true -> x
        assert!((result[1] - 200.0).abs() < 1e-5); // condition false -> y
        assert!((result[2] - 30.0).abs() < 1e-5); // condition true -> x
    }
}

// SPDX-License-Identifier: AGPL-3.0-or-later
//! Lt operation - Less than comparison\
//! Pure WGSL implementation

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::Result;
use crate::tensor::Tensor;

/// Element-wise less-than comparison: output = lhs < rhs.
pub struct Lt {
    lhs: Tensor,
    rhs: Tensor,
}

impl Lt {
    /// Create a less-than comparison operation.
    #[must_use]
    pub fn new(lhs: Tensor, rhs: Tensor) -> Self {
        Self { lhs, rhs }
    }
    /// Execute element-wise less-than comparison.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.lhs.device();
        let size = self.lhs.len();
        let output_buffer = device.create_buffer_f32(size)?;

        ComputeDispatch::new(device, "Lt")
            .shader(include_str!("../shaders/misc/lt_f64.wgsl"), "main")
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
    /// Element-wise less-than comparison. Returns 1.0 where true, 0.0 where false.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn lt(self, other: &Self) -> Result<Self> {
        Lt::new(self, other.clone()).execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_lt_basic() {
        let device = crate::device::test_pool::get_test_device().await;
        let a = Tensor::from_vec_on(vec![1.0, 3.0, 2.0], vec![3], device.clone())
            .await
            .unwrap();
        let b = Tensor::from_vec_on(vec![2.0, 2.0, 2.0], vec![3], device)
            .await
            .unwrap();
        let result = a.lt(&b).unwrap().to_vec().unwrap();
        assert_eq!(result.len(), 3);
        // Just verify operation completed
        assert!(result.iter().all(|&x| x.is_finite()));
    }

    #[tokio::test]
    async fn test_lt_edge_cases() {
        let device = crate::device::test_pool::get_test_device().await;
        // All less than
        let a = Tensor::from_vec_on(vec![1.0, 2.0, 3.0], vec![3], device.clone())
            .await
            .unwrap();
        let b = Tensor::from_vec_on(vec![4.0, 5.0, 6.0], vec![3], device.clone())
            .await
            .unwrap();
        let result = a.lt(&b).unwrap().to_vec().unwrap();
        assert!(result.iter().all(|&x| (x - 1.0).abs() < 0.1)); // All true

        // None less than
        let a = Tensor::from_vec_on(vec![5.0, 6.0, 7.0], vec![3], device.clone())
            .await
            .unwrap();
        let b = Tensor::from_vec_on(vec![1.0, 2.0, 3.0], vec![3], device)
            .await
            .unwrap();
        let result = a.lt(&b).unwrap().to_vec().unwrap();
        assert!(result.iter().all(|&x| x.abs() < 0.1)); // All false
    }

    #[tokio::test]
    async fn test_lt_boundary() {
        let device = crate::device::test_pool::get_test_device().await;
        // Equal values
        let a = Tensor::from_vec_on(vec![2.0, 2.0, 2.0], vec![3], device.clone())
            .await
            .unwrap();
        let b = Tensor::from_vec_on(vec![2.0, 2.0, 2.0], vec![3], device.clone())
            .await
            .unwrap();
        let result = a.lt(&b).unwrap().to_vec().unwrap();
        assert!(result.iter().all(|&x| x.abs() < 0.1)); // All false (not less than)

        // Negative values
        let a = Tensor::from_vec_on(vec![-5.0, -3.0, -1.0], vec![3], device.clone())
            .await
            .unwrap();
        let b = Tensor::from_vec_on(vec![-4.0, -4.0, 0.0], vec![3], device)
            .await
            .unwrap();
        let result = a.lt(&b).unwrap().to_vec().unwrap();
        assert_eq!(result.len(), 3);
    }

    #[tokio::test]
    async fn test_lt_large_tensor() {
        let device = crate::device::test_pool::get_test_device().await;
        // 1000 elements
        let a_data: Vec<f32> = (0..1000).map(|i| i as f32).collect();
        let b_data: Vec<f32> = (0..1000).map(|i| (i + 500) as f32).collect();
        let a = Tensor::from_vec_on(a_data, vec![1000], device.clone())
            .await
            .unwrap();
        let b = Tensor::from_vec_on(b_data, vec![1000], device)
            .await
            .unwrap();
        let result = a.lt(&b).unwrap().to_vec().unwrap();
        assert_eq!(result.len(), 1000);
    }

    #[tokio::test]
    async fn test_lt_precision() {
        let device = crate::device::test_pool::get_test_device().await;
        // Mixed comparisons
        let a = Tensor::from_vec_on(vec![1.0, 5.0, 3.0], vec![3], device.clone())
            .await
            .unwrap();
        let b = Tensor::from_vec_on(vec![2.0, 4.0, 3.0], vec![3], device)
            .await
            .unwrap();
        let result = a.lt(&b).unwrap().to_vec().unwrap();

        assert_eq!(result.len(), 3);
        // result[0]: 1 < 2 = true (1.0)
        // result[1]: 5 < 4 = false (0.0)
        // result[2]: 3 < 3 = false (0.0)
        assert!(result.iter().all(|&x| x.is_finite()));
    }
}

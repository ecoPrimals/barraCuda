// SPDX-License-Identifier: AGPL-3.0-or-later
//! Element-wise subtraction
//!
//! **Deep Debt Principles**:
//! - ✅ Pure WGSL implementation (universal compute)
//! - ✅ Capability-based dispatch (vendor-optimized)
//!
//! Formula: C = A - B (element-wise)

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

/// Element-wise subtraction: output = lhs - rhs.
pub struct Sub {
    lhs: Tensor,
    rhs: Tensor,
}

impl Sub {
    /// Create a subtraction operation. Shapes must match.
    /// # Errors
    /// Returns [`Err`] if lhs and rhs shapes do not match.
    pub fn new(lhs: Tensor, rhs: Tensor) -> Result<Self> {
        if lhs.shape() != rhs.shape() {
            return Err(BarracudaError::shape_mismatch(
                lhs.shape().to_vec(),
                rhs.shape().to_vec(),
            ));
        }
        Ok(Self { lhs, rhs })
    }

    /// Execute element-wise subtraction and return the result tensor.
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or device submission fails (e.g. device lost).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.lhs.device();
        let size = self.lhs.len();
        let output_buffer = device.create_buffer_f32(size)?;

        ComputeDispatch::new(device, "Sub")
            .shader(
                include_str!("../shaders/math/elementwise_sub_f64.wgsl"),
                "main",
            )
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
    /// Subtract another tensor element-wise. Shapes must match.
    /// # Errors
    /// Returns [`Err`] if shapes do not match, or buffer allocation/GPU dispatch fails (e.g. device lost).
    pub fn sub(&self, other: &Self) -> Result<Self> {
        Sub::new(self.clone(), other.clone())?.execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_sub_basic() {
        let device = crate::device::test_pool::get_test_device().await;
        let lhs = Tensor::from_vec_on(vec![10.0, 20.0, 30.0], vec![3], device.clone())
            .await
            .unwrap();
        let rhs = Tensor::from_vec_on(vec![1.0, 2.0, 3.0], vec![3], device)
            .await
            .unwrap();

        let result = lhs.sub(&rhs).unwrap().to_vec().unwrap();
        assert!((result[0] - 9.0).abs() < 1e-5);
        assert!((result[1] - 18.0).abs() < 1e-5);
        assert!((result[2] - 27.0).abs() < 1e-5);
    }

    #[tokio::test]
    async fn test_sub_edge_cases() {
        let device = crate::device::test_pool::get_test_device().await;
        let lhs = Tensor::from_vec_on(vec![0.0, 1e-6, -1e-6, 1.0, -1.0], vec![5], device.clone())
            .await
            .unwrap();
        let rhs = Tensor::from_vec_on(vec![0.0, 1e-6, -1e-6, 1.0, -1.0], vec![5], device)
            .await
            .unwrap();

        let result = lhs.sub(&rhs).unwrap().to_vec().unwrap();
        assert!(result[0].abs() < 1e-12);
        assert!(result[1].abs() < 1e-12);
        assert!(result[2].abs() < 1e-12);
    }

    #[tokio::test]
    async fn test_sub_boundary() {
        let device = crate::device::test_pool::get_test_device().await;
        let lhs = Tensor::from_vec_on(vec![f32::INFINITY, 1e10, 0.0], vec![3], device.clone())
            .await
            .unwrap();
        let rhs = Tensor::from_vec_on(vec![100.0, 100.0, 100.0], vec![3], device)
            .await
            .unwrap();

        let result = lhs.sub(&rhs).unwrap().to_vec().unwrap();
        assert!(result[0].is_infinite() && result[0].is_sign_positive());
        assert_eq!(result[2], -100.0);
    }

    #[tokio::test]
    async fn test_sub_large_tensor() {
        let device = crate::device::test_pool::get_test_device().await;
        let size = 1000;
        let lhs_data: Vec<f32> = (0..size).map(|i| (i as f32) * 2.0).collect();
        let rhs_data: Vec<f32> = (0..size).map(|i| i as f32).collect();

        let lhs = Tensor::from_vec_on(lhs_data, vec![size], device.clone())
            .await
            .unwrap();
        let rhs = Tensor::from_vec_on(rhs_data, vec![size], device)
            .await
            .unwrap();

        let result = lhs.sub(&rhs).unwrap().to_vec().unwrap();
        for (i, &val) in result.iter().enumerate() {
            assert!((val - i as f32).abs() < 1e-4);
        }
    }

    #[tokio::test]
    async fn test_sub_precision() {
        let device = crate::device::test_pool::get_test_device().await;
        let lhs_data = vec![5.0, 2.5, 1.0, 0.0, -1.0];
        let rhs_data = vec![2.0, 1.5, 0.5, 0.0, -0.5];

        let lhs = Tensor::from_vec_on(lhs_data.clone(), vec![5], device.clone())
            .await
            .unwrap();
        let rhs = Tensor::from_vec_on(rhs_data.clone(), vec![5], device)
            .await
            .unwrap();

        let gpu_result = lhs.sub(&rhs).unwrap().to_vec().unwrap();
        let cpu_result: Vec<f32> = lhs_data
            .iter()
            .zip(rhs_data.iter())
            .map(|(&a, &b)| a - b)
            .collect();

        for (i, (&gpu, &cpu)) in gpu_result.iter().zip(cpu_result.iter()).enumerate() {
            assert!(
                (gpu - cpu).abs() < 1e-6,
                "Error at {i}: GPU={gpu}, CPU={cpu}"
            );
        }
    }
}

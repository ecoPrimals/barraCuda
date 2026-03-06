// SPDX-License-Identifier: AGPL-3.0-or-later
//! Eq operation - Element-wise equality
//! Pure WGSL implementation

use crate::device::ComputeDispatch;
use crate::error::Result;
use crate::tensor::Tensor;

/// f64 is the canonical source — f32 derived via `downcast_f64_to_f32` when needed.
const SHADER_F64: &str = include_str!("../shaders/misc/eq_f64.wgsl");

static SHADER_F32: std::sync::LazyLock<String> =
    std::sync::LazyLock::new(|| crate::shaders::precision::downcast_f64_to_f32(SHADER_F64));

/// Element-wise equality operation (WGSL).
pub struct Eq {
    lhs: Tensor,
    rhs: Tensor,
}

impl Eq {
    /// Create an element-wise equality operation.
    #[must_use]
    pub fn new(lhs: Tensor, rhs: Tensor) -> Self {
        Self { lhs, rhs }
    }

    fn wgsl_shader() -> &'static str {
        &SHADER_F32
    }

    /// Execute element-wise equality and return the output tensor.
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.lhs.device();
        let size = self.lhs.len();
        let output_buffer = device.create_buffer_f32(size)?;

        ComputeDispatch::new(device, "Eq")
            .shader(Self::wgsl_shader(), "main")
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
    /// Element-wise equality (1.0 where equal, 0.0 where not).
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn eq(self, other: &Self) -> Result<Self> {
        Eq::new(self, other.clone()).execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::test_prelude::with_device_retry;

    #[tokio::test]
    async fn test_eq_basic() {
        with_device_retry(|device| async move {
            let a = Tensor::from_vec_on(vec![1.0, 2.0, 3.0], vec![3], device.clone()).await?;
            let b = Tensor::from_vec_on(vec![1.0, 2.1, 3.0], vec![3], device).await?;

            let result = a.eq(&b)?.to_vec()?;
            assert!((result[0] - 1.0).abs() < 1e-5);
            assert!((result[1] - 0.0).abs() < 1e-5);
            assert!((result[2] - 1.0).abs() < 1e-5);
            Ok(())
        })
        .await;
    }

    #[tokio::test]
    async fn test_eq_edge_cases() {
        with_device_retry(|device| async move {
            let a = Tensor::from_vec_on(vec![5.0, 5.0, 5.0], vec![3], device.clone()).await?;
            let b = Tensor::from_vec_on(vec![5.0, 5.0, 5.0], vec![3], device.clone()).await?;
            let result = a.eq(&b)?.to_vec()?;
            assert!(result.iter().all(|&x| (x - 1.0).abs() < 1e-5));

            let a = Tensor::from_vec_on(vec![1.0, 2.0, 3.0], vec![3], device.clone()).await?;
            let b = Tensor::from_vec_on(vec![4.0, 5.0, 6.0], vec![3], device).await?;
            let result = a.eq(&b)?.to_vec()?;
            assert!(result.iter().all(|&x| x.abs() < 1e-5));
            Ok(())
        })
        .await;
    }

    #[tokio::test]
    async fn test_eq_boundary() {
        with_device_retry(|device| async move {
            let a = Tensor::from_vec_on(vec![-1.0, -2.0, -3.0], vec![3], device.clone()).await?;
            let b = Tensor::from_vec_on(vec![-1.0, -2.0, -3.0], vec![3], device.clone()).await?;
            let result = a.eq(&b)?.to_vec()?;
            assert!(result.iter().all(|&x| (x - 1.0).abs() < 1e-5));

            let a = Tensor::from_vec_on(vec![0.0, 0.0, 1.0], vec![3], device.clone()).await?;
            let b = Tensor::from_vec_on(vec![0.0, 0.0, 1.0], vec![3], device).await?;
            let result = a.eq(&b)?.to_vec()?;
            assert!(result.iter().all(|&x| (x - 1.0).abs() < 1e-5));
            Ok(())
        })
        .await;
    }

    #[tokio::test]
    async fn test_eq_large_tensor() {
        with_device_retry(|device| async move {
            let a_data: Vec<f32> = (0..1000).map(|i| i as f32).collect();
            let b_data: Vec<f32> = (0..1000).map(|i| i as f32).collect();

            let a = Tensor::from_vec_on(a_data, vec![1000], device.clone()).await?;
            let b = Tensor::from_vec_on(b_data, vec![1000], device).await?;

            let result = a.eq(&b)?.to_vec()?;
            assert_eq!(result.len(), 1000);
            assert!(result.iter().all(|&x| (x - 1.0).abs() < 1e-5));
            Ok(())
        })
        .await;
    }

    #[tokio::test]
    async fn test_eq_precision() {
        with_device_retry(|device| async move {
            let a = Tensor::from_vec_on(vec![1.0, 2.0, 3.0], vec![3], device.clone()).await?;
            let b = Tensor::from_vec_on(vec![1.0, 2.0, 3.0], vec![3], device.clone()).await?;

            let result = a.eq(&b)?.to_vec()?;
            assert!(result.iter().all(|&x| (x - 1.0).abs() < 1e-5));

            let a = Tensor::from_vec_on(vec![1.0, 2.0, 3.0], vec![3], device.clone()).await?;
            let b = Tensor::from_vec_on(vec![1.0, 5.0, 3.0], vec![3], device).await?;

            let result = a.eq(&b)?.to_vec()?;
            assert!((result[0] - 1.0).abs() < 1e-5);
            assert!(result[1].abs() < 1e-5);
            assert!((result[2] - 1.0).abs() < 1e-5);
            Ok(())
        })
        .await;
    }
}

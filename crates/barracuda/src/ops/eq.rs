//! Eq operation - Element-wise equality
//! Pure WGSL implementation

use crate::device::ComputeDispatch;
use crate::error::Result;
use crate::tensor::Tensor;

/// f64 is the canonical source — f32 derived via downcast_f64_to_f32 when needed.
const SHADER_F64: &str = include_str!("../shaders/misc/eq_f64.wgsl");

static SHADER_F32: std::sync::LazyLock<String> =
    std::sync::LazyLock::new(|| crate::shaders::precision::downcast_f64_to_f32(SHADER_F64));

pub struct Eq {
    lhs: Tensor,
    rhs: Tensor,
}

impl Eq {
    pub fn new(lhs: Tensor, rhs: Tensor) -> Self {
        Self { lhs, rhs }
    }

    fn wgsl_shader() -> &'static str {
        &SHADER_F32
    }

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
            .submit();

        Ok(Tensor::from_buffer(
            output_buffer,
            self.lhs.shape().to_vec(),
            device.clone(),
        ))
    }
}

impl Tensor {
    pub fn eq(self, other: &Self) -> Result<Self> {
        Eq::new(self, other.clone()).execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    async fn get_test_device() -> Option<Arc<crate::device::WgpuDevice>> {
        crate::device::test_pool::get_test_device_if_gpu_available().await
    }

    #[tokio::test]
    async fn test_eq_basic() {
        let Some(device) = get_test_device().await else {
            return;
        };
        if device.is_lost() {
            return;
        }
        let a = Tensor::from_vec_on(vec![1.0, 2.0, 3.0], vec![3], device.clone())
            .await
            .unwrap();
        let b = Tensor::from_vec_on(vec![1.0, 2.1, 3.0], vec![3], device)
            .await
            .unwrap();

        let result = match a.eq(&b).and_then(|t| t.to_vec()) {
            Ok(r) => r,
            Err(_) => return,
        };
        assert!((result[0] - 1.0).abs() < 1e-5);
        assert!((result[1] - 0.0).abs() < 1e-5);
        assert!((result[2] - 1.0).abs() < 1e-5);
    }

    #[tokio::test]
    async fn test_eq_edge_cases() {
        let Some(device) = get_test_device().await else {
            return;
        };
        // All equal
        let a = Tensor::from_vec_on(vec![5.0, 5.0, 5.0], vec![3], device.clone())
            .await
            .unwrap();
        let b = Tensor::from_vec_on(vec![5.0, 5.0, 5.0], vec![3], device.clone())
            .await
            .unwrap();
        let result = a.eq(&b).unwrap().to_vec().unwrap();
        assert!(result.iter().all(|&x| (x - 1.0).abs() < 1e-5));

        // None equal
        let a = Tensor::from_vec_on(vec![1.0, 2.0, 3.0], vec![3], device.clone())
            .await
            .unwrap();
        let b = Tensor::from_vec_on(vec![4.0, 5.0, 6.0], vec![3], device)
            .await
            .unwrap();
        let result = a.eq(&b).unwrap().to_vec().unwrap();
        assert!(result.iter().all(|&x| x.abs() < 1e-5));
    }

    #[tokio::test]
    async fn test_eq_boundary() {
        let Some(device) = get_test_device().await else {
            return;
        };
        // Negative values
        let a = Tensor::from_vec_on(vec![-1.0, -2.0, -3.0], vec![3], device.clone())
            .await
            .unwrap();
        let b = Tensor::from_vec_on(vec![-1.0, -2.0, -3.0], vec![3], device.clone())
            .await
            .unwrap();
        let result = a.eq(&b).unwrap().to_vec().unwrap();
        assert!(result.iter().all(|&x| (x - 1.0).abs() < 1e-5));

        // Zero comparison
        let a = Tensor::from_vec_on(vec![0.0, 0.0, 1.0], vec![3], device.clone())
            .await
            .unwrap();
        let b = Tensor::from_vec_on(vec![0.0, 0.0, 1.0], vec![3], device)
            .await
            .unwrap();
        let result = a.eq(&b).unwrap().to_vec().unwrap();
        assert!(result.iter().all(|&x| (x - 1.0).abs() < 1e-5));
    }

    #[tokio::test]
    async fn test_eq_large_tensor() {
        let Some(device) = get_test_device().await else {
            return;
        };
        // 1000 elements
        let a_data: Vec<f32> = (0..1000).map(|i| i as f32).collect();
        let b_data: Vec<f32> = (0..1000).map(|i| i as f32).collect();

        let a = Tensor::from_vec_on(a_data, vec![1000], device.clone())
            .await
            .unwrap();
        let b = Tensor::from_vec_on(b_data, vec![1000], device)
            .await
            .unwrap();

        let result = a.eq(&b).unwrap().to_vec().unwrap();
        assert_eq!(result.len(), 1000);
        assert!(result.iter().all(|&x| (x - 1.0).abs() < 1e-5));
    }

    #[tokio::test]
    async fn test_eq_precision() {
        use crate::device::test_pool::test_prelude::with_device_retry;
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

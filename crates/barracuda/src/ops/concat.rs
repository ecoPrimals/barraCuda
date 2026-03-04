// SPDX-License-Identifier: AGPL-3.0-or-later
//! Concatenate operation - Pure WGSL

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::Result;
use crate::tensor::Tensor;

/// f64 is the canonical source — f32 derived via downcast_f64_to_f32 when needed.
const SHADER_F64: &str = include_str!("../shaders/tensor/concat_f64.wgsl");

static SHADER_F32: std::sync::LazyLock<String> =
    std::sync::LazyLock::new(|| crate::shaders::precision::downcast_f64_to_f32(SHADER_F64));

pub struct Concat {
    lhs: Tensor,
    rhs: Tensor,
}

impl Concat {
    pub fn new(lhs: Tensor, rhs: Tensor) -> Self {
        Self { lhs, rhs }
    }

    fn wgsl_shader() -> &'static str {
        &SHADER_F32
    }

    pub fn execute(self) -> Result<Tensor> {
        let device = self.lhs.device();
        let size1 = self.lhs.len();
        let size2 = self.rhs.len();
        let output_size = size1 + size2;

        let output_buffer = device.create_buffer_f32(output_size)?;

        ComputeDispatch::new(device, "Concat")
            .shader(Self::wgsl_shader(), "main")
            .storage_read(0, self.lhs.buffer())
            .storage_read(1, self.rhs.buffer())
            .storage_rw(2, &output_buffer)
            .dispatch_1d(output_size as u32)
            .submit();

        Ok(Tensor::from_buffer(
            output_buffer,
            vec![output_size],
            device.clone(),
        ))
    }
}

impl Tensor {
    pub fn concat(self, other: &Self) -> Result<Self> {
        Concat::new(self, other.clone()).execute()
    }
}

#[expect(clippy::unwrap_used, reason = "tests")]
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_concat_basic() {
        let device = crate::device::Auto::new().await.unwrap();
        // device is already Arc from Auto::new()

        let t1 = Tensor::from_vec_on(vec![1.0, 2.0, 3.0], vec![3], device.clone())
            .await
            .unwrap();
        let t2 = Tensor::from_vec_on(vec![4.0, 5.0], vec![2], device)
            .await
            .unwrap();

        let result = t1.concat(&t2).unwrap().to_vec().unwrap();

        assert_eq!(result.len(), 5);
        assert!((result[0] - 1.0).abs() < 1e-5);
        assert!((result[1] - 2.0).abs() < 1e-5);
        assert!((result[2] - 3.0).abs() < 1e-5);
        assert!((result[3] - 4.0).abs() < 1e-5);
        assert!((result[4] - 5.0).abs() < 1e-5);
    }

    #[tokio::test]
    async fn test_concat_edge_cases() {
        let device = crate::device::Auto::new().await.unwrap();
        // device is already Arc from Auto::new()

        // Single element tensors
        let t1 = Tensor::from_vec_on(vec![1.0], vec![1], device.clone())
            .await
            .unwrap();
        let t2 = Tensor::from_vec_on(vec![2.0], vec![1], device.clone())
            .await
            .unwrap();

        let result = t1.concat(&t2).unwrap().to_vec().unwrap();
        assert_eq!(result, vec![1.0, 2.0]);

        // Same size tensors
        let t3 = Tensor::from_vec_on(vec![3.0, 4.0], vec![2], device.clone())
            .await
            .unwrap();
        let t4 = Tensor::from_vec_on(vec![5.0, 6.0], vec![2], device)
            .await
            .unwrap();

        let result = t3.concat(&t4).unwrap().to_vec().unwrap();
        assert_eq!(result.len(), 4);
    }

    #[tokio::test]
    async fn test_concat_boundary() {
        let device = crate::device::Auto::new().await.unwrap();
        // device is already Arc from Auto::new()

        // Different sized tensors
        let t1 = Tensor::from_vec_on(vec![1.0; 10], vec![10], device.clone())
            .await
            .unwrap();
        let t2 = Tensor::from_vec_on(vec![2.0; 5], vec![5], device)
            .await
            .unwrap();

        let result = t1.concat(&t2).unwrap().to_vec().unwrap();
        assert_eq!(result.len(), 15);

        // First 10 should be 1.0
        assert!(result[0..10].iter().all(|&x| (x - 1.0).abs() < 1e-5));
        // Next 5 should be 2.0
        assert!(result[10..15].iter().all(|&x| (x - 2.0).abs() < 1e-5));
    }

    #[tokio::test]
    async fn test_concat_large_tensors() {
        let device = crate::device::Auto::new().await.unwrap();
        // device is already Arc from Auto::new()

        // Large tensors
        let size1 = 1000;
        let size2 = 500;

        let t1 = Tensor::from_vec_on(vec![1.0; size1], vec![size1], device.clone())
            .await
            .unwrap();
        let t2 = Tensor::from_vec_on(vec![2.0; size2], vec![size2], device)
            .await
            .unwrap();

        let result = t1.concat(&t2).unwrap().to_vec().unwrap();

        assert_eq!(result.len(), size1 + size2);
        assert!(result[0..size1].iter().all(|&x| (x - 1.0).abs() < 1e-5));
        assert!(result[size1..].iter().all(|&x| (x - 2.0).abs() < 1e-5));
    }

    #[tokio::test]
    async fn test_concat_precision() {
        let device = crate::device::Auto::new().await.unwrap();
        // device is already Arc from Auto::new()

        // Test with specific values
        let t1 = Tensor::from_vec_on(vec![1.5, 2.5, 3.5], vec![3], device.clone())
            .await
            .unwrap();
        let t2 = Tensor::from_vec_on(vec![4.5, 5.5], vec![2], device)
            .await
            .unwrap();

        let result = t1.concat(&t2).unwrap().to_vec().unwrap();

        assert_eq!(result.len(), 5);
        assert!((result[0] - 1.5).abs() < 1e-5);
        assert!((result[1] - 2.5).abs() < 1e-5);
        assert!((result[2] - 3.5).abs() < 1e-5);
        assert!((result[3] - 4.5).abs() < 1e-5);
        assert!((result[4] - 5.5).abs() < 1e-5);
    }
}

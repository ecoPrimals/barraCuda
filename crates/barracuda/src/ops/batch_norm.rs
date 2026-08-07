// SPDX-License-Identifier: AGPL-3.0-or-later
//! `BatchNorm` operation - Batch normalization
//! Pure WGSL implementation
//!
//! Deep Debt Principles:
//! - ✅ Capability-based dispatch (vendor-optimized workgroups)

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::Result;
use crate::tensor::Tensor;

/// f32 batch norm shader (training mode, running mean/var).
pub const WGSL_BATCHNORM_TRAINING: &str = include_str!("../shaders/norm/batchnorm_f64.wgsl");

/// f32 2D batch norm shader (NCHW format, per-channel stats).
pub const WGSL_BATCH_NORM_2D: &str = include_str!("../shaders/norm/batch_norm2d_f64.wgsl");

/// f64 canonical — per-tensor batch norm (simplified).
pub const SHADER_BATCH_NORM_F64: &str = include_str!("../shaders/norm/batch_norm_f64.wgsl");

/// GPU shader for group normalization (groups within channels).
pub const WGSL_GROUPNORM: &str = include_str!("../shaders/norm/groupnorm_f64.wgsl");

/// GPU shader for instance normalization (per-instance per-channel).
pub const WGSL_INSTANCENORM: &str = include_str!("../shaders/norm/instancenorm_f64.wgsl");

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct BatchNormParams {
    epsilon: f32,
    _padding: [f32; 7],
}

/// Per-tensor batch normalization (WGSL).
pub struct BatchNorm {
    input: Tensor,
    epsilon: f32,
}

impl BatchNorm {
    /// Create batch norm with given epsilon for numerical stability.
    #[must_use]
    pub fn new(input: Tensor, epsilon: f32) -> Self {
        Self { input, epsilon }
    }

    /// Execute batch normalization and return the output tensor.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let size = self.input.len();
        let output_buffer = device.create_buffer_f32(size)?;

        let params = BatchNormParams {
            epsilon: self.epsilon,
            _padding: [0.0; 7],
        };

        let params_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("BatchNorm Params"),
            size: std::mem::size_of::<BatchNormParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        device
            .queue
            .write_buffer(&params_buffer, 0, bytemuck::bytes_of(&params));

        ComputeDispatch::new(device, "BatchNorm")
            .shader(SHADER_BATCH_NORM_F64, "main")
            .storage_read(0, self.input.buffer())
            .storage_rw(1, &output_buffer)
            .uniform(2, &params_buffer)
            .dispatch_1d(size as u32)
            .submit()?;

        Ok(Tensor::from_buffer(
            output_buffer,
            self.input.shape().to_vec(),
            device.clone(),
        ))
    }
}

impl Tensor {
    /// Apply per-tensor batch normalization.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn batch_norm(self, epsilon: f32) -> Result<Self> {
        BatchNorm::new(self, epsilon).execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn batch_norm_cpu(input: &[f32], epsilon: f32) -> Vec<f32> {
        let n = input.len() as f32;
        let mean: f32 = input.iter().sum::<f32>() / n;
        let variance: f32 = input.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n;
        let std = (variance + epsilon).sqrt();
        input.iter().map(|x| (x - mean) / std).collect()
    }

    #[tokio::test]
    async fn test_batch_norm_basic() {
        let device = crate::device::test_pool::get_test_device().await;
        let input_data = vec![1.0, 2.0, 3.0, 4.0];
        let input = Tensor::from_vec_on(input_data.clone(), vec![4], device)
            .await
            .unwrap();
        let result = input.batch_norm(1e-5).unwrap();

        let data = result.to_vec().unwrap();
        assert_eq!(data.len(), 4);

        let expected = batch_norm_cpu(&input_data, 1e-5);
        for (r, e) in data.iter().zip(expected.iter()) {
            assert!((r - e).abs() < 1e-4);
        }
    }

    #[tokio::test]
    async fn test_batch_norm_edge_cases() {
        let device = crate::device::test_pool::get_test_device().await;
        // All same values (zero variance)
        let input_data = vec![5.0, 5.0, 5.0, 5.0];
        let input = Tensor::from_vec_on(input_data.clone(), vec![4], device.clone())
            .await
            .unwrap();
        let result = input.batch_norm(1e-5).unwrap();
        let data = result.to_vec().unwrap();
        // Should be all zeros (normalized to mean)
        for val in &data {
            assert!(val.abs() < 1e-3);
        }

        // Negative values
        let input_data = vec![-2.0, -1.0, 1.0, 2.0];
        let input = Tensor::from_vec_on(input_data.clone(), vec![4], device.clone())
            .await
            .unwrap();
        let result = input.batch_norm(1e-5).unwrap();
        let data = result.to_vec().unwrap();
        let expected = batch_norm_cpu(&input_data, 1e-5);
        for (r, e) in data.iter().zip(expected.iter()) {
            assert!((r - e).abs() < 1e-4);
        }
    }

    #[tokio::test]
    async fn test_batch_norm_boundary() {
        let device = crate::device::test_pool::get_test_device().await;
        // Single element
        let input_data = vec![5.0];
        let input = Tensor::from_vec_on(input_data.clone(), vec![1], device.clone())
            .await
            .unwrap();
        let result = input.batch_norm(1e-5).unwrap();
        let data = result.to_vec().unwrap();
        assert!(data[0].abs() < 1e-3); // Should be ~0

        // Wide range of values
        let input_data = vec![-100.0, -50.0, 0.0, 50.0, 100.0];
        let input = Tensor::from_vec_on(input_data.clone(), vec![5], device.clone())
            .await
            .unwrap();
        let result = input.batch_norm(1e-5).unwrap();
        let data = result.to_vec().unwrap();
        let expected = batch_norm_cpu(&input_data, 1e-5);
        for (r, e) in data.iter().zip(expected.iter()) {
            assert!((r - e).abs() < 1e-4);
        }
    }

    #[tokio::test]
    async fn test_batch_norm_large_tensor() {
        let device = crate::device::test_pool::get_test_device().await;
        // 1000 elements
        let input_data: Vec<f32> = (0..1000).map(|i| i as f32 * 0.1).collect();
        let input = Tensor::from_vec_on(input_data.clone(), vec![1000], device)
            .await
            .unwrap();
        let result = input.batch_norm(1e-5).unwrap();

        let data = result.to_vec().unwrap();
        let expected = batch_norm_cpu(&input_data, 1e-5);

        for (r, e) in data.iter().zip(expected.iter()) {
            assert!((r - e).abs() < 1e-3);
        }
    }

    #[tokio::test]
    async fn test_batch_norm_precision() {
        let device = crate::device::test_pool::get_test_device().await;
        // Test FP32 precision
        let input_data = vec![1.234, 5.678, 9.012, 3.456, 7.890];
        let input = Tensor::from_vec_on(input_data.clone(), vec![5], device)
            .await
            .unwrap();
        let result = input.batch_norm(1e-5).unwrap();

        let data = result.to_vec().unwrap();
        let expected = batch_norm_cpu(&input_data, 1e-5);

        // Verify FP32 precision
        let max_error = data
            .iter()
            .zip(expected.iter())
            .map(|(r, e)| (r - e).abs())
            .fold(0.0f32, f32::max);

        assert!(
            max_error < 1e-4,
            "Max error: {max_error} exceeds FP32 threshold"
        );
    }
}

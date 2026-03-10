// SPDX-License-Identifier: AGPL-3.0-only
//! MSE Loss - Mean Squared Error
//! Pure WGSL implementation

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::Result;
use crate::tensor::Tensor;

/// Mean squared error loss between predictions and targets.
pub struct MseLoss {
    predictions: Tensor,
    targets: Tensor,
}

impl MseLoss {
    /// Creates a new MSE loss. Shapes must match.
    #[must_use]
    pub fn new(predictions: Tensor, targets: Tensor) -> Self {
        Self {
            predictions,
            targets,
        }
    }

    fn wgsl_shader() -> &'static str {
        include_str!("../shaders/loss/mse_loss.wgsl")
    }

    /// f64 MSE loss (tree reduction for accumulation accuracy).
    #[must_use]
    pub fn wgsl_shader_f64() -> &'static str {
        include_str!("../shaders/loss/mse_loss_f64.wgsl")
    }

    /// Executes MSE loss and returns a scalar loss tensor.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.predictions.device();
        let size: usize = self.predictions.shape().iter().product();

        // Create output buffer for single loss value
        let output_buffer = device.create_buffer_f32(1)?;

        ComputeDispatch::new(device, "MSE Loss")
            .shader(Self::wgsl_shader(), "main")
            .storage_read(0, self.predictions.buffer())
            .storage_read(1, self.targets.buffer())
            .storage_rw(2, &output_buffer)
            .dispatch_1d(size.max(1) as u32)
            .submit()?;

        Ok(Tensor::from_buffer(output_buffer, vec![1], device.clone()))
    }
}

impl Tensor {
    /// Compute Mean Squared Error loss
    /// # Arguments
    /// * `targets` - Target values
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn mse_loss(self, targets: Tensor) -> Result<Self> {
        MseLoss::new(self, targets).execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device_if_gpu_available;

    fn mse_loss_cpu(predictions: &[f32], targets: &[f32]) -> f32 {
        let n = predictions.len() as f32;
        let sum: f32 = predictions
            .iter()
            .zip(targets.iter())
            .map(|(p, t)| (p - t).powi(2))
            .sum();
        sum / n
    }

    #[tokio::test]
    async fn test_mse_loss_basic() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        // Predictions: [1, 2, 3]
        let pred_data = vec![1.0f32, 2.0, 3.0];
        let predictions = Tensor::from_data(&pred_data, vec![3], device.clone()).unwrap();

        // Targets: [1, 2, 3] (perfect match)
        let target_data = vec![1.0f32, 2.0, 3.0];
        let targets = Tensor::from_data(&target_data, vec![3], device.clone()).unwrap();

        // MSE should be 0
        let result = predictions.mse_loss(targets).unwrap();
        let output = result.to_vec().unwrap();

        assert_eq!(output.len(), 1);
        assert!(output[0] < 0.001); // Should be ~0
    }

    #[tokio::test]
    async fn test_mse_loss_with_error() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        // Predictions: [2, 4, 6]
        let pred_data = vec![2.0f32, 4.0, 6.0];
        let predictions = Tensor::from_data(&pred_data, vec![3], device.clone()).unwrap();

        // Targets: [1, 2, 3]
        let target_data = vec![1.0f32, 2.0, 3.0];
        let targets = Tensor::from_data(&target_data, vec![3], device.clone()).unwrap();

        // MSE = ((2-1)² + (4-2)² + (6-3)²) / 3 = (1 + 4 + 9) / 3 = 4.67
        let result = predictions.mse_loss(targets).unwrap();
        let output = result.to_vec().unwrap();

        assert_eq!(output.len(), 1);
        assert!((output[0] - 4.67).abs() < 0.1);
    }

    #[tokio::test]
    async fn test_mse_loss_edge_cases() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        // Test with zeros
        let pred_data = vec![0.0f32, 0.0, 0.0];
        let predictions = Tensor::from_data(&pred_data, vec![3], device.clone()).unwrap();
        let target_data = vec![0.0f32, 0.0, 0.0];
        let targets = Tensor::from_data(&target_data, vec![3], device.clone()).unwrap();
        let result = predictions.mse_loss(targets).unwrap();
        let output = result.to_vec().unwrap();
        assert!(output[0] < 1e-6);

        // Test with negative values
        let pred_data = vec![-1.0f32, -2.0, -3.0];
        let predictions = Tensor::from_data(&pred_data, vec![3], device.clone()).unwrap();
        let target_data = vec![-1.5f32, -2.5, -3.5];
        let targets = Tensor::from_data(&target_data, vec![3], device.clone()).unwrap();
        let expected = mse_loss_cpu(&pred_data, &target_data);
        let result = predictions.mse_loss(targets).unwrap();
        let output = result.to_vec().unwrap();
        assert!((output[0] - expected).abs() < 1e-5);
    }

    #[tokio::test]
    async fn test_mse_loss_boundary() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        // Single element
        let pred_data = vec![5.0f32];
        let predictions = Tensor::from_data(&pred_data, vec![1], device.clone()).unwrap();
        let target_data = vec![3.0f32];
        let targets = Tensor::from_data(&target_data, vec![1], device.clone()).unwrap();
        let expected = mse_loss_cpu(&pred_data, &target_data);
        let result = predictions.mse_loss(targets).unwrap();
        let output = result.to_vec().unwrap();
        assert!((output[0] - expected).abs() < 1e-5);

        // Large error
        let pred_data = vec![100.0f32, 200.0, 300.0];
        let predictions = Tensor::from_data(&pred_data, vec![3], device.clone()).unwrap();
        let target_data = vec![0.0f32, 0.0, 0.0];
        let targets = Tensor::from_data(&target_data, vec![3], device.clone()).unwrap();
        let expected = mse_loss_cpu(&pred_data, &target_data);
        let result = predictions.mse_loss(targets).unwrap();
        let output = result.to_vec().unwrap();
        assert!((output[0] - expected).abs() < 1e-2);
    }

    #[tokio::test]
    async fn test_mse_loss_large_tensor() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        // 1000 elements
        let pred_data: Vec<f32> = (0..1000).map(|i| i as f32 * 0.1).collect();
        let target_data: Vec<f32> = (0..1000).map(|i| i as f32 * 0.1 + 0.5).collect();

        let predictions = Tensor::from_data(&pred_data, vec![1000], device.clone()).unwrap();
        let targets = Tensor::from_data(&target_data, vec![1000], device.clone()).unwrap();

        let expected = mse_loss_cpu(&pred_data, &target_data);
        let result = predictions.mse_loss(targets).unwrap();
        let output = result.to_vec().unwrap();

        assert_eq!(output.len(), 1);
        assert!((output[0] - expected).abs() < 1e-4);
    }

    #[tokio::test]
    async fn test_mse_loss_precision() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        // Test precision against CPU reference
        let pred_data = vec![1.234f32, 5.678, 9.012, 3.456, 7.890];
        let target_data = vec![1.111f32, 6.789, 8.901, 3.333, 8.000];

        let predictions = Tensor::from_data(&pred_data, vec![5], device.clone()).unwrap();
        let targets = Tensor::from_data(&target_data, vec![5], device.clone()).unwrap();

        let expected = mse_loss_cpu(&pred_data, &target_data);
        let result = predictions.mse_loss(targets).unwrap();
        let output = result.to_vec().unwrap();

        // Verify FP32 precision
        let error = (output[0] - expected).abs();
        assert!(
            error < 1e-5,
            "GPU MSE: {}, CPU MSE: {}, Error: {}",
            output[0],
            expected,
            error
        );
    }
}

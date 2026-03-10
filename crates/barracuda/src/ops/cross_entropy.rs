// SPDX-License-Identifier: AGPL-3.0-only
//! Cross Entropy Loss
//! Pure WGSL implementation

use crate::device::{DeviceCapabilities, WorkloadType};
use crate::error::Result;
use crate::tensor::Tensor;

/// f64 is the canonical source — math is universal, precision is silicon.
static SHADER_F64: &str = include_str!("../shaders/loss/cross_entropy_f64.wgsl");
static SHADER_F32: std::sync::LazyLock<String> =
    std::sync::LazyLock::new(|| SHADER_F64.to_string());

/// Cross entropy loss for classification.
pub struct CrossEntropy {
    predictions: Tensor,
    targets: Tensor,
}

impl CrossEntropy {
    /// Creates a new cross entropy loss. Shapes must match.
    #[must_use]
    pub fn new(predictions: Tensor, targets: Tensor) -> Self {
        Self {
            predictions,
            targets,
        }
    }

    fn wgsl_shader() -> &'static str {
        std::sync::LazyLock::force(&SHADER_F32).as_str()
    }

    /// Executes cross entropy loss and returns a scalar loss tensor.
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.predictions.device();
        let size: usize = self.predictions.shape().iter().product();

        // Create output buffer for single loss value
        let output_buffer = device.create_buffer_f32(1)?;

        // Create shader module
        let shader = device.compile_shader(Self::wgsl_shader(), Some("Cross Entropy Shader"));

        // Create compute pipeline
        let pipeline = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Cross Entropy Pipeline"),
                layout: None,
                module: &shader,
                entry_point: Some("main"),
                cache: None,
                compilation_options: Default::default(),
            });

        // Create bind group
        let bind_group_layout = pipeline.get_bind_group_layout(0);
        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Cross Entropy Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.predictions.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.targets.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buffer.as_entire_binding(),
                },
            ],
        });

        // Execute
        let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("Cross Entropy Encoder"),
        });
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Cross Entropy Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, Some(&bind_group), &[]);
            // Deep Debt Evolution: Capability-based dispatch
            // Cross entropy loss is a reduction over prediction elements
            let caps = DeviceCapabilities::from_device(device);
            let optimal_wg_size = caps.optimal_workgroup_size(WorkloadType::Reduction);
            let workgroups = (size as u32).div_ceil(optimal_wg_size);
            compute_pass.dispatch_workgroups(workgroups.max(1), 1, 1);
        }
        device.submit_and_poll(Some(encoder.finish()));

        Ok(Tensor::from_buffer(output_buffer, vec![1], device.clone()))
    }
}

impl Tensor {
    /// Compute Cross Entropy loss
    /// # Arguments
    /// * `targets` - Target probabilities
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn cross_entropy(self, targets: Tensor) -> Result<Self> {
        CrossEntropy::new(self, targets).execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device_if_gpu_available;

    fn cross_entropy_cpu(predictions: &[f32], targets: &[f32]) -> f32 {
        let n = predictions.len() as f32;
        let sum: f32 = predictions
            .iter()
            .zip(targets.iter())
            .map(|(p, t)| {
                if *t > 0.0 {
                    -t * p.max(1e-10).ln()
                } else {
                    0.0
                }
            })
            .sum();
        sum / n
    }

    #[tokio::test]
    async fn test_cross_entropy_basic() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        // Predictions (probabilities): [0.7, 0.2, 0.1]
        let pred_data = vec![0.7f32, 0.2, 0.1];
        let predictions = Tensor::from_data(&pred_data, vec![3], device.clone()).unwrap();

        // Targets (one-hot): [1, 0, 0]
        let target_data = vec![1.0f32, 0.0, 0.0];
        let targets = Tensor::from_data(&target_data, vec![3], device.clone()).unwrap();

        // Cross entropy = -log(0.7) ≈ 0.357
        let result = predictions.cross_entropy(targets).unwrap();
        let output = result.to_vec().unwrap();

        assert_eq!(output.len(), 1);
        assert!(output[0] > 0.0); // Should be positive
        let expected = cross_entropy_cpu(&pred_data, &target_data);
        assert!((output[0] - expected).abs() < 0.1);
    }

    #[tokio::test]
    async fn test_cross_entropy_edge_cases() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        // Perfect prediction (target = prediction)
        let pred_data = vec![1.0f32, 0.0, 0.0];
        let predictions = Tensor::from_data(&pred_data, vec![3], device.clone()).unwrap();
        let target_data = vec![1.0f32, 0.0, 0.0];
        let targets = Tensor::from_data(&target_data, vec![3], device.clone()).unwrap();
        let result = predictions.cross_entropy(targets).unwrap();
        let output = result.to_vec().unwrap();
        assert!(output[0] < 0.1); // Should be close to 0

        // Uniform distribution
        let pred_data = vec![0.33f32, 0.33, 0.34];
        let predictions = Tensor::from_data(&pred_data, vec![3], device.clone()).unwrap();
        let target_data = vec![0.33f32, 0.33, 0.34];
        let targets = Tensor::from_data(&target_data, vec![3], device.clone()).unwrap();
        let result = predictions.cross_entropy(targets).unwrap();
        let output = result.to_vec().unwrap();
        let expected = cross_entropy_cpu(&pred_data, &target_data);
        assert!((output[0] - expected).abs() < 0.1);
    }

    #[tokio::test]
    async fn test_cross_entropy_boundary() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        // Binary classification
        let pred_data = vec![0.8f32, 0.2];
        let predictions = Tensor::from_data(&pred_data, vec![2], device.clone()).unwrap();
        let target_data = vec![1.0f32, 0.0];
        let targets = Tensor::from_data(&target_data, vec![2], device.clone()).unwrap();
        let result = predictions.cross_entropy(targets).unwrap();
        let output = result.to_vec().unwrap();
        let expected = cross_entropy_cpu(&pred_data, &target_data);
        assert!((output[0] - expected).abs() < 0.1);

        // Many classes
        let pred_data = vec![0.5, 0.2, 0.1, 0.1, 0.05, 0.03, 0.02];
        let predictions = Tensor::from_data(&pred_data, vec![7], device.clone()).unwrap();
        let target_data = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let targets = Tensor::from_data(&target_data, vec![7], device.clone()).unwrap();
        let result = predictions.cross_entropy(targets).unwrap();
        let output = result.to_vec().unwrap();
        let expected = cross_entropy_cpu(&pred_data, &target_data);
        assert!((output[0] - expected).abs() < 0.1);
    }

    #[tokio::test]
    async fn test_cross_entropy_large_tensor() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        // 100 classes
        let mut pred_data = vec![0.01f32; 100];
        pred_data[0] = 0.5;
        let mut target_data = vec![0.0f32; 100];
        target_data[0] = 1.0;

        let predictions = Tensor::from_data(&pred_data, vec![100], device.clone()).unwrap();
        let targets = Tensor::from_data(&target_data, vec![100], device).unwrap();

        let result = predictions.cross_entropy(targets).unwrap();
        let output = result.to_vec().unwrap();
        let expected = cross_entropy_cpu(&pred_data, &target_data);

        assert!((output[0] - expected).abs() < 0.1);
    }

    #[tokio::test]
    async fn test_cross_entropy_precision() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        // Test FP32 precision
        let pred_data = vec![0.123, 0.234, 0.345, 0.298];
        let target_data = vec![0.0, 1.0, 0.0, 0.0];

        let predictions = Tensor::from_data(&pred_data, vec![4], device.clone()).unwrap();
        let targets = Tensor::from_data(&target_data, vec![4], device).unwrap();

        let result = predictions.cross_entropy(targets).unwrap();
        let output = result.to_vec().unwrap();
        let expected = cross_entropy_cpu(&pred_data, &target_data);

        // Verify FP32 precision
        let error = (output[0] - expected).abs();
        assert!(
            error < 0.01,
            "GPU CE: {}, CPU CE: {}, Error: {}",
            output[0],
            expected,
            error
        );
    }
}

// SPDX-License-Identifier: AGPL-3.0-or-later
//! Binary Cross Entropy Loss
//! Pure WGSL implementation

use crate::device::{DeviceCapabilities, WorkloadType};
use crate::error::Result;
use crate::tensor::Tensor;

/// f64 is the canonical source — math is universal, precision is silicon.
static SHADER_F64: &str = include_str!("../shaders/loss/binary_cross_entropy_f64.wgsl");
static SHADER_F32: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
    crate::shaders::precision::downcast_f64_to_f32_with_transcendentals(SHADER_F64)
});

pub struct BinaryCrossEntropy {
    predictions: Tensor,
    targets: Tensor,
}

impl BinaryCrossEntropy {
    pub fn new(predictions: Tensor, targets: Tensor) -> Self {
        Self {
            predictions,
            targets,
        }
    }

    fn wgsl_shader() -> &'static str {
        &SHADER_F32
    }

    pub fn execute(self) -> Result<Tensor> {
        let device = self.predictions.device();

        // Create output buffer for single loss value
        let output_buffer = device.create_buffer_f32(1)?;

        // Create shader module
        let shader = device
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Binary Cross Entropy Shader"),
                source: wgpu::ShaderSource::Wgsl(Self::wgsl_shader().into()),
            });

        // Create compute pipeline
        let pipeline = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Binary Cross Entropy Pipeline"),
                layout: None,
                module: &shader,
                entry_point: Some("main"),
                cache: None,
                compilation_options: Default::default(),
            });

        // Create bind group
        let bind_group_layout = pipeline.get_bind_group_layout(0);
        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Binary Cross Entropy Bind Group"),
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
            label: Some("Binary Cross Entropy Encoder"),
        });
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Binary Cross Entropy Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, Some(&bind_group), &[]);
            // Deep Debt Evolution: Capability-based dispatch
            // Binary cross entropy is a reduction over prediction elements
            let caps = DeviceCapabilities::from_device(device);
            let optimal_wg_size = caps.optimal_workgroup_size(WorkloadType::Reduction);
            let size = self.predictions.shape().iter().product::<usize>();
            let workgroups = (size as u32).div_ceil(optimal_wg_size);
            compute_pass.dispatch_workgroups(workgroups.max(1), 1, 1);
        }
        device.submit_and_poll(Some(encoder.finish()));

        Ok(Tensor::from_buffer(output_buffer, vec![1], device.clone()))
    }
}

impl Tensor {
    /// Compute Binary Cross Entropy loss
    /// # Arguments
    /// * `targets` - Target binary labels (0 or 1)
    pub fn binary_cross_entropy(self, targets: Tensor) -> Result<Self> {
        BinaryCrossEntropy::new(self, targets).execute()
    }
}

#[expect(clippy::unwrap_used, reason = "tests")]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device_if_gpu_available;

    #[tokio::test]
    async fn test_binary_cross_entropy_basic() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        // Predictions (probabilities): [0.9, 0.1, 0.8]
        let pred_data = vec![0.9f32, 0.1, 0.8];
        let predictions = Tensor::from_data(&pred_data, vec![3], device.clone()).unwrap();

        // Targets (binary): [1, 0, 1]
        let target_data = vec![1.0f32, 0.0, 1.0];
        let targets = Tensor::from_data(&target_data, vec![3], device.clone()).unwrap();

        // BCE should be low (good predictions)
        let result = predictions.binary_cross_entropy(targets).unwrap();
        let output = result.to_vec().unwrap();

        assert_eq!(output.len(), 1);
        assert!(output[0].is_finite());
        assert!(output[0] > 0.0); // Should be positive
    }

    #[tokio::test]
    async fn test_binary_cross_entropy_edge_cases() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        // Perfect predictions
        let pred_data = vec![1.0f32, 0.0, 1.0];
        let predictions = Tensor::from_data(&pred_data, vec![3], device.clone()).unwrap();

        let target_data = vec![1.0f32, 0.0, 1.0];
        let targets = Tensor::from_data(&target_data, vec![3], device.clone()).unwrap();

        let result = predictions.binary_cross_entropy(targets).unwrap();
        let output = result.to_vec().unwrap();

        assert!(output[0].is_finite());
        // Perfect predictions should have low loss
        assert!(output[0] < 0.1);
    }

    #[tokio::test]
    async fn test_binary_cross_entropy_boundary() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        // Worst case predictions (opposite of targets)
        let pred_data = vec![0.1f32, 0.9, 0.1];
        let predictions = Tensor::from_data(&pred_data, vec![3], device.clone()).unwrap();

        let target_data = vec![1.0f32, 0.0, 1.0];
        let targets = Tensor::from_data(&target_data, vec![3], device.clone()).unwrap();

        let result = predictions.binary_cross_entropy(targets).unwrap();
        let output = result.to_vec().unwrap();

        assert!(output[0].is_finite());
        // Bad predictions should have high loss
        assert!(output[0] > 1.0);
    }

    #[tokio::test]
    async fn test_binary_cross_entropy_large_batch() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        // Large batch size
        let size = 1000;
        let pred_data = vec![0.7f32; size];
        let predictions = Tensor::from_data(&pred_data, vec![size], device.clone()).unwrap();

        let target_data = vec![1.0f32; size];
        let targets = Tensor::from_data(&target_data, vec![size], device.clone()).unwrap();

        let result = predictions.binary_cross_entropy(targets).unwrap();
        let output = result.to_vec().unwrap();

        assert_eq!(output.len(), 1);
        assert!(output[0].is_finite());
        assert!(output[0] > 0.0);
    }

    #[tokio::test]
    async fn test_binary_cross_entropy_precision() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        // Test determinism
        let pred_data = vec![0.6f32, 0.4, 0.8, 0.2];
        let target_data = vec![1.0f32, 0.0, 1.0, 0.0];

        let predictions1 = Tensor::from_data(&pred_data, vec![4], device.clone()).unwrap();
        let targets1 = Tensor::from_data(&target_data, vec![4], device.clone()).unwrap();

        let predictions2 = Tensor::from_data(&pred_data, vec![4], device.clone()).unwrap();
        let targets2 = Tensor::from_data(&target_data, vec![4], device.clone()).unwrap();

        let result1 = predictions1.binary_cross_entropy(targets1).unwrap();
        let result2 = predictions2.binary_cross_entropy(targets2).unwrap();

        let output1 = result1.to_vec().unwrap();
        let output2 = result2.to_vec().unwrap();

        // Should be deterministic
        assert_eq!(output1[0], output2[0]);
    }
}

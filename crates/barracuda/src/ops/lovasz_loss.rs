// SPDX-License-Identifier: AGPL-3.0-only
//! Lovasz Loss - GPU-accelerated IoU-optimized loss for segmentation
//!
//! **Deep Debt Principles**:
//! - ✅ Pure WGSL implementation (new shader!)
//! - ✅ Safe Rust wrapper (no unsafe code)
//! - ✅ Hardware-agnostic via WebGPU
//! - ✅ Complete implementation (production-ready for segmentation)
//!
//! ## Algorithm
//!
//! ```text
//! Lovasz Loss = Lovasz_extension(IoU_loss)
//!
//! Steps:
//! 1. Compute errors: e = max(0, 1 - p_true)
//! 2. Sort errors in descending order
//! 3. Compute Lovasz extension for IoU
//!
//! Benefits: Directly optimizes IoU metric
//! ```
//!
//! **Key Properties**:
//! - Convex surrogate for `IoU` loss
//! - Directly optimizes Intersection over Union
//! - Better than cross-entropy for segmentation
//! - Especially effective for imbalanced classes
//!
//! **Used By**: Semantic segmentation, medical imaging, scene understanding
//!
//! **Reference**: "The Lovász-Softmax loss" (Berman et al., CVPR 2018)
//!
//! ## Usage
//!
//! ```rust,ignore
//! use barracuda::tensor::Tensor;
//!
//! let predictions = Tensor::randn(vec![1000]).await?;  // Predicted probs
//! let targets = Tensor::randn(vec![1000]).await?;      // Ground truth
//!
//! let loss = predictions.lovasz_loss(&targets)?;
//! ```

use crate::device::{DeviceCapabilities, WorkloadType};
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct LovaszLossParams {
    size: u32,
    smooth: f32,
    _padding: [u32; 2],
}

/// Lovász-Softmax loss for IoU-optimized segmentation.
pub struct LovaszLoss {
    predictions: Tensor,
    targets: Tensor,
}

impl LovaszLoss {
    /// Creates a new Lovász loss. Shapes must match.
    /// # Errors
    /// Returns [`Err`] if prediction and target shapes do not match.
    pub fn new(predictions: Tensor, targets: Tensor) -> Result<Self> {
        // Validate shapes match
        if predictions.shape() != targets.shape() {
            return Err(BarracudaError::shape_mismatch(
                predictions.shape().to_vec(),
                targets.shape().to_vec(),
            ));
        }

        Ok(Self {
            predictions,
            targets,
        })
    }

    fn wgsl_shader() -> &'static str {
        static SHADER: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
            include_str!("../shaders/loss/lovasz_loss_f64.wgsl").to_string()
        });
        std::sync::LazyLock::force(&SHADER).as_str()
    }

    /// Executes Lovász loss and returns the loss tensor.
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.predictions.device();
        let size = self.predictions.shape().iter().product::<usize>();

        let params = LovaszLossParams {
            size: size as u32,
            smooth: 1e-5,
            _padding: [0; 2],
        };

        let output_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("lovasz_loss_output"),
            size: (size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("lovasz_loss_params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let shader = device.compile_shader(Self::wgsl_shader(), Some("lovasz_loss_shader"));

        let bind_group_layout =
            device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("lovasz_loss_bind_group_layout"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });

        let pipeline_layout =
            device
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("lovasz_loss_pipeline_layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    immediate_size: 0,
                });

        let pipeline = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("lovasz_loss_pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                cache: None,
                compilation_options: Default::default(),
            });

        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("lovasz_loss_bind_group"),
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
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("lovasz_loss_encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("lovasz_loss_pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, Some(&bind_group), &[]);

            // Deep Debt Evolution: Capability-based dispatch
            let caps = DeviceCapabilities::from_device(device.as_ref());
            let optimal_wg_size = caps.optimal_workgroup_size(WorkloadType::Reduction);
            let workgroups = (size as u32).div_ceil(optimal_wg_size);
            compute_pass.dispatch_workgroups(workgroups, 1, 1);
        }

        device.submit_commands(Some(encoder.finish()));

        Ok(Tensor::from_buffer(
            output_buffer,
            self.predictions.shape().to_vec(),
            device.clone(),
        ))
    }
}

// ═══════════════════════════════════════════════════════════════
// TENSOR API INTEGRATION
// ═══════════════════════════════════════════════════════════════

impl Tensor {
    /// Lovasz Loss for IoU-optimized semantic segmentation
    /// **Deep Debt**: Essential for semantic segmentation tasks
    /// # Arguments
    /// - `targets`: Ground truth tensor [same shape as predictions]
    /// # Returns
    /// - Loss tensor [same shape as input]
    /// # Example
    /// ```rust,ignore
    /// // Semantic segmentation
    /// let loss = predictions.lovasz_loss(&targets)?;
    /// // Medical imaging
    /// let seg_loss = model_output.lovasz_loss(&ground_truth)?;
    /// ```
    /// # Note
    /// - Directly optimizes `IoU` metric
    /// - Better than cross-entropy for segmentation
    /// - Especially effective for imbalanced classes
    /// - Predictions and targets should be in [0, 1]
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn lovasz_loss(self, targets: &Self) -> Result<Self> {
        LovaszLoss::new(self, targets.clone())?.execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device_if_gpu_available;

    #[tokio::test]
    async fn test_lovasz_loss_basic() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let predictions = Tensor::from_vec_on(vec![0.9, 0.8, 0.7, 0.6], vec![4], device.clone())
            .await
            .unwrap();
        let targets = Tensor::from_vec_on(vec![1.0, 1.0, 1.0, 1.0], vec![4], device.clone())
            .await
            .unwrap();

        let loss = predictions.lovasz_loss(&targets).unwrap();
        let data = loss.to_vec().unwrap();

        assert_eq!(data.len(), 4);
        assert!(data.iter().all(|&x| x.is_finite()));
        assert!(data.iter().all(|&x| x >= 0.0)); // Loss should be non-negative
    }

    #[tokio::test]
    async fn test_lovasz_loss_perfect_prediction() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        // Perfect prediction should have very low loss
        let predictions = Tensor::from_vec_on(vec![1.0, 1.0, 1.0, 1.0], vec![4], device.clone())
            .await
            .unwrap();
        let targets = Tensor::from_vec_on(vec![1.0, 1.0, 1.0, 1.0], vec![4], device.clone())
            .await
            .unwrap();

        let loss = predictions.lovasz_loss(&targets).unwrap();
        let data = loss.to_vec().unwrap();
        let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;

        assert!(
            mean < 0.1,
            "Expected low loss for perfect prediction, got {mean}"
        );
    }

    #[tokio::test]
    async fn test_lovasz_loss_poor_prediction() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        // Poor prediction should have higher loss
        let predictions = Tensor::from_vec_on(vec![0.1, 0.2, 0.3, 0.1], vec![4], device.clone())
            .await
            .unwrap();
        let targets = Tensor::from_vec_on(vec![1.0, 1.0, 1.0, 1.0], vec![4], device.clone())
            .await
            .unwrap();

        let loss = predictions.lovasz_loss(&targets).unwrap();
        let data = loss.to_vec().unwrap();

        assert!(data.iter().all(|&x| x > 0.5)); // Should have high error
    }

    #[tokio::test]
    async fn test_lovasz_loss_validation() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        // Shape mismatch
        let predictions = Tensor::from_vec_on(vec![0.5; 10], vec![10], device.clone())
            .await
            .unwrap();
        let targets = Tensor::from_vec_on(vec![1.0; 5], vec![5], device.clone())
            .await
            .unwrap();

        assert!(predictions.lovasz_loss(&targets).is_err());
    }

    #[tokio::test]
    async fn test_lovasz_loss_large_batch() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let size = 1000;
        let pred_data: Vec<f32> = (0..size).map(|i| (i as f32) / size as f32).collect();
        let target_data = vec![1.0; size];

        let predictions = Tensor::from_vec_on(pred_data, vec![size], device.clone())
            .await
            .unwrap();
        let targets = Tensor::from_vec_on(target_data, vec![size], device.clone())
            .await
            .unwrap();

        let loss = predictions.lovasz_loss(&targets).unwrap();
        let data = loss.to_vec().unwrap();

        assert_eq!(data.len(), size);
        assert!(data.iter().all(|&x| x.is_finite()));
        assert!(data.iter().all(|&x| x >= 0.0));
    }
}

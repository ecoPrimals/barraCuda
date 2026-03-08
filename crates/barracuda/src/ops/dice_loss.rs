// SPDX-License-Identifier: AGPL-3.0-or-later
//! Dice Loss - Medical image segmentation loss function
//!
//! **Deep Debt Principles**:
//! - ✅ Pure WGSL implementation
//! - ✅ Safe Rust wrapper
//! - ✅ Handles class imbalance (medical imaging standard)
//!
//! ## Algorithm
//!
//! Dice Loss = 1 - Dice Coefficient
//! Dice = (2 * |X ∩ Y| + smooth) / (|X| + |Y| + smooth)
//!
//! Where X = predicted, Y = target, smooth prevents division by zero
//!
//! ## Usage
//!
//! ```rust,ignore
//! let predicted = Tensor::sigmoid(logits)?; // [0, 1] probabilities
//! let target = Tensor::from_vec(ground_truth, shape).await?;
//! let loss = predicted.dice_loss(&target, 1.0)?; // smooth = 1.0
//! ```

use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct DiceLossParams {
    size: u32,
    smooth_val: f32,
    _padding: [u32; 2],
}

/// Dice loss for medical image segmentation (handles class imbalance).
pub struct DiceLoss {
    predicted: Tensor,
    target: Tensor,
    smooth: f32,
}

impl DiceLoss {
    /// Creates a new Dice loss. Smooth prevents division by zero.
    /// # Errors
    /// Returns [`Err`] if predicted and target shapes do not match.
    pub fn new(predicted: Tensor, target: Tensor, smooth: f32) -> Result<Self> {
        if predicted.shape() != target.shape() {
            return Err(BarracudaError::shape_mismatch(
                predicted.shape().to_vec(),
                target.shape().to_vec(),
            ));
        }

        Ok(Self {
            predicted,
            target,
            smooth,
        })
    }

    fn wgsl_shader() -> &'static str {
        static SHADER: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
            include_str!("../shaders/loss/dice_loss_f64.wgsl").to_string()
        });
        std::sync::LazyLock::force(&SHADER).as_str()
    }

    /// Executes Dice loss and returns a scalar loss tensor.
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.predicted.device();
        let size = self.predicted.len();

        // Output is scalar loss value
        let output_buffer = device.create_buffer_f32(1)?;

        let params = DiceLossParams {
            size: size as u32,
            smooth_val: self.smooth,
            _padding: [0, 0],
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Dice Loss Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind_group_layout =
            device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Dice Loss Bind Group Layout"),
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

        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Dice Loss Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.predicted.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.target.buffer().as_entire_binding(),
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

        let shader_module = device.compile_shader(Self::wgsl_shader(), Some("Dice Loss Shader"));

        let pipeline_layout =
            device
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Dice Loss Pipeline Layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    immediate_size: 0,
                });

        let pipeline = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Dice Loss Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: Some("main"),
                cache: None,
                compilation_options: Default::default(),
            });

        let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("Dice Loss Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Dice Loss Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, Some(&bind_group), &[]);

            // Deep Debt Evolution: Capability-based dispatch
            use crate::device::{DeviceCapabilities, WorkloadType};
            let caps = DeviceCapabilities::from_device(device);
            let optimal_wg_size = caps.optimal_workgroup_size(WorkloadType::ElementWise);
            let workgroups = (size as u32).div_ceil(optimal_wg_size);
            compute_pass.dispatch_workgroups(workgroups, 1, 1);
        }

        device.submit_and_poll(Some(encoder.finish()));

        Ok(Tensor::from_buffer(output_buffer, vec![1], device.clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device_if_gpu_available;

    #[tokio::test]
    async fn test_dice_loss_perfect_overlap() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        // Perfect prediction = target
        let pred = Tensor::from_vec_on(vec![1.0, 1.0, 0.0, 0.0], vec![4], device.clone())
            .await
            .unwrap();
        let target = Tensor::from_vec_on(vec![1.0, 1.0, 0.0, 0.0], vec![4], device)
            .await
            .unwrap();

        let loss = DiceLoss::new(pred, target, 1.0).unwrap().execute().unwrap();
        let result = loss.to_vec().unwrap();

        // Perfect overlap: Dice = 1.0, Loss = 0.0
        assert!(result[0] < 0.1, "Perfect overlap should have loss ≈ 0");
    }

    #[tokio::test]
    async fn test_dice_loss_no_overlap() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        // No overlap
        let pred = Tensor::from_vec_on(vec![1.0, 1.0, 0.0, 0.0], vec![4], device.clone())
            .await
            .unwrap();
        let target = Tensor::from_vec_on(vec![0.0, 0.0, 1.0, 1.0], vec![4], device)
            .await
            .unwrap();

        let loss = DiceLoss::new(pred, target, 1.0).unwrap().execute().unwrap();
        let result = loss.to_vec().unwrap();

        // No overlap: Loss should be high (close to 1.0)
        assert!(result[0] > 0.5, "No overlap should have high loss");
    }

    #[tokio::test]
    async fn test_dice_loss_partial_overlap() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let pred = Tensor::from_vec_on(vec![0.8, 0.6, 0.2, 0.1], vec![4], device.clone())
            .await
            .unwrap();
        let target = Tensor::from_vec_on(vec![1.0, 0.0, 0.0, 1.0], vec![4], device)
            .await
            .unwrap();

        let loss = DiceLoss::new(pred, target, 1.0).unwrap().execute().unwrap();
        let result = loss.to_vec().unwrap();

        // Partial overlap: 0 < Loss < 1
        assert!(result[0] > 0.0 && result[0] < 1.0);
        assert!(result[0].is_finite());
    }
}

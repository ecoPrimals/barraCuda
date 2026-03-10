// SPDX-License-Identifier: AGPL-3.0-only
//! Center Loss for metric learning
//!
//! **Pure WGSL**: Single implementation via WebGPU shader
//! Learns class centers and penalizes intra-class variance

use crate::device::{DeviceCapabilities, WorkloadType};
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct CenterLossParams {
    batch_size: u32,
    feature_dim: u32,
    num_classes: u32,
    _padding: u32,
}

/// Center loss for metric learning: penalizes intra-class variance.
pub struct CenterLoss {
    features: Tensor,
    centers: Tensor,
    labels: Tensor,
}

impl CenterLoss {
    /// Create `CenterLoss` operation
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn new(features: Tensor, centers: Tensor, labels: Tensor) -> Result<Self> {
        // Validate shapes
        if features.shape().len() != 2 {
            return Err(BarracudaError::invalid_op(
                "CenterLoss",
                format!(
                    "features must be 2D [batch, feature_dim], got shape {:?}",
                    features.shape()
                ),
            ));
        }

        if centers.shape().len() != 2 {
            return Err(BarracudaError::invalid_op(
                "CenterLoss",
                format!(
                    "centers must be 2D [num_classes, feature_dim], got shape {:?}",
                    centers.shape()
                ),
            ));
        }

        if labels.shape().len() != 1 {
            return Err(BarracudaError::invalid_op(
                "CenterLoss",
                format!("labels must be 1D [batch], got shape {:?}", labels.shape()),
            ));
        }

        Ok(Self {
            features,
            centers,
            labels,
        })
    }

    /// WGSL shader source (embedded at compile time)
    fn wgsl_shader() -> &'static str {
        static SHADER: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
            include_str!("../shaders/loss/center_loss_f64.wgsl").to_string()
        });
        &SHADER
    }

    /// Execute `CenterLoss` on tensor
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.features.device();
        let features_shape = self.features.shape();
        let batch_size = features_shape[0];
        let feature_dim = features_shape[1];

        // Create output buffer: [batch] - per-sample loss
        let output_buffer = device.create_buffer_f32(batch_size)?;

        let params = CenterLossParams {
            batch_size: batch_size as u32,
            feature_dim: feature_dim as u32,
            num_classes: self.centers.shape()[0] as u32,
            _padding: 0,
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("CenterLoss Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        // Create bind group layout
        let bind_group_layout =
            device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("CenterLoss Bind Group Layout"),
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
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 4,
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

        // Create bind group
        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("CenterLoss Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.features.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.centers.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.labels.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        // Compile shader
        let shader = device.compile_shader(Self::wgsl_shader(), Some("CenterLoss"));

        // Create pipeline
        let pipeline_layout =
            device
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("CenterLoss Pipeline Layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    immediate_size: 0,
                });

        let pipeline = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("CenterLoss Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                cache: None,
                compilation_options: Default::default(),
            });

        // Encode and execute
        let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("CenterLoss Encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("CenterLoss Pass"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&bind_group), &[]);

            // Deep Debt Evolution: Capability-based dispatch
            let caps = DeviceCapabilities::from_device(device);
            let optimal_wg_size = caps.optimal_workgroup_size(WorkloadType::Reduction);
            let workgroups = (batch_size as u32).div_ceil(optimal_wg_size);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        device.submit_and_poll(Some(encoder.finish()));

        // Create output tensor
        Ok(Tensor::from_buffer(
            output_buffer,
            vec![batch_size],
            device.clone(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device_if_gpu_available;

    #[tokio::test]
    async fn test_center_loss_basic() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let batch_size = 4;
        let feature_dim = 3;
        let num_classes = 2;

        let features = Tensor::from_vec_on(
            vec![1.0; batch_size * feature_dim],
            vec![batch_size, feature_dim],
            device.clone(),
        )
        .await
        .unwrap();

        let centers = Tensor::from_vec_on(
            vec![0.5; num_classes * feature_dim],
            vec![num_classes, feature_dim],
            device.clone(),
        )
        .await
        .unwrap();

        let labels =
            Tensor::from_vec_on(vec![0.0, 1.0, 0.0, 1.0], vec![batch_size], device.clone())
                .await
                .unwrap();

        let result = CenterLoss::new(features, centers, labels)
            .unwrap()
            .execute()
            .unwrap();

        assert_eq!(result.shape(), &[batch_size]);
    }
}

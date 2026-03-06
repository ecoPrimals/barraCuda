// SPDX-License-Identifier: AGPL-3.0-or-later
//! `NLLLoss` - Pure WGSL
//!
//! Deep Debt Principles:
//! - Self-knowledge: Operation knows its computation
//! - Zero hardcoding: Hardware-agnostic implementation
//! - Modern idiomatic Rust: Safe, zero unsafe code
//! - Complete implementation: Production-ready, no mocks
//! - Hardware-agnostic: Pure WGSL for universal compute

use crate::device::{DeviceCapabilities, WorkloadType};
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

/// Negative Log Likelihood Loss operation
pub struct NLLLoss {
    log_probs: Tensor,
    targets: Tensor,
    weights: Option<Tensor>,
    ignore_index: i32,
    reduction: u32, // 0 = none, 1 = mean, 2 = sum
}

impl NLLLoss {
    /// Create a new NLL loss operation
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn new(
        log_probs: Tensor,
        targets: Tensor,
        weights: Option<Tensor>,
        ignore_index: Option<i32>,
        reduction: Option<u32>,
    ) -> Result<Self> {
        let log_probs_shape = log_probs.shape();
        let batch_size = log_probs_shape[0];
        let num_classes = log_probs_shape[1..].iter().product::<usize>();

        let targets_shape = targets.shape();
        if targets_shape[0] != batch_size {
            return Err(BarracudaError::invalid_op(
                "nll_loss",
                "targets batch size mismatch",
            ));
        }

        if let Some(ref w) = weights {
            if w.shape().iter().product::<usize>() != num_classes {
                return Err(BarracudaError::invalid_op(
                    "nll_loss",
                    "weights must have num_classes elements",
                ));
            }
        }

        Ok(Self {
            log_probs,
            targets,
            weights,
            ignore_index: ignore_index.unwrap_or(-1),
            reduction: reduction.unwrap_or(1), // mean by default
        })
    }

    /// Get the WGSL shader source
    fn wgsl_shader() -> &'static str {
        static SHADER: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
            crate::shaders::precision::downcast_f64_to_f32_with_transcendentals(include_str!(
                "../shaders/loss/nll_loss_f64.wgsl"
            ))
        });
        &SHADER
    }

    /// Execute the NLL loss operation
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.log_probs.device();

        let batch_size = self.log_probs.shape()[0];
        let num_classes = self.log_probs.shape()[1..].iter().product::<usize>();

        // Output size depends on reduction
        let output_size = if self.reduction == 0 { batch_size } else { 1 };
        let output_buffer = device.create_buffer_f32(output_size)?;

        // Create uniform buffer for parameters
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            batch_size: u32,
            num_classes: u32,
            ignore_index: i32,
            reduction: u32,
        }

        let params = Params {
            batch_size: batch_size as u32,
            num_classes: num_classes as u32,
            ignore_index: self.ignore_index,
            reduction: self.reduction,
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("NLLLoss Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        // Create weight buffer (use ones if not provided)
        let ones_buffer;
        let weight_buffer = if let Some(ref w) = self.weights {
            w.buffer()
        } else {
            let ones = vec![1.0f32; num_classes];
            ones_buffer = device
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("NLLLoss Weight"),
                    contents: bytemuck::cast_slice(&ones),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                });
            &ones_buffer
        };

        // Compile shader
        let shader_module = device.compile_shader(Self::wgsl_shader(), Some("NLLLoss Shader"));

        // Create bind group layout
        let bind_group_layout =
            device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("NLLLoss Bind Group Layout"),
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
            label: Some("NLLLoss Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.log_probs.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.targets.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: weight_buffer.as_entire_binding(),
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

        // Create pipeline
        let pipeline_layout =
            device
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("NLLLoss Pipeline Layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    immediate_size: 0,
                });

        let pipeline = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("NLLLoss Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: Some("main"),
                cache: None,
                compilation_options: Default::default(),
            });

        // Encode and execute
        let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("NLLLoss Encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("NLLLoss Pass"),
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
        let output_shape = if self.reduction == 0 {
            vec![batch_size]
        } else {
            vec![1]
        };

        Ok(Tensor::from_buffer(
            output_buffer,
            output_shape,
            device.clone(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device_if_gpu_available;

    #[tokio::test]
    async fn test_nll_loss_basic() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let batch_size = 2;
        let num_classes = 3;

        let log_probs = Tensor::from_vec_on(
            vec![-0.9, -2.3, -1.2, -2.1, -0.2, -1.5],
            vec![batch_size, num_classes],
            device.clone(),
        )
        .await
        .unwrap();

        let targets = Tensor::from_vec_on(vec![0.0, 1.0], vec![batch_size], device.clone())
            .await
            .unwrap();

        let output = NLLLoss::new(log_probs, targets, None, None, None)
            .unwrap()
            .execute()
            .unwrap();

        assert_eq!(output.shape(), &[1]);
    }
}

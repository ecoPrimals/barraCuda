//! MultiMarginLoss - Pure WGSL
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
use wgpu::util::DeviceExt;

/// Multi-Margin Loss operation
pub struct MultiMarginLoss {
    input: Tensor,
    target: Tensor,
    weight: Option<Tensor>,
    num_classes: usize,
    p: u32,
    margin: f32,
}

impl MultiMarginLoss {
    /// Create a new multi-margin loss operation
    pub fn new(
        input: Tensor,
        target: Tensor,
        weight: Option<Tensor>,
        num_classes: usize,
        p: u32,
        margin: f32,
    ) -> Result<Self> {
        let input_shape = input.shape();
        let batch_size = input_shape[0];
        let input_classes = input_shape[1..].iter().product::<usize>();

        if input_classes != num_classes {
            return Err(BarracudaError::invalid_op(
                "multi_margin_loss",
                "input must have num_classes columns",
            ));
        }

        let target_shape = target.shape();
        if target_shape[0] != batch_size {
            return Err(BarracudaError::invalid_op(
                "multi_margin_loss",
                "target batch size mismatch",
            ));
        }

        if let Some(ref w) = weight {
            if w.shape().iter().product::<usize>() != num_classes {
                return Err(BarracudaError::invalid_op(
                    "multi_margin_loss",
                    "weight must have num_classes elements",
                ));
            }
        }

        if p != 1 && p != 2 {
            return Err(BarracudaError::invalid_op(
                "multi_margin_loss",
                "p must be 1 or 2",
            ));
        }

        Ok(Self {
            input,
            target,
            weight,
            num_classes,
            p,
            margin,
        })
    }

    /// Get the WGSL shader source
    fn wgsl_shader() -> &'static str {
        static SHADER: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
            crate::shaders::precision::downcast_f64_to_f32_with_transcendentals(include_str!(
                "../shaders/math/multi_margin_loss_f64.wgsl"
            ))
        });
        &SHADER
    }

    /// Execute the multi-margin loss operation
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();

        let batch_size = self.input.shape()[0];
        let output_size = batch_size;
        let output_buffer = device.create_buffer_f32(output_size)?;

        // Create uniform buffer for parameters
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            batch_size: u32,
            num_classes: u32,
            p: u32,
            margin: f32,
        }

        let params = Params {
            batch_size: batch_size as u32,
            num_classes: self.num_classes as u32,
            p: self.p,
            margin: self.margin,
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("MultiMarginLoss Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        // Create weight buffer (use ones if not provided)
        let ones_buffer;
        let weight_buffer = if let Some(ref w) = self.weight {
            w.buffer()
        } else {
            let ones = vec![1.0f32; self.num_classes];
            ones_buffer = device
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("MultiMarginLoss Weight"),
                    contents: bytemuck::cast_slice(&ones),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                });
            &ones_buffer
        };

        // Compile shader
        let shader_module =
            device.compile_shader(Self::wgsl_shader(), Some("MultiMarginLoss Shader"));

        // Create bind group layout
        let bind_group_layout =
            device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("MultiMarginLoss Bind Group Layout"),
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
            label: Some("MultiMarginLoss Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.input.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.target.buffer().as_entire_binding(),
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
                    label: Some("MultiMarginLoss Pipeline Layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });

        let pipeline = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("MultiMarginLoss Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: "main",
                cache: None,
                compilation_options: Default::default(),
            });

        // Encode and execute
        let mut encoder = device
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("MultiMarginLoss Encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("MultiMarginLoss Pass"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);

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
    async fn test_multi_margin_loss_basic() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let batch_size = 2;
        let num_classes = 3;

        let input = Tensor::from_vec_on(
            vec![0.9, 0.1, 0.1, 0.1, 0.8, 0.2],
            vec![batch_size, num_classes],
            device.clone(),
        )
        .await
        .unwrap();

        let target = Tensor::from_vec_on(vec![0.0f32, 1.0f32], vec![batch_size], device.clone())
            .await
            .unwrap();

        let output = MultiMarginLoss::new(input, target, None, num_classes, 1, 1.0)
            .unwrap()
            .execute()
            .unwrap();

        assert_eq!(output.shape(), &[batch_size]);
    }
}

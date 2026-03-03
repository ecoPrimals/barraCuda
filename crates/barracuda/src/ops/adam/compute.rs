//! GPU compute operations for Adam Optimizer
//!
//! This module contains the single-pass GPU execution for Adam optimizer
//! with bias correction.

use super::{Adam, AdamParams};
use crate::device::{DeviceCapabilities, WorkloadType};
use crate::error::Result;
use crate::tensor::Tensor;
use wgpu::util::DeviceExt;

impl Adam {
    /// Execute Adam optimizer step (GPU single-pass with bias correction)
    ///
    /// **Deep Debt**: Efficient single-pass update with bias correction
    ///
    /// Returns: (new_params, new_m, new_v)
    pub fn execute(self) -> Result<(Tensor, Tensor, Tensor)> {
        let device = self.params().device();
        let size = self.params().shape().iter().product::<usize>();

        let adam_params = AdamParams {
            num_params: size as u32,
            learning_rate: self.learning_rate(),
            beta1: self.beta1(),
            beta2: self.beta2(),
            epsilon: 1e-8,
            weight_decay: 0.0,
            step: self.step() as u32,
        };

        // Create writable buffers (shader does in-place updates)
        let zeros = vec![0.0f32; size];

        // Copy params to writable buffer
        let params_data = self.params().to_vec()?;
        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("adam_params"),
                contents: bytemuck::cast_slice(&params_data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });

        // Copy or create m buffer
        let m_data = if let Some(m_tensor) = self.m() {
            m_tensor.to_vec()?
        } else {
            zeros.clone()
        };
        let m_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("adam_m"),
                contents: bytemuck::cast_slice(&m_data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });

        // Copy or create v buffer
        let v_data = if let Some(v_tensor) = self.v() {
            v_tensor.to_vec()?
        } else {
            zeros
        };
        let v_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("adam_v"),
                contents: bytemuck::cast_slice(&v_data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });

        let adam_params_buffer =
            device
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("adam_params"),
                    contents: bytemuck::cast_slice(&[adam_params]),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

        let shader = device
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("adam_shader"),
                source: wgpu::ShaderSource::Wgsl(Self::shader().into()),
            });

        let bind_group_layout =
            device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("adam_bind_group_layout"),
                    entries: &[
                        // binding 0: gradients (read)
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
                        // binding 1: params (read_write)
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // binding 2: m (read_write)
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
                        // binding 3: v (read_write)
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
                        // binding 4: adam_params (uniform)
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

        let pipeline_layout =
            device
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("adam_pipeline_layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });

        let pipeline = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("adam_pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: "main",
                cache: None,
                compilation_options: Default::default(),
            });

        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("adam_bind_group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.gradients().buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: m_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: v_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: adam_params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("adam_encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("adam_pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            // Deep Debt Evolution: Capability-based dispatch
            let caps = DeviceCapabilities::from_device(device);
            let optimal_wg_size = caps.optimal_workgroup_size(WorkloadType::ElementWise);
            let workgroups = (size as u32).div_ceil(optimal_wg_size);
            compute_pass.dispatch_workgroups(workgroups, 1, 1);
        }

        device.submit_and_poll(Some(encoder.finish()));

        let updated_params = Tensor::from_buffer(
            params_buffer,
            self.params().shape().to_vec(),
            device.clone(),
        );

        let updated_m =
            Tensor::from_buffer(m_buffer, self.params().shape().to_vec(), device.clone());

        let updated_v =
            Tensor::from_buffer(v_buffer, self.params().shape().to_vec(), device.clone());

        Ok((updated_params, updated_m, updated_v))
    }
}

// SPDX-License-Identifier: AGPL-3.0-or-later
//! Cyclical learning rate operation
//!
//! Cycles learning rate between bounds for better convergence
//! Reference: "Cyclical Learning Rates for Training Neural Networks" by Smith (2017)

use crate::device::{DeviceCapabilities, WorkloadType};
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;
use bytemuck::{Pod, Zeroable};

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct CyclicalLrParams {
    current_iter: u32,
    step_size: u32,
    base_lr: f32,
    max_lr: f32,
    mode: u32,
    gamma: f32,
    _padding: [u32; 2],
}

/// Cyclical learning rate operation
pub struct CyclicalLr {
    current_iter: u32,
    step_size: u32,
    base_lr: f32,
    max_lr: f32,
    mode: CyclicalLrMode,
    gamma: f32,
}

/// Cyclical learning rate mode
#[derive(Copy, Clone, Debug)]
pub enum CyclicalLrMode {
    Triangular = 0,
    Triangular2 = 1,
    ExpRange = 2,
}

impl CyclicalLr {
    /// Create cyclical learning rate operation
    pub fn new(
        current_iter: u32,
        step_size: u32,
        base_lr: f32,
        max_lr: f32,
        mode: CyclicalLrMode,
        gamma: f32,
    ) -> Result<Self> {
        if step_size == 0 {
            return Err(BarracudaError::invalid_op(
                "cyclical_lr",
                "step_size must be greater than 0",
            ));
        }

        if base_lr < 0.0 || max_lr < 0.0 {
            return Err(BarracudaError::invalid_op(
                "cyclical_lr",
                "base_lr and max_lr must be non-negative",
            ));
        }

        if base_lr > max_lr {
            return Err(BarracudaError::invalid_op(
                "cyclical_lr",
                format!("base_lr {base_lr} must be <= max_lr {max_lr}"),
            ));
        }

        if matches!(mode, CyclicalLrMode::ExpRange) && gamma <= 0.0 {
            return Err(BarracudaError::invalid_op(
                "cyclical_lr",
                "gamma must be positive for ExpRange mode",
            ));
        }

        Ok(Self {
            current_iter,
            step_size,
            base_lr,
            max_lr,
            mode,
            gamma,
        })
    }

    /// WGSL shader source (embedded at compile time)
    fn wgsl_shader() -> &'static str {
        {
            static SHADER: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
                crate::shaders::precision::downcast_f64_to_f32_with_transcendentals(include_str!(
                    "../shaders/optimizer/cyclical_lr_f64.wgsl"
                ))
            });
            std::sync::LazyLock::force(&SHADER).as_str()
        }
    }

    /// Execute cyclical learning rate computation
    pub fn execute(self, device: &crate::device::WgpuDevice) -> Result<Tensor> {
        // Create output buffer (scalar LR value)
        let output_buffer = device.create_buffer_f32(1)?;

        // Create params
        let params = CyclicalLrParams {
            current_iter: self.current_iter,
            step_size: self.step_size,
            base_lr: self.base_lr,
            max_lr: self.max_lr,
            mode: self.mode as u32,
            gamma: self.gamma,
            _padding: [0; 2],
        };

        let params_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("CyclicalLr Params"),
            size: std::mem::size_of::<CyclicalLrParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        device
            .queue
            .write_buffer(&params_buffer, 0, bytemuck::bytes_of(&params));

        // Create bind group layout
        let bind_group_layout =
            device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("CyclicalLr Bind Group Layout"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
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
            label: Some("CyclicalLr Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        // Compile shader
        let shader = device.compile_shader(Self::wgsl_shader(), Some("CyclicalLr"));

        // Create pipeline
        let pipeline_layout =
            device
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("CyclicalLr Pipeline Layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });

        let pipeline = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("CyclicalLr Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: "main",
                cache: None,
                compilation_options: Default::default(),
            });

        // Encode and execute
        let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("CyclicalLr Encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("CyclicalLr Pass"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);

            // Deep Debt Evolution: Capability-based dispatch
            // Note: This is a scalar operation (single LR value), but using capability pattern for consistency
            let caps = DeviceCapabilities::from_device(device);
            let optimal_wg_size = caps.optimal_workgroup_size(WorkloadType::ElementWise);
            let workgroups = 1u32.div_ceil(optimal_wg_size);
            pass.dispatch_workgroups(workgroups.max(1), 1, 1);
        }

        device.submit_and_poll(Some(encoder.finish()));

        // Create output tensor (scalar)
        Ok(Tensor::from_buffer(
            output_buffer,
            vec![1],
            std::sync::Arc::new(device.clone()),
        ))
    }
}

#[expect(clippy::unwrap_used, reason = "tests")]
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_cyclical_lr_basic() {
        // Note: This test requires device creation which is async
        // For now, we'll test the validation logic
        let result = CyclicalLr::new(0, 100, 0.001, 0.01, CyclicalLrMode::Triangular, 0.9);
        assert!(result.is_ok());

        // Test invalid step_size
        let result = CyclicalLr::new(0, 0, 0.001, 0.01, CyclicalLrMode::Triangular, 0.9);
        assert!(result.is_err());

        // Test invalid lr range
        let result = CyclicalLr::new(0, 100, 0.01, 0.001, CyclicalLrMode::Triangular, 0.9);
        assert!(result.is_err());
    }
}

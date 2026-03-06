// SPDX-License-Identifier: AGPL-3.0-or-later
//! `PoissonNLLLoss` - Pure WGSL
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

/// Poisson Negative Log Likelihood Loss operation
pub struct PoissonNLLLoss {
    input: Tensor,
    target: Tensor,
    log_input: bool,
    full: bool,
    epsilon: f32,
}

impl PoissonNLLLoss {
    /// Create a new Poisson NLL loss operation
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if input and target shapes do not match.
    pub fn new(
        input: Tensor,
        target: Tensor,
        log_input: Option<bool>,
        full: Option<bool>,
        epsilon: Option<f32>,
    ) -> Result<Self> {
        if input.shape() != target.shape() {
            return Err(BarracudaError::invalid_op(
                "poisson_nll_loss",
                "input and target shapes must match",
            ));
        }

        Ok(Self {
            input,
            target,
            log_input: log_input.unwrap_or(false),
            full: full.unwrap_or(false),
            epsilon: epsilon.unwrap_or(1e-8),
        })
    }

    /// Get the WGSL shader source
    fn wgsl_shader() -> &'static str {
        static SHADER: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
            crate::shaders::precision::downcast_f64_to_f32_with_transcendentals(include_str!(
                "../shaders/loss/poisson_nll_loss_f64.wgsl"
            ))
        });
        &SHADER
    }

    /// Execute the Poisson NLL loss operation
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();

        let size = self.input.shape().iter().product::<usize>();
        let output_buffer = device.create_buffer_f32(size)?;

        // Create uniform buffer for parameters
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            size: u32,
            log_input: u32,
            full: u32,
            epsilon: f32,
        }

        let params = Params {
            size: size as u32,
            log_input: u32::from(self.log_input),
            full: u32::from(self.full),
            epsilon: self.epsilon,
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("PoissonNLLLoss Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        // Compile shader
        let shader_module =
            device.compile_shader(Self::wgsl_shader(), Some("PoissonNLLLoss Shader"));

        // Create bind group layout
        let bind_group_layout =
            device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("PoissonNLLLoss Bind Group Layout"),
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

        // Create bind group
        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("PoissonNLLLoss Bind Group"),
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
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        // Create pipeline
        let pipeline_layout =
            device
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("PoissonNLLLoss Pipeline Layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    immediate_size: 0,
                });

        let pipeline = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("PoissonNLLLoss Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: Some("main"),
                cache: None,
                compilation_options: Default::default(),
            });

        // Encode and execute
        let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("PoissonNLLLoss Encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("PoissonNLLLoss Pass"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&bind_group), &[]);

            // Deep Debt Evolution: Capability-based dispatch
            let caps = DeviceCapabilities::from_device(device);
            let optimal_wg_size = caps.optimal_workgroup_size(WorkloadType::Reduction);
            let workgroups = (size as u32).div_ceil(optimal_wg_size);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        device.submit_and_poll(Some(encoder.finish()));

        // Create output tensor
        Ok(Tensor::from_buffer(
            output_buffer,
            self.input.shape().to_vec(),
            device.clone(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device_if_gpu_available;

    #[tokio::test]
    async fn test_poisson_nll_loss_basic() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let size = 10;

        let input = Tensor::from_vec_on(vec![1.0; size], vec![size], device.clone())
            .await
            .unwrap();

        let target = Tensor::from_vec_on(vec![2.0; size], vec![size], device.clone())
            .await
            .unwrap();

        let output = PoissonNLLLoss::new(input, target, None, None, None)
            .unwrap()
            .execute()
            .unwrap();

        assert_eq!(output.shape(), &[size]);
    }
}

// SPDX-License-Identifier: AGPL-3.0-only
//! BETA - Beta function B(a,b) - Pure WGSL
//!
//! Deep Debt Principles:
//! - Self-knowledge: Operation knows its computation
//! - Zero hardcoding: Hardware-agnostic implementation
//! - Modern idiomatic Rust: Safe, zero unsafe code
//! - Complete implementation: Production-ready, no mocks
//! - Hardware-agnostic: Pure WGSL for universal compute

use crate::device::{DeviceCapabilities, WorkloadType};
use crate::error::Result;
use crate::tensor::Tensor;

/// Beta function B(a,b) = Γ(a)Γ(b)/Γ(a+b)
pub struct Beta {
    input: Tensor, // Interleaved pairs [a₀, b₀, a₁, b₁, ...]
}

impl Beta {
    /// Create new Beta function operation
    /// Input tensor must have even length: [a₀, b₀, a₁, b₁, ...]
    #[must_use]
    pub fn new(input: Tensor) -> Self {
        Self { input }
    }

    fn wgsl_shader() -> &'static str {
        include_str!("../shaders/special/beta.wgsl")
    }

    /// Execute Beta function B(a,b) = Γ(a)Γ(b)/Γ(a+b) on input pairs.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let input_size: usize = self.input.shape().iter().product();
        let output_size = input_size / 2; // Output is half the input size

        let output_buffer = device.create_buffer_f32(output_size)?;

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            size: u32,
        }

        let params = Params {
            size: output_size as u32,
        };
        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Beta Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind_group_layout =
            device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Beta Bind Group Layout"),
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
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
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
            label: Some("Beta Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.input.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let shader = device.compile_shader(Self::wgsl_shader(), Some("Beta"));

        let pipeline_layout =
            device
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Beta Pipeline Layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    immediate_size: 0,
                });

        let pipeline = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Beta Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                cache: None,
                compilation_options: Default::default(),
            });

        let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("Beta Encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Beta Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&bind_group), &[]);
            let caps = DeviceCapabilities::from_device(device);
            let optimal_wg_size = caps.optimal_workgroup_size(WorkloadType::ElementWise);
            let workgroups = (output_size as u32).div_ceil(optimal_wg_size);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        device.submit_commands(Some(encoder.finish()));

        Ok(Tensor::from_buffer(
            output_buffer,
            vec![output_size],
            device.clone(),
        ))
    }
}

impl Tensor {
    /// Compute Beta function B(a,b) for interleaved pairs
    /// Input: [a₀, b₀, a₁, b₁, ...]
    /// Output: [B(a₀,b₀), B(a₁,b₁), ...]
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn beta(self) -> Result<Self> {
        Beta::new(self).execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn get_test_device() -> Option<std::sync::Arc<crate::device::WgpuDevice>> {
        crate::device::test_pool::get_test_device_if_gpu_available().await
    }

    #[tokio::test]
    async fn test_beta_1_1() {
        let Some(device) = get_test_device().await else {
            return;
        };
        // B(1,1) = Γ(1)Γ(1)/Γ(2) = 1*1/1 = 1
        let input = Tensor::new(vec![1.0, 1.0], vec![2], device);
        let output = input.beta().unwrap();
        let result = output.to_vec().unwrap();
        assert!(
            (result[0] - 1.0).abs() < 0.01,
            "B(1,1) = {}, expected 1",
            result[0]
        );
    }

    #[tokio::test]
    async fn test_beta_2_2() {
        let Some(device) = get_test_device().await else {
            return;
        };
        // B(2,2) = Γ(2)Γ(2)/Γ(4) = 1*1/6 = 1/6 ≈ 0.1667
        let input = Tensor::new(vec![2.0, 2.0], vec![2], device);
        let output = input.beta().unwrap();
        let result = output.to_vec().unwrap();
        let expected = 1.0 / 6.0;
        assert!(
            (result[0] - expected).abs() < 0.01,
            "B(2,2) = {}, expected {}",
            result[0],
            expected
        );
    }

    #[tokio::test]
    async fn test_beta_multiple() {
        let Some(device) = get_test_device().await else {
            return;
        };
        // B(1,2) = 1/2, B(2,3) = 1/12, B(3,4) = 1/60
        let input = Tensor::new(vec![1.0, 2.0, 2.0, 3.0, 3.0, 4.0], vec![6], device);
        let output = input.beta().unwrap();
        let result = output.to_vec().unwrap();

        assert!(
            (result[0] - 0.5).abs() < 0.01,
            "B(1,2) = {}, expected 0.5",
            result[0]
        );
        assert!(
            (result[1] - 1.0 / 12.0).abs() < 0.01,
            "B(2,3) = {}, expected {}",
            result[1],
            1.0 / 12.0
        );
        assert!(
            (result[2] - 1.0 / 60.0).abs() < 0.01,
            "B(3,4) = {}, expected {}",
            result[2],
            1.0 / 60.0
        );
    }
}

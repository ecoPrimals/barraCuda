// SPDX-License-Identifier: AGPL-3.0-only
//! HERMITE - Physicist's Hermite polynomials - Pure WGSL
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

/// Hermite polynomial evaluator Hₙ(x)
pub struct Hermite {
    input: Tensor,
    n: u32,
}

impl Hermite {
    /// Create new Hermite polynomial operation for order n
    #[must_use]
    pub fn new(input: Tensor, n: u32) -> Self {
        Self { input, n }
    }

    fn wgsl_shader() -> &'static str {
        include_str!("../shaders/special/hermite.wgsl")
    }

    /// Execute Hermite polynomial evaluation on the input tensor.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let size: usize = self.input.shape().iter().product();

        let output_buffer = device.create_buffer_f32(size)?;

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            size: u32,
            n: u32,
        }

        let params = Params {
            size: size as u32,
            n: self.n,
        };
        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Hermite Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind_group_layout =
            device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Hermite Bind Group Layout"),
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
            label: Some("Hermite Bind Group"),
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

        let shader = device.compile_shader(Self::wgsl_shader(), Some("Hermite"));

        let pipeline_layout =
            device
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Hermite Pipeline Layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    immediate_size: 0,
                });

        let pipeline = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Hermite Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                cache: None,
                compilation_options: Default::default(),
            });

        let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("Hermite Encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Hermite Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&bind_group), &[]);
            let caps = DeviceCapabilities::from_device(device);
            let optimal_wg_size = caps.optimal_workgroup_size(WorkloadType::ElementWise);
            let workgroups = (size as u32).div_ceil(optimal_wg_size);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        device.submit_and_poll(Some(encoder.finish()));

        Ok(Tensor::from_buffer(
            output_buffer,
            self.input.shape().to_vec(),
            device.clone(),
        ))
    }
}

impl Tensor {
    /// Compute Hermite polynomial Hₙ(x) for each element
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn hermite(self, n: u32) -> Result<Self> {
        Hermite::new(self, n).execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn get_test_device() -> Option<std::sync::Arc<crate::device::WgpuDevice>> {
        crate::device::test_pool::get_test_device_if_gpu_available().await
    }

    #[tokio::test]
    async fn test_hermite_h0() {
        let Some(device) = get_test_device().await else {
            return;
        };
        let data = vec![0.0, 1.0, 2.0, -1.0, 0.5];
        let input = Tensor::new(data, vec![5], device);
        let output = input.hermite(0).unwrap();
        let result = output.to_vec().unwrap();
        // H₀(x) = 1 for all x
        for &v in &result {
            assert!((v - 1.0).abs() < 1e-5, "H₀ should be 1, got {v}");
        }
    }

    #[tokio::test]
    async fn test_hermite_h1() {
        let Some(device) = get_test_device().await else {
            return;
        };
        let data = vec![0.0, 1.0, 2.0, -1.0, 0.5];
        let input = Tensor::new(data.clone(), vec![5], device);
        let output = input.hermite(1).unwrap();
        let result = output.to_vec().unwrap();
        // H₁(x) = 2x
        for (i, &v) in result.iter().enumerate() {
            let expected = 2.0 * data[i];
            assert!(
                (v - expected).abs() < 1e-5,
                "H₁({}) = {}, expected {}",
                data[i],
                v,
                expected
            );
        }
    }

    #[tokio::test]
    async fn test_hermite_h2() {
        let Some(device) = get_test_device().await else {
            return;
        };
        let data = vec![0.0, 1.0, 2.0, -1.0, 0.5];
        let input = Tensor::new(data.clone(), vec![5], device);
        let output = input.hermite(2).unwrap();
        let result = output.to_vec().unwrap();
        // H₂(x) = 4x² - 2
        for (i, &v) in result.iter().enumerate() {
            let x = data[i];
            let expected = 4.0 * x * x - 2.0;
            assert!(
                (v - expected).abs() < 1e-4,
                "H₂({x}) = {v}, expected {expected}"
            );
        }
    }
}

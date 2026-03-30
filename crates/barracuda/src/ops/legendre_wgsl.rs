// SPDX-License-Identifier: AGPL-3.0-or-later
//! LEGENDRE - Legendre polynomials and associated functions - Pure WGSL
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

/// Legendre polynomial evaluator Pₙ(x) and associated Legendre Pₙᵐ(x)
pub struct Legendre {
    input: Tensor,
    n: u32,
    m: u32,
    is_associated: bool,
}

impl Legendre {
    /// Create new Legendre polynomial Pₙ(x)
    #[must_use]
    pub fn new(input: Tensor, n: u32) -> Self {
        Self {
            input,
            n,
            m: 0,
            is_associated: false,
        }
    }

    /// Create new associated Legendre function Pₙᵐ(x)
    #[must_use]
    pub fn associated(input: Tensor, n: u32, m: u32) -> Self {
        Self {
            input,
            n,
            m,
            is_associated: true,
        }
    }

    fn wgsl_shader() -> &'static str {
        include_str!("../shaders/special/legendre.wgsl")
    }

    /// Execute Legendre polynomial evaluation on the input tensor.
    /// # Errors
    /// Returns [`Err`] if buffer allocation fails, GPU dispatch fails, or the device is lost.
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let size: usize = self.input.shape().iter().product();

        let output_buffer = device.create_buffer_f32(size)?;

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            size: u32,
            n: u32,
            m: u32,
            is_assoc: u32,
        }

        let params = Params {
            size: size as u32,
            n: self.n,
            m: self.m,
            is_assoc: u32::from(self.is_associated),
        };
        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Legendre Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind_group_layout =
            device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Legendre Bind Group Layout"),
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
            label: Some("Legendre Bind Group"),
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

        let shader = device.compile_shader(Self::wgsl_shader(), Some("Legendre"));

        let pipeline_layout =
            device
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Legendre Pipeline Layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    immediate_size: 0,
                });

        let pipeline = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Legendre Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                cache: None,
                compilation_options: Default::default(),
            });

        let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("Legendre Encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Legendre Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&bind_group), &[]);
            let caps = DeviceCapabilities::from_device(device);
            let optimal_wg_size = caps.optimal_workgroup_size(WorkloadType::ElementWise);
            let workgroups = (size as u32).div_ceil(optimal_wg_size);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        device.submit_commands(Some(encoder.finish()));

        Ok(Tensor::from_buffer(
            output_buffer,
            self.input.shape().to_vec(),
            device.clone(),
        ))
    }
}

impl Tensor {
    /// Compute Legendre polynomial Pₙ(x) for each element
    /// # Errors
    /// Returns [`Err`] if buffer allocation fails, GPU dispatch fails, or the device is lost.
    pub fn legendre(self, n: u32) -> Result<Self> {
        Legendre::new(self, n).execute()
    }

    /// Compute associated Legendre function Pₙᵐ(x) for each element
    /// # Errors
    /// Returns [`Err`] if buffer allocation fails, GPU dispatch fails, or the device is lost.
    pub fn assoc_legendre(self, n: u32, m: u32) -> Result<Self> {
        Legendre::associated(self, n, m).execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_legendre_p0() {
        let device = crate::device::test_pool::get_test_device().await;
        let data = vec![-1.0, -0.5, 0.0, 0.5, 1.0];
        let input = Tensor::new(data, vec![5], device);
        let output = input.legendre(0).unwrap();
        let result = output.to_vec().unwrap();
        // P₀(x) = 1 for all x
        for &v in &result {
            assert!((v - 1.0).abs() < 1e-5, "P₀ should be 1, got {v}");
        }
    }

    #[tokio::test]
    async fn test_legendre_p1() {
        let device = crate::device::test_pool::get_test_device().await;
        let data = vec![-1.0, -0.5, 0.0, 0.5, 1.0];
        let input = Tensor::new(data.clone(), vec![5], device);
        let output = input.legendre(1).unwrap();
        let result = output.to_vec().unwrap();
        // P₁(x) = x
        for (i, &v) in result.iter().enumerate() {
            assert!(
                (v - data[i]).abs() < 1e-5,
                "P₁({}) = {}, expected {}",
                data[i],
                v,
                data[i]
            );
        }
    }

    #[tokio::test]
    async fn test_legendre_p2() {
        let device = crate::device::test_pool::get_test_device().await;
        let data = vec![-1.0, -0.5, 0.0, 0.5, 1.0];
        let input = Tensor::new(data.clone(), vec![5], device);
        let output = input.legendre(2).unwrap();
        let result = output.to_vec().unwrap();
        // P₂(x) = (3x² - 1) / 2
        for (i, &v) in result.iter().enumerate() {
            let x = data[i];
            let expected = (3.0 * x).mul_add(x, -1.0) / 2.0;
            assert!(
                (v - expected).abs() < 1e-4,
                "P₂({x}) = {v}, expected {expected}"
            );
        }
    }

    #[tokio::test]
    async fn test_assoc_legendre_p11() {
        let device = crate::device::test_pool::get_test_device().await;
        // P₁¹(x) = -sqrt(1 - x²) (Condon-Shortley)
        let data = vec![0.0, 0.5, -0.5];
        let input = Tensor::new(data.clone(), vec![3], device);
        let output = input.assoc_legendre(1, 1).unwrap();
        let result = output.to_vec().unwrap();
        for (i, &v) in result.iter().enumerate() {
            let x = data[i];
            let expected = -(1.0 - x * x).sqrt();
            assert!(
                (v - expected).abs() < 1e-4,
                "P₁¹({x}) = {v}, expected {expected}"
            );
        }
    }
}

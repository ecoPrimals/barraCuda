// SPDX-License-Identifier: AGPL-3.0-or-later
//! `NORM_PPF` - Inverse Normal CDF (Probit) - Pure WGSL
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

/// Inverse Normal CDF (Percent Point Function / Probit)
pub struct NormPpf {
    input: Tensor,
    mu: f32,
    sigma: f32,
}

impl NormPpf {
    /// Create standard normal inverse CDF (μ=0, σ=1)
    #[must_use]
    pub fn standard(input: Tensor) -> Self {
        Self {
            input,
            mu: 0.0,
            sigma: 1.0,
        }
    }

    /// Create general normal inverse CDF with custom μ, σ
    #[must_use]
    pub fn general(input: Tensor, mu: f32, sigma: f32) -> Self {
        Self { input, mu, sigma }
    }

    fn wgsl_shader() -> &'static str {
        static SHADER: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
            include_str!("../shaders/special/norm_ppf_f64.wgsl").to_string()
        });
        &SHADER
    }

    /// Execute inverse normal CDF (probit) on the input tensor.
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or device submission fails (e.g. device lost).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let size: usize = self.input.shape().iter().product();

        let output_buffer = device.create_buffer_f32(size)?;

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            size: u32,
            mu: f32,
            sigma: f32,
        }

        let params = Params {
            size: size as u32,
            mu: self.mu,
            sigma: self.sigma,
        };
        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("NormPpf Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind_group_layout =
            device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("NormPpf Bind Group Layout"),
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
            label: Some("NormPpf Bind Group"),
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

        let shader = device.compile_shader(Self::wgsl_shader(), Some("NormPpf"));

        let pipeline_layout =
            device
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("NormPpf Pipeline Layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    immediate_size: 0,
                });

        let pipeline = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("NormPpf Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                cache: None,
                compilation_options: Default::default(),
            });

        let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("NormPpf Encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("NormPpf Pass"),
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
    /// Compute standard normal inverse CDF Φ⁻¹(p) for each element
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or device submission fails (e.g. device lost).
    pub fn norm_ppf(self) -> Result<Self> {
        NormPpf::standard(self).execute()
    }

    /// Compute general normal inverse CDF with custom μ, σ
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or device submission fails (e.g. device lost).
    pub fn norm_ppf_params(self, mu: f32, sigma: f32) -> Result<Self> {
        NormPpf::general(self, mu, sigma).execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_norm_ppf_median() {
        let device = crate::device::test_pool::get_test_device().await;
        // Φ⁻¹(0.5) = 0
        let input = Tensor::new(vec![0.5], vec![1], device);
        let output = input.norm_ppf().unwrap();
        let result = output.to_vec().unwrap();
        assert!(
            result[0].abs() < 0.001,
            "Φ⁻¹(0.5) = {}, expected 0",
            result[0]
        );
    }

    #[tokio::test]
    async fn test_norm_ppf_quartiles() {
        let device = crate::device::test_pool::get_test_device().await;
        // Φ⁻¹(0.25) ≈ -0.6745, Φ⁻¹(0.75) ≈ 0.6745
        let input = Tensor::new(vec![0.25, 0.75], vec![2], device);
        let output = input.norm_ppf().unwrap();
        let result = output.to_vec().unwrap();
        assert!(
            (result[0] - (-0.6745)).abs() < 0.01,
            "Φ⁻¹(0.25) = {}, expected -0.6745",
            result[0]
        );
        assert!(
            (result[1] - 0.6745).abs() < 0.01,
            "Φ⁻¹(0.75) = {}, expected 0.6745",
            result[1]
        );
    }

    #[tokio::test]
    async fn test_norm_ppf_critical() {
        let device = crate::device::test_pool::get_test_device().await;
        // Φ⁻¹(0.025) ≈ -1.96, Φ⁻¹(0.975) ≈ 1.96
        let input = Tensor::new(vec![0.025, 0.975], vec![2], device);
        let output = input.norm_ppf().unwrap();
        let result = output.to_vec().unwrap();
        assert!(
            (result[0] - (-1.96)).abs() < 0.02,
            "Φ⁻¹(0.025) = {}, expected -1.96",
            result[0]
        );
        assert!(
            (result[1] - 1.96).abs() < 0.02,
            "Φ⁻¹(0.975) = {}, expected 1.96",
            result[1]
        );
    }

    #[tokio::test]
    async fn test_norm_ppf_general() {
        let device = crate::device::test_pool::get_test_device().await;
        // For N(μ=10, σ=2): Φ⁻¹(0.5) = μ = 10
        let input = Tensor::new(vec![0.5], vec![1], device);
        let output = input.norm_ppf_params(10.0, 2.0).unwrap();
        let result = output.to_vec().unwrap();
        assert!(
            (result[0] - 10.0).abs() < 0.01,
            "Φ⁻¹(0.5; 10, 2) = {}, expected 10",
            result[0]
        );
    }
}

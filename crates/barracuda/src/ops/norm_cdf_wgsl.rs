// SPDX-License-Identifier: AGPL-3.0-or-later
//! NORM_CDF - Normal distribution CDF and PDF - Pure WGSL
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
use wgpu::util::DeviceExt;

/// Normal distribution CDF Φ(x) and PDF φ(x)
pub struct NormCdf {
    input: Tensor,
    mu: f32,
    sigma: f32,
    compute_pdf: bool,
}

impl NormCdf {
    /// Create standard normal CDF operation (μ=0, σ=1)
    pub fn standard_cdf(input: Tensor) -> Self {
        Self {
            input,
            mu: 0.0,
            sigma: 1.0,
            compute_pdf: false,
        }
    }

    /// Create standard normal PDF operation (μ=0, σ=1)
    pub fn standard_pdf(input: Tensor) -> Self {
        Self {
            input,
            mu: 0.0,
            sigma: 1.0,
            compute_pdf: true,
        }
    }

    /// Create general normal CDF operation with custom μ, σ
    pub fn cdf(input: Tensor, mu: f32, sigma: f32) -> Self {
        Self {
            input,
            mu,
            sigma,
            compute_pdf: false,
        }
    }

    /// Create general normal PDF operation with custom μ, σ
    pub fn pdf(input: Tensor, mu: f32, sigma: f32) -> Self {
        Self {
            input,
            mu,
            sigma,
            compute_pdf: true,
        }
    }

    fn wgsl_shader() -> &'static str {
        static SHADER: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
            crate::shaders::precision::downcast_f64_to_f32_with_transcendentals(include_str!(
                "../shaders/special/norm_cdf_f64.wgsl"
            ))
        });
        &SHADER
    }

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
            mode: u32, // 0 = CDF, 1 = PDF
        }

        let params = Params {
            size: size as u32,
            mu: self.mu,
            sigma: self.sigma,
            mode: if self.compute_pdf { 1 } else { 0 },
        };
        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("NormCdf Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind_group_layout =
            device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("NormCdf Bind Group Layout"),
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
            label: Some("NormCdf Bind Group"),
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

        let shader = device.compile_shader(Self::wgsl_shader(), Some("NormCdf"));

        let pipeline_layout =
            device
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("NormCdf Pipeline Layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });

        let pipeline = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("NormCdf Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: "main",
                cache: None,
                compilation_options: Default::default(),
            });

        let mut encoder = device
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("NormCdf Encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("NormCdf Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
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
    /// Compute standard normal CDF Φ(x) for each element
    pub fn norm_cdf(self) -> Result<Self> {
        NormCdf::standard_cdf(self).execute()
    }

    /// Compute standard normal PDF φ(x) for each element
    pub fn norm_pdf(self) -> Result<Self> {
        NormCdf::standard_pdf(self).execute()
    }

    /// Compute general normal CDF Φ(x; μ, σ) for each element
    pub fn norm_cdf_params(self, mu: f32, sigma: f32) -> Result<Self> {
        NormCdf::cdf(self, mu, sigma).execute()
    }

    /// Compute general normal PDF φ(x; μ, σ) for each element
    pub fn norm_pdf_params(self, mu: f32, sigma: f32) -> Result<Self> {
        NormCdf::pdf(self, mu, sigma).execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn get_test_device() -> Option<std::sync::Arc<crate::device::WgpuDevice>> {
        crate::device::test_pool::get_test_device_if_gpu_available().await
    }

    #[tokio::test]
    async fn test_norm_cdf_zero() {
        let Some(device) = get_test_device().await else {
            return;
        };
        // Φ(0) = 0.5
        let input = Tensor::new(vec![0.0], vec![1], device.clone());
        let output = input.norm_cdf().unwrap();
        let result = output.to_vec().unwrap();
        assert!(
            (result[0] - 0.5).abs() < 0.001,
            "Φ(0) = {}, expected 0.5",
            result[0]
        );
    }

    #[tokio::test]
    async fn test_norm_cdf_critical() {
        let Some(device) = get_test_device().await else {
            return;
        };
        // Φ(-1.96) ≈ 0.025, Φ(1.96) ≈ 0.975
        let input = Tensor::new(vec![-1.96, 1.96], vec![2], device.clone());
        let output = input.norm_cdf().unwrap();
        let result = output.to_vec().unwrap();
        assert!(
            (result[0] - 0.025).abs() < 0.01,
            "Φ(-1.96) = {}, expected ~0.025",
            result[0]
        );
        assert!(
            (result[1] - 0.975).abs() < 0.01,
            "Φ(1.96) = {}, expected ~0.975",
            result[1]
        );
    }

    #[tokio::test]
    async fn test_norm_pdf_peak() {
        let Some(device) = get_test_device().await else {
            return;
        };
        // φ(0) = 1/√(2π) ≈ 0.3989
        let input = Tensor::new(vec![0.0], vec![1], device.clone());
        let output = input.norm_pdf().unwrap();
        let result = output.to_vec().unwrap();
        let expected = 1.0 / (2.0 * std::f32::consts::PI).sqrt();
        assert!(
            (result[0] - expected).abs() < 0.001,
            "φ(0) = {}, expected {}",
            result[0],
            expected
        );
    }

    #[tokio::test]
    async fn test_norm_cdf_general() {
        let Some(device) = get_test_device().await else {
            return;
        };
        // Φ(μ; μ, σ) = 0.5 for any σ > 0
        let input = Tensor::new(vec![5.0], vec![1], device.clone());
        let output = input.norm_cdf_params(5.0, 2.0).unwrap();
        let result = output.to_vec().unwrap();
        assert!(
            (result[0] - 0.5).abs() < 0.001,
            "Φ(5; 5, 2) = {}, expected 0.5",
            result[0]
        );
    }
}

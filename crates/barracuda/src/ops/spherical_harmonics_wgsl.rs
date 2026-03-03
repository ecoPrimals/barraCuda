//! Real spherical harmonics Y_l^m(theta, phi) for multipole expansion - Pure WGSL
//!
//! Deep Debt Principles:
//! - Self-knowledge: Operation knows its computation
//! - Zero hardcoding: Hardware-agnostic implementation
//! - Modern idiomatic Rust: Safe, zero unsafe code
//! - Complete implementation: Production-ready, no mocks
//! - Hardware-agnostic: Pure WGSL for universal compute

use crate::device::DeviceCapabilities;
use crate::error::Result;
use crate::tensor::Tensor;
use wgpu::util::DeviceExt;

/// Real spherical harmonics Y_l^m(theta, phi).
/// theta_phi: interleaved [theta0, phi0, theta1, phi1, ...]
/// l: degree (0..6), m: order (can be negative)
pub struct SphericalHarmonics {
    theta_phi: Tensor,
    l: u32,
    m: i32,
}

impl SphericalHarmonics {
    pub fn new(theta_phi: Tensor, l: u32, m: i32) -> Self {
        Self { theta_phi, l, m }
    }

    fn wgsl_shader() -> &'static str {
        include_str!("../shaders/special/spherical_harmonics.wgsl")
    }

    pub fn execute(self) -> Result<Tensor> {
        let device = self.theta_phi.device();
        let total_elements: usize = self.theta_phi.shape().iter().product();
        assert!(
            total_elements.is_multiple_of(2),
            "theta_phi must have even length (theta, phi pairs)"
        );
        let size = total_elements / 2;

        if size == 0 {
            return Ok(Tensor::new(vec![], vec![0], device.clone()));
        }

        let output_buffer = device.create_buffer_f32(size)?;

        let abs_m = self.m.unsigned_abs();
        let m_is_positive = if self.m > 0 { 1u32 } else { 0u32 };

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            size: u32,
            l: u32,
            abs_m: u32,
            m_is_positive: u32,
        }

        let params = Params {
            size: size as u32,
            l: self.l,
            abs_m,
            m_is_positive,
        };
        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("SphericalHarmonics Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind_group_layout =
            device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("SphericalHarmonics Bind Group Layout"),
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
            label: Some("SphericalHarmonics Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.theta_phi.buffer().as_entire_binding(),
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

        let shader = device.compile_shader(Self::wgsl_shader(), Some("SphericalHarmonics"));

        let pipeline_layout =
            device
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("SphericalHarmonics Pipeline Layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });

        let pipeline = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("SphericalHarmonics Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: "main",
                cache: None,
                compilation_options: Default::default(),
            });

        let mut encoder = device
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("SphericalHarmonics Encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("SphericalHarmonics Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let caps = DeviceCapabilities::from_device(device);
            let workgroups = caps.dispatch_1d(size as u32);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        device.submit_and_poll(Some(encoder.finish()));

        Ok(Tensor::from_buffer(
            output_buffer,
            vec![size],
            device.clone(),
        ))
    }
}

impl Tensor {
    /// Compute real spherical harmonic Y_l^m at (theta, phi) points.
    /// theta_phi: interleaved [theta0, phi0, theta1, phi1, ...]
    pub fn spherical_harmonics(self, l: u32, m: i32) -> Result<Self> {
        SphericalHarmonics::new(self, l, m).execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    async fn get_test_device() -> Option<Arc<crate::device::WgpuDevice>> {
        crate::device::test_pool::get_test_device_if_gpu_available().await
    }

    #[tokio::test]
    async fn test_spherical_harmonics_y00() {
        let Some(device) = get_test_device().await else {
            return;
        };
        // Y_0^0 = 1/(2*sqrt(pi)) = 0.282094...
        let theta_phi = vec![0.0f32, 0.0, 1.0, 2.0]; // two points
        let input = Tensor::new(theta_phi, vec![4], device.clone());
        let output = input.spherical_harmonics(0, 0).unwrap();
        let result = output.to_vec().unwrap();
        let expected = 0.5 / std::f32::consts::PI.sqrt(); // Y_0^0 = 1/sqrt(4*pi)
        assert!((result[0] - expected).abs() < 1e-5);
        assert!((result[1] - expected).abs() < 1e-5);
    }
}

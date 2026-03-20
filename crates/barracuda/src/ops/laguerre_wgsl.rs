// SPDX-License-Identifier: AGPL-3.0-or-later
//! LAGUERRE - Generalized Laguerre polynomials - Pure WGSL
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

/// Generalized Laguerre polynomial evaluator `L_n^(α)(x)`
pub struct Laguerre {
    input: Tensor,
    n: u32,
    alpha: f32,
}

impl Laguerre {
    /// Create new Laguerre polynomial operation for degree n and parameter α
    #[must_use]
    pub fn new(input: Tensor, n: u32, alpha: f32) -> Self {
        Self { input, n, alpha }
    }

    /// Create simple Laguerre polynomial `L_n(x)` = `L_n^(0)(x)`
    #[must_use]
    pub fn simple(input: Tensor, n: u32) -> Self {
        Self::new(input, n, 0.0)
    }

    fn wgsl_shader() -> &'static str {
        include_str!("../shaders/special/laguerre.wgsl")
    }

    /// Execute Laguerre polynomial evaluation on the input tensor.
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
            alpha: f32,
        }

        let params = Params {
            size: size as u32,
            n: self.n,
            alpha: self.alpha,
        };
        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Laguerre Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind_group_layout =
            device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Laguerre Bind Group Layout"),
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
            label: Some("Laguerre Bind Group"),
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

        let shader = device.compile_shader(Self::wgsl_shader(), Some("Laguerre"));

        let pipeline_layout =
            device
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Laguerre Pipeline Layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    immediate_size: 0,
                });

        let pipeline = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Laguerre Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                cache: None,
                compilation_options: Default::default(),
            });

        let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("Laguerre Encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Laguerre Pass"),
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
    /// Compute generalized Laguerre polynomial `L_n^(α)(x)` for each element
    /// # Errors
    /// Returns [`Err`] if buffer allocation fails, GPU dispatch fails, or the device is lost.
    pub fn laguerre(self, n: u32, alpha: f32) -> Result<Self> {
        Laguerre::new(self, n, alpha).execute()
    }

    /// Compute simple Laguerre polynomial `L_n(x)` = `L_n^(0)(x)` for each element
    /// # Errors
    /// Returns [`Err`] if buffer allocation fails, GPU dispatch fails, or the device is lost.
    pub fn laguerre_simple(self, n: u32) -> Result<Self> {
        Laguerre::simple(self, n).execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn get_test_device() -> Option<std::sync::Arc<crate::device::WgpuDevice>> {
        crate::device::test_pool::get_test_device_if_gpu_available().await
    }

    #[tokio::test]
    async fn test_laguerre_l0() {
        let Some(device) = get_test_device().await else {
            return;
        };
        let data = vec![0.0, 1.0, 2.0, 5.0];
        let input = Tensor::new(data, vec![4], device);
        let output = input.laguerre_simple(0).unwrap();
        let result = output.to_vec().unwrap();
        // L_0(x) = 1 for all x
        for &v in &result {
            assert!((v - 1.0).abs() < 1e-5, "L_0 should be 1, got {v}");
        }
    }

    #[tokio::test]
    async fn test_laguerre_l1() {
        let Some(device) = get_test_device().await else {
            return;
        };
        let data = vec![0.0, 1.0, 3.0];
        let input = Tensor::new(data.clone(), vec![3], device);
        let output = input.laguerre_simple(1).unwrap();
        let result = output.to_vec().unwrap();
        // L_1(x) = 1 - x
        let expected = [1.0, 0.0, -2.0];
        for (i, &v) in result.iter().enumerate() {
            assert!(
                (v - expected[i]).abs() < 1e-5,
                "L_1({}) = {}, expected {}",
                data[i],
                v,
                expected[i]
            );
        }
    }

    #[tokio::test]
    async fn test_laguerre_l2() {
        let Some(device) = get_test_device().await else {
            return;
        };
        let data = vec![0.0, 1.0, 2.0];
        let input = Tensor::new(data.clone(), vec![3], device);
        let output = input.laguerre_simple(2).unwrap();
        let result = output.to_vec().unwrap();
        // L_2(x) = (x² - 4x + 2) / 2
        for (i, &v) in result.iter().enumerate() {
            let x = data[i];
            let expected = f32::midpoint(x * x - 4.0 * x, 2.0);
            assert!(
                (v - expected).abs() < 1e-4,
                "L_2({x}) = {v}, expected {expected}"
            );
        }
    }

    #[tokio::test]
    async fn test_laguerre_generalized() {
        let Some(device) = get_test_device().await else {
            return;
        };
        // L_1^(1)(x) = 2 - x
        let data = vec![0.0, 2.0];
        let input = Tensor::new(data, vec![2], device);
        let output = input.laguerre(1, 1.0).unwrap();
        let result = output.to_vec().unwrap();
        assert!(
            (result[0] - 2.0).abs() < 1e-5,
            "L_1^(1)(0) = {}, expected 2",
            result[0]
        );
        assert!(
            (result[1] - 0.0).abs() < 1e-5,
            "L_1^(1)(2) = {}, expected 0",
            result[1]
        );
    }
}

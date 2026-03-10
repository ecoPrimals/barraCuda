// SPDX-License-Identifier: AGPL-3.0-only
//! `RANDOM_UNIFORM` - Uniform random sampling - Pure WGSL
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
use std::sync::Arc;

/// Uniform random sampling (GPU accelerated)
pub struct RandomUniformGpu {
    device: Arc<crate::device::WgpuDevice>,
    n_samples: u32,
    bounds: Vec<f32>, // Interleaved [lo0, hi0, lo1, hi1, ...]
    seed: u32,
}

impl RandomUniformGpu {
    /// Create new uniform random sampler
    #[must_use]
    pub fn new(
        device: Arc<crate::device::WgpuDevice>,
        n_samples: u32,
        bounds: &[(f32, f32)],
        seed: u32,
    ) -> Self {
        let bounds_flat: Vec<f32> = bounds.iter().flat_map(|&(lo, hi)| [lo, hi]).collect();
        Self {
            device,
            n_samples,
            bounds: bounds_flat,
            seed,
        }
    }

    fn wgsl_shader() -> &'static str {
        static SHADER: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
            include_str!("../shaders/sample/random_uniform_f64.wgsl").to_string()
        });
        &SHADER
    }

    /// Generate uniform random samples
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn generate(self) -> Result<Tensor> {
        let device = &self.device;
        let n_dims = self.bounds.len() / 2;
        let output_size = (self.n_samples as usize) * n_dims;

        let output_buffer = device.create_buffer_f32(output_size)?;

        // Create bounds buffer
        let bounds_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("RandomUniform Bounds"),
                contents: bytemuck::cast_slice(&self.bounds),
                usage: wgpu::BufferUsages::STORAGE,
            });

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            n_samples: u32,
            n_dims: u32,
            seed: u32,
            _pad: u32,
        }

        let params = Params {
            n_samples: self.n_samples,
            n_dims: n_dims as u32,
            seed: self.seed,
            _pad: 0,
        };
        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("RandomUniform Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind_group_layout =
            device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("RandomUniform Bind Group Layout"),
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
            label: Some("RandomUniform Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: bounds_buffer.as_entire_binding(),
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

        let shader = device.compile_shader(Self::wgsl_shader(), Some("RandomUniform"));

        let pipeline_layout =
            device
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("RandomUniform Pipeline Layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    immediate_size: 0,
                });

        let pipeline = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("RandomUniform Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                cache: None,
                compilation_options: Default::default(),
            });

        let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("RandomUniform Encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("RandomUniform Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&bind_group), &[]);
            let caps = DeviceCapabilities::from_device(device);
            let optimal_wg_size = caps.optimal_workgroup_size(WorkloadType::ElementWise);
            let workgroups = self.n_samples.div_ceil(optimal_wg_size);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        device.submit_and_poll(Some(encoder.finish()));

        Ok(Tensor::from_buffer(
            output_buffer,
            vec![self.n_samples as usize, n_dims],
            self.device.clone(),
        ))
    }
}

/// Generate uniform random samples on GPU
///
/// # Errors
///
/// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
/// readback fails (e.g. device lost or out of memory).
pub fn random_uniform_gpu(
    device: Arc<crate::device::WgpuDevice>,
    n_samples: u32,
    bounds: &[(f32, f32)],
    seed: u32,
) -> Result<Tensor> {
    RandomUniformGpu::new(device, n_samples, bounds, seed).generate()
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn get_test_device() -> Option<Arc<crate::device::WgpuDevice>> {
        crate::device::test_pool::get_test_device_if_gpu_available().await
    }

    #[tokio::test]
    async fn test_random_uniform_basic() {
        let Some(device) = get_test_device().await else {
            return;
        };
        let bounds = vec![(0.0, 1.0), (-1.0, 1.0)];
        let result = random_uniform_gpu(device, 100, &bounds, 42).unwrap();
        let data = result.to_vec().unwrap();

        // Should have 100 * 2 = 200 values
        assert_eq!(data.len(), 200);

        // Check bounds for each sample
        for i in 0..100 {
            let x = data[i * 2];
            let y = data[i * 2 + 1];
            assert!((0.0..=1.0).contains(&x), "x={x} out of [0,1]");
            assert!((-1.0..=1.0).contains(&y), "y={y} out of [-1,1]");
        }
    }

    #[tokio::test]
    async fn test_random_uniform_different_seeds() {
        let Some(device) = get_test_device().await else {
            return;
        };
        let bounds = vec![(0.0, 1.0)];

        let r1 = random_uniform_gpu(device.clone(), 10, &bounds, 42).unwrap();
        let r2 = random_uniform_gpu(device, 10, &bounds, 99).unwrap();

        let d1 = r1.to_vec().unwrap();
        let d2 = r2.to_vec().unwrap();

        // Different seeds should give different results
        let different = d1.iter().zip(d2.iter()).any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(different, "Different seeds should give different results");
    }
}

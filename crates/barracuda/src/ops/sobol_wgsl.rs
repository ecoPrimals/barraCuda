// SPDX-License-Identifier: AGPL-3.0-only
//! SOBOL - Quasi-random Sobol sequences - Pure WGSL
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

/// Sobol quasi-random sequence generator (GPU accelerated)
pub struct SobolGpu {
    device: Arc<crate::device::WgpuDevice>,
    n_samples: u32,
    n_dims: u32,
    skip: u32,
}

impl SobolGpu {
    /// Create new Sobol sequence generator
    #[must_use]
    pub fn new(device: Arc<crate::device::WgpuDevice>, n_samples: u32, n_dims: u32) -> Self {
        Self {
            device,
            n_samples,
            n_dims,
            skip: 0,
        }
    }

    /// Skip the first `n` points (useful for continuing a sequence)
    #[must_use]
    pub fn with_skip(mut self, skip: u32) -> Self {
        self.skip = skip;
        self
    }

    fn wgsl_shader() -> &'static str {
        static SHADER: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
            include_str!("../shaders/sample/sobol_f64.wgsl").to_string()
        });
        &SHADER
    }

    /// Generate the Sobol sequence
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or device submission fails (e.g. device lost).
    pub fn generate(self) -> Result<Tensor> {
        let device = &self.device;
        let output_size = (self.n_samples * self.n_dims) as usize;

        let output_buffer = device.create_buffer_f32(output_size)?;

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            n_samples: u32,
            n_dims: u32,
            skip: u32,
            _pad: u32,
        }

        let params = Params {
            n_samples: self.n_samples,
            n_dims: self.n_dims,
            skip: self.skip,
            _pad: 0,
        };
        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Sobol Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind_group_layout =
            device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Sobol Bind Group Layout"),
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

        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Sobol Bind Group"),
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

        let shader = device.compile_shader(Self::wgsl_shader(), Some("Sobol"));

        let pipeline_layout =
            device
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Sobol Pipeline Layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    immediate_size: 0,
                });

        let pipeline = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Sobol Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                cache: None,
                compilation_options: Default::default(),
            });

        let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("Sobol Encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Sobol Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&bind_group), &[]);
            let caps = DeviceCapabilities::from_device(device);
            let optimal_wg_size = caps.optimal_workgroup_size(WorkloadType::ElementWise);
            let workgroups = self.n_samples.div_ceil(optimal_wg_size);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        device.submit_commands(Some(encoder.finish()));

        Ok(Tensor::from_buffer(
            output_buffer,
            vec![self.n_samples as usize, self.n_dims as usize],
            self.device.clone(),
        ))
    }
}

/// Generate Sobol sequence on GPU
///
/// # Errors
///
/// Returns [`Err`] if buffer allocation, GPU dispatch, or device submission fails (e.g. device lost).
pub fn sobol_gpu(
    device: Arc<crate::device::WgpuDevice>,
    n_samples: u32,
    n_dims: u32,
) -> Result<Tensor> {
    SobolGpu::new(device, n_samples, n_dims).generate()
}

/// Generate Sobol sequence with skip on GPU
///
/// # Errors
///
/// Returns [`Err`] if buffer allocation, GPU dispatch, or device submission fails (e.g. device lost).
pub fn sobol_gpu_skip(
    device: Arc<crate::device::WgpuDevice>,
    n_samples: u32,
    n_dims: u32,
    skip: u32,
) -> Result<Tensor> {
    SobolGpu::new(device, n_samples, n_dims)
        .with_skip(skip)
        .generate()
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn get_test_device() -> Option<Arc<crate::device::WgpuDevice>> {
        crate::device::test_pool::get_test_device_if_gpu_available().await
    }

    #[tokio::test]
    async fn test_sobol_basic() {
        let Some(device) = get_test_device().await else {
            return;
        };
        let result = sobol_gpu(device, 16, 2).unwrap();
        let data = result.to_vec().unwrap();

        // Should have 16 * 2 = 32 values
        assert_eq!(data.len(), 32);

        // All values should be in [0, 1)
        for &v in &data {
            assert!((0.0..1.0).contains(&v), "Value {v} out of range");
        }

        // First point should be 0
        assert!(data[0].abs() < 1e-6, "First point should be 0");
    }

    #[tokio::test]
    async fn test_sobol_low_discrepancy() {
        let Some(device) = get_test_device().await else {
            return;
        };
        // Generate 64 points in 1D
        let result = sobol_gpu(device, 64, 1).unwrap();
        let data = result.to_vec().unwrap();

        // Sobol should cover intervals more evenly than random
        // Check that we have points in each quarter
        let q1 = data.iter().filter(|&&x| x < 0.25).count();
        let q2 = data.iter().filter(|&&x| (0.25..0.5).contains(&x)).count();
        let q3 = data.iter().filter(|&&x| (0.5..0.75).contains(&x)).count();
        let q4 = data.iter().filter(|&&x| x >= 0.75).count();

        // Each quarter should have roughly 16 points (allow some slack)
        assert!((10..=22).contains(&q1), "Q1 has {q1} points");
        assert!((10..=22).contains(&q2), "Q2 has {q2} points");
        assert!((10..=22).contains(&q3), "Q3 has {q3} points");
        assert!((10..=22).contains(&q4), "Q4 has {q4} points");
    }

    #[tokio::test]
    async fn test_sobol_with_skip() {
        let Some(device) = get_test_device().await else {
            return;
        };
        // Generate with skip=8 should give different first point
        let result = sobol_gpu_skip(device, 8, 1, 8).unwrap();
        let data = result.to_vec().unwrap();

        // With skip=8, first point won't be 0
        assert!(
            data[0].abs() > 1e-6,
            "First point after skip shouldn't be 0"
        );
    }
}

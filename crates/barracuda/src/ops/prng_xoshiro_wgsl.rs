// SPDX-License-Identifier: AGPL-3.0-or-later
//! PRNG Xoshiro128** - High-quality pseudorandom f64 generator - Pure WGSL
//!
//! Deep Debt Principles:
//! - Self-knowledge: Operation knows its computation
//! - Zero hardcoding: Hardware-agnostic implementation
//! - Modern idiomatic Rust: Safe, zero unsafe code
//! - Complete implementation: Production-ready, no mocks
//! - Hardware-agnostic: Pure WGSL for universal compute
//!
//! f64 pipeline: seeds as `array<u32>` (1 per output, expanded to 4-stride),
//! output as `array<f64>` in [0, 1).

use crate::device::DeviceCapabilities;
use crate::error::Result;
use crate::tensor::Tensor;

/// Xoshiro128** PRNG for GPU random number generation.
pub struct PrngXoshiro {
    seeds: Tensor,
    offset: u32,
}

impl PrngXoshiro {
    /// Creates a new PRNG with the given seeds tensor (u32) and offset.
    #[must_use]
    pub fn new(seeds: Tensor, offset: u32) -> Self {
        Self { seeds, offset }
    }

    /// Xoshiro128** stateful PRNG (neuralSpring): per-thread state, `n_samples` per thread.
    #[must_use]
    pub fn wgsl_xoshiro128ss() -> &'static str {
        static SHADER: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
            include_str!("../shaders/misc/xoshiro128ss_f64.wgsl").to_string()
        });
        std::sync::LazyLock::force(&SHADER).as_str()
    }

    /// WGSL kernel for Xoshiro PRNG (f32 variant).
    pub const WGSL_PRNG_XOSHIRO_F32: &str = include_str!("../shaders/misc/prng_xoshiro.wgsl");

    /// f64 version for universal math library portability.
    #[must_use]
    pub fn wgsl_shader_f64() -> &'static str {
        include_str!("../shaders/misc/prng_xoshiro_f64.wgsl")
    }

    /// Executes the PRNG and returns random values in [0, 1).
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.seeds.device();
        let seed_count: usize = self.seeds.shape().iter().product();

        if seed_count == 0 {
            return Ok(Tensor::new(vec![], vec![0], device.clone()));
        }

        // f64 shader expects 4 u32s per output (seed_base = idx * 4); expand 1→4 stride
        let seeds_data = device.read_buffer_u32(self.seeds.buffer(), seed_count)?;
        let expanded: Vec<u32> = (0..seed_count)
            .flat_map(|i| [seeds_data[i], 0u32, 0u32, 0u32])
            .collect();
        let seeds_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("PRNG Xoshiro seeds expanded"),
                contents: bytemuck::cast_slice(&expanded),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let output_buffer = device.create_buffer_f64(seed_count)?;

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            size: u32,
            offset: u32,
        }

        let params = Params {
            size: seed_count as u32,
            offset: self.offset,
        };
        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("PRNG Xoshiro Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind_group_layout =
            device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("PRNG Xoshiro f64 Bind Group Layout"),
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
            label: Some("PRNG Xoshiro Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: seeds_buffer.as_entire_binding(),
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

        let shader = device.compile_shader_f64(Self::wgsl_shader_f64(), Some("PRNG Xoshiro f64"));

        let pipeline_layout =
            device
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("PRNG Xoshiro Pipeline Layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    immediate_size: 0,
                });

        let pipeline = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("PRNG Xoshiro Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                cache: None,
                compilation_options: Default::default(),
            });

        let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("PRNG Xoshiro Encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("PRNG Xoshiro Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&bind_group), &[]);
            let caps = DeviceCapabilities::from_device(device);
            let workgroups = caps.dispatch_1d(seed_count as u32);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        device.submit_commands(Some(encoder.finish()));

        Ok(Tensor::from_buffer(
            output_buffer,
            self.seeds.shape().to_vec(),
            device.clone(),
        ))
    }
}

impl Tensor {
    /// Generate random f64 values in [0, 1) using xoshiro128** PRNG.
    /// Seeds tensor must contain u32 data (use `Tensor::from_data_pod` with u32).
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn prng_xoshiro(self, offset: u32) -> Result<Self> {
        PrngXoshiro::new(self, offset).execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    async fn get_test_device() -> Option<Arc<crate::device::WgpuDevice>> {
        crate::device::test_pool::get_test_device_if_f64_gpu_available().await
    }

    #[tokio::test]
    async fn test_prng_xoshiro() {
        let Some(device) = get_test_device().await else {
            return;
        };
        // Seeds as u32 (one per output element)
        let seeds: Vec<u32> = vec![1, 2, 3, 4, 5, 100, 200, 300];
        let seeds_tensor = Tensor::from_data_pod(&seeds, vec![8], device).unwrap();
        let output = seeds_tensor.prng_xoshiro(0).unwrap();
        let result = output.to_f64_vec().unwrap();
        assert_eq!(result.len(), 8);
        assert!(
            result
                .iter()
                .all(|&x| (0.0..1.0).contains(&x) && x.is_finite())
        );
    }
}

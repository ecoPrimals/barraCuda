// SPDX-License-Identifier: AGPL-3.0-or-later
//! Single-dispatch Jacobi eigensolver — one GPU submit for n≤32
//!
//! Eliminates poll bottleneck: instead of ~8000 queue.submit() calls per batch,
//! uses exactly ONE dispatch. Limited to n≤32 by workgroup shared memory.

use super::params::SingleDispatchParams;
use super::BatchedEighGpu;
use crate::device::capabilities::{EigensolveStrategy, GpuDriverProfile};
use crate::device::WgpuDevice;
use crate::error::{BarracudaError, Result};
use crate::shaders::precision::ShaderTemplate;
use std::sync::Arc;

impl BatchedEighGpu {
    /// **SINGLE-DISPATCH** batched eigenvalue decomposition — eliminates poll bottleneck
    ///
    /// For n=12, batch=40: Previous 7,920 submits → This: 1 submit.
    /// Maximum n=32 (workgroup shared memory limit).
    pub fn execute_single_dispatch(
        device: Arc<WgpuDevice>,
        data: &[f64],
        n: usize,
        batch_size: usize,
        max_sweeps: u32,
        tolerance: f64,
    ) -> Result<(Vec<f64>, Vec<f64>)> {
        const MAX_N: usize = 32;

        if n > MAX_N {
            return Err(BarracudaError::InvalidInput {
                message: format!(
                    "Single-dispatch eigensolve limited to n≤{MAX_N}, got n={n}. Use execute_f64() for larger matrices."
                ),
            });
        }

        if data.len() != batch_size * n * n {
            return Err(BarracudaError::InvalidInput {
                message: format!(
                    "Data length {} does not match batch_size={} × n²={}",
                    data.len(),
                    batch_size,
                    n * n
                ),
            });
        }

        let a_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("SingleDispatch A"),
                contents: bytemuck::cast_slice(data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });

        let v_size = (batch_size * n * n * 8) as u64;
        let v_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SingleDispatch V"),
            size: v_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let eig_size = (batch_size * n * 8) as u64;
        let eig_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("SingleDispatch Eigenvalues"),
            size: eig_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params = SingleDispatchParams {
            n: n as u32,
            batch_size: batch_size as u32,
            max_sweeps,
            tolerance: tolerance as f32,
        };
        let params_buffer = device.create_uniform_buffer("SingleDispatch Params", &params);

        // Determine optimal wave/warp size from driver profile
        let wave_size = match GpuDriverProfile::from_device(&device).optimal_eigensolve_strategy() {
            EigensolveStrategy::WarpPacked { wg_size } => wg_size,
            EigensolveStrategy::WavePacked { wave_size } => wave_size,
            EigensolveStrategy::Standard => 1,
        };
        let patched_shader = ShaderTemplate::patch_warp_size(
            Self::single_dispatch_shader_for_device(&device),
            wave_size,
        );
        let shader =
            device.compile_shader_f64(&patched_shader, Some("Batched Eigh Single-Dispatch f64"));

        let bgl = device
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("SingleDispatch BGL"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
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
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let pipeline_layout =
            device
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("SingleDispatch PL"),
                    bind_group_layouts: &[&bgl],
                    immediate_size: 0,
                });

        let pipeline = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("SingleDispatch Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("batched_eigh_single_dispatch"),
                cache: None,
                compilation_options: Default::default(),
            });

        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SingleDispatch BG"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: a_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: v_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: eig_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("SingleDispatch Encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("SingleDispatch Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&bind_group), &[]);
            pass.dispatch_workgroups((batch_size as u32).div_ceil(wave_size), 1, 1);
        }
        device.submit_and_poll(Some(encoder.finish()));

        let eigenvalues = device.read_f64_buffer(&eig_buffer, batch_size * n)?;
        let eigenvectors = device.read_f64_buffer(&v_buffer, batch_size * n * n)?;

        Ok((eigenvalues, eigenvectors))
    }

    /// **SINGLE-DISPATCH** buffer-based eigensolve — no CPU readback
    pub fn execute_single_dispatch_buffers(
        device: &Arc<WgpuDevice>,
        matrices_buffer: &wgpu::Buffer,
        eigenvalues_buffer: &wgpu::Buffer,
        eigenvectors_buffer: &wgpu::Buffer,
        n: usize,
        batch_size: usize,
        max_sweeps: u32,
        tolerance: f64,
    ) -> Result<()> {
        const MAX_N: usize = 32;

        if n > MAX_N {
            return Err(BarracudaError::InvalidInput {
                message: format!("Single-dispatch eigensolve limited to n≤{MAX_N}, got n={n}"),
            });
        }

        let params = SingleDispatchParams {
            n: n as u32,
            batch_size: batch_size as u32,
            max_sweeps,
            tolerance: tolerance as f32,
        };
        let params_buffer =
            device.create_uniform_buffer("SingleDispatch Params (buffers)", &params);

        let wave_size = match GpuDriverProfile::from_device(device).optimal_eigensolve_strategy() {
            EigensolveStrategy::WarpPacked { wg_size } => wg_size,
            EigensolveStrategy::WavePacked { wave_size } => wave_size,
            EigensolveStrategy::Standard => 1,
        };
        let patched_shader = ShaderTemplate::patch_warp_size(
            Self::single_dispatch_shader_for_device(device),
            wave_size,
        );
        let shader = device.compile_shader_f64(
            &patched_shader,
            Some("Batched Eigh Single-Dispatch f64 (buffers)"),
        );

        let bgl = device
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("SingleDispatch BGL (buffers)"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
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
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let pipeline_layout =
            device
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("SingleDispatch PL (buffers)"),
                    bind_group_layouts: &[&bgl],
                    immediate_size: 0,
                });

        let pipeline = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("SingleDispatch Pipeline (buffers)"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("batched_eigh_single_dispatch"),
                cache: None,
                compilation_options: Default::default(),
            });

        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SingleDispatch BG (buffers)"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: matrices_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: eigenvectors_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: eigenvalues_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("SingleDispatch Encoder (buffers)"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("SingleDispatch Pass (buffers)"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&bind_group), &[]);
            pass.dispatch_workgroups((batch_size as u32).div_ceil(wave_size), 1, 1);
        }
        device.submit_and_poll(Some(encoder.finish()));

        Ok(())
    }
}

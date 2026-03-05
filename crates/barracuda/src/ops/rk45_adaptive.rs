// SPDX-License-Identifier: AGPL-3.0-or-later

//! Adaptive Dormand-Prince RK45 for regulatory networks — GPU kernel (f64).
//!
//! Single adaptive step of the Dormand-Prince 5(4) embedded pair.
//! Each thread handles one independent ODE system with Hill function kinetics.
//!
//! Output: new state (5th order) and per-variable absolute error.
//! Host uses error to adapt step size:
//!   h_new = h × min(5, max(0.2, 0.9 × (tol/err)^0.2))
//!
//! **Provenance**: neuralSpring metalForge → toadStool absorption (Feb 2026)
//! **Papers**: 020 (regulatory network), 021 (signal integration)

use std::sync::Arc;

use wgpu::util::DeviceExt;

use crate::device::capabilities::WORKGROUP_SIZE_COMPACT;
use crate::device::WgpuDevice;

/// WGSL source for adaptive RK45 (f32).
pub const WGSL_RK45_ADAPTIVE: &str = include_str!("../shaders/numerical/rk45_adaptive.wgsl");

/// f64 version for universal math library portability.
pub const WGSL_RK45_ADAPTIVE_F64: &str =
    include_str!("../shaders/numerical/rk45_adaptive_f64.wgsl");

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Rk45Params {
    n_systems: u32,
    dim: u32,
    n_coeffs: u32,
    _pad: u32,
    dt: f64,
    _pad2: f64,
}

/// Adaptive Dormand-Prince RK45 GPU kernel for regulatory network ODEs.
pub struct Rk45AdaptiveGpu {
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    device: Arc<WgpuDevice>,
}

impl Rk45AdaptiveGpu {
    /// Create an adaptive RK45 GPU kernel for regulatory network ODEs.
    pub fn new(device: Arc<WgpuDevice>) -> Self {
        let d = device.device();

        let bgl = d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("RK45 BGL"),
            entries: &[
                storage_entry(0, true),  // state
                storage_entry(1, true),  // coeffs
                storage_entry(2, false), // new_state
                storage_entry(3, false), // error
                uniform_entry(4),        // params
                storage_entry(5, false), // scratch
            ],
        });

        let layout = d.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("RK45 Layout"),
            bind_group_layouts: &[&bgl],
            immediate_size: 0,
        });

        let module = device.compile_shader_f64(WGSL_RK45_ADAPTIVE_F64, Some("RK45 f64"));

        let pipeline = d.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("RK45 Pipeline"),
            layout: Some(&layout),
            module: &module,
            entry_point: Some("rk45_step"),
            compilation_options: Default::default(),
            cache: None,
        });

        Self {
            pipeline,
            bgl,
            device,
        }
    }

    /// Dispatch one adaptive RK45 step (f64 pipeline).
    ///
    /// `state_buf`:     `[n_systems × dim]` f64 — current ODE state
    /// `coeffs_buf`:    `[n_systems × n_coeffs]` f64 — per-system coefficients
    /// `new_state_buf`: `[n_systems × dim]` f64 — output state (5th order)
    /// `error_buf`:     `[n_systems × dim]` f64 — per-variable absolute error
    /// `scratch_buf`:   `[n_systems × dim × 8]` f64 — k-stage + tmp workspace
    #[expect(clippy::too_many_arguments, reason = "API")]
    pub fn dispatch(
        &self,
        state_buf: &wgpu::Buffer,
        coeffs_buf: &wgpu::Buffer,
        new_state_buf: &wgpu::Buffer,
        error_buf: &wgpu::Buffer,
        scratch_buf: &wgpu::Buffer,
        n_systems: u32,
        dim: u32,
        n_coeffs: u32,
        dt: f64,
    ) {
        let d = self.device.device();
        let q = self.device.queue();

        let params = Rk45Params {
            n_systems,
            dim,
            n_coeffs,
            _pad: 0,
            dt,
            _pad2: 0.0,
        };
        let params_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("RK45 Params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bg = d.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("RK45 BG"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: state_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: coeffs_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: new_state_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: error_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: scratch_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = self
            .device
            .create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
                label: Some("RK45"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("RK45 Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups(n_systems.div_ceil(WORKGROUP_SIZE_COMPACT), 1, 1);
        }
        q.submit(std::iter::once(encoder.finish()));
    }
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

// SPDX-License-Identifier: AGPL-3.0-or-later

//! Adaptive Dormand-Prince RK45 for regulatory networks — GPU kernel (f64).
//!
//! Single adaptive step of the Dormand-Prince 5(4) embedded pair.
//! Each thread handles one independent ODE system with Hill function kinetics.
//!
//! Output: new state (5th order) and per-variable absolute error.
//! Host uses error to adapt step size:
//!   `h_new` = h × min(5, max(0.2, 0.9 × (tol/err)^0.2))
//!
//! **Provenance**: neuralSpring metalForge → toadStool absorption (Feb 2026)
//! **Papers**: 020 (regulatory network), 021 (signal integration)

use std::sync::Arc;

use wgpu::util::DeviceExt;

use crate::device::WgpuDevice;
use crate::device::capabilities::WORKGROUP_SIZE_COMPACT;

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

/// GPU-resident buffers for an RK45 adaptive step.
pub struct Rk45Buffers<'a> {
    /// Current ODE state `[n_systems × dim]`.
    pub state_buf: &'a wgpu::Buffer,
    /// Per-system coefficients `[n_systems × n_coeffs]`.
    pub coeffs_buf: &'a wgpu::Buffer,
    /// Output: 5th-order solution `[n_systems × dim]`.
    pub new_state_buf: &'a wgpu::Buffer,
    /// Output: per-variable absolute error `[n_systems × dim]`.
    pub error_buf: &'a wgpu::Buffer,
    /// Workspace for k-stages `[n_systems × dim × 8]`.
    pub scratch_buf: &'a wgpu::Buffer,
}

/// Scalar parameters for an RK45 adaptive step.
pub struct Rk45DispatchParams {
    /// Number of independent ODE systems.
    pub n_systems: u32,
    /// State dimension per system.
    pub dim: u32,
    /// Coefficient count per system.
    pub n_coeffs: u32,
    /// Time step.
    pub dt: f64,
}

/// Grouped arguments for [`Rk45AdaptiveGpu::dispatch`].
pub struct Rk45DispatchArgs<'a> {
    /// GPU buffers.
    pub buffers: Rk45Buffers<'a>,
    /// Scalar parameters.
    pub params: Rk45DispatchParams,
}

/// Adaptive Dormand-Prince RK45 GPU kernel for regulatory network ODEs.
pub struct Rk45AdaptiveGpu {
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    device: Arc<WgpuDevice>,
}

impl Rk45AdaptiveGpu {
    /// Create an adaptive RK45 GPU kernel for regulatory network ODEs.
    #[must_use]
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
    /// `buffers.state_buf`:     `[n_systems × dim]` f64 — current ODE state
    /// `buffers.coeffs_buf`:    `[n_systems × n_coeffs]` f64 — per-system coefficients
    /// `buffers.new_state_buf`: `[n_systems × dim]` f64 — output state (5th order)
    /// `buffers.error_buf`:     `[n_systems × dim]` f64 — per-variable absolute error
    /// `buffers.scratch_buf`:   `[n_systems × dim × 8]` f64 — k-stage + tmp workspace
    pub fn dispatch(&self, args: &Rk45DispatchArgs<'_>) {
        let d = self.device.device();
        let q = self.device.queue();

        let params = Rk45Params {
            n_systems: args.params.n_systems,
            dim: args.params.dim,
            n_coeffs: args.params.n_coeffs,
            _pad: 0,
            dt: args.params.dt,
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
                    resource: args.buffers.state_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: args.buffers.coeffs_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: args.buffers.new_state_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: args.buffers.error_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: args.buffers.scratch_buf.as_entire_binding(),
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
            pass.dispatch_workgroups(args.params.n_systems.div_ceil(WORKGROUP_SIZE_COMPACT), 1, 1);
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

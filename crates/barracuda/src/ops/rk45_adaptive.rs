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

// ── Batched ODE RK45 integrator (wetSpring V95) ─────────────────────────────

/// Configuration for batched adaptive RK45 integration.
pub struct BatchedRk45Config {
    /// Number of independent ODE systems to integrate.
    pub n_systems: u32,
    /// State dimension per system.
    pub dim: u32,
    /// Coefficient count per system.
    pub n_coeffs: u32,
    /// Initial time step.
    pub dt_init: f64,
    /// Absolute tolerance for step-size control.
    pub atol: f64,
    /// Relative tolerance for step-size control.
    pub rtol: f64,
    /// Maximum number of adaptive steps before returning.
    pub max_steps: u32,
    /// Integration endpoint (`t_final`).
    pub t_final: f64,
}

/// Result of a batched RK45 integration.
pub struct BatchedRk45Result {
    /// Final state `[n_systems × dim]`.
    pub states: Vec<f64>,
    /// Total steps taken per system.
    pub steps_taken: u32,
    /// Whether integration reached `t_final` within `max_steps`.
    pub converged: bool,
}

/// Batched adaptive ODE integrator: multiple full trajectories on GPU.
///
/// Wraps [`Rk45AdaptiveGpu`] with host-side adaptive step-size control
/// (Dormand-Prince 5(4) error estimate). Each call to [`integrate`]
/// advances all systems from `t=0` to `t_final` using adaptive dt.
///
/// **Provenance**: wetSpring V95 requested this for batched bio ODE systems
/// where RK45 vs RK4 achieves 18.5× fewer steps for stiff regulatory networks.
pub struct BatchedOdeRK45F64 {
    kernel: Rk45AdaptiveGpu,
    device: Arc<WgpuDevice>,
}

impl BatchedOdeRK45F64 {
    /// Create a batched RK45 integrator.
    #[must_use]
    pub fn new(device: Arc<WgpuDevice>) -> Self {
        let kernel = Rk45AdaptiveGpu::new(Arc::clone(&device));
        Self { kernel, device }
    }

    /// Integrate all systems from `t=0` to `config.t_final` using adaptive stepping.
    ///
    /// The step-size controller uses the standard embedded-pair formula:
    /// `dt_new = dt * min(5, max(0.2, 0.9 * (tol/err)^0.2))`
    ///
    /// Returns the final states and integration metadata.
    ///
    /// # Errors
    /// Returns [`Err`] if GPU buffer creation or readback fails.
    pub fn integrate(
        &self,
        initial_states: &[f64],
        coefficients: &[f64],
        config: &BatchedRk45Config,
    ) -> crate::error::Result<BatchedRk45Result> {
        let sys_dim = (config.n_systems * config.dim) as usize;
        let scratch_size = (config.n_systems * config.dim * 8) as usize;
        let d = self.device.device();

        let state_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("RK45Batch state"),
            contents: bytemuck::cast_slice(initial_states),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });
        let coeffs_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("RK45Batch coeffs"),
            contents: bytemuck::cast_slice(coefficients),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let new_state_buf = self.device.create_buffer_f64(sys_dim)?;
        let error_buf = self.device.create_buffer_f64(sys_dim)?;
        let scratch_buf = self.device.create_buffer_f64(scratch_size)?;

        let mut dt = config.dt_init;
        let mut t = 0.0_f64;
        let mut steps: u32 = 0;
        let safety = 0.9_f64;
        let dt_min_factor = 0.2_f64;
        let dt_max_factor = 5.0_f64;

        while t < config.t_final && steps < config.max_steps {
            let dt_step = dt.min(config.t_final - t);

            let args = Rk45DispatchArgs {
                buffers: Rk45Buffers {
                    state_buf: &state_buf,
                    coeffs_buf: &coeffs_buf,
                    new_state_buf: &new_state_buf,
                    error_buf: &error_buf,
                    scratch_buf: &scratch_buf,
                },
                params: Rk45DispatchParams {
                    n_systems: config.n_systems,
                    dim: config.dim,
                    n_coeffs: config.n_coeffs,
                    dt: dt_step,
                },
            };

            self.kernel.dispatch(&args);

            let errors = self.device.read_buffer_f64(&error_buf, sys_dim)?;
            let max_err = errors.iter().copied().fold(0.0_f64, f64::max).max(1e-15);

            let tol = config.rtol.mul_add(max_err, config.atol);
            let err_ratio = tol / max_err;

            if err_ratio >= 1.0 {
                // Accept step: copy new_state → state
                let mut encoder =
                    self.device
                        .create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
                            label: Some("RK45Batch copy"),
                        });
                encoder.copy_buffer_to_buffer(
                    &new_state_buf,
                    0,
                    &state_buf,
                    0,
                    (sys_dim * 8) as u64,
                );
                self.device
                    .queue()
                    .submit(std::iter::once(encoder.finish()));

                t += dt_step;
                steps += 1;
            }

            // Adapt step size
            let factor = safety * err_ratio.powf(0.2);
            dt *= factor.clamp(dt_min_factor, dt_max_factor);
        }

        let final_states = self.device.read_buffer_f64(&state_buf, sys_dim)?;

        Ok(BatchedRk45Result {
            states: final_states,
            steps_taken: steps,
            converged: t >= config.t_final,
        })
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

// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU dynamical fermion HMC trajectory.
//!
//! Orchestrates all lattice QCD GPU primitives into a complete HMC trajectory:
//! leapfrog integration, gauge force, pseudofermion force, CG solver,
//! Wilson action, kinetic energy, and Metropolis accept/reject.
//!
//! All math runs on GPU. The host loop only reads scalar reduction results
//! for convergence checks and the accept/reject decision.
//!
//! ## Streaming Mode
//!
//! For pure gauge HMC (no dynamical fermions), `run_streaming()` batches the
//! entire MD integration into a single GPU command encoder submission. This
//! eliminates per-dispatch host-device round-trip overhead (~0.5ms × N_md × 8
//! dispatches/step) and dramatically improves GPU utilization (43% → 85-95%).
//!
//! Provenance: hotSpring `gpu_streaming_md_encoder` (validated on RTX 3090).

use crate::device::WgpuDevice;
use crate::device::capabilities::WORKGROUP_SIZE_COMPACT;
use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::Result;
use crate::pipeline::ReduceScalarPipeline;
use std::sync::Arc;

use super::dirac::DiracGpuLayout;
use super::gpu_cg_solver::{GpuCgBuffers, GpuCgSolver};
use super::gpu_hmc_leapfrog::{GpuHmcLeapfrog, LeapfrogBuffers};
use super::gpu_hmc_types::{AxpyParamsLocal, DotParamsLocal, HostRng};
pub use super::gpu_hmc_types::{GpuHmcBuffers, GpuHmcConfig, GpuHmcResult};
use super::gpu_kinetic_energy::GpuKineticEnergy;
use super::gpu_pseudofermion::{GpuPseudofermionForce, GpuPseudofermionHeatbath};
use super::gpu_wilson_action::GpuWilsonAction;
use super::hmc_force_su3::Su3HmcForce;

/// Full GPU HMC trajectory engine.
pub struct GpuHmcTrajectory {
    device: Arc<WgpuDevice>,
    config: GpuHmcConfig,
    volume: u32,
    n_links: u32,
    leapfrog: GpuHmcLeapfrog,
    omelyan: super::omelyan_integrator::OmelyanIntegrator,
    gauge_force: Su3HmcForce,
    wilson_action: GpuWilsonAction,
    kinetic: GpuKineticEnergy,
    heatbath: GpuPseudofermionHeatbath,
    pf_force: GpuPseudofermionForce,
    cg_solver: GpuCgSolver,
    action_reducer: ReduceScalarPipeline,
    energy_reducer: ReduceScalarPipeline,
    host_rng: std::cell::RefCell<HostRng>,
}

impl GpuHmcTrajectory {
    /// Create a new HMC trajectory engine with default RNG seed.
    /// # Errors
    /// Returns [`Err`] if any sub-component (leapfrog, gauge force, Wilson action, etc.) fails to initialize due to shader compilation, buffer allocation, or device loss.
    pub fn new(device: Arc<WgpuDevice>, config: GpuHmcConfig) -> Result<Self> {
        Self::with_seed(device, config, 42)
    }

    /// Create with an explicit host RNG seed for reproducible Metropolis.
    /// # Errors
    /// Returns [`Err`] if any sub-component (leapfrog, gauge force, Wilson action, etc.) fails to initialize due to shader compilation, buffer allocation, or device loss.
    pub fn with_seed(device: Arc<WgpuDevice>, config: GpuHmcConfig, seed: u64) -> Result<Self> {
        let volume = config.nt * config.nx * config.ny * config.nz;
        let n_links = volume * 4;
        let leapfrog = GpuHmcLeapfrog::new(device.clone(), volume)?;
        let omelyan = super::omelyan_integrator::OmelyanIntegrator::new(GpuHmcLeapfrog::new(
            device.clone(),
            volume,
        )?);

        Ok(Self {
            leapfrog,
            omelyan,
            gauge_force: Su3HmcForce::new(
                device.clone(),
                config.nt,
                config.nx,
                config.ny,
                config.nz,
                config.beta,
            )?,
            wilson_action: GpuWilsonAction::new(
                device.clone(),
                config.nt,
                config.nx,
                config.ny,
                config.nz,
            )?,
            kinetic: GpuKineticEnergy::new(device.clone(), volume)?,
            heatbath: GpuPseudofermionHeatbath::new(device.clone(), volume)?,
            pf_force: GpuPseudofermionForce::new(
                device.clone(),
                config.nt,
                config.nx,
                config.ny,
                config.nz,
            )?,
            cg_solver: GpuCgSolver::new(device.clone(), volume)?,
            action_reducer: ReduceScalarPipeline::new(device.clone(), volume as usize)?,
            energy_reducer: ReduceScalarPipeline::new(device.clone(), n_links as usize)?,
            device,
            config,
            volume,
            n_links,
            host_rng: std::cell::RefCell::new(HostRng::new(seed)),
        })
    }

    /// Active FP64 strategy used by this trajectory engine (from its leapfrog integrator).
    #[must_use]
    pub fn strategy(&self) -> crate::device::capabilities::Fp64Strategy {
        self.leapfrog.strategy()
    }

    /// Upload lattice topology (neighbors + staggered phases) from a `DiracGpuLayout`.
    pub fn upload_topology(&self, layout: &DiracGpuLayout, bufs: &GpuHmcBuffers) {
        self.device
            .queue
            .write_buffer(&bufs.nbr, 0, bytemuck::cast_slice(&layout.neighbors));
        self.device
            .queue
            .write_buffer(&bufs.phases, 0, bytemuck::cast_slice(&layout.phases));
    }

    /// Seed RNG buffers from a host seed.
    pub fn seed_rng(&self, seed: u32, bufs: &GpuHmcBuffers) {
        let link_seeds: Vec<u32> = (0..self.n_links)
            .map(|i| seed.wrapping_mul(2_654_435_761).wrapping_add(i))
            .collect();
        let site_seeds: Vec<u32> = (0..self.volume)
            .map(|i| {
                seed.wrapping_mul(1_103_515_245)
                    .wrapping_add(i)
                    .wrapping_add(1)
            })
            .collect();
        self.device
            .queue
            .write_buffer(&bufs.rng_links, 0, bytemuck::cast_slice(&link_seeds));
        self.device
            .queue
            .write_buffer(&bufs.rng_sites, 0, bytemuck::cast_slice(&site_seeds));
    }

    /// Run one HMC trajectory. All computation on GPU.
    /// # Errors
    /// Returns [`Err`] if any GPU operation fails (heatbath, Dirac dispatch, CG solve, force computation, reduction, or buffer mapping) due to invalid buffer dimensions, command submission failure, or device loss.
    pub fn run(&self, bufs: &GpuHmcBuffers) -> Result<GpuHmcResult> {
        // Backup links for possible reject
        self.copy_buffer(&bufs.links, &bufs.links_backup);

        let mut total_cg_iters = 0;

        // Generate pseudofermion fields: η ~ N(0,1), then φ = D†η
        let dirac_heatbath = super::dirac::StaggeredDirac::new(self.device.clone(), self.volume)?;
        for phi_buf in &bufs.phi_fields {
            self.heatbath.generate(&bufs.eta, &bufs.rng_sites)?;
            dirac_heatbath.dispatch(
                self.config.mass,
                -1.0, // hop_sign = -1 for D†
                &bufs.links,
                &bufs.eta,
                phi_buf,
                &bufs.nbr,
                &bufs.phases,
            )?;
        }

        // Compute initial gauge action S_G
        self.wilson_action
            .compute(&bufs.links, &bufs.action_per_site)?;
        let gauge_action_before =
            self.config.beta * self.action_reducer.sum_f64(&bufs.action_per_site)?;

        // Compute initial fermion action S_F = Σ φ†(D†D)⁻¹φ
        let mut fermion_action_before = 0.0;
        let lattice = super::gpu_cg_solver::CgLatticeBuffers {
            links: &bufs.links,
            nbr: &bufs.nbr,
            phases: &bufs.phases,
        };
        let cg_config = super::gpu_cg_solver::CgSolverConfig {
            mass: self.config.mass,
            tol: self.config.cg_tol,
            max_iter: self.config.cg_max_iter,
        };
        for phi_buf in &bufs.phi_fields {
            let cg_result = self
                .cg_solver
                .solve(phi_buf, &bufs.cg, &lattice, &cg_config)?;
            total_cg_iters += cg_result.iterations;
            fermion_action_before += self.fermion_action_from_cg(phi_buf, &bufs.cg)?;
        }

        // Generate random momenta
        self.leapfrog.generate_momenta(
            &LeapfrogBuffers {
                links_buf: &bufs.links,
                momenta_buf: &bufs.momenta,
                force_buf: &bufs.gauge_force,
                rng_buf: &bufs.rng_links,
            },
            self.volume,
        )?;

        // Compute initial kinetic energy
        self.kinetic.compute(&bufs.momenta, &bufs.energy_per_link)?;
        let kinetic_before = self.energy_reducer.sum_f64(&bufs.energy_per_link)?;

        let h_old = kinetic_before + gauge_action_before + fermion_action_before;

        // Omelyan 2MN integration (O(ε⁴) energy conservation)
        self.omelyan_integration(bufs, &mut total_cg_iters)?;

        // Compute final Hamiltonian
        self.wilson_action
            .compute(&bufs.links, &bufs.action_per_site)?;
        let gauge_action_after =
            self.config.beta * self.action_reducer.sum_f64(&bufs.action_per_site)?;

        let mut fermion_action_after = 0.0;
        for phi_buf in &bufs.phi_fields {
            let cg_result = self
                .cg_solver
                .solve(phi_buf, &bufs.cg, &lattice, &cg_config)?;
            total_cg_iters += cg_result.iterations;
            fermion_action_after += self.fermion_action_from_cg(phi_buf, &bufs.cg)?;
        }

        self.kinetic.compute(&bufs.momenta, &bufs.energy_per_link)?;
        let kinetic_after = self.energy_reducer.sum_f64(&bufs.energy_per_link)?;

        let h_new = kinetic_after + gauge_action_after + fermion_action_after;
        let delta_h = h_new - h_old;

        // Metropolis accept/reject (only scalar comparison on host)
        let accepted = if delta_h <= 0.0 {
            true
        } else {
            let r: f64 = self.host_rng.borrow_mut().uniform();
            r < (-delta_h).exp()
        };

        if !accepted {
            self.copy_buffer(&bufs.links_backup, &bufs.links);
        }

        Ok(GpuHmcResult {
            accepted,
            delta_h,
            gauge_action: if accepted {
                gauge_action_after
            } else {
                gauge_action_before
            },
            fermion_action: if accepted {
                fermion_action_after
            } else {
                fermion_action_before
            },
            kinetic_energy: if accepted {
                kinetic_after
            } else {
                kinetic_before
            },
            total_cg_iterations: total_cg_iters,
        })
    }

    /// Omelyan 2MN integration with force recomputation between position updates.
    ///
    /// Per step: F → π(λε) → U(ε/2) → F → π((1-2λ)ε) → U(ε/2) → F → π(λε)
    ///
    /// Force is recomputed 3× per step (after each drift) to maintain the
    /// symplectic property. Without this, the integrator is non-reversible and
    /// produces O(1) energy violations instead of O(ε⁴).
    fn omelyan_integration(&self, bufs: &GpuHmcBuffers, total_cg_iters: &mut usize) -> Result<()> {
        let dt = self.config.dt;
        let lam = super::omelyan_integrator::OMELYAN_LAMBDA;
        let lf_bufs = LeapfrogBuffers {
            links_buf: &bufs.links,
            momenta_buf: &bufs.momenta,
            force_buf: &bufs.total_force,
            rng_buf: &bufs.rng_links,
        };

        for _ in 0..self.config.n_md_steps {
            self.compute_total_force(bufs, total_cg_iters)?;
            self.leapfrog.momentum_kick(&lf_bufs, self.volume, lam * dt)?;

            self.leapfrog.link_update(&lf_bufs, self.volume, 0.5 * dt)?;

            self.compute_total_force(bufs, total_cg_iters)?;
            self.leapfrog.momentum_kick(&lf_bufs, self.volume, 2.0f64.mul_add(-lam, 1.0) * dt)?;

            self.leapfrog.link_update(&lf_bufs, self.volume, 0.5 * dt)?;

            self.compute_total_force(bufs, total_cg_iters)?;
            self.leapfrog.momentum_kick(&lf_bufs, self.volume, lam * dt)?;
        }

        Ok(())
    }

    /// Streaming Omelyan integration: all MD passes in a single GPU submission.
    ///
    /// Pre-compiles pipelines for force, momentum kick, and link update, then
    /// records all `n_md_steps × 8` passes (3 force + 3 kick + 2 link per step)
    /// into one command encoder. Eliminates per-dispatch submit+poll overhead.
    ///
    /// Pure gauge only — no CG convergence readbacks needed during MD.
    fn omelyan_integration_streaming(&self, bufs: &GpuHmcBuffers) -> Result<()> {
        let dt = self.config.dt;
        let lam = super::omelyan_integrator::OMELYAN_LAMBDA;
        let n_md_steps = self.config.n_md_steps;

        let force_wg = self.gauge_force.workgroup_count();
        let lf_wg = self.leapfrog.workgroup_count();

        let force_module = self.device.compile_shader_f64(
            self.gauge_force.shader_src(),
            Some("streaming_force"),
        );
        let force_bgl = self.device.device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("streaming_force_bgl"),
                entries: &[
                    crate::device::compute_pipeline::uniform_bgl_entry(0),
                    crate::device::compute_pipeline::storage_bgl_entry(1, true),
                    crate::device::compute_pipeline::storage_bgl_entry(2, false),
                ],
            },
        );
        let force_pl = self.device.device.create_pipeline_layout(
            &wgpu::PipelineLayoutDescriptor {
                label: Some("streaming_force_pl"),
                bind_group_layouts: &[&force_bgl],
                immediate_size: 0,
            },
        );
        let force_pipeline = self.device.device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some("streaming_force"),
                layout: Some(&force_pl),
                module: &force_module,
                entry_point: Some("hmc_force"),
                cache: None,
                compilation_options: Default::default(),
            },
        );
        let force_bg = self.device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("streaming_force_bg"),
            layout: &force_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.gauge_force.params_buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: bufs.links.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: bufs.total_force.as_entire_binding(),
                },
            ],
        });

        let leapfrog_bgl_entries = &[
            crate::device::compute_pipeline::uniform_bgl_entry(0),
            crate::device::compute_pipeline::storage_bgl_entry(1, false),
            crate::device::compute_pipeline::storage_bgl_entry(2, false),
            crate::device::compute_pipeline::storage_bgl_entry(3, true),
            crate::device::compute_pipeline::storage_bgl_entry(4, false),
        ];
        let lf_bgl = self.device.device.create_bind_group_layout(
            &wgpu::BindGroupLayoutDescriptor {
                label: Some("streaming_lf_bgl"),
                entries: leapfrog_bgl_entries,
            },
        );
        let lf_pl = self.device.device.create_pipeline_layout(
            &wgpu::PipelineLayoutDescriptor {
                label: Some("streaming_lf_pl"),
                bind_group_layouts: &[&lf_bgl],
                immediate_size: 0,
            },
        );

        let lf_wg_df64 = self.leapfrog.workgroup_count_df64();
        let (kick_pipeline, link_pipeline, lf_wg_actual) =
            if let (Some(mom_src), Some(link_src)) = (
                self.leapfrog.df64_momentum_src(),
                self.leapfrog.df64_link_src(),
            ) {
                let mom_mod = self.device.compile_shader(mom_src, Some("streaming_kick_df64"));
                let link_mod = self.device.compile_shader(link_src, Some("streaming_link_df64"));
                let kick_pl = self.device.device.create_compute_pipeline(
                    &wgpu::ComputePipelineDescriptor {
                        label: Some("streaming_kick_df64"),
                        layout: Some(&lf_pl),
                        module: &mom_mod,
                        entry_point: Some("momentum_update_df64"),
                        cache: None,
                        compilation_options: Default::default(),
                    },
                );
                let link_pl_pipe = self.device.device.create_compute_pipeline(
                    &wgpu::ComputePipelineDescriptor {
                        label: Some("streaming_link_df64"),
                        layout: Some(&lf_pl),
                        module: &link_mod,
                        entry_point: Some("link_update_df64"),
                        cache: None,
                        compilation_options: Default::default(),
                    },
                );
                (kick_pl, link_pl_pipe, lf_wg_df64)
            } else {
                let native_mod = self.device.compile_shader_f64(
                    self.leapfrog.native_shader_src(),
                    Some("streaming_lf_native"),
                );
                let kick_pl = self.device.device.create_compute_pipeline(
                    &wgpu::ComputePipelineDescriptor {
                        label: Some("streaming_kick_native"),
                        layout: Some(&lf_pl),
                        module: &native_mod,
                        entry_point: Some("momentum_kick"),
                        cache: None,
                        compilation_options: Default::default(),
                    },
                );
                let link_pl_pipe = self.device.device.create_compute_pipeline(
                    &wgpu::ComputePipelineDescriptor {
                        label: Some("streaming_link_native"),
                        layout: Some(&lf_pl),
                        module: &native_mod,
                        entry_point: Some("link_update"),
                        cache: None,
                        compilation_options: Default::default(),
                    },
                );
                (kick_pl, link_pl_pipe, lf_wg)
            };

        use super::gpu_hmc_leapfrog::LeapfrogParams;
        let make_lf_params_buf = |dt_val: f64, label: &str| -> wgpu::Buffer {
            let data = LeapfrogParams {
                volume: self.volume,
                n_links: self.n_links,
                _pad0: 0,
                _pad1: 0,
                dt: dt_val,
                _padf: 0.0,
            };
            let buf = self.device.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: std::mem::size_of::<LeapfrogParams>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            self.device
                .queue
                .write_buffer(&buf, 0, bytemuck::bytes_of(&data));
            buf
        };

        let kick_lam_buf = make_lf_params_buf(lam * dt, "stream_kick_lam");
        let kick_mid_buf = make_lf_params_buf(2.0f64.mul_add(-lam, 1.0) * dt, "stream_kick_mid");
        let link_half_buf = make_lf_params_buf(0.5 * dt, "stream_link_half");

        let make_lf_bg = |params_buf: &wgpu::Buffer, label: &str| -> wgpu::BindGroup {
            self.device.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some(label),
                layout: &lf_bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: params_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: bufs.links.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: bufs.momenta.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: bufs.total_force.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: bufs.rng_links.as_entire_binding(),
                    },
                ],
            })
        };

        let kick_lam_bg = make_lf_bg(&kick_lam_buf, "stream_kick_lam_bg");
        let kick_mid_bg = make_lf_bg(&kick_mid_buf, "stream_kick_mid_bg");
        let link_half_bg = make_lf_bg(&link_half_buf, "stream_link_half_bg");

        let _permit = self.device.acquire_dispatch();
        let mut encoder = self.device.create_encoder_guarded(
            &wgpu::CommandEncoderDescriptor {
                label: Some("streaming_md"),
            },
        );

        for _step in 0..n_md_steps {
            Self::encode_compute_pass(&mut encoder, &force_pipeline, &force_bg, force_wg);
            Self::encode_compute_pass(&mut encoder, &kick_pipeline, &kick_lam_bg, lf_wg_actual);
            Self::encode_compute_pass(&mut encoder, &link_pipeline, &link_half_bg, lf_wg_actual);
            Self::encode_compute_pass(&mut encoder, &force_pipeline, &force_bg, force_wg);
            Self::encode_compute_pass(&mut encoder, &kick_pipeline, &kick_mid_bg, lf_wg_actual);
            Self::encode_compute_pass(&mut encoder, &link_pipeline, &link_half_bg, lf_wg_actual);
            Self::encode_compute_pass(&mut encoder, &force_pipeline, &force_bg, force_wg);
            Self::encode_compute_pass(&mut encoder, &kick_pipeline, &kick_lam_bg, lf_wg_actual);
        }

        self.device.submit_and_poll_inner(Some(encoder.finish()));
        Ok(())
    }

    fn encode_compute_pass(
        encoder: &mut crate::device::GuardedEncoder,
        pipeline: &wgpu::ComputePipeline,
        bind_group: &wgpu::BindGroup,
        workgroups: u32,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: None,
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, Some(bind_group), &[]);
        let (wx, wy) = if workgroups <= 65535 {
            (workgroups, 1)
        } else {
            let y = workgroups.div_ceil(65535);
            (workgroups.div_ceil(y), y)
        };
        pass.dispatch_workgroups(wx, wy, 1);
    }

    /// Run one HMC trajectory using streaming encoder (single GPU submission for MD).
    ///
    /// Pure gauge only — requires `n_flavors_over_4 == 0`. For dynamical fermions,
    /// falls back to `run()` (CG solves require per-iteration host readbacks).
    ///
    /// Performance: eliminates `n_md_steps × 8` submit+poll cycles (~0.5ms each),
    /// reducing 40-step trajectory dispatch overhead from ~160ms to ~0.5ms.
    /// # Errors
    /// Returns [`Err`] if any GPU operation fails.
    pub fn run_streaming(&self, bufs: &GpuHmcBuffers) -> Result<GpuHmcResult> {
        if !bufs.phi_fields.is_empty() {
            return self.run(bufs);
        }

        self.copy_buffer(&bufs.links, &bufs.links_backup);

        self.leapfrog.generate_momenta(
            &LeapfrogBuffers {
                links_buf: &bufs.links,
                momenta_buf: &bufs.momenta,
                force_buf: &bufs.gauge_force,
                rng_buf: &bufs.rng_links,
            },
            self.volume,
        )?;

        self.kinetic.compute(&bufs.momenta, &bufs.energy_per_link)?;
        let kinetic_before = self.energy_reducer.sum_f64(&bufs.energy_per_link)?;

        self.wilson_action
            .compute(&bufs.links, &bufs.action_per_site)?;
        let gauge_action_before =
            self.config.beta * self.action_reducer.sum_f64(&bufs.action_per_site)?;

        let h_old = kinetic_before + gauge_action_before;

        self.omelyan_integration_streaming(bufs)?;

        self.wilson_action
            .compute(&bufs.links, &bufs.action_per_site)?;
        let gauge_action_after =
            self.config.beta * self.action_reducer.sum_f64(&bufs.action_per_site)?;

        self.kinetic.compute(&bufs.momenta, &bufs.energy_per_link)?;
        let kinetic_after = self.energy_reducer.sum_f64(&bufs.energy_per_link)?;

        let h_new = kinetic_after + gauge_action_after;
        let delta_h = h_new - h_old;

        let accepted = if delta_h <= 0.0 {
            true
        } else {
            let r: f64 = self.host_rng.borrow_mut().uniform();
            r < (-delta_h).exp()
        };

        if !accepted {
            self.copy_buffer(&bufs.links_backup, &bufs.links);
        }

        Ok(GpuHmcResult {
            accepted,
            delta_h,
            gauge_action: if accepted {
                gauge_action_after
            } else {
                gauge_action_before
            },
            fermion_action: 0.0,
            kinetic_energy: if accepted {
                kinetic_after
            } else {
                kinetic_before
            },
            total_cg_iterations: 0,
        })
    }

    fn compute_total_force(&self, bufs: &GpuHmcBuffers, total_cg_iters: &mut usize) -> Result<()> {
        // Gauge force
        self.gauge_force.compute(&bufs.links, &bufs.gauge_force)?;

        // Zero total force, then accumulate
        let force_bytes = (self.n_links as usize * 18 * std::mem::size_of::<f64>()) as u64;
        self.device
            .queue
            .write_buffer(&bufs.total_force, 0, &vec![0u8; force_bytes as usize]);

        // Copy gauge force to total force
        self.copy_buffer_sized(&bufs.gauge_force, &bufs.total_force, force_bytes);

        // Fermion force from each pseudofermion field
        let lattice = super::gpu_cg_solver::CgLatticeBuffers {
            links: &bufs.links,
            nbr: &bufs.nbr,
            phases: &bufs.phases,
        };
        let cg_config = super::gpu_cg_solver::CgSolverConfig {
            mass: self.config.mass,
            tol: self.config.cg_tol,
            max_iter: self.config.cg_max_iter,
        };
        for phi_buf in &bufs.phi_fields {
            let cg_result = self
                .cg_solver
                .solve(phi_buf, &bufs.cg, &lattice, &cg_config)?;
            *total_cg_iters += cg_result.iterations;

            // y = D·x (apply Dirac to CG solution)
            use super::dirac::StaggeredDirac;
            let dirac = StaggeredDirac::new(self.device.clone(), self.volume)?;
            dirac.dispatch(
                self.config.mass,
                1.0,
                &bufs.links,
                &bufs.cg.x,
                &bufs.dirac_tmp,
                &bufs.nbr,
                &bufs.phases,
            )?;

            // Compute fermion force
            self.pf_force.compute(
                &bufs.links,
                &bufs.cg.x,
                &bufs.dirac_tmp,
                &bufs.fermion_force,
            )?;

            // Accumulate: total_force += fermion_force (element-wise add via axpy)
            self.add_force_buffers(&bufs.fermion_force, &bufs.total_force, force_bytes)?;
        }

        Ok(())
    }

    fn fermion_action_from_cg(
        &self,
        phi_buf: &wgpu::Buffer,
        cg_bufs: &GpuCgBuffers,
    ) -> Result<f64> {
        // S_F = Re<φ|x> — uses the dot product from the CG solver
        // We compile a quick dot_re dispatch
        let n_pairs = self.volume * 3;
        let dot_params = DotParamsLocal {
            n_pairs,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };
        let params_buf = self.device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hmc_dot:params"),
            size: std::mem::size_of::<DotParamsLocal>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.device
            .queue
            .write_buffer(&params_buf, 0, bytemuck::bytes_of(&dot_params));

        ComputeDispatch::new(self.device.as_ref(), "hmc_dot")
            .shader(super::cg::WGSL_COMPLEX_DOT_RE_F64, "main")
            .f64()
            .uniform(0, &params_buf)
            .storage_read(1, phi_buf)
            .storage_read(2, &cg_bufs.x)
            .storage_rw(3, &cg_bufs.dot_out)
            .dispatch(n_pairs.div_ceil(WORKGROUP_SIZE_COMPACT), 1, 1)
            .submit()?;

        let reducer = ReduceScalarPipeline::new(self.device.clone(), n_pairs as usize)?;
        reducer.sum_f64(&cg_bufs.dot_out)
    }

    fn add_force_buffers(&self, src: &wgpu::Buffer, dst: &wgpu::Buffer, _size: u64) -> Result<()> {
        // Use axpy with alpha=1.0 on the flat f64 arrays
        let n = self.n_links * 18;
        let params_data = AxpyParamsLocal {
            n,
            pad0: 0,
            alpha: 1.0,
        };
        let params_buf = self.device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("force_add:params"),
            size: std::mem::size_of::<AxpyParamsLocal>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.device
            .queue
            .write_buffer(&params_buf, 0, bytemuck::bytes_of(&params_data));

        ComputeDispatch::new(self.device.as_ref(), "force_add")
            .shader(super::cg::WGSL_AXPY_F64, "main")
            .f64()
            .uniform(0, &params_buf)
            .storage_read(1, src)
            .storage_rw(2, dst)
            .dispatch(n.div_ceil(WORKGROUP_SIZE_COMPACT), 1, 1)
            .submit()?;
        Ok(())
    }

    fn copy_buffer(&self, src: &wgpu::Buffer, dst: &wgpu::Buffer) {
        let size = (self.n_links as usize * 18 * std::mem::size_of::<f64>()) as u64;
        self.copy_buffer_sized(src, dst, size);
    }

    fn copy_buffer_sized(&self, src: &wgpu::Buffer, dst: &wgpu::Buffer, size: u64) {
        let mut enc = self.device.create_encoder_guarded(&Default::default());
        enc.copy_buffer_to_buffer(src, 0, dst, 0, size);
        self.device.submit_commands(Some(enc.finish()));
    }
}

#[cfg(test)]
#[path = "gpu_hmc_trajectory_tests.rs"]
mod tests;

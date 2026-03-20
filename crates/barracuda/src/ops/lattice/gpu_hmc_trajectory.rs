// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU dynamical fermion HMC trajectory.
//!
//! Orchestrates all lattice QCD GPU primitives into a complete HMC trajectory:
//! leapfrog integration, gauge force, pseudofermion force, CG solver,
//! Wilson action, kinetic energy, and Metropolis accept/reject.
//!
//! All math runs on GPU. The host loop only reads scalar reduction results
//! for convergence checks and the accept/reject decision.

use crate::device::WgpuDevice;
use crate::device::capabilities::WORKGROUP_SIZE_COMPACT;
use crate::error::Result;
use crate::pipeline::ReduceScalarPipeline;
use std::sync::Arc;

use super::dirac::DiracGpuLayout;
use super::gpu_cg_solver::{GpuCgBuffers, GpuCgSolver};
use super::gpu_hmc_leapfrog::{GpuHmcLeapfrog, LeapfrogBuffers};
use super::gpu_hmc_types::{AxpyParamsLocal, DotParamsLocal, HostRng, storage_bgl, uniform_bgl};
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

    /// Omelyan 2MN integration with force recomputation between steps.
    /// Uses π(λε) → U(ε/2) → π((1-2λ)ε) → U(ε/2) → π(λε) per step,
    /// achieving O(ε⁴) energy conservation vs O(ε²) for plain leapfrog.
    fn omelyan_integration(&self, bufs: &GpuHmcBuffers, total_cg_iters: &mut usize) -> Result<()> {
        let dt = self.config.dt;

        for _ in 0..self.config.n_md_steps {
            self.compute_total_force(bufs, total_cg_iters)?;
            self.omelyan.step(
                &bufs.links,
                &bufs.momenta,
                &bufs.total_force,
                &bufs.rng_links,
                self.volume,
                dt,
            )?;
        }

        Ok(())
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
        let module = self
            .device
            .compile_shader_f64(super::cg::WGSL_COMPLEX_DOT_RE_F64, Some("hmc_dot"));
        let bgl = self
            .device
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("hmc_dot:bgl"),
                entries: &[
                    uniform_bgl(0),
                    storage_bgl(1, true),
                    storage_bgl(2, true),
                    storage_bgl(3, false),
                ],
            });
        let layout = self
            .device
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("hmc_dot:layout"),
                bind_group_layouts: &[&bgl],
                immediate_size: 0,
            });
        let pipeline =
            self.device
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("hmc_dot:pipeline"),
                    layout: Some(&layout),
                    module: &module,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                });

        let params_buf = self.device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("hmc_dot:params"),
            size: std::mem::size_of::<DotParamsLocal>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.device
            .queue
            .write_buffer(&params_buf, 0, bytemuck::bytes_of(&dot_params));

        let bg = self
            .device
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("hmc_dot:bg"),
                layout: &bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: params_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: phi_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: cg_bufs.x.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: cg_bufs.dot_out.as_entire_binding(),
                    },
                ],
            });

        let mut enc = self.device.create_encoder_guarded(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups(n_pairs.div_ceil(WORKGROUP_SIZE_COMPACT), 1, 1);
        }
        self.device.submit_commands(Some(enc.finish()));

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
        let module = self
            .device
            .compile_shader_f64(super::cg::WGSL_AXPY_F64, Some("force_add"));
        let bgl = self
            .device
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("force_add:bgl"),
                entries: &[uniform_bgl(0), storage_bgl(1, true), storage_bgl(2, false)],
            });
        let layout = self
            .device
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("force_add:layout"),
                bind_group_layouts: &[&bgl],
                immediate_size: 0,
            });
        let pipeline =
            self.device
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("force_add:pipeline"),
                    layout: Some(&layout),
                    module: &module,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                });

        let params_buf = self.device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("force_add:params"),
            size: std::mem::size_of::<AxpyParamsLocal>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.device
            .queue
            .write_buffer(&params_buf, 0, bytemuck::bytes_of(&params_data));

        let bg = self
            .device
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("force_add:bg"),
                layout: &bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: params_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: src.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: dst.as_entire_binding(),
                    },
                ],
            });

        let mut enc = self.device.create_encoder_guarded(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups(n.div_ceil(WORKGROUP_SIZE_COMPACT), 1, 1);
        }
        self.device.submit_commands(Some(enc.finish()));
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

// SPDX-License-Identifier: AGPL-3.0-or-later

//! VPC (Visual Predictive Check) Monte Carlo simulation — GPU kernel.
//!
//! Simulates individual PK profiles under a population model with random
//! inter-individual variability. Each simulation is independent, making this
//! embarrassingly parallel on the GPU.
//!
//! At 1,000+ simulations, GPU execution is ~200× faster than sequential CPU.
//!
//! Provenance: healthSpring V14 → barraCuda absorption (Mar 2026)

use std::sync::Arc;

use wgpu::util::DeviceExt;

use crate::device::WgpuDevice;
use crate::device::capabilities::WORKGROUP_SIZE_1D;
use crate::device::compute_pipeline::{storage_bgl_entry, uniform_bgl_entry};
use crate::error::Result;

/// WGSL shader for VPC Monte Carlo simulation.
pub const WGSL_VPC_SIMULATE: &str = include_str!("../../shaders/pharma/vpc_simulate_f64.wgsl");

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct VpcConfig {
    n_simulations: u32,
    n_time_points: u32,
    n_compartments: u32,
    n_steps_per_interval: u32,
    seed_base: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

/// Parameters for a VPC simulation run.
pub struct VpcSimulateParams {
    /// Population parameter means (e.g. CL, V, ka).
    pub pop_params: Vec<f64>,
    /// Diagonal of the omega matrix (inter-individual variability variances).
    pub omega_diag: Vec<f64>,
    /// Time points at which to sample concentrations.
    pub time_points: Vec<f64>,
    /// Number of Monte Carlo simulations to run.
    pub n_simulations: u32,
    /// Number of RK4 sub-steps between time points (higher = more accurate).
    pub n_steps_per_interval: u32,
    /// PRNG seed.
    pub seed: u32,
}

/// GPU kernel for VPC Monte Carlo PK simulation.
pub struct VpcSimulateGpu {
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    device: Arc<WgpuDevice>,
}

impl VpcSimulateGpu {
    /// Create the VPC simulation kernel.
    #[must_use]
    pub fn new(device: Arc<WgpuDevice>) -> Self {
        let d = device.device();

        let bgl = d.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("VpcSimulate BGL"),
            entries: &[
                uniform_bgl_entry(0),
                storage_bgl_entry(1, true),
                storage_bgl_entry(2, true),
                storage_bgl_entry(3, true),
                storage_bgl_entry(4, false),
            ],
        });

        let layout = d.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("VpcSimulate Layout"),
            bind_group_layouts: &[&bgl],
            immediate_size: 0,
        });

        let module = device.compile_shader_f64(WGSL_VPC_SIMULATE, Some("vpc_simulate"));

        let pipeline = d.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("VpcSimulate Pipeline"),
            layout: Some(&layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: device.pipeline_cache(),
        });

        Self {
            pipeline,
            bgl,
            device,
        }
    }

    /// Run VPC Monte Carlo simulation.
    ///
    /// Returns concentration profiles: `[n_simulations × n_time_points]`.
    ///
    /// # Errors
    /// Returns [`Err`] if the device is lost or poll fails.
    pub fn simulate(&self, params: &VpcSimulateParams) -> Result<Vec<f64>> {
        let n_tp = params.time_points.len() as u32;

        let config = VpcConfig {
            n_simulations: params.n_simulations,
            n_time_points: n_tp,
            n_compartments: 2,
            n_steps_per_interval: params.n_steps_per_interval,
            seed_base: params.seed,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };

        let d = self.device.device();

        let config_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("vpc_config"),
            contents: bytemuck::bytes_of(&config),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let pop_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("vpc_pop_params"),
            contents: bytemuck::cast_slice(&params.pop_params),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let omega_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("vpc_omega"),
            contents: bytemuck::cast_slice(&params.omega_diag),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let time_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("vpc_times"),
            contents: bytemuck::cast_slice(&params.time_points),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let out_size = (params.n_simulations * n_tp) as u64 * 8;
        let conc_buf = d.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vpc_concentrations"),
            size: out_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging_buf = d.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vpc_staging"),
            size: out_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bg = d.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("vpc_bg"),
            layout: &self.bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: config_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: pop_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: omega_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: time_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: conc_buf.as_entire_binding(),
                },
            ],
        });

        let wg_count = params.n_simulations.div_ceil(WORKGROUP_SIZE_1D);
        let mut encoder = self
            .device
            .create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
                label: Some("vpc_encode"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("vpc_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups(wg_count, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&conc_buf, 0, &staging_buf, 0, out_size);
        self.device
            .queue()
            .submit(std::iter::once(encoder.finish()));
        self.device.poll_safe()?;

        let slice = staging_buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            tx.send(r).ok();
        });
        self.device.poll_safe()?;
        rx.recv()
            .map_err(|_| {
                crate::error::BarracudaError::DeviceLost("readback channel closed".into())
            })?
            .map_err(|e| {
                crate::error::BarracudaError::DeviceLost(format!("buffer map failed: {e:?}"))
            })?;

        let data = slice.get_mapped_range();
        let count = (params.n_simulations * n_tp) as usize;
        let result: Vec<f64> = bytemuck::cast_slice(&data)[..count].to_vec();
        drop(data);
        staging_buf.unmap();

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_layout() {
        assert_eq!(std::mem::size_of::<VpcConfig>(), 32);
    }
}

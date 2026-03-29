// SPDX-License-Identifier: AGPL-3.0-or-later

//! GPU-resident scalar observables — O(1) readback for plaquette, KE, and Hamiltonian.
//!
//! Replaces O(V)-readback patterns with GPU reduce chains that produce single f64
//! scalars on-device. CPU only reads back 8–72 bytes instead of O(V) per-site arrays.
//!
//! # Architecture
//!
//! ```text
//! per-site plaquette ──reduce──→ `plaq_sum` (1 f64)  ─┐
//! per-link KE        ──reduce──→ `ke` (1 f64)         ─┼→ Hamiltonian assembly (GPU)
//! fermion action     ──accumulate──→ `s_ferm` (1 f64) ─┘        │
//!                                                        `ΔH` + Metropolis (GPU)
//!                                                             │
//!                                                    ← 72 bytes readback →
//! ```
//!
//! Absorbed from hotSpring lattice QCD `resident_observables.rs` (Mar 2026),
//! rewritten to barraCuda's `WgpuDevice` / `ReduceScalarPipeline` patterns.

use crate::device::WgpuDevice;
use crate::error::Result;
use crate::pipeline::ReduceScalarPipeline;
use std::sync::Arc;

use super::absorbed_shaders::{
    WGSL_FERMION_ACTION_SUM_F64, WGSL_GPU_METROPOLIS_F64, WGSL_HAMILTONIAN_ASSEMBLY_F64,
};

/// Pre-compiled GPU pipelines for resident observable computation.
pub struct ResidentObservablePipelines {
    device: Arc<WgpuDevice>,
    /// Hamiltonian assembly: `H = S_gauge + T + S_ferm`.
    pub hamiltonian_pipeline: wgpu::ComputePipeline,
    /// Bind group layout for [`Self::hamiltonian_pipeline`].
    pub hamiltonian_bgl: wgpu::BindGroupLayout,
    /// Fermion action accumulation for RHMC sectors.
    pub fermion_action_pipeline: wgpu::ComputePipeline,
    /// Bind group layout for [`Self::fermion_action_pipeline`].
    pub fermion_action_bgl: wgpu::BindGroupLayout,
    /// Metropolis accept/reject test with diagnostics.
    pub metropolis_pipeline: wgpu::ComputePipeline,
    /// Bind group layout for [`Self::metropolis_pipeline`].
    pub metropolis_bgl: wgpu::BindGroupLayout,
    /// Scalar reduction pipeline.
    pub reducer: ReduceScalarPipeline,
}

impl ResidentObservablePipelines {
    /// Compile all resident observable pipelines.
    ///
    /// # Errors
    /// Returns [`Err`] if shader compilation or pipeline creation fails.
    pub fn new(device: Arc<WgpuDevice>, max_reduce_elements: usize) -> Result<Self> {
        let h_mod =
            device.compile_shader_f64(WGSL_HAMILTONIAN_ASSEMBLY_F64, Some("hamiltonian_assembly"));
        let fa_mod =
            device.compile_shader_f64(WGSL_FERMION_ACTION_SUM_F64, Some("fermion_action_sum"));
        let metro_mod = device.compile_shader_f64(WGSL_GPU_METROPOLIS_F64, Some("gpu_metropolis"));

        let hamiltonian_bgl =
            device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("hamiltonian:bgl"),
                    entries: &[
                        storage_bgl(0, true),  // params (beta, 6V)
                        storage_bgl(1, true),  // plaq_sum
                        storage_bgl(2, true),  // t_ke
                        storage_bgl(3, true),  // s_ferm
                        storage_bgl(4, false), // h_out
                        storage_bgl(5, false), // diag_out
                    ],
                });

        let fermion_action_bgl =
            device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("fermion_action:bgl"),
                    entries: &[
                        storage_bgl(0, true),  // params (n_dots, alpha_0)
                        storage_bgl(1, true),  // dots
                        storage_bgl(2, true),  // alphas
                        storage_bgl(3, false), // s_ferm
                    ],
                });

        let metropolis_bgl =
            device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("metropolis:bgl"),
                    entries: &[
                        storage_bgl(0, true),  // params (rand, 6V)
                        storage_bgl(1, true),  // h_old
                        storage_bgl(2, true),  // h_new
                        storage_bgl(3, true),  // plaq_sum
                        storage_bgl(4, true),  // diag_old
                        storage_bgl(5, true),  // diag_new
                        storage_bgl(6, false), // result
                    ],
                });

        let make_pipeline = |bgl: &wgpu::BindGroupLayout,
                             module: &wgpu::ShaderModule,
                             label: &str| {
            let layout = device
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some(&format!("{label}:layout")),
                    bind_group_layouts: &[bgl],
                    immediate_size: 0,
                });
            device
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(label),
                    layout: Some(&layout),
                    module,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                })
        };

        let hamiltonian_pipeline = make_pipeline(&hamiltonian_bgl, &h_mod, "hamiltonian");
        let fermion_action_pipeline = make_pipeline(&fermion_action_bgl, &fa_mod, "fermion_action");
        let metropolis_pipeline = make_pipeline(&metropolis_bgl, &metro_mod, "metropolis");

        let reducer = ReduceScalarPipeline::new(device.clone(), max_reduce_elements)?;

        Ok(Self {
            device,
            hamiltonian_pipeline,
            hamiltonian_bgl,
            fermion_action_pipeline,
            fermion_action_bgl,
            metropolis_pipeline,
            metropolis_bgl,
            reducer,
        })
    }

    /// The device this pipeline was compiled on.
    #[must_use]
    pub fn device(&self) -> &Arc<WgpuDevice> {
        &self.device
    }
}

/// GPU-resident buffers for scalar observables.
///
/// Allocated once per lattice volume, reused across trajectories.
pub struct ResidentObservableBuffers {
    /// Reduced plaquette sum (1 f64, GPU-resident).
    pub plaq_sum: wgpu::Buffer,
    /// Reduced kinetic energy (1 f64, GPU-resident).
    pub kinetic_energy: wgpu::Buffer,
    /// Accumulated fermion action (1 f64, GPU-resident).
    pub s_ferm: wgpu::Buffer,
    /// Hamiltonian output (1 f64, GPU-resident).
    pub h_out: wgpu::Buffer,
    /// Hamiltonian diagnostics: `[S_gauge, T, S_ferm]` (3 f64, GPU-resident).
    pub diag: wgpu::Buffer,
    /// Metropolis result (9 f64 — accepted, `delta_h`, plaquette, plus per-sector diagnostics).
    pub metropolis_result: wgpu::Buffer,
}

impl ResidentObservableBuffers {
    /// Allocate observable buffers.
    #[must_use]
    pub fn new(device: &WgpuDevice) -> Self {
        let make = |label: &str, n: usize| {
            device.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: (n * std::mem::size_of::<f64>()) as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };

        Self {
            plaq_sum: make("obs:plaq_sum", 1),
            kinetic_energy: make("obs:ke", 1),
            s_ferm: make("obs:s_ferm", 1),
            h_out: make("obs:h_out", 1),
            diag: make("obs:diag", 3),
            metropolis_result: make("obs:metropolis", 9),
        }
    }
}

/// Metropolis test result from a single GPU readback (72 bytes).
#[derive(Debug, Clone)]
pub struct MetropolisResult {
    /// Whether the trajectory was accepted.
    pub accepted: bool,
    /// `ΔH = H_new − H_old`.
    pub delta_h: f64,
    /// Normalized plaquette: `plaq_sum / 6V`.
    pub plaquette: f64,
    /// `S_gauge` (old configuration).
    pub s_gauge_old: f64,
    /// `S_gauge` (new configuration).
    pub s_gauge_new: f64,
    /// Kinetic energy (old momenta).
    pub t_old: f64,
    /// Kinetic energy (new momenta).
    pub t_new: f64,
    /// Fermion action (old configuration).
    pub s_ferm_old: f64,
    /// Fermion action (new configuration).
    pub s_ferm_new: f64,
}

impl MetropolisResult {
    /// Parse from the 9-entry f64 array written by the GPU Metropolis shader.
    #[must_use]
    pub fn from_gpu_result(data: &[f64; 9]) -> Self {
        Self {
            accepted: data[0] > 0.5,
            delta_h: data[1],
            plaquette: data[2],
            s_gauge_old: data[3],
            s_gauge_new: data[4],
            t_old: data[5],
            t_new: data[6],
            s_ferm_old: data[7],
            s_ferm_new: data[8],
        }
    }
}

fn storage_bgl(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metropolis_result_accepted() {
        let data = [1.0, -0.05, 0.59, 10.0, 9.5, 5.0, 5.1, 3.0, 3.2];
        let result = MetropolisResult::from_gpu_result(&data);
        assert!(result.accepted);
        assert!((result.delta_h - (-0.05)).abs() < 1e-15);
        assert!((result.plaquette - 0.59).abs() < 1e-15);
    }

    #[test]
    fn metropolis_result_rejected() {
        let data = [0.0, 2.5, 0.55, 8.0, 10.5, 4.0, 4.1, 2.0, 2.5];
        let result = MetropolisResult::from_gpu_result(&data);
        assert!(!result.accepted);
        assert!((result.delta_h - 2.5).abs() < 1e-15);
    }

    #[test]
    fn pipeline_creation_with_gpu() {
        let Some(device) = crate::device::test_pool::get_test_device_if_f64_gpu_available_sync()
        else {
            return;
        };
        let pipelines = ResidentObservablePipelines::new(device.clone(), 256).unwrap();
        assert!(Arc::ptr_eq(pipelines.device(), &device));
    }

    #[test]
    fn buffer_allocation() {
        let Some(device) = crate::device::test_pool::get_test_device_if_f64_gpu_available_sync()
        else {
            return;
        };
        let _bufs = ResidentObservableBuffers::new(&device);
    }
}

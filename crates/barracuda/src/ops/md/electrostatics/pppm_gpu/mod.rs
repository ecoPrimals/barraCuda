// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU-accelerated PPPM/Ewald electrostatics (universal via WGSL)
//!
//! This module provides hardware-agnostic PPPM using WGSL shaders that run
//! on any GPU/NPU/CPU via wgpu. All math is f64 precision.
//!
//! # Refactoring (Feb 2026)
//! - Shader sources in `shaders.rs`
//! - Pipeline creation in `pipelines.rs`
//! - Main logic in this file

mod kspace;
mod kspace_gpu;
pub(crate) mod pipelines;
mod shaders;

use std::sync::Arc;

use crate::device::capabilities::WORKGROUP_SIZE_COMPACT;
use crate::device::WgpuDevice;
use crate::error::{BarracudaError, Result};
use crate::linalg::sparse::SparseBuffers;
use crate::shaders::precision::ShaderTemplate;

use super::{GreensFunction, PppmParams};

use pipelines::{PppmBindGroupLayouts, PppmPipelines};

/// GPU-accelerated PPPM solver
pub struct PppmGpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    wgpu_device: Arc<WgpuDevice>,
    params: PppmParams,
    greens: GreensFunction,
    pipelines: PppmPipelines,
    layouts: PppmBindGroupLayouts,
}

impl PppmGpu {
    /// Create from a WgpuDevice (preferred - driver-aware f64 compilation).
    pub async fn from_device(wgpu_device: &Arc<WgpuDevice>, params: PppmParams) -> Result<Self> {
        let greens = GreensFunction::new(&params);
        let bspline_module = wgpu_device.compile_shader_f64(shaders::BSPLINE, Some("pppm_bspline"));
        let charge_spread_module =
            wgpu_device.compile_shader_f64(shaders::CHARGE_SPREAD, Some("pppm_charge_spread"));
        let greens_apply_module =
            wgpu_device.compile_shader_f64(shaders::GREENS_APPLY, Some("pppm_greens_apply"));
        let force_interp_module =
            wgpu_device.compile_shader_f64(shaders::FORCE_INTERP, Some("pppm_force_interp"));
        let erfc_forces_module =
            wgpu_device.compile_shader_f64(shaders::ERFC_FORCES, Some("pppm_erfc_forces"));

        Self::build_from_modules(
            wgpu_device.device_clone(),
            wgpu_device.queue_clone(),
            Arc::clone(wgpu_device),
            params,
            greens,
            bspline_module,
            charge_spread_module,
            greens_apply_module,
            force_interp_module,
            erfc_forces_module,
        )
        .await
    }

    /// Create from raw wgpu device/queue (legacy API).
    #[deprecated(
        since = "0.3.0",
        note = "Use from_device() for proper adapter detection"
    )]
    pub async fn new(device: wgpu::Device, queue: wgpu::Queue, params: PppmParams) -> Result<Self> {
        #[expect(
            deprecated,
            reason = "forwarding to deprecated constructor during migration"
        )]
        Self::new_with_driver(device, queue, params, false).await
    }

    /// Create with explicit driver awareness.
    #[deprecated(
        since = "0.3.0",
        note = "Use from_device() for proper adapter detection"
    )]
    pub async fn new_with_driver(
        device: wgpu::Device,
        queue: wgpu::Queue,
        params: PppmParams,
        is_nvk: bool,
    ) -> Result<Self> {
        let greens = GreensFunction::new(&params);
        let bspline_shader = ShaderTemplate::for_driver_auto(shaders::BSPLINE, is_nvk);
        let charge_spread_shader = ShaderTemplate::for_driver_auto(shaders::CHARGE_SPREAD, is_nvk);
        let greens_apply_shader = ShaderTemplate::for_driver_auto(shaders::GREENS_APPLY, is_nvk);
        let force_interp_shader = ShaderTemplate::for_driver_auto(shaders::FORCE_INTERP, is_nvk);
        let erfc_forces_shader = ShaderTemplate::for_driver_auto(shaders::ERFC_FORCES, is_nvk);

        let bspline_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pppm_bspline"),
            source: wgpu::ShaderSource::Wgsl(bspline_shader.into()),
        });
        let charge_spread_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pppm_charge_spread"),
            source: wgpu::ShaderSource::Wgsl(charge_spread_shader.into()),
        });
        let greens_apply_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pppm_greens_apply"),
            source: wgpu::ShaderSource::Wgsl(greens_apply_shader.into()),
        });
        let force_interp_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pppm_force_interp"),
            source: wgpu::ShaderSource::Wgsl(force_interp_shader.into()),
        });
        let erfc_forces_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("pppm_erfc_forces"),
            source: wgpu::ShaderSource::Wgsl(erfc_forces_shader.into()),
        });

        let wgpu_device = Arc::new(WgpuDevice::from_existing(
            device.clone(),
            queue.clone(),
            wgpu::AdapterInfo {
                name: "Legacy PPPM Device".to_string(),
                vendor: 0,
                device: 0,
                device_type: wgpu::DeviceType::Other,
                driver: if is_nvk { "nvk" } else { "unknown" }.to_string(),
                driver_info: "created via deprecated new_with_driver()".to_string(),
                backend: wgpu::Backend::Vulkan,
                device_pci_bus_id: String::new(),
                subgroup_min_size: 1,
                subgroup_max_size: 128,
                transient_saves_memory: false,
            },
        ));
        Self::build_from_modules(
            device,
            queue,
            wgpu_device,
            params,
            greens,
            bspline_module,
            charge_spread_module,
            greens_apply_module,
            force_interp_module,
            erfc_forces_module,
        )
        .await
    }

    async fn build_from_modules(
        device: wgpu::Device,
        queue: wgpu::Queue,
        wgpu_device: Arc<WgpuDevice>,
        params: PppmParams,
        greens: GreensFunction,
        bspline_module: wgpu::ShaderModule,
        charge_spread_module: wgpu::ShaderModule,
        greens_apply_module: wgpu::ShaderModule,
        force_interp_module: wgpu::ShaderModule,
        erfc_forces_module: wgpu::ShaderModule,
    ) -> Result<Self> {
        let layouts = PppmBindGroupLayouts::new(&device);
        let pipelines = PppmPipelines::new(
            &device,
            &layouts,
            &bspline_module,
            &charge_spread_module,
            &greens_apply_module,
            &force_interp_module,
            &erfc_forces_module,
        );
        Ok(Self {
            device,
            queue,
            wgpu_device,
            params,
            greens,
            pipelines,
            layouts,
        })
    }

    pub fn params(&self) -> &PppmParams {
        &self.params
    }

    pub fn greens(&self) -> &GreensFunction {
        &self.greens
    }

    pub(crate) fn device(&self) -> &wgpu::Device {
        &self.device
    }

    pub(crate) fn queue(&self) -> &wgpu::Queue {
        &self.queue
    }

    pub(crate) fn wgpu_device(&self) -> &Arc<WgpuDevice> {
        &self.wgpu_device
    }

    pub(crate) fn layouts(&self) -> &PppmBindGroupLayouts {
        &self.layouts
    }

    pub(crate) fn pipelines(&self) -> &PppmPipelines {
        &self.pipelines
    }

    pub async fn compute(&self, positions: &[f64], charges: &[f64]) -> Result<(Vec<f64>, f64)> {
        let n = charges.len();
        if positions.len() != n * 3 {
            return Err(BarracudaError::InvalidInput {
                message: format!(
                    "positions length {} != charges length {} * 3",
                    positions.len(),
                    n
                ),
            });
        }
        let order = self.params.interpolation_order;
        let [kx, ky, kz] = self.params.mesh_dims;
        let o3 = order * order * order;

        let positions_buffer =
            SparseBuffers::f64_from_slice_raw(&self.device, "positions", positions);
        let charges_buffer = SparseBuffers::f64_from_slice_raw(&self.device, "charges", charges);
        let coeffs_size = n * order * 3;
        let coeffs_buffer = SparseBuffers::f64_zeros_raw(&self.device, "coeffs", coeffs_size);
        let derivs_buffer = SparseBuffers::f64_zeros_raw(&self.device, "derivs", coeffs_size);
        let base_idx_buffer = SparseBuffers::i32_zeros_raw(&self.device, "base_idx", n * 3);
        let bspline_params: Vec<f64> = vec![
            n as f64,
            order as f64,
            kx as f64,
            ky as f64,
            kz as f64,
            self.params.box_dims[0],
            self.params.box_dims[1],
            self.params.box_dims[2],
        ];
        let bspline_params_buffer =
            SparseBuffers::f64_from_slice_raw(&self.device, "bspline_params", &bspline_params);
        let per_particle_mesh_buffer =
            SparseBuffers::f64_zeros_raw(&self.device, "per_particle_mesh", n * o3);
        let spread_params: Vec<f64> = vec![n as f64, order as f64, kx as f64, ky as f64, kz as f64];
        let spread_params_buffer =
            SparseBuffers::f64_from_slice_raw(&self.device, "spread_params", &spread_params);
        let forces_buffer = SparseBuffers::f64_zeros_raw(&self.device, "forces", n * 3);
        let pe_buffer = SparseBuffers::f64_zeros_raw(&self.device, "pe", n);
        let erfc_params: Vec<f64> = vec![
            n as f64,
            self.params.alpha,
            self.params.real_cutoff * self.params.real_cutoff,
            self.params.box_dims[0],
            self.params.box_dims[1],
            self.params.box_dims[2],
            self.params.coulomb_constant,
        ];
        let erfc_params_buffer =
            SparseBuffers::f64_from_slice_raw(&self.device, "erfc_params", &erfc_params);

        let bspline_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bspline_bind_group"),
            layout: &self.layouts.bspline,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: positions_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: coeffs_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: derivs_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: base_idx_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: bspline_params_buffer.as_entire_binding(),
                },
            ],
        });
        let charge_spread_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("charge_spread_bind_group"),
            layout: &self.layouts.charge_spread,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: charges_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: coeffs_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: base_idx_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: per_particle_mesh_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: spread_params_buffer.as_entire_binding(),
                },
            ],
        });
        let erfc_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("erfc_bind_group"),
            layout: &self.layouts.erfc_forces,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: positions_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: charges_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: forces_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: pe_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: erfc_params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder =
            self.wgpu_device
                .create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
                    label: Some("PPPM Encoder"),
                });
        let particle_workgroups = (n as u32).div_ceil(WORKGROUP_SIZE_COMPACT);

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("PPPM B-spline Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.bspline);
            pass.set_bind_group(0, Some(&bspline_bind_group), &[]);
            pass.dispatch_workgroups(particle_workgroups, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("PPPM Charge Spread Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.charge_spread);
            pass.set_bind_group(0, Some(&charge_spread_bind_group), &[]);
            pass.dispatch_workgroups(particle_workgroups, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("PPPM erfc Forces Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.erfc_forces);
            pass.set_bind_group(0, Some(&erfc_bind_group), &[]);
            pass.dispatch_workgroups(particle_workgroups, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("PPPM Self Energy Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipelines.self_energy);
            pass.set_bind_group(0, Some(&erfc_bind_group), &[]);
            pass.dispatch_workgroups(particle_workgroups, 1, 1);
        }

        self.wgpu_device.submit_and_poll(Some(encoder.finish()));
        let forces =
            SparseBuffers::read_f64_raw(self.wgpu_device().as_ref(), &forces_buffer, n * 3)?;
        let pe_values = SparseBuffers::read_f64_raw(self.wgpu_device().as_ref(), &pe_buffer, n)?;
        let total_energy: f64 = pe_values.iter().sum();
        Ok((forces, total_energy))
    }

    pub async fn compute_forces(&self, positions: &[f64], charges: &[f64]) -> Result<Vec<f64>> {
        let (forces, _) = self.compute(positions, charges).await?;
        Ok(forces)
    }

    pub async fn compute_with_kspace(
        &self,
        positions: &[f64],
        charges: &[f64],
    ) -> Result<(Vec<f64>, f64)> {
        kspace::compute_with_kspace(self, positions, charges).await
    }

    pub async fn compute_with_kspace_gpu(
        &self,
        positions: &[f64],
        charges: &[f64],
    ) -> Result<(Vec<f64>, f64)> {
        kspace_gpu::compute_with_kspace_gpu(self, positions, charges).await
    }
}

#[expect(clippy::unwrap_used, reason = "tests")]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pppm_gpu_params() {
        let params = PppmParams::custom(100, [10.0, 10.0, 10.0], [16, 16, 16], 1.0, 3.0, 4);
        assert_eq!(params.mesh_dims, [16, 16, 16]);
        assert_eq!(params.alpha, 1.0);
    }

    #[tokio::test]
    async fn test_pppm_gpu_opposite_charges_energy() {
        use crate::device::test_pool::get_test_device_if_f64_gpu_available;
        let Some(device) = get_test_device_if_f64_gpu_available().await else {
            return;
        };
        let params = PppmParams::custom(2, [10.0, 10.0, 10.0], [8, 8, 8], 2.0, 3.0, 4);
        let pppm = PppmGpu::from_device(&device, params)
            .await
            .expect("Failed to create PppmGpu");
        let positions: Vec<f64> = vec![4.0, 5.0, 5.0, 6.0, 5.0, 5.0];
        let charges: Vec<f64> = vec![1.0, -1.0];
        let (_forces, energy) = pppm
            .compute_with_kspace(&positions, &charges)
            .await
            .unwrap();
        assert!(
            energy < 0.0,
            "Opposite charges should have negative energy, got {}",
            energy
        );
    }

    #[tokio::test]
    async fn test_pppm_gpu_newtons_third_law() {
        use crate::device::test_pool::get_test_device_if_f64_gpu_available;
        let Some(device) = get_test_device_if_f64_gpu_available().await else {
            return;
        };
        let params = PppmParams::custom(2, [10.0, 10.0, 10.0], [8, 8, 8], 2.0, 3.0, 4);
        let pppm = PppmGpu::from_device(&device, params)
            .await
            .expect("Failed to create PppmGpu");
        let positions: Vec<f64> = vec![4.0, 5.0, 5.0, 6.0, 5.0, 5.0];
        let charges: Vec<f64> = vec![1.0, -1.0];
        let (forces, _energy) = pppm
            .compute_with_kspace(&positions, &charges)
            .await
            .unwrap();
        let fx_sum = forces[0] + forces[3];
        let fy_sum = forces[1] + forces[4];
        let fz_sum = forces[2] + forces[5];
        let f1_mag = (forces[0].powi(2) + forces[1].powi(2) + forces[2].powi(2)).sqrt();
        let relative_error = if f1_mag > 1e-14 {
            (fx_sum.powi(2) + fy_sum.powi(2) + fz_sum.powi(2)).sqrt() / f1_mag
        } else {
            0.0
        };
        assert!(
            relative_error < 1e-3,
            "Newton's 3rd law violation: |F1+F2|/|F1| = {}",
            relative_error
        );
    }

    #[tokio::test]
    async fn test_pppm_gpu_like_charges_repel() {
        use crate::device::test_pool::get_test_device_if_f64_gpu_available;
        let Some(device) = get_test_device_if_f64_gpu_available().await else {
            return;
        };
        let params = PppmParams::custom(2, [10.0, 10.0, 10.0], [8, 8, 8], 2.0, 3.0, 4);
        let pppm = PppmGpu::from_device(&device, params)
            .await
            .expect("Failed to create PppmGpu");
        let positions: Vec<f64> = vec![4.0, 5.0, 5.0, 6.0, 5.0, 5.0];
        let charges: Vec<f64> = vec![1.0, 1.0];
        let (forces, _energy) = pppm
            .compute_with_kspace(&positions, &charges)
            .await
            .unwrap();
        assert!(
            forces[0] < 0.0,
            "Like charges should repel: F0_x should be negative, got {}",
            forces[0]
        );
        assert!(
            forces[3] > 0.0,
            "Like charges should repel: F1_x should be positive, got {}",
            forces[3]
        );
    }

    #[tokio::test]
    #[ignore = "W-002: diagnostic - compare CPU vs GPU B-spline coeffs"]
    async fn test_pppm_gpu_bspline_vs_cpu() {
        use crate::device::test_pool::get_test_device_if_f64_gpu_available;
        use crate::linalg::sparse::SparseBuffers;
        use crate::ops::md::electrostatics::bspline::BsplineCoeffs;

        let Some(device) = get_test_device_if_f64_gpu_available().await else {
            return;
        };
        let params = PppmParams::custom(2, [10.0, 10.0, 10.0], [8, 8, 8], 2.0, 3.0, 4);
        let positions = vec![4.0, 5.0, 5.0, 6.0, 5.0, 5.0];
        let box_dims = params.box_dims;
        let mesh_dims = params.mesh_dims;
        let order = params.interpolation_order;

        let wrap_one = |x: f64, l: f64| x - (x / l).floor() * l;
        let pos0 = [
            wrap_one(positions[0], box_dims[0]),
            wrap_one(positions[1], box_dims[1]),
            wrap_one(positions[2], box_dims[2]),
        ];
        let pos1 = [
            wrap_one(positions[3], box_dims[0]),
            wrap_one(positions[4], box_dims[1]),
            wrap_one(positions[5], box_dims[2]),
        ];

        let cpu_coeffs0 = BsplineCoeffs::compute(order, pos0, mesh_dims, box_dims);
        let cpu_coeffs1 = BsplineCoeffs::compute(order, pos1, mesh_dims, box_dims);

        let pppm = PppmGpu::from_device(&device, params)
            .await
            .expect("GPU PPPM create failed");
        let (device_ref, queue, wgpu_dev) = (pppm.device(), pppm.queue(), pppm.wgpu_device());
        let positions_buffer = SparseBuffers::f64_from_slice_raw(device_ref, "pos", &positions);
        let coeffs_buffer = SparseBuffers::f64_zeros_raw(device_ref, "coeffs", 2 * order * 3);
        let derivs_buffer = SparseBuffers::f64_zeros_raw(device_ref, "derivs", 2 * order * 3);
        let base_idx_buffer = SparseBuffers::i32_zeros_raw(device_ref, "base", 6);
        let bspline_params: Vec<f64> = vec![
            2.0,
            order as f64,
            8.0,
            8.0,
            8.0,
            box_dims[0],
            box_dims[1],
            box_dims[2],
        ];
        let params_buffer = SparseBuffers::f64_from_slice_raw(device_ref, "bp", &bspline_params);
        let layout = &pppm.layouts().bspline;
        let bg = device_ref.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: positions_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: coeffs_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: derivs_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: base_idx_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });
        let mut enc = wgpu_dev.create_encoder_guarded(&Default::default());
        {
            let mut pass = enc.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pppm.pipelines().bspline);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        queue.submit(Some(enc.finish()));

        let gpu_coeffs =
            SparseBuffers::read_f64_raw(wgpu_dev.as_ref(), &coeffs_buffer, 2 * order * 3).unwrap();
        let gpu_derivs =
            SparseBuffers::read_f64_raw(wgpu_dev.as_ref(), &derivs_buffer, 2 * order * 3).unwrap();
        let gpu_base = SparseBuffers::read_i32_raw(wgpu_dev.as_ref(), &base_idx_buffer, 6).unwrap();

        for d in 0..3 {
            for k in 0..order {
                let cpu_c = cpu_coeffs0.coeffs[d][k];
                let gpu_c = gpu_coeffs[d * order + k];
                assert!(
                    (cpu_c - gpu_c).abs() < 1e-10,
                    "coeff0 d{} k{}: CPU {} GPU {}",
                    d,
                    k,
                    cpu_c,
                    gpu_c
                );
                let cpu_d = cpu_coeffs0.derivs[d][k];
                let gpu_d = gpu_derivs[d * order + k];
                assert!(
                    (cpu_d - gpu_d).abs() < 1e-10,
                    "deriv0 d{} k{}: CPU {} GPU {}",
                    d,
                    k,
                    cpu_d,
                    gpu_d
                );
            }
            assert_eq!(cpu_coeffs0.base_idx[d], gpu_base[d], "base0 d{}", d);
        }
        for d in 0..3 {
            for k in 0..order {
                let cpu_c = cpu_coeffs1.coeffs[d][k];
                let gpu_c = gpu_coeffs[3 * order + d * order + k];
                assert!(
                    (cpu_c - gpu_c).abs() < 1e-10,
                    "coeff1 d{} k{}: CPU {} GPU {}",
                    d,
                    k,
                    cpu_c,
                    gpu_c
                );
            }
        }
    }

    #[tokio::test]
    #[ignore = "W-002: diagnostic - erfc short-range only"]
    async fn test_pppm_gpu_erfc_only() {
        use crate::device::test_pool::get_test_device_if_f64_gpu_available;
        use crate::ops::md::electrostatics::compute_short_range;

        // Test erfc-only path (compute without kspace) vs CPU compute_short_range
        let Some(device) = get_test_device_if_f64_gpu_available().await else {
            return;
        };
        let params = PppmParams::custom(2, [10.0, 10.0, 10.0], [8, 8, 8], 2.0, 3.0, 4);
        let positions = vec![4.0, 5.0, 5.0, 6.0, 5.0, 5.0];
        let charges = vec![1.0, -1.0];
        let pos_3d: Vec<[f64; 3]> = positions.chunks(3).map(|c| [c[0], c[1], c[2]]).collect();

        let (cpu_short_forces, _) = compute_short_range(&pos_3d, &charges, &params);
        let pppm = PppmGpu::from_device(&device, params)
            .await
            .expect("GPU PPPM create failed");
        let (gpu_forces, _gpu_energy) = pppm.compute(&positions, &charges).await.unwrap();

        // Erfc forces should attract: F0_x > 0, F1_x < 0
        assert!(
            cpu_short_forces[0][0] > 0.0,
            "CPU erfc: +q should be pulled right"
        );
        assert!(
            cpu_short_forces[1][0] < 0.0,
            "CPU erfc: -q should be pulled left"
        );
        assert!(
            gpu_forces[0] > 0.0,
            "GPU erfc: +q should be pulled right, got {}",
            gpu_forces[0]
        );
        assert!(
            gpu_forces[3] < 0.0,
            "GPU erfc: -q should be pulled left, got {}",
            gpu_forces[3]
        );
    }

    #[tokio::test]
    #[ignore = "requires f64 GPU hardware"]
    async fn test_pppm_gpu_matches_cpu_reference() {
        use crate::device::test_pool::get_test_device_if_f64_gpu_available;
        use crate::ops::md::electrostatics::Pppm;

        let Some(device) = get_test_device_if_f64_gpu_available().await else {
            return;
        };

        let positions = vec![4.0, 5.0, 5.0, 6.0, 5.0, 5.0]; // flat [x,y,z,x,y,z]
        let charges = vec![1.0, -1.0];
        let params = PppmParams::custom(2, [10.0, 10.0, 10.0], [8, 8, 8], 2.0, 3.0, 4);

        // Reference (Pppm now uses GPU FFT)
        let pos_3d: Vec<[f64; 3]> = positions.chunks(3).map(|c| [c[0], c[1], c[2]]).collect();
        let cpu_pppm = Pppm::new(device.clone(), params.clone());
        let (cpu_forces, cpu_energy) = cpu_pppm.compute(&pos_3d, &charges).unwrap();

        // GPU version
        let pppm = PppmGpu::from_device(&device, params)
            .await
            .expect("GPU PPPM create failed");
        let (gpu_forces, gpu_energy) = pppm
            .compute_with_kspace(&positions, &charges)
            .await
            .unwrap();

        // Energy should match within 10% (mesh resolution limits accuracy)
        let energy_denom = cpu_energy.abs().max(1e-14);
        let energy_rel_error = ((gpu_energy - cpu_energy) / energy_denom).abs();
        assert!(
            energy_rel_error < 0.1,
            "GPU energy {} vs CPU energy {}",
            gpu_energy,
            cpu_energy
        );

        // Force directions must match (sign check)
        let cpu_fx0 = cpu_forces[0][0];
        let gpu_fx0 = gpu_forces[0];
        assert!(
            cpu_fx0 * gpu_fx0 > 0.0,
            "Force direction mismatch: CPU {} GPU {}",
            cpu_fx0,
            gpu_fx0
        );
    }
}

// SPDX-License-Identifier: AGPL-3.0-only
//! `PopulationPkF64` — GPU-vectorized population pharmacokinetics Monte Carlo
//!
//! Simulates N virtual patients with inter-individual clearance (CL) variability
//! using a single-compartment model:
//!
//! `AUC_i = F × Dose / CL_i`
//!
//! where `CL_i = base_cl × U(cl_low, cl_high)` for each patient.
//!
//! Absorbed from healthSpring and evolved to fully parameterized (no hardcoded
//! CL ranges or base values).

use crate::device::WgpuDevice;
use crate::device::capabilities::WORKGROUP_SIZE_1D;
use crate::error::{BarracudaError, Result};
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct PopPkParamsGpu {
    n_patients: u32,
    base_seed: u32,
    dose_mg: f64,
    f_bioavail: f64,
    base_cl: f64,
    cl_low: f64,
    cl_high: f64,
}

/// Population PK simulation configuration.
pub struct PopulationPkConfig {
    /// Dose in mg.
    pub dose_mg: f64,
    /// Oral bioavailability fraction F (0, 1\].
    pub f_bioavail: f64,
    /// Base clearance in L/hr.
    pub base_cl: f64,
    /// Lower bound of CL multiplier (e.g. 0.5 for 50% of base).
    pub cl_low: f64,
    /// Upper bound of CL multiplier (e.g. 1.5 for 150% of base).
    pub cl_high: f64,
}

/// GPU-vectorized population PK Monte Carlo.
///
/// Each GPU thread simulates one virtual patient with randomized clearance.
/// Returns a Vec of AUC values (one per patient).
pub struct PopulationPkF64 {
    device: Arc<WgpuDevice>,
    config: PopulationPkConfig,
}

impl PopulationPkF64 {
    fn wgsl_shader() -> &'static str {
        include_str!("../shaders/science/population_pk_f64.wgsl")
    }

    /// Create a new population PK simulation.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if any parameter is invalid.
    pub fn new(device: Arc<WgpuDevice>, config: PopulationPkConfig) -> Result<Self> {
        if config.dose_mg <= 0.0 {
            return Err(BarracudaError::InvalidInput {
                message: format!(
                    "PopulationPkF64: dose_mg must be > 0, got {}",
                    config.dose_mg
                ),
            });
        }
        if config.f_bioavail <= 0.0 || config.f_bioavail > 1.0 {
            return Err(BarracudaError::InvalidInput {
                message: format!(
                    "PopulationPkF64: f_bioavail must be in (0, 1], got {}",
                    config.f_bioavail
                ),
            });
        }
        if config.base_cl <= 0.0 {
            return Err(BarracudaError::InvalidInput {
                message: format!(
                    "PopulationPkF64: base_cl must be > 0, got {}",
                    config.base_cl
                ),
            });
        }
        if config.cl_low <= 0.0 || config.cl_high <= config.cl_low {
            return Err(BarracudaError::InvalidInput {
                message: format!(
                    "PopulationPkF64: need 0 < cl_low < cl_high, got [{}, {}]",
                    config.cl_low, config.cl_high
                ),
            });
        }
        Ok(Self { device, config })
    }

    /// Run the Monte Carlo simulation for `n_patients` virtual patients.
    ///
    /// Returns a `Vec<f64>` of AUC values, one per patient.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if GPU dispatch or buffer readback fails.
    pub fn simulate(&self, n_patients: u32, seed: u32) -> Result<Vec<f64>> {
        if n_patients == 0 {
            return Ok(Vec::new());
        }

        let dev = &self.device;
        let params = PopPkParamsGpu {
            n_patients,
            base_seed: seed,
            dose_mg: self.config.dose_mg,
            f_bioavail: self.config.f_bioavail,
            base_cl: self.config.base_cl,
            cl_low: self.config.cl_low,
            cl_high: self.config.cl_high,
        };

        let params_buf = dev
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("PopPK Params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let output_size = (n_patients as usize) * std::mem::size_of::<f64>();
        let output_buf = dev.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("PopPK Output"),
            size: output_size as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let bgl = dev
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("PopPK BGL"),
                entries: &[bgl_storage_rw(0), bgl_uniform(1)],
            });

        let bg = dev.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("PopPK BG"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: output_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });

        let shader = dev.compile_shader_f64(Self::wgsl_shader(), Some("PopPK"));
        let pl = dev
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("PopPK PL"),
                bind_group_layouts: &[&bgl],
                immediate_size: 0,
            });
        let pipeline = dev
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("PopPK Pipeline"),
                layout: Some(&pl),
                module: &shader,
                entry_point: Some("main"),
                cache: None,
                compilation_options: Default::default(),
            });

        let mut encoder = dev.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("PopPK Encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("PopPK Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups(n_patients.div_ceil(WORKGROUP_SIZE_1D), 1, 1);
        }
        dev.submit_and_poll(Some(encoder.finish()));

        crate::utils::read_buffer_f64(dev, &output_buf, n_patients as usize)
    }
}

fn bgl_storage_rw(idx: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding: idx,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn bgl_uniform(idx: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding: idx,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device_if_f64_gpu_available;

    fn default_config() -> PopulationPkConfig {
        PopulationPkConfig {
            dose_mg: 500.0,
            f_bioavail: 0.8,
            base_cl: 10.0,
            cl_low: 0.5,
            cl_high: 1.5,
        }
    }

    #[tokio::test]
    async fn test_population_pk_basic() {
        let Some(device) = get_test_device_if_f64_gpu_available().await else {
            return;
        };
        let pk = PopulationPkF64::new(device, default_config()).unwrap();
        let results = pk.simulate(1024, 42).unwrap();
        assert_eq!(results.len(), 1024);

        for &auc in &results {
            assert!(auc > 0.0, "AUC must be positive, got {auc}");
            assert!(auc.is_finite(), "AUC must be finite, got {auc}");
        }
    }

    #[tokio::test]
    async fn test_population_pk_auc_bounds() {
        let Some(device) = get_test_device_if_f64_gpu_available().await else {
            return;
        };
        let cfg = default_config();
        let auc_min = cfg.f_bioavail * cfg.dose_mg / (cfg.base_cl * cfg.cl_high);
        let auc_max = cfg.f_bioavail * cfg.dose_mg / (cfg.base_cl * cfg.cl_low);

        let pk = PopulationPkF64::new(device, cfg).unwrap();
        let results = pk.simulate(4096, 123).unwrap();

        for &auc in &results {
            assert!(
                auc >= auc_min * 0.99 && auc <= auc_max * 1.01,
                "AUC {auc} outside expected range [{auc_min}, {auc_max}]"
            );
        }
    }

    #[tokio::test]
    async fn test_population_pk_deterministic() {
        let Some(device) = get_test_device_if_f64_gpu_available().await else {
            return;
        };
        let pk = PopulationPkF64::new(device, default_config()).unwrap();
        let r1 = pk.simulate(256, 42).unwrap();
        let r2 = pk.simulate(256, 42).unwrap();
        assert_eq!(r1, r2, "same seed must produce identical results");
    }

    #[tokio::test]
    async fn test_population_pk_different_seeds() {
        let Some(device) = get_test_device_if_f64_gpu_available().await else {
            return;
        };
        let pk = PopulationPkF64::new(device, default_config()).unwrap();
        let r1 = pk.simulate(256, 1).unwrap();
        let r2 = pk.simulate(256, 2).unwrap();
        assert_ne!(r1, r2, "different seeds should give different results");
    }

    #[tokio::test]
    async fn test_population_pk_empty() {
        let Some(device) = get_test_device_if_f64_gpu_available().await else {
            return;
        };
        let pk = PopulationPkF64::new(device, default_config()).unwrap();
        let results = pk.simulate(0, 42).unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_population_pk_validation() {
        let Some(device) = get_test_device_if_f64_gpu_available().await else {
            return;
        };
        assert!(
            PopulationPkF64::new(
                device.clone(),
                PopulationPkConfig {
                    dose_mg: -10.0,
                    ..default_config()
                }
            )
            .is_err(),
            "negative dose"
        );
        assert!(
            PopulationPkF64::new(
                device.clone(),
                PopulationPkConfig {
                    f_bioavail: 1.5,
                    ..default_config()
                }
            )
            .is_err(),
            "bioavailability > 1"
        );
        assert!(
            PopulationPkF64::new(
                device,
                PopulationPkConfig {
                    cl_low: 1.5,
                    cl_high: 0.5,
                    ..default_config()
                }
            )
            .is_err(),
            "cl_low > cl_high"
        );
    }
}

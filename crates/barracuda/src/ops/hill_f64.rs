// SPDX-License-Identifier: AGPL-3.0-or-later
//! `HillFunctionF64` — element-wise Hill dose-response activation (f64 precision)
//!
//! `E(x) = Emax × xⁿ / (Kⁿ + xⁿ)`
//!
//! Used in:
//! - Quorum-sensing / c-di-GMP regulatory cascades (wetSpring)
//! - Michaelis-Menten kinetics (Hill with n=1)
//! - Cooperativity models (ligand binding with n>1)
//! - Drug dose-response curves (healthSpring absorption)
//! - PFAS degradation rate models
//!
//! Set `emax = 1.0` for the normalized Hill activation in \[0, 1\].

use crate::device::WgpuDevice;
use crate::device::capabilities::WORKGROUP_SIZE_1D;
use crate::error::{BarracudaError, Result};
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct HillParamsGpu {
    n_elements: u32,
    _pad: u32,
    k: f64,
    n: f64,
    emax: f64,
}

/// Element-wise Hill dose-response: `E(x_i) = Emax × x_iⁿ / (Kⁿ + x_iⁿ)`
///
/// Output is in \[0, Emax\]; inputs are clamped to ≥ 0 before evaluation.
/// With `emax = 1.0` this is the classic normalized Hill activation.
///
/// # Example
/// ```ignore
/// // Autoinducer concentrations → HapR activation probabilities
/// let activations = HillFunctionF64::new(device, K_h, n_h)?
///     .apply(&autoinducer_concs)?;
///
/// // Drug dose-response with 95% max efficacy
/// let responses = HillFunctionF64::dose_response(device, ec50, hill_n, 0.95)?
///     .apply(&drug_concentrations)?;
/// ```
pub struct HillFunctionF64 {
    device: Arc<WgpuDevice>,
    /// Half-saturation constant K (EC₅₀).
    pub k: f64,
    /// Hill coefficient n (cooperativity exponent; 1 = Michaelis-Menten).
    pub n: f64,
    /// Maximum effect (Emax). Default 1.0 for normalized output.
    pub emax: f64,
}

impl HillFunctionF64 {
    fn wgsl_shader() -> &'static str {
        include_str!("../shaders/math/hill_f64.wgsl")
    }

    /// Create a Hill activation with the given K (EC₅₀) and cooperativity n.
    ///
    /// Emax defaults to 1.0 (normalized Hill in \[0, 1\]).
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if K or n are not positive.
    pub fn new(device: Arc<WgpuDevice>, k: f64, n: f64) -> Result<Self> {
        Self::dose_response(device, k, n, 1.0)
    }

    /// Create a Hill dose-response with explicit Emax.
    ///
    /// `E(x) = emax × xⁿ / (Kⁿ + xⁿ)`
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if K, n, or emax are not positive.
    pub fn dose_response(device: Arc<WgpuDevice>, k: f64, n: f64, emax: f64) -> Result<Self> {
        if k <= 0.0 {
            return Err(BarracudaError::InvalidInput {
                message: format!("HillFunctionF64: K must be > 0, got {k}"),
            });
        }
        if n <= 0.0 {
            return Err(BarracudaError::InvalidInput {
                message: format!("HillFunctionF64: n must be > 0, got {n}"),
            });
        }
        if emax <= 0.0 {
            return Err(BarracudaError::InvalidInput {
                message: format!("HillFunctionF64: emax must be > 0, got {emax}"),
            });
        }
        Ok(Self { device, k, n, emax })
    }

    /// Apply Hill dose-response element-wise.
    ///
    /// `input` is a flat f64 slice. Returns a flat f64 Vec of the same length.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn apply(&self, input: &[f64]) -> Result<Vec<f64>> {
        let n_elements = input.len();
        if n_elements == 0 {
            return Ok(Vec::new());
        }

        let dev = &self.device;
        let params = HillParamsGpu {
            n_elements: n_elements as u32,
            _pad: 0,
            k: self.k,
            n: self.n,
            emax: self.emax,
        };

        let params_buf = dev
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("HillF64 Params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let input_buf = dev
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("HillF64 Input"),
                contents: bytemuck::cast_slice(input),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let output_size = std::mem::size_of_val(input) as u64;
        let output_buf = dev.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("HillF64 Output"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let bgl = dev
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("HillF64 BGL"),
                entries: &[bgl_storage_ro(0), bgl_storage_rw(1), bgl_uniform(2)],
            });

        let bg = dev.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("HillF64 BG"),
            layout: &bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buf.as_entire_binding(),
                },
            ],
        });

        let shader = dev.compile_shader_f64(Self::wgsl_shader(), Some("HillF64"));
        let pl = dev
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("HillF64 PL"),
                bind_group_layouts: &[&bgl],
                immediate_size: 0,
            });
        let pipeline = dev
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("HillF64 Pipeline"),
                layout: Some(&pl),
                module: &shader,
                entry_point: Some("main"),
                cache: None,
                compilation_options: Default::default(),
            });

        let mut encoder = dev.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("HillF64 Encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("HillF64 Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups((n_elements as u32).div_ceil(WORKGROUP_SIZE_1D), 1, 1);
        }
        dev.submit_and_poll(Some(encoder.finish()));

        crate::utils::read_buffer_f64(dev, &output_buf, n_elements)
    }
}

fn bgl_storage_ro(idx: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding: idx,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
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
    use crate::device::test_pool::get_test_device_if_gpu_available;

    #[tokio::test]
    async fn test_hill_michaelis_menten() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let k = 10.0_f64;
        let hill = HillFunctionF64::new(device, k, 1.0).unwrap();
        let result = hill.apply(&[0.0, k, 2.0 * k, 100.0 * k]).unwrap();
        assert!(result[0].abs() < 1e-10, "Hill(0)=0");
        assert!(
            (result[1] - 0.5).abs() < 1e-10,
            "Hill(K)=0.5, got {}",
            result[1]
        );
        assert!((result[2] - 2.0 / 3.0).abs() < 1e-10, "Hill(2K)=2/3");
        assert!(result[3] > 0.99, "Hill(100K)≈1");
    }

    #[tokio::test]
    async fn test_hill_cooperativity() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let k = 5.0_f64;
        let hill = HillFunctionF64::new(device, k, 2.0).unwrap();
        let result = hill.apply(&[k]).unwrap();
        assert!(
            (result[0] - 0.5).abs() < 1e-9,
            "Hill(K, K, 2)=0.5, got {}",
            result[0]
        );
    }

    #[tokio::test]
    async fn test_hill_output_range() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let vals: Vec<f64> = (0..=100).map(|i| i as f64).collect();
        let hill = HillFunctionF64::new(device, 50.0, 1.0).unwrap();
        let result = hill.apply(&vals).unwrap();
        for &v in &result {
            assert!((0.0..=1.0).contains(&v), "Hill output out of [0,1]: {v}");
        }
    }

    #[tokio::test]
    async fn test_dose_response_emax() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let ec50 = 10.0_f64;
        let emax = 0.95;
        let hill = HillFunctionF64::dose_response(device, ec50, 2.0, emax).unwrap();
        let result = hill.apply(&[0.0, ec50, 100.0 * ec50]).unwrap();

        assert!(result[0].abs() < 1e-10, "E(0)=0");
        assert!(
            (result[1] - emax * 0.5).abs() < 1e-9,
            "E(EC50)=Emax/2, got {}",
            result[1]
        );
        assert!(
            (result[2] - emax).abs() < 0.01,
            "E(100×EC50)≈Emax, got {}",
            result[2]
        );
    }

    #[tokio::test]
    async fn test_dose_response_output_bounded_by_emax() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let emax = 42.0;
        let vals: Vec<f64> = (0..=200).map(|i| i as f64).collect();
        let hill = HillFunctionF64::dose_response(device, 50.0, 1.0, emax).unwrap();
        let result = hill.apply(&vals).unwrap();
        for &v in &result {
            assert!(
                (0.0..=emax).contains(&v),
                "dose-response output {v} exceeds Emax {emax}"
            );
        }
    }

    #[tokio::test]
    async fn test_dose_response_validation() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        assert!(
            HillFunctionF64::dose_response(device.clone(), 10.0, 1.0, -1.0).is_err(),
            "negative emax should fail"
        );
        assert!(
            HillFunctionF64::dose_response(device.clone(), 10.0, 1.0, 0.0).is_err(),
            "zero emax should fail"
        );
        assert!(
            HillFunctionF64::dose_response(device, 10.0, 1.0, 0.5).is_ok(),
            "positive emax should succeed"
        );
    }
}

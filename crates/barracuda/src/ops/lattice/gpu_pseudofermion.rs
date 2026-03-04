// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU pseudofermion operations: heatbath noise and fermion force.
//!
//! The heatbath generates Gaussian noise η; the actual φ = D†η is performed
//! by dispatching the staggered Dirac operator from `dirac.rs`.
//!
//! The fermion force computes dS_F/dU from CG solution fields.

use crate::device::WgpuDevice;
use crate::error::Result;
use std::sync::Arc;

use super::complex_f64::WGSL_COMPLEX64;
use super::lcg::WGSL_LCG_F64;
use super::su3::su3_preamble;

const WG: u32 = 64;
const HEATBATH_SHADER: &str = include_str!("../../shaders/lattice/pseudofermion_heatbath_f64.wgsl");
const FORCE_SHADER: &str = include_str!("../../shaders/lattice/pseudofermion_force_f64.wgsl");

// ── Heatbath ────────────────────────────────────────────────────────────────

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct HeatbathParams {
    volume: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

/// GPU pseudofermion heatbath — generates Gaussian noise fermion field.
pub struct GpuPseudofermionHeatbath {
    device: Arc<WgpuDevice>,
    volume: u32,
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    params: wgpu::Buffer,
}

impl GpuPseudofermionHeatbath {
    pub fn new(device: Arc<WgpuDevice>, volume: u32) -> Result<Self> {
        let src = format!("{WGSL_COMPLEX64}\n{WGSL_LCG_F64}\n{HEATBATH_SHADER}");
        let module = device.compile_shader_f64(&src, Some("pf_heatbath"));

        let bgl = device
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("GpuPfHeatbath:bgl"),
                entries: &[
                    uniform_bgl(0),
                    storage_bgl(1, false), // eta
                    storage_bgl(2, false), // rng_state
                ],
            });

        let layout = device
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("GpuPfHeatbath:layout"),
                bind_group_layouts: &[&bgl],
                push_constant_ranges: &[],
            });

        let pipeline = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("GpuPfHeatbath:pipeline"),
                layout: Some(&layout),
                module: &module,
                entry_point: "heatbath_noise",
                compilation_options: Default::default(),
                cache: None,
            });

        let params_data = HeatbathParams {
            volume,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let params = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GpuPfHeatbath:params"),
            size: std::mem::size_of::<HeatbathParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        device
            .queue
            .write_buffer(&params, 0, bytemuck::bytes_of(&params_data));

        Ok(Self {
            device,
            volume,
            pipeline,
            bgl,
            params,
        })
    }

    /// Generate Gaussian noise into `eta_buf`.
    ///
    /// * `eta_buf`     — `[V × 6]` f64 (3 colors × 2)
    /// * `rng_buf`     — `[V]` u64 (per-site RNG state)
    pub fn generate(&self, eta_buf: &wgpu::Buffer, rng_buf: &wgpu::Buffer) -> Result<()> {
        let bg = self
            .device
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("GpuPfHeatbath:bg"),
                layout: &self.bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.params.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: eta_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: rng_buf.as_entire_binding(),
                    },
                ],
            });

        let mut enc = self
            .device
            .create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
                label: Some("GpuPfHeatbath:enc"),
            });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("GpuPfHeatbath:pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(self.volume.div_ceil(WG), 1, 1);
        }
        self.device.submit_and_poll(Some(enc.finish()));
        Ok(())
    }

    pub fn volume(&self) -> u32 {
        self.volume
    }
}

// ── Pseudofermion Force ─────────────────────────────────────────────────────

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct PFForceParams {
    nt: u32,
    nx: u32,
    ny: u32,
    nz: u32,
    volume: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

/// GPU pseudofermion force: dS_F/dU from CG solution fields.
pub struct GpuPseudofermionForce {
    device: Arc<WgpuDevice>,
    volume: u32,
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
    params: wgpu::Buffer,
}

impl GpuPseudofermionForce {
    pub fn new(device: Arc<WgpuDevice>, nt: u32, nx: u32, ny: u32, nz: u32) -> Result<Self> {
        let volume = nt * nx * ny * nz;
        let src = format!("{}{}", su3_preamble(), FORCE_SHADER);
        let module = device.compile_shader_f64(&src, Some("pf_force"));

        let bgl = device
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("GpuPfForce:bgl"),
                entries: &[
                    uniform_bgl(0),
                    storage_bgl(1, true),  // links
                    storage_bgl(2, true),  // x_field
                    storage_bgl(3, true),  // y_field
                    storage_bgl(4, false), // force
                ],
            });

        let layout = device
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("GpuPfForce:layout"),
                bind_group_layouts: &[&bgl],
                push_constant_ranges: &[],
            });

        let pipeline = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("GpuPfForce:pipeline"),
                layout: Some(&layout),
                module: &module,
                entry_point: "pseudofermion_force_kernel",
                compilation_options: Default::default(),
                cache: None,
            });

        let params_data = PFForceParams {
            nt,
            nx,
            ny,
            nz,
            volume,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let params = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GpuPfForce:params"),
            size: std::mem::size_of::<PFForceParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        device
            .queue
            .write_buffer(&params, 0, bytemuck::bytes_of(&params_data));

        Ok(Self {
            device,
            volume,
            pipeline,
            bgl,
            params,
        })
    }

    /// Compute pseudofermion force for all links.
    ///
    /// * `links_buf`   — `[V × 4 × 18]` f64
    /// * `x_field_buf` — `[V × 6]` f64 (CG solution)
    /// * `y_field_buf` — `[V × 6]` f64 (D·X)
    /// * `force_buf`   — `[V × 4 × 18]` f64 (output)
    pub fn compute(
        &self,
        links_buf: &wgpu::Buffer,
        x_field_buf: &wgpu::Buffer,
        y_field_buf: &wgpu::Buffer,
        force_buf: &wgpu::Buffer,
    ) -> Result<()> {
        let bg = self
            .device
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("GpuPfForce:bg"),
                layout: &self.bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.params.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: links_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: x_field_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: y_field_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: force_buf.as_entire_binding(),
                    },
                ],
            });

        let mut enc = self
            .device
            .create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
                label: Some("GpuPfForce:enc"),
            });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("GpuPfForce:pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(self.volume.div_ceil(WG), 1, 1);
        }
        self.device.submit_and_poll(Some(enc.finish()));
        Ok(())
    }

    pub fn volume(&self) -> u32 {
        self.volume
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

fn uniform_bgl(binding: u32) -> wgpu::BindGroupLayoutEntry {
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

#[expect(clippy::unwrap_used, reason = "tests")]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_heatbath_pipeline_creation() {
        let Some(device) = crate::device::test_pool::get_test_device_if_f64_gpu_available_sync()
        else {
            return;
        };
        let op = GpuPseudofermionHeatbath::new(device, 16).unwrap();
        assert_eq!(op.volume(), 16);
    }

    #[test]
    fn test_pf_force_pipeline_creation() {
        let Some(device) = crate::device::test_pool::get_test_device_if_f64_gpu_available_sync()
        else {
            return;
        };
        let op = GpuPseudofermionForce::new(device, 2, 2, 2, 2).unwrap();
        assert_eq!(op.volume(), 16);
    }
}

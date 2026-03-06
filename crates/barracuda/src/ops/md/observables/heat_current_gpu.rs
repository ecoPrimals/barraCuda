// SPDX-License-Identifier: AGPL-3.0-or-later

//! GPU heat current for Green-Kubo thermal conductivity (f64).
//!
//! Computes the microscopic heat current `J_q` per particle from positions,
//! velocities, and Yukawa interaction parameters. Output is per-particle
//! [`J_x`, `J_y`, `J_z`] f64 vectors; host sums to get total `J_q(t)`.
//!
//! Absorbed from hotSpring CPU `compute_heat_current()` → GPU shader.

use crate::device::WgpuDevice;
use crate::error::Result;
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;

const SHADER: &str = include_str!("../../../shaders/md/observables/heat_current_f64.wgsl");
const WG: u32 = 64;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct HeatParams {
    n: u32,
    _pad0: u32,
    box_side: f64,
    kappa: f64,
    mass: f64,
}

/// Per-particle heat current GPU kernel (Yukawa interaction, f64).
pub struct HeatCurrentGpu {
    device: Arc<WgpuDevice>,
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
}

impl HeatCurrentGpu {
    /// Creates a new heat current GPU kernel for the given WGPU device.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn new(device: Arc<WgpuDevice>) -> Result<Self> {
        let module = device.compile_shader_f64(SHADER, Some("heat_current_f64"));

        let bgl = device
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("HeatCurrent:bgl"),
                entries: &[
                    uniform_bgl(0),
                    storage_bgl(1, true),  // pos
                    storage_bgl(2, true),  // vel
                    storage_bgl(3, false), // jq_out
                ],
            });

        let layout = device
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("HeatCurrent:layout"),
                bind_group_layouts: &[&bgl],
                immediate_size: 0,
            });

        let pipeline = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("HeatCurrent:pipeline"),
                layout: Some(&layout),
                module: &module,
                entry_point: Some("heat_current"),
                compilation_options: Default::default(),
                cache: None,
            });

        Ok(Self {
            device,
            pipeline,
            bgl,
        })
    }

    /// Dispatch heat current computation.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation or GPU dispatch fails (e.g. device lost).
    ///
    /// * `pos_buf` — `[N × 3]` f64 positions
    /// * `vel_buf` — `[N × 3]` f64 velocities
    /// * `jq_buf`  — `[N × 3]` f64 output (per-particle `J_q`)
    pub fn dispatch(
        &self,
        pos_buf: &wgpu::Buffer,
        vel_buf: &wgpu::Buffer,
        jq_buf: &wgpu::Buffer,
        n: u32,
        box_side: f64,
        kappa: f64,
        mass: f64,
    ) -> Result<()> {
        let params_data = HeatParams {
            n,
            _pad0: 0,
            box_side,
            kappa,
            mass,
        };
        let params = self.device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("HeatCurrent:params"),
            size: std::mem::size_of::<HeatParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.device
            .queue
            .write_buffer(&params, 0, bytemuck::bytes_of(&params_data));

        let bg = self
            .device
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("HeatCurrent:bg"),
                layout: &self.bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: params.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: pos_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: vel_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: jq_buf.as_entire_binding(),
                    },
                ],
            });

        let mut enc = self
            .device
            .create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
                label: Some("HeatCurrent:enc"),
            });
        {
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("HeatCurrent:pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups(n.div_ceil(WG), 1, 1);
        }
        self.device.submit_and_poll(Some(enc.finish()));
        Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_heat_current_pipeline_creation() {
        let Some(device) = crate::device::test_pool::get_test_device_if_f64_gpu_available_sync()
        else {
            return;
        };
        let hc = HeatCurrentGpu::new(device).unwrap();
        assert!(std::mem::size_of_val(&hc) > 0);
    }
}

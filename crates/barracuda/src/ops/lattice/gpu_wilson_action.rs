// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU Wilson action computation.
//!
//! Per-site action contribution dispatched on GPU; host-side reduction
//! via `ReduceScalarPipeline` yields the total Wilson action.

use crate::device::WgpuDevice;
use crate::device::capabilities::WORKGROUP_SIZE_COMPACT;
use crate::device::compute_pipeline::ComputeDispatch;
use crate::device::driver_profile::{Fp64Strategy, GpuDriverProfile};
use crate::error::Result;
use std::sync::Arc;

use super::su3::{su3_df64_preamble, su3_preamble};
const SHADER_BODY: &str = include_str!("../../shaders/lattice/wilson_action_f64.wgsl");
const SHADER_DF64: &str = include_str!("../../shaders/lattice/wilson_action_df64.wgsl");

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ActionParams {
    nt: u32,
    nx: u32,
    ny: u32,
    nz: u32,
    volume: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

/// GPU Wilson action operator.
pub struct GpuWilsonAction {
    device: Arc<WgpuDevice>,
    volume: u32,
    shader_src: String,
    params: wgpu::Buffer,
}

impl GpuWilsonAction {
    /// Compile the Wilson action pipeline.
    /// Automatically selects DF64 (f32-pair) shaders on consumer GPUs,
    /// routing plaquette products through FP32 cores for ~10x throughput.
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn new(device: Arc<WgpuDevice>, nt: u32, nx: u32, ny: u32, nz: u32) -> Result<Self> {
        let volume = nt * nx * ny * nz;

        let profile = GpuDriverProfile::from_device(&device);
        let strategy = profile.fp64_strategy();
        let shader_src = match strategy {
            Fp64Strategy::Sovereign | Fp64Strategy::Native | Fp64Strategy::Concurrent => {
                format!("{}{}", su3_preamble(), SHADER_BODY)
            }
            Fp64Strategy::Hybrid => format!("{}{}", su3_df64_preamble(), SHADER_DF64),
        };
        tracing::info!(
            ?strategy,
            "GpuWilsonAction: compiled with {:?} FP64 strategy",
            strategy
        );

        let params_data = ActionParams {
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
            label: Some("GpuWilsonAction:params"),
            size: std::mem::size_of::<ActionParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        device
            .queue
            .write_buffer(&params, 0, bytemuck::bytes_of(&params_data));

        Ok(Self {
            device,
            volume,
            shader_src,
            params,
        })
    }

    /// Compute per-site Wilson action contributions.
    ///   * `links_buf`  — `[V × 4 × 18]` f64 (gauge config)
    ///   * `action_buf` — `[V]` f64 (per-site output)
    ///
    /// Multiply total sum by β for the full Wilson action.
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn compute(&self, links_buf: &wgpu::Buffer, action_buf: &wgpu::Buffer) -> Result<()> {
        ComputeDispatch::new(self.device.as_ref(), "GpuWilsonAction")
            .shader(&self.shader_src, "wilson_action_kernel")
            .f64()
            .uniform(0, &self.params)
            .storage_read(1, links_buf)
            .storage_rw(2, action_buf)
            .dispatch(self.volume.div_ceil(WORKGROUP_SIZE_COMPACT), 1, 1)
            .submit()?;
        Ok(())
    }

    /// Lattice volume.
    #[must_use]
    pub fn volume(&self) -> u32 {
        self.volume
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wilson_action_pipeline_creation() {
        let Some(device) = crate::device::test_pool::get_test_device_if_f64_gpu_available_sync()
        else {
            return;
        };
        let op = GpuWilsonAction::new(device, 2, 2, 2, 2).unwrap();
        assert_eq!(op.volume(), 16);
    }

    #[test]
    fn test_wilson_action_cold_start_is_zero_gpu() {
        let Some(device) = crate::device::test_pool::get_test_device_if_f64_gpu_available_sync()
        else {
            return;
        };

        let (nt, nx, ny, nz) = (2u32, 2, 2, 2);
        let volume = (nt * nx * ny * nz) as usize;

        let identity_18: [f64; 18] = [
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            0.0,
        ];
        let links_f64: Vec<f64> = std::iter::repeat_n(identity_18.iter().copied(), volume * 4)
            .flatten()
            .collect();

        let link_bytes: &[u8] = bytemuck::cast_slice(&links_f64);
        let links_buf = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test:links"),
            size: link_bytes.len() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        device.queue.write_buffer(&links_buf, 0, link_bytes);

        let action_bytes = volume * std::mem::size_of::<f64>();
        let action_buf = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test:action"),
            size: action_bytes as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let op = GpuWilsonAction::new(device.clone(), nt, nx, ny, nz).unwrap();
        op.compute(&links_buf, &action_buf).unwrap();

        let staging = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test:staging"),
            size: action_bytes as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = device.create_encoder_guarded(&Default::default());
        enc.copy_buffer_to_buffer(&action_buf, 0, &staging, 0, action_bytes as u64);
        device.submit_and_poll(Some(enc.finish()));

        let action_out: Vec<f64> = device.map_staging_buffer(&staging, volume).unwrap();

        let total: f64 = action_out.iter().sum();
        assert!(
            total.abs() < 1e-10,
            "Wilson action for cold start should be 0, got {total}"
        );
    }
}

// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU lattice initialization: cold start and hot start.
//!
//! Replaces CPU-only `wilson.rs` `cold_start/hot_start` with GPU shaders.

use crate::device::WgpuDevice;
use crate::device::capabilities::WORKGROUP_SIZE_COMPACT;
use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::Result;
use std::sync::Arc;

use super::su3_extended::su3_extended_preamble;
const SHADER_BODY: &str = include_str!("../../shaders/lattice/lattice_init_f64.wgsl");

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct InitParams {
    volume: u32,
    n_links: u32,
    _pad0: u32,
    _pad1: u32,
    epsilon: f64,
    _padf: f64,
}

/// GPU lattice initializer — cold start (identity) or hot start (random near identity).
pub struct GpuLatticeInit {
    device: Arc<WgpuDevice>,
    n_links: u32,
    shader_src: String,
}

impl GpuLatticeInit {
    /// Create lattice initializer for given volume.
    /// # Errors
    /// Returns [`Err`] if shader compilation or pipeline creation fails.
    pub fn new(device: Arc<WgpuDevice>, volume: u32) -> Result<Self> {
        let n_links = volume * 4;
        let shader_src = format!("{}{}", su3_extended_preamble(), SHADER_BODY);

        Ok(Self {
            device,
            n_links,
            shader_src,
        })
    }

    /// Initialize all links to SU(3) identity.
    /// # Errors
    /// Returns [`Err`] if GPU dispatch fails or the device is lost.
    pub fn cold_start(
        &self,
        links_buf: &wgpu::Buffer,
        rng_buf: &wgpu::Buffer,
        volume: u32,
    ) -> Result<()> {
        self.dispatch(links_buf, rng_buf, volume, 0.0, "cold_start", "cold_start")
    }

    /// Initialize links with random SU(3) near identity.
    /// # Errors
    /// Returns [`Err`] if GPU dispatch fails or the device is lost.
    pub fn hot_start(
        &self,
        links_buf: &wgpu::Buffer,
        rng_buf: &wgpu::Buffer,
        volume: u32,
        epsilon: f64,
    ) -> Result<()> {
        self.dispatch(
            links_buf,
            rng_buf,
            volume,
            epsilon,
            "hot_start",
            "hot_start",
        )
    }

    fn dispatch(
        &self,
        links_buf: &wgpu::Buffer,
        rng_buf: &wgpu::Buffer,
        volume: u32,
        epsilon: f64,
        entry_point: &str,
        label: &str,
    ) -> Result<()> {
        let params_data = InitParams {
            volume,
            n_links: self.n_links,
            _pad0: 0,
            _pad1: 0,
            epsilon,
            _padf: 0.0,
        };
        let params = self.device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(&format!("GpuLatticeInit:{label}:params")),
            size: std::mem::size_of::<InitParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.device
            .queue
            .write_buffer(&params, 0, bytemuck::bytes_of(&params_data));

        ComputeDispatch::new(&self.device, &format!("GpuLatticeInit:{label}"))
            .shader(&self.shader_src, entry_point)
            .f64()
            .uniform(0, &params)
            .storage_rw(1, links_buf)
            .storage_rw(2, rng_buf)
            .dispatch(self.n_links.div_ceil(WORKGROUP_SIZE_COMPACT), 1, 1)
            .submit()?;

        Ok(())
    }

    /// Number of gauge links (volume × 4).
    #[must_use]
    pub fn n_links(&self) -> u32 {
        self.n_links
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init_pipeline_creation() {
        let Some(device) = crate::device::test_pool::get_test_device_if_f64_gpu_available_sync()
        else {
            return;
        };
        let init = GpuLatticeInit::new(device, 16).unwrap();
        assert_eq!(init.n_links(), 64);
    }

    #[test]
    fn test_cold_start_identity_gpu() {
        let Some(device) = crate::device::test_pool::get_test_device_if_f64_gpu_available_sync()
        else {
            return;
        };

        let volume = 16u32;
        let n_links = volume * 4;
        let init = GpuLatticeInit::new(device.clone(), volume).unwrap();

        let links_bytes = (n_links as usize) * 18 * std::mem::size_of::<f64>();
        let links_buf = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test:links"),
            size: links_bytes as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let rng_bytes = (n_links as usize) * std::mem::size_of::<u32>();
        let rng_buf = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test:rng"),
            size: rng_bytes as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        init.cold_start(&links_buf, &rng_buf, volume).unwrap();

        let staging = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test:staging"),
            size: links_bytes as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = device.create_encoder_guarded(&Default::default());
        enc.copy_buffer_to_buffer(&links_buf, 0, &staging, 0, links_bytes as u64);
        device.submit_commands(Some(enc.finish()));

        let n_f64 = links_bytes / std::mem::size_of::<f64>();
        let data: Vec<f64> = device.map_staging_buffer(&staging, n_f64).unwrap();

        for link in 0..n_links as usize {
            for i in 0..9 {
                let re = data[link * 18 + i * 2];
                let im = data[link * 18 + i * 2 + 1];
                let (exp_re, exp_im) = if i == 0 || i == 4 || i == 8 {
                    (1.0, 0.0)
                } else {
                    (0.0, 0.0)
                };
                assert!(
                    (re - exp_re).abs() < 1e-10 && (im - exp_im).abs() < 1e-10,
                    "link {link} elem {i}: ({re}, {im}) expected ({exp_re}, {exp_im})"
                );
            }
        }
    }
}

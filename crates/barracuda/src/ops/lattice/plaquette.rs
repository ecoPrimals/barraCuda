// SPDX-License-Identifier: AGPL-3.0-or-later
//! Wilson plaquette GPU operation for SU(3) lattice gauge theory.
//!
//! Computes `Re Tr(U_p) / 3` for all 6 plane orientations at every site on a
//! 4D periodic lattice in a single GPU dispatch.
//!
//! # Usage
//!
//! ```ignore
//! let op = WilsonPlaquette::new(device.clone(), nt, nx, ny, nz)?;
//! // `links_buf` holds [V × 4 × 18] f64 in row-major SU(3) storage format
//! op.compute(&links_buf, &plaq_buf)?;
//! // Average plaquette via ReduceScalarPipeline::sum_f64(&plaq_buf, volume*6)
//! // then divide by (volume * 6).
//! ```
//!
//! # hotSpring validation
//!
//! CPU reference in hotSpring `lattice/wilson.rs`.  Expected average plaquette
//! for a thermalized SU(3) config at β=6: ≈ 0.5937 (Wilson action).

use crate::device::compute_pipeline::ComputeDispatch;
use crate::device::driver_profile::{Fp64Strategy, GpuDriverProfile};
use crate::device::WgpuDevice;
use crate::error::Result;
use std::sync::Arc;

use super::su3::{su3_df64_preamble, su3_preamble};

const PLAQ_WG: u32 = 64;
const PLAQ_SHADER_BODY: &str = include_str!("../../shaders/lattice/wilson_plaquette_f64.wgsl");
const PLAQ_SHADER_DF64: &str = include_str!("../../shaders/lattice/wilson_plaquette_df64.wgsl");

/// Wilson plaquette operator on a 4D SU(3) lattice.
pub struct WilsonPlaquette {
    device: Arc<WgpuDevice>,
    volume: u32,
    shader_src: String,
    params: wgpu::Buffer,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct PlaqParams {
    nt: u32,
    nx: u32,
    ny: u32,
    nz: u32,
    volume: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

impl WilsonPlaquette {
    /// Compile the plaquette pipeline for a lattice of dimensions `nt×nx×ny×nz`.
    ///
    /// Automatically selects DF64 (f32-pair) shaders on consumer GPUs,
    /// routing plaquette SU(3) products through FP32 cores for ~10x throughput.
    pub fn new(device: Arc<WgpuDevice>, nt: u32, nx: u32, ny: u32, nz: u32) -> Result<Self> {
        let volume = nt * nx * ny * nz;

        let profile = GpuDriverProfile::from_device(&device);
        let strategy = profile.fp64_strategy();
        let shader_src = match strategy {
            Fp64Strategy::Native | Fp64Strategy::Concurrent => {
                format!("{}{}", su3_preamble(), PLAQ_SHADER_BODY)
            }
            Fp64Strategy::Hybrid => format!("{}{}", su3_df64_preamble(), PLAQ_SHADER_DF64),
        };
        tracing::info!(
            ?strategy,
            "WilsonPlaquette: compiled with {:?} FP64 strategy",
            strategy
        );

        let params_data = PlaqParams {
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
            label: Some("WilsonPlaquette:params"),
            size: std::mem::size_of::<PlaqParams>() as u64,
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

    /// Compute `Re Tr(U_p) / 3` for all plaquettes.
    ///
    /// * `links_buf` — `[V × 4 × 18]` f64 storage buffer (GPU-resident)
    /// * `plaq_buf`  — `[V × 6]` f64 storage buffer (output, GPU-resident)
    pub fn compute(&self, links_buf: &wgpu::Buffer, plaq_buf: &wgpu::Buffer) -> Result<()> {
        ComputeDispatch::new(self.device.as_ref(), "WilsonPlaquette")
            .shader(&self.shader_src, "plaquette")
            .f64()
            .uniform(0, &self.params)
            .storage_read(1, links_buf)
            .storage_rw(2, plaq_buf)
            .dispatch(self.volume.div_ceil(PLAQ_WG), 1, 1)
            .submit();
        Ok(())
    }

    /// Number of lattice sites.
    pub fn volume(&self) -> u32 {
        self.volume
    }

    /// Total number of plaquette values in the output buffer (`volume × 6`).
    pub fn n_plaquettes(&self) -> u32 {
        self.volume * 6
    }
}

#[expect(clippy::unwrap_used, reason = "tests")]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_n_plaquettes() {
        // 4^4 = 256 sites × 6 plane orientations
        let vol: u32 = 4 * 4 * 4 * 4;
        assert_eq!(vol * 6, 1536);
    }

    #[test]
    fn test_shader_source_includes_preamble() {
        let src = format!("{}{}", su3_preamble(), PLAQ_SHADER_BODY);
        assert!(src.contains("fn c64_mul"));
        assert!(src.contains("fn su3_mul"));
        assert!(src.contains("fn plaquette"));
    }

    /// All-identity link configuration must produce plaquette = 1.0 for every
    /// plane at every site.  This validates the full WGSL path:
    ///   su3_load → su3_plaquette → su3_re_trace / 3 → write.
    ///
    /// On a 2×2×2×2 lattice (16 sites):
    ///   U_p = I · I · I† · I† = I  →  Re Tr(I)/3 = 3/3 = 1.0
    #[test]
    fn test_plaquette_identity_links_gpu() {
        let Some(device) = crate::device::test_pool::get_test_device_if_f64_gpu_available_sync()
        else {
            return;
        };

        let (nt, nx, ny, nz) = (2u32, 2, 2, 2);
        let volume = (nt * nx * ny * nz) as usize; // 16

        // SU(3) identity: 9 complex entries row-major → 18 f64
        // diagonal (1+0i, 1+0i, 1+0i), off-diagonal 0+0i
        let identity_18: [f64; 18] = [
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            0.0,
        ];

        let links_f64: Vec<f64> = std::iter::repeat_n(identity_18.iter().copied(), volume * 4)
            .flatten()
            .collect(); // V × 4 directions

        let plaq_len = volume * 6; // V × 6 planes

        // Upload links
        let link_bytes: &[u8] = bytemuck::cast_slice(&links_f64);
        let links_buf = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test:links"),
            size: link_bytes.len() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        device.queue.write_buffer(&links_buf, 0, link_bytes);

        // Output buffer
        let plaq_bytes = plaq_len * std::mem::size_of::<f64>();
        let plaq_buf = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test:plaq"),
            size: plaq_bytes as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let op = WilsonPlaquette::new(device.clone(), nt, nx, ny, nz).unwrap();
        op.compute(&links_buf, &plaq_buf).unwrap();

        // Readback
        let staging = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test:plaq_staging"),
            size: plaq_bytes as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = device.create_encoder_guarded(&Default::default());
        enc.copy_buffer_to_buffer(&plaq_buf, 0, &staging, 0, plaq_bytes as u64);
        device.submit_and_poll(Some(enc.finish()));

        let plaq_out: Vec<f64> = device.map_staging_buffer(&staging, plaq_len).unwrap();

        for (i, &v) in plaq_out.iter().enumerate() {
            assert!(
                (v - 1.0).abs() < 1e-10,
                "plaq[{i}] = {v:.15e}, expected 1.0 (identity links)"
            );
        }
    }
}

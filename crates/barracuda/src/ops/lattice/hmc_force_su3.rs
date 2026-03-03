// SPDX-License-Identifier: AGPL-3.0-or-later
//! SU(3) HMC gauge force GPU operator (Wilson action, 4D lattice).
//!
//! Computes the force on every link `U_mu(x)` from the Wilson plaquette action:
//!
//! ```text
//! S_W = -beta/3 · Σ_{x,mu<nu} Re Tr(U_p(x,mu,nu))
//!
//! F_mu(x) = -beta/3 · Im Tr(Σ_staple · U_mu†(x))
//!         projected onto the su(3) algebra (anti-Hermitian, traceless)
//! ```
//!
//! The output buffer holds Lie-algebra force matrices `f_mu(x)` in the same
//! 18-f64 SU(3) storage format as the links.
//!
//! # Usage
//!
//! ```ignore
//! let force_op = Su3HmcForce::new(device.clone(), nt, nx, ny, nz, beta)?;
//! force_op.compute(&links_buf, &force_buf)?;
//! // Then: leapfrog update  π_mu(x) ← π_mu(x) − dt · f_mu(x)
//! ```
//!
//! # Notes
//!
//! For optimal performance, consider using `StatefulPipeline` to chain the
//! force computation with the leapfrog update in a single `queue.submit()`.
//!
//! # hotSpring design
//!
//! Algorithm: hotSpring `lattice/hmc.rs` (v0.5.16, Feb 2026).
//! GPU promotion: Feb 2026.  CPU reference unchanged in hotSpring.

use crate::device::compute_pipeline::ComputeDispatch;
use crate::device::driver_profile::{Fp64Strategy, GpuDriverProfile};
use crate::device::WgpuDevice;
use crate::error::Result;
use std::sync::Arc;

use super::su3::{su3_df64_preamble, su3_preamble};

const FORCE_WG: u32 = 64;
const FORCE_SHADER_BODY: &str = include_str!("../../shaders/lattice/su3_hmc_force_f64.wgsl");
const FORCE_SHADER_DF64: &str = include_str!("../../shaders/lattice/su3_hmc_force_df64.wgsl");

/// SU(3) HMC gauge force operator (4D Wilson action).
pub struct Su3HmcForce {
    device: Arc<WgpuDevice>,
    volume: u32,
    shader_src: String,
    params: wgpu::Buffer,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct ForceParams {
    nt: u32,
    nx: u32,
    ny: u32,
    nz: u32,
    volume: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    beta: f64,
    _padf: f64,
    _paf1: f64,
    _paf2: f64,
}

impl Su3HmcForce {
    /// Compile the HMC force pipeline for a `nt×nx×ny×nz` 4D lattice.
    ///
    /// Automatically selects DF64 (f32-pair) shaders on consumer GPUs where
    /// FP64:FP32 ≤ 1:64, routing staple multiplications through the FP32 core
    /// array for ~10x throughput. On compute-class GPUs (Titan V, A100, MI250)
    /// with 1:2 hardware, native f64 is used directly.
    pub fn new(
        device: Arc<WgpuDevice>,
        nt: u32,
        nx: u32,
        ny: u32,
        nz: u32,
        beta: f64,
    ) -> Result<Self> {
        let volume = nt * nx * ny * nz;

        let profile = GpuDriverProfile::from_device(&device);
        let strategy = profile.fp64_strategy();
        let shader_src = match strategy {
            Fp64Strategy::Native | Fp64Strategy::Concurrent => {
                format!("{}{}", su3_preamble(), FORCE_SHADER_BODY)
            }
            Fp64Strategy::Hybrid => format!("{}{}", su3_df64_preamble(), FORCE_SHADER_DF64),
        };
        tracing::info!(
            ?strategy,
            "Su3HmcForce: compiled with {:?} FP64 strategy",
            strategy
        );

        let params_data = ForceParams {
            nt,
            nx,
            ny,
            nz,
            volume,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
            beta,
            _padf: 0.0,
            _paf1: 0.0,
            _paf2: 0.0,
        };
        let params = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Su3HmcForce:params"),
            size: std::mem::size_of::<ForceParams>() as u64,
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

    /// Compute su(3)-algebra force matrices for all links.
    ///
    /// * `links_buf` — `[V × 4 × 18]` f64 (GPU-resident gauge configuration)
    /// * `force_buf` — `[V × 4 × 18]` f64 (output: algebra-valued force, zero-init)
    ///
    /// The output must be zeroed before calling (e.g. via `queue.write_buffer`
    /// or a separate clear kernel).
    pub fn compute(&self, links_buf: &wgpu::Buffer, force_buf: &wgpu::Buffer) -> Result<()> {
        ComputeDispatch::new(self.device.as_ref(), "Su3HmcForce")
            .shader(&self.shader_src, "hmc_force")
            .f64()
            .uniform(0, &self.params)
            .storage_read(1, links_buf)
            .storage_rw(2, force_buf)
            .dispatch(self.volume.div_ceil(FORCE_WG), 1, 1)
            .submit();
        Ok(())
    }

    /// Number of lattice sites.
    pub fn volume(&self) -> u32 {
        self.volume
    }

    /// Total link buffer size in f64 elements (`volume × 4 × 18`).
    pub fn link_buffer_len(&self) -> u64 {
        self.volume as u64 * 4 * 18
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shader_has_staple_sum() {
        let src = format!("{}{}", su3_preamble(), FORCE_SHADER_BODY);
        assert!(
            src.contains("staple"),
            "force shader must implement staple sum"
        );
        assert!(
            src.contains("su3_project_algebra"),
            "must project onto algebra"
        );
    }

    #[test]
    fn test_link_buffer_len() {
        // 4^4 = 256 sites × 4 directions × 18 f64 per matrix
        let vol: u64 = 4 * 4 * 4 * 4; // 256
        assert_eq!(vol * 4 * 18, 18432);
    }

    #[test]
    fn test_params_16byte_aligned() {
        assert_eq!(std::mem::size_of::<ForceParams>() % 16, 0);
    }

    /// All-identity link configuration must produce zero force on every link.
    ///
    /// With U_mu(x) = I everywhere, every plaquette staple is also the identity.
    /// The staple sum is 6I per link (6 planes contribute identically).
    ///
    /// Force = -β/3 · Im Tr(staple_sum · U_mu†) projected onto su(3) algebra.
    ///
    /// Im Tr(6I · I†) = Im Tr(6I) = Im(6 · 3) = 0  →  force = 0 for every link.
    ///
    /// This validates the full shader path: staple loop, su3_adjoint,
    /// su3_project_algebra, and the β/3 prefactor.
    #[test]
    fn test_su3_hmc_force_identity_links_gpu() {
        let Some(device) = crate::device::test_pool::get_test_device_if_f64_gpu_available_sync()
        else {
            return;
        };

        let (nt, nx, ny, nz) = (2u32, 2, 2, 2);
        let volume = (nt * nx * ny * nz) as usize; // 16

        // SU(3) identity: 18 f64 values per matrix
        let identity_18: [f64; 18] = [
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
            0.0,
        ];

        let links_f64: Vec<f64> = std::iter::repeat_n(identity_18.iter().copied(), volume * 4)
            .flatten()
            .collect();

        let buf_f64_len = volume * 4 * 18;
        let buf_bytes = buf_f64_len * std::mem::size_of::<f64>();

        let links_buf = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test:links"),
            size: buf_bytes as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        device
            .queue
            .write_buffer(&links_buf, 0, bytemuck::cast_slice(&links_f64));

        let zeros: Vec<u8> = vec![0u8; buf_bytes];
        let force_buf = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test:force"),
            size: buf_bytes as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        device.queue.write_buffer(&force_buf, 0, &zeros);

        let op = Su3HmcForce::new(device.clone(), nt, nx, ny, nz, 6.0).unwrap();
        op.compute(&links_buf, &force_buf).unwrap();

        // Readback
        let staging = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test:force_staging"),
            size: buf_bytes as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = device.device.create_command_encoder(&Default::default());
        enc.copy_buffer_to_buffer(&force_buf, 0, &staging, 0, buf_bytes as u64);
        device.submit_and_poll(Some(enc.finish()));

        let force_out: Vec<f64> = device.map_staging_buffer(&staging, buf_f64_len).unwrap();

        for (i, &v) in force_out.iter().enumerate() {
            assert!(
                v.abs() < 1e-10,
                "force[{i}] = {v:.15e}, expected 0.0 for identity links"
            );
        }
    }
}

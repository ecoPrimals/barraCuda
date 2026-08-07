// SPDX-License-Identifier: AGPL-3.0-or-later

//! GPU-resident transport observables — batched VACF + stress-tensor ACF.
//!
//! **Absorbed from**: hotSpring v0.64 (Feb 2026)
//!
//! ## Architecture
//!
//! During MD production, velocity snapshots are stored in a single flat GPU
//! buffer ([`GpuVelocityRing`]). After production, the batched VACF shader
//! computes C(lag) for each lag in ONE dispatch (iterating over all time
//! origins inside the shader), then reduces to a scalar via
//! [`crate::ops::sum_reduce_f64`]. Total GPU round-trips: `n_lag × 2`.
//!
//! ## Shaders
//!
//! | File | Entry Point | Purpose |
//! |------|-------------|---------|
//! | `vacf_batch_f64.wgsl` | `main` | Batched C(lag) across all origins |
//! | `stress_virial_f64.wgsl` | `main` | Per-particle `σ_xy` for Green-Kubo viscosity |
//!
//! ## Deep Debt Compliance
//!
//! - WGSL shader-first (separate .wgsl files)
//! - Full f64 precision
//! - Zero unsafe code
//! - Capability-based: CPU fallback when no GPU available

use crate::device::WgpuDevice;
use crate::device::capabilities::WORKGROUP_SIZE_COMPACT;
use crate::device::compute_pipeline::ComputeDispatch;
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;

const VACF_BATCH_SHADER: &str =
    include_str!("../../../shaders/md/vacf_batch_per_particle_f64.wgsl");
const STRESS_VIRIAL_SHADER: &str =
    include_str!("../../../shaders/md/stress_virial_per_particle_f64.wgsl");

// ─── Batched VACF ────────────────────────────────────────────────────────────

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct VacfBatchParams {
    n: u32,
    n_frames: u32,
    lag: u32,
    stride: u32,
}

/// Batched VACF GPU pipeline — computes C(lag) in a single dispatch per lag.
///
/// Replaces `n_frames` individual dispatches with ONE dispatch that iterates
/// over all valid time origins inside the shader.
pub struct VacfBatchGpu {
    device: Arc<WgpuDevice>,
}

impl VacfBatchGpu {
    /// Create a batched VACF GPU pipeline.
    #[must_use]
    pub fn new(device: Arc<WgpuDevice>) -> Self {
        Self { device }
    }

    /// Compute the per-particle VACF contribution for a single `lag` value.
    ///
    /// `vel_ring_buf` must hold `[n_frames × n_particles × 3]` f64 values.
    /// Returns a buffer of `[n_particles]` f64 values (per-particle C(lag)),
    /// ready for reduction via `ReduceScalarPipeline`.
    #[expect(clippy::missing_panics_doc, reason = "dispatch submit is infallible on valid device")]
    pub fn dispatch(
        &self,
        vel_ring_buf: &wgpu::Buffer,
        out_buf: &wgpu::Buffer,
        n_particles: u32,
        n_frames: u32,
        lag: u32,
    ) {
        let stride = n_particles * 3;
        let params = VacfBatchParams {
            n: n_particles,
            n_frames,
            lag,
            stride,
        };
        let params_buf = self
            .device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("VacfBatch Params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        ComputeDispatch::new(&self.device, "VacfBatch")
            .shader(VACF_BATCH_SHADER, "main")
            .f64()
            .storage_read(0, vel_ring_buf)
            .storage_rw(1, out_buf)
            .uniform(2, &params_buf)
            .dispatch(n_particles.div_ceil(WORKGROUP_SIZE_COMPACT), 1, 1)
            .submit()
            .expect("VacfBatch dispatch");
    }
}

// ─── Stress virial ───────────────────────────────────────────────────────────

/// GPU stress tensor `σ_xy` operator for Green-Kubo viscosity.
///
/// Per-particle `σ_xy` = m·vx·vy + Σ_{j≠i} (`F_ij_x` · `r_ij_y)/(2r`)
/// Output is `[N]` f64 values, reduced via `ReduceScalarPipeline`.
pub struct StressVirialGpu {
    device: Arc<WgpuDevice>,
}

impl StressVirialGpu {
    /// Create a stress-virial GPU pipeline.
    #[must_use]
    pub fn new(device: Arc<WgpuDevice>) -> Self {
        Self { device }
    }

    /// Dispatch the stress-virial kernel.
    ///
    /// `pos_buf`:  `[N×3]` f64 — particle positions
    /// `vel_buf`:  `[N×3]` f64 — particle velocities
    /// `out_buf`:  `[N]`   f64 — per-particle `σ_xy` contribution
    /// `params`:   simulation parameters packed as `[8]` f64
    #[expect(clippy::missing_panics_doc, reason = "dispatch submit is infallible on valid device")]
    pub fn dispatch(
        &self,
        pos_buf: &wgpu::Buffer,
        vel_buf: &wgpu::Buffer,
        out_buf: &wgpu::Buffer,
        params_buf: &wgpu::Buffer,
        n_particles: u32,
    ) {
        ComputeDispatch::new(&self.device, "StressVirial")
            .shader(STRESS_VIRIAL_SHADER, "main")
            .f64()
            .storage_read(0, pos_buf)
            .storage_read(1, vel_buf)
            .storage_rw(2, out_buf)
            .storage_read(3, params_buf)
            .dispatch(n_particles.div_ceil(WORKGROUP_SIZE_COMPACT), 1, 1)
            .submit()
            .expect("StressVirial dispatch");
    }
}

// ─── GPU velocity ring buffer ────────────────────────────────────────────────

/// GPU-resident velocity ring buffer backed by a single flat wgpu buffer.
///
/// Layout: `vel_flat[snapshot_idx × stride + particle × 3 + component]`
/// where `stride = n_particles × 3`.
///
/// During MD production, velocity snapshots are GPU→GPU copied into sequential
/// slots. After production, the flat buffer is passed directly to
/// [`VacfBatchGpu`] for correlation.
pub struct GpuVelocityRing {
    /// Flat GPU buffer holding all velocity snapshots
    pub flat_buf: wgpu::Buffer,
    /// Number of snapshot slots in the ring
    pub n_slots: usize,
    /// Current write index (ring position)
    pub write_idx: usize,
    /// Total snapshots stored (capped at `n_slots`)
    pub total_stored: usize,
    /// Number of particles per snapshot
    pub n_particles: usize,
    /// Stride per snapshot (`n_particles` * 3)
    pub stride: usize,
}

impl GpuVelocityRing {
    /// Create a velocity ring buffer for the given particle count and slot count.
    #[must_use]
    pub fn new(device: &WgpuDevice, n_particles: usize, n_slots: usize) -> Self {
        let stride = n_particles * 3;
        let total_f64 = n_slots * stride;
        let total_bytes = (total_f64 * 8) as u64;

        let flat_buf = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("vel_ring_flat"),
            size: total_bytes.max(8),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        Self {
            flat_buf,
            n_slots,
            write_idx: 0,
            total_stored: 0,
            n_particles,
            stride,
        }
    }

    /// Copy a velocity snapshot from `src_buf` into the next ring slot.
    ///
    /// `src_buf` must hold `[N×3]` f64 values (the current velocity array).
    pub fn push_snapshot(&mut self, encoder: &mut wgpu::CommandEncoder, src_buf: &wgpu::Buffer) {
        let offset_bytes = (self.write_idx * self.stride * 8) as u64;
        let size_bytes = (self.stride * 8) as u64;

        encoder.copy_buffer_to_buffer(src_buf, 0, &self.flat_buf, offset_bytes, size_bytes);

        self.write_idx = (self.write_idx + 1) % self.n_slots;
        self.total_stored = (self.total_stored + 1).min(self.n_slots);
    }

    /// Number of snapshots currently stored.
    #[must_use]
    pub fn stored(&self) -> usize {
        self.total_stored
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool;

    #[test]
    fn test_velocity_ring_creation() {
        let Some(device) = test_pool::get_test_device_if_f64_gpu_available_sync() else {
            return;
        };
        let ring = GpuVelocityRing::new(&device, 100, 500);
        assert_eq!(ring.n_slots, 500);
        assert_eq!(ring.stride, 300);
        assert_eq!(ring.stored(), 0);
    }

    #[test]
    fn test_vacf_batch_pipeline_creation() {
        let Some(device) = test_pool::get_test_device_if_f64_gpu_available_sync() else {
            return;
        };
        let _pipeline = VacfBatchGpu::new(device);
    }

    #[test]
    fn test_stress_virial_pipeline_creation() {
        let Some(device) = test_pool::get_test_device_if_f64_gpu_available_sync() else {
            return;
        };
        let _pipeline = StressVirialGpu::new(device);
    }
}

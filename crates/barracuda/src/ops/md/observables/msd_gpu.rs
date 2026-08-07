// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU-accelerated Mean-Squared Displacement (f64).
//!
//! Dispatches `msd_f64.wgsl` for each lag value, reducing the per-pair
//! squared displacements on the host to produce MSD(τ).
//!
//! Positions must be PBC-unwrapped before calling.

use crate::device::WgpuDevice;
use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::Result;
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;

const SHADER: &str = include_str!("msd_f64.wgsl");

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct MsdParams {
    n_particles: u32,
    n_frames: u32,
    lag: u32,
    _pad0: u32,
}

/// GPU MSD calculator.
pub struct MsdGpu {
    device: Arc<WgpuDevice>,
}

impl MsdGpu {
    /// Creates a new GPU MSD calculator for the given WGPU device.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn new(device: Arc<WgpuDevice>) -> Result<Self> {
        Ok(Self { device })
    }

    /// Compute MSD for a range of lags.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    ///
    /// * `unwrapped_positions` — contiguous `[n_frames * n * 3]` f64 (PBC-unwrapped)
    /// * `n` — number of particles
    /// * `n_frames` — number of snapshots
    /// * `max_lag` — maximum lag in snapshot intervals
    /// * `dt` — time between snapshots
    ///
    /// Returns `(t_values, msd_values)` where `msd_values[i]` is MSD at lag `i+1`.
    pub fn compute(
        &self,
        unwrapped_positions: &[f64],
        n: usize,
        n_frames: usize,
        max_lag: usize,
        dt: f64,
    ) -> Result<(Vec<f64>, Vec<f64>)> {
        let actual_max_lag = max_lag.min(n_frames - 1);
        let d = &self.device.device;
        let q = &self.device.queue;

        let pos_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("MSD:pos"),
            contents: bytemuck::cast_slice(unwrapped_positions),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let mut t_values = Vec::with_capacity(actual_max_lag);
        let mut msd_values = Vec::with_capacity(actual_max_lag);

        for lag in 1..=actual_max_lag {
            let n_origins = n_frames - lag;
            let total = n_origins * n;

            let out_size = (total * 8) as u64;
            let out_buf = d.create_buffer(&wgpu::BufferDescriptor {
                label: Some("MSD:out"),
                size: out_size.max(8),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });

            let params = MsdParams {
                n_particles: n as u32,
                n_frames: n_frames as u32,
                lag: lag as u32,
                _pad0: 0,
            };
            let params_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("MSD:params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

            ComputeDispatch::new(&self.device, "MSD")
                .shader(SHADER, "main")
                .f64()
                .storage_read(0, &pos_buf)
                .storage_rw(1, &out_buf)
                .uniform(2, &params_buf)
                .dispatch_1d(total as u32)
                .submit()?;

            let rb = d.create_buffer(&wgpu::BufferDescriptor {
                label: Some("MSD:rb"),
                size: out_size.max(8),
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let mut enc = self.device.create_encoder_guarded(&Default::default());
            enc.copy_buffer_to_buffer(&out_buf, 0, &rb, 0, out_size.max(8));
            q.submit(Some(enc.finish()));

            let vals = self.device.map_staging_buffer::<f64>(&rb, total)?;
            let msd: f64 = vals.iter().sum::<f64>() / total as f64;

            t_values.push(lag as f64 * dt);
            msd_values.push(msd);
        }

        Ok((t_values, msd_values))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_device() -> Option<Arc<WgpuDevice>> {
        crate::device::test_pool::get_test_device_if_f64_gpu_available_sync()
    }

    #[test]
    fn test_msd_stationary_particles() -> Result<()> {
        let Some(device) = test_device() else {
            return Ok(());
        };
        let msd = MsdGpu::new(device)?;

        // 2 particles, 4 frames, stationary → MSD = 0
        let positions = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, // frame 0
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, // frame 1
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, // frame 2
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, // frame 3
        ];

        let (t, m) = msd.compute(&positions, 2, 4, 3, 0.1)?;
        assert_eq!(t.len(), 3);
        for val in &m {
            assert!(
                val.abs() < 1e-12,
                "MSD should be 0 for stationary particles"
            );
        }
        Ok(())
    }

    #[test]
    fn test_msd_linear_motion() -> Result<()> {
        let Some(device) = test_device() else {
            return Ok(());
        };
        let msd = MsdGpu::new(device)?;

        // 1 particle moving at constant velocity: v = (1, 0, 0)
        // dt = 1.0, so at frame t, x = t
        let positions = vec![
            0.0, 0.0, 0.0, // frame 0
            1.0, 0.0, 0.0, // frame 1
            2.0, 0.0, 0.0, // frame 2
            3.0, 0.0, 0.0, // frame 3
        ];

        let (t, m) = msd.compute(&positions, 1, 4, 3, 1.0)?;

        // MSD(lag=1) = avg of (1-0)², (2-1)², (3-2)² = 1.0
        assert!((m[0] - 1.0).abs() < 1e-10, "MSD(1) = {}", m[0]);
        // MSD(lag=2) = avg of (2-0)², (3-1)² = (4+4)/2 = 4.0
        assert!((m[1] - 4.0).abs() < 1e-10, "MSD(2) = {}", m[1]);
        // MSD(lag=3) = (3-0)² = 9.0
        assert!((m[2] - 9.0).abs() < 1e-10, "MSD(3) = {}", m[2]);

        assert!((t[0] - 1.0).abs() < 1e-10);
        assert!((t[1] - 2.0).abs() < 1e-10);
        assert!((t[2] - 3.0).abs() < 1e-10);
        Ok(())
    }
}

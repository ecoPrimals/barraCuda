// SPDX-License-Identifier: AGPL-3.0-or-later
//! RDF F64 - Radial Distribution Function histogram - f64 precision WGSL
//!
//! GPU-accelerated O(N²) pair-distance histogram with PBC.
//! Uses `atomic<u32>` bins in WGSL for race-free accumulation.
//!
//! Applications:
//! - Structure analysis in MD
//! - Phase identification (solid/liquid/gas)
//! - Validation against experiment/theory

use crate::device::compute_pipeline::ComputeDispatch;
use crate::device::WgpuDevice;
use crate::error::Result;
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;

const SHADER: &str = include_str!("rdf_histogram_f64.wgsl");
const WG: u32 = 64;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct RdfParams {
    n_particles: u32,
    n_bins: u32,
    dr: f64,
    box_x: f64,
    box_y: f64,
    box_z: f64,
}

/// GPU-accelerated RDF histogram calculator (f64 positions, u32 bins).
pub struct RdfHistogramF64 {
    device: Arc<WgpuDevice>,
}

impl RdfHistogramF64 {
    pub fn new(device: Arc<WgpuDevice>) -> Result<Self> {
        Ok(Self { device })
    }

    /// WGSL kernel for RDF histogram (f32 variant, for GPUs without f64).
    pub const WGSL_RDF_HISTOGRAM_F32: &str = include_str!("rdf_histogram.wgsl");

    /// Compute RDF histogram on GPU.
    ///
    /// * `positions` — `[N*3]` f64 particle positions
    /// * `n_bins`    — number of histogram bins
    /// * `r_max`     — maximum radius
    /// * `box_size`  — `[Lx, Ly, Lz]` (PBC)
    pub fn histogram(
        &self,
        positions: &[f64],
        n_bins: usize,
        r_max: f64,
        box_size: [f64; 3],
    ) -> Result<Vec<u32>> {
        let n = positions.len() / 3;
        let dr = r_max / n_bins as f64;

        let params_data = RdfParams {
            n_particles: n as u32,
            n_bins: n_bins as u32,
            dr,
            box_x: box_size[0],
            box_y: box_size[1],
            box_z: box_size[2],
        };

        let d = &self.device.device;

        let pos_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("RDF:pos"),
            contents: bytemuck::cast_slice(positions),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let hist_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("RDF:hist"),
            contents: bytemuck::cast_slice(&vec![0u32; n_bins]),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        });

        let params_buf = d.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("RDF:params"),
            contents: bytemuck::bytes_of(&params_data),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let wg_count = (n as u32).div_ceil(WG);
        ComputeDispatch::new(&self.device, "rdf_histogram_f64")
            .shader(SHADER, "main")
            .f64()
            .storage_read(0, &pos_buf)
            .storage_rw(1, &hist_buf)
            .uniform(2, &params_buf)
            .dispatch(wg_count, 1, 1)
            .submit();

        let readback = d.create_buffer(&wgpu::BufferDescriptor {
            label: Some("RDF:readback"),
            size: (n_bins * 4) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut enc = self
            .device
            .create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
                label: Some("RDF:enc"),
            });
        enc.copy_buffer_to_buffer(&hist_buf, 0, &readback, 0, (n_bins * 4) as u64);
        self.device.submit_and_poll(Some(enc.finish()));

        self.device.map_staging_buffer::<u32>(&readback, n_bins)
    }

    /// Compute normalized g(r) on GPU.
    ///
    /// g(r) = histogram / (N * ρ * V_shell)
    /// where V_shell = 4π/3 * ((r+dr)³ - r³)
    pub fn compute_gr(
        &self,
        positions: &[f64],
        n_bins: usize,
        r_max: f64,
        box_size: [f64; 3],
    ) -> Result<(Vec<f64>, Vec<f64>)> {
        let hist = self.histogram(positions, n_bins, r_max, box_size)?;
        let dr = r_max / n_bins as f64;
        let n = positions.len() / 3;
        let volume = box_size[0] * box_size[1] * box_size[2];
        let density = n as f64 / volume;

        let mut r = Vec::with_capacity(n_bins);
        let mut gr = Vec::with_capacity(n_bins);

        for i in 0..n_bins {
            let r_lo = i as f64 * dr;
            let r_hi = (i + 1) as f64 * dr;
            let r_mid = (r_lo + r_hi) / 2.0;
            let v_shell = 4.0 / 3.0 * std::f64::consts::PI * (r_hi.powi(3) - r_lo.powi(3));
            let expected = density * v_shell * (n - 1) as f64 / 2.0;

            r.push(r_mid);
            gr.push(if expected > 0.0 {
                hist[i] as f64 / expected
            } else {
                0.0
            });
        }

        Ok((r, gr))
    }
}

#[expect(clippy::unwrap_used, reason = "tests")]
#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_device() -> Option<Arc<crate::device::WgpuDevice>> {
        crate::device::test_pool::get_test_device_if_f64_gpu_available_sync()
    }

    #[test]
    fn test_two_particles() -> Result<()> {
        let Some(device) = create_test_device() else {
            return Ok(());
        };
        let rdf = RdfHistogramF64::new(device)?;

        // Two particles at distance 1.0
        let positions = vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let box_size = [10.0, 10.0, 10.0];

        let hist = rdf.histogram(&positions, 10, 5.0, box_size)?;

        // dr = 0.5, distance 1.0 falls in bin 2 (0.5-1.0... wait, 1.0/0.5 = 2, so bin 2)
        assert_eq!(hist[2], 1, "Pair at r=1.0 should be in bin 2");

        Ok(())
    }

    #[test]
    fn test_pbc() -> Result<()> {
        let Some(device) = create_test_device() else {
            return Ok(());
        };
        let rdf = RdfHistogramF64::new(device)?;

        // Two particles at opposite corners - should see minimum image
        let positions = vec![0.5, 0.5, 0.5, 9.5, 9.5, 9.5];
        let box_size = [10.0, 10.0, 10.0];

        let hist = rdf.histogram(&positions, 10, 5.0, box_size)?;

        // Real distance = sqrt((9-0.5)² * 3) ≈ 15.6 (no PBC)
        // PBC distance = sqrt((1.0)² * 3) ≈ 1.73
        // With dr = 0.5, bin = 3 (1.5-2.0)
        let total: u32 = hist.iter().sum();
        assert_eq!(total, 1, "Should have exactly one pair");

        // The pair should be in a small-r bin due to PBC
        let small_r_counts: u32 = hist[0..5].iter().sum();
        assert_eq!(small_r_counts, 1, "Pair should be at small r due to PBC");

        Ok(())
    }

    #[test]
    fn test_gr_ideal_gas() -> Result<()> {
        let Some(device) = create_test_device() else {
            return Ok(());
        };
        let rdf = RdfHistogramF64::new(device)?;

        // Random positions (approximating ideal gas)
        let n = 100;
        let box_size = [10.0, 10.0, 10.0];
        let mut positions = Vec::with_capacity(n * 3);

        // Simple pseudo-random placement
        for i in 0..n {
            positions.push((i as f64 * 0.97) % box_size[0]);
            positions.push((i as f64 * 1.13) % box_size[1]);
            positions.push((i as f64 * 0.89) % box_size[2]);
        }

        let (r, gr) = rdf.compute_gr(&positions, 20, 4.0, box_size)?;

        assert_eq!(r.len(), 20);
        assert_eq!(gr.len(), 20);

        // g(r) should be roughly 1 for an ideal gas at large r
        // (This is approximate due to small N)
        let avg_gr: f64 = gr[10..].iter().sum::<f64>() / 10.0;
        assert!(
            avg_gr > 0.5 && avg_gr < 2.0,
            "g(r) ~ 1 expected for random distribution, got {}",
            avg_gr
        );

        Ok(())
    }
}

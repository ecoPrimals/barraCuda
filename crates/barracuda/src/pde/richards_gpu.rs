// SPDX-License-Identifier: AGPL-3.0-only

//! GPU-accelerated Richards PDE solver using Picard iteration.
//!
//! Dispatches three WGSL kernels per Picard iteration:
//! 1. `compute_hydraulics` — K(h), C(h), θ(h) per node (parallel)
//! 2. `assemble_tridiag` — Build Crank-Nicolson tridiagonal system (parallel)
//! 3. `thomas_solve` — Thomas algorithm for tridiagonal solve (sequential)
//!
//! Convergence is checked on CPU after each Picard iteration by reading
//! back `h_new` and comparing to h. When converged, h ← `h_new` and the
//! next time step begins.
//!
//! Provenance: airSpring V045 → toadStool absorption

use super::richards::{RichardsBc, RichardsConfig, RichardsResult};
use crate::device::WgpuDevice;
use crate::device::capabilities::WORKGROUP_SIZE_COMPACT;
use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::{BarracudaError, Result};
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;

const SHADER: &str = include_str!("../shaders/science/richards_picard_f64.wgsl");

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct RichardsGpuParams {
    n_nodes: u32,
    bc_top: u32,
    bc_bottom: u32,
    _pad: u32,
    dz: f64,
    dt: f64,
    theta_s: f64,
    theta_r: f64,
    alpha: f64,
    vg_n: f64,
    k_sat: f64,
    bc_top_val: f64,
    bc_bottom_val: f64,
    _padf: f64,
}

fn bc_type(bc: &RichardsBc) -> u32 {
    match bc {
        RichardsBc::PressureHead(_) => 0,
        RichardsBc::Flux(_) => 1,
    }
}

fn bc_val(bc: &RichardsBc) -> f64 {
    match bc {
        RichardsBc::PressureHead(v) | RichardsBc::Flux(v) => *v,
    }
}

/// GPU-accelerated Richards PDE solver.
///
/// Uses Picard iteration with Crank-Nicolson time discretization.
/// The tridiagonal solve runs on a single GPU workgroup (Thomas algorithm
/// is inherently sequential), but the hydraulic property computation and
/// system assembly are fully parallel across nodes.
///
/// For small grids (< 1000 nodes), the CPU solver may be faster due to
/// GPU dispatch overhead. For large grids or batched solves, the GPU
/// solver amortizes overhead.
pub struct RichardsGpu {
    device: Arc<WgpuDevice>,
}

impl RichardsGpu {
    /// Create a GPU-accelerated Richards PDE solver.
    #[must_use]
    pub fn new(device: Arc<WgpuDevice>) -> Self {
        Self { device }
    }

    /// Solve the Richards equation for the given configuration and initial conditions.
    ///
    /// Returns the final pressure head `h`, volumetric water content `theta`, and iteration counts.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if config validation fails, h0 length mismatch, or if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn solve(
        &self,
        config: &RichardsConfig,
        h0: &[f64],
        n_steps: usize,
        top_bc: RichardsBc,
        bottom_bc: RichardsBc,
    ) -> Result<RichardsResult> {
        config.validate()?;
        let n = config.n_nodes;

        if h0.len() != n {
            return Err(BarracudaError::InvalidInput {
                message: format!("h0 length {} != n_nodes {}", h0.len(), n),
            });
        }

        let gpu_params = RichardsGpuParams {
            n_nodes: n as u32,
            bc_top: bc_type(&top_bc),
            bc_bottom: bc_type(&bottom_bc),
            _pad: 0,
            dz: config.dz,
            dt: config.dt,
            theta_s: config.soil.theta_s,
            theta_r: config.soil.theta_r,
            alpha: config.soil.alpha,
            vg_n: config.soil.n,
            k_sat: config.soil.k_sat,
            bc_top_val: bc_val(&top_bc),
            bc_bottom_val: bc_val(&bottom_bc),
            _padf: 0.0,
        };

        let params_buf = self
            .device
            .create_uniform_buffer("Richards params", &gpu_params);

        let buf_size = (n * 8) as u64;
        let create_rw = |label: &str| {
            self.device.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: buf_size,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let create_rw_iface = |label: &str| {
            self.device.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: ((n - 1) * 8) as u64,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };

        let h_bytes: &[u8] = bytemuck::cast_slice(h0);
        let h_buf = self
            .device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Richards h"),
                contents: h_bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            });
        let h_old_buf = self
            .device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Richards h_old"),
                contents: h_bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            });

        let k_buf = create_rw("Richards K");
        let c_buf = create_rw("Richards C");
        let theta_buf = create_rw("Richards theta");
        let k_half_buf = create_rw_iface("Richards k_half");
        let a_buf = create_rw("Richards a");
        let b_buf = create_rw("Richards b");
        let c_tri_buf = create_rw("Richards c_tri");
        let d_buf = create_rw("Richards d");
        let h_new_buf = create_rw("Richards h_new");

        let wg = n.div_ceil(WORKGROUP_SIZE_COMPACT as usize) as u32;
        let mut total_picard = 0usize;

        for _step in 0..n_steps {
            // Copy h → h_old
            let mut enc = self
                .device
                .create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
                    label: Some("Richards copy h→h_old"),
                });
            enc.copy_buffer_to_buffer(&h_buf, 0, &h_old_buf, 0, buf_size);
            self.device.submit_commands(Some(enc.finish()));

            for _picard in 0..config.max_picard_iter {
                total_picard += 1;

                ComputeDispatch::new(self.device.as_ref(), "richards_hydraulics")
                    .shader(SHADER, "compute_hydraulics")
                    .f64()
                    .uniform(0, &params_buf)
                    .storage_read(1, &h_buf)
                    .storage_read(2, &h_old_buf)
                    .storage_rw(3, &k_buf)
                    .storage_rw(4, &c_buf)
                    .storage_rw(5, &theta_buf)
                    .storage_rw(6, &k_half_buf)
                    .storage_rw(7, &a_buf)
                    .storage_rw(8, &b_buf)
                    .storage_rw(9, &c_tri_buf)
                    .storage_rw(10, &d_buf)
                    .storage_rw(11, &h_new_buf)
                    .dispatch(wg, 1, 1)
                    .submit()?;

                ComputeDispatch::new(self.device.as_ref(), "richards_tridiag")
                    .shader(SHADER, "assemble_tridiag")
                    .f64()
                    .uniform(0, &params_buf)
                    .storage_read(1, &h_buf)
                    .storage_read(2, &h_old_buf)
                    .storage_rw(3, &k_buf)
                    .storage_rw(4, &c_buf)
                    .storage_rw(5, &theta_buf)
                    .storage_rw(6, &k_half_buf)
                    .storage_rw(7, &a_buf)
                    .storage_rw(8, &b_buf)
                    .storage_rw(9, &c_tri_buf)
                    .storage_rw(10, &d_buf)
                    .storage_rw(11, &h_new_buf)
                    .dispatch(wg, 1, 1)
                    .submit()?;

                ComputeDispatch::new(self.device.as_ref(), "richards_thomas")
                    .shader(SHADER, "thomas_solve")
                    .f64()
                    .uniform(0, &params_buf)
                    .storage_read(1, &h_buf)
                    .storage_read(2, &h_old_buf)
                    .storage_rw(3, &k_buf)
                    .storage_rw(4, &c_buf)
                    .storage_rw(5, &theta_buf)
                    .storage_rw(6, &k_half_buf)
                    .storage_rw(7, &a_buf)
                    .storage_rw(8, &b_buf)
                    .storage_rw(9, &c_tri_buf)
                    .storage_rw(10, &d_buf)
                    .storage_rw(11, &h_new_buf)
                    .dispatch(1, 1, 1)
                    .submit()?;

                let h_current = self.device.read_f64_buffer(&h_buf, n)?;
                let h_updated = self.device.read_f64_buffer(&h_new_buf, n)?;

                let max_diff = h_current
                    .iter()
                    .zip(h_updated.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f64, f64::max);

                // h ← h_new
                let new_bytes: &[u8] = bytemuck::cast_slice(&h_updated);
                self.device.queue.write_buffer(&h_buf, 0, new_bytes);

                if max_diff < config.picard_tol {
                    break;
                }
            }
        }

        let h_final = self.device.read_f64_buffer(&h_buf, n)?;
        let theta: Vec<f64> = h_final.iter().map(|&hi| config.soil.theta(hi)).collect();

        Ok(RichardsResult {
            h: h_final,
            theta,
            total_picard_iterations: total_picard,
            time_steps_completed: n_steps,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_gpu_device;
    use crate::pde::richards::SoilParams;

    #[tokio::test]
    async fn test_richards_gpu_steady_state() {
        let Some(device) = get_test_gpu_device().await else {
            return;
        };
        let solver = RichardsGpu::new(device);
        let n = 20;
        let h0 = vec![-50.0; n];
        let config = RichardsConfig {
            soil: SoilParams::SANDY_LOAM,
            dz: 5.0,
            dt: 60.0,
            n_nodes: n,
            max_picard_iter: 20,
            picard_tol: 1e-4,
        };

        let result = solver
            .solve(
                &config,
                &h0,
                3,
                RichardsBc::PressureHead(-50.0),
                RichardsBc::PressureHead(-50.0),
            )
            .unwrap();

        assert_eq!(result.h.len(), n, "output must have n_nodes elements");
        assert!(
            result.time_steps_completed > 0,
            "must complete at least 1 step"
        );
        for &hi in &result.h {
            assert!(hi.is_finite(), "GPU Richards h must be finite, got {hi}");
        }
    }
}

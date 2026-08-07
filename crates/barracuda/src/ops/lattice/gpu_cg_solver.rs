// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU Conjugate Gradient solver for D†D on staggered fermion fields.
//!
//! Orchestrates the existing [`StaggeredDirac`] and CG vector kernels
//! (`complex_dot_re`, axpy, xpay) with [`ReduceScalarPipeline`] reductions
//! in a host-side loop. All math runs on GPU — no CPU fallback.
//!
//! # Algorithm
//!
//! Solves `(D†D) x = b` where D is the staggered Dirac operator:
//!
//! 1. r = b - D†D·x  (initial residual, x=0 → r=b)
//! 2. p = r
//! 3. Loop:
//!    a. Ap = D†D·p  (two Dirac dispatches: D·p → tmp, D†·tmp → Ap)
//!    b. rr = Re<r|r>  (`complex_dot_re` + reduce)
//!    c. pAp = Re<p|Ap>  (`complex_dot_re` + reduce)
//!    d. α = rr / pAp
//!    e. x += α·p  (axpy)
//!    f. r -= α·Ap  (axpy with -α)
//!    g. `new_rr` = Re<r|r>
//!    h. β = `new_rr` / rr
//!    i. p = r + β·p  (xpay)
//!    j. Check convergence: `new_rr` < tol² × `b_norm²`

use crate::device::compute_pipeline::ComputeDispatch;
use crate::device::WgpuDevice;
use crate::device::capabilities::WORKGROUP_SIZE_COMPACT;
use crate::error::Result;
use crate::pipeline::ReduceScalarPipeline;
use std::sync::Arc;

use super::dirac::StaggeredDirac;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct DotParams {
    n_pairs: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct AxpyParams {
    n: u32,
    pad0: u32,
    alpha: f64,
}

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct XpayParams {
    n: u32,
    pad0: u32,
    beta: f64,
}

/// Lattice geometry buffers for the Dirac operator (links, neighbors, phases).
pub struct CgLatticeBuffers<'a> {
    /// Gauge link buffer.
    pub links: &'a wgpu::Buffer,
    /// Neighbor index buffer.
    pub nbr: &'a wgpu::Buffer,
    /// Staggered phase buffer.
    pub phases: &'a wgpu::Buffer,
}

/// Solver convergence parameters.
#[derive(Clone, Debug)]
pub struct CgSolverConfig {
    /// Fermion mass.
    pub mass: f64,
    /// Relative residual tolerance.
    pub tol: f64,
    /// Maximum CG iterations.
    pub max_iter: usize,
}

/// Result of a GPU CG solve.
#[derive(Clone, Debug)]
pub struct GpuCgResult {
    /// Whether the solver converged
    pub converged: bool,
    /// Number of iterations performed
    pub iterations: usize,
    /// Final residual squared
    pub residual_sq: f64,
}

/// GPU-resident buffers for the CG solver workspace.
pub struct GpuCgBuffers {
    /// Solution vector
    pub x: wgpu::Buffer,
    /// Residual vector
    pub r: wgpu::Buffer,
    /// Search direction
    pub p: wgpu::Buffer,
    /// A·p workspace
    pub ap: wgpu::Buffer,
    /// Temporary for Dirac application
    pub tmp: wgpu::Buffer,
    /// Dot product reduction output
    pub dot_out: wgpu::Buffer,
}

impl GpuCgBuffers {
    /// Create GPU buffers for the given lattice volume.
    #[must_use]
    pub fn new(device: &WgpuDevice, volume: usize) -> Self {
        let field_bytes = (volume * 6 * std::mem::size_of::<f64>()) as u64;
        let dot_bytes = (volume * 3 * std::mem::size_of::<f64>()) as u64;
        let make_field = |label: &str| {
            device.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: field_bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        Self {
            x: make_field("cg:x"),
            r: make_field("cg:r"),
            p: make_field("cg:p"),
            ap: make_field("cg:ap"),
            tmp: make_field("cg:tmp"),
            dot_out: device.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("cg:dot_out"),
                size: dot_bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
        }
    }
}

/// GPU CG solver orchestrating Dirac + CG vector ops + reduction.
pub struct GpuCgSolver {
    device: Arc<WgpuDevice>,
    volume: u32,
    n_f64: u32,
    n_pairs: u32,
    dirac: StaggeredDirac,
    reducer: ReduceScalarPipeline,
}

impl GpuCgSolver {
    /// Create a GPU CG solver for the given lattice volume.
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn new(device: Arc<WgpuDevice>, volume: u32) -> Result<Self> {
        let dirac = StaggeredDirac::new(device.clone(), volume)?;
        let n_f64 = volume * 6;
        let n_pairs = volume * 3;

        let reducer = ReduceScalarPipeline::new(device.clone(), n_pairs as usize)?;

        Ok(Self {
            device,
            volume,
            n_f64,
            n_pairs,
            dirac,
            reducer,
        })
    }

    /// Solve (D†D)x = b on GPU.
    /// All buffers must be GPU-resident. `x` is zeroed at start.
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn solve(
        &self,
        b_buf: &wgpu::Buffer,
        bufs: &GpuCgBuffers,
        lattice: &CgLatticeBuffers<'_>,
        config: &CgSolverConfig,
    ) -> Result<GpuCgResult> {
        let n = self.n_f64 as usize;
        let n_bytes = (n * std::mem::size_of::<f64>()) as u64;

        self.device
            .queue
            .write_buffer(&bufs.x, 0, &vec![0u8; n_bytes as usize]);

        self.copy_buffer(b_buf, &bufs.r, n_bytes);
        self.copy_buffer(&bufs.r, &bufs.p, n_bytes);

        let b_norm_sq = self.complex_dot_re(b_buf, b_buf, &bufs.dot_out)?;
        if b_norm_sq < 1e-30 {
            return Ok(GpuCgResult {
                converged: true,
                iterations: 0,
                residual_sq: 0.0,
            });
        }
        let tol_sq = config.tol * config.tol * b_norm_sq;

        let mut rr = b_norm_sq;

        for iter in 0..config.max_iter {
            self.dirac.dispatch(
                config.mass,
                1.0,
                lattice.links,
                &bufs.p,
                &bufs.tmp,
                lattice.nbr,
                lattice.phases,
            )?;
            self.dirac.dispatch(
                config.mass,
                -1.0,
                lattice.links,
                &bufs.tmp,
                &bufs.ap,
                lattice.nbr,
                lattice.phases,
            )?;

            let p_ap = self.complex_dot_re(&bufs.p, &bufs.ap, &bufs.dot_out)?;

            if p_ap.abs() < 1e-30 {
                return Ok(GpuCgResult {
                    converged: false,
                    iterations: iter,
                    residual_sq: rr,
                });
            }
            let alpha = rr / p_ap;

            self.axpy(alpha, &bufs.p, &bufs.x)?;
            self.axpy(-alpha, &bufs.ap, &bufs.r)?;

            let new_rr = self.complex_dot_re(&bufs.r, &bufs.r, &bufs.dot_out)?;

            if new_rr < tol_sq {
                return Ok(GpuCgResult {
                    converged: true,
                    iterations: iter + 1,
                    residual_sq: new_rr,
                });
            }

            let beta = new_rr / rr;
            rr = new_rr;

            self.xpay(&bufs.r, beta, &bufs.p)?;
        }

        Ok(GpuCgResult {
            converged: false,
            iterations: config.max_iter,
            residual_sq: rr,
        })
    }

    fn copy_buffer(&self, src: &wgpu::Buffer, dst: &wgpu::Buffer, size: u64) {
        let mut enc = self.device.create_encoder_guarded(&Default::default());
        enc.copy_buffer_to_buffer(src, 0, dst, 0, size);
        self.device.submit_commands(Some(enc.finish()));
    }

    fn complex_dot_re(
        &self,
        a: &wgpu::Buffer,
        b: &wgpu::Buffer,
        out: &wgpu::Buffer,
    ) -> Result<f64> {
        let params_data = DotParams {
            n_pairs: self.n_pairs,
            pad0: 0,
            pad1: 0,
            pad2: 0,
        };
        let params = self.device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cg_dot:params"),
            size: std::mem::size_of::<DotParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.device
            .queue
            .write_buffer(&params, 0, bytemuck::bytes_of(&params_data));

        ComputeDispatch::new(&self.device, "cg_dot")
            .shader(super::cg::WGSL_COMPLEX_DOT_RE_F64, "main")
            .f64()
            .uniform(0, &params)
            .storage_read(1, a)
            .storage_read(2, b)
            .storage_rw(3, out)
            .dispatch(self.n_pairs.div_ceil(WORKGROUP_SIZE_COMPACT), 1, 1)
            .submit()?;

        self.reducer.sum_f64(out)
    }

    fn axpy(&self, alpha: f64, x: &wgpu::Buffer, y: &wgpu::Buffer) -> Result<()> {
        let params_data = AxpyParams {
            n: self.n_f64,
            pad0: 0,
            alpha,
        };
        let params = self.device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cg_axpy:params"),
            size: std::mem::size_of::<AxpyParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.device
            .queue
            .write_buffer(&params, 0, bytemuck::bytes_of(&params_data));

        ComputeDispatch::new(&self.device, "cg_axpy")
            .shader(super::cg::WGSL_AXPY_F64, "main")
            .f64()
            .uniform(0, &params)
            .storage_read(1, x)
            .storage_rw(2, y)
            .dispatch(self.n_f64.div_ceil(WORKGROUP_SIZE_COMPACT), 1, 1)
            .submit()?;

        Ok(())
    }

    fn xpay(&self, x: &wgpu::Buffer, beta: f64, p: &wgpu::Buffer) -> Result<()> {
        let params_data = XpayParams {
            n: self.n_f64,
            pad0: 0,
            beta,
        };
        let params = self.device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("cg_xpay:params"),
            size: std::mem::size_of::<XpayParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.device
            .queue
            .write_buffer(&params, 0, bytemuck::bytes_of(&params_data));

        ComputeDispatch::new(&self.device, "cg_xpay")
            .shader(super::cg::WGSL_XPAY_F64, "main")
            .f64()
            .uniform(0, &params)
            .storage_read(1, x)
            .storage_rw(2, p)
            .dispatch(self.n_f64.div_ceil(WORKGROUP_SIZE_COMPACT), 1, 1)
            .submit()?;

        Ok(())
    }

    /// Lattice volume (number of sites).
    #[must_use]
    pub fn volume(&self) -> u32 {
        self.volume
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cg_solver_pipeline_creation() {
        let Some(device) = crate::device::test_pool::get_test_device_if_f64_gpu_available_sync()
        else {
            return;
        };
        let solver = GpuCgSolver::new(device, 16).unwrap();
        assert_eq!(solver.volume(), 16);
    }
}

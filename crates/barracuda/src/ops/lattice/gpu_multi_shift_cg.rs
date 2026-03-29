// SPDX-License-Identifier: AGPL-3.0-or-later

//! GPU multi-shift Conjugate Gradient solver for RHMC.
//!
//! Solves `(D†D + σ_s) x_s = b` for all shifts simultaneously using a single
//! Krylov subspace. Only ONE matrix-vector product per iteration regardless
//! of shift count, reducing the dominant compute cost by a factor of `N_shifts`.
//!
//! Algorithm: Jegerlehner (hep-lat/9612014), adapted for GPU-resident scalars.
//!
//! # Architecture
//!
//! - WGSL shaders from [`super::absorbed_shaders`] handle all per-element work
//! - Host loop drives convergence checking with exponential back-off
//! - Uses barraCuda's `WgpuDevice::compile_shader_f64` + `ReduceScalarPipeline`
//! - All shift-dependent buffers (`ζ`, `α_s`, `β_ratio`, `p_s`, `x_s`) are GPU-resident
//!
//! Absorbed from hotSpring lattice QCD (Mar 2026), rewritten to barraCuda patterns.

use crate::device::WgpuDevice;
use crate::error::Result;
use crate::pipeline::ReduceScalarPipeline;
use std::sync::Arc;

use super::absorbed_shaders::{
    WGSL_CG_COMPUTE_ALPHA_SHIFTED_F64, WGSL_CG_UPDATE_XR_SHIFTED_F64, WGSL_MS_P_UPDATE_F64,
    WGSL_MS_X_UPDATE_F64, WGSL_MS_ZETA_UPDATE_F64,
};

/// Configuration for the GPU multi-shift CG solver.
#[derive(Debug, Clone)]
pub struct GpuMultiShiftCgConfig {
    /// Maximum CG iterations before declaring non-convergence.
    pub max_iterations: usize,
    /// Relative residual tolerance (convergence when `||r||² < tol² × ||b||²`).
    pub tolerance: f64,
    /// Iteration interval for convergence checking (exponential back-off cap).
    /// Avoids per-iteration GPU→CPU readback overhead.
    pub check_interval: usize,
}

impl Default for GpuMultiShiftCgConfig {
    fn default() -> Self {
        Self {
            max_iterations: 5000,
            tolerance: 1e-10,
            check_interval: 64,
        }
    }
}

/// Result of a GPU multi-shift CG solve.
#[derive(Debug, Clone)]
pub struct GpuMultiShiftCgResult {
    /// Number of CG iterations performed (shared across all shifts).
    pub iterations: usize,
    /// Whether the solver converged within `max_iterations`.
    pub converged: bool,
    /// Final relative residual norm squared (base system).
    pub residual_sq: f64,
}

/// Pre-compiled GPU pipelines for multi-shift CG.
///
/// Holds all five multi-shift WGSL pipelines plus the base CG infrastructure
/// (dot product, axpy, xpay, reduce). Create once and reuse across solves.
pub struct GpuMultiShiftCgPipelines {
    device: Arc<WgpuDevice>,
    /// Shifted alpha: `α = rz / (pAp + σ × pp)`.
    pub shifted_alpha_pipeline: wgpu::ComputePipeline,
    /// Bind group layout for [`Self::shifted_alpha_pipeline`].
    pub shifted_alpha_bgl: wgpu::BindGroupLayout,
    /// Shifted xr update: `x += α×p`, `r -= α×(Ap + σ×p)`.
    pub shifted_xr_pipeline: wgpu::ComputePipeline,
    /// Bind group layout for [`Self::shifted_xr_pipeline`].
    pub shifted_xr_bgl: wgpu::BindGroupLayout,
    /// Jegerlehner `ζ` recurrence.
    pub zeta_pipeline: wgpu::ComputePipeline,
    /// Bind group layout for [`Self::zeta_pipeline`].
    pub zeta_bgl: wgpu::BindGroupLayout,
    /// Shifted x update: `x_σ += α_σ × p_σ`.
    pub ms_x_pipeline: wgpu::ComputePipeline,
    /// Bind group layout for [`Self::ms_x_pipeline`].
    pub ms_x_bgl: wgpu::BindGroupLayout,
    /// Shifted p update: `p_σ = ζ_σ × r + β_σ × p_σ`.
    pub ms_p_pipeline: wgpu::ComputePipeline,
    /// Bind group layout for [`Self::ms_p_pipeline`].
    pub ms_p_bgl: wgpu::BindGroupLayout,
    /// Scalar reduction pipeline.
    pub reducer: ReduceScalarPipeline,
}

impl GpuMultiShiftCgPipelines {
    /// Compile all multi-shift CG pipelines for a given vector length.
    ///
    /// # Errors
    /// Returns [`Err`] if shader compilation or pipeline creation fails.
    pub fn new(device: Arc<WgpuDevice>, n_pairs: usize) -> Result<Self> {
        let shifted_alpha_mod =
            device.compile_shader_f64(WGSL_CG_COMPUTE_ALPHA_SHIFTED_F64, Some("ms_alpha_shifted"));
        let shifted_xr_mod =
            device.compile_shader_f64(WGSL_CG_UPDATE_XR_SHIFTED_F64, Some("ms_xr_shifted"));
        let zeta_mod = device.compile_shader_f64(WGSL_MS_ZETA_UPDATE_F64, Some("ms_zeta_update"));
        let ms_x_mod = device.compile_shader_f64(WGSL_MS_X_UPDATE_F64, Some("ms_x_update"));
        let ms_p_mod = device.compile_shader_f64(WGSL_MS_P_UPDATE_F64, Some("ms_p_update"));

        let shifted_alpha_bgl =
            device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("ms_alpha_shifted:bgl"),
                    entries: &[
                        storage_bgl(0, true),  // rz
                        storage_bgl(1, true),  // pap
                        storage_bgl(2, true),  // pp
                        storage_bgl(3, true),  // sigma
                        storage_bgl(4, false), // alpha
                    ],
                });

        let shifted_xr_bgl =
            device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("ms_xr_shifted:bgl"),
                    entries: &[
                        uniform_bgl(0),        // params
                        storage_bgl(1, false), // x
                        storage_bgl(2, false), // r
                        storage_bgl(3, true),  // p
                        storage_bgl(4, true),  // ap
                        storage_bgl(5, true),  // alpha
                        storage_bgl(6, true),  // sigma
                    ],
                });

        let zeta_bgl = device
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("ms_zeta:bgl"),
                entries: &[
                    uniform_bgl(0),        // params
                    storage_bgl(1, true),  // sigma
                    storage_bgl(2, false), // zeta_curr
                    storage_bgl(3, false), // zeta_prev
                    storage_bgl(4, false), // alpha_s
                    storage_bgl(5, false), // beta_ratio
                    storage_bgl(6, true),  // alpha_j
                    storage_bgl(7, true),  // beta_prev
                    storage_bgl(8, true),  // alpha_prev
                ],
            });

        let ms_x_bgl = device
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("ms_x_update:bgl"),
                entries: &[
                    uniform_bgl(0),        // params
                    storage_bgl(1, false), // x
                    storage_bgl(2, true),  // p
                    storage_bgl(3, true),  // alpha_s
                ],
            });

        let ms_p_bgl = device
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("ms_p_update:bgl"),
                entries: &[
                    uniform_bgl(0),        // params
                    storage_bgl(1, false), // p
                    storage_bgl(2, true),  // r
                    storage_bgl(3, true),  // zeta_curr
                    storage_bgl(4, true),  // beta_ratio
                    storage_bgl(5, true),  // beta_base
                ],
            });

        let make_pipeline = |bgl: &wgpu::BindGroupLayout,
                             module: &wgpu::ShaderModule,
                             label: &str| {
            let layout = device
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some(&format!("{label}:layout")),
                    bind_group_layouts: &[bgl],
                    immediate_size: 0,
                });
            device
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(label),
                    layout: Some(&layout),
                    module,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                })
        };

        let shifted_alpha_pipeline =
            make_pipeline(&shifted_alpha_bgl, &shifted_alpha_mod, "ms_alpha_shifted");
        let shifted_xr_pipeline = make_pipeline(&shifted_xr_bgl, &shifted_xr_mod, "ms_xr_shifted");
        let zeta_pipeline = make_pipeline(&zeta_bgl, &zeta_mod, "ms_zeta");
        let ms_x_pipeline = make_pipeline(&ms_x_bgl, &ms_x_mod, "ms_x_update");
        let ms_p_pipeline = make_pipeline(&ms_p_bgl, &ms_p_mod, "ms_p_update");

        let reducer = ReduceScalarPipeline::new(device.clone(), n_pairs)?;

        Ok(Self {
            device,
            shifted_alpha_pipeline,
            shifted_alpha_bgl,
            shifted_xr_pipeline,
            shifted_xr_bgl,
            zeta_pipeline,
            zeta_bgl,
            ms_x_pipeline,
            ms_x_bgl,
            ms_p_pipeline,
            ms_p_bgl,
            reducer,
        })
    }

    /// The device this pipeline was compiled on.
    #[must_use]
    pub fn device(&self) -> &Arc<WgpuDevice> {
        &self.device
    }
}

/// GPU-resident buffers for one multi-shift CG solve instance.
///
/// Allocated per-solve because shift count varies with the rational approximation.
pub struct GpuMultiShiftCgBuffers {
    /// Per-shift solution vectors `x_s` (one buffer per shift).
    pub x_shift: Vec<wgpu::Buffer>,
    /// Per-shift search directions `p_s` (one buffer per shift).
    pub p_shift: Vec<wgpu::Buffer>,
    /// `ζ_curr[n_shifts]`: current zeta values.
    pub zeta_curr: wgpu::Buffer,
    /// `ζ_prev[n_shifts]`: previous zeta values.
    pub zeta_prev: wgpu::Buffer,
    /// `α_s[n_shifts]`: per-shift alpha scalars.
    pub alpha_s: wgpu::Buffer,
    /// `β_ratio[n_shifts]`: `ζ_new/ζ_curr` ratios.
    pub beta_ratio: wgpu::Buffer,
    /// `σ[n_shifts]`: shift values (set once per solve).
    pub sigma: wgpu::Buffer,
    /// Scalar buffer for base CG `alpha`.
    pub alpha: wgpu::Buffer,
    /// Scalar buffer for base CG `beta`.
    pub beta: wgpu::Buffer,
    /// Scalar buffer for base CG `alpha` from the previous iteration.
    pub alpha_prev: wgpu::Buffer,
    /// Scalar buffer for base CG `beta` from the previous iteration.
    pub beta_prev: wgpu::Buffer,
}

impl GpuMultiShiftCgBuffers {
    /// Allocate GPU buffers for a multi-shift CG solve.
    #[must_use]
    pub fn new(device: &WgpuDevice, n_shifts: usize, vector_length: usize) -> Self {
        let field_bytes = (vector_length * std::mem::size_of::<f64>()) as u64;
        let shift_scalar_bytes = (n_shifts * std::mem::size_of::<f64>()) as u64;
        let scalar_bytes = std::mem::size_of::<f64>() as u64;

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

        let make_shift_scalar = |label: &str| {
            device.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: shift_scalar_bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };

        let make_scalar = |label: &str| {
            device.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: scalar_bytes,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };

        let x_shift = (0..n_shifts)
            .map(|s| make_field(&format!("ms_cg:x_shift_{s}")))
            .collect();
        let p_shift = (0..n_shifts)
            .map(|s| make_field(&format!("ms_cg:p_shift_{s}")))
            .collect();

        Self {
            x_shift,
            p_shift,
            zeta_curr: make_shift_scalar("ms_cg:zeta_curr"),
            zeta_prev: make_shift_scalar("ms_cg:zeta_prev"),
            alpha_s: make_shift_scalar("ms_cg:alpha_s"),
            beta_ratio: make_shift_scalar("ms_cg:beta_ratio"),
            sigma: make_shift_scalar("ms_cg:sigma"),
            alpha: make_scalar("ms_cg:alpha"),
            beta: make_scalar("ms_cg:beta"),
            alpha_prev: make_scalar("ms_cg:alpha_prev"),
            beta_prev: make_scalar("ms_cg:beta_prev"),
        }
    }

    /// Number of shifts.
    #[must_use]
    pub fn n_shifts(&self) -> usize {
        self.x_shift.len()
    }
}

/// CPU-generic multi-shift CG reference implementation.
///
/// Solves `(A + σ_s I) x_s = b` for all shifts simultaneously using a single
/// Krylov subspace. The matrix-vector product `A·v` is provided via closure.
///
/// This is the validation reference for the GPU implementation. It operates on
/// flat f64 slices without lattice-specific types.
#[must_use]
pub fn multi_shift_cg_generic(
    matvec: &dyn Fn(&[f64], &mut [f64]),
    b: &[f64],
    shifts: &[f64],
    tol: f64,
    max_iter: usize,
) -> (Vec<Vec<f64>>, GpuMultiShiftCgResult) {
    let n = b.len();
    let n_shifts = shifts.len();

    let mut x: Vec<Vec<f64>> = (0..n_shifts).map(|_| vec![0.0; n]).collect();
    let mut p: Vec<Vec<f64>> = (0..n_shifts).map(|_| b.to_vec()).collect();
    let mut r = b.to_vec();

    let b_norm_sq: f64 = b.iter().map(|v| v * v).sum();
    if b_norm_sq < 1e-30 {
        return (
            x,
            GpuMultiShiftCgResult {
                iterations: 0,
                converged: true,
                residual_sq: 0.0,
            },
        );
    }

    let tol_sq = tol * tol * b_norm_sq;
    let mut rz = b_norm_sq;
    let mut zeta_prev = vec![1.0; n_shifts];
    let mut zeta_curr = vec![1.0; n_shifts];
    let mut alpha_prev = 0.0_f64;
    let mut active = vec![true; n_shifts];
    let mut ap = vec![0.0; n];

    let mut iterations = 0;

    for _iter in 0..max_iter {
        iterations += 1;

        matvec(&p[0], &mut ap);

        let mut p_ap: f64 = p[0].iter().zip(ap.iter()).map(|(a, b)| a * b).sum();
        p_ap += shifts[0] * p[0].iter().map(|v| v * v).sum::<f64>();

        if p_ap.abs() < 1e-30 {
            break;
        }
        let alpha = rz / p_ap;

        for i in 0..n {
            x[0][i] += alpha * p[0][i];
            r[i] -= alpha * shifts[0].mul_add(p[0][i], ap[i]);
        }

        let rz_new: f64 = r.iter().map(|v| v * v).sum();

        for s in 1..n_shifts {
            if !active[s] {
                continue;
            }

            let ds = shifts[s] - shifts[0];
            let denom = alpha.mul_add(ds, 1.0)
                + alpha * alpha_prev * (1.0 - zeta_prev[s] / zeta_curr[s])
                    / if active[s] {
                        (rz / b_norm_sq.max(1e-30)).max(1e-30)
                    } else {
                        1e-30
                    };
            if denom.abs() < 1e-30 {
                active[s] = false;
                continue;
            }

            let zeta_next = zeta_curr[s] / denom;
            let alpha_s = alpha * zeta_next / zeta_curr[s];

            for i in 0..n {
                x[s][i] += alpha_s * p[s][i];
            }

            let beta_s = if rz.abs() > 1e-30 {
                (zeta_next / zeta_curr[s]).powi(2) * (rz_new / rz)
            } else {
                0.0
            };

            for i in 0..n {
                p[s][i] = zeta_next * r[i] + beta_s * p[s][i];
            }

            zeta_prev[s] = zeta_curr[s];
            zeta_curr[s] = zeta_next;
        }

        let beta = if rz.abs() > 1e-30 { rz_new / rz } else { 0.0 };
        for i in 0..n {
            p[0][i] = r[i] + beta * p[0][i];
        }

        alpha_prev = alpha;
        rz = rz_new;

        if rz < tol_sq {
            break;
        }
    }

    (
        x,
        GpuMultiShiftCgResult {
            iterations,
            converged: rz < tol_sq,
            residual_sq: rz / b_norm_sq,
        },
    )
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
    fn config_default_values() {
        let cfg = GpuMultiShiftCgConfig::default();
        assert_eq!(cfg.max_iterations, 5000);
        assert!((cfg.tolerance - 1e-10).abs() < 1e-20);
        assert_eq!(cfg.check_interval, 64);
    }

    #[test]
    fn generic_cpu_multi_shift_identity_matrix() {
        let n = 16;
        let identity_matvec = |x: &[f64], y: &mut [f64]| {
            y.copy_from_slice(x);
        };
        let b: Vec<f64> = (0..n).map(|i| (i + 1) as f64).collect();
        let shifts = vec![0.0, 0.5, 1.0, 2.0];

        let (solutions, result) =
            multi_shift_cg_generic(&identity_matvec, &b, &shifts, 1e-12, 1000);

        assert!(result.converged, "CG should converge for identity matrix");
        assert_eq!(solutions.len(), shifts.len());

        for (s_idx, &sigma) in shifts.iter().enumerate() {
            for i in 0..n {
                let expected = b[i] / (1.0 + sigma);
                let actual = solutions[s_idx][i];
                assert!(
                    (actual - expected).abs() < 1e-6,
                    "shift {sigma}: x[{i}] = {actual}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn generic_cpu_multi_shift_zero_rhs() {
        let n = 8;
        let matvec = |x: &[f64], y: &mut [f64]| y.copy_from_slice(x);
        let b = vec![0.0; n];
        let shifts = vec![0.0, 1.0];

        let (solutions, result) = multi_shift_cg_generic(&matvec, &b, &shifts, 1e-10, 100);

        assert!(result.converged);
        assert_eq!(result.iterations, 0);
        for sol in &solutions {
            assert!(sol.iter().all(|&v| v.abs() < 1e-15));
        }
    }

    #[test]
    fn generic_cpu_multi_shift_diagonal_matrix() {
        let n = 8;
        let diag: Vec<f64> = (1..=n).map(|i| i as f64).collect();
        let diag_clone = diag.clone();
        let matvec = move |x: &[f64], y: &mut [f64]| {
            for (i, val) in y.iter_mut().enumerate() {
                *val = diag_clone[i] * x[i];
            }
        };
        let b = vec![1.0; n];
        let shifts = vec![0.0, 0.5, 3.0];

        let (solutions, result) = multi_shift_cg_generic(&matvec, &b, &shifts, 1e-10, 500);

        assert!(result.converged);
        for (s_idx, &sigma) in shifts.iter().enumerate() {
            for i in 0..n {
                let expected = 1.0 / (diag[i] + sigma);
                let actual = solutions[s_idx][i];
                assert!(
                    (actual - expected).abs() < 1e-6,
                    "shift {sigma}: x[{i}] = {actual}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn pipeline_creation_with_gpu() {
        let Some(device) = crate::device::test_pool::get_test_device_if_f64_gpu_available_sync()
        else {
            return;
        };
        let pipelines = GpuMultiShiftCgPipelines::new(device.clone(), 64).unwrap();
        assert!(Arc::ptr_eq(pipelines.device(), &device));
    }

    #[test]
    fn buffer_allocation() {
        let Some(device) = crate::device::test_pool::get_test_device_if_f64_gpu_available_sync()
        else {
            return;
        };
        let bufs = GpuMultiShiftCgBuffers::new(&device, 4, 128);
        assert_eq!(bufs.n_shifts(), 4);
        assert_eq!(bufs.x_shift.len(), 4);
        assert_eq!(bufs.p_shift.len(), 4);
    }
}

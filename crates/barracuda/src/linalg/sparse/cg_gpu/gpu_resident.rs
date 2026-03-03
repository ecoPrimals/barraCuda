// SPDX-License-Identifier: AGPL-3.0-or-later
//! GPU-resident Conjugate Gradient solver
//!
//! Scalar values (α, β, ρ) remain on GPU; residual read only every `check_interval` iterations.

use super::CgGpu;
use super::CgGpuResult;
use crate::device::WgpuDevice;
use crate::error::{BarracudaError, Result};
use crate::linalg::sparse::csr::CsrMatrix;
use crate::linalg::sparse::gpu_helpers::{
    cg_dispatch_pass, CgPipelineSet, SparseBindGroupLayouts, SparseBuffers,
};
use std::sync::Arc;
use wgpu::util::DeviceExt;

impl CgGpu {
    /// Solve Ax = b using GPU-resident Conjugate Gradient (reduced CPU sync)
    ///
    /// This is the **recommended method** for large systems.
    /// Scalar values (α, β, ρ) remain on GPU; residual is only read every `check_interval` iterations.
    ///
    /// # Arguments
    /// * `device` - WgpuDevice to execute on
    /// * `a` - Symmetric positive definite CSR matrix (f64)
    /// * `b` - Right-hand side vector (f64)
    /// * `tol` - Convergence tolerance
    /// * `max_iter` - Maximum iterations
    /// * `check_interval` - How often to read residual from GPU (default: 10)
    ///
    /// # Performance
    /// For a 1000×1000 matrix:
    /// - Original: ~100 GPU↔CPU syncs for 100 iterations
    /// - GPU-resident (check_interval=10): ~10 GPU↔CPU syncs
    pub fn solve_gpu_resident(
        device: Arc<WgpuDevice>,
        a: &CsrMatrix,
        b: &[f64],
        tol: f64,
        max_iter: usize,
        check_interval: usize,
    ) -> Result<CgGpuResult> {
        let n = a.n_rows;
        if a.n_cols != n {
            return Err(BarracudaError::InvalidInput {
                message: "CG requires square matrix".to_string(),
            });
        }
        if b.len() != n {
            return Err(BarracudaError::InvalidInput {
                message: format!("Vector length {} doesn't match matrix size {}", b.len(), n),
            });
        }

        let check_interval = check_interval.max(1);

        // Early exit for zero RHS
        let b_norm: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
        if b_norm < 1e-14 {
            return Ok(CgGpuResult {
                x: vec![0.0; n],
                iterations: 0,
                residual: 0.0,
                converged: true,
            });
        }

        // Create GPU buffers for CSR matrix
        let values_buffer = SparseBuffers::f64_from_slice(&device, "CG values", &a.values);
        let col_indices_buffer =
            SparseBuffers::u32_from_usize(&device, "CG col_idx", &a.col_indices);
        let row_ptrs_buffer = SparseBuffers::u32_from_usize(&device, "CG row_ptr", &a.row_ptr);

        // Create GPU buffers for vectors
        let x_buffer = SparseBuffers::f64_zeros(&device, "CG x", n);
        let r_buffer = SparseBuffers::f64_from_slice(&device, "CG r", b);
        let p_buffer = SparseBuffers::f64_from_slice(&device, "CG p", b);
        let ap_buffer = SparseBuffers::f64_zeros(&device, "CG Ap", n);

        // Scalar buffers (stay on GPU)
        let num_workgroups = n.div_ceil(256);
        let partial_sums_buffer = SparseBuffers::f64_zeros(&device, "CG partial", num_workgroups);
        let rz_buffer = SparseBuffers::f64_zeros(&device, "CG rz", 1);
        let rz_new_buffer = SparseBuffers::f64_zeros(&device, "CG rz_new", 1);
        let pap_buffer = SparseBuffers::f64_zeros(&device, "CG pAp", 1);
        let alpha_buffer = SparseBuffers::f64_zeros(&device, "CG alpha", 1);
        let beta_buffer = SparseBuffers::f64_zeros(&device, "CG beta", 1);

        // Compile shader modules
        let spmv_shader = device.compile_shader_f64(CgGpu::spmv_shader(), Some("CG SpMV"));
        let dot_reduce_shader =
            device.compile_shader_f64(CgGpu::dot_reduce_shader(), Some("CG Dot/Reduce"));
        let cg_kernels_shader =
            device.compile_shader_f64(CgGpu::cg_kernels_shader(), Some("CG Kernels"));

        // Create bind group layouts and pipelines
        let spmv_bgl = SparseBindGroupLayouts::spmv(&device);
        let dot_bgl = SparseBindGroupLayouts::dot(&device);
        let reduce_bgl = SparseBindGroupLayouts::reduce(&device);
        let update_xr_bgl = SparseBindGroupLayouts::cg_update_xr(&device);
        let update_p_bgl = SparseBindGroupLayouts::cg_update_p(&device);
        let compute_alpha_bgl = SparseBindGroupLayouts::compute_alpha(&device);
        let compute_beta_bgl = SparseBindGroupLayouts::compute_beta(&device);

        let pl = CgPipelineSet::new(
            &device,
            &spmv_shader,
            &dot_reduce_shader,
            &cg_kernels_shader,
            &spmv_bgl,
            &dot_bgl,
            &reduce_bgl,
            &update_xr_bgl,
            &update_p_bgl,
            &compute_alpha_bgl,
            &compute_beta_bgl,
        );

        // Create bind groups
        let spmv_params = [n as u32];
        let spmv_params_buf = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("SpMV params"),
                contents: bytemuck::cast_slice(&spmv_params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let spmv_bg = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SpMV BG"),
            layout: &spmv_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: values_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: col_indices_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: row_ptrs_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: p_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: ap_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: spmv_params_buf.as_entire_binding(),
                },
            ],
        });

        let dot_params = [n as u32];
        let dot_params_buf = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Dot params"),
                contents: bytemuck::cast_slice(&dot_params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        // rᵀr dot product bind group
        let rr_bg = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rr BG"),
            layout: &dot_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: r_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: r_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: partial_sums_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: dot_params_buf.as_entire_binding(),
                },
            ],
        });

        // pᵀAp dot product bind group
        let pap_bg = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("pAp BG"),
            layout: &dot_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: p_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: ap_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: partial_sums_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: dot_params_buf.as_entire_binding(),
                },
            ],
        });

        let reduce_params = [num_workgroups as u32];
        let reduce_params_buf =
            device
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Reduce params"),
                    contents: bytemuck::cast_slice(&reduce_params),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

        // Reduce to rz_buffer
        let reduce_rz_bg = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Reduce rz BG"),
            layout: &reduce_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: partial_sums_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: rz_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: reduce_params_buf.as_entire_binding(),
                },
            ],
        });

        // Reduce to rz_new_buffer
        let reduce_rz_new_bg = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Reduce rz_new BG"),
            layout: &reduce_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: partial_sums_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: rz_new_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: reduce_params_buf.as_entire_binding(),
                },
            ],
        });

        // Reduce to pap_buffer
        let reduce_pap_bg = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Reduce pAp BG"),
            layout: &reduce_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: partial_sums_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: pap_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: reduce_params_buf.as_entire_binding(),
                },
            ],
        });

        let cg_params = [n as u32];
        let cg_params_buf = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("CG params"),
                contents: bytemuck::cast_slice(&cg_params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let update_xr_bg = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Update xr BG"),
            layout: &update_xr_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: x_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: r_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: p_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: ap_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: alpha_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: cg_params_buf.as_entire_binding(),
                },
            ],
        });

        let update_p_bg = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Update p BG"),
            layout: &update_p_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: r_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: p_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: beta_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: cg_params_buf.as_entire_binding(),
                },
            ],
        });

        let compute_alpha_bg = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Compute alpha BG"),
            layout: &compute_alpha_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: rz_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: pap_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: alpha_buffer.as_entire_binding(),
                },
            ],
        });

        let compute_beta_bg = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Compute beta BG"),
            layout: &compute_beta_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: rz_new_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: rz_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: beta_buffer.as_entire_binding(),
                },
            ],
        });

        // Initialize: compute rᵀr and store in rz_buffer
        {
            let mut encoder =
                device
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("Init rz"),
                    });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Dot rr Pass"),
                    timestamp_writes: None,
                });
                cg_dispatch_pass(&mut pass, &pl.dot, &rr_bg, num_workgroups as u32, 1, 1);
            }
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Reduce rz Pass"),
                    timestamp_writes: None,
                });
                cg_dispatch_pass(&mut pass, &pl.reduce, &reduce_rz_bg, 1, 1, 1);
            }
            device.submit_and_poll(Some(encoder.finish()));
        }

        // Main CG iteration loop
        let mut last_residual = 1.0;
        for iter in 0..max_iter {
            // 1. Compute Ap
            // 2. Compute pᵀAp
            // 3. α = rz / pAp
            // 4. x = x + α*p, r = r - α*Ap
            // 5. Compute new rᵀr
            // 6. β = rz_new / rz, update rz
            // 7. p = r + β*p

            let mut encoder =
                device
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("CG iter"),
                    });

            // 1. SpMV: Ap = A * p
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("SpMV"),
                    timestamp_writes: None,
                });
                cg_dispatch_pass(
                    &mut pass,
                    &pl.spmv,
                    &spmv_bg,
                    (n as u32).div_ceil(256),
                    1,
                    1,
                );
            }
            // 2. Dot: pᵀAp, then reduce
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Dot pAp"),
                    timestamp_writes: None,
                });
                cg_dispatch_pass(&mut pass, &pl.dot, &pap_bg, num_workgroups as u32, 1, 1);
            }
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Reduce pAp"),
                    timestamp_writes: None,
                });
                cg_dispatch_pass(&mut pass, &pl.reduce, &reduce_pap_bg, 1, 1, 1);
            }
            // 3. α = rz / pAp
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Compute alpha"),
                    timestamp_writes: None,
                });
                cg_dispatch_pass(&mut pass, &pl.compute_alpha, &compute_alpha_bg, 1, 1, 1);
            }
            // 4. x = x + α*p, r = r - α*Ap
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Update xr"),
                    timestamp_writes: None,
                });
                cg_dispatch_pass(
                    &mut pass,
                    &pl.update_xr,
                    &update_xr_bg,
                    (n as u32).div_ceil(256),
                    1,
                    1,
                );
            }
            // 5. New rᵀr
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Dot rr new"),
                    timestamp_writes: None,
                });
                cg_dispatch_pass(&mut pass, &pl.dot, &rr_bg, num_workgroups as u32, 1, 1);
            }
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Reduce rz new"),
                    timestamp_writes: None,
                });
                cg_dispatch_pass(&mut pass, &pl.reduce, &reduce_rz_new_bg, 1, 1, 1);
            }
            // 6. β = rz_new / rz, then rz = rz_new
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Compute beta"),
                    timestamp_writes: None,
                });
                cg_dispatch_pass(&mut pass, &pl.compute_beta, &compute_beta_bg, 1, 1, 1);
            }
            // 7. p = r + β*p
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Update p"),
                    timestamp_writes: None,
                });
                cg_dispatch_pass(
                    &mut pass,
                    &pl.update_p,
                    &update_p_bg,
                    (n as u32).div_ceil(256),
                    1,
                    1,
                );
            }

            device.submit_and_poll(Some(encoder.finish()));

            // Check convergence only every check_interval iterations
            if (iter + 1) % check_interval == 0 || iter == max_iter - 1 {
                let rz_new = SparseBuffers::read_f64(&device, &rz_new_buffer, 1)?;
                let r_norm = rz_new[0].sqrt();
                last_residual = r_norm / b_norm;

                if last_residual < tol {
                    let x_data = SparseBuffers::read_f64(&device, &x_buffer, n)?;
                    return Ok(CgGpuResult {
                        x: x_data,
                        iterations: iter + 1,
                        residual: last_residual,
                        converged: true,
                    });
                }
            }
        }

        // Did not converge
        let x_data = SparseBuffers::read_f64(&device, &x_buffer, n)?;

        Ok(CgGpuResult {
            x: x_data,
            iterations: max_iter,
            residual: last_residual,
            converged: false,
        })
    }
}

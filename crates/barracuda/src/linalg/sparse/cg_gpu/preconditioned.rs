// SPDX-License-Identifier: AGPL-3.0-or-later
//! Preconditioned Conjugate Gradient solver (Jacobi/diagonal preconditioning)
//!
//! M = diag(A) → z = M⁻¹r = r / diag(A). Typically halves iteration count.

use super::{CgGpu, CgGpuResult};
use crate::device::WgpuDevice;
use crate::device::capabilities::WORKGROUP_SIZE_1D;
use crate::error::{BarracudaError, Result};
use crate::linalg::sparse::csr::CsrMatrix;
use crate::linalg::sparse::gpu_helpers::{
    CgPipelineSet, SparseBindGroupLayouts, SparseBuffers, cg_dispatch_pass,
};
use std::sync::Arc;

impl CgGpu {
    /// Solve Ax = b using Preconditioned Conjugate Gradient (GPU-resident)
    /// This version uses Jacobi (diagonal) preconditioning for faster convergence.
    /// M = diag(A) → z = M⁻¹r = r / diag(A)
    /// Preconditioning typically halves the iteration count for poorly-conditioned matrices.
    /// # Errors
    /// Returns [`Err`] if the matrix is not square, if `b.len() != n`, or if
    /// buffer allocation, GPU dispatch, or readback fails (e.g. device lost or
    /// out of memory).
    pub fn solve_preconditioned(
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

        // Extract diagonal for Jacobi preconditioner
        let diag: Vec<f64> = (0..n)
            .map(|i| {
                let row_start = a.row_ptr[i];
                let row_end = a.row_ptr[i + 1];
                for k in row_start..row_end {
                    if a.col_indices[k] == i {
                        return a.values[k];
                    }
                }
                1.0 // Fallback if diagonal not found
            })
            .collect();

        // Create GPU buffers
        let values_buffer = SparseBuffers::f64_from_slice(&device, "PCG values", &a.values);
        let col_indices_buffer =
            SparseBuffers::u32_from_usize(&device, "PCG col_idx", &a.col_indices);
        let row_ptrs_buffer = SparseBuffers::u32_from_usize(&device, "PCG row_ptr", &a.row_ptr);
        let diag_buffer = SparseBuffers::f64_from_slice(&device, "PCG diag", &diag);

        let x_buffer = SparseBuffers::f64_zeros(&device, "PCG x", n);
        let r_buffer = SparseBuffers::f64_from_slice(&device, "PCG r", b);
        let z_buffer = SparseBuffers::f64_zeros(&device, "PCG z", n); // z = M⁻¹r
        let p_buffer = SparseBuffers::f64_zeros(&device, "PCG p", n);
        let ap_buffer = SparseBuffers::f64_zeros(&device, "PCG Ap", n);

        // Scalar buffers
        let num_workgroups = n.div_ceil(WORKGROUP_SIZE_1D as usize);
        let partial_sums_buffer = SparseBuffers::f64_zeros(&device, "PCG partial", num_workgroups);
        let rz_buffer = SparseBuffers::f64_zeros(&device, "PCG rz", 1);
        let rz_new_buffer = SparseBuffers::f64_zeros(&device, "PCG rz_new", 1);
        let pap_buffer = SparseBuffers::f64_zeros(&device, "PCG pAp", 1);
        let alpha_buffer = SparseBuffers::f64_zeros(&device, "PCG alpha", 1);
        let beta_buffer = SparseBuffers::f64_zeros(&device, "PCG beta", 1);

        // Compile shader modules
        let spmv_shader = device.compile_shader_f64(Self::spmv_shader(), Some("PCG SpMV"));
        let dot_reduce_shader =
            device.compile_shader_f64(Self::dot_reduce_shader(), Some("PCG Dot/Reduce"));
        let cg_kernels_shader =
            device.compile_shader_f64(Self::cg_kernels_shader(), Some("PCG Kernels"));
        let vector_ops_shader = device.compile_shader_f64(
            include_str!("../../../shaders/sparse/vector_ops_f64.wgsl"),
            Some("PCG VecOps"),
        );

        // Create bind group layouts and CG pipelines
        let spmv_bgl = SparseBindGroupLayouts::spmv(&device);
        let dot_bgl = SparseBindGroupLayouts::dot(&device);
        let reduce_bgl = SparseBindGroupLayouts::reduce(&device);
        let update_xr_bgl = SparseBindGroupLayouts::cg_update_xr(&device);
        let update_p_bgl = SparseBindGroupLayouts::cg_update_p(&device);
        let compute_alpha_bgl = SparseBindGroupLayouts::compute_alpha(&device);
        let compute_beta_bgl = SparseBindGroupLayouts::compute_beta(&device);
        let precond_bgl = SparseBindGroupLayouts::precond(&device);

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

        let precond_pipeline =
            device
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("Precond f64"),
                    layout: Some(&device.device.create_pipeline_layout(
                        &wgpu::PipelineLayoutDescriptor {
                            label: Some("Precond PL"),
                            bind_group_layouts: &[&precond_bgl],
                            immediate_size: 0,
                        },
                    )),
                    module: &vector_ops_shader,
                    entry_point: Some("precond_f64"),
                    cache: None,
                    compilation_options: Default::default(),
                });

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

        // rᵀz dot product bind group
        let rz_bg = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rz BG"),
            layout: &dot_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: r_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: z_buffer.as_entire_binding(),
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

        // rᵀr for convergence check
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

        let reduce_params = [num_workgroups as u32];
        let reduce_params_buf =
            device
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Reduce params"),
                    contents: bytemuck::cast_slice(&reduce_params),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

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

        let precond_params = [n as u32];
        let precond_params_buf =
            device
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Precond params"),
                    contents: bytemuck::cast_slice(&precond_params),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

        let precond_bg = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Precond BG"),
            layout: &precond_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: r_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: diag_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: z_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: precond_params_buf.as_entire_binding(),
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

        // p = z + β*p (using z instead of r)
        let update_p_bg = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Update p BG"),
            layout: &update_p_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: z_buffer.as_entire_binding(),
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

        // Initialize: z₀ = M⁻¹r₀, p₀ = z₀, compute r₀ᵀz₀
        {
            let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
                label: Some("PCG init"),
            });

            // z = M⁻¹r (apply preconditioner)
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Precond Pass"),
                    timestamp_writes: None,
                });
                cg_dispatch_pass(
                    &mut pass,
                    &precond_pipeline,
                    &precond_bg,
                    (n as u32).div_ceil(WORKGROUP_SIZE_1D),
                    1,
                    1,
                );
            }

            device.submit_commands(Some(encoder.finish()));

            // Copy z to p: p₀ = z₀
            SparseBuffers::copy_f64(&device, &z_buffer, &p_buffer, n);

            // Compute rᵀz
            let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
                label: Some("Init rz"),
            });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Dot rz Pass"),
                    timestamp_writes: None,
                });
                cg_dispatch_pass(&mut pass, &pl.dot, &rz_bg, num_workgroups as u32, 1, 1);
            }
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Reduce rz Pass"),
                    timestamp_writes: None,
                });
                cg_dispatch_pass(&mut pass, &pl.reduce, &reduce_rz_bg, 1, 1, 1);
            }
            device.submit_commands(Some(encoder.finish()));
        }

        // Main PCG iteration
        let mut last_residual = 1.0;
        for iter in 0..max_iter {
            let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
                label: Some("PCG iter"),
            });

            // 1. Ap = A * p
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("SpMV"),
                    timestamp_writes: None,
                });
                cg_dispatch_pass(
                    &mut pass,
                    &pl.spmv,
                    &spmv_bg,
                    (n as u32).div_ceil(WORKGROUP_SIZE_1D),
                    1,
                    1,
                );
            }
            // 2. pᵀAp, then reduce
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
            // 3. α = (rᵀz) / (pᵀAp)
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Compute alpha"),
                    timestamp_writes: None,
                });
                cg_dispatch_pass(&mut pass, &pl.compute_alpha, &compute_alpha_bg, 1, 1, 1);
            }
            // 4. x = x + αp, r = r - αAp
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Update xr"),
                    timestamp_writes: None,
                });
                cg_dispatch_pass(
                    &mut pass,
                    &pl.update_xr,
                    &update_xr_bg,
                    (n as u32).div_ceil(WORKGROUP_SIZE_1D),
                    1,
                    1,
                );
            }
            // 5. z = M⁻¹r
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Precond"),
                    timestamp_writes: None,
                });
                cg_dispatch_pass(
                    &mut pass,
                    &precond_pipeline,
                    &precond_bg,
                    (n as u32).div_ceil(WORKGROUP_SIZE_1D),
                    1,
                    1,
                );
            }
            // 6. New rᵀz
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Dot rz new"),
                    timestamp_writes: None,
                });
                cg_dispatch_pass(&mut pass, &pl.dot, &rz_bg, num_workgroups as u32, 1, 1);
            }
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Reduce rz new"),
                    timestamp_writes: None,
                });
                cg_dispatch_pass(&mut pass, &pl.reduce, &reduce_rz_new_bg, 1, 1, 1);
            }
            // 7. β = (r_new ᵀ z_new) / (rᵀz), then rz = rz_new
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Compute beta"),
                    timestamp_writes: None,
                });
                cg_dispatch_pass(&mut pass, &pl.compute_beta, &compute_beta_bg, 1, 1, 1);
            }
            // 8. p = z + βp
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Update p"),
                    timestamp_writes: None,
                });
                cg_dispatch_pass(
                    &mut pass,
                    &pl.update_p,
                    &update_p_bg,
                    (n as u32).div_ceil(WORKGROUP_SIZE_1D),
                    1,
                    1,
                );
            }

            device.submit_commands(Some(encoder.finish()));

            // Check convergence
            if (iter + 1) % check_interval == 0 || iter == max_iter - 1 {
                let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
                    label: Some("Check convergence"),
                });
                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("Dot rr"),
                        timestamp_writes: None,
                    });
                    cg_dispatch_pass(&mut pass, &pl.dot, &rr_bg, num_workgroups as u32, 1, 1);
                }
                {
                    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: Some("Reduce rr"),
                        timestamp_writes: None,
                    });
                    cg_dispatch_pass(&mut pass, &pl.reduce, &reduce_rz_new_bg, 1, 1, 1);
                }
                device.submit_commands(Some(encoder.finish()));

                let rr = SparseBuffers::read_f64(&device, &rz_new_buffer, 1)?;
                let r_norm = rr[0].sqrt();
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

        let x_data = SparseBuffers::read_f64(&device, &x_buffer, n)?;

        Ok(CgGpuResult {
            x: x_data,
            iterations: max_iter,
            residual: last_residual,
            converged: false,
        })
    }
}

// SPDX-License-Identifier: AGPL-3.0-or-later
//! Standard Jacobi sweep — multi-dispatch variant (all matrix sizes)
//!
//! Iterates over (p,q) pairs, dispatching compute angles → rotate A → update blocks → rotate V
//! for each pair. Used by execute_f64 and execute_f64_buffers.

use super::params::{BatchedEighParams, ParallelSweepParams};
use super::pipelines::create_eigh_pipelines;
use super::sweep::run_sweep_pass;
use super::BatchedEighGpu;
use crate::device::capabilities::WORKGROUP_SIZE_1D;
use crate::device::WgpuDevice;
use crate::error::{BarracudaError, Result};
use std::sync::Arc;

impl BatchedEighGpu {
    /// Execute batched eigenvalue decomposition on GPU with full f64 precision
    ///
    /// This processes multiple symmetric matrices in parallel, ideal for HFB
    /// Hamiltonian diagonalization where 52 nuclei need simultaneous eigensolves.
    ///
    /// # Arguments
    /// * `device` - WgpuDevice to execute on
    /// * `data` - Packed matrices [batch_size × n × n] in row-major order (f64)
    /// * `n` - Matrix dimension (same for all matrices)
    /// * `batch_size` - Number of matrices
    /// * `max_sweeps` - Maximum Jacobi sweeps (default: 30)
    ///
    /// # Returns
    /// Tuple (eigenvalues, eigenvectors) where:
    /// - eigenvalues: [batch_size × n] f64
    /// - eigenvectors: [batch_size × n × n] f64
    pub fn execute_f64(
        device: Arc<WgpuDevice>,
        data: &[f64],
        n: usize,
        batch_size: usize,
        max_sweeps: u32,
    ) -> Result<(Vec<f64>, Vec<f64>)> {
        let expected_len = batch_size * n * n;
        if data.len() != expected_len {
            return Err(BarracudaError::InvalidInput {
                message: format!(
                    "Expected {} elements for {} matrices of {}x{}, got {}",
                    expected_len,
                    batch_size,
                    n,
                    n,
                    data.len()
                ),
            });
        }

        let nu = n as u32;
        let batch_u = batch_size as u32;

        // Create A buffer (input matrices, will be modified in-place)
        let a_buffer = {
            let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
            device
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Batched A f64"),
                    contents: &bytes,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                })
        };

        // Create V buffer (eigenvectors output)
        let v_size = (batch_size * n * n * 8) as u64;
        let v_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Batched V f64"),
            size: v_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create eigenvalues buffer
        let eig_size = (batch_size * n * 8) as u64;
        let eig_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Batched eigenvalues f64"),
            size: eig_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create cos/sin buffer for rotation angles [batch_size × 2]
        let cs_size = (batch_size * 2 * 8) as u64;
        let cs_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Batched cs f64"),
            size: cs_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let shader = device.compile_shader_f64(Self::wgsl_shader(), Some("Batched Eigh f64"));
        let pipelines = create_eigh_pipelines(&device, &shader);

        let params = BatchedEighParams {
            n: nu,
            batch_size: batch_u,
            max_sweeps,
            _pad: 0,
        };
        let params_buffer = device.create_uniform_buffer("Batched Eigh Params", &params);

        let init_bg = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Batched Init BG"),
            layout: &pipelines.init_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: a_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: v_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: eig_buffer.as_entire_binding(),
                },
            ],
        });

        // Step 1: Initialize V = Identity for all matrices
        {
            let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
                label: Some("Init V Encoder"),
            });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Init V Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&pipelines.init_v_pipeline);
                pass.set_bind_group(0, &init_bg, &[]);
                let wg_xy = nu.div_ceil(16);
                pass.dispatch_workgroups(wg_xy, wg_xy, batch_u);
            }
            device.submit_and_poll(Some(encoder.finish()));
        }

        // Step 2: Jacobi sweeps
        for _sweep in 0..max_sweeps {
            for p in 0..(n - 1) {
                for q in (p + 1)..n {
                    let sweep_params = ParallelSweepParams {
                        n: nu,
                        batch_size: batch_u,
                        current_p: p as u32,
                        current_q: q as u32,
                    };
                    let sweep_params_buffer =
                        device.create_uniform_buffer("Sweep Params", &sweep_params);

                    let sweep_bg = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("Sweep BG"),
                        layout: &pipelines.sweep_bgl,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: sweep_params_buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: a_buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: v_buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: cs_buffer.as_entire_binding(),
                            },
                        ],
                    });

                    run_sweep_pass(
                        &device,
                        &sweep_bg,
                        &pipelines.compute_angles_pipeline,
                        (batch_u.div_ceil(64), 1, 1),
                    );
                    run_sweep_pass(
                        &device,
                        &sweep_bg,
                        &pipelines.rotate_a_pipeline,
                        (nu.div_ceil(64), batch_u, 1),
                    );
                    run_sweep_pass(
                        &device,
                        &sweep_bg,
                        &pipelines.update_blocks_pipeline,
                        (batch_u.div_ceil(64), 1, 1),
                    );
                    run_sweep_pass(
                        &device,
                        &sweep_bg,
                        &pipelines.rotate_v_pipeline,
                        (nu.div_ceil(64), batch_u, 1),
                    );
                }
            }
        }

        // Step 3: Extract eigenvalues from diagonal of A
        {
            let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
                label: Some("Extract Eigenvalues"),
            });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Extract Eigenvalues Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&pipelines.extract_pipeline);
                pass.set_bind_group(0, &init_bg, &[]);
                pass.dispatch_workgroups(nu.div_ceil(WORKGROUP_SIZE_1D), batch_u, 1);
            }
            device.submit_and_poll(Some(encoder.finish()));
        }

        let eigenvalues = device.read_f64_buffer(&eig_buffer, batch_size * n)?;
        let eigenvectors = device.read_f64_buffer(&v_buffer, batch_size * n * n)?;

        Ok((eigenvalues, eigenvectors))
    }

    /// Execute batched eigenvalue decomposition on GPU **without CPU readback**
    ///
    /// This is the GPU-resident variant for use in pipelines where data should
    /// stay on GPU between operations (e.g., SCF iteration loops).
    pub fn execute_f64_buffers(
        device: &Arc<WgpuDevice>,
        matrices_buffer: &wgpu::Buffer,
        eigenvalues_buffer: &wgpu::Buffer,
        eigenvectors_buffer: &wgpu::Buffer,
        n: usize,
        batch_size: usize,
        max_sweeps: u32,
    ) -> Result<()> {
        let nu = n as u32;
        let batch_u = batch_size as u32;

        let cs_size = (batch_size * 2 * 8) as u64;
        let cs_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("BatchedEigh cs f64"),
            size: cs_size,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let shader =
            device.compile_shader_f64(Self::wgsl_shader(), Some("Batched Eigh f64 (buffers)"));
        let pipelines = create_eigh_pipelines(device, &shader);

        let params = BatchedEighParams {
            n: nu,
            batch_size: batch_u,
            max_sweeps,
            _pad: 0,
        };
        let params_buffer = device.create_uniform_buffer("Batched Eigh Params (buffers)", &params);

        let init_bg = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Batched Init BG (buffers)"),
            layout: &pipelines.init_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: matrices_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: eigenvectors_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: eigenvalues_buffer.as_entire_binding(),
                },
            ],
        });

        {
            let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
                label: Some("Init V Encoder (buffers)"),
            });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Init V Pass (buffers)"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&pipelines.init_v_pipeline);
                pass.set_bind_group(0, &init_bg, &[]);
                let wg_xy = nu.div_ceil(16);
                pass.dispatch_workgroups(wg_xy, wg_xy, batch_u);
            }
            device.submit_and_poll(Some(encoder.finish()));
        }

        for _sweep in 0..max_sweeps {
            for p in 0..(n - 1) {
                for q in (p + 1)..n {
                    let sweep_params = ParallelSweepParams {
                        n: nu,
                        batch_size: batch_u,
                        current_p: p as u32,
                        current_q: q as u32,
                    };
                    let sweep_params_buffer =
                        device.create_uniform_buffer("Sweep Params (buffers)", &sweep_params);

                    let sweep_bg = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("Sweep BG (buffers)"),
                        layout: &pipelines.sweep_bgl,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: sweep_params_buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: matrices_buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: eigenvectors_buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 3,
                                resource: cs_buffer.as_entire_binding(),
                            },
                        ],
                    });

                    run_sweep_pass(
                        device,
                        &sweep_bg,
                        &pipelines.compute_angles_pipeline,
                        (batch_u.div_ceil(64), 1, 1),
                    );
                    run_sweep_pass(
                        device,
                        &sweep_bg,
                        &pipelines.rotate_a_pipeline,
                        (nu.div_ceil(64), batch_u, 1),
                    );
                    run_sweep_pass(
                        device,
                        &sweep_bg,
                        &pipelines.update_blocks_pipeline,
                        (batch_u.div_ceil(64), 1, 1),
                    );
                    run_sweep_pass(
                        device,
                        &sweep_bg,
                        &pipelines.rotate_v_pipeline,
                        (nu.div_ceil(64), batch_u, 1),
                    );
                }
            }
        }

        {
            let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
                label: Some("Extract Eigenvalues (buffers)"),
            });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Extract Eigenvalues Pass (buffers)"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&pipelines.extract_pipeline);
                pass.set_bind_group(0, &init_bg, &[]);
                pass.dispatch_workgroups(nu.div_ceil(WORKGROUP_SIZE_1D), batch_u, 1);
            }
            device.submit_and_poll(Some(encoder.finish()));
        }

        Ok(())
    }
}

// SPDX-License-Identifier: AGPL-3.0-or-later
//! Cyclic Reduction (f64) — Parallel Tridiagonal Solver for PDEs
//!
//! Solves tridiagonal systems: a[i]*x[i-1] + b[i]*x[i] + c[i]*x[i+1] = d[i]
//!
//! **Use cases**:
//! - Crank-Nicolson PDE (heat, diffusion, Schrödinger) — all springs
//! - Richards equation for unsaturated flow — airSpring, wetSpring
//! - Implicit finite difference schemes — hotSpring
//! - Cubic spline interpolation
//!
//! **Deep Debt Principles**:
//! - Pure WGSL implementation (hardware-agnostic)
//! - Full f64 precision for science-grade stability
//! - O(log n) parallel complexity vs O(n) sequential Thomas
//! - Safe Rust wrapper (no unsafe code)

use crate::device::WgpuDevice;
use crate::error::{BarracudaError, Result};
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;

/// Parameters for cyclic reduction shader
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct CyclicParams {
    n: u32,
    step: u32,
    phase: u32, // 0 = reduction, 1 = substitution
    _pad: u32,
}

/// Type alias for tridiagonal system: (sub_diag, main_diag, super_diag, rhs)
pub type TridiagonalSystem = (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>);

/// GPU-accelerated f64 tridiagonal solver via cyclic reduction
pub struct CyclicReductionF64 {
    device: Arc<WgpuDevice>,
}

impl CyclicReductionF64 {
    fn wgsl_shader() -> &'static str {
        include_str!("../../shaders/linalg/cyclic_reduction_f64.wgsl")
    }

    /// Create a new CyclicReductionF64 orchestrator
    pub fn new(device: Arc<WgpuDevice>) -> Result<Self> {
        Ok(Self { device })
    }

    /// Solve tridiagonal system Ax = d where A is tridiagonal
    ///
    /// # Arguments
    /// * `a` - Sub-diagonal (length n, a[0] unused)
    /// * `b` - Main diagonal (length n)
    /// * `c` - Super-diagonal (length n, c[n-1] unused)
    /// * `d` - Right-hand side (length n)
    ///
    /// # Returns
    /// Solution vector x of length n
    ///
    /// # Example
    /// ```ignore
    /// // Solve: 4x₀ + x₁ = 5
    /// //        x₀ + 4x₁ + x₂ = 6
    /// //        x₁ + 4x₂ = 5
    /// let a = vec![0.0, 1.0, 1.0];
    /// let b = vec![4.0, 4.0, 4.0];
    /// let c = vec![1.0, 1.0, 0.0];
    /// let d = vec![5.0, 6.0, 5.0];
    /// let x = solver.solve(&a, &b, &c, &d)?;
    /// // x ≈ [1.0, 1.0, 1.0]
    /// ```
    pub fn solve(&self, a: &[f64], b: &[f64], c: &[f64], d: &[f64]) -> Result<Vec<f64>> {
        let n = b.len();

        if a.len() != n || c.len() != n || d.len() != n {
            return Err(BarracudaError::InvalidInput {
                message: format!(
                    "All vectors must have length {}: a={}, b={}, c={}, d={}",
                    n,
                    a.len(),
                    b.len(),
                    c.len(),
                    d.len()
                ),
            });
        }

        if n == 0 {
            return Ok(vec![]);
        }

        if n == 1 {
            if b[0].abs() < 1e-14 {
                return Err(BarracudaError::InvalidInput {
                    message: "Singular matrix: b[0] = 0".to_string(),
                });
            }
            return Ok(vec![d[0] / b[0]]);
        }

        if n >= 2048 {
            self.solve_gpu_parallel(a, b, c, d)
        } else {
            self.solve_gpu_serial(a, b, c, d)
        }
    }

    /// Batched solve for multiple independent systems
    ///
    /// # Arguments
    /// * `systems` - Vector of `TridiagonalSystem` (a, b, c, d) tuples
    ///
    /// # Returns
    /// Vector of solution vectors
    pub fn solve_batch(&self, systems: &[TridiagonalSystem]) -> Result<Vec<Vec<f64>>> {
        // For now, solve sequentially (batched GPU version coming)
        systems
            .iter()
            .map(|(a, b, c, d)| self.solve(a, b, c, d))
            .collect()
    }

    /// GPU serial solver using Thomas algorithm in a single kernel
    /// No synchronization issues - O(n) but runs on GPU memory
    fn solve_gpu_serial(&self, a: &[f64], b: &[f64], c: &[f64], d: &[f64]) -> Result<Vec<f64>> {
        let n = b.len();

        let shader = self
            .device
            .compile_shader_f64(Self::wgsl_shader(), Some("Cyclic Serial f64"));

        // Create GPU buffers
        let a_buf = self
            .device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("A (sub-diag)"),
                contents: bytemuck::cast_slice(a),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

        let b_buf = self
            .device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("B (main-diag)"),
                contents: bytemuck::cast_slice(b),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

        let c_buf = self
            .device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("C (super-diag)"),
                contents: bytemuck::cast_slice(c),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

        let d_buf = self
            .device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("D (RHS/solution)"),
                contents: bytemuck::cast_slice(d),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            });

        // Create bind group layout
        let bgl = self
            .device
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Serial BGL"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let pl = self
            .device
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Serial PL"),
                bind_group_layouts: &[&bgl],
                immediate_size: 0,
            });

        let serial_pipeline =
            self.device
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("Serial Pipeline"),
                    layout: Some(&pl),
                    module: &shader,
                    entry_point: Some("solve_serial_f64"),
                    cache: None,
                    compilation_options: Default::default(),
                });

        let params = CyclicParams {
            n: n as u32,
            step: 0,
            phase: 0,
            _pad: 0,
        };

        let params_buf = self
            .device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bg = self
            .device
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Serial BG"),
                layout: &bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: params_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: a_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: b_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: c_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: d_buf.as_entire_binding(),
                    },
                ],
            });

        let mut encoder = self
            .device
            .create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
                label: Some("Serial Encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Serial Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&serial_pipeline);
            pass.set_bind_group(0, Some(&bg), &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }

        // Read back solution
        let staging = self.device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging"),
            size: (n * 8) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        encoder.copy_buffer_to_buffer(&d_buf, 0, &staging, 0, (n * 8) as u64);
        self.device.submit_and_poll(Some(encoder.finish()));

        let result: Vec<f64> = self.device.map_staging_buffer(&staging, n)?;
        Ok(result)
    }

    /// GPU parallel cyclic reduction solver
    ///
    /// O(log n) parallel — dispatched for n >= 2048 where parallelism amortizes
    /// the extra passes. For smaller systems, `solve_gpu_serial` is preferred.
    fn solve_gpu_parallel(&self, a: &[f64], b: &[f64], c: &[f64], d: &[f64]) -> Result<Vec<f64>> {
        let n = b.len();
        let n_padded = n.next_power_of_two();
        let num_steps = (n_padded as f64).log2() as u32;

        // Pad arrays to power of 2
        let mut a_data: Vec<f64> = a.to_vec();
        let mut b_data: Vec<f64> = b.to_vec();
        let mut c_data: Vec<f64> = c.to_vec();
        let mut d_data: Vec<f64> = d.to_vec();

        a_data.resize(n_padded, 0.0);
        b_data.resize(n_padded, 1.0); // Identity for padded elements
        c_data.resize(n_padded, 0.0);
        d_data.resize(n_padded, 0.0);

        let shader = self
            .device
            .compile_shader_f64(Self::wgsl_shader(), Some("Cyclic Reduction f64"));

        // Create mutable GPU buffers
        let a_buf = self
            .device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("A (sub-diag)"),
                contents: bytemuck::cast_slice(&a_data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

        let b_buf = self
            .device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("B (main-diag)"),
                contents: bytemuck::cast_slice(&b_data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

        let c_buf = self
            .device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("C (super-diag)"),
                contents: bytemuck::cast_slice(&c_data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

        let d_buf = self
            .device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("D (RHS/solution)"),
                contents: bytemuck::cast_slice(&d_data),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            });

        // Create bind group layout
        let bgl = self
            .device
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Cyclic Reduction BGL"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let pl = self
            .device
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Cyclic Reduction PL"),
                bind_group_layouts: &[&bgl],
                immediate_size: 0,
            });

        // Create pipelines for reduction and substitution
        let reduction_pipeline =
            self.device
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("Reduction Pipeline"),
                    layout: Some(&pl),
                    module: &shader,
                    entry_point: Some("reduction_f64"),
                    cache: None,
                    compilation_options: Default::default(),
                });

        let substitution_pipeline =
            self.device
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("Substitution Pipeline"),
                    layout: Some(&pl),
                    module: &shader,
                    entry_point: Some("substitution_f64"),
                    cache: None,
                    compilation_options: Default::default(),
                });

        // Multi-pass cyclic reduction
        let workgroup_size = 256;

        // Reduction phase: O(log n) passes
        for step in 0..num_steps {
            let params = CyclicParams {
                n: n_padded as u32,
                step,
                phase: 0,
                _pad: 0,
            };

            let params_buf =
                self.device
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("Params"),
                        contents: bytemuck::bytes_of(&params),
                        usage: wgpu::BufferUsages::UNIFORM,
                    });

            let bg = self
                .device
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Reduction BG"),
                    layout: &bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: params_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: a_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: b_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: c_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: d_buf.as_entire_binding(),
                        },
                    ],
                });

            let n_threads = n_padded >> (step + 1);
            let n_workgroups = n_threads.div_ceil(workgroup_size);

            let mut encoder = self
                .device
                .create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
                    label: Some("Reduction Encoder"),
                });

            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Reduction Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&reduction_pipeline);
                pass.set_bind_group(0, Some(&bg), &[]);
                pass.dispatch_workgroups(n_workgroups.max(1) as u32, 1, 1);
            }

            self.device.submit_and_poll(Some(encoder.finish()));
        }

        // Substitution phase: O(log n) passes in reverse
        for step in (0..num_steps).rev() {
            let params = CyclicParams {
                n: n_padded as u32,
                step,
                phase: 1,
                _pad: 0,
            };

            let params_buf =
                self.device
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("Params"),
                        contents: bytemuck::bytes_of(&params),
                        usage: wgpu::BufferUsages::UNIFORM,
                    });

            let bg = self
                .device
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("Substitution BG"),
                    layout: &bgl,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: params_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: a_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: b_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: c_buf.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: d_buf.as_entire_binding(),
                        },
                    ],
                });

            let n_threads = n_padded >> (step + 1);
            let n_workgroups = n_threads.div_ceil(workgroup_size);

            let mut encoder = self
                .device
                .create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
                    label: Some("Substitution Encoder"),
                });

            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Substitution Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&substitution_pipeline);
                pass.set_bind_group(0, Some(&bg), &[]);
                pass.dispatch_workgroups(n_workgroups.max(1) as u32, 1, 1);
            }

            self.device.submit_and_poll(Some(encoder.finish()));
        }

        // Read back solution (stored in d_buf)
        let staging = self.device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging"),
            size: (n * 8) as u64, // f64 = 8 bytes, only read first n
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
                label: Some("Copy Encoder"),
            });
        encoder.copy_buffer_to_buffer(&d_buf, 0, &staging, 0, (n * 8) as u64);
        self.device.submit_and_poll(Some(encoder.finish()));

        let result: Vec<f64> = self.device.map_staging_buffer(&staging, n)?;
        Ok(result)
    }
}

#[expect(clippy::unwrap_used, reason = "tests")]
#[cfg(test)]
mod cpu_reference;

#[expect(clippy::unwrap_used, reason = "tests")]
#[cfg(test)]
mod tests;

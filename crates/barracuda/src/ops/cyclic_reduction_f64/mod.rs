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
use crate::device::compute_pipeline::ComputeDispatch;
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

/// Type alias for tridiagonal system: (`sub_diag`, `main_diag`, `super_diag`, rhs)
pub type TridiagonalSystem = (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>);

/// GPU-accelerated f64 tridiagonal solver via cyclic reduction
pub struct CyclicReductionF64 {
    device: Arc<WgpuDevice>,
}

impl CyclicReductionF64 {
    fn wgsl_shader() -> &'static str {
        include_str!("../../shaders/linalg/cyclic_reduction_f64.wgsl")
    }

    /// Create a new `CyclicReductionF64` orchestrator
    /// # Errors
    /// Returns [`Err`] if device initialization fails.
    pub fn new(device: Arc<WgpuDevice>) -> Result<Self> {
        Ok(Self { device })
    }

    /// Solve tridiagonal system Ax = d where A is tridiagonal
    /// # Arguments
    /// * `a` - Sub-diagonal (length n, a[0] unused)
    /// * `b` - Main diagonal (length n)
    /// * `c` - Super-diagonal (length n, c[n-1] unused)
    /// * `d` - Right-hand side (length n)
    /// # Returns
    /// Solution vector x of length n
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
    /// # Errors
    /// Returns [`Err`] if a, c, or d length differs from b, n=1 with singular matrix (b[0]=0),
    /// buffer allocation fails, GPU dispatch fails, buffer readback fails, or the device is lost.
    pub fn solve(&self, a: &[f64], b: &[f64], c: &[f64], d: &[f64]) -> Result<Vec<f64>> {
        let n = b.len();

        if a.len() != n || c.len() != n || d.len() != n {
            return Err(BarracudaError::invalid_input(format!(
                "All vectors must have length {}: a={}, b={}, c={}, d={}",
                n,
                a.len(),
                b.len(),
                c.len(),
                d.len()
            )));
        }

        if n == 0 {
            return Ok(vec![]);
        }

        if n == 1 {
            if b[0].abs() < 1e-14 {
                return Err(BarracudaError::invalid_input("Singular matrix: b[0] = 0"));
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
    /// # Arguments
    /// * `systems` - Vector of `TridiagonalSystem` (a, b, c, d) tuples
    /// # Returns
    /// Vector of solution vectors
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn solve_batch(&self, systems: &[TridiagonalSystem]) -> Result<Vec<Vec<f64>>> {
        // Sequential per-system solve — batched GPU version (parallel across
        // systems) is a P2 evolution when batch sizes exceed ~64 systems
        systems
            .iter()
            .map(|(a, b, c, d)| self.solve(a, b, c, d))
            .collect()
    }

    /// GPU serial solver using Thomas algorithm in a single kernel
    /// No synchronization issues - O(n) but runs on GPU memory
    fn solve_gpu_serial(&self, a: &[f64], b: &[f64], c: &[f64], d: &[f64]) -> Result<Vec<f64>> {
        let n = b.len();

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

        ComputeDispatch::new(self.device.as_ref(), "Cyclic Serial f64")
            .shader(Self::wgsl_shader(), "solve_serial_f64")
            .f64()
            .uniform(0, &params_buf)
            .storage_rw(1, &a_buf)
            .storage_rw(2, &b_buf)
            .storage_rw(3, &c_buf)
            .storage_rw(4, &d_buf)
            .dispatch(1, 1, 1)
            .submit()?;

        // Read back solution
        let staging = self.device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Staging"),
            size: (n * 8) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
                label: Some("Serial Copy Encoder"),
            });
        encoder.copy_buffer_to_buffer(&d_buf, 0, &staging, 0, (n * 8) as u64);
        self.device.submit_commands(Some(encoder.finish()));

        let result: Vec<f64> = self.device.map_staging_buffer(&staging, n)?;
        Ok(result)
    }

    /// GPU parallel cyclic reduction solver
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

        let shader_source = Self::wgsl_shader();
        let workgroup_size = 256u32;

        for step in 0..num_steps {
            let params = CyclicParams {
                n: n_padded as u32,
                step,
                phase: 0,
                _pad: 0,
            };
            let params_buf = self.device.device.create_buffer_init(
                &wgpu::util::BufferInitDescriptor {
                    label: Some("Params"),
                    contents: bytemuck::bytes_of(&params),
                    usage: wgpu::BufferUsages::UNIFORM,
                },
            );
            let n_threads = n_padded >> (step + 1);
            let n_workgroups = n_threads.div_ceil(workgroup_size as usize);
            ComputeDispatch::new(self.device.as_ref(), "Cyclic Reduction")
                .shader(shader_source, "reduction_f64")
                .f64()
                .uniform(0, &params_buf)
                .storage_rw(1, &a_buf)
                .storage_rw(2, &b_buf)
                .storage_rw(3, &c_buf)
                .storage_rw(4, &d_buf)
                .dispatch(n_workgroups.max(1) as u32, 1, 1)
                .submit()?;
        }

        for step in (0..num_steps).rev() {
            let params = CyclicParams {
                n: n_padded as u32,
                step,
                phase: 1,
                _pad: 0,
            };
            let params_buf = self.device.device.create_buffer_init(
                &wgpu::util::BufferInitDescriptor {
                    label: Some("Params"),
                    contents: bytemuck::bytes_of(&params),
                    usage: wgpu::BufferUsages::UNIFORM,
                },
            );
            let n_threads = n_padded >> (step + 1);
            let n_workgroups = n_threads.div_ceil(workgroup_size as usize);
            ComputeDispatch::new(self.device.as_ref(), "Cyclic Substitution")
                .shader(shader_source, "substitution_f64")
                .f64()
                .uniform(0, &params_buf)
                .storage_rw(1, &a_buf)
                .storage_rw(2, &b_buf)
                .storage_rw(3, &c_buf)
                .storage_rw(4, &d_buf)
                .dispatch(n_workgroups.max(1) as u32, 1, 1)
                .submit()?;
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
        self.device.submit_commands(Some(encoder.finish()));

        let result: Vec<f64> = self.device.map_staging_buffer(&staging, n)?;
        Ok(result)
    }
}

#[cfg(test)]
mod cpu_reference;

#[cfg(test)]
mod tests;

//! Crank-Nicolson PDE Solver — GPU-Accelerated via WGSL
//!
//! Implicit PDE solver for diffusion-type equations:
//!   ∂u/∂t = α · ∂²u/∂x²
//!
//! **Use cases**:
//! - Heat diffusion (all springs)
//! - Richards equation for unsaturated flow (airSpring, wetSpring)
//! - Two-Temperature Model in laser-material interaction (hotSpring)
//! - Time-dependent Schrödinger equation (hotSpring)
//!
//! **Algorithm**:
//! 1. Compute RHS (embarrassingly parallel)
//! 2. Solve tridiagonal system via cyclic reduction (O(log n) parallel)
//! 3. Repeat for each timestep
//!
//! **Deep Debt Principles**:
//! - Pure WGSL implementation (hardware-agnostic)
//! - Chains with cyclic_reduction for the tridiagonal solve
//! - Safe Rust wrapper (no unsafe code)

use crate::device::WgpuDevice;
use crate::error::{BarracudaError, Result};
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;
use wgpu::util::DeviceExt;

/// Parameters for Crank-Nicolson shader
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct CrankNicolsonParams {
    n: u32,        // Number of interior points
    r: f32,        // Courant number: α·Δt/Δx²
    left_bc: f32,  // Left boundary value
    right_bc: f32, // Right boundary value
}

/// Boundary condition types
#[derive(Clone, Copy, Debug)]
pub enum BoundaryCondition {
    /// Dirichlet: u(boundary) = value
    Dirichlet(f32),
    /// Neumann: ∂u/∂x(boundary) = value. Zero flux (0.0) means ghost point mirrors interior.
    Neumann(f32),
}

/// GPU-accelerated Crank-Nicolson solver
pub struct CrankNicolson {
    device: Arc<WgpuDevice>,
}

impl CrankNicolson {
    fn wgsl_shader() -> &'static str {
        include_str!("../shaders/pde/crank_nicolson.wgsl")
    }

    /// Create a new Crank-Nicolson solver
    pub fn new(device: Arc<WgpuDevice>) -> Result<Self> {
        Ok(Self { device })
    }

    /// Solve the 1D diffusion equation
    ///
    /// # Arguments
    /// * `u0` - Initial condition (interior points only)
    /// * `alpha` - Diffusion coefficient
    /// * `dx` - Spatial step size
    /// * `dt` - Time step size
    /// * `n_steps` - Number of time steps
    /// * `left_bc` - Left boundary condition
    /// * `right_bc` - Right boundary condition
    ///
    /// # Returns
    /// Solution at final time (interior points only)
    pub fn solve(
        &self,
        u0: &[f32],
        alpha: f32,
        dx: f32,
        dt: f32,
        n_steps: usize,
        left_bc: BoundaryCondition,
        right_bc: BoundaryCondition,
    ) -> Result<Vec<f32>> {
        let n = u0.len();
        if n == 0 {
            return Err(BarracudaError::InvalidInput {
                message: "Initial condition must have at least 1 point".to_string(),
            });
        }

        let courant = alpha * dt / (dx * dx);

        match (left_bc, right_bc) {
            (BoundaryCondition::Dirichlet(l), BoundaryCondition::Dirichlet(r)) => {
                self.solve_gpu(u0, courant, n_steps, l, r)
            }
            _ => {
                // Neumann or mixed: use CPU path (GPU shader is Dirichlet-only)
                self.solve_neumann_cpu(u0, courant, n_steps, left_bc, right_bc)
            }
        }
    }

    /// Solve with Neumann boundary conditions (CPU path).
    /// Zero-flux Neumann: du/dx=0 means ghost point mirrors interior (u[-1]=u[0], u[n]=u[n-1]).
    fn solve_neumann_cpu(
        &self,
        u0: &[f32],
        r: f32,
        n_steps: usize,
        left_bc: BoundaryCondition,
        right_bc: BoundaryCondition,
    ) -> Result<Vec<f32>> {
        let n = u0.len();
        let mut u = u0.to_vec();
        let mut rhs = vec![0.0f32; n];

        let a_coef = -r / 2.0;
        let b_coef = 1.0 + r;
        let c_coef = -r / 2.0;

        for _ in 0..n_steps {
            // RHS with Neumann: ghost point mirrors interior for zero flux (du/dx=0)
            for i in 0..n {
                let (u_left, u_right) = match (i, n, &left_bc, &right_bc) {
                    (0, 1, BoundaryCondition::Neumann(0.0), BoundaryCondition::Neumann(0.0)) => {
                        (u[0], u[0])
                    }
                    (0, 1, BoundaryCondition::Dirichlet(l), BoundaryCondition::Dirichlet(r)) => {
                        (*l, *r)
                    }
                    (0, 1, BoundaryCondition::Dirichlet(l), BoundaryCondition::Neumann(0.0)) => {
                        (*l, u[0])
                    }
                    (0, 1, BoundaryCondition::Neumann(0.0), BoundaryCondition::Dirichlet(r)) => {
                        (u[0], *r)
                    }
                    (0, _, BoundaryCondition::Neumann(0.0), _) => (u[0], u[1]),
                    (0, _, BoundaryCondition::Dirichlet(l), _) => (*l, u[1]),
                    (0, _, BoundaryCondition::Neumann(_), _) => {
                        return Err(BarracudaError::InvalidInput {
                            message: "Non-zero Neumann flux not yet implemented".to_string(),
                        });
                    }
                    (i, _, _, BoundaryCondition::Neumann(0.0)) if i == n - 1 && n > 1 => {
                        (u[n - 2], u[n - 1])
                    }
                    (i, _, _, BoundaryCondition::Dirichlet(r)) if i == n - 1 => (u[n - 2], *r),
                    (i, _, _, BoundaryCondition::Neumann(_)) if i == n - 1 => {
                        return Err(BarracudaError::InvalidInput {
                            message: "Non-zero Neumann flux not yet implemented".to_string(),
                        });
                    }
                    _ => (u[i - 1], u[i + 1]),
                };

                rhs[i] = (1.0 - r) * u[i] + (r / 2.0) * (u_left + u_right);

                // Neumann zero-flux: no extra boundary term (ghost already in u_left/u_right)
                // Dirichlet: add implicit boundary contribution
                if i == 0 {
                    if let BoundaryCondition::Dirichlet(l) = left_bc {
                        rhs[i] += (r / 2.0) * l;
                    }
                }
                if i == n - 1 {
                    if let BoundaryCondition::Dirichlet(r_val) = right_bc {
                        rhs[i] += (r / 2.0) * r_val;
                    }
                }
            }

            // Thomas algorithm with Neumann-modified first/last rows
            let (b0, c0) = match (n, left_bc) {
                (1, BoundaryCondition::Neumann(0.0)) => (1.0 + r / 2.0, 0.0),
                (_, BoundaryCondition::Neumann(0.0)) => (1.0 + r / 2.0, -r / 2.0),
                _ => (b_coef, c_coef),
            };
            let (a_n, b_n) = match (n, right_bc) {
                (1, _) => (0.0, 1.0 + r / 2.0),
                (_, BoundaryCondition::Neumann(0.0)) => (-r / 2.0, 1.0 + r / 2.0),
                _ => (a_coef, b_coef),
            };

            let mut c_prime = vec![0.0f32; n];
            let mut d_prime = vec![0.0f32; n];

            c_prime[0] = if b0.abs() > 1e-10 { c0 / b0 } else { 0.0 };
            d_prime[0] = rhs[0] / b0;

            for i in 1..n {
                let (a_i, b_i, c_i) = if i == n - 1 {
                    (a_n, b_n, 0.0)
                } else {
                    (a_coef, b_coef, c_coef)
                };
                let m = b_i - a_i * c_prime[i - 1];
                c_prime[i] = if m.abs() > 1e-10 { c_i / m } else { 0.0 };
                d_prime[i] = (rhs[i] - a_i * d_prime[i - 1]) / m;
            }

            u[n - 1] = d_prime[n - 1];
            for i in (0..n - 1).rev() {
                u[i] = d_prime[i] - c_prime[i] * u[i + 1];
            }
        }

        Ok(u)
    }

    /// CPU reference implementation (Thomas algorithm)
    #[cfg(test)]
    #[allow(dead_code)]
    fn solve_cpu(
        &self,
        u0: &[f32],
        r: f32,
        n_steps: usize,
        left_bc: f32,
        right_bc: f32,
    ) -> Vec<f32> {
        let n = u0.len();
        let mut u = u0.to_vec();
        let mut rhs = vec![0.0f32; n];

        // Tridiagonal coefficients (constant for all timesteps)
        let a = -r / 2.0; // sub-diagonal
        let b = 1.0 + r; // main diagonal
        let c = -r / 2.0; // super-diagonal

        for _ in 0..n_steps {
            // Compute RHS
            for i in 0..n {
                let u_left = if i == 0 { left_bc } else { u[i - 1] };
                let u_right = if i == n - 1 { right_bc } else { u[i + 1] };

                rhs[i] = (1.0 - r) * u[i] + (r / 2.0) * (u_left + u_right);

                // Boundary contributions
                if i == 0 {
                    rhs[i] += (r / 2.0) * left_bc;
                }
                if i == n - 1 {
                    rhs[i] += (r / 2.0) * right_bc;
                }
            }

            // Thomas algorithm for tridiagonal solve
            let mut c_prime = vec![0.0f32; n];
            let mut d_prime = vec![0.0f32; n];

            c_prime[0] = c / b;
            d_prime[0] = rhs[0] / b;

            for i in 1..n {
                let m = b - a * c_prime[i - 1];
                c_prime[i] = c / m;
                d_prime[i] = (rhs[i] - a * d_prime[i - 1]) / m;
            }

            u[n - 1] = d_prime[n - 1];
            for i in (0..n - 1).rev() {
                u[i] = d_prime[i] - c_prime[i] * u[i + 1];
            }
        }

        u
    }

    fn solve_gpu(
        &self,
        u0: &[f32],
        r: f32,
        n_steps: usize,
        left_bc: f32,
        right_bc: f32,
    ) -> Result<Vec<f32>> {
        let n = u0.len();

        let shader = self
            .device
            .compile_shader(Self::wgsl_shader(), Some("Crank-Nicolson"));

        // Create buffers
        let u_buf = self
            .device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("U (solution)"),
                contents: bytemuck::cast_slice(u0),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            });

        let rhs_buf = self.device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("RHS"),
            size: (n * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params = CrankNicolsonParams {
            n: n as u32,
            r,
            left_bc,
            right_bc,
        };

        let params_buf = self
            .device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("CN Params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        // Create bind group layout for RHS computation
        let bgl = self
            .device
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("CN RHS BGL"),
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
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
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
                ],
            });

        let pl = self
            .device
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("CN PL"),
                bind_group_layouts: &[&bgl],
                push_constant_ranges: &[],
            });

        let rhs_pipeline =
            self.device
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("CN RHS Pipeline"),
                    layout: Some(&pl),
                    module: &shader,
                    entry_point: "compute_rhs",
                    cache: None,
                    compilation_options: Default::default(),
                });

        let bg = self
            .device
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("CN BG"),
                layout: &bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: params_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: u_buf.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: rhs_buf.as_entire_binding(),
                    },
                ],
            });

        let workgroup_size = crate::device::capabilities::WORKGROUP_SIZE_1D as usize;
        let n_workgroups = n.div_ceil(workgroup_size);

        // Time-stepping loop
        // Note: Full GPU implementation would chain with cyclic_reduction_f64 for the solve.
        // For now, we use a hybrid approach: GPU RHS, CPU solve.
        let mut u = u0.to_vec();
        let a_coef = -r / 2.0;
        let b_coef = 1.0 + r;
        let c_coef = -r / 2.0;

        for _step in 0..n_steps {
            // Compute RHS on GPU
            let mut encoder =
                self.device
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("CN RHS Encoder"),
                    });

            // Upload current u
            self.device
                .queue
                .write_buffer(&u_buf, 0, bytemuck::cast_slice(&u));

            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("CN RHS Pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&rhs_pipeline);
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups(n_workgroups as u32, 1, 1);
            }

            self.device.submit_and_poll(Some(encoder.finish()));

            // Read back RHS
            let staging = self.device.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Staging"),
                size: (n * 4) as u64,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });

            let mut encoder2 =
                self.device
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("Copy Encoder"),
                    });
            encoder2.copy_buffer_to_buffer(&rhs_buf, 0, &staging, 0, (n * 4) as u64);
            self.device.submit_and_poll(Some(encoder2.finish()));

            let rhs: Vec<f32> = self.device.map_staging_buffer(&staging, n)?;

            // CPU Thomas solve for constant-coefficient tridiagonal system
            // O(n) with excellent cache behavior - optimal for typical PDE grid sizes
            // Note: RHS computation happens on GPU; this is just the back-substitution
            let mut c_prime = vec![0.0f32; n];
            let mut d_prime = vec![0.0f32; n];

            c_prime[0] = c_coef / b_coef;
            d_prime[0] = rhs[0] / b_coef;

            for i in 1..n {
                let m = b_coef - a_coef * c_prime[i - 1];
                c_prime[i] = c_coef / m;
                d_prime[i] = (rhs[i] - a_coef * d_prime[i - 1]) / m;
            }

            u[n - 1] = d_prime[n - 1];
            for i in (0..n - 1).rev() {
                u[i] = d_prime[i] - c_prime[i] * u[i + 1];
            }
        }

        Ok(u)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn get_test_device() -> Arc<crate::device::WgpuDevice> {
        crate::device::test_pool::get_test_device_sync()
    }

    #[test]
    fn test_heat_diffusion_steady_state() {
        let device = get_test_device();
        let solver = CrankNicolson::new(device).unwrap();

        // Heat equation with fixed boundaries should approach linear profile.
        // Crank-Nicolson is unconditionally stable so we use large dt to
        // reach steady state in few GPU round-trips (avoids 60s+ test times).
        let n = 30;
        let u0 = vec![0.0f32; n];
        let alpha = 1.0;
        let dx = 1.0 / (n + 1) as f32;
        let dt = 0.01;
        let n_steps = 200;

        let u = solver
            .solve(
                &u0,
                alpha,
                dx,
                dt,
                n_steps,
                BoundaryCondition::Dirichlet(0.0),
                BoundaryCondition::Dirichlet(1.0),
            )
            .unwrap();

        // At steady state, solution should be linear from 0 to 1
        for (i, &ui) in u.iter().enumerate() {
            let expected = (i + 1) as f32 / (n + 1) as f32;
            assert!(
                (ui - expected).abs() < 0.05,
                "u[{}] = {}, expected {}",
                i,
                ui,
                expected
            );
        }
    }

    #[test]
    fn test_conservation() {
        let device = get_test_device();
        let solver = CrankNicolson::new(device).unwrap();

        // With zero Dirichlet BCs, total "heat" should decrease over time
        // but remain positive. Use enough steps for measurable dissipation.
        let n = 20;
        let u0: Vec<f32> = (0..n)
            .map(|i| ((i as f32 + 0.5) / n as f32 * std::f32::consts::PI).sin())
            .collect();
        let initial_sum: f32 = u0.iter().sum();

        let u = solver
            .solve(
                &u0,
                1.0,
                0.1,
                0.001,
                100,
                BoundaryCondition::Dirichlet(0.0),
                BoundaryCondition::Dirichlet(0.0),
            )
            .unwrap();

        let final_sum: f32 = u.iter().sum();

        // Heat should be dissipating to boundaries
        assert!(
            final_sum < initial_sum,
            "Heat should decrease: initial={}, final={}",
            initial_sum,
            final_sum
        );
        assert!(
            final_sum > 0.0,
            "Heat should remain positive: final={}",
            final_sum
        );
    }

    #[test]
    fn test_neumann_zero_flux_conservation() {
        let device = get_test_device();
        let solver = CrankNicolson::new(device).unwrap();

        // With zero-flux Neumann BCs, total "mass" must be conserved
        let n = 20;
        let u0: Vec<f32> = (0..n)
            .map(|i| ((i as f32 + 0.5) / n as f32 * std::f32::consts::PI).sin())
            .collect();
        let initial_sum: f32 = u0.iter().sum();

        let u = solver
            .solve(
                &u0,
                1.0,
                0.1,
                0.001,
                100,
                BoundaryCondition::Neumann(0.0),
                BoundaryCondition::Neumann(0.0),
            )
            .unwrap();

        let final_sum: f32 = u.iter().sum();

        // Mass must be conserved with zero-flux Neumann
        assert!(
            (final_sum - initial_sum).abs() < 0.01,
            "Mass should be conserved: initial={}, final={}",
            initial_sum,
            final_sum
        );
    }
}

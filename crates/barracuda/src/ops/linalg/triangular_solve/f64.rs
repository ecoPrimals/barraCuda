// SPDX-License-Identifier: AGPL-3.0-or-later
//! f64 triangular solve (science-grade precision)

use super::f32::TriangularSolve;
use crate::error::{BarracudaError, Result};

/// Triangular solve for f64 data (GPU)
///
/// **Deep Debt Evolution (Feb 16, 2026)**:
/// - Science-grade f64 precision
/// - Native Vulkan fp64 arithmetic
/// - WGSL as unified math language
/// - Includes transpose solve for Cholesky pipeline
pub struct TriangularSolveF64;

impl TriangularSolveF64 {
    /// Solve triangular system L·x = b or U·x = b with f64 precision
    /// # Arguments
    /// * `device` - GPU device (Arc-wrapped)
    /// * `matrix` - Triangular matrix (row-major f64)
    /// * `rhs` - Right-hand side vector b
    /// * `n` - Matrix/vector dimension
    /// * `lower` - true for lower triangular (forward), false for upper (backward)
    /// * `unit_diagonal` - true if diagonal is implicitly 1.0
    /// # Returns
    /// Solution vector x
    /// # Errors
    /// Returns [`Err`] if `matrix.len() != n * n` or `rhs.len() != n` (invalid dimensions),
    /// if buffer allocation fails, or if GPU dispatch/readback fails (e.g., device lost).
    pub fn execute(
        device: std::sync::Arc<crate::device::WgpuDevice>,
        matrix: &[f64],
        rhs: &[f64],
        n: usize,
        lower: bool,
        unit_diagonal: bool,
    ) -> Result<Vec<f64>> {
        if matrix.len() != n * n {
            return Err(BarracudaError::InvalidShape {
                expected: vec![n * n],
                actual: vec![matrix.len()],
            });
        }
        if rhs.len() != n {
            return Err(BarracudaError::InvalidShape {
                expected: vec![n],
                actual: vec![rhs.len()],
            });
        }

        // Create buffers
        let matrix_buffer = device.create_buffer_f64(n * n)?;
        device
            .queue
            .write_buffer(&matrix_buffer, 0, bytemuck::cast_slice(matrix));

        let rhs_buffer = device.create_buffer_f64(n)?;
        device
            .queue
            .write_buffer(&rhs_buffer, 0, bytemuck::cast_slice(rhs));

        let solution_buffer = device.create_buffer_f64(n)?;

        // Params: n, is_lower, is_unit, _pad
        let is_lower = u32::from(lower);
        let is_unit = u32::from(unit_diagonal);
        let params_buffer = device.create_uniform_buffer(
            "TriangularSolve F64 Params",
            &[n as u32, is_lower, is_unit, 0u32],
        );

        // Create bind group layout
        let bind_group_layout =
            device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("TriangularSolve F64 BGL"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
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
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });

        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("TriangularSolve F64 BG"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: matrix_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: rhs_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: solution_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let shader = device.compile_shader(
            TriangularSolve::wgsl_shader_f64(),
            Some("TriangularSolve F64"),
        );

        let pipeline_layout =
            device
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("TriangularSolve F64 PL"),
                    bind_group_layouts: &[&bind_group_layout],
                    immediate_size: 0,
                });

        let pipeline = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("TriangularSolve F64 Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("triangular_solve_f64"),
                cache: None,
                compilation_options: Default::default(),
            });

        let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("TriangularSolve F64 Encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("TriangularSolve F64 Pass"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&bind_group), &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }

        device.submit_and_poll(Some(encoder.finish()));

        crate::utils::read_buffer_f64(&device, &solution_buffer, n)
    }

    /// Solve L·x = b (forward substitution) with f64
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn forward(
        device: std::sync::Arc<crate::device::WgpuDevice>,
        matrix: &[f64],
        rhs: &[f64],
        n: usize,
    ) -> Result<Vec<f64>> {
        Self::execute(device, matrix, rhs, n, true, false)
    }

    /// Solve U·x = b (backward substitution) with f64
    /// # Errors
    /// Returns [`Err`] if [`execute`](Self::execute) fails (invalid dimensions, buffer allocation,
    /// or GPU dispatch/readback failure).
    pub fn backward(
        device: std::sync::Arc<crate::device::WgpuDevice>,
        matrix: &[f64],
        rhs: &[f64],
        n: usize,
    ) -> Result<Vec<f64>> {
        Self::execute(device, matrix, rhs, n, false, false)
    }

    /// Solve Lᵀ·x = b using stored L (transpose solve).
    /// This is the second step of Cholesky solve:
    /// 1. L·z = b (forward)
    /// 2. Lᵀ·x = z (this method)
    ///    The matrix is accessed as transpose internally.
    /// # Errors
    /// Returns [`Err`] if `matrix.len() != n * n` or `rhs.len() != n` (invalid dimensions),
    /// if buffer allocation fails, or if GPU dispatch/readback fails (e.g., device lost).
    pub fn solve_transpose(
        device: std::sync::Arc<crate::device::WgpuDevice>,
        matrix: &[f64],
        rhs: &[f64],
        n: usize,
    ) -> Result<Vec<f64>> {
        if matrix.len() != n * n {
            return Err(BarracudaError::InvalidShape {
                expected: vec![n * n],
                actual: vec![matrix.len()],
            });
        }
        if rhs.len() != n {
            return Err(BarracudaError::InvalidShape {
                expected: vec![n],
                actual: vec![rhs.len()],
            });
        }

        // Create buffers
        let matrix_buffer = device.create_buffer_f64(n * n)?;
        device
            .queue
            .write_buffer(&matrix_buffer, 0, bytemuck::cast_slice(matrix));

        let rhs_buffer = device.create_buffer_f64(n)?;
        device
            .queue
            .write_buffer(&rhs_buffer, 0, bytemuck::cast_slice(rhs));

        let solution_buffer = device.create_buffer_f64(n)?;

        // Params: n, is_lower=1 (but we use transpose kernel), is_unit=0, _pad
        let params_buffer = device.create_uniform_buffer(
            "TriangularSolve Transpose F64 Params",
            &[n as u32, 1u32, 0u32, 0u32],
        );

        // Create bind group layout
        let bind_group_layout =
            device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("TriangularSolve Transpose F64 BGL"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
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
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });

        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("TriangularSolve Transpose F64 BG"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: matrix_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: rhs_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: solution_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let shader = device.compile_shader(
            TriangularSolve::wgsl_shader_f64(),
            Some("TriangularSolve Transpose F64"),
        );

        let pipeline_layout =
            device
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("TriangularSolve Transpose F64 PL"),
                    bind_group_layouts: &[&bind_group_layout],
                    immediate_size: 0,
                });

        let pipeline = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("TriangularSolve Transpose F64 Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("triangular_solve_transpose_f64"),
                cache: None,
                compilation_options: Default::default(),
            });

        let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("TriangularSolve Transpose F64 Encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("TriangularSolve Transpose F64 Pass"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&bind_group), &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }

        device.submit_and_poll(Some(encoder.finish()));

        crate::utils::read_buffer_f64(&device, &solution_buffer, n)
    }

    /// Complete Cholesky solve: Given L from Cholesky(A), solve A·x = b
    /// Performs:
    /// 1. L·z = b (forward substitution)
    /// 2. Lᵀ·x = z (backward with transpose)
    /// # Arguments
    /// * `device` - GPU device (Arc-wrapped)
    /// * `l_matrix` - Lower triangular Cholesky factor L
    /// * `b` - Right-hand side vector
    /// * `n` - System dimension
    /// # Returns
    /// Solution vector x where A·x = b
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn cholesky_solve(
        device: std::sync::Arc<crate::device::WgpuDevice>,
        l_matrix: &[f64],
        b: &[f64],
        n: usize,
    ) -> Result<Vec<f64>> {
        // Step 1: L·z = b (forward)
        let z = Self::forward(device.clone(), l_matrix, b, n)?;

        // Step 2: Lᵀ·x = z (transpose solve)
        Self::solve_transpose(device, l_matrix, &z, n)
    }
}

//! f32 triangular solve (Tensor-based) and Tensor extension

use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

/// Triangular solve operation
///
/// Solves L·x = b (forward) or Uᵀ·x = b (backward)
pub struct TriangularSolve {
    matrix: Tensor, // Triangular matrix (L or U)
    rhs: Tensor,    // Right-hand side vector b
    lower: bool,    // true for lower triangular (forward), false for upper (backward)
}

impl TriangularSolve {
    /// Create new triangular solve operation
    ///
    /// # Arguments
    /// * `matrix` - Triangular matrix [N, N]
    /// * `rhs` - Right-hand side vector [N]
    /// * `lower` - true for lower triangular (forward substitution)
    ///
    /// # Deep Debt Compliance
    /// - No hardcoded sizes (runtime N)
    /// - No unsafe blocks
    /// - Agnostic design (works with any triangular system)
    pub fn new(matrix: Tensor, rhs: Tensor, lower: bool) -> Self {
        Self { matrix, rhs, lower }
    }

    /// Create forward substitution: L·x = b
    pub fn forward(matrix: Tensor, rhs: Tensor) -> Self {
        Self::new(matrix, rhs, true)
    }

    /// Create backward substitution: Uᵀ·x = b
    pub fn backward(matrix: Tensor, rhs: Tensor) -> Self {
        Self::new(matrix, rhs, false)
    }

    fn wgsl_shader() -> &'static str {
        include_str!("../../../shaders/linalg/triangular_solve.wgsl")
    }

    /// Shared with f64 module for transpose solve
    pub(crate) fn wgsl_shader_f64() -> &'static str {
        include_str!("../../../shaders/linalg/triangular_solve_f64.wgsl")
    }

    /// Execute triangular solve on GPU
    ///
    /// # Returns
    /// Solution vector x
    ///
    /// # Errors
    /// - Returns error if matrix is not square
    /// - Returns error if rhs size doesn't match matrix
    /// - Returns zero vector if matrix is singular
    ///
    /// # Deep Debt Compliance
    /// - Pure WGSL execution (no CPU fallback)
    /// - Capability-based workgroup dispatch
    /// - Safe buffer management
    pub fn execute(self) -> Result<Tensor> {
        let device = self.matrix.device();
        let matrix_shape = self.matrix.shape();
        let rhs_shape = self.rhs.shape();

        // Validate square matrix
        if matrix_shape.len() != 2 || matrix_shape[0] != matrix_shape[1] {
            return Err(BarracudaError::InvalidShape {
                expected: vec![0, 0],
                actual: matrix_shape.to_vec(),
            });
        }

        let n = matrix_shape[0];

        // Validate rhs is a vector of length n
        if rhs_shape.len() != 1 || rhs_shape[0] != n {
            return Err(BarracudaError::InvalidShape {
                expected: vec![n],
                actual: rhs_shape.to_vec(),
            });
        }

        // Create output buffer for solution vector x
        let solution_buffer = device.create_buffer_f32(n)?;

        // Create params buffer with matrix size and substitution type
        let is_lower = if self.lower { 1u32 } else { 0u32 };
        let params_buffer =
            device.create_uniform_buffer("TriangularSolve Params", &[n as u32, is_lower]);

        // Create bind group layout
        let bind_group_layout =
            device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("TriangularSolve BGL"),
                    entries: &[
                        // Triangular matrix (L or U)
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
                        // Right-hand side vector b
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
                        // Solution vector x
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
                        // Parameters (n, is_lower)
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

        // Create bind group
        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("TriangularSolve BG"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.matrix.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.rhs.buffer().as_entire_binding(),
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

        // Compile shader
        let shader = device.compile_shader(Self::wgsl_shader(), Some("TriangularSolve"));

        // Create pipeline
        let pipeline_layout =
            device
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("TriangularSolve PL"),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });

        let pipeline = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("TriangularSolve Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: "main",
                cache: None,
                compilation_options: Default::default(),
            });

        // Create command encoder
        let mut encoder = device
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("TriangularSolve Encoder"),
            });

        // Execute compute pass
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("TriangularSolve Pass"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);

            // Deep Debt Evolution: Sequential algorithm (dependency chain)
            // Single-threaded execution (workgroup_size=1 in shader)
            pass.dispatch_workgroups(1, 1, 1);
        }

        device.submit_and_poll(Some(encoder.finish()));

        let output_data = crate::utils::read_buffer(device, &solution_buffer, n)?;
        Ok(Tensor::new(output_data, vec![n], device.clone()))
    }
}

/// Tensor extension for triangular solve
impl Tensor {
    /// Solve L·x = b (forward substitution)
    ///
    /// # Arguments
    /// * `rhs` - Right-hand side vector b
    ///
    /// # Returns
    /// Solution vector x
    ///
    /// # Example
    /// ```ignore
    /// let l = tensor.cholesky()?;  // Get lower triangular L
    /// let x = l.solve_triangular_forward(&b)?;
    /// ```
    pub fn solve_triangular_forward(&self, rhs: &Tensor) -> Result<Tensor> {
        TriangularSolve::forward(self.clone(), rhs.clone()).execute()
    }

    /// Solve Uᵀ·x = b (backward substitution)
    ///
    /// # Arguments
    /// * `rhs` - Right-hand side vector b
    ///
    /// # Returns
    /// Solution vector x
    pub fn solve_triangular_backward(&self, rhs: &Tensor) -> Result<Tensor> {
        TriangularSolve::backward(self.clone(), rhs.clone()).execute()
    }

    /// Solve triangular system L·x = b or Uᵀ·x = b
    ///
    /// # Arguments
    /// * `rhs` - Right-hand side vector b
    /// * `lower` - true for lower triangular, false for upper
    pub fn solve_triangular(&self, rhs: &Tensor, lower: bool) -> Result<Tensor> {
        TriangularSolve::new(self.clone(), rhs.clone(), lower).execute()
    }
}

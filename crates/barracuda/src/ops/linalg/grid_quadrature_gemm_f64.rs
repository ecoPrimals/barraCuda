// SPDX-License-Identifier: AGPL-3.0-or-later
//! Grid Quadrature GEMM (f64) — GPU Hamiltonian Construction
//!
//! Computes batched Hamiltonian matrices via numerical quadrature:
//!
//! ```text
//! H[b,i,j] = Σ_k φ[b,i,k] * W[b,k] * φ[b,j,k] * quad_weights[k]
//! ```
//!
//! This is the core operation for constructing Hamiltonian matrices
//! from basis functions evaluated on a numerical grid.
//!
//! **Use cases**:
//! - HFB Hamiltonian construction
//! - DFT matrix assembly
//! - Any basis function integral on a grid
//!
//! **Deep Debt Principles**:
//! - Pure WGSL implementation (hardware-agnostic)
//! - Full f64 precision
//! - Safe Rust wrapper (no unsafe code)
//!
//! # Example
//!
//! ```rust,ignore
//! let device = WgpuDevice::new().await?;
//! let gemm = GridQuadratureGemm::new(device.clone(), 791, 12, 100)?;
//!
//! // phi[batch, n, grid] - basis functions
//! // w[batch, grid] - weight function (potential etc.)
//! // quad_weights[grid] - quadrature weights
//! let h = gemm.execute(&phi, &w, &quad_weights).await?;
//! // h[batch, n, n] - Hamiltonian matrices
//! ```

use crate::device::WgpuDevice;
use crate::error::{BarracudaError, Result};
use bytemuck::{Pod, Zeroable};
use std::sync::Arc;

/// Parameters for grid quadrature GEMM shader
#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable)]
struct QuadParams {
    batch_size: u32,
    n: u32,
    grid_size: u32,
    _pad: u32,
}

/// GPU-accelerated grid quadrature for Hamiltonian construction
///
/// Computes `H[b,i,j] = Σ_k φ[b,i,k] * W[b,k] * φ[b,j,k] * quad_weights[k]`
pub struct GridQuadratureGemm {
    device: Arc<WgpuDevice>,
    batch_size: usize,
    n: usize,         // Basis size (matrix dimension)
    grid_size: usize, // Number of quadrature points
    symmetric: bool,  // Use symmetric optimization
}

impl GridQuadratureGemm {
    /// Create a new grid quadrature GEMM operator
    ///
    /// # Arguments
    /// * `device` - WgpuDevice
    /// * `batch_size` - Number of matrices to compute
    /// * `n` - Basis size (matrix dimension)
    /// * `grid_size` - Number of quadrature points
    pub fn new(
        device: Arc<WgpuDevice>,
        batch_size: usize,
        n: usize,
        grid_size: usize,
    ) -> Result<Self> {
        if batch_size == 0 || n == 0 || grid_size == 0 {
            return Err(BarracudaError::InvalidInput {
                message: "batch_size, n, and grid_size must be positive".to_string(),
            });
        }
        Ok(Self {
            device,
            batch_size,
            n,
            grid_size,
            symmetric: false,
        })
    }

    /// Enable symmetric optimization (only compute upper triangle)
    ///
    /// Use this when H is symmetric, cuts computation in half.
    pub fn with_symmetric(mut self, symmetric: bool) -> Self {
        self.symmetric = symmetric;
        self
    }

    fn wgsl_shader() -> &'static str {
        include_str!("../../shaders/linalg/grid_quadrature_gemm_f64.wgsl")
    }

    /// Execute grid quadrature GEMM
    ///
    /// # Arguments
    /// * `phi` - Basis functions [batch, n, grid]
    /// * `w` - Weight function [batch, grid]
    /// * `quad_weights` - Quadrature weights [grid]
    ///
    /// # Returns
    /// Hamiltonian matrices [batch, n, n]
    pub fn execute(&self, phi: &[f64], w: &[f64], quad_weights: &[f64]) -> Result<Vec<f64>> {
        // Validate dimensions
        let expected_phi_len = self.batch_size * self.n * self.grid_size;
        let expected_w_len = self.batch_size * self.grid_size;

        if phi.len() != expected_phi_len {
            return Err(BarracudaError::InvalidInput {
                message: format!(
                    "phi length {} doesn't match expected {} (batch={}, n={}, grid={})",
                    phi.len(),
                    expected_phi_len,
                    self.batch_size,
                    self.n,
                    self.grid_size
                ),
            });
        }
        if w.len() != expected_w_len {
            return Err(BarracudaError::InvalidInput {
                message: format!(
                    "w length {} doesn't match expected {} (batch={}, grid={})",
                    w.len(),
                    expected_w_len,
                    self.batch_size,
                    self.grid_size
                ),
            });
        }
        if quad_weights.len() != self.grid_size {
            return Err(BarracudaError::InvalidInput {
                message: format!(
                    "quad_weights length {} doesn't match grid_size {}",
                    quad_weights.len(),
                    self.grid_size
                ),
            });
        }

        let shader = self
            .device
            .compile_shader_f64(Self::wgsl_shader(), Some("Grid Quadrature GEMM f64"));

        // Choose entry point based on grid size and symmetry
        let entry_point = if self.symmetric {
            "grid_quadrature_gemm_symmetric"
        } else if self.grid_size <= 256 {
            "grid_quadrature_gemm_small"
        } else {
            "grid_quadrature_gemm"
        };

        // Create bind group layout
        let bgl = self
            .device
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("GridQuadGEMM BGL"),
                entries: &[
                    // phi [batch, n, grid]
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
                    // w [batch, grid]
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
                    // quad_weights [grid]
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // output [batch, n, n]
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
                    // params
                    wgpu::BindGroupLayoutEntry {
                        binding: 4,
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

        let pl = self
            .device
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("GridQuadGEMM PL"),
                bind_group_layouts: &[&bgl],
                push_constant_ranges: &[],
            });

        let pipeline =
            self.device
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(entry_point),
                    layout: Some(&pl),
                    module: &shader,
                    entry_point,
                    cache: None,
                    compilation_options: Default::default(),
                });

        // Create buffers
        let phi_bytes: Vec<u8> = phi.iter().flat_map(|v| v.to_le_bytes()).collect();
        let phi_buffer = self
            .device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("GridQuadGEMM phi"),
                contents: &phi_bytes,
                usage: wgpu::BufferUsages::STORAGE,
            });

        let w_bytes: Vec<u8> = w.iter().flat_map(|v| v.to_le_bytes()).collect();
        let w_buffer = self
            .device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("GridQuadGEMM w"),
                contents: &w_bytes,
                usage: wgpu::BufferUsages::STORAGE,
            });

        let qw_bytes: Vec<u8> = quad_weights.iter().flat_map(|v| v.to_le_bytes()).collect();
        let qw_buffer = self
            .device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("GridQuadGEMM quad_weights"),
                contents: &qw_bytes,
                usage: wgpu::BufferUsages::STORAGE,
            });

        let output_size = self.batch_size * self.n * self.n;
        let output_buffer = self.device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GridQuadGEMM output"),
            size: (output_size * 8) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params = QuadParams {
            batch_size: self.batch_size as u32,
            n: self.n as u32,
            grid_size: self.grid_size as u32,
            _pad: 0,
        };
        let params_buffer = self
            .device
            .create_uniform_buffer("GridQuadGEMM params", &params);

        // Create bind group
        let bg = self
            .device
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("GridQuadGEMM BG"),
                layout: &bgl,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: phi_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: w_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: qw_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: output_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: params_buffer.as_entire_binding(),
                    },
                ],
            });

        // Calculate workgroup dimensions
        let (wg_x, wg_y) = if self.symmetric {
            let n_upper = (self.n * (self.n + 1)) / 2;
            (n_upper as u32, self.batch_size as u32)
        } else {
            ((self.n * self.n) as u32, self.batch_size as u32)
        };

        // Execute
        {
            let mut encoder = self
                .device
                .create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
                    label: Some("GridQuadGEMM encoder"),
                });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("GridQuadGEMM pass"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &bg, &[]);
                pass.dispatch_workgroups(wg_x, wg_y, 1);
            }
            self.device.submit_and_poll(Some(encoder.finish()));
        }

        // Read back results
        self.device.read_f64_buffer(&output_buffer, output_size)
    }
}

#[expect(clippy::unwrap_used, reason = "tests")]
#[cfg(test)]
mod tests {
    use super::*;

    const GRID_QUADRATURE_SHADER: &str =
        include_str!("../../shaders/linalg/grid_quadrature_gemm_f64.wgsl");

    #[test]
    fn quad_params_layout() {
        assert_eq!(std::mem::size_of::<QuadParams>(), 16);
    }

    #[test]
    fn grid_quadrature_gemm_shader_source_valid() {
        assert!(!GRID_QUADRATURE_SHADER.is_empty());
        assert!(
            GRID_QUADRATURE_SHADER.contains("fn main")
                || GRID_QUADRATURE_SHADER.contains("@compute")
        );
    }

    #[tokio::test]
    async fn test_grid_quadrature_gemm_identity() {
        let Some(device) = crate::device::test_pool::get_test_device_if_f64_gpu_available().await
        else {
            return;
        };

        // Simple test: phi = identity-like, w = 1, quad_weights = 1
        // Should give identity matrix
        let batch = 1;
        let n = 4;
        let grid = 4;

        // phi[0, i, k] = 1 if i==k else 0
        let mut phi = vec![0.0; batch * n * grid];
        for i in 0..n {
            phi[i * grid + i] = 1.0;
        }

        let w = vec![1.0; batch * grid];
        let quad_weights = vec![1.0; grid];

        let gemm = GridQuadratureGemm::new(device, batch, n, grid).unwrap();
        let result = gemm.execute(&phi, &w, &quad_weights).unwrap();

        // Should be identity matrix
        for i in 0..n {
            for j in 0..n {
                let expected = if i == j { 1.0 } else { 0.0 };
                let actual = result[i * n + j];
                assert!(
                    (actual - expected).abs() < 1e-10,
                    "H[{},{}] = {}, expected {}",
                    i,
                    j,
                    actual,
                    expected
                );
            }
        }
    }

    #[tokio::test]
    async fn test_grid_quadrature_gemm_weighted() {
        let Some(device) = crate::device::test_pool::get_test_device_if_f64_gpu_available().await
        else {
            return;
        };

        let batch = 1;
        let n = 2;
        let grid = 3;

        // phi[0, 0, :] = [1, 2, 3]
        // phi[0, 1, :] = [4, 5, 6]
        let phi = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        // w[0, :] = [1, 1, 1]
        let w = vec![1.0, 1.0, 1.0];

        // quad_weights = [1, 1, 1]
        let quad_weights = vec![1.0, 1.0, 1.0];

        // H[0,0] = 1*1*1 + 2*1*2 + 3*1*3 = 1 + 4 + 9 = 14
        // H[0,1] = 1*1*4 + 2*1*5 + 3*1*6 = 4 + 10 + 18 = 32
        // H[1,0] = 4*1*1 + 5*1*2 + 6*1*3 = 4 + 10 + 18 = 32
        // H[1,1] = 4*1*4 + 5*1*5 + 6*1*6 = 16 + 25 + 36 = 77

        let gemm = GridQuadratureGemm::new(device, batch, n, grid).unwrap();
        let result = gemm.execute(&phi, &w, &quad_weights).unwrap();

        assert!(
            (result[0] - 14.0).abs() < 1e-10,
            "H[0,0] = {}, expected 14",
            result[0]
        );
        assert!(
            (result[1] - 32.0).abs() < 1e-10,
            "H[0,1] = {}, expected 32",
            result[1]
        );
        assert!(
            (result[2] - 32.0).abs() < 1e-10,
            "H[1,0] = {}, expected 32",
            result[2]
        );
        assert!(
            (result[3] - 77.0).abs() < 1e-10,
            "H[1,1] = {}, expected 77",
            result[3]
        );
    }

    #[tokio::test]
    async fn test_grid_quadrature_gemm_batched() {
        let Some(device) = crate::device::test_pool::get_test_device_if_f64_gpu_available().await
        else {
            return;
        };

        let batch = 10;
        let n = 8;
        let grid = 50;

        // Random-ish data
        let phi: Vec<f64> = (0..batch * n * grid)
            .map(|i| ((i as f64) * 0.01).sin())
            .collect();
        let w: Vec<f64> = (0..batch * grid)
            .map(|i| ((i as f64) * 0.02).cos())
            .collect();
        let quad_weights: Vec<f64> = (0..grid).map(|i| 1.0 / (1.0 + i as f64 * 0.1)).collect();

        let gemm = GridQuadratureGemm::new(device.clone(), batch, n, grid).unwrap();
        let result = gemm.execute(&phi, &w, &quad_weights).unwrap();

        // Verify output size
        assert_eq!(result.len(), batch * n * n);

        // Verify symmetry-ish (H should be symmetric if phi basis is real)
        for b in 0..batch {
            for i in 0..n {
                for j in i..n {
                    let h_ij = result[b * n * n + i * n + j];
                    let h_ji = result[b * n * n + j * n + i];
                    assert!(
                        (h_ij - h_ji).abs() < 1e-10,
                        "Batch {}: H[{},{}] = {}, H[{},{}] = {}",
                        b,
                        i,
                        j,
                        h_ij,
                        j,
                        i,
                        h_ji
                    );
                }
            }
        }
    }
}

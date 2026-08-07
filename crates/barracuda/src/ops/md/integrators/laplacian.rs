// SPDX-License-Identifier: AGPL-3.0-or-later
//! Laplacian Stencil (7-point 3D)
//!
//! **Physics**: Finite difference approximation of ∇²u
//! **Use Case**: Diffusion, electrostatics (PPPM), wave equations
//!
//! **Deep Debt Compliance**:
//! - ✅ Pure WGSL shader  
//! - ✅ Periodic boundaries
//! - ✅ Zero unsafe code

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

/// Laplacian stencil operation (7-point 3D)
///
/// Computes ∇²u using central difference formula.
/// Includes periodic boundary conditions.
pub struct Laplacian {
    field: Tensor,     // [nx, ny, nz] input field
    grid_spacing: f32, // h (mesh spacing)
}

impl Laplacian {
    /// Creates a Laplacian stencil for a 3D field with given grid spacing.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if field is not 3D or `grid_spacing` is not positive.
    pub fn new(field: Tensor, grid_spacing: f32) -> Result<Self> {
        let shape = field.shape();
        if shape.len() != 3 {
            return Err(BarracudaError::InvalidShape {
                expected: vec![0, 0, 0],
                actual: shape.to_vec(),
            });
        }

        if grid_spacing <= 0.0 {
            return Err(BarracudaError::Device(
                "Grid spacing h must be positive".to_string(),
            ));
        }

        Ok(Self {
            field,
            grid_spacing,
        })
    }

    /// Execute Laplacian calculation
    ///
    /// # Returns
    /// Laplacian field [nx, ny, nz]
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.field.device();
        let shape = self.field.shape();
        let (nx, ny, nz) = (shape[0], shape[1], shape[2]);

        // Create output buffer
        let output_size = (nx * ny * nz * std::mem::size_of::<f32>()) as u64;
        let output_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Laplacian Output"),
            size: output_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            nx: u32,
            ny: u32,
            nz: u32,
            h_squared: f32,
        }

        let params = Params {
            nx: nx as u32,
            ny: ny as u32,
            nz: nz as u32,
            h_squared: self.grid_spacing * self.grid_spacing,
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Laplacian Params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let workgroups_x = (nx as u32).div_ceil(4);
        let workgroups_y = (ny as u32).div_ceil(4);
        let workgroups_z = (nz as u32).div_ceil(4);

        ComputeDispatch::new(device, "Laplacian")
            .shader(include_str!("laplacian.wgsl"), "main")
            .storage_read(0, self.field.buffer())
            .storage_rw(1, &output_buffer)
            .uniform(2, &params_buffer)
            .dispatch(workgroups_x, workgroups_y, workgroups_z)
            .submit()?;

        Ok(Tensor::from_buffer(
            output_buffer,
            vec![nx, ny, nz],
            device.clone(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    // Test un-ignored - issue was test code structure, not tensor implementation
    async fn test_laplacian_simple() {
        let device = crate::device::test_pool::get_test_device().await;

        // Simple 3x3x3 grid
        let (nx, ny, nz) = (3, 3, 3);
        let size = nx * ny * nz;

        // Set all values to same number, Laplacian should be zero everywhere
        let data = vec![1.0f32; size];

        let field_tensor = Tensor::from_data(&data, vec![nx, ny, nz], device).unwrap();

        // Verify input (explicit validation to prevent rustc optimization issues)
        let field_check = field_tensor.to_vec().unwrap();
        assert_eq!(field_check.len(), size, "Field size mismatch");

        // All values should be 1.0
        for (i, &val) in field_check.iter().enumerate() {
            assert_eq!(
                val, 1.0,
                "Input corrupted at index {i}: expected 1.0, got {val}"
            );
        }

        let laplacian = Laplacian::new(field_tensor, 1.0).unwrap();
        let result = laplacian.execute().unwrap();

        let lap_data = result.to_vec().unwrap();

        // For constant field, Laplacian should be zero everywhere
        // ∇²(constant) = 0
        for (i, &val) in lap_data.iter().enumerate() {
            assert!(
                val.abs() < 1e-5,
                "Index {i} Laplacian should be ~0, got {val}"
            );
        }
    }
}

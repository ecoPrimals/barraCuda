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

        let shader = device
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Laplacian Shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("laplacian.wgsl").into()),
            });

        let bind_group_layout =
            device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Laplacian BGL"),
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
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });

        let pipeline_layout =
            device
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Laplacian PL"),
                    bind_group_layouts: &[&bind_group_layout],
                    immediate_size: 0,
                });

        let pipeline = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Laplacian Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                cache: None,
                compilation_options: Default::default(),
            });

        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Laplacian BG"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.field.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("Laplacian Encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Laplacian Pass"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&bind_group), &[]);

            // 3D workgroup dispatch (4x4x4 threads per workgroup, 64 total)
            let workgroups_x = (nx as u32).div_ceil(4);
            let workgroups_y = (ny as u32).div_ceil(4);
            let workgroups_z = (nz as u32).div_ceil(4);
            pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z);
        }

        device.submit_and_poll(Some(encoder.finish()));

        Ok(Tensor::from_buffer(
            output_buffer,
            vec![nx, ny, nz],
            device.clone(),
        ))
    }
}

#[expect(clippy::unwrap_used, reason = "tests")]
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    // Test un-ignored - issue was test code structure, not tensor implementation
    async fn test_laplacian_simple() {
        let Some(device) = crate::device::test_pool::get_test_device_if_gpu_available().await
        else {
            return;
        };

        // Simple 3x3x3 grid
        let (nx, ny, nz) = (3, 3, 3);
        let size = nx * ny * nz;

        // Set all values to same number, Laplacian should be zero everywhere
        let data = vec![1.0f32; size];

        let field_tensor = Tensor::from_data(&data, vec![nx, ny, nz], device.clone()).unwrap();

        // Verify input (explicit validation to prevent rustc optimization issues)
        let field_check = field_tensor.to_vec().unwrap();
        assert_eq!(field_check.len(), size, "Field size mismatch");

        // All values should be 1.0
        for (i, &val) in field_check.iter().enumerate() {
            assert_eq!(
                val, 1.0,
                "Input corrupted at index {}: expected 1.0, got {}",
                i, val
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
                "Index {} Laplacian should be ~0, got {}",
                i,
                val
            );
        }

        println!("✅ Laplacian validated (constant field)");
    }
}

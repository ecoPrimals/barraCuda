// SPDX-License-Identifier: AGPL-3.0-or-later
//! AffineGrid - Affine Grid Generator
//!
//! **Deep Debt Principles**:
//! - ✅ Pure WGSL implementation
//! - ✅ Safe Rust wrapper (no unsafe code)
//! - ✅ Hardware-agnostic via WebGPU
//! - ✅ Complete implementation (production-ready)
//! - ✅ Modern idiomatic Rust (no traits, direct impl)
//!
//! Generates sampling grid for spatial transformer networks
//! Takes affine transformation matrix and produces coordinate grid
//!
//! Reference: "Spatial Transformer Networks" by Jaderberg et al. (2015)

use crate::device::{DeviceCapabilities, WorkloadType};
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct AffineGridParams {
    batch_size: u32,
    height: u32,
    width: u32,
    align_corners: u32,
}

pub struct AffineGrid {
    theta: Tensor,
    size: (usize, usize),
    align_corners: bool,
}

impl AffineGrid {
    pub fn new(theta: Tensor, size: (usize, usize), align_corners: bool) -> Result<Self> {
        // Validate theta shape: must be [B, 2, 3] for affine matrices
        let shape = theta.shape();
        if shape.len() != 3 || shape[1] != 2 || shape[2] != 3 {
            return Err(BarracudaError::invalid_op(
                "affine_grid",
                "theta must be 3D tensor [B, 2, 3]",
            ));
        }

        let _batch_size = shape[0];
        let height = size.0;
        let width = size.1;

        if height == 0 || width == 0 {
            return Err(BarracudaError::invalid_op(
                "affine_grid",
                "height and width must be positive",
            ));
        }

        Ok(Self {
            theta,
            size,
            align_corners,
        })
    }

    fn wgsl_shader() -> &'static str {
        static SHADER: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
            crate::shaders::precision::downcast_f64_to_f32_with_transcendentals(include_str!(
                "../shaders/misc/affine_grid_f64.wgsl"
            ))
        });
        &SHADER
    }

    pub fn execute(self) -> Result<Tensor> {
        let device = self.theta.device();
        let shape = self.theta.shape();
        let batch_size = shape[0];
        let height = self.size.0;
        let width = self.size.1;

        // Output shape: [B, H, W, 2]
        let output_size = batch_size * height * width * 2;
        let output_buffer = device.create_buffer_f32(output_size)?;

        let params = AffineGridParams {
            batch_size: batch_size as u32,
            height: height as u32,
            width: width as u32,
            align_corners: if self.align_corners { 1 } else { 0 },
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("affine_grid_params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let shader = device.compile_shader(Self::wgsl_shader(), Some("affine_grid_shader"));

        let bind_group_layout =
            device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("affine_grid_bind_group_layout"),
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
                    label: Some("affine_grid_pipeline_layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });

        let pipeline = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("affine_grid_pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: "main",
                cache: None,
                compilation_options: Default::default(),
            });

        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("affine_grid_bind_group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.theta.buffer().as_entire_binding(),
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
            label: Some("affine_grid_encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("affine_grid_pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            // Deep Debt Evolution: Capability-based dispatch
            let caps = DeviceCapabilities::from_device(device);
            let (wg_x, wg_y, wg_z) = caps.optimal_workgroup_size_3d(WorkloadType::MatMul);
            let workgroups_x = (width as u32).div_ceil(wg_x);
            let workgroups_y = (height as u32).div_ceil(wg_y);
            let workgroups_z = (batch_size as u32).div_ceil(wg_z);
            compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z);
        }

        device.submit_and_poll(Some(encoder.finish()));

        Ok(Tensor::from_buffer(
            output_buffer,
            vec![batch_size, height, width, 2],
            device.clone(),
        ))
    }
}

impl Tensor {
    /// Generate affine grid from transformation matrices
    ///
    /// # Arguments
    /// - `size`: (height, width) output grid size
    /// - `align_corners`: Whether to align corners
    pub fn affine_grid(self, size: (usize, usize), align_corners: bool) -> Result<Self> {
        AffineGrid::new(self, size, align_corners)?.execute()
    }
}

#[expect(clippy::unwrap_used, reason = "tests")]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device_if_gpu_available;

    #[tokio::test]
    async fn test_affine_grid_basic() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        // Identity transformation matrix [B=1, 2, 3]
        let theta_data = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let theta = Tensor::from_vec_on(theta_data, vec![1, 2, 3], device.clone())
            .await
            .unwrap();

        let output = theta.affine_grid((4, 4), false).unwrap();
        let result = output.to_vec().unwrap();

        assert_eq!(output.shape(), &[1, 4, 4, 2]);
        assert_eq!(result.len(), 32);
        assert!(result.iter().all(|&x| x.is_finite()));
    }
}

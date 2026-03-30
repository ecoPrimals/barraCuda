// SPDX-License-Identifier: AGPL-3.0-or-later
//! Triu - Complete triangular (GPU implementation)
//!
//! **Deep Debt Principles**:
//! - Complete GPU implementation: OVERWRITE existing CPU version
//! - No CPU fallbacks: All computation on GPU
//! - Self-knowledge: Validates matrix dimensions
//! - Modern idiomatic Rust: Result<T, E>

use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

/// f64 is the canonical source — math is universal, precision is silicon.
const SHADER_F64: &str = include_str!("../shaders/linalg/triu_f64.wgsl");

/// f32 variant derived from f64 via precision downcast.
static SHADER_F32: std::sync::LazyLock<String> =
    std::sync::LazyLock::new(|| SHADER_F64.to_string());

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct TriuParams {
    rows: u32,
    cols: u32,
    diagonal: i32,
    _pad1: u32,
}

/// Upper triangular matrix extraction operation.
pub struct Triu {
    input: Tensor,
    diagonal: i32,
}

impl Triu {
    /// Creates a new triu operation. `diagonal` selects which diagonal to include (0 = main).
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn new(input: Tensor, diagonal: i32) -> Result<Self> {
        let shape = input.shape();
        if shape.len() < 2 {
            return Err(BarracudaError::invalid_op(
                "triu",
                "Requires at least 2D tensor",
            ));
        }

        Ok(Self { input, diagonal })
    }

    fn wgsl_shader() -> &'static str {
        &SHADER_F32
    }

    /// Executes the triu operation and returns the upper triangular result.
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let shape = self.input.shape();
        let rows = shape[shape.len() - 2];
        let cols = shape[shape.len() - 1];
        let matrix_size = rows * cols;

        let output_buffer = device.create_buffer_f32(matrix_size)?;

        let params = TriuParams {
            rows: rows as u32,
            cols: cols as u32,
            diagonal: self.diagonal,
            _pad1: 0,
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Triu Params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let bind_group_layout =
            device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Triu Bind Group Layout"),
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

        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Triu Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.input.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buffer.as_entire_binding(),
                },
            ],
        });

        let shader = device.compile_shader(Self::wgsl_shader(), Some("Triu"));
        let pipeline_layout =
            device
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Triu Pipeline Layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    immediate_size: 0,
                });

        let pipeline = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Triu Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                cache: None,
                compilation_options: Default::default(),
            });

        let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("Triu Encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Triu Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&bind_group), &[]);
            let workgroups_x = (cols as u32).div_ceil(16);
            let workgroups_y = (rows as u32).div_ceil(16);
            pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        device.submit_commands(Some(encoder.finish()));

        Ok(Tensor::from_buffer(
            output_buffer,
            shape.to_vec(),
            device.clone(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_triu_basic() {
        let device = crate::device::test_pool::get_test_device().await;
        let matrix = Tensor::from_vec_on(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
            vec![3, 3],
            device.clone(),
        )
        .await
        .unwrap();

        let result = Triu::new(matrix, 0).unwrap().execute().unwrap();
        let output = result.to_vec().unwrap();
        assert_eq!(output[0], 1.0);
        assert_eq!(output[3], 0.0); // Below diagonal
        assert_eq!(output[4], 5.0);
    }
}

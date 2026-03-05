// SPDX-License-Identifier: AGPL-3.0-or-later
//! Grid Sample - Spatial transformer sampling - Pure WGSL
//!
//! **Deep Debt Principles**:
//! - ✅ Pure WGSL implementation (GPU-optimized)
//! - ✅ Safe Rust wrapper (no unsafe code)
//! - ✅ Hardware-agnostic via WebGPU
//! - ✅ Complete implementation (production-ready)
//!
//! ## Algorithm
//!
//! Samples input at arbitrary grid positions (Spatial Transformer Networks):
//! ```text
//! Input:  [B, C, H_in, W_in] - Input tensor
//! Grid:   [B, H_out, W_out, 2] - Sampling coordinates (normalized [-1, 1])
//! Output: [B, C, H_out, W_out] - Sampled output
//!
//! Uses bilinear interpolation for smooth sampling
//! ```

use crate::device::DeviceCapabilities;
use crate::error::Result;
use crate::tensor::Tensor;

/// Spatial transformer sampling: sample input at grid coordinates.
pub struct GridSample {
    input: Tensor,
    grid: Tensor,
}

impl GridSample {
    /// Create a grid sample operation. Input [B,C,H,W], grid [B,H_out,W_out,2].
    pub fn new(input: Tensor, grid: Tensor) -> Self {
        Self { input, grid }
    }

    fn wgsl_shader() -> &'static str {
        static SHADER: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
            crate::shaders::precision::downcast_f64_to_f32_with_transcendentals(include_str!(
                "../shaders/misc/grid_sample_f64.wgsl"
            ))
        });
        &SHADER
    }

    /// Execute grid sampling on GPU with bilinear interpolation.
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let input_shape = self.input.shape();
        let grid_shape = self.grid.shape();

        // Expect input: [B, C, H, W]
        if input_shape.len() != 4 {
            return Err(crate::error::BarracudaError::InvalidShape {
                expected: vec![0, 0, 0, 0],
                actual: input_shape.to_vec(),
            });
        }

        // Expect grid: [B, H_out, W_out, 2]
        if grid_shape.len() != 4 || grid_shape[3] != 2 {
            return Err(crate::error::BarracudaError::InvalidShape {
                expected: vec![input_shape[0], 0, 0, 2],
                actual: grid_shape.to_vec(),
            });
        }

        let batch_size = input_shape[0];
        let channels = input_shape[1];
        let in_height = input_shape[2];
        let in_width = input_shape[3];
        let out_height = grid_shape[1];
        let out_width = grid_shape[2];

        if grid_shape[0] != batch_size {
            return Err(crate::error::BarracudaError::InvalidShape {
                expected: vec![batch_size, out_height, out_width, 2],
                actual: grid_shape.to_vec(),
            });
        }

        let output_size = batch_size * channels * out_height * out_width;
        let output_buffer = device.create_buffer_f32(output_size)?;

        // Create params buffer
        let params_data = [
            batch_size as u32,
            channels as u32,
            in_height as u32,
            in_width as u32,
            out_height as u32,
            out_width as u32,
        ];
        let params_buffer = device.create_uniform_buffer("Params", &params_data);

        let bind_group_layout =
            device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("GridSample BGL"),
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
            label: Some("GridSample BG"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.input.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.grid.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let shader = device.compile_shader(Self::wgsl_shader(), Some("GridSample"));
        let pipeline_layout =
            device
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("GridSample PL"),
                    bind_group_layouts: &[&bind_group_layout],
                    immediate_size: 0,
                });

        let pipeline = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("GridSample Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                cache: None,
                compilation_options: Default::default(),
            });

        let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("GridSample Encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("GridSample Pass"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&bind_group), &[]);

            // Dispatch using standard 2D shader workgroup size (16, 16)
            let caps = DeviceCapabilities::from_device(device);
            let (workgroups_x, workgroups_y) =
                caps.dispatch_2d(out_width as u32, out_height as u32);
            pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        device.submit_and_poll(Some(encoder.finish()));

        let output_data = crate::utils::read_buffer(device, &output_buffer, output_size)?;
        Ok(Tensor::new(
            output_data,
            vec![batch_size, channels, out_height, out_width],
            device.clone(),
        ))
    }
}

impl Tensor {
    /// Sample this tensor at grid coordinates. Uses bilinear interpolation.
    pub fn grid_sample_wgsl(self, grid: Tensor) -> Result<Self> {
        GridSample::new(self, grid).execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device_if_gpu_available;

    #[tokio::test]
    async fn test_grid_sample_identity() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        // Simple 1x1x2x2 input
        let input_data = vec![1.0, 2.0, 3.0, 4.0];
        let input = Tensor::from_vec_on(input_data, vec![1, 1, 2, 2], device.clone())
            .await
            .unwrap();

        // Identity grid (sample at original positions)
        // Grid coordinates in normalized space [-1, 1]
        let grid_data = vec![
            -1.0, -1.0, // Top-left
            1.0, -1.0, // Top-right
            -1.0, 1.0, // Bottom-left
            1.0, 1.0, // Bottom-right
        ];
        let grid = Tensor::from_vec_on(grid_data, vec![1, 2, 2, 2], device)
            .await
            .unwrap();

        let result = input.grid_sample_wgsl(grid).unwrap();
        let output = result.to_vec().unwrap();

        // Should be close to original (allowing for interpolation)
        assert_eq!(output.len(), 4);
        assert!((output[0] - 1.0).abs() < 0.2);
    }
}

// SPDX-License-Identifier: AGPL-3.0-or-later
//! Interpolate Nearest - Nearest neighbor interpolation - Pure WGSL
//!
//! **Deep Debt Principles**:
//! - ✅ Pure WGSL implementation (GPU-optimized)
//! - ✅ Safe Rust wrapper (no unsafe code)
//! - ✅ Hardware-agnostic via WebGPU
//! - ✅ Complete implementation (production-ready)
//!
//! ## Algorithm
//!
//! Resizes images using nearest neighbor sampling:
//! ```text
//! Input:  [B, C, H_in, W_in]
//! Output: [B, C, H_out, W_out]
//!
//! For each output pixel, finds nearest input pixel:
//! in_y = floor(out_y * scale_h)
//! in_x = floor(out_x * scale_w)
//! ```

use crate::device::{DeviceCapabilities, WorkloadType};
use crate::error::Result;
use crate::tensor::Tensor;

/// Nearest-neighbor interpolation for 4D tensors [batch, channels, height, width].
pub struct InterpolateNearest {
    input: Tensor,
    output_size: (usize, usize), // (height, width)
}

impl InterpolateNearest {
    /// Create interpolator with target (height, width).
    pub fn new(input: Tensor, output_size: (usize, usize)) -> Self {
        Self { input, output_size }
    }

    fn wgsl_shader() -> &'static str {
        static SHADER: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
            crate::shaders::precision::downcast_f64_to_f32_with_transcendentals(include_str!(
                "../shaders/misc/interpolate_nearest_f64.wgsl"
            ))
        });
        &SHADER
    }

    /// Execute nearest-neighbor interpolation on GPU.
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let shape = self.input.shape();

        // Expect 4D tensor [batch, channels, height, width]
        if shape.len() != 4 {
            return Err(crate::error::BarracudaError::InvalidShape {
                expected: vec![0, 0, 0, 0],
                actual: shape.to_vec(),
            });
        }

        let batch_size = shape[0];
        let channels = shape[1];
        let in_height = shape[2];
        let in_width = shape[3];
        let (out_height, out_width) = self.output_size;

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
                    label: Some("InterpolateNearest BGL"),
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

        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("InterpolateNearest BG"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.input.buffer().as_entire_binding(),
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

        let shader = device.compile_shader(Self::wgsl_shader(), Some("InterpolateNearest"));
        let pipeline_layout =
            device
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("InterpolateNearest PL"),
                    bind_group_layouts: &[&bind_group_layout],
                    immediate_size: 0,
                });

        let pipeline = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("InterpolateNearest Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                cache: None,
                compilation_options: Default::default(),
            });

        let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("InterpolateNearest Encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("InterpolateNearest Pass"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&bind_group), &[]);

            // Deep Debt Evolution: Capability-based dispatch
            let caps = DeviceCapabilities::from_device(device);
            let optimal_wg_size = caps.optimal_workgroup_size(WorkloadType::Convolution);
            let workgroups_x = (out_width as u32).div_ceil(optimal_wg_size);
            let workgroups_y = (out_height as u32).div_ceil(optimal_wg_size);
            pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        device.submit_and_poll(Some(encoder.finish()));

        Ok(Tensor::from_buffer(
            output_buffer,
            vec![batch_size, channels, out_height, out_width],
            device.clone(),
        ))
    }
}

impl Tensor {
    /// Resize image using nearest-neighbor sampling. Expects [B, C, H, W].
    pub fn interpolate_nearest_wgsl(self, output_size: (usize, usize)) -> Result<Self> {
        InterpolateNearest::new(self, output_size).execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device_if_gpu_available;

    #[tokio::test]
    async fn test_interpolate_nearest_upsample() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        // 1x1x2x2 image, upsample to 4x4
        let input_data = vec![1.0, 2.0, 3.0, 4.0];
        let input = Tensor::from_vec_on(input_data, vec![1, 1, 2, 2], device)
            .await
            .unwrap();

        let result = input.interpolate_nearest_wgsl((4, 4)).unwrap();
        let output = result.to_vec().unwrap();

        // Each input pixel should be replicated 2x2
        assert_eq!(output.len(), 16);
        // Top-left quadrant should all be 1.0
        assert!((output[0] - 1.0).abs() < 1e-5);
    }
}

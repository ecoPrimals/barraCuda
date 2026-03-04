// SPDX-License-Identifier: AGPL-3.0-or-later
//! Bucketize - Assign values to bins - Pure WGSL
//!
//! **Deep Debt Principles**:
//! - ✅ Pure WGSL implementation (GPU-optimized)
//! - ✅ Safe Rust wrapper (no unsafe code)
//! - ✅ Hardware-agnostic via WebGPU
//! - ✅ Complete implementation (production-ready)
//!
//! ## Algorithm
//!
//! Maps each value to a bucket index based on boundaries:
//! ```text
//! Input:  [0.5, 1.5, 2.5, 3.5]
//! Boundaries: [1.0, 2.0, 3.0]
//! Output: [0, 1, 2, 3]  (bucket indices)
//! ```

use crate::device::{DeviceCapabilities, WorkloadType};
use crate::error::Result;
use crate::tensor::Tensor;

pub struct Bucketize {
    input: Tensor,
    boundaries: Vec<f32>,
}

impl Bucketize {
    pub fn new(input: Tensor, boundaries: Vec<f32>) -> Self {
        Self { input, boundaries }
    }

    fn wgsl_shader() -> &'static str {
        static SHADER: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
            crate::shaders::precision::downcast_f64_to_f32_with_transcendentals(include_str!(
                "../shaders/misc/bucketize_f64.wgsl"
            ))
        });
        &SHADER
    }

    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let input_size = self.input.len();
        let num_boundaries = self.boundaries.len();

        // Create output buffer (u32 for bucket indices)
        let output_buffer = device.create_buffer_u32(input_size)?;

        // Create boundaries buffer
        let boundaries_buffer =
            device.create_storage_buffer("Data", bytemuck::cast_slice(&self.boundaries));

        // Create params buffer
        let params_data = [input_size as u32, num_boundaries as u32];
        let params_buffer = device.create_uniform_buffer("Params", &params_data);

        let bind_group_layout =
            device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Bucketize BGL"),
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
            label: Some("Bucketize BG"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.input.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: boundaries_buffer.as_entire_binding(),
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

        let shader = device.compile_shader(Self::wgsl_shader(), Some("Bucketize"));
        let pipeline_layout =
            device
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Bucketize PL"),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });

        let pipeline = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Bucketize Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: "main",
                cache: None,
                compilation_options: Default::default(),
            });

        let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("Bucketize Encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Bucketize Pass"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);

            // Deep Debt Evolution: Capability-based dispatch
            let caps = DeviceCapabilities::from_device(device);
            let optimal_wg_size = caps.optimal_workgroup_size(WorkloadType::ElementWise);
            let workgroups = (input_size as u32).div_ceil(optimal_wg_size);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        device.submit_and_poll(Some(encoder.finish()));

        // Read u32 buffer and convert to f32 for Tensor compatibility
        let u32_data = crate::utils::read_buffer_u32(device, &output_buffer, input_size)?;
        let f32_data: Vec<f32> = u32_data.iter().map(|&x| x as f32).collect();

        Ok(Tensor::new(
            f32_data,
            self.input.shape().to_vec(),
            device.clone(),
        ))
    }
}

impl Tensor {
    pub fn bucketize_wgsl(self, boundaries: Vec<f32>) -> Result<Self> {
        Bucketize::new(self, boundaries).execute()
    }
}

#[expect(clippy::unwrap_used, reason = "tests")]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device_if_gpu_available;

    #[tokio::test]
    async fn test_bucketize_basic() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let input_data = vec![0.5, 1.5, 2.5, 3.5];
        let input = Tensor::from_vec_on(input_data, vec![4], device)
            .await
            .unwrap();

        let boundaries = vec![1.0, 2.0, 3.0];
        let result = input.bucketize_wgsl(boundaries).unwrap();
        let output_f32 = result.to_vec().unwrap();
        let output: Vec<u32> = output_f32.iter().map(|&x| x as u32).collect();

        // 0.5 < 1.0 → bucket 0
        // 1.0 <= 1.5 < 2.0 → bucket 1
        // 2.0 <= 2.5 < 3.0 → bucket 2
        // 3.5 >= 3.0 → bucket 3
        assert_eq!(output, vec![0, 1, 2, 3]);
    }

    #[tokio::test]
    async fn test_bucketize_edge_cases() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let input_data = vec![0.0, 1.0, 2.0, 10.0];
        let input = Tensor::from_vec_on(input_data, vec![4], device)
            .await
            .unwrap();

        let boundaries = vec![1.0, 2.0];
        let result = input.bucketize_wgsl(boundaries).unwrap();
        let output_f32 = result.to_vec().unwrap();
        let output: Vec<u32> = output_f32.iter().map(|&x| x as u32).collect();

        // 0.0 < 1.0 → bucket 0
        // 1.0 (boundary) → bucket 1
        // 2.0 (boundary) → bucket 2
        // 10.0 >= 2.0 → bucket 2
        assert_eq!(output, vec![0, 1, 2, 2]);
    }
}

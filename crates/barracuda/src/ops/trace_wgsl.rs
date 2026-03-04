// SPDX-License-Identifier: AGPL-3.0-or-later
//! Trace - Sum of diagonal elements - Pure WGSL
//!
//! **Deep Debt Principles**:
//! - ✅ Pure WGSL implementation (GPU-optimized)
//! - ✅ Safe Rust wrapper (no unsafe code)
//! - ✅ Hardware-agnostic via WebGPU
//! - ✅ Complete implementation (production-ready)
//!
//! ## Algorithm
//!
//! Computes the trace of a square matrix:
//! ```text
//! trace(A) = sum of diagonal elements
//! For [[a, b], [c, d]]: trace = a + d
//! ```

use crate::device::{DeviceCapabilities, WorkloadType};
use crate::error::Result;
use crate::tensor::Tensor;

/// f64 is the canonical source — math is universal, precision is silicon.
const SHADER_F64: &str = include_str!("../shaders/linalg/trace_f64.wgsl");

/// f32 variant derived from f64 via precision downcast.
static SHADER_F32: std::sync::LazyLock<String> =
    std::sync::LazyLock::new(|| crate::shaders::precision::downcast_f64_to_f32(SHADER_F64));

pub struct Trace {
    input: Tensor,
}

impl Trace {
    pub fn new(input: Tensor) -> Self {
        Self { input }
    }

    fn wgsl_shader() -> &'static str {
        &SHADER_F32
    }

    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let shape = self.input.shape();

        // Expect 2D square matrix
        if shape.len() != 2 || shape[0] != shape[1] {
            return Err(crate::error::BarracudaError::InvalidShape {
                expected: vec![0, 0],
                actual: shape.to_vec(),
            });
        }

        let n = shape[0];

        // Deep Debt Evolution: Capability-based dispatch
        let caps = DeviceCapabilities::from_device(device);
        let optimal_wg_size = caps.optimal_workgroup_size(WorkloadType::Reduction);
        let workgroups = (n as u32).div_ceil(optimal_wg_size);

        // Output buffer: single element for final result, or partial results if multi-workgroup
        let output_size = if workgroups > 1 {
            workgroups as usize
        } else {
            1
        };
        let output_buffer = device.create_buffer_f32(output_size)?;

        let params_buffer = device.create_uniform_buffer("Params", &[n as u32, 0u32, 0u32, 0u32]);

        let bind_group_layout =
            device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Trace BGL"),
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
            label: Some("Trace BG"),
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

        let shader = device.compile_shader(Self::wgsl_shader(), Some("Trace"));
        let pipeline_layout =
            device
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Trace PL"),
                    bind_group_layouts: &[&bind_group_layout],
                    immediate_size: 0,
                });

        let pipeline = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Trace Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                cache: None,
                compilation_options: Default::default(),
            });

        let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("Trace Encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Trace Pass"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&bind_group), &[]);

            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        device.submit_and_poll(Some(encoder.finish()));

        // If multiple workgroups, reduce partial results in a second pass using reduce shader
        let final_buffer = if workgroups > 1 {
            // Second pass: reduce partial results using reduce shader
            let reduce_shader_source = crate::ops::reduce::Reduce::wgsl_shader();
            let reduce_shader = device
                .device
                .create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some("Trace Reduce Shader"),
                    source: wgpu::ShaderSource::Wgsl(reduce_shader_source.into()),
                });

            #[repr(C)]
            #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
            struct ReduceParams {
                size: u32,
                operation: u32, // 0 = Sum
                _pad0: u32,
                _pad1: u32,
            }

            let reduce_params = ReduceParams {
                size: workgroups,
                operation: 0u32, // Sum operation
                _pad0: 0,
                _pad1: 0,
            };

            let final_output_buffer = device.create_buffer_f32(1)?;
            let reduce_params_buffer =
                device
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("Trace Reduce Params"),
                        contents: bytemuck::cast_slice(&[reduce_params]),
                        usage: wgpu::BufferUsages::UNIFORM,
                    });

            let bind_group_layout_2 =
                device
                    .device
                    .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: Some("Trace Reduce BGL"),
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

            let bind_group_2 = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Trace Reduce BG"),
                layout: &bind_group_layout_2,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: output_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: final_output_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: reduce_params_buffer.as_entire_binding(),
                    },
                ],
            });

            let pipeline_layout_2 =
                device
                    .device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("Trace Reduce PL"),
                        bind_group_layouts: &[&bind_group_layout_2],
                        immediate_size: 0,
                    });

            let pipeline_2 =
                device
                    .device
                    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some("Trace Reduce Pipeline"),
                        layout: Some(&pipeline_layout_2),
                        module: &reduce_shader,
                        entry_point: Some("main"),
                        cache: None,
                        compilation_options: Default::default(),
                    });

            let mut encoder_2 = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
                label: Some("Trace Reduce Encoder"),
            });

            {
                let mut pass_2 = encoder_2.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Trace Reduce Pass"),
                    timestamp_writes: None,
                });

                pass_2.set_pipeline(&pipeline_2);
                pass_2.set_bind_group(0, Some(&bind_group_2), &[]);
                // Deep Debt Evolution: Capability-based dispatch for reduction pass
                let caps_2 = DeviceCapabilities::from_device(device);
                let optimal_wg_size_2 = caps_2.optimal_workgroup_size(WorkloadType::Reduction);
                let workgroups_2 = workgroups.div_ceil(optimal_wg_size_2);
                pass_2.dispatch_workgroups(workgroups_2.max(1), 1, 1);
            }

            device.submit_and_poll(Some(encoder_2.finish()));
            final_output_buffer
        } else {
            output_buffer
        };

        // Return scalar tensor with trace result
        Ok(Tensor::from_buffer(final_buffer, vec![1], device.clone()))
    }
}

impl Tensor {
    pub fn trace_wgsl(self) -> Result<Self> {
        Trace::new(self).execute()
    }
}

#[expect(clippy::unwrap_used, reason = "tests")]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device_if_gpu_available;

    #[tokio::test]
    async fn test_trace_2x2() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let input_data = vec![1.0, 2.0, 3.0, 4.0];
        let input = Tensor::from_vec_on(input_data, vec![2, 2], device)
            .await
            .unwrap();

        let result = input.trace_wgsl().unwrap();
        let trace_result = result.to_vec().unwrap();

        // Result should be scalar tensor [trace_value]
        assert_eq!(trace_result.len(), 1);
        // Trace = 1.0 + 4.0 = 5.0
        assert!((trace_result[0] - 5.0).abs() < 1e-5);
    }

    #[tokio::test]
    async fn test_trace_3x3() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        // Matrix: [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        // Diagonal: [1, 5, 9], trace = 15
        let input_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let input = Tensor::from_vec_on(input_data, vec![3, 3], device)
            .await
            .unwrap();

        let result = input.trace_wgsl().unwrap();
        let trace_result = result.to_vec().unwrap();

        assert_eq!(trace_result.len(), 1);
        assert!((trace_result[0] - 15.0).abs() < 1e-5);
    }

    #[tokio::test]
    async fn test_trace_large_matrix() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let n = 512; // Larger than workgroup size to test multi-workgroup reduction
        let mut input_data = vec![0.0; n * n];

        // Fill diagonal with sequential values: 1, 2, 3, ..., n
        for i in 0..n {
            input_data[i * n + i] = (i + 1) as f32;
        }

        let input = Tensor::from_vec_on(input_data, vec![n, n], device)
            .await
            .unwrap();

        let result = input.trace_wgsl().unwrap();
        let trace_result = result.to_vec().unwrap();

        assert_eq!(trace_result.len(), 1);
        // Sum of 1..n = n*(n+1)/2
        let expected = (n * (n + 1) / 2) as f32;
        assert!((trace_result[0] - expected).abs() < 1e-4);
    }
}

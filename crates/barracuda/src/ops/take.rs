// SPDX-License-Identifier: AGPL-3.0-only
//! Take - Pure WGSL
//!
//! Deep Debt Principles:
//! - Self-knowledge: Operation knows its computation
//! - Zero hardcoding: Hardware-agnostic implementation
//! - Modern idiomatic Rust: Safe, zero unsafe code
//! - Complete implementation: Production-ready, no mocks
//! - Hardware-agnostic: Pure WGSL for universal compute

use crate::device::DeviceCapabilities;
use crate::error::Result;
use crate::tensor::Tensor;

/// Take operation (advanced indexing/gather)
pub struct Take {
    input: Tensor,
    indices: Vec<u32>,
}

impl Take {
    /// Create a new take operation
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn new(input: Tensor, indices: Vec<u32>) -> Result<Self> {
        let input_size = input.shape().iter().product::<usize>();
        if indices.iter().any(|&idx| idx as usize >= input_size) {
            return Err(crate::error::BarracudaError::InvalidInput {
                message: format!(
                    "Index out of bounds: input_size={input_size}, indices={indices:?}"
                ),
            });
        }
        Ok(Self { input, indices })
    }

    /// Get the WGSL shader source
    fn wgsl_shader() -> &'static str {
        static S: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
            include_str!("../shaders/tensor/take_f64.wgsl").to_string()
        });
        &S
    }

    /// Execute the take operation
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let output_size = self.indices.len();

        if output_size == 0 {
            return Ok(Tensor::new(vec![], vec![0], device.clone()));
        }

        // Access input buffer directly (zero-copy)
        let input_buffer = self.input.buffer();

        // Create indices buffer
        let indices_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Take Indices"),
                contents: bytemuck::cast_slice(&self.indices),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

        // Create output buffer (ensure minimum 32 bytes for WebGPU storage binding)
        let output_byte_size = (output_size * std::mem::size_of::<f32>()).max(32);
        let output_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Take Output Buffer"),
            size: output_byte_size as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create uniform buffer for parameters
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            output_size: u32,
            input_size: u32,
            _pad1: u32,
            _pad2: u32,
        }

        let input_size = self.input.shape().iter().product::<usize>();
        let params = Params {
            output_size: output_size as u32,
            input_size: input_size as u32,
            _pad1: 0,
            _pad2: 0,
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Take Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        // Compile shader
        let shader_module = device.compile_shader(Self::wgsl_shader(), Some("Take Shader"));

        // Create bind group layout
        let bind_group_layout =
            device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Take Bind Group Layout"),
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
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
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
                    ],
                });

        // Create bind group
        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Take Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: indices_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: output_buffer.as_entire_binding(),
                },
            ],
        });

        // Create compute pipeline
        let pipeline_layout =
            device
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Take Pipeline Layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    immediate_size: 0,
                });

        let compute_pipeline =
            device
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("Take Pipeline"),
                    layout: Some(&pipeline_layout),
                    module: &shader_module,
                    entry_point: Some("main"),
                    cache: None,
                    compilation_options: Default::default(),
                });

        // Execute compute shader
        let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("Take Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Take Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&compute_pipeline);
            compute_pass.set_bind_group(0, Some(&bind_group), &[]);

            // Dispatch using standard 1D shader workgroup size (256)
            let caps = DeviceCapabilities::from_device(device);
            let workgroups = caps.dispatch_1d(output_size as u32);
            compute_pass.dispatch_workgroups(workgroups, 1, 1);
        }

        device.submit_commands(Some(encoder.finish()));
        let output_data = crate::utils::read_buffer(device, &output_buffer, output_size)?;
        Ok(Tensor::new(output_data, vec![output_size], device.clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device_if_gpu_available;

    #[tokio::test]
    async fn test_take_basic() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let input = Tensor::from_data(&[10.0, 20.0, 30.0, 40.0], vec![4], device).unwrap();

        let result = Take::new(input, vec![0, 2, 1]).unwrap().execute().unwrap();
        let output = result.to_vec().unwrap();

        assert_eq!(output.len(), 3);
        assert_eq!(output[0], 10.0);
        assert_eq!(output[1], 30.0);
        assert_eq!(output[2], 20.0);
    }

    #[tokio::test]
    async fn test_take_repeated() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let input = Tensor::from_data(&[1.0, 2.0, 3.0], vec![3], device).unwrap();

        let result = Take::new(input, vec![0, 0, 1, 1, 2])
            .unwrap()
            .execute()
            .unwrap();
        let output = result.to_vec().unwrap();

        assert_eq!(output.len(), 5);
        assert_eq!(output[0], 1.0);
        assert_eq!(output[1], 1.0);
        assert_eq!(output[2], 2.0);
        assert_eq!(output[3], 2.0);
        assert_eq!(output[4], 3.0);
    }

    #[tokio::test]
    async fn test_take_large() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let data: Vec<f32> = (0..1000).map(|i| i as f32).collect();
        let input = Tensor::from_data(&data, vec![1000], device).unwrap();

        let indices: Vec<u32> = (0..100).map(|i| (i * 10) as u32).collect();
        let result = Take::new(input, indices).unwrap().execute().unwrap();
        let output = result.to_vec().unwrap();

        assert_eq!(output.len(), 100);
        for i in 0..100 {
            assert_eq!(output[i], (i * 10) as f32);
        }
    }

    #[tokio::test]
    async fn test_take_empty() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let input = Tensor::from_data(&[1.0, 2.0, 3.0], vec![3], device).unwrap();

        let result = Take::new(input, vec![]).unwrap().execute().unwrap();
        let output = result.to_vec().unwrap();

        assert_eq!(output.len(), 0);
    }

    #[tokio::test]
    async fn test_take_invalid() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let input = Tensor::from_data(&[1.0, 2.0, 3.0], vec![3], device).unwrap();

        assert!(Take::new(input, vec![0, 5]).is_err());
    }
}

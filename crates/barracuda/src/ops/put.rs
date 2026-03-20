// SPDX-License-Identifier: AGPL-3.0-or-later
//! Put - Pure WGSL
//!
//! Deep Debt Principles:
//! - Self-knowledge: Operation knows its computation
//! - Zero hardcoding: Hardware-agnostic implementation
//! - Modern idiomatic Rust: Safe, zero unsafe code
//! - Complete implementation: Production-ready, no mocks
//! - Hardware-agnostic: Pure WGSL for universal compute
//!
//! NOTE: Uses atomic operations when accumulate=true

use crate::device::DeviceCapabilities;
use crate::error::Result;
use crate::tensor::Tensor;

/// Put operation (scatter with indexing)
pub struct Put {
    output: Tensor,
    indices: Vec<u32>,
    values: Tensor,
    accumulate: bool,
}

impl Put {
    /// Create a new put operation
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if indices length does not match values size, or any index is out of bounds.
    pub fn new(
        output: Tensor,
        indices: Vec<u32>,
        values: Tensor,
        accumulate: bool,
    ) -> Result<Self> {
        let output_size = output.shape().iter().product::<usize>();
        let values_size = values.shape().iter().product::<usize>();

        if indices.len() != values_size {
            return Err(crate::error::BarracudaError::InvalidInput {
                message: format!(
                    "Indices length {} doesn't match values size {}",
                    indices.len(),
                    values_size
                ),
            });
        }

        // Validate indices are in bounds
        for &idx in &indices {
            if idx as usize >= output_size {
                return Err(crate::error::BarracudaError::InvalidInput {
                    message: format!("Index {idx} out of bounds for output size {output_size}"),
                });
            }
        }

        Ok(Self {
            output,
            indices,
            values,
            accumulate,
        })
    }

    /// Get the WGSL shader source
    fn wgsl_shader() -> &'static str {
        {
            static S: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
                include_str!("../shaders/tensor/put_f64.wgsl").to_string()
            });
            &S
        }
    }

    /// Execute the put operation
    /// Note: This modifies the output tensor in-place
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.output.device();
        let output_size: usize = self.output.shape().iter().product();
        let num_values = self.values.shape().iter().product::<usize>();

        // Ensure minimum 32 bytes for WebGPU storage buffer binding requirements
        let byte_size = (output_size * std::mem::size_of::<f32>()).max(32);
        let mut work_contents = self.output.to_vec()?;
        work_contents.resize(byte_size / std::mem::size_of::<f32>(), 0.0);
        let work_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Put Work Buffer"),
                contents: bytemuck::cast_slice(&work_contents),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            });

        // Access buffers directly (zero-copy)
        let values_buffer = self.values.buffer();

        // Create indices buffer
        let indices_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Put Indices"),
                contents: bytemuck::cast_slice(&self.indices),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

        // Create uniform buffer for parameters
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            output_size: u32,
            num_values: u32,
            accumulate: u32,
            _pad1: u32,
        }

        let params = Params {
            output_size: output_size as u32,
            num_values: num_values as u32,
            accumulate: u32::from(self.accumulate),
            _pad1: 0,
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Put Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        // Compile shader
        let shader_module = device.compile_shader(Self::wgsl_shader(), Some("Put Shader"));

        // Create bind group layout
        let bind_group_layout =
            device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Put Bind Group Layout"),
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
            label: Some("Put Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: values_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: indices_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: work_buffer.as_entire_binding(),
                },
            ],
        });

        // Create compute pipeline
        let pipeline_layout =
            device
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Put Pipeline Layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    immediate_size: 0,
                });

        let compute_pipeline =
            device
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("Put Pipeline"),
                    layout: Some(&pipeline_layout),
                    module: &shader_module,
                    entry_point: Some("main"),
                    cache: None,
                    compilation_options: Default::default(),
                });

        // Execute compute shader
        let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("Put Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Put Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&compute_pipeline);
            compute_pass.set_bind_group(0, Some(&bind_group), &[]);

            // Dispatch using standard 1D shader workgroup size (256)
            let caps = DeviceCapabilities::from_device(device);
            let workgroups = caps.dispatch_1d(num_values as u32);
            compute_pass.dispatch_workgroups(workgroups, 1, 1);
        }

        device.submit_commands(Some(encoder.finish()));

        // Read back via device (ensures GPU writes are visible)
        let output_data = device.read_buffer_f32(&work_buffer, output_size)?;
        Ok(Tensor::new(
            output_data,
            self.output.shape().to_vec(),
            device.clone(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::WgpuDevice;
    use std::sync::Arc;

    async fn get_test_device() -> Option<Arc<WgpuDevice>> {
        crate::device::test_pool::get_test_device_if_gpu_available().await
    }

    #[tokio::test]
    async fn test_put_basic() {
        let Some(device) = get_test_device().await else {
            return;
        };
        let output = Tensor::from_data(&[0.0, 0.0, 0.0, 0.0], vec![4], device.clone()).unwrap();
        let values = Tensor::from_data(&[10.0, 30.0], vec![2], device).unwrap();

        let result = Put::new(output, vec![0, 2], values, false)
            .unwrap()
            .execute()
            .unwrap();
        let output_data = result.to_vec().unwrap();

        assert_eq!(output_data[0], 10.0);
        assert_eq!(output_data[2], 30.0);
        assert_eq!(output_data[1], 0.0);
        assert_eq!(output_data[3], 0.0);
    }

    #[tokio::test]
    async fn test_put_accumulate() {
        let Some(device) = get_test_device().await else {
            return;
        };
        let output = Tensor::from_data(&[1.0, 2.0, 3.0, 4.0], vec![4], device.clone()).unwrap();
        let values = Tensor::from_data(&[10.0, 20.0], vec![2], device).unwrap();

        let result = Put::new(output, vec![0, 1], values, true)
            .unwrap()
            .execute()
            .unwrap();
        let output_data = result.to_vec().unwrap();

        // With accumulate, values are added
        assert_eq!(output_data[0], 11.0);
        assert_eq!(output_data[1], 22.0);
    }

    #[tokio::test]
    async fn test_put_invalid_index() {
        let Some(device) = get_test_device().await else {
            return;
        };
        let output = Tensor::from_data(&[0.0, 0.0], vec![2], device.clone()).unwrap();
        let values = Tensor::from_data(&[1.0], vec![1], device).unwrap();

        assert!(Put::new(output, vec![5], values, false).is_err());
    }

    #[tokio::test]
    async fn test_put_length_mismatch() {
        let Some(device) = get_test_device().await else {
            return;
        };
        let output = Tensor::from_data(&[0.0, 0.0], vec![2], device.clone()).unwrap();
        let values = Tensor::from_data(&[1.0, 2.0, 3.0], vec![3], device).unwrap();

        assert!(Put::new(output, vec![0], values, false).is_err());
    }

    #[tokio::test]
    async fn test_put_repeated_indices() {
        let Some(device) = get_test_device().await else {
            return;
        };
        let output = Tensor::from_data(&[0.0, 0.0], vec![2], device.clone()).unwrap();
        let values = Tensor::from_data(&[1.0, 2.0], vec![2], device).unwrap();

        // Same index twice - race condition without atomics: either write can win
        let result = Put::new(output, vec![0, 0], values, false)
            .unwrap()
            .execute()
            .unwrap();
        let output_data = result.to_vec().unwrap();
        // GPU non-atomic writes to same location are non-deterministic; accept either
        assert!(output_data[0] == 1.0 || output_data[0] == 2.0);
    }
}

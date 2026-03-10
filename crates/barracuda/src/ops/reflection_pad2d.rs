// SPDX-License-Identifier: AGPL-3.0-only
//! ReflectionPad2D - Padding with reflection
//!
//! Pads image by reflecting pixels at borders.
//! Better for image tasks than zero-padding.
//!
//! Deep Debt Principles:
//! - Pure GPU/WGSL execution
//! - Safe Rust wrappers
//! - Hardware-agnostic via WebGPU
//! - Runtime device discovery
//! - Zero CPU fallbacks in execution

use crate::error::Result;
use crate::tensor::Tensor;
use wgpu::util::DeviceExt;

/// ReflectionPad2D operation
pub struct ReflectionPad2D {
    input: Tensor,
    padding: (usize, usize, usize, usize), // (left, right, top, bottom)
}

impl ReflectionPad2D {
    /// Create a new reflection pad 2D operation
    pub fn new(input: Tensor, padding: (usize, usize, usize, usize)) -> Result<Self> {
        let shape = input.shape();
        if shape.len() != 4 {
            return Err(crate::error::BarracudaError::invalid_op(
                "ReflectionPad2D",
                format!("Expected 4D tensor (NCHW), got {}D", shape.len()),
            ));
        }
        Ok(Self { input, padding })
    }

    /// Get the WGSL shader source
    fn wgsl_shader() -> &'static str {
        {
            static S: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
                include_str!(
                    "../shaders/tensor/reflection_pad_f64.wgsl"
                ).to_string()
            });
            &S
        }
    }

    /// Execute the reflection pad 2D operation
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let shape = self.input.shape();

        // Assume NCHW format
        let batch = shape[0];
        let channels = shape[1];
        let in_height = shape[2];
        let in_width = shape[3];
        
        let (pad_left, pad_right, pad_top, pad_bottom) = self.padding;
        let out_height = in_height + pad_top + pad_bottom;
        let out_width = in_width + pad_left + pad_right;
        
        let output_size = batch * channels * out_height * out_width;

        // Create buffers
        let input_buffer = self.input.buffer();

        let output_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ReflectionPad2D Output"),
            size: (output_size * std::mem::size_of::<f32>()) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        // Create uniform buffer for parameters
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            batch: u32,
            channels: u32,
            in_height: u32,
            in_width: u32,
            out_height: u32,
            out_width: u32,
            pad_top: u32,
            pad_left: u32,
        }

        let params = Params {
            batch: batch as u32,
            channels: channels as u32,
            in_height: in_height as u32,
            in_width: in_width as u32,
            out_height: out_height as u32,
            out_width: out_width as u32,
            pad_top: pad_top as u32,
            pad_left: pad_left as u32,
        };

        let params_buffer = device.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ReflectionPad2D Params"),
            contents: bytemuck::cast_slice(&[params]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Create bind group layout
        let bind_group_layout = device.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ReflectionPad2D Bind Group Layout"),
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

        // Create bind group
        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ReflectionPad2D Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buffer.as_entire_binding(),
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

        // Create compute pipeline
        let shader_module = device.compile_shader(Self::wgsl_shader(), Some("ReflectionPad2D Shader"));

        let pipeline_layout = device.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("ReflectionPad2D Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
                    immediate_size: 0,
        });

        let compute_pipeline = device.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("ReflectionPad2D Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("main"),
        cache: None,
        compilation_options: Default::default(),
        });

        // Execute compute shader
        let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("ReflectionPad2D Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("ReflectionPad2D Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&compute_pipeline);
            compute_pass.set_bind_group(0, Some(&bind_group), &[]);
            
            let workgroups_x = (out_width as u32 + 15) / 16;
            let workgroups_y = (out_height as u32 + 15) / 16;
            compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        device.submit_and_poll(Some(encoder.finish()));

        // Read back results
        let output_data = crate::utils::read_buffer(device, &output_buffer, output_size)?;

        Ok(Tensor::new(
            output_data,
            vec![batch, channels, out_height, out_width],
            device.clone(),
        ))
    }
}

impl Tensor {
    /// Apply reflection padding 2D (NCHW format)
    ///
    /// # Arguments
    ///
    /// * `padding` - (left, right, top, bottom) padding amounts
    pub fn reflection_pad2d(self, padding: (usize, usize, usize, usize)) -> Result<Self> {
        ReflectionPad2D::new(self, padding)?.execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn get_test_device() -> Option<std::sync::Arc<crate::device::WgpuDevice>> {
        crate::device::test_pool::get_test_device_if_gpu_available().await
    }

    #[tokio::test]
    async fn test_reflection_pad2d_basic() {
        let Some(device) = get_test_device().await else { return };
        let input = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2], device.clone());
        let output = input.reflection_pad2d((1, 1, 1, 1)).unwrap();
        assert_eq!(output.shape(), &[1, 1, 4, 4]);
    }

    #[tokio::test]
    async fn test_reflection_pad2d_edge_cases() {
        let Some(device) = get_test_device().await else { return };
        // No padding
        let input = Tensor::new(vec![1.0; 1 * 1 * 4 * 4], vec![1, 1, 4, 4], device.clone());
        let output = input.reflection_pad2d((0, 0, 0, 0)).unwrap();
        assert_eq!(output.shape(), &[1, 1, 4, 4]);

        // Single pixel
        let input = Tensor::new(vec![5.0], vec![1, 1, 1, 1], device.clone());
        let output = input.reflection_pad2d((1, 1, 1, 1)).unwrap();
        assert_eq!(output.shape(), &[1, 1, 3, 3]);
    }

    #[tokio::test]
    async fn test_reflection_pad2d_boundary() {
        let Some(device) = get_test_device().await else { return };
        // Large padding
        let input = Tensor::new(vec![1.0; 1 * 3 * 4 * 4], vec![1, 3, 4, 4], device.clone());
        let output = input.reflection_pad2d((2, 2, 2, 2)).unwrap();
        assert_eq!(output.shape(), &[1, 3, 8, 8]);

        // Asymmetric padding
        let input = Tensor::new(vec![1.0; 1 * 1 * 8 * 8], vec![1, 1, 8, 8], device.clone());
        let output = input.reflection_pad2d((3, 1, 2, 4)).unwrap();
        assert_eq!(output.shape(), &[1, 1, 14, 12]);
    }

    #[tokio::test]
    async fn test_reflection_pad2d_large_batch() {
        let Some(device) = get_test_device().await else { return };
        // Batch size 4, multiple channels
        let batch_size = 4;
        let channels = 3;
        let input = Tensor::new(
            vec![1.0; batch_size * channels * 16 * 16],
            vec![batch_size, channels, 16, 16],
            device.clone(),
        );
        let output = input.reflection_pad2d((2, 2, 2, 2)).unwrap();
        assert_eq!(output.shape(), &[batch_size, channels, 20, 20]);
    }
}

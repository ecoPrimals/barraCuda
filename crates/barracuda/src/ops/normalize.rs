//! Normalize - L2 normalization along dimension (Pure WGSL)
//!
//! Normalizes vectors to unit length: x_normalized = x / ||x||_2
//!
//! **Deep Debt Principles**:
//! - Pure WGSL implementation (no CPU code)
//! - Safe Rust wrapper (no unsafe code)
//! - Hardware-agnostic via WebGPU
//! - Complete implementation (production-ready)

use crate::device::{DeviceCapabilities, WorkloadType};
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;
use wgpu::util::DeviceExt;

/// L2 Normalization
pub struct Normalize {
    input: Tensor,
    dim: usize,
    epsilon: f32,
}

impl Normalize {
    pub fn new(input: Tensor, dim: usize, epsilon: f32) -> Result<Self> {
        let input_shape = input.shape();
        if dim >= input_shape.len() {
            return Err(BarracudaError::invalid_op(
                "normalize",
                "dim must be less than input rank",
            ));
        }

        if epsilon <= 0.0 {
            return Err(BarracudaError::invalid_op(
                "normalize",
                "epsilon must be positive",
            ));
        }

        Ok(Self {
            input,
            dim,
            epsilon,
        })
    }

    fn wgsl_shader() -> &'static str {
        {
            static SHADER: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
                crate::shaders::precision::downcast_f64_to_f32_with_transcendentals(include_str!(
                    "../shaders/norm/normalize_f64.wgsl"
                ))
            });
            std::sync::LazyLock::force(&SHADER).as_str()
        }
    }

    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let input_shape = self.input.shape();
        let dim_size = input_shape[self.dim];

        // Compute outer (product of dimensions before dim)
        let outer = input_shape[..self.dim].iter().product::<usize>();

        // Compute inner (product of dimensions after dim)
        let inner = input_shape[self.dim + 1..].iter().product::<usize>();

        let output_size = input_shape.iter().product::<usize>();
        let output_buffer = device.create_buffer_f32(output_size)?;

        // Create uniform buffer
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            outer: u32,
            dim_size: u32,
            inner: u32,
            epsilon: f32,
        }

        let params = Params {
            outer: outer as u32,
            dim_size: dim_size as u32,
            inner: inner as u32,
            epsilon: self.epsilon,
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Normalize Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        // Compile shader
        let shader_module = device.compile_shader(Self::wgsl_shader(), Some("Normalize Shader"));

        // Create bind group layout
        let bind_group_layout =
            device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Normalize Bind Group Layout"),
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

        // Create bind group
        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Normalize Bind Group"),
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

        // Create pipeline layout
        let pipeline_layout =
            device
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Normalize Pipeline Layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });

        // Create pipeline
        let pipeline = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Normalize Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: "main",
                cache: None,
                compilation_options: Default::default(),
            });

        // Encode and execute
        let mut encoder = device
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Normalize Encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Normalize Pass"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);

            // Deep Debt Evolution: Capability-based dispatch
            let caps = DeviceCapabilities::from_device(device);
            let optimal_wg_size = caps.optimal_workgroup_size(WorkloadType::Reduction);
            let workgroups = ((outer * inner) as u32).div_ceil(optimal_wg_size);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        device.submit_and_poll(Some(encoder.finish()));

        Ok(Tensor::from_buffer(
            output_buffer,
            input_shape.to_vec(),
            device.clone(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device_if_gpu_available;

    #[tokio::test]
    async fn test_normalize_basic() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let input = Tensor::from_vec_on(vec![1.0, 2.0, 3.0, 4.0], vec![4], device.clone())
            .await
            .unwrap();

        let normalize = Normalize::new(input, 0, 1e-8).unwrap();
        let output = normalize.execute().unwrap();

        assert_eq!(output.shape(), &[4]);
    }

    #[tokio::test]
    async fn test_normalize_2d() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let input = Tensor::from_vec_on(vec![1.0; 12], vec![3, 4], device.clone())
            .await
            .unwrap();

        let normalize = Normalize::new(input, 1, 1e-8).unwrap();
        let output = normalize.execute().unwrap();

        assert_eq!(output.shape(), &[3, 4]);
    }
}

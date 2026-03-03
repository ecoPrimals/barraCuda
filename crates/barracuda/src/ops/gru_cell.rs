//! GRU Cell - Pure WGSL
//!
//! Deep Debt Principles:
//! - Self-knowledge: Operation knows its computation
//! - Zero hardcoding: Hardware-agnostic implementation
//! - Modern idiomatic Rust: Safe, zero unsafe code
//! - Complete implementation: Production-ready, no mocks
//! - Hardware-agnostic: Pure WGSL for universal compute

use crate::device::{DeviceCapabilities, WorkloadType};
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;
use wgpu::util::DeviceExt;

/// GRU Cell operation
pub struct GRUCell {
    input: Tensor,
    weight_ih: Tensor,
    weight_hh: Tensor,
    bias_ih: Tensor,
    bias_hh: Tensor,
    h_prev: Tensor,
    batch_size: usize,
    input_size: usize,
    hidden_size: usize,
}

impl GRUCell {
    /// Create a new GRU cell operation
    pub fn new(
        input: Tensor,
        weight_ih: Tensor,
        weight_hh: Tensor,
        bias_ih: Tensor,
        bias_hh: Tensor,
        h_prev: Tensor,
    ) -> Result<Self> {
        let input_shape = input.shape();
        let batch_size = input_shape[0];
        let input_size = input_shape[1..].iter().product::<usize>();

        let hidden_size = h_prev.shape()[1..].iter().product::<usize>();

        // Validate dimensions
        if weight_ih.shape().iter().product::<usize>() != 3 * hidden_size * input_size {
            return Err(BarracudaError::invalid_op(
                "gru_cell",
                "weight_ih must be [3*hidden_size, input_size]",
            ));
        }

        if weight_hh.shape().iter().product::<usize>() != 3 * hidden_size * hidden_size {
            return Err(BarracudaError::invalid_op(
                "gru_cell",
                "weight_hh must be [3*hidden_size, hidden_size]",
            ));
        }

        if h_prev.shape()[0] != batch_size
            || h_prev.shape()[1..].iter().product::<usize>() != hidden_size
        {
            return Err(BarracudaError::invalid_op(
                "gru_cell",
                "h_prev shape mismatch",
            ));
        }

        Ok(Self {
            input,
            weight_ih,
            weight_hh,
            bias_ih,
            bias_hh,
            h_prev,
            batch_size,
            input_size,
            hidden_size,
        })
    }

    /// Get the WGSL shader source
    fn wgsl_shader() -> &'static str {
        {
            static SHADER: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
                crate::shaders::precision::downcast_f64_to_f32_with_transcendentals(include_str!(
                    "../shaders/rnn/gru_cell_f64.wgsl"
                ))
            });
            std::sync::LazyLock::force(&SHADER).as_str()
        }
    }

    /// Execute the GRU cell operation
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();

        // Create output buffer for h_next
        let h_next_size = self.batch_size * self.hidden_size;
        let h_next_buffer = device.create_buffer_f32(h_next_size)?;

        // Create uniform buffer for parameters
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            batch_size: u32,
            input_size: u32,
            hidden_size: u32,
            _padding: u32,
        }

        let params = Params {
            batch_size: self.batch_size as u32,
            input_size: self.input_size as u32,
            hidden_size: self.hidden_size as u32,
            _padding: 0,
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("GRUCell Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        // Compile shader
        let shader_module = device.compile_shader(Self::wgsl_shader(), Some("GRUCell Shader"));

        // Create bind group layout
        let bind_group_layout =
            device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("GRUCell Bind Group Layout"),
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
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 4,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 5,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 6,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 7,
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
            label: Some("GRUCell Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.input.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.weight_ih.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.weight_hh.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.bias_ih.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.bias_hh.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.h_prev.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: h_next_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        // Create pipeline
        let pipeline_layout =
            device
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("GRUCell Pipeline Layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });

        let pipeline = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("GRUCell Pipeline"),
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
                label: Some("GRUCell Encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("GRUCell Pass"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);

            // Deep Debt Evolution: Capability-based dispatch
            let caps = DeviceCapabilities::from_device(device);
            let optimal_wg_size = caps.optimal_workgroup_size(WorkloadType::MatMul);
            let workgroups = (self.batch_size as u32).div_ceil(optimal_wg_size);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        device.submit_and_poll(Some(encoder.finish()));

        // Create output tensor
        Ok(Tensor::from_buffer(
            h_next_buffer,
            vec![self.batch_size, self.hidden_size],
            device.clone(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device_if_gpu_available;

    #[tokio::test]
    async fn test_gru_cell_basic() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let batch_size = 2;
        let input_size = 4;
        let hidden_size = 8;

        let input = Tensor::from_vec_on(
            vec![0.5; batch_size * input_size],
            vec![batch_size, input_size],
            device.clone(),
        )
        .await
        .unwrap();

        let weight_ih = Tensor::from_vec_on(
            vec![0.01; 3 * hidden_size * input_size],
            vec![3 * hidden_size, input_size],
            device.clone(),
        )
        .await
        .unwrap();

        let weight_hh = Tensor::from_vec_on(
            vec![0.01; 3 * hidden_size * hidden_size],
            vec![3 * hidden_size, hidden_size],
            device.clone(),
        )
        .await
        .unwrap();

        let bias_ih = Tensor::from_vec_on(
            vec![0.0; 3 * hidden_size],
            vec![3 * hidden_size],
            device.clone(),
        )
        .await
        .unwrap();

        let bias_hh = Tensor::from_vec_on(
            vec![0.0; 3 * hidden_size],
            vec![3 * hidden_size],
            device.clone(),
        )
        .await
        .unwrap();

        let h_prev = Tensor::from_vec_on(
            vec![0.0; batch_size * hidden_size],
            vec![batch_size, hidden_size],
            device.clone(),
        )
        .await
        .unwrap();

        let h_next = GRUCell::new(input, weight_ih, weight_hh, bias_ih, bias_hh, h_prev)
            .unwrap()
            .execute()
            .unwrap();

        assert_eq!(h_next.shape(), &[batch_size, hidden_size]);
    }
}

// SPDX-License-Identifier: AGPL-3.0-or-later
//! LSTM Cell - Pure WGSL
//!
//! Deep Debt Principles:
//! - Self-knowledge: Operation knows its computation
//! - Zero hardcoding: Hardware-agnostic implementation
//! - Modern idiomatic Rust: Safe, zero unsafe code
//! - Complete implementation: Production-ready, no mocks
//! - Hardware-agnostic: Pure WGSL for universal compute

use crate::device::DeviceCapabilities;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

/// LSTM Cell operation
pub struct LSTMCell {
    input: Tensor,
    weight_ih: Tensor,
    weight_hh: Tensor,
    bias_ih: Tensor,
    bias_hh: Tensor,
    h_prev: Tensor,
    c_prev: Tensor,
    batch_size: usize,
    input_size: usize,
    hidden_size: usize,
}

impl LSTMCell {
    /// Create a new LSTM cell operation
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn new(
        input: Tensor,
        weight_ih: Tensor,
        weight_hh: Tensor,
        bias_ih: Tensor,
        bias_hh: Tensor,
        h_prev: Tensor,
        c_prev: Tensor,
    ) -> Result<Self> {
        let input_shape = input.shape();
        let batch_size = input_shape[0];
        let input_size = input_shape[1..].iter().product::<usize>();

        let hidden_size = h_prev.shape()[1..].iter().product::<usize>();

        // Validate dimensions
        if weight_ih.shape().iter().product::<usize>() != 4 * hidden_size * input_size {
            return Err(BarracudaError::invalid_op(
                "lstm_cell",
                "weight_ih must be [4*hidden_size, input_size]",
            ));
        }

        if weight_hh.shape().iter().product::<usize>() != 4 * hidden_size * hidden_size {
            return Err(BarracudaError::invalid_op(
                "lstm_cell",
                "weight_hh must be [4*hidden_size, hidden_size]",
            ));
        }

        if h_prev.shape()[0] != batch_size
            || h_prev.shape()[1..].iter().product::<usize>() != hidden_size
        {
            return Err(BarracudaError::invalid_op(
                "lstm_cell",
                "h_prev shape mismatch",
            ));
        }

        if c_prev.shape()[0] != batch_size
            || c_prev.shape()[1..].iter().product::<usize>() != hidden_size
        {
            return Err(BarracudaError::invalid_op(
                "lstm_cell",
                "c_prev shape mismatch",
            ));
        }

        Ok(Self {
            input,
            weight_ih,
            weight_hh,
            bias_ih,
            bias_hh,
            h_prev,
            c_prev,
            batch_size,
            input_size,
            hidden_size,
        })
    }

    /// Get the WGSL shader source
    fn wgsl_shader() -> &'static str {
        {
            static SHADER: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
                include_str!("../shaders/rnn/lstm_cell_f64.wgsl").to_string()
            });
            std::sync::LazyLock::force(&SHADER).as_str()
        }
    }

    /// Execute the LSTM cell operation
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<(Tensor, Tensor)> {
        let device = self.input.device();

        // Create output buffers for h_next and c_next
        let h_next_size = self.batch_size * self.hidden_size;
        let c_next_size = self.batch_size * self.hidden_size;

        let h_next_buffer = device.create_buffer_f32(h_next_size)?;
        let c_next_buffer = device.create_buffer_f32(c_next_size)?;

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
                label: Some("LSTMCell Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        // Compile shader
        let shader_module = device.compile_shader(Self::wgsl_shader(), Some("LSTMCell Shader"));

        // Combine bias_ih and bias_hh into single buffer [bias_ih..., bias_hh...]
        let bias_ih_data = self.bias_ih.to_vec()?;
        let bias_hh_data = self.bias_hh.to_vec()?;
        let mut bias_combined: Vec<f32> =
            Vec::with_capacity(bias_ih_data.len() + bias_hh_data.len());
        bias_combined.extend_from_slice(&bias_ih_data);
        bias_combined.extend_from_slice(&bias_hh_data);

        let bias_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("LSTMCell Bias Combined"),
                contents: bytemuck::cast_slice(&bias_combined),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

        // Create bind group layout
        let bind_group_layout =
            device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("LSTMCell Bind Group Layout"),
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
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 5,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
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
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 8,
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
            label: Some("LSTMCell Bind Group"),
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
                    resource: bias_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: self.h_prev.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: self.c_prev.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: h_next_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: c_next_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        // Create pipeline
        let pipeline_layout =
            device
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("LSTMCell Pipeline Layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    immediate_size: 0,
                });

        let pipeline = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("LSTMCell Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: Some("main"),
                cache: None,
                compilation_options: Default::default(),
            });

        // Encode and execute
        let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("LSTMCell Encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("LSTMCell Pass"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&bind_group), &[]);

            // Dispatch using standard 1D shader workgroup size (256)
            let caps = DeviceCapabilities::from_device(device);
            let workgroups = caps.dispatch_1d(self.batch_size as u32);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        device.submit_and_poll(Some(encoder.finish()));

        let h_size = self.batch_size * self.hidden_size;
        let h_data = crate::utils::read_buffer(device, &h_next_buffer, h_size)?;
        let c_data = crate::utils::read_buffer(device, &c_next_buffer, h_size)?;

        let h_next = Tensor::new(
            h_data,
            vec![self.batch_size, self.hidden_size],
            device.clone(),
        );
        let c_next = Tensor::new(
            c_data,
            vec![self.batch_size, self.hidden_size],
            device.clone(),
        );

        Ok((h_next, c_next))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device_if_gpu_available;

    #[tokio::test]
    async fn test_lstm_cell_basic() {
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
            vec![0.01; 4 * hidden_size * input_size],
            vec![4 * hidden_size, input_size],
            device.clone(),
        )
        .await
        .unwrap();

        let weight_hh = Tensor::from_vec_on(
            vec![0.01; 4 * hidden_size * hidden_size],
            vec![4 * hidden_size, hidden_size],
            device.clone(),
        )
        .await
        .unwrap();

        let bias_ih = Tensor::from_vec_on(
            vec![0.0; 4 * hidden_size],
            vec![4 * hidden_size],
            device.clone(),
        )
        .await
        .unwrap();

        let bias_hh = Tensor::from_vec_on(
            vec![0.0; 4 * hidden_size],
            vec![4 * hidden_size],
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

        let c_prev = Tensor::from_vec_on(
            vec![0.0; batch_size * hidden_size],
            vec![batch_size, hidden_size],
            device.clone(),
        )
        .await
        .unwrap();

        let (h_next, c_next) = LSTMCell::new(
            input, weight_ih, weight_hh, bias_ih, bias_hh, h_prev, c_prev,
        )
        .unwrap()
        .execute()
        .unwrap();

        assert_eq!(h_next.shape(), &[batch_size, hidden_size]);
        assert_eq!(c_next.shape(), &[batch_size, hidden_size]);
    }
}

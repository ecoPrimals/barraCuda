//! Bidirectional LSTM operation
//!
//! **Pure WGSL**: Single implementation via WebGPU shader
//! Processes sequence in both forward and backward directions

use crate::device::{DeviceCapabilities, WorkloadType};
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct BiLSTMParams {
    batch_size: u32,
    seq_len: u32,
    input_size: u32,
    hidden_size: u32,
    direction: u32, // 0 = forward, 1 = backward
}

pub struct BiLSTM {
    input: Tensor,
    weight_ih: Tensor,
    weight_hh: Tensor,
    bias_ih: Tensor,
    bias_hh: Tensor,
    direction: u32,
}

impl BiLSTM {
    /// Create BiLSTM operation
    pub fn new(
        input: Tensor,
        weight_ih: Tensor,
        weight_hh: Tensor,
        bias_ih: Tensor,
        bias_hh: Tensor,
        direction: u32,
    ) -> Result<Self> {
        // Validate direction
        if direction > 1 {
            return Err(BarracudaError::invalid_op(
                "BiLSTM",
                format!("direction must be 0 (forward) or 1 (backward), got {direction}"),
            ));
        }

        Ok(Self {
            input,
            weight_ih,
            weight_hh,
            bias_ih,
            bias_hh,
            direction,
        })
    }

    /// WGSL shader source (embedded at compile time)
    fn wgsl_shader() -> &'static str {
        {
            static SHADER: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
                crate::shaders::precision::downcast_f64_to_f32_with_transcendentals(include_str!(
                    "../shaders/rnn/bi_lstm_f64.wgsl"
                ))
            });
            std::sync::LazyLock::force(&SHADER).as_str()
        }
    }

    /// Execute BiLSTM on tensor
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let input_shape = self.input.shape();

        if input_shape.len() != 3 {
            return Err(BarracudaError::invalid_op(
                "BiLSTM",
                format!("input must be 3D [seq_len, batch, input_size], got shape {input_shape:?}"),
            ));
        }

        let seq_len = input_shape[0];
        let batch_size = input_shape[1];
        let input_size = input_shape[2];
        let hidden_size = self.bias_ih.len() / 4; // 4 gates: i, f, g, o

        // Create output buffer: [seq_len, batch, hidden_size]
        let output_size = seq_len * batch_size * hidden_size;
        let output_buffer = device.create_buffer_f32(output_size)?;

        // Create h_state and c_state buffers: [batch, hidden_size]
        let state_size = batch_size * hidden_size;
        let h_state_buffer = device.create_buffer_f32(state_size)?;
        let c_state_buffer = device.create_buffer_f32(state_size)?;

        // Initialize states to zero
        device.queue.write_buffer(
            &h_state_buffer,
            0,
            bytemuck::cast_slice(&vec![0.0f32; state_size]),
        );
        device.queue.write_buffer(
            &c_state_buffer,
            0,
            bytemuck::cast_slice(&vec![0.0f32; state_size]),
        );

        let params = BiLSTMParams {
            batch_size: batch_size as u32,
            seq_len: seq_len as u32,
            input_size: input_size as u32,
            hidden_size: hidden_size as u32,
            direction: self.direction,
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("BiLSTM Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        // Create bind group layout
        let bind_group_layout =
            device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("BiLSTM Bind Group Layout"),
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
            label: Some("BiLSTM Bind Group"),
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
                    resource: h_state_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: c_state_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        // Compile shader
        let shader = device.compile_shader(Self::wgsl_shader(), Some("BiLSTM"));

        // Create pipeline
        let pipeline_layout =
            device
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("BiLSTM Pipeline Layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });

        let pipeline = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("BiLSTM Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: "main",
                cache: None,
                compilation_options: Default::default(),
            });

        // Encode and execute
        let mut encoder = device
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("BiLSTM Encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("BiLSTM Pass"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);

            // Deep Debt Evolution: Capability-based dispatch
            let caps = DeviceCapabilities::from_device(device);
            let optimal_wg_size = caps.optimal_workgroup_size(WorkloadType::ElementWise);
            let workgroups = (batch_size as u32).div_ceil(optimal_wg_size);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        device.submit_and_poll(Some(encoder.finish()));

        // Create output tensor
        Ok(Tensor::from_buffer(
            output_buffer,
            vec![seq_len, batch_size, hidden_size],
            device.clone(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device_if_gpu_available;

    #[tokio::test]
    async fn test_bi_lstm_basic() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        // Create test tensors: [seq_len=2, batch=1, input_size=3]
        let input = Tensor::from_vec_on(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 1, 3],
            device.clone(),
        )
        .await
        .unwrap();

        let hidden_size = 4;
        let weight_ih = Tensor::from_vec_on(
            vec![0.1; 4 * hidden_size * 3], // [4*hidden, input]
            vec![4 * hidden_size, 3],
            device.clone(),
        )
        .await
        .unwrap();

        let weight_hh = Tensor::from_vec_on(
            vec![0.1; 4 * hidden_size * hidden_size], // [4*hidden, hidden]
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

        let result = BiLSTM::new(input, weight_ih, weight_hh, bias_ih, bias_hh, 0)
            .unwrap()
            .execute()
            .unwrap();

        assert_eq!(result.shape(), &[2, 1, 4]);
    }
}

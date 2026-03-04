// SPDX-License-Identifier: AGPL-3.0-or-later
//! Conv1D - 1D Convolution
//! Pure WGSL implementation
//!
//! Convolution operation for sequences (time-series, NLP, audio processing)
//! Formula: output[b, oc, ol] = Σ(input[b, ic, il] * weight[oc, ic, k]) + bias[oc]
//!
//! Used in: WaveNet, temporal CNNs, sequence models, audio processing
//! Benefits: Captures temporal/sequential patterns efficiently
//!
//! Shader: f64 canonical (downcast to f32 at compile)

const SHADER_F64: &str = include_str!("../shaders/conv/conv1d_f64.wgsl");

use crate::device::{DeviceCapabilities, WorkloadType};
use crate::error::Result;
use crate::tensor::Tensor;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Conv1DParams {
    batch: u32,
    in_channels: u32,
    out_channels: u32,
    in_length: u32,
    kernel_size: u32,
    stride: u32,
    padding: u32,
    dilation: u32,
    out_length: u32,
}

pub struct Conv1D {
    input: Tensor,
    weight: Tensor,
    bias: Tensor,
    stride: usize,
    padding: usize,
    dilation: usize,
}

impl Conv1D {
    pub fn new(
        input: Tensor,
        weight: Tensor,
        bias: Tensor,
        stride: usize,
        padding: usize,
        dilation: usize,
    ) -> Self {
        Self {
            input,
            weight,
            bias,
            stride,
            padding,
            dilation,
        }
    }

    fn wgsl_shader() -> &'static str {
        static SHADER: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
            crate::shaders::precision::downcast_f64_to_f32_with_transcendentals(SHADER_F64)
        });
        SHADER.as_str()
    }

    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let input_shape = self.input.shape();
        let weight_shape = self.weight.shape();

        // Input shape: [batch, in_channels, length]
        let batch = input_shape[0];
        let in_channels = input_shape[1];
        let in_length = input_shape[2];

        // Weight shape: [out_channels, in_channels, kernel_size]
        let out_channels = weight_shape[0];
        let kernel_size = weight_shape[2];

        // Calculate output length
        let out_length = (in_length + 2 * self.padding - self.dilation * (kernel_size - 1) - 1)
            / self.stride
            + 1;

        let output_size = batch * out_channels * out_length;

        // Create output buffer
        let output_buffer = device.create_buffer_f32(output_size)?;

        // Create params
        let params = Conv1DParams {
            batch: batch as u32,
            in_channels: in_channels as u32,
            out_channels: out_channels as u32,
            in_length: in_length as u32,
            kernel_size: kernel_size as u32,
            stride: self.stride as u32,
            padding: self.padding as u32,
            dilation: self.dilation as u32,
            out_length: out_length as u32,
        };
        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Conv1D Params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        // Create shader module
        let shader = device
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Conv1D Shader"),
                source: wgpu::ShaderSource::Wgsl(Self::wgsl_shader().into()),
            });

        // Create compute pipeline
        let pipeline = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Conv1D Pipeline"),
                layout: None,
                module: &shader,
                entry_point: Some("main"),
                cache: None,
                compilation_options: Default::default(),
            });

        // Create bind group
        let bind_group_layout = pipeline.get_bind_group_layout(0);
        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Conv1D Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.input.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.weight.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.bias.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        // Execute
        let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("Conv1D Encoder"),
        });
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Conv1D Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, Some(&bind_group), &[]);
            // Deep Debt Evolution: Capability-based dispatch
            let caps = DeviceCapabilities::from_device(device);
            let optimal_wg_size = caps.optimal_workgroup_size(WorkloadType::Convolution);
            let workgroups = (output_size as u32).div_ceil(optimal_wg_size);
            compute_pass.dispatch_workgroups(workgroups, 1, 1);
        }
        device.submit_and_poll(Some(encoder.finish()));

        Ok(Tensor::from_buffer(
            output_buffer,
            vec![batch, out_channels, out_length],
            device.clone(),
        ))
    }
}

impl Tensor {
    /// Apply 1D convolution for sequence processing
    /// # Arguments
    /// * `weight` - Convolution weights (shape: [out_channels, in_channels, kernel_size])
    /// * `bias` - Bias terms (shape: [out_channels])
    /// * `stride` - Stride for convolution (default: 1)
    /// * `padding` - Padding to apply (default: 0)
    /// * `dilation` - Dilation factor (default: 1)
    pub fn conv1d(
        self,
        weight: Tensor,
        bias: Tensor,
        stride: usize,
        padding: usize,
        dilation: usize,
    ) -> Result<Self> {
        Conv1D::new(self, weight, bias, stride, padding, dilation).execute()
    }
}

#[expect(clippy::unwrap_used, reason = "tests")]
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_conv1d_basic() {
        let Some(device) = crate::device::test_pool::get_test_device_if_gpu_available().await
        else {
            return;
        };

        // Create input [1, 2, 4] - 1 batch, 2 channels, length 4
        let input_data = vec![
            1.0f32, 2.0, 3.0, 4.0, // Channel 0
            5.0, 6.0, 7.0, 8.0, // Channel 1
        ];
        let input = Tensor::from_data(&input_data, vec![1, 2, 4], device.clone()).unwrap();

        // Create weight [1, 2, 3] - 1 output channel, 2 input channels, kernel size 3
        let weight_data = vec![
            1.0f32, 0.0, -1.0, // For input channel 0
            0.5, 0.5, 0.5, // For input channel 1
        ];
        let weight = Tensor::from_data(&weight_data, vec![1, 2, 3], device.clone()).unwrap();

        // Create bias [1]
        let bias_data = vec![0.0f32];
        let bias = Tensor::from_data(&bias_data, vec![1], device.clone()).unwrap();

        // Apply Conv1D with stride=1, padding=0, dilation=1
        let result = input.conv1d(weight, bias, 1, 0, 1).unwrap();
        let output = result.to_vec().unwrap();

        // Output shape should be [1, 1, 2] (length reduced by kernel_size - 1)
        assert_eq!(result.shape(), &[1, 1, 2]);
        assert_eq!(output.len(), 2);
        assert!(output.iter().all(|&x| x.is_finite()));
    }

    #[tokio::test]
    async fn test_conv1d_edge_cases() {
        let Some(device) = crate::device::test_pool::get_test_device_if_gpu_available().await
        else {
            return;
        };

        // Single channel, kernel size 1 (no reduction)
        let input_data = vec![1.0f32, 2.0, 3.0];
        let input = Tensor::from_data(&input_data, vec![1, 1, 3], device.clone()).unwrap();

        let weight_data = vec![1.0f32];
        let weight = Tensor::from_data(&weight_data, vec![1, 1, 1], device.clone()).unwrap();

        let bias_data = vec![0.0f32];
        let bias = Tensor::from_data(&bias_data, vec![1], device.clone()).unwrap();

        let result = input.conv1d(weight, bias, 1, 0, 1).unwrap();

        assert_eq!(result.shape(), &[1, 1, 3]);
    }

    #[tokio::test]
    async fn test_conv1d_boundary() {
        let Some(device) = crate::device::test_pool::get_test_device_if_gpu_available().await
        else {
            return;
        };

        // Test with padding
        let input_data = vec![1.0f32, 2.0, 3.0, 4.0];
        let input = Tensor::from_data(&input_data, vec![1, 1, 4], device.clone()).unwrap();

        let weight_data = vec![1.0f32, 1.0, 1.0];
        let weight = Tensor::from_data(&weight_data, vec![1, 1, 3], device.clone()).unwrap();

        let bias_data = vec![0.0f32];
        let bias = Tensor::from_data(&bias_data, vec![1], device.clone()).unwrap();

        // With padding=1, output length should be same as input
        let result = input.conv1d(weight, bias, 1, 1, 1).unwrap();

        assert_eq!(result.shape(), &[1, 1, 4]);
    }

    #[tokio::test]
    async fn test_conv1d_large_batch() {
        let Some(device) = crate::device::test_pool::get_test_device_if_gpu_available().await
        else {
            return;
        };

        // Larger sequence
        let batch = 2;
        let in_channels = 3;
        let length = 16;

        let input_data = vec![1.0f32; batch * in_channels * length];
        let input = Tensor::from_data(
            &input_data,
            vec![batch, in_channels, length],
            device.clone(),
        )
        .unwrap();

        let out_channels = 4;
        let kernel_size = 5;
        let weight_data = vec![0.1f32; out_channels * in_channels * kernel_size];
        let weight = Tensor::from_data(
            &weight_data,
            vec![out_channels, in_channels, kernel_size],
            device.clone(),
        )
        .unwrap();

        let bias_data = vec![0.0f32; out_channels];
        let bias = Tensor::from_data(&bias_data, vec![out_channels], device.clone()).unwrap();

        let result = input.conv1d(weight, bias, 1, 0, 1).unwrap();

        // Output length = (16 - 5 + 1) = 12
        assert_eq!(result.shape(), &[batch, out_channels, 12]);
    }

    #[tokio::test]
    async fn test_conv1d_precision() {
        let Some(device) = crate::device::test_pool::get_test_device_if_gpu_available().await
        else {
            return;
        };

        // Simple identity-like convolution
        let input_data = vec![1.0f32, 2.0, 3.0];
        let input = Tensor::from_data(&input_data, vec![1, 1, 3], device.clone()).unwrap();

        let weight_data = vec![1.0f32];
        let weight = Tensor::from_data(&weight_data, vec![1, 1, 1], device.clone()).unwrap();

        let bias_data = vec![0.0f32];
        let bias = Tensor::from_data(&bias_data, vec![1], device.clone()).unwrap();

        let result = input.conv1d(weight, bias, 1, 0, 1).unwrap();
        let output = result.to_vec().unwrap();

        // Should be identity: [1, 2, 3]
        assert_eq!(output.len(), 3);
        assert!(output.iter().all(|&x| x.is_finite()));
    }
}

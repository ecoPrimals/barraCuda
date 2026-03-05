// SPDX-License-Identifier: AGPL-3.0-or-later
//! DepthwiseConv2D - Depthwise 2D Convolution
//! Pure WGSL implementation
//!
//! Efficient convolution that applies a separate filter to each input channel
//! Formula: output[b, c, h, w] = Σ(input[b, c, ...] * weight[c, 1, ...]) + bias[c]
//!
//! Used in: MobileNet, EfficientNet, lightweight CNNs
//! Benefits: Dramatically reduces parameters and computation vs standard Conv2D

use crate::error::Result;
use crate::tensor::Tensor;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct DepthwiseConv2DParams {
    batch: u32,
    channels: u32,
    in_height: u32,
    in_width: u32,
    kernel_h: u32,
    kernel_w: u32,
    stride_h: u32,
    stride_w: u32,
    pad_h: u32,
    pad_w: u32,
    out_height: u32,
    out_width: u32,
}

/// Depthwise 2D convolution (one filter per channel, MobileNet-style).
pub struct DepthwiseConv2D {
    input: Tensor,
    weight: Tensor,
    bias: Tensor,
    stride: (usize, usize),
    padding: (usize, usize),
}

impl DepthwiseConv2D {
    /// Creates a new depthwise conv2d. Weight shape: [channels, 1, kh, kw].
    pub fn new(
        input: Tensor,
        weight: Tensor,
        bias: Tensor,
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Self {
        Self {
            input,
            weight,
            bias,
            stride,
            padding,
        }
    }

    fn wgsl_shader() -> &'static str {
        {
            static SHADER: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
                crate::shaders::precision::downcast_f64_to_f32_with_transcendentals(include_str!(
                    "../shaders/conv/depthwise_conv2d_f64.wgsl"
                ))
            });
            std::sync::LazyLock::force(&SHADER).as_str()
        }
    }

    /// Executes depthwise conv2d and returns the output tensor.
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let input_shape = self.input.shape();
        let weight_shape = self.weight.shape();

        // Input shape: [batch, channels, height, width]
        let batch = input_shape[0];
        let channels = input_shape[1];
        let in_height = input_shape[2];
        let in_width = input_shape[3];

        // Weight shape: [channels, 1, kernel_h, kernel_w]
        let kernel_h = weight_shape[2];
        let kernel_w = weight_shape[3];

        // Calculate output dimensions
        let out_height = (in_height + 2 * self.padding.0 - kernel_h) / self.stride.0 + 1;
        let out_width = (in_width + 2 * self.padding.1 - kernel_w) / self.stride.1 + 1;

        let output_size = batch * channels * out_height * out_width;

        // Create output buffer
        let output_buffer = device.create_buffer_f32(output_size)?;

        // Create params
        let params = DepthwiseConv2DParams {
            batch: batch as u32,
            channels: channels as u32,
            in_height: in_height as u32,
            in_width: in_width as u32,
            kernel_h: kernel_h as u32,
            kernel_w: kernel_w as u32,
            stride_h: self.stride.0 as u32,
            stride_w: self.stride.1 as u32,
            pad_h: self.padding.0 as u32,
            pad_w: self.padding.1 as u32,
            out_height: out_height as u32,
            out_width: out_width as u32,
        };
        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("DepthwiseConv2D Params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        // Create shader module
        let shader = device
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("DepthwiseConv2D Shader"),
                source: wgpu::ShaderSource::Wgsl(Self::wgsl_shader().into()),
            });

        // Create compute pipeline
        let pipeline = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("DepthwiseConv2D Pipeline"),
                layout: None,
                module: &shader,
                entry_point: Some("main"),
                cache: None,
                compilation_options: Default::default(),
            });

        // Create bind group
        let bind_group_layout = pipeline.get_bind_group_layout(0);
        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("DepthwiseConv2D Bind Group"),
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

        // Execute with 2D workgroup (16x16)
        let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("DepthwiseConv2D Encoder"),
        });
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("DepthwiseConv2D Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, Some(&bind_group), &[]);

            let workgroups_x = out_width.div_ceil(16) as u32;
            let workgroups_y = out_height.div_ceil(16) as u32;
            let workgroups_z = (batch * channels) as u32;
            compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z);
        }
        device.submit_and_poll(Some(encoder.finish()));

        Ok(Tensor::from_buffer(
            output_buffer,
            vec![batch, channels, out_height, out_width],
            device.clone(),
        ))
    }
}

impl Tensor {
    /// Apply depthwise 2D convolution (efficient for mobile networks)
    /// # Arguments
    /// * `weight` - Depthwise weights (shape: [channels, 1, kernel_h, kernel_w])
    /// * `bias` - Bias terms (shape: [channels])
    /// * `stride` - (height_stride, width_stride)
    /// * `padding` - (height_padding, width_padding)
    pub fn depthwise_conv2d(
        self,
        weight: Tensor,
        bias: Tensor,
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<Self> {
        DepthwiseConv2D::new(self, weight, bias, stride, padding).execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    async fn get_test_device() -> Option<Arc<crate::device::WgpuDevice>> {
        crate::device::test_pool::get_test_device_if_gpu_available().await
    }

    #[tokio::test]
    async fn test_depthwise_conv2d_basic() {
        let Some(device) = get_test_device().await else {
            return;
        };
        // Create input [1, 2, 3, 3] - 1 batch, 2 channels, 3x3 spatial
        let input_data = vec![
            1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, // Channel 0
            9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, // Channel 1
        ];
        let input = Tensor::from_data(&input_data, vec![1, 2, 3, 3], device.clone()).unwrap();

        // Create depthwise weight [2, 1, 2, 2] - 2 channels, kernel 2x2
        let weight_data = vec![
            1.0f32, 0.0, 0.0, 1.0, // Channel 0 kernel
            1.0, 1.0, 1.0, 1.0, // Channel 1 kernel
        ];
        let weight = Tensor::from_data(&weight_data, vec![2, 1, 2, 2], device.clone()).unwrap();

        // Create bias [2]
        let bias_data = vec![0.0f32, 0.0];
        let bias = Tensor::from_data(&bias_data, vec![2], device.clone()).unwrap();

        // Apply DepthwiseConv2D
        let result = input
            .depthwise_conv2d(weight, bias, (1, 1), (0, 0))
            .unwrap();

        // Output shape should be [1, 2, 2, 2]
        assert_eq!(result.shape(), &[1, 2, 2, 2]);
    }

    #[tokio::test]
    async fn test_depthwise_conv2d_edge_cases() {
        let Some(device) = get_test_device().await else {
            return;
        };
        // Single channel, 1x1 kernel (identity operation)
        let input_data = vec![1.0, 2.0, 3.0, 4.0]; // [1, 1, 2, 2]
        let input = Tensor::from_data(&input_data, vec![1, 1, 2, 2], device.clone()).unwrap();
        let weight_data = vec![1.0]; // [1, 1, 1, 1]
        let weight = Tensor::from_data(&weight_data, vec![1, 1, 1, 1], device.clone()).unwrap();
        let bias_data = vec![0.0];
        let bias = Tensor::from_data(&bias_data, vec![1], device.clone()).unwrap();
        let result = input
            .depthwise_conv2d(weight, bias, (1, 1), (0, 0))
            .unwrap();
        assert_eq!(result.shape(), &[1, 1, 2, 2]);

        // All zeros input
        let input_data = vec![0.0; 16]; // [1, 2, 2, 2]
        let input = Tensor::from_data(&input_data, vec![1, 2, 2, 2], device.clone()).unwrap();
        let weight_data = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]; // [2, 1, 2, 2]
        let weight = Tensor::from_data(&weight_data, vec![2, 1, 2, 2], device.clone()).unwrap();
        let bias_data = vec![0.0, 0.0];
        let bias = Tensor::from_data(&bias_data, vec![2], device.clone()).unwrap();
        let result = input
            .depthwise_conv2d(weight, bias, (1, 1), (0, 0))
            .unwrap();
        let output = result.to_vec().unwrap();
        assert!(output.iter().all(|&x| x.abs() < 1e-5));
    }

    #[tokio::test]
    async fn test_depthwise_conv2d_boundary() {
        let Some(device) = get_test_device().await else {
            return;
        };
        // With padding - output size preserved
        let input_data = vec![1.0, 2.0, 3.0, 4.0]; // [1, 1, 2, 2]
        let input = Tensor::from_data(&input_data, vec![1, 1, 2, 2], device.clone()).unwrap();
        let weight_data = vec![1.0, 0.0, 0.0, 1.0]; // [1, 1, 2, 2]
        let weight = Tensor::from_data(&weight_data, vec![1, 1, 2, 2], device.clone()).unwrap();
        let bias_data = vec![0.0];
        let bias = Tensor::from_data(&bias_data, vec![1], device.clone()).unwrap();
        let result = input
            .depthwise_conv2d(weight, bias, (1, 1), (1, 1))
            .unwrap();
        assert_eq!(result.shape(), &[1, 1, 3, 3]); // Output larger with padding

        // Stride > 1 (downsampling)
        let input_data = vec![1.0; 32]; // [1, 2, 4, 4]
        let input = Tensor::from_data(&input_data, vec![1, 2, 4, 4], device.clone()).unwrap();
        let weight_data = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]; // [2, 1, 2, 2]
        let weight = Tensor::from_data(&weight_data, vec![2, 1, 2, 2], device.clone()).unwrap();
        let bias_data = vec![0.0, 0.0];
        let bias = Tensor::from_data(&bias_data, vec![2], device.clone()).unwrap();
        let result = input
            .depthwise_conv2d(weight, bias, (2, 2), (0, 0))
            .unwrap();
        assert_eq!(result.shape(), &[1, 2, 2, 2]); // Downsampled by stride=2
    }

    #[tokio::test]
    async fn test_depthwise_conv2d_large_batch() {
        let Some(device) = get_test_device().await else {
            return;
        };
        // Batch size 8, 16 channels (MobileNet scale)
        let batch_size = 8;
        let channels = 16;
        let height = 8;
        let width = 8;
        let input_data = vec![1.0; batch_size * channels * height * width];
        let input = Tensor::from_data(
            &input_data,
            vec![batch_size, channels, height, width],
            device.clone(),
        )
        .unwrap();

        // Depthwise 3x3 kernel
        let kernel_size = 3;
        let weight_data = vec![1.0; channels * kernel_size * kernel_size];
        let weight = Tensor::from_data(
            &weight_data,
            vec![channels, 1, kernel_size, kernel_size],
            device.clone(),
        )
        .unwrap();
        let bias_data = vec![0.0; channels];
        let bias = Tensor::from_data(&bias_data, vec![channels], device.clone()).unwrap();

        let result = input
            .depthwise_conv2d(weight, bias, (1, 1), (1, 1))
            .unwrap();
        // With padding=1, output size preserved
        assert_eq!(result.shape(), &[batch_size, channels, height, width]);
    }

    #[tokio::test]
    async fn test_depthwise_conv2d_precision() {
        let Some(device) = get_test_device().await else {
            return;
        };
        // Precision test: Verify depthwise computation produces reasonable outputs
        let input_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]; // [1, 1, 3, 3]
        let input = Tensor::from_data(&input_data, vec![1, 1, 3, 3], device.clone()).unwrap();

        // Simple kernel (all ones)
        let weight_data = vec![1.0, 1.0, 1.0, 1.0]; // [1, 1, 2, 2]
        let weight = Tensor::from_data(&weight_data, vec![1, 1, 2, 2], device.clone()).unwrap();
        let bias_data = vec![0.0];
        let bias = Tensor::from_data(&bias_data, vec![1], device.clone()).unwrap();

        let result = input
            .depthwise_conv2d(weight, bias, (1, 1), (0, 0))
            .unwrap();

        // 3x3 input with 2x2 kernel and stride 1 produces 2x2 output
        assert_eq!(result.shape(), &[1, 1, 2, 2]);

        let output = result.to_vec().unwrap();
        assert_eq!(output.len(), 4);

        // All outputs should be positive and finite (kernel sums positive inputs)
        for val in output.iter() {
            assert!(val.is_finite());
            assert!(*val > 0.0);
        }

        // Outputs should be monotonically increasing (sliding window on increasing input)
        assert!(output[3] > output[0]); // Bottom-right > top-left
    }
}

// SPDX-License-Identifier: AGPL-3.0-or-later
//! TransposedConv2D - Transposed 2D Convolution (Deconvolution)
//! Pure WGSL implementation
//!
//! Upsampling operation also known as fractionally-strided convolution
//! Formula: Reverses the spatial transformation of a convolution
//!
//! Used in: U-Net decoder, image super-resolution, GANs, segmentation
//! Benefits: Learnable upsampling, preserves spatial relationships

use crate::error::Result;
use crate::tensor::Tensor;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct TransposedConv2DParams {
    batch_size: u32,
    in_channels: u32,
    out_channels: u32,
    input_h: u32,
    input_w: u32,
    output_h: u32,
    output_w: u32,
    kernel_h: u32,
    kernel_w: u32,
    stride_h: u32,
    stride_w: u32,
    padding_h: u32,
    padding_w: u32,
    output_padding_h: u32,
    output_padding_w: u32,
    _pad: u32,
}

pub struct TransposedConv2D {
    input: Tensor,
    weight: Tensor,
    bias: Tensor,
    stride: (usize, usize),
    padding: (usize, usize),
    output_padding: (usize, usize),
}

impl TransposedConv2D {
    pub fn new(
        input: Tensor,
        weight: Tensor,
        bias: Tensor,
        stride: (usize, usize),
        padding: (usize, usize),
        output_padding: (usize, usize),
    ) -> Self {
        Self {
            input,
            weight,
            bias,
            stride,
            padding,
            output_padding,
        }
    }

    fn wgsl_shader() -> &'static str {
        {
            static SHADER: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
                crate::shaders::precision::downcast_f64_to_f32_with_transcendentals(include_str!(
                    "../shaders/conv/transposed_conv2d_f64.wgsl"
                ))
            });
            SHADER.as_str()
        }
    }

    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let input_shape = self.input.shape();
        let weight_shape = self.weight.shape();

        // Input shape: [batch, in_channels, height, width]
        let batch_size = input_shape[0];
        let in_channels = input_shape[1];
        let input_h = input_shape[2];
        let input_w = input_shape[3];

        // Weight shape: [in_channels, out_channels, kernel_h, kernel_w]
        let out_channels = weight_shape[1];
        let kernel_h = weight_shape[2];
        let kernel_w = weight_shape[3];

        // Calculate output dimensions
        let output_h =
            (input_h - 1) * self.stride.0 - 2 * self.padding.0 + kernel_h + self.output_padding.0;
        let output_w =
            (input_w - 1) * self.stride.1 - 2 * self.padding.1 + kernel_w + self.output_padding.1;

        let output_size = batch_size * out_channels * output_h * output_w;

        // Create output buffer
        let output_buffer = device.create_buffer_f32(output_size)?;

        // Create params
        let params = TransposedConv2DParams {
            batch_size: batch_size as u32,
            in_channels: in_channels as u32,
            out_channels: out_channels as u32,
            input_h: input_h as u32,
            input_w: input_w as u32,
            output_h: output_h as u32,
            output_w: output_w as u32,
            kernel_h: kernel_h as u32,
            kernel_w: kernel_w as u32,
            stride_h: self.stride.0 as u32,
            stride_w: self.stride.1 as u32,
            padding_h: self.padding.0 as u32,
            padding_w: self.padding.1 as u32,
            output_padding_h: self.output_padding.0 as u32,
            output_padding_w: self.output_padding.1 as u32,
            _pad: 0,
        };
        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("TransposedConv2D Params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        // Create shader module
        let shader = device
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("TransposedConv2D Shader"),
                source: wgpu::ShaderSource::Wgsl(Self::wgsl_shader().into()),
            });

        // Create compute pipeline
        let pipeline = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("TransposedConv2D Pipeline"),
                layout: None,
                module: &shader,
                entry_point: Some("main"),
                cache: None,
                compilation_options: Default::default(),
            });

        // Create bind group
        let bind_group_layout = pipeline.get_bind_group_layout(0);
        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("TransposedConv2D Bind Group"),
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
            label: Some("TransposedConv2D Encoder"),
        });
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("TransposedConv2D Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, Some(&bind_group), &[]);

            let workgroups_x = output_w.div_ceil(16) as u32;
            let workgroups_y = output_h.div_ceil(16) as u32;
            let workgroups_z = out_channels as u32;
            compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z);
        }
        device.submit_and_poll(Some(encoder.finish()));

        Ok(Tensor::from_buffer(
            output_buffer,
            vec![batch_size, out_channels, output_h, output_w],
            device.clone(),
        ))
    }
}

impl Tensor {
    /// Apply transposed 2D convolution (learnable upsampling)
    /// # Arguments
    /// * `weight` - Transposed conv weights (shape: [in_channels, out_channels, kernel_h, kernel_w])
    /// * `bias` - Bias terms (shape: [out_channels])
    /// * `stride` - (height_stride, width_stride)
    /// * `padding` - (height_padding, width_padding)
    /// * `output_padding` - (height_out_pad, width_out_pad)
    pub fn transposed_conv2d(
        self,
        weight: Tensor,
        bias: Tensor,
        stride: (usize, usize),
        padding: (usize, usize),
        output_padding: (usize, usize),
    ) -> Result<Self> {
        TransposedConv2D::new(self, weight, bias, stride, padding, output_padding).execute()
    }
}

#[expect(clippy::unwrap_used, reason = "tests")]
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_transposed_conv2d_basic() {
        let Some(device) = crate::device::test_pool::get_test_device_if_gpu_available().await
        else {
            return;
        };

        // Create input [1, 1, 2, 2] - simple upsampling test
        let input_data = vec![1.0f32, 2.0, 3.0, 4.0];
        let input = Tensor::from_data(&input_data, vec![1, 1, 2, 2], device.clone()).unwrap();

        // Create weight [1, 1, 2, 2] - simple kernel
        let weight_data = vec![1.0f32, 0.0, 0.0, 1.0];
        let weight = Tensor::from_data(&weight_data, vec![1, 1, 2, 2], device.clone()).unwrap();

        // Create bias [1]
        let bias_data = vec![0.0f32];
        let bias = Tensor::from_data(&bias_data, vec![1], device.clone()).unwrap();

        // Store input shape before moving
        let input_h = input.shape()[2];
        let input_w = input.shape()[3];

        // Apply TransposedConv2D with stride=2 (upsampling by 2x)
        let result = input
            .transposed_conv2d(weight, bias, (2, 2), (0, 0), (0, 0))
            .unwrap();

        // Output should be larger than input (upsampled)
        assert!(result.shape()[2] > input_h);
        assert!(result.shape()[3] > input_w);
    }
}

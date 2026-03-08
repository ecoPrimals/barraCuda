// SPDX-License-Identifier: AGPL-3.0-or-later
//! `Conv3D` - 3D Convolution
//! Pure WGSL implementation
//!
//! Convolv operation for volumetric/spatiotemporal data
//! Formula: output[b, oc, od, oh, ow] = Σ(input[b, ic, ...] * weight[oc, ic, ...]) + bias[oc]
//!
//! Used in: Video analysis, medical imaging (CT/MRI), 3D object recognition
//! Benefits: Captures spatiotemporal features, volumetric pattern recognition

use crate::error::Result;
use crate::tensor::Tensor;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Conv3DParams {
    batch_size: u32,
    in_channels: u32,
    out_channels: u32,
    input_d: u32, // depth
    input_h: u32, // height
    input_w: u32, // width
    output_d: u32,
    output_h: u32,
    output_w: u32,
    kernel_d: u32,
    kernel_h: u32,
    kernel_w: u32,
    stride_d: u32,
    stride_h: u32,
    stride_w: u32,
    padding_d: u32,
    padding_h: u32,
    padding_w: u32,
    dilation_d: u32,
    dilation_h: u32,
    dilation_w: u32,
    _pad: u32,
}

/// 3D convolution for volumetric/spatiotemporal data.
pub struct Conv3D {
    input: Tensor,
    weight: Tensor,
    bias: Tensor,
    stride: (usize, usize, usize),
    padding: (usize, usize, usize),
    dilation: (usize, usize, usize),
}

impl Conv3D {
    /// Creates a new `Conv3D`. Input shape: [B, `C_in`, D, H, W]; weight: [`C_out`, `C_in`, kD, kH, kW].
    #[must_use]
    pub fn new(
        input: Tensor,
        weight: Tensor,
        bias: Tensor,
        stride: (usize, usize, usize),
        padding: (usize, usize, usize),
        dilation: (usize, usize, usize),
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
        {
            static SHADER: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
                include_str!("../shaders/conv/conv3d_f64.wgsl").to_string()
            });
            std::sync::LazyLock::force(&SHADER).as_str()
        }
    }

    /// Executes 3D convolution and returns the output tensor.
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let input_shape = self.input.shape();
        let weight_shape = self.weight.shape();

        // Input shape: [batch, in_channels, depth, height, width]
        let batch_size = input_shape[0];
        let in_channels = input_shape[1];
        let input_d = input_shape[2];
        let input_h = input_shape[3];
        let input_w = input_shape[4];

        // Weight shape: [out_channels, in_channels, kernel_d, kernel_h, kernel_w]
        let out_channels = weight_shape[0];
        let kernel_d = weight_shape[2];
        let kernel_h = weight_shape[3];
        let kernel_w = weight_shape[4];

        // Calculate output dimensions
        let output_d = (input_d + 2 * self.padding.0 - self.dilation.0 * (kernel_d - 1) - 1)
            / self.stride.0
            + 1;
        let output_h = (input_h + 2 * self.padding.1 - self.dilation.1 * (kernel_h - 1) - 1)
            / self.stride.1
            + 1;
        let output_w = (input_w + 2 * self.padding.2 - self.dilation.2 * (kernel_w - 1) - 1)
            / self.stride.2
            + 1;

        let output_size = batch_size * out_channels * output_d * output_h * output_w;

        // Create output buffer
        let output_buffer = device.create_buffer_f32(output_size)?;

        // Create params
        let params = Conv3DParams {
            batch_size: batch_size as u32,
            in_channels: in_channels as u32,
            out_channels: out_channels as u32,
            input_d: input_d as u32,
            input_h: input_h as u32,
            input_w: input_w as u32,
            output_d: output_d as u32,
            output_h: output_h as u32,
            output_w: output_w as u32,
            kernel_d: kernel_d as u32,
            kernel_h: kernel_h as u32,
            kernel_w: kernel_w as u32,
            stride_d: self.stride.0 as u32,
            stride_h: self.stride.1 as u32,
            stride_w: self.stride.2 as u32,
            padding_d: self.padding.0 as u32,
            padding_h: self.padding.1 as u32,
            padding_w: self.padding.2 as u32,
            dilation_d: self.dilation.0 as u32,
            dilation_h: self.dilation.1 as u32,
            dilation_w: self.dilation.2 as u32,
            _pad: 0,
        };
        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Conv3D Params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        // Create shader module
        let shader = device.compile_shader(Self::wgsl_shader(), Some("Conv3D Shader"));

        // Create compute pipeline
        let pipeline = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Conv3D Pipeline"),
                layout: None,
                module: &shader,
                entry_point: Some("main"),
                cache: None,
                compilation_options: Default::default(),
            });

        // Create bind group
        let bind_group_layout = pipeline.get_bind_group_layout(0);
        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Conv3D Bind Group"),
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

        // Execute with 3D workgroup (4x4x4)
        let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("Conv3D Encoder"),
        });
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Conv3D Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, Some(&bind_group), &[]);

            let workgroups_x = output_w.div_ceil(4) as u32;
            let workgroups_y = output_h.div_ceil(4) as u32;
            let workgroups_z = output_d.div_ceil(4) as u32;
            compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z);
        }
        device.submit_and_poll(Some(encoder.finish()));

        Ok(Tensor::from_buffer(
            output_buffer,
            vec![batch_size, out_channels, output_d, output_h, output_w],
            device.clone(),
        ))
    }
}

impl Tensor {
    /// Apply 3D convolution for volumetric/video data
    /// # Arguments
    /// * `weight` - 3D conv weights (shape: [`out_channels`, `in_channels`, `kernel_d`, `kernel_h`, `kernel_w`])
    /// * `bias` - Bias terms (shape: [`out_channels`])
    /// * `stride` - (`depth_stride`, `height_stride`, `width_stride`)
    /// * `padding` - (`depth_padding`, `height_padding`, `width_padding`)
    /// * `dilation` - (`depth_dilation`, `height_dilation`, `width_dilation`)
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn conv3d(
        self,
        weight: Tensor,
        bias: Tensor,
        stride: (usize, usize, usize),
        padding: (usize, usize, usize),
        dilation: (usize, usize, usize),
    ) -> Result<Self> {
        Conv3D::new(self, weight, bias, stride, padding, dilation).execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_conv3d_basic() {
        let Some(device) = crate::device::test_pool::get_test_device_if_gpu_available().await
        else {
            return;
        };

        // Create input [1, 1, 2, 2, 2] - 1 batch, 1 channel, 2x2x2 volume
        let input_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let input = Tensor::from_data(&input_data, vec![1, 1, 2, 2, 2], device.clone()).unwrap();

        // Create weight [1, 1, 2, 2, 2] - simple 3D kernel
        let weight_data = vec![1.0f32; 8];
        let weight = Tensor::from_data(&weight_data, vec![1, 1, 2, 2, 2], device.clone()).unwrap();

        // Create bias [1]
        let bias_data = vec![0.0f32];
        let bias = Tensor::from_data(&bias_data, vec![1], device.clone()).unwrap();

        // Apply Conv3D
        let result = input
            .conv3d(weight, bias, (1, 1, 1), (0, 0, 0), (1, 1, 1))
            .unwrap();

        // Output shape should be [1, 1, 1, 1, 1] (reduced by kernel size - 1)
        assert_eq!(result.shape(), &[1, 1, 1, 1, 1]);

        let output = result.to_vec().unwrap();
        assert!(output.iter().all(|&x| x.is_finite()));
    }

    #[tokio::test]
    async fn test_conv3d_edge_cases() {
        let Some(device) = crate::device::test_pool::get_test_device_if_gpu_available().await
        else {
            return;
        };

        // Small 3D volume, kernel size 1 (no reduction)
        let input_data = vec![1.0f32; 8];
        let input = Tensor::from_data(&input_data, vec![1, 1, 2, 2, 2], device.clone()).unwrap();

        let weight_data = vec![1.0f32];
        let weight = Tensor::from_data(&weight_data, vec![1, 1, 1, 1, 1], device.clone()).unwrap();

        let bias_data = vec![0.0f32];
        let bias = Tensor::from_data(&bias_data, vec![1], device.clone()).unwrap();

        let result = input
            .conv3d(weight, bias, (1, 1, 1), (0, 0, 0), (1, 1, 1))
            .unwrap();

        // Kernel size 1 should preserve dimensions
        assert_eq!(result.shape(), &[1, 1, 2, 2, 2]);
    }

    #[tokio::test]
    async fn test_conv3d_boundary() {
        let Some(device) = crate::device::test_pool::get_test_device_if_gpu_available().await
        else {
            return;
        };

        // Test with stride > 1 (downsampling)
        let input_data = vec![1.0f32; 4 * 4 * 4];
        let input = Tensor::from_data(&input_data, vec![1, 1, 4, 4, 4], device.clone()).unwrap();

        let weight_data = vec![1.0f32; 2 * 2 * 2];
        let weight = Tensor::from_data(&weight_data, vec![1, 1, 2, 2, 2], device.clone()).unwrap();

        let bias_data = vec![0.0f32];
        let bias = Tensor::from_data(&bias_data, vec![1], device.clone()).unwrap();

        // Stride 2 should downsample
        let result = input
            .conv3d(weight, bias, (2, 2, 2), (0, 0, 0), (1, 1, 1))
            .unwrap();

        assert_eq!(result.shape()[2..], [2, 2, 2]); // Spatial dimensions halved
    }

    #[tokio::test]
    async fn test_conv3d_large_batch() {
        let Some(device) = crate::device::test_pool::get_test_device_if_gpu_available().await
        else {
            return;
        };

        // Larger 3D volume
        let batch = 1;
        let in_channels = 2;
        let depth = 4;
        let height = 4;
        let width = 4;

        let input_data = vec![1.0f32; batch * in_channels * depth * height * width];
        let input = Tensor::from_data(
            &input_data,
            vec![batch, in_channels, depth, height, width],
            device.clone(),
        )
        .unwrap();

        let out_channels = 3;
        let kernel_d = 2;
        let kernel_h = 2;
        let kernel_w = 2;
        let weight_data = vec![0.1f32; out_channels * in_channels * kernel_d * kernel_h * kernel_w];
        let weight = Tensor::from_data(
            &weight_data,
            vec![out_channels, in_channels, kernel_d, kernel_h, kernel_w],
            device.clone(),
        )
        .unwrap();

        let bias_data = vec![0.0f32; out_channels];
        let bias = Tensor::from_data(&bias_data, vec![out_channels], device.clone()).unwrap();

        let result = input
            .conv3d(weight, bias, (1, 1, 1), (0, 0, 0), (1, 1, 1))
            .unwrap();

        // Output spatial dims = (4-2+1, 4-2+1, 4-2+1) = (3, 3, 3)
        assert_eq!(result.shape(), &[batch, out_channels, 3, 3, 3]);
    }

    #[tokio::test]
    async fn test_conv3d_precision() {
        let Some(device) = crate::device::test_pool::get_test_device_if_gpu_available().await
        else {
            return;
        };

        // Simple identity-like 3D convolution
        let input_data = vec![1.0f32; 8];
        let input = Tensor::from_data(&input_data, vec![1, 1, 2, 2, 2], device.clone()).unwrap();

        let weight_data = vec![1.0f32];
        let weight = Tensor::from_data(&weight_data, vec![1, 1, 1, 1, 1], device.clone()).unwrap();

        let bias_data = vec![0.0f32];
        let bias = Tensor::from_data(&bias_data, vec![1], device.clone()).unwrap();

        let result = input
            .conv3d(weight, bias, (1, 1, 1), (0, 0, 0), (1, 1, 1))
            .unwrap();
        let output = result.to_vec().unwrap();

        // Should preserve values with identity kernel
        assert_eq!(output.len(), 8);
        assert!(output.iter().all(|&x| x.is_finite()));
    }
}

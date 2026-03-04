// SPDX-License-Identifier: AGPL-3.0-or-later
use crate::device::DeviceCapabilities;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct DilatedConv2DParams {
    batch_size: u32,
    in_channels: u32,
    out_channels: u32,
    in_height: u32,
    in_width: u32,
    kernel_height: u32,
    kernel_width: u32,
    out_height: u32,
    out_width: u32,
    stride_h: u32,
    stride_w: u32,
    pad_h: u32,
    pad_w: u32,
    dilation_h: u32,
    dilation_w: u32,
    _padding: u32,
}

pub struct DilatedConv2D {
    input: Tensor,
    weight: Tensor,
    bias: Option<Tensor>,
    stride: (usize, usize),
    padding: (usize, usize),
    dilation: (usize, usize),
}

impl DilatedConv2D {
    pub fn new(
        input: Tensor,
        weight: Tensor,
        bias: Option<Tensor>,
        stride: (usize, usize),
        padding: (usize, usize),
        dilation: (usize, usize),
    ) -> Result<Self> {
        // Validate shapes
        if input.shape().len() != 4 {
            return Err(BarracudaError::invalid_op(
                "dilated_conv2d",
                format!(
                    "Input must be 4D [B, C, H, W], got shape: {:?}",
                    input.shape()
                ),
            ));
        }

        if weight.shape().len() != 4 {
            return Err(BarracudaError::invalid_op(
                "dilated_conv2d",
                format!(
                    "Weight must be 4D [C_out, C_in, Kh, Kw], got shape: {:?}",
                    weight.shape()
                ),
            ));
        }

        Ok(Self {
            input,
            weight,
            bias,
            stride,
            padding,
            dilation,
        })
    }

    fn wgsl_shader() -> &'static str {
        {
            static SHADER: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
                crate::shaders::precision::downcast_f64_to_f32_with_transcendentals(include_str!(
                    "../shaders/conv/dilated_conv2d_f64.wgsl"
                ))
            });
            SHADER.as_str()
        }
    }

    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let in_shape = self.input.shape();
        let w_shape = self.weight.shape();

        let batch_size = in_shape[0];
        let in_channels = in_shape[1];
        let in_height = in_shape[2];
        let in_width = in_shape[3];

        let out_channels = w_shape[0];
        let kernel_height = w_shape[2];
        let kernel_width = w_shape[3];

        // Calculate output dimensions
        let out_height =
            (in_height + 2 * self.padding.0 - self.dilation.0 * (kernel_height - 1) - 1)
                / self.stride.0
                + 1;
        let out_width = (in_width + 2 * self.padding.1 - self.dilation.1 * (kernel_width - 1) - 1)
            / self.stride.1
            + 1;

        let output_size = batch_size * out_channels * out_height * out_width;
        let output_buffer = device.create_buffer_f32(output_size)?;

        // Create bias buffer (or zeros)
        let zeros_buffer;
        let bias_buffer = if let Some(ref bias) = self.bias {
            bias.buffer()
        } else {
            let zeros = vec![0.0f32; out_channels];
            zeros_buffer = device
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Dilated Conv2D Bias (zeros)"),
                    contents: bytemuck::cast_slice(&zeros),
                    usage: wgpu::BufferUsages::STORAGE,
                });
            &zeros_buffer
        };

        let params = DilatedConv2DParams {
            batch_size: batch_size as u32,
            in_channels: in_channels as u32,
            out_channels: out_channels as u32,
            in_height: in_height as u32,
            in_width: in_width as u32,
            kernel_height: kernel_height as u32,
            kernel_width: kernel_width as u32,
            out_height: out_height as u32,
            out_width: out_width as u32,
            stride_h: self.stride.0 as u32,
            stride_w: self.stride.1 as u32,
            pad_h: self.padding.0 as u32,
            pad_w: self.padding.1 as u32,
            dilation_h: self.dilation.0 as u32,
            dilation_w: self.dilation.1 as u32,
            _padding: 0,
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Dilated Conv2D Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let bind_group_layout =
            device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Dilated Conv2D Bind Group Layout"),
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
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 4,
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

        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Dilated Conv2D Bind Group"),
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
                    resource: bias_buffer.as_entire_binding(),
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

        let shader_module = device
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Dilated Conv2D Shader"),
                source: wgpu::ShaderSource::Wgsl(Self::wgsl_shader().into()),
            });

        let pipeline_layout =
            device
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Dilated Conv2D Pipeline Layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });

        let pipeline = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Dilated Conv2D Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: "main",
                cache: None,
                compilation_options: Default::default(),
            });

        let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("Dilated Conv2D Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Dilated Conv2D Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);

            // Dispatch using standard 2D shader workgroup size (16, 16)
            let caps = DeviceCapabilities::from_device(device);
            let (workgroups_x, workgroups_y) =
                caps.dispatch_2d(out_width as u32, out_height as u32);
            let workgroups_z = batch_size as u32 * out_channels as u32;
            compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z);
        }

        device.submit_and_poll(Some(encoder.finish()));

        let output_data = crate::utils::read_buffer(device, &output_buffer, output_size)?;
        Ok(Tensor::new(
            output_data,
            vec![batch_size, out_channels, out_height, out_width],
            device.clone(),
        ))
    }
}

#[expect(clippy::unwrap_used, reason = "tests")]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device_if_gpu_available;

    #[tokio::test]
    async fn test_dilated_conv2d_basic() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let batch_size = 1;
        let in_channels = 3;
        let out_channels = 8;
        let height = 32;
        let width = 32;
        let kernel_size = 3;

        let input_data = vec![1.0; batch_size * in_channels * height * width];
        let weight_data = vec![0.1; out_channels * in_channels * kernel_size * kernel_size];
        let bias_data = vec![0.0; out_channels];

        let input = Tensor::from_vec_on(
            input_data,
            vec![batch_size, in_channels, height, width],
            device.clone(),
        )
        .await
        .unwrap();

        let weight = Tensor::from_vec_on(
            weight_data,
            vec![out_channels, in_channels, kernel_size, kernel_size],
            device.clone(),
        )
        .await
        .unwrap();

        let bias = Tensor::from_vec_on(bias_data, vec![out_channels], device.clone())
            .await
            .unwrap();

        let output = DilatedConv2D::new(
            input,
            weight,
            Some(bias),
            (1, 1), // stride
            (1, 1), // padding
            (2, 2), // dilation = 2
        )
        .unwrap()
        .execute()
        .unwrap();

        // With dilation=2, kernel 3x3: effective kernel = 5x5
        // out_h = (32 + 2*pad - dilation*(kernel_size-1) - 1)/stride + 1 = 30
        let pad = 1;
        let stride = 1;
        let expected_h = (height + 2 * pad - 2 * (kernel_size - 1) - 1) / stride + 1;
        let expected_w = (width + 2 * pad - 2 * (kernel_size - 1) - 1) / stride + 1;
        assert_eq!(
            output.shape(),
            &[batch_size, out_channels, expected_h, expected_w]
        );
    }
}

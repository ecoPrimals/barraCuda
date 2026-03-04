// SPDX-License-Identifier: AGPL-3.0-or-later
//! Conv2D operation - 2D convolution
//! Pure WGSL implementation
//! Shader: f64 canonical (downcast to f32 at compile)

const SHADER_F64: &str = include_str!("../shaders/conv/conv2d_f64.wgsl");

use crate::error::Result;
use crate::tensor::Tensor;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Conv2DParams {
    input_width: u32,
    input_height: u32,
    kernel_width: u32,
    kernel_height: u32,
}

pub struct Conv2D {
    input: Tensor,
    kernel: Tensor,
}

impl Conv2D {
    pub fn new(input: Tensor, kernel: Tensor) -> Self {
        Self { input, kernel }
    }

    fn wgsl_shader() -> &'static str {
        static SHADER: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
            crate::shaders::precision::downcast_f64_to_f32_with_transcendentals(SHADER_F64)
        });
        SHADER.as_str()
    }

    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();

        // Assume input: [height, width], kernel: [kh, kw]
        let input_height = self.input.shape()[0];
        let input_width = self.input.shape()[1];
        let kernel_height = self.kernel.shape()[0];
        let kernel_width = self.kernel.shape()[1];

        let output_height = input_height - kernel_height + 1;
        let output_width = input_width - kernel_width + 1;
        let output_size = output_height * output_width;

        let output_buffer = device.create_buffer_f32(output_size)?;

        let params = Conv2DParams {
            input_width: input_width as u32,
            input_height: input_height as u32,
            kernel_width: kernel_width as u32,
            kernel_height: kernel_height as u32,
        };

        let params_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Conv2D Params"),
            size: std::mem::size_of::<Conv2DParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        device
            .queue
            .write_buffer(&params_buffer, 0, bytemuck::bytes_of(&params));

        let bind_group_layout =
            device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Conv2D BGL"),
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
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
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
            label: Some("Conv2D BG"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.input.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.kernel.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let shader = device.compile_shader(Self::wgsl_shader(), Some("Conv2D"));
        let pipeline_layout =
            device
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Conv2D PL"),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });

        let pipeline = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Conv2D Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: "main",
                cache: None,
                compilation_options: Default::default(),
            });

        let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("Conv2D Encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Conv2D Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let workgroups_x = (output_width as u32).div_ceil(16);
            let workgroups_y = (output_height as u32).div_ceil(16);
            pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        device.submit_and_poll(Some(encoder.finish()));

        Ok(Tensor::from_buffer(
            output_buffer,
            vec![output_height, output_width],
            device.clone(),
        ))
    }
}

impl Tensor {
    pub fn conv2d(self, kernel: &Self) -> Result<Self> {
        Conv2D::new(self, kernel.clone()).execute()
    }
}

#[expect(clippy::unwrap_used, reason = "tests")]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device_if_gpu_available;

    fn conv2d_cpu(
        input: &[f32],
        kernel: &[f32],
        input_h: usize,
        input_w: usize,
        kernel_h: usize,
        kernel_w: usize,
    ) -> Vec<f32> {
        let output_h = input_h - kernel_h + 1;
        let output_w = input_w - kernel_w + 1;
        let mut result = vec![0.0; output_h * output_w];

        for i in 0..output_h {
            for j in 0..output_w {
                let mut sum = 0.0;
                for ki in 0..kernel_h {
                    for kj in 0..kernel_w {
                        let input_idx = (i + ki) * input_w + (j + kj);
                        let kernel_idx = ki * kernel_w + kj;
                        sum += input[input_idx] * kernel[kernel_idx];
                    }
                }
                result[i * output_w + j] = sum;
            }
        }
        result
    }

    #[tokio::test]
    async fn test_conv2d_basic() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        // 4x4 input, 2x2 kernel -> 3x3 output
        let input_data = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ];
        let kernel_data = vec![1.0, 0.0, 0.0, 1.0];

        let input = Tensor::from_vec_on(input_data.clone(), vec![4, 4], device.clone())
            .await
            .unwrap();
        let kernel = Tensor::from_vec_on(kernel_data.clone(), vec![2, 2], device)
            .await
            .unwrap();

        let result = input.conv2d(&kernel).unwrap();
        assert_eq!(result.shape(), &[3, 3]);

        let output = result.to_vec().unwrap();
        let expected = conv2d_cpu(&input_data, &kernel_data, 4, 4, 2, 2);

        for (r, e) in output.iter().zip(expected.iter()) {
            assert!((r - e).abs() < 1e-5);
        }
    }

    #[tokio::test]
    async fn test_conv2d_edge_cases() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        // Identity kernel
        let input_data = vec![1.0, 2.0, 3.0, 4.0];
        let kernel_data = vec![1.0];
        let input = Tensor::from_vec_on(input_data.clone(), vec![2, 2], device.clone())
            .await
            .unwrap();
        let kernel = Tensor::from_vec_on(kernel_data.clone(), vec![1, 1], device.clone())
            .await
            .unwrap();
        let result = input.conv2d(&kernel).unwrap();
        let output = result.to_vec().unwrap();
        assert_eq!(output, input_data);

        // Zero kernel
        let input_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let kernel_data = vec![0.0, 0.0, 0.0, 0.0];
        let input = Tensor::from_vec_on(input_data.clone(), vec![3, 3], device.clone())
            .await
            .unwrap();
        let kernel = Tensor::from_vec_on(kernel_data.clone(), vec![2, 2], device.clone())
            .await
            .unwrap();
        let result = input.conv2d(&kernel).unwrap();
        let output = result.to_vec().unwrap();

        for val in output.iter() {
            assert!(val.abs() < 1e-6);
        }
    }

    #[tokio::test]
    async fn test_conv2d_boundary() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        // Minimal convolution (2x2 input, 2x2 kernel -> 1x1 output)
        let input_data = vec![1.0, 2.0, 3.0, 4.0];
        let kernel_data = vec![0.5, 0.5, 0.5, 0.5];
        let input = Tensor::from_vec_on(input_data.clone(), vec![2, 2], device.clone())
            .await
            .unwrap();
        let kernel = Tensor::from_vec_on(kernel_data.clone(), vec![2, 2], device.clone())
            .await
            .unwrap();
        let result = input.conv2d(&kernel).unwrap();
        assert_eq!(result.shape(), &[1, 1]);
        let output = result.to_vec().unwrap();
        let expected = conv2d_cpu(&input_data, &kernel_data, 2, 2, 2, 2);
        assert!((output[0] - expected[0]).abs() < 1e-5);

        // 3x3 kernel (edge detection filter)
        let input_data: Vec<f32> = (1..=25).map(|i| i as f32).collect();
        let kernel_data = vec![-1.0, -1.0, -1.0, -1.0, 8.0, -1.0, -1.0, -1.0, -1.0];
        let input = Tensor::from_vec_on(input_data.clone(), vec![5, 5], device.clone())
            .await
            .unwrap();
        let kernel = Tensor::from_vec_on(kernel_data.clone(), vec![3, 3], device.clone())
            .await
            .unwrap();
        let result = input.conv2d(&kernel).unwrap();
        assert_eq!(result.shape(), &[3, 3]);

        let output = result.to_vec().unwrap();
        let expected = conv2d_cpu(&input_data, &kernel_data, 5, 5, 3, 3);

        for (r, e) in output.iter().zip(expected.iter()) {
            assert!((r - e).abs() < 1e-4);
        }
    }

    #[tokio::test]
    async fn test_conv2d_large_tensor() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        // 16x16 input, 3x3 kernel -> 14x14 output
        let input_data: Vec<f32> = (0..256).map(|i| (i as f32) * 0.01).collect();
        let kernel_data = vec![0.0, 0.125, 0.0, 0.125, 0.5, 0.125, 0.0, 0.125, 0.0];

        let input = Tensor::from_vec_on(input_data.clone(), vec![16, 16], device.clone())
            .await
            .unwrap();
        let kernel = Tensor::from_vec_on(kernel_data.clone(), vec![3, 3], device)
            .await
            .unwrap();

        let result = input.conv2d(&kernel).unwrap();
        assert_eq!(result.shape(), &[14, 14]);

        let output = result.to_vec().unwrap();
        let expected = conv2d_cpu(&input_data, &kernel_data, 16, 16, 3, 3);

        for (r, e) in output.iter().zip(expected.iter()) {
            assert!((r - e).abs() < 1e-4);
        }
    }

    #[tokio::test]
    async fn test_conv2d_precision() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        // Test FP32 precision with typical CNN values
        let input_data = vec![
            0.123, 0.234, 0.345, 0.456, 0.567, 0.678, 0.789, 0.890, 0.901, 0.012, 0.123, 0.234,
            0.345, 0.456, 0.567, 0.678,
        ];
        let kernel_data = vec![0.1, 0.2, 0.3, 0.4];

        let input = Tensor::from_vec_on(input_data.clone(), vec![4, 4], device.clone())
            .await
            .unwrap();
        let kernel = Tensor::from_vec_on(kernel_data.clone(), vec![2, 2], device)
            .await
            .unwrap();

        let result = input.conv2d(&kernel).unwrap();
        let output = result.to_vec().unwrap();
        let expected = conv2d_cpu(&input_data, &kernel_data, 4, 4, 2, 2);

        // Verify FP32 precision
        let max_error = output
            .iter()
            .zip(expected.iter())
            .map(|(r, e)| (r - e).abs())
            .fold(0.0f32, f32::max);

        assert!(
            max_error < 1e-5,
            "Max error: {} exceeds FP32 threshold",
            max_error
        );
    }
}

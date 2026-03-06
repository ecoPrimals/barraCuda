// SPDX-License-Identifier: AGPL-3.0-or-later
//! `AvgPool2D` operation - Average pooling for 2D tensors
//! Pure WGSL implementation

use crate::error::Result;
use crate::tensor::Tensor;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct AvgPool2DParams {
    input_width: u32,
    input_height: u32,
    pool_size: u32,
    stride: u32,
    pad_h: u32,
    pad_w: u32,
}

/// Average pooling for 2D tensors (WGSL implementation).
pub struct AvgPool2D {
    input: Tensor,
    pool_size: usize,
    stride: usize,
    pad_h: usize,
    pad_w: usize,
}

impl AvgPool2D {
    /// Create average pooling without padding.
    #[must_use]
    pub fn new(input: Tensor, pool_size: usize, stride: usize) -> Self {
        Self {
            input,
            pool_size,
            stride,
            pad_h: 0,
            pad_w: 0,
        }
    }

    /// Create average pooling with explicit padding.
    #[must_use]
    pub fn with_padding(
        input: Tensor,
        pool_size: usize,
        stride: usize,
        pad_h: usize,
        pad_w: usize,
    ) -> Self {
        Self {
            input,
            pool_size,
            stride,
            pad_h,
            pad_w,
        }
    }

    fn wgsl_shader() -> &'static str {
        {
            static SHADER: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
                crate::shaders::precision::downcast_f64_to_f32(include_str!(
                    "../shaders/pooling/avgpool2d_f64.wgsl"
                ))
            });
            std::sync::LazyLock::force(&SHADER).as_str()
        }
    }

    /// Execute average pooling and return the output tensor.
    /// # Errors
    /// Returns [`Err`] if buffer allocation fails, GPU dispatch fails, or the device is lost.
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();

        let input_height = self.input.shape()[0];
        let input_width = self.input.shape()[1];
        let output_height = (input_height + 2 * self.pad_h - self.pool_size) / self.stride + 1;
        let output_width = (input_width + 2 * self.pad_w - self.pool_size) / self.stride + 1;
        let output_size = output_height * output_width;

        let output_buffer = device.create_buffer_f32(output_size)?;

        let params = AvgPool2DParams {
            input_width: input_width as u32,
            input_height: input_height as u32,
            pool_size: self.pool_size as u32,
            stride: self.stride as u32,
            pad_h: self.pad_h as u32,
            pad_w: self.pad_w as u32,
        };

        let params_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("AvgPool2D Params"),
            size: std::mem::size_of::<AvgPool2DParams>() as u64,
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
                    label: Some("AvgPool2D BGL"),
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
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
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
            label: Some("AvgPool2D BG"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.input.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        let shader = device.compile_shader(Self::wgsl_shader(), Some("AvgPool2D"));
        let pipeline_layout =
            device
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("AvgPool2D PL"),
                    bind_group_layouts: &[&bind_group_layout],
                    immediate_size: 0,
                });

        let pipeline = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("AvgPool2D Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                cache: None,
                compilation_options: Default::default(),
            });

        let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("AvgPool2D Encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("AvgPool2D Pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&bind_group), &[]);
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
    /// Apply 2D average pooling.
    /// # Errors
    /// Returns [`Err`] if buffer allocation fails, GPU dispatch fails, or the device is lost.
    pub fn avgpool2d(self, pool_size: usize, stride: usize) -> Result<Self> {
        AvgPool2D::new(self, pool_size, stride).execute()
    }

    /// Apply 2D average pooling with explicit padding.
    /// # Errors
    /// Returns [`Err`] if buffer allocation fails, GPU dispatch fails, or the device is lost.
    pub fn avgpool2d_padded(
        self,
        pool_size: usize,
        stride: usize,
        pad_h: usize,
        pad_w: usize,
    ) -> Result<Self> {
        AvgPool2D::with_padding(self, pool_size, stride, pad_h, pad_w).execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device_if_gpu_available;

    fn avgpool2d_cpu(
        input: &[f32],
        input_h: usize,
        input_w: usize,
        pool_size: usize,
        stride: usize,
    ) -> Vec<f32> {
        avgpool2d_cpu_padded(input, input_h, input_w, pool_size, stride, 0, 0)
    }

    fn avgpool2d_cpu_padded(
        input: &[f32],
        input_h: usize,
        input_w: usize,
        pool_size: usize,
        stride: usize,
        pad_h: usize,
        pad_w: usize,
    ) -> Vec<f32> {
        let output_h = (input_h + 2 * pad_h - pool_size) / stride + 1;
        let output_w = (input_w + 2 * pad_w - pool_size) / stride + 1;
        let mut result = vec![0.0; output_h * output_w];

        for i in 0..output_h {
            for j in 0..output_w {
                let mut sum = 0.0;
                let mut count = 0;
                for pi in 0..pool_size {
                    for pj in 0..pool_size {
                        let in_i = (i * stride + pi) as isize - pad_h as isize;
                        let in_j = (j * stride + pj) as isize - pad_w as isize;
                        if in_i >= 0
                            && (in_i as usize) < input_h
                            && in_j >= 0
                            && (in_j as usize) < input_w
                        {
                            sum += input[in_i as usize * input_w + in_j as usize];
                            count += 1;
                        }
                    }
                }
                result[i * output_w + j] = sum / count as f32;
            }
        }
        result
    }

    #[tokio::test]
    async fn test_avgpool2d_basic() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        // 4x4 input, 2x2 pool with stride 2 -> 2x2 output
        let input_data = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
        ];

        let input = Tensor::from_vec_on(input_data.clone(), vec![4, 4], device)
            .await
            .unwrap();

        let result = input.avgpool2d(2, 2).unwrap();
        assert_eq!(result.shape(), &[2, 2]);

        let output = result.to_vec().unwrap();
        let expected = avgpool2d_cpu(&input_data, 4, 4, 2, 2);

        for (r, e) in output.iter().zip(expected.iter()) {
            assert!((r - e).abs() < 1e-5);
        }
    }

    #[tokio::test]
    async fn test_avgpool2d_edge_cases() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        // All same values
        let input_data = vec![7.0; 16];
        let input = Tensor::from_vec_on(input_data.clone(), vec![4, 4], device.clone())
            .await
            .unwrap();
        let result = input.avgpool2d(2, 2).unwrap();
        let output = result.to_vec().unwrap();

        for val in &output {
            assert!((val - 7.0).abs() < 1e-5);
        }

        // Mixed positive and negative
        let input_data = vec![
            -1.0, 2.0, -3.0, 4.0, 5.0, -6.0, 7.0, -8.0, -9.0, 10.0, -11.0, 12.0, 13.0, -14.0, 15.0,
            -16.0,
        ];
        let input = Tensor::from_vec_on(input_data.clone(), vec![4, 4], device.clone())
            .await
            .unwrap();
        let result = input.avgpool2d(2, 2).unwrap();
        let output = result.to_vec().unwrap();
        let expected = avgpool2d_cpu(&input_data, 4, 4, 2, 2);

        for (r, e) in output.iter().zip(expected.iter()) {
            assert!((r - e).abs() < 1e-5);
        }
    }

    #[tokio::test]
    async fn test_avgpool2d_boundary() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        // Small 2x2 input
        let input_data = vec![1.0, 2.0, 3.0, 4.0];
        let input = Tensor::from_vec_on(input_data.clone(), vec![2, 2], device.clone())
            .await
            .unwrap();
        let result = input.avgpool2d(2, 2).unwrap();
        assert_eq!(result.shape(), &[1, 1]);
        let output = result.to_vec().unwrap();
        let expected = avgpool2d_cpu(&input_data, 2, 2, 2, 2);
        assert!((output[0] - expected[0]).abs() < 1e-5);

        // Large stride relative to pool size
        let input_data: Vec<f32> = (1..=64).map(|i| i as f32).collect();
        let input = Tensor::from_vec_on(input_data.clone(), vec![8, 8], device.clone())
            .await
            .unwrap();
        let result = input.avgpool2d(2, 4).unwrap();
        assert_eq!(result.shape(), &[2, 2]);

        let output = result.to_vec().unwrap();
        let expected = avgpool2d_cpu(&input_data, 8, 8, 2, 4);

        for (r, e) in output.iter().zip(expected.iter()) {
            assert!((r - e).abs() < 1e-5);
        }
    }

    #[tokio::test]
    async fn test_avgpool2d_large_tensor() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        // 32x32 input, 2x2 pool with stride 2 -> 16x16 output
        let input_data: Vec<f32> = (0..1024).map(|i| (i as f32) * 0.1).collect();
        let input = Tensor::from_vec_on(input_data.clone(), vec![32, 32], device)
            .await
            .unwrap();

        let result = input.avgpool2d(2, 2).unwrap();
        assert_eq!(result.shape(), &[16, 16]);

        let output = result.to_vec().unwrap();
        let expected = avgpool2d_cpu(&input_data, 32, 32, 2, 2);

        for (r, e) in output.iter().zip(expected.iter()) {
            assert!((r - e).abs() < 1e-4);
        }
    }

    #[tokio::test]
    async fn test_avgpool2d_precision() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        // Test FP32 precision with typical CNN values
        let input_data = vec![
            1.234, 2.345, 3.456, 4.567, 5.678, 6.789, 7.890, 8.901, 9.012, 10.123, 11.234, 12.345,
            13.456, 14.567, 15.678, 16.789,
        ];

        let input = Tensor::from_vec_on(input_data.clone(), vec![4, 4], device)
            .await
            .unwrap();
        let result = input.avgpool2d(2, 2).unwrap();
        let output = result.to_vec().unwrap();
        let expected = avgpool2d_cpu(&input_data, 4, 4, 2, 2);

        // Verify FP32 precision
        let max_error = output
            .iter()
            .zip(expected.iter())
            .map(|(r, e)| (r - e).abs())
            .fold(0.0f32, f32::max);

        assert!(
            max_error < 1e-5,
            "Max error: {max_error} exceeds FP32 threshold"
        );
    }

    #[tokio::test]
    async fn test_avgpool2d_with_padding() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        // 4x4 input, pool=3, stride=1, pad=1 -> 4x4 output
        let input_data: Vec<f32> = (1..=16).map(|i| i as f32).collect();

        let input = Tensor::from_vec_on(input_data.clone(), vec![4, 4], device)
            .await
            .unwrap();

        let result = input.avgpool2d_padded(3, 1, 1, 1).unwrap();
        assert_eq!(result.shape(), &[4, 4]);

        let output = result.to_vec().unwrap();
        let expected = avgpool2d_cpu_padded(&input_data, 4, 4, 3, 1, 1, 1);

        for (i, (r, e)) in output.iter().zip(expected.iter()).enumerate() {
            assert!((r - e).abs() < 1e-5, "mismatch at {i}: gpu={r}, cpu={e}");
        }
    }
}

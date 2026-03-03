// SPDX-License-Identifier: AGPL-3.0-only

//! Full NCHW Conv2D GPU orchestrator.
//!
//! Wraps `ops/nn/conv2d.wgsl` to support stride, padding, dilation, groups, and
//! bias on the GPU via a single flat dispatch.

use std::sync::Arc;

use wgpu::util::DeviceExt;

use crate::device::WgpuDevice;
use crate::error::Result;
use crate::tensor::Tensor;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct Conv2dGpuParams {
    n: u32,
    c_in: u32,
    h_in: u32,
    w_in: u32,
    c_out: u32,
    k_h: u32,
    k_w: u32,
    stride_h: u32,
    stride_w: u32,
    pad_h: u32,
    pad_w: u32,
    dil_h: u32,
    dil_w: u32,
    h_out: u32,
    w_out: u32,
    groups: u32,
}

/// Full NCHW Conv2D on GPU with stride, padding, dilation, groups.
pub struct Conv2dGpu {
    pub input: Tensor,
    pub kernel: Tensor,
    pub bias: Option<Tensor>,
    pub stride: (usize, usize),
    pub padding: (usize, usize),
    pub dilation: (usize, usize),
    pub groups: usize,
}

impl Conv2dGpu {
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let in_shape = self.input.shape();
        assert!(in_shape.len() == 4, "Conv2dGpu input must be [N,C,H,W]");

        let (n, c_in, h_in, w_in) = (in_shape[0], in_shape[1], in_shape[2], in_shape[3]);
        let k_shape = self.kernel.shape();
        let (c_out, _c_in_per_group, k_h, k_w) = (k_shape[0], k_shape[1], k_shape[2], k_shape[3]);

        let h_out =
            (h_in + 2 * self.padding.0 - self.dilation.0 * (k_h - 1) - 1) / self.stride.0 + 1;
        let w_out =
            (w_in + 2 * self.padding.1 - self.dilation.1 * (k_w - 1) - 1) / self.stride.1 + 1;

        let output_size = n * c_out * h_out * w_out;
        let output_buffer = device.create_buffer_f32(output_size)?;

        let zero_bias;
        let bias_buffer: &wgpu::Buffer = if let Some(ref bias_t) = self.bias {
            bias_t.buffer()
        } else {
            zero_bias = Self::zeros_buffer(device, c_out);
            &zero_bias
        };

        let params = Conv2dGpuParams {
            n: n as u32,
            c_in: c_in as u32,
            h_in: h_in as u32,
            w_in: w_in as u32,
            c_out: c_out as u32,
            k_h: k_h as u32,
            k_w: k_w as u32,
            stride_h: self.stride.0 as u32,
            stride_w: self.stride.1 as u32,
            pad_h: self.padding.0 as u32,
            pad_w: self.padding.1 as u32,
            dil_h: self.dilation.0 as u32,
            dil_w: self.dilation.1 as u32,
            h_out: h_out as u32,
            w_out: w_out as u32,
            groups: self.groups as u32,
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("conv2d_gpu params"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let shader = device.compile_shader(include_str!("conv2d.wgsl"), Some("conv2d_gpu"));

        let bgl = device
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("conv2d_gpu BGL"),
                entries: &[
                    storage_entry(0, true),
                    storage_entry(1, true),
                    storage_entry(2, true),
                    storage_entry(3, false),
                    uniform_entry(4),
                ],
            });

        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("conv2d_gpu BG"),
            layout: &bgl,
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

        let pipeline_layout =
            device
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("conv2d_gpu PL"),
                    bind_group_layouts: &[&bgl],
                    push_constant_ranges: &[],
                });

        let pipeline = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("conv2d_gpu"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: "main",
                cache: None,
                compilation_options: Default::default(),
            });

        let mut encoder = device
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("conv2d_gpu"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("conv2d_gpu"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((output_size as u32).div_ceil(256), 1, 1);
        }
        device.submit_and_poll(Some(encoder.finish()));

        Ok(Tensor::from_buffer(
            output_buffer,
            vec![n, c_out, h_out, w_out],
            device.clone(),
        ))
    }

    fn zeros_buffer(device: &Arc<WgpuDevice>, len: usize) -> wgpu::Buffer {
        let zeros = vec![0.0f32; len];
        device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("conv2d_gpu zero bias"),
                contents: bytemuck::cast_slice(&zeros),
                usage: wgpu::BufferUsages::STORAGE,
            })
    }
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device_if_gpu_available;

    fn conv2d_cpu_nchw(
        input: &[f32],
        kernel: &[f32],
        bias: &[f32],
        n: usize,
        c_in: usize,
        h_in: usize,
        w_in: usize,
        c_out: usize,
        k_h: usize,
        k_w: usize,
        stride: (usize, usize),
        pad: (usize, usize),
        dil: (usize, usize),
        groups: usize,
    ) -> Vec<f32> {
        let h_out = (h_in + 2 * pad.0 - dil.0 * (k_h - 1) - 1) / stride.0 + 1;
        let w_out = (w_in + 2 * pad.1 - dil.1 * (k_w - 1) - 1) / stride.1 + 1;
        let c_in_per_group = c_in / groups;
        let c_out_per_group = c_out / groups;
        let mut output = vec![0.0f32; n * c_out * h_out * w_out];

        for batch in 0..n {
            for co in 0..c_out {
                let group = co / c_out_per_group;
                let c_in_start = group * c_in_per_group;
                for oh in 0..h_out {
                    for ow in 0..w_out {
                        let mut sum = 0.0f32;
                        for ci in 0..c_in_per_group {
                            for ky in 0..k_h {
                                for kx in 0..k_w {
                                    let ih = oh as isize * stride.0 as isize
                                        + ky as isize * dil.0 as isize
                                        - pad.0 as isize;
                                    let iw = ow as isize * stride.1 as isize
                                        + kx as isize * dil.1 as isize
                                        - pad.1 as isize;
                                    if ih >= 0
                                        && (ih as usize) < h_in
                                        && iw >= 0
                                        && (iw as usize) < w_in
                                    {
                                        let i_idx = batch * c_in * h_in * w_in
                                            + (c_in_start + ci) * h_in * w_in
                                            + ih as usize * w_in
                                            + iw as usize;
                                        let k_idx = co * c_in_per_group * k_h * k_w
                                            + ci * k_h * k_w
                                            + ky * k_w
                                            + kx;
                                        sum += input[i_idx] * kernel[k_idx];
                                    }
                                }
                            }
                        }
                        let o_idx =
                            batch * c_out * h_out * w_out + co * h_out * w_out + oh * w_out + ow;
                        output[o_idx] = sum + bias[co];
                    }
                }
            }
        }
        output
    }

    #[tokio::test]
    async fn test_conv2d_gpu_stride_pad() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let (n, c_in, h, w) = (1, 3, 8, 8);
        let (c_out, k_h, k_w) = (4, 3, 3);
        let stride = (2, 2);
        let pad = (1, 1);
        let dil = (1, 1);
        let groups = 1;

        let input_data: Vec<f32> = (0..n * c_in * h * w).map(|i| (i as f32) * 0.01).collect();
        let kernel_data: Vec<f32> = (0..c_out * c_in * k_h * k_w)
            .map(|i| ((i % 7) as f32 - 3.0) * 0.1)
            .collect();
        let bias_data: Vec<f32> = (0..c_out).map(|i| i as f32 * 0.05).collect();

        let expected = conv2d_cpu_nchw(
            &input_data,
            &kernel_data,
            &bias_data,
            n,
            c_in,
            h,
            w,
            c_out,
            k_h,
            k_w,
            stride,
            pad,
            dil,
            groups,
        );

        let input_t = Tensor::from_vec_on(input_data, vec![n, c_in, h, w], device.clone())
            .await
            .unwrap();
        let kernel_t =
            Tensor::from_vec_on(kernel_data, vec![c_out, c_in, k_h, k_w], device.clone())
                .await
                .unwrap();
        let bias_t = Tensor::from_vec_on(bias_data, vec![c_out], device.clone())
            .await
            .unwrap();

        let op = Conv2dGpu {
            input: input_t,
            kernel: kernel_t,
            bias: Some(bias_t),
            stride,
            padding: pad,
            dilation: dil,
            groups,
        };
        let result = op.execute().unwrap();

        let h_out = (h + 2 * pad.0 - k_h) / stride.0 + 1;
        let w_out = (w + 2 * pad.1 - k_w) / stride.1 + 1;
        assert_eq!(result.shape(), &[n, c_out, h_out, w_out]);

        let got = result.to_vec().unwrap();
        for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
            assert!(
                (g - e).abs() < 1e-3,
                "mismatch at {i}: got {g}, expected {e}"
            );
        }
    }

    #[tokio::test]
    async fn test_conv2d_gpu_dilation() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let (n, c_in, h, w) = (1, 1, 7, 7);
        let (c_out, k_h, k_w) = (1, 3, 3);
        let stride = (1, 1);
        let pad = (2, 2);
        let dil = (2, 2);
        let groups = 1;

        let input_data: Vec<f32> = (0..n * c_in * h * w).map(|i| (i as f32) * 0.1).collect();
        let kernel_data: Vec<f32> = vec![1.0; c_out * c_in * k_h * k_w];
        let bias_data = vec![0.0f32; c_out];

        let expected = conv2d_cpu_nchw(
            &input_data,
            &kernel_data,
            &bias_data,
            n,
            c_in,
            h,
            w,
            c_out,
            k_h,
            k_w,
            stride,
            pad,
            dil,
            groups,
        );

        let input_t = Tensor::from_vec_on(input_data, vec![n, c_in, h, w], device.clone())
            .await
            .unwrap();
        let kernel_t =
            Tensor::from_vec_on(kernel_data, vec![c_out, c_in, k_h, k_w], device.clone())
                .await
                .unwrap();

        let op = Conv2dGpu {
            input: input_t,
            kernel: kernel_t,
            bias: None,
            stride,
            padding: pad,
            dilation: dil,
            groups,
        };
        let result = op.execute().unwrap();
        let got = result.to_vec().unwrap();

        for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
            assert!(
                (g - e).abs() < 1e-3,
                "dilation mismatch at {i}: got {g}, expected {e}"
            );
        }
    }

    #[tokio::test]
    async fn test_conv2d_gpu_no_bias() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let (n, c_in, h, w) = (2, 2, 6, 6);
        let (c_out, k_h, k_w) = (3, 3, 3);

        let input_data: Vec<f32> = (0..n * c_in * h * w).map(|i| (i as f32) * 0.01).collect();
        let kernel_data: Vec<f32> = (0..c_out * c_in * k_h * k_w)
            .map(|i| ((i % 5) as f32 - 2.0) * 0.1)
            .collect();
        let bias_data = vec![0.0f32; c_out];

        let expected = conv2d_cpu_nchw(
            &input_data,
            &kernel_data,
            &bias_data,
            n,
            c_in,
            h,
            w,
            c_out,
            k_h,
            k_w,
            (1, 1),
            (0, 0),
            (1, 1),
            1,
        );

        let input_t = Tensor::from_vec_on(input_data, vec![n, c_in, h, w], device.clone())
            .await
            .unwrap();
        let kernel_t =
            Tensor::from_vec_on(kernel_data, vec![c_out, c_in, k_h, k_w], device.clone())
                .await
                .unwrap();

        let op = Conv2dGpu {
            input: input_t,
            kernel: kernel_t,
            bias: None,
            stride: (1, 1),
            padding: (0, 0),
            dilation: (1, 1),
            groups: 1,
        };
        let result = op.execute().unwrap();
        let got = result.to_vec().unwrap();

        for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
            assert!(
                (g - e).abs() < 1e-3,
                "no-bias mismatch at {i}: got {g}, expected {e}"
            );
        }
    }
}

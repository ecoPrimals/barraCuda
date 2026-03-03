//! Fractional max pool 2D operation - Stochastic pooling with non-integer pooling ratios
//!
//! Improves generalization by introducing randomness
//! Reference: "Fractional Max-Pooling" by Graham (2014)

use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
struct FractionalMaxPool2dParams {
    batch_size: u32,
    channels: u32,
    in_height: u32,
    in_width: u32,
    out_height: u32,
    out_width: u32,
    _padding: [u32; 2],
}

/// Fractional max pool 2D operation
pub struct FractionalMaxPool2d {
    input: Tensor,
    pool_seq_h: Tensor,
    pool_seq_w: Tensor,
}

impl FractionalMaxPool2d {
    /// Create fractional max pool 2D operation
    pub fn new(input: Tensor, pool_seq_h: Tensor, pool_seq_w: Tensor) -> Result<Self> {
        let shape = input.shape();
        if shape.len() != 4 {
            return Err(BarracudaError::invalid_op(
                "fractional_max_pool2d",
                format!("input must be 4D [B, C, H, W], got shape {shape:?}"),
            ));
        }

        let _in_height = shape[2];
        let _in_width = shape[3];

        // pool_seq_h should have out_height + 1 elements
        // pool_seq_w should have out_width + 1 elements
        let pool_seq_h_shape = pool_seq_h.shape();
        let pool_seq_w_shape = pool_seq_w.shape();

        if pool_seq_h_shape.len() != 1 {
            return Err(BarracudaError::invalid_op(
                "fractional_max_pool2d",
                format!("pool_seq_h must be 1D, got shape {pool_seq_h_shape:?}"),
            ));
        }

        if pool_seq_w_shape.len() != 1 {
            return Err(BarracudaError::invalid_op(
                "fractional_max_pool2d",
                format!("pool_seq_w must be 1D, got shape {pool_seq_w_shape:?}"),
            ));
        }

        let _out_height = pool_seq_h_shape[0] - 1;
        let _out_width = pool_seq_w_shape[0] - 1;

        if pool_seq_h_shape[0] < 2 {
            return Err(BarracudaError::invalid_op(
                "fractional_max_pool2d",
                format!(
                    "pool_seq_h must have at least 2 elements, got {}",
                    pool_seq_h_shape[0]
                ),
            ));
        }

        if pool_seq_w_shape[0] < 2 {
            return Err(BarracudaError::invalid_op(
                "fractional_max_pool2d",
                format!(
                    "pool_seq_w must have at least 2 elements, got {}",
                    pool_seq_w_shape[0]
                ),
            ));
        }

        Ok(Self {
            input,
            pool_seq_h,
            pool_seq_w,
        })
    }

    /// WGSL shader source (embedded at compile time)
    fn wgsl_shader() -> &'static str {
        {
            static SHADER: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
                crate::shaders::precision::downcast_f64_to_f32(include_str!(
                    "../shaders/pooling/fractional_max_pool2d_f64.wgsl"
                ))
            });
            std::sync::LazyLock::force(&SHADER).as_str()
        }
    }

    /// Execute fractional max pool 2D on tensor
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let shape = self.input.shape();
        let batch_size = shape[0];
        let channels = shape[1];
        let in_height = shape[2];
        let in_width = shape[3];

        let pool_seq_h_shape = self.pool_seq_h.shape();
        let pool_seq_w_shape = self.pool_seq_w.shape();
        let out_height = pool_seq_h_shape[0] - 1;
        let out_width = pool_seq_w_shape[0] - 1;
        let output_size = batch_size * channels * out_height * out_width;

        // Create output buffer
        let output_buffer = device.create_buffer_f32(output_size)?;

        // Create params
        let params = FractionalMaxPool2dParams {
            batch_size: batch_size as u32,
            channels: channels as u32,
            in_height: in_height as u32,
            in_width: in_width as u32,
            out_height: out_height as u32,
            out_width: out_width as u32,
            _padding: [0; 2],
        };

        let params_buffer = device.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("FractionalMaxPool2d Params"),
            size: std::mem::size_of::<FractionalMaxPool2dParams>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        device
            .queue
            .write_buffer(&params_buffer, 0, bytemuck::bytes_of(&params));

        // Convert pool sequences to u32 buffers
        let pool_seq_h_data: Vec<u32> = self
            .pool_seq_h
            .to_vec()?
            .iter()
            .map(|&x| x as u32)
            .collect();
        let pool_seq_w_data: Vec<u32> = self
            .pool_seq_w
            .to_vec()?
            .iter()
            .map(|&x| x as u32)
            .collect();

        let pool_seq_h_buffer =
            device
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("FractionalMaxPool2d PoolSeqH"),
                    contents: bytemuck::cast_slice(&pool_seq_h_data),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                });

        let pool_seq_w_buffer =
            device
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("FractionalMaxPool2d PoolSeqW"),
                    contents: bytemuck::cast_slice(&pool_seq_w_data),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                });

        // Create bind group layout
        let bind_group_layout =
            device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("FractionalMaxPool2d Bind Group Layout"),
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

        // Create bind group
        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("FractionalMaxPool2d Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.input.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: pool_seq_h_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: pool_seq_w_buffer.as_entire_binding(),
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

        // Compile shader
        let shader = device.compile_shader(Self::wgsl_shader(), Some("FractionalMaxPool2d"));

        // Create pipeline
        let pipeline_layout =
            device
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("FractionalMaxPool2d Pipeline Layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });

        let pipeline = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("FractionalMaxPool2d Pipeline"),
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
                label: Some("FractionalMaxPool2d Encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("FractionalMaxPool2d Pass"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);

            // Dispatch workgroups (8x8x1 workgroup size)
            let workgroups_x = (out_width as u32).div_ceil(8);
            let workgroups_y = (out_height as u32).div_ceil(8);
            let workgroups_z = (batch_size * channels) as u32;
            pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z);
        }

        device.submit_and_poll(Some(encoder.finish()));

        // Create output tensor
        Ok(Tensor::from_buffer(
            output_buffer,
            vec![batch_size, channels, out_height, out_width],
            device.clone(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device_if_gpu_available;

    #[tokio::test]
    async fn test_fractional_max_pool2d_basic() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let input = Tensor::from_vec_on(vec![1.0; 2 * 3 * 8 * 8], vec![2, 3, 8, 8], device.clone())
            .await
            .unwrap();

        // Pool sequence: [0, 2, 4, 6, 8] for 4x4 output
        let pool_seq_h =
            Tensor::from_vec_on(vec![0.0, 2.0, 4.0, 6.0, 8.0], vec![5], device.clone())
                .await
                .unwrap();

        let pool_seq_w = Tensor::from_vec_on(vec![0.0, 2.0, 4.0, 6.0, 8.0], vec![5], device)
            .await
            .unwrap();

        let output = FractionalMaxPool2d::new(input, pool_seq_h, pool_seq_w)
            .unwrap()
            .execute()
            .unwrap();
        let result = output.to_vec().unwrap();

        // Output should be [2, 3, 4, 4]
        assert_eq!(result.len(), 2 * 3 * 4 * 4);
    }
}

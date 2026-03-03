// SPDX-License-Identifier: AGPL-3.0-or-later
//! ROI Align - Region of Interest Align (Pure WGSL)
//!
//! Extracts fixed-size feature maps from regions using bilinear interpolation
//! Avoids quantization artifacts of ROI Pooling
//!
//! **Deep Debt Principles**:
//! - Pure WGSL implementation (no CPU code)
//! - Safe Rust wrapper (no unsafe code)
//! - Hardware-agnostic via WebGPU
//! - Complete implementation (production-ready)

use crate::device::{DeviceCapabilities, WorkloadType};
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;
use wgpu::util::DeviceExt;

/// Region of Interest Align
pub struct RoiAlign {
    features: Tensor,
    rois: Tensor,
    pooled_height: usize,
    pooled_width: usize,
    spatial_scale: f32,
    sampling_ratio: usize,
}

impl RoiAlign {
    pub fn new(
        features: Tensor,
        rois: Tensor,
        pooled_height: usize,
        pooled_width: usize,
        spatial_scale: f32,
        sampling_ratio: usize,
    ) -> Result<Self> {
        let features_shape = features.shape();
        if features_shape.len() != 4 {
            return Err(BarracudaError::invalid_op(
                "roi_align",
                "features must be 4D [1, channels, height, width]",
            ));
        }

        let rois_shape = rois.shape();
        if rois_shape.len() != 2 || rois_shape[1] != 4 {
            return Err(BarracudaError::invalid_op(
                "roi_align",
                "rois must be 2D [num_rois, 4]",
            ));
        }

        if spatial_scale <= 0.0 {
            return Err(BarracudaError::invalid_op(
                "roi_align",
                "spatial_scale must be positive",
            ));
        }

        Ok(Self {
            features,
            rois,
            pooled_height,
            pooled_width,
            spatial_scale,
            sampling_ratio,
        })
    }

    fn wgsl_shader() -> &'static str {
        {
            static SHADER: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
                crate::shaders::precision::downcast_f64_to_f32(include_str!(
                    "../shaders/pooling/roi_align_f64.wgsl"
                ))
            });
            std::sync::LazyLock::force(&SHADER).as_str()
        }
    }

    pub fn execute(self) -> Result<Tensor> {
        let device = self.features.device();
        let features_shape = self.features.shape();
        let channels = features_shape[1];
        let height = features_shape[2];
        let width = features_shape[3];

        let rois_shape = self.rois.shape();
        let num_rois = rois_shape[0];

        let output_size = num_rois * channels * self.pooled_height * self.pooled_width;
        let output_buffer = device.create_buffer_f32(output_size)?;

        // Create uniform buffer
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            num_rois: u32,
            channels: u32,
            height: u32,
            width: u32,
            pooled_height: u32,
            pooled_width: u32,
            spatial_scale: f32,
            sampling_ratio: u32,
        }

        let params = Params {
            num_rois: num_rois as u32,
            channels: channels as u32,
            height: height as u32,
            width: width as u32,
            pooled_height: self.pooled_height as u32,
            pooled_width: self.pooled_width as u32,
            spatial_scale: self.spatial_scale,
            sampling_ratio: self.sampling_ratio as u32,
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("RoiAlign Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        // Compile shader
        let shader_module = device.compile_shader(Self::wgsl_shader(), Some("RoiAlign Shader"));

        // Create bind group layout
        let bind_group_layout =
            device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("RoiAlign Bind Group Layout"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
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
                    ],
                });

        // Create bind group
        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("RoiAlign Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.features.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.rois.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: output_buffer.as_entire_binding(),
                },
            ],
        });

        // Create pipeline layout
        let pipeline_layout =
            device
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("RoiAlign Pipeline Layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });

        // Create pipeline
        let pipeline = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("RoiAlign Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point: "main",
                cache: None,
                compilation_options: Default::default(),
            });

        // Encode and execute
        let mut encoder = device
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("RoiAlign Encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("RoiAlign Pass"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);

            // Deep Debt Evolution: Capability-based dispatch
            let caps = DeviceCapabilities::from_device(device);
            let optimal_wg_size = caps.optimal_workgroup_size(WorkloadType::Convolution);
            let spatial_size = self.pooled_height * self.pooled_width;
            let workgroups_x = (spatial_size as u32).div_ceil(optimal_wg_size);
            let workgroups_y = (channels as u32).div_ceil(optimal_wg_size);
            let workgroups_z = num_rois as u32;
            pass.dispatch_workgroups(workgroups_x, workgroups_y, workgroups_z);
        }

        device.submit_and_poll(Some(encoder.finish()));

        Ok(Tensor::from_buffer(
            output_buffer,
            vec![num_rois, channels, self.pooled_height, self.pooled_width],
            device.clone(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device_if_gpu_available;

    #[tokio::test]
    async fn test_roi_align_basic() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let channels = 64;
        let height = 32;
        let width = 32;
        let num_rois = 2;
        let pooled_height = 7;
        let pooled_width = 7;

        let features = Tensor::from_vec_on(
            vec![1.0; channels * height * width],
            vec![1, channels, height, width],
            device.clone(),
        )
        .await
        .unwrap();

        let rois = Tensor::from_vec_on(
            vec![0.0, 0.0, 10.0, 10.0, 5.0, 5.0, 15.0, 15.0],
            vec![num_rois, 4],
            device.clone(),
        )
        .await
        .unwrap();

        let roi_align = RoiAlign::new(features, rois, pooled_height, pooled_width, 1.0, 2).unwrap();
        let output = roi_align.execute().unwrap();

        assert_eq!(
            output.shape(),
            &[num_rois, channels, pooled_height, pooled_width]
        );
    }
}

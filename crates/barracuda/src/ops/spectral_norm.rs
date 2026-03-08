// SPDX-License-Identifier: AGPL-3.0-or-later
//! Spectral Normalization
//!
//! **Pure WGSL**: Single implementation via WebGPU shader
//! Normalizes weights by their spectral norm (largest singular value)

use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct SpectralNormParams {
    rows: u32,
    cols: u32,
    num_iterations: u32,
    _padding: u32,
}

/// Spectral normalization: normalizes weights by their spectral norm (largest singular value).
pub struct SpectralNorm {
    weight: Tensor,
    u: Tensor,
    v: Tensor,
    num_iterations: u32,
}

impl SpectralNorm {
    /// Create `SpectralNorm` operation
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn new(weight: Tensor, u: Tensor, v: Tensor, num_iterations: u32) -> Result<Self> {
        if num_iterations == 0 {
            return Err(BarracudaError::invalid_op(
                "SpectralNorm",
                "num_iterations must be > 0",
            ));
        }

        Ok(Self {
            weight,
            u,
            v,
            num_iterations,
        })
    }

    /// WGSL shader source (embedded at compile time)
    fn wgsl_shader() -> &'static str {
        {
            static SHADER: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
                include_str!("../shaders/norm/spectral_norm_f64.wgsl").to_string()
            });
            std::sync::LazyLock::force(&SHADER).as_str()
        }
    }

    /// Execute `SpectralNorm` on tensor (modifies weight in-place)
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.weight.device();
        let weight_shape = self.weight.shape();

        if weight_shape.len() != 2 {
            return Err(BarracudaError::invalid_op(
                "SpectralNorm",
                format!("weight must be 2D [rows, cols], got shape {weight_shape:?}"),
            ));
        }

        let rows = weight_shape[0];
        let cols = weight_shape[1];

        // Validate u and v shapes
        if self.u.shape() != [rows] {
            return Err(BarracudaError::invalid_op(
                "SpectralNorm",
                format!("u must be 1D [rows], got shape {:?}", self.u.shape()),
            ));
        }

        if self.v.shape() != [cols] {
            return Err(BarracudaError::invalid_op(
                "SpectralNorm",
                format!("v must be 1D [cols], got shape {:?}", self.v.shape()),
            ));
        }

        let params = SpectralNormParams {
            rows: rows as u32,
            cols: cols as u32,
            num_iterations: self.num_iterations,
            _padding: 0,
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("SpectralNorm Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        // Create bind group layout
        let bind_group_layout =
            device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("SpectralNorm Bind Group Layout"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
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

        // Create bind group
        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SpectralNorm Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.weight.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.u.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.v.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        // Compile shader
        let shader = device.compile_shader(Self::wgsl_shader(), Some("SpectralNorm"));

        // Create pipeline
        let pipeline_layout =
            device
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("SpectralNorm Pipeline Layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    immediate_size: 0,
                });

        let pipeline = device
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("SpectralNorm Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                cache: None,
                compilation_options: Default::default(),
            });

        // Encode and execute
        let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("SpectralNorm Encoder"),
        });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("SpectralNorm Pass"),
                timestamp_writes: None,
            });

            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, Some(&bind_group), &[]);

            // Deep Debt Evolution: Capability-based dispatch
            use crate::device::{DeviceCapabilities, WorkloadType};
            let caps = DeviceCapabilities::from_device(device);
            let optimal_wg_size = caps.optimal_workgroup_size(WorkloadType::ElementWise);
            let max_dim = rows.max(cols);
            let workgroups = (max_dim as u32).div_ceil(optimal_wg_size);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        device.submit_and_poll(Some(encoder.finish()));

        // Return normalized weight (modified in-place)
        Ok(self.weight)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device_if_gpu_available;

    #[tokio::test]
    async fn test_spectral_norm_basic() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let rows = 10;
        let cols = 8;

        let weight = Tensor::from_vec_on(vec![0.1; rows * cols], vec![rows, cols], device.clone())
            .await
            .unwrap();

        let u = Tensor::from_vec_on(vec![1.0; rows], vec![rows], device.clone())
            .await
            .unwrap();

        let v = Tensor::from_vec_on(vec![1.0; cols], vec![cols], device.clone())
            .await
            .unwrap();

        let result = SpectralNorm::new(weight, u, v, 1)
            .unwrap()
            .execute()
            .unwrap();

        assert_eq!(result.shape(), &[rows, cols]);
    }
}

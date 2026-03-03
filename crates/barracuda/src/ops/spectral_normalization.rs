//! Spectral Normalization - Pure WGSL
//!
//! Deep Debt Principles:
//! - Self-knowledge: Operation knows its computation
//! - Zero hardcoding: Hardware-agnostic implementation
//! - Modern idiomatic Rust: Safe, zero unsafe code
//! - Complete implementation: Production-ready, no mocks
//! - Hardware-agnostic: Pure WGSL for universal compute
//!
//! Stabilizes GAN training by constraining Lipschitz constant.
//! Used in SNGAN, BigGAN.
//!
//! Normalizes weights by their spectral norm (largest singular value).

use crate::device::{DeviceCapabilities, WorkloadType};
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;
use wgpu::util::DeviceExt;

/// Spectral Normalization operation
pub struct SpectralNormalization {
    weight: Tensor,
    u: Tensor,
    v: Tensor,
    num_iterations: u32,
}

impl SpectralNormalization {
    /// Create a new spectral normalization operation
    ///
    /// # Arguments
    /// * `weight` - Weight matrix [rows, cols]
    /// * `u` - Left singular vector [rows]
    /// * `v` - Right singular vector [cols]
    /// * `num_iterations` - Number of power iteration steps (typically 1)
    pub fn new(weight: Tensor, u: Tensor, v: Tensor, num_iterations: u32) -> Result<Self> {
        if num_iterations == 0 {
            return Err(BarracudaError::invalid_op(
                "SpectralNormalization",
                "num_iterations must be > 0",
            ));
        }

        // Validate weight shape
        let weight_shape = weight.shape();
        if weight_shape.len() != 2 {
            return Err(BarracudaError::invalid_op(
                "SpectralNormalization",
                format!("weight must be 2D [rows, cols], got shape {weight_shape:?}"),
            ));
        }

        let rows = weight_shape[0];
        let cols = weight_shape[1];

        // Validate u and v shapes
        if u.shape() != [rows] {
            return Err(BarracudaError::invalid_op(
                "SpectralNormalization",
                format!("u must be 1D [rows], got shape {:?}", u.shape()),
            ));
        }

        if v.shape() != [cols] {
            return Err(BarracudaError::invalid_op(
                "SpectralNormalization",
                format!("v must be 1D [cols], got shape {:?}", v.shape()),
            ));
        }

        Ok(Self {
            weight,
            u,
            v,
            num_iterations,
        })
    }

    /// Get the WGSL shader source
    fn wgsl_shader() -> &'static str {
        {
            static SHADER: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
                crate::shaders::precision::downcast_f64_to_f32_with_transcendentals(include_str!(
                    "../shaders/norm/spectral_norm_f64.wgsl"
                ))
            });
            std::sync::LazyLock::force(&SHADER).as_str()
        }
    }

    /// Execute the spectral normalization operation (modifies weight in-place)
    pub fn execute(self) -> Result<Tensor> {
        let device = self.weight.device();
        let weight_shape = self.weight.shape();
        let rows = weight_shape[0];
        let cols = weight_shape[1];

        // Create uniform buffer for parameters
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            rows: u32,
            cols: u32,
            num_iterations: u32,
            _padding: u32,
        }

        let params = Params {
            rows: rows as u32,
            cols: cols as u32,
            num_iterations: self.num_iterations,
            _padding: 0,
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("SpectralNormalization Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        // Compile shader
        let shader_module =
            device.compile_shader(Self::wgsl_shader(), Some("SpectralNormalization Shader"));

        // Create bind group layout
        let bind_group_layout =
            device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("SpectralNormalization Bind Group Layout"),
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
            label: Some("SpectralNormalization Bind Group"),
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

        // Create compute pipeline
        let pipeline_layout =
            device
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("SpectralNormalization Pipeline Layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });

        let compute_pipeline =
            device
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("SpectralNormalization Pipeline"),
                    layout: Some(&pipeline_layout),
                    module: &shader_module,
                    entry_point: "main",
                    cache: None,
                    compilation_options: Default::default(),
                });

        // Execute compute shader
        let mut encoder = device
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("SpectralNormalization Encoder"),
            });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("SpectralNormalization Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&compute_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            // Deep Debt Evolution: Capability-based dispatch
            let caps = DeviceCapabilities::from_device(device);
            let optimal_wg_size = caps.optimal_workgroup_size(WorkloadType::Reduction);
            let max_dim = rows.max(cols);
            let workgroups = (max_dim as u32).div_ceil(optimal_wg_size);
            compute_pass.dispatch_workgroups(workgroups, 1, 1);
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
    async fn test_spectral_normalization() {
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

        let result = SpectralNormalization::new(weight, u, v, 1)
            .unwrap()
            .execute()
            .unwrap();

        assert_eq!(result.shape(), &[rows, cols]);
    }
}

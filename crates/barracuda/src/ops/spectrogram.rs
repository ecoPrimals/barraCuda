// SPDX-License-Identifier: AGPL-3.0-or-later
//! Spectrogram - Power spectrogram computation
//!
//! Computes magnitude squared of STFT.
//! Visualizes frequency content over time.
//!
//! Deep Debt Principles:
//! - Self-knowledge: Operation knows its computation
//! - Zero hardcoding: Hardware-agnostic implementation
//! - Modern idiomatic Rust: Safe, zero unsafe code
//! - Complete implementation: Production-ready, no mocks
//! - Hardware-agnostic: Pure WGSL for universal compute

use crate::device::{DeviceCapabilities, WorkloadType};
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

/// f64 is the canonical source — math is universal, precision is silicon.
static SHADER_F64: &str = include_str!("../shaders/audio/spectrogram_f64.wgsl");
static SHADER_F32: std::sync::LazyLock<String> =
    std::sync::LazyLock::new(|| SHADER_F64.to_string());

/// Spectrogram operation
pub struct Spectrogram {
    stft_data: Tensor, // Complex STFT [real, imag, real, imag, ...]
    power: f32,        // 1.0 for magnitude, 2.0 for power
}

impl Spectrogram {
    /// Create a new spectrogram operation
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn new(stft_data: Tensor, power: f32) -> Result<Self> {
        let size = stft_data.shape().iter().product::<usize>();
        if size % 2 != 0 {
            return Err(BarracudaError::InvalidInput {
                message: "STFT data must contain even number of elements (complex pairs)"
                    .to_string(),
            });
        }
        Ok(Self { stft_data, power })
    }

    /// Get the WGSL shader source
    fn wgsl_shader() -> &'static str {
        &SHADER_F32
    }

    /// Execute the spectrogram operation
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.stft_data.device();
        let size: usize = self.stft_data.shape().iter().product();
        let num_complex_pairs = size / 2;

        // Access input buffer directly (zero-copy)
        let input_buffer = self.stft_data.buffer();

        // Create output buffer
        let output_buffer = device.create_buffer_f32(num_complex_pairs)?;

        // Create uniform buffer for parameters
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            size: u32,
            power: f32,
        }

        let params = Params {
            size: num_complex_pairs as u32,
            power: self.power,
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Spectrogram Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        // Compile shader
        let shader_module = device.compile_shader(Self::wgsl_shader(), Some("Spectrogram Shader"));

        // Create bind group layout
        let bind_group_layout =
            device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Spectrogram Bind Group Layout"),
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

        // Create bind group
        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Spectrogram Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buffer.as_entire_binding(),
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

        // Create compute pipeline
        let pipeline_layout =
            device
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Spectrogram Pipeline Layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    immediate_size: 0,
                });

        let compute_pipeline =
            device
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("Spectrogram Pipeline"),
                    layout: Some(&pipeline_layout),
                    module: &shader_module,
                    entry_point: Some("main"),
                    cache: None,
                    compilation_options: Default::default(),
                });

        // Execute compute shader
        let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("Spectrogram Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Spectrogram Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&compute_pipeline);
            compute_pass.set_bind_group(0, Some(&bind_group), &[]);
            // Deep Debt Evolution: Capability-based dispatch
            let caps = DeviceCapabilities::from_device(device);
            let optimal_wg_size = caps.optimal_workgroup_size(WorkloadType::MatMul);
            let workgroups = (num_complex_pairs as u32).div_ceil(optimal_wg_size);
            compute_pass.dispatch_workgroups(workgroups, 1, 1);
        }

        device.submit_commands(Some(encoder.finish()));

        // Output shape: [num_complex_pairs] (flattened from original shape)
        let mut output_shape = self.stft_data.shape().to_vec();
        if let Some(last) = output_shape.last_mut() {
            *last /= 2;
        }

        // Return tensor without reading back (zero-copy)
        Ok(Tensor::from_buffer(
            output_buffer,
            output_shape,
            device.clone(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::tensor::Tensor;
    #[expect(unused_imports, reason = "conditional imports")]
    use std::sync::Arc;

    #[tokio::test]
    async fn test_spectrogram_basic() {
        // Create complex STFT data: [real, imag, real, imag, ...]
        let device = crate::device::test_pool::get_test_device().await;
        let stft_data = vec![3.0, 4.0, 3.0, 4.0, 3.0, 4.0]; // 3 complex pairs, magnitude = 5.0
        let stft_tensor = Tensor::from_vec_on(stft_data, vec![3, 2], device.clone())
            .await
            .unwrap();

        let power_spec = Spectrogram::new(stft_tensor, 2.0)
            .unwrap()
            .execute()
            .unwrap();
        assert_eq!(power_spec.shape(), &[3, 1]);
    }

    #[tokio::test]
    async fn test_spectrogram_edge_cases() {
        let device = crate::device::test_pool::get_test_device().await;
        // Single complex pair
        let stft_data = vec![1.0, 0.0];
        let stft_tensor = Tensor::from_vec_on(stft_data, vec![1, 2], device.clone())
            .await
            .unwrap();
        let mag_spec = Spectrogram::new(stft_tensor, 1.0)
            .unwrap()
            .execute()
            .unwrap();
        assert_eq!(mag_spec.shape(), &[1, 1]);
    }
}

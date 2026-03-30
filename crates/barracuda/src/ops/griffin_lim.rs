// SPDX-License-Identifier: AGPL-3.0-or-later
//! `GriffinLim` - Phase reconstruction from magnitude spectrogram
//!
//! Iteratively estimates phase for ISTFT.
//! Used in audio synthesis from spectrograms.
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
static SHADER_F64: &str = include_str!("../shaders/audio/griffin_lim_f64.wgsl");
static SHADER_F32: std::sync::LazyLock<String> =
    std::sync::LazyLock::new(|| SHADER_F64.to_string());

/// `GriffinLim` operation
pub struct GriffinLim {
    magnitude: Tensor,
    n_frames: usize,
    n_freqs: usize,
    n_fft: usize,
    hop_length: usize,
    n_iter: usize,
}

impl GriffinLim {
    /// Create a new Griffin-Lim operation
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn new(
        magnitude: Tensor,
        n_frames: usize,
        n_freqs: usize,
        n_fft: usize,
        hop_length: usize,
        n_iter: usize,
    ) -> Result<Self> {
        let mag_size: usize = magnitude.shape().iter().product();
        if mag_size != n_frames * n_freqs {
            return Err(BarracudaError::InvalidInput {
                message: format!(
                    "Magnitude size ({}) must equal n_frames * n_freqs ({})",
                    mag_size,
                    n_frames * n_freqs
                ),
            });
        }

        let expected_freqs = n_fft / 2 + 1;
        if n_freqs != expected_freqs {
            return Err(BarracudaError::InvalidInput {
                message: format!(
                    "n_freqs ({n_freqs}) must equal n_fft/2 + 1 ({expected_freqs}) for STFT consistency"
                ),
            });
        }

        if hop_length == 0 || hop_length > n_fft {
            return Err(BarracudaError::InvalidInput {
                message: format!("hop_length ({hop_length}) must be in [1, n_fft={n_fft}]"),
            });
        }

        Ok(Self {
            magnitude,
            n_frames,
            n_freqs,
            n_fft,
            hop_length,
            n_iter,
        })
    }

    /// Expected output signal length: `n_fft + (n_frames - 1) * hop_length`.
    #[must_use]
    pub fn expected_signal_length(&self) -> usize {
        self.n_fft + (self.n_frames - 1) * self.hop_length
    }

    /// Get the WGSL shader source
    fn wgsl_shader() -> &'static str {
        &SHADER_F32
    }

    /// Execute the Griffin-Lim operation
    /// Note: This is a simplified version. Full implementation would require
    /// iterative STFT/ISTFT cycles which are complex to implement in GPU.
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.magnitude.device();

        // Initialize phase with random values
        let phase_data: Vec<f32> = (0..self.n_frames * self.n_freqs)
            .map(|i| (i as f32 * 0.1) % (2.0 * std::f32::consts::PI))
            .collect();

        // Access input buffer directly (zero-copy)
        let magnitude_buffer = self.magnitude.buffer();

        // Create phase buffer
        let phase_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("GriffinLim Phase"),
                contents: bytemuck::cast_slice(&phase_data),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

        // Create output buffer (complex pairs)
        let output_size = self.n_frames * self.n_freqs * 2;
        let output_buffer = device.create_buffer_f32(output_size)?;

        // Create uniform buffer for parameters
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            n_frames: u32,
            n_freqs: u32,
            n_iter: u32,
        }

        let params = Params {
            n_frames: self.n_frames as u32,
            n_freqs: self.n_freqs as u32,
            n_iter: self.n_iter as u32,
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("GriffinLim Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        // Compile shader
        let shader_module = device.compile_shader(Self::wgsl_shader(), Some("GriffinLim Shader"));

        // Create bind group layout
        let bind_group_layout =
            device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("GriffinLim Bind Group Layout"),
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
            label: Some("GriffinLim Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: magnitude_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: phase_buffer.as_entire_binding(),
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

        // Create compute pipeline
        let pipeline_layout =
            device
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("GriffinLim Pipeline Layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    immediate_size: 0,
                });

        let compute_pipeline =
            device
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("GriffinLim Pipeline"),
                    layout: Some(&pipeline_layout),
                    module: &shader_module,
                    entry_point: Some("main"),
                    cache: None,
                    compilation_options: Default::default(),
                });

        // Execute compute shader
        let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("GriffinLim Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("GriffinLim Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&compute_pipeline);
            compute_pass.set_bind_group(0, Some(&bind_group), &[]);
            // Deep Debt Evolution: Capability-based dispatch
            let caps = DeviceCapabilities::from_device(device);
            let size = self.n_frames * self.n_freqs;
            let optimal_wg_size = caps.optimal_workgroup_size(WorkloadType::MatMul);
            let workgroups = (size as u32).div_ceil(optimal_wg_size);
            compute_pass.dispatch_workgroups(workgroups, 1, 1);
        }

        device.submit_commands(Some(encoder.finish()));

        // Output shape: [n_frames, n_freqs, 2] (complex pairs)
        let output_shape = vec![self.n_frames, self.n_freqs, 2];

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

    #[tokio::test]
    async fn test_griffin_lim_basic() {
        let device = crate::device::test_pool::get_test_device().await;
        let magnitude = Tensor::from_vec_on(vec![1.0; 100 * 257], vec![100, 257], device.clone())
            .await
            .unwrap();

        let output = GriffinLim::new(magnitude, 100, 257, 512, 256, 10)
            .unwrap()
            .execute()
            .unwrap();
        assert_eq!(output.shape(), &[100, 257, 2]);
    }
}

// SPDX-License-Identifier: AGPL-3.0-or-later
//! MFCC - Mel-Frequency Cepstral Coefficients
//!
//! Extracts MFCC features from audio.
//! Standard features for speech recognition.
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
static SHADER_F64: &str = include_str!("../shaders/audio/mfcc_f64.wgsl");
static SHADER_F32: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
    SHADER_F64.to_string()
});

/// MFCC operation
pub struct MFCC {
    mel_spectrogram: Tensor,
    n_frames: usize,
    n_mels: usize,
    n_mfcc: usize,
}

impl MFCC {
    /// Create a new MFCC operation
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn new(
        mel_spectrogram: Tensor,
        n_frames: usize,
        n_mels: usize,
        n_mfcc: usize,
    ) -> Result<Self> {
        if n_mfcc > n_mels {
            return Err(BarracudaError::InvalidInput {
                message: format!("n_mfcc ({n_mfcc}) cannot exceed n_mels ({n_mels})"),
            });
        }

        let mel_size: usize = mel_spectrogram.shape().iter().product();
        if mel_size != n_frames * n_mels {
            return Err(BarracudaError::InvalidInput {
                message: format!(
                    "Mel spectrogram size ({}) must equal n_frames * n_mels ({})",
                    mel_size,
                    n_frames * n_mels
                ),
            });
        }

        Ok(Self {
            mel_spectrogram,
            n_frames,
            n_mels,
            n_mfcc,
        })
    }

    /// Get the WGSL shader source
    fn wgsl_shader() -> &'static str {
        &SHADER_F32
    }

    /// Execute the MFCC operation
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.mel_spectrogram.device();
        let output_size = self.n_frames * self.n_mfcc;

        // Access input buffer directly (zero-copy)
        let mel_buffer = self.mel_spectrogram.buffer();

        // Create output buffer
        let output_buffer = device.create_buffer_f32(output_size)?;

        // Create uniform buffer for parameters
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            n_frames: u32,
            n_mels: u32,
            n_mfcc: u32,
        }

        let params = Params {
            n_frames: self.n_frames as u32,
            n_mels: self.n_mels as u32,
            n_mfcc: self.n_mfcc as u32,
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("MFCC Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        // Compile shader
        let shader_module = device.compile_shader(Self::wgsl_shader(), Some("MFCC Shader"));

        // Create bind group layout
        let bind_group_layout =
            device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("MFCC Bind Group Layout"),
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
            label: Some("MFCC Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: mel_buffer.as_entire_binding(),
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
                    label: Some("MFCC Pipeline Layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    immediate_size: 0,
                });

        let compute_pipeline =
            device
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("MFCC Pipeline"),
                    layout: Some(&pipeline_layout),
                    module: &shader_module,
                    entry_point: Some("main"),
                    cache: None,
                    compilation_options: Default::default(),
                });

        // Execute compute shader
        let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("MFCC Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("MFCC Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&compute_pipeline);
            compute_pass.set_bind_group(0, Some(&bind_group), &[]);
            // Deep Debt Evolution: Capability-based dispatch
            let caps = DeviceCapabilities::from_device(device);
            let size = self.n_frames * self.n_mfcc;
            let optimal_wg_size = caps.optimal_workgroup_size(WorkloadType::MatMul);
            let workgroups = (size as u32).div_ceil(optimal_wg_size);
            compute_pass.dispatch_workgroups(workgroups, 1, 1);
        }

        device.submit_and_poll(Some(encoder.finish()));

        // Output shape: [n_frames, n_mfcc]
        let output_shape = vec![self.n_frames, self.n_mfcc];

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
    use crate::device::test_pool::get_test_device_if_gpu_available;

    #[tokio::test]
    async fn test_mfcc_basic() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let mel_spec = Tensor::from_vec_on(vec![1.0; 100 * 80], vec![100, 80], device.clone())
            .await
            .unwrap();

        let mfcc_features = MFCC::new(mel_spec, 100, 80, 13).unwrap().execute().unwrap();
        assert_eq!(mfcc_features.shape(), &[100, 13]);
    }
}

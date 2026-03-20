// SPDX-License-Identifier: AGPL-3.0-or-later
//! STFT - Short-Time Fourier Transform
//!
//! Converts time-domain signal to time-frequency representation.
//! Foundation for spectrograms and audio analysis.
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
static SHADER_F64: &str = include_str!("../shaders/audio/stft_f64.wgsl");
static SHADER_F32: std::sync::LazyLock<String> =
    std::sync::LazyLock::new(|| SHADER_F64.to_string());

/// STFT operation
pub struct STFT {
    signal: Tensor,
    window: Tensor,
    n_fft: usize,
    hop_length: usize,
}

impl STFT {
    /// Create a new STFT operation
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn new(signal: Tensor, window: Tensor, n_fft: usize, hop_length: usize) -> Result<Self> {
        // Validate window length
        let window_size: usize = window.shape().iter().product();
        if window_size != n_fft {
            return Err(BarracudaError::InvalidInput {
                message: format!("Window length ({window_size}) must match n_fft ({n_fft})"),
            });
        }

        // Validate signal length
        let signal_size: usize = signal.shape().iter().product();
        if signal_size < n_fft {
            return Err(BarracudaError::InvalidInput {
                message: format!("Signal length ({signal_size}) must be at least n_fft ({n_fft})"),
            });
        }

        // Ensure same device
        if !std::ptr::eq(signal.device().as_ref(), window.device().as_ref()) {
            return Err(BarracudaError::InvalidInput {
                message: "Signal and window must be on the same device".to_string(),
            });
        }

        Ok(Self {
            signal,
            window,
            n_fft,
            hop_length,
        })
    }

    /// Get the WGSL shader source
    fn wgsl_shader() -> &'static str {
        &SHADER_F32
    }

    /// Execute the STFT operation
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.signal.device();
        let signal_length: usize = self.signal.shape().iter().product();
        let num_frames = (signal_length - self.n_fft) / self.hop_length + 1;
        let bins_per_frame = self.n_fft / 2 + 1;
        let output_size = num_frames * bins_per_frame * 2; // Complex pairs

        // Access input buffers directly (zero-copy)
        let signal_buffer = self.signal.buffer();
        let window_buffer = self.window.buffer();

        // Create output buffer
        let output_buffer = device.create_buffer_f32(output_size)?;

        // Create uniform buffer for parameters
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            signal_length: u32,
            n_fft: u32,
            hop_length: u32,
            num_frames: u32,
            bins_per_frame: u32,
        }

        let params = Params {
            signal_length: signal_length as u32,
            n_fft: self.n_fft as u32,
            hop_length: self.hop_length as u32,
            num_frames: num_frames as u32,
            bins_per_frame: bins_per_frame as u32,
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("STFT Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        // Compile shader
        let shader_module = device.compile_shader(Self::wgsl_shader(), Some("STFT Shader"));

        // Create bind group layout
        let bind_group_layout =
            device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("STFT Bind Group Layout"),
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

        // Create bind group
        let bind_group = device.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("STFT Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: signal_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: window_buffer.as_entire_binding(),
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
                    label: Some("STFT Pipeline Layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    immediate_size: 0,
                });

        let compute_pipeline =
            device
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("STFT Pipeline"),
                    layout: Some(&pipeline_layout),
                    module: &shader_module,
                    entry_point: Some("main"),
                    cache: None,
                    compilation_options: Default::default(),
                });

        // Execute compute shader
        let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("STFT Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("STFT Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&compute_pipeline);
            compute_pass.set_bind_group(0, Some(&bind_group), &[]);
            // Deep Debt Evolution: Capability-based dispatch
            let caps = DeviceCapabilities::from_device(device);
            let optimal_wg_size = caps.optimal_workgroup_size(WorkloadType::MatMul);
            let workgroups_x = (num_frames as u32).div_ceil(optimal_wg_size);
            let workgroups_y = (bins_per_frame as u32).div_ceil(optimal_wg_size);
            compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        device.submit_commands(Some(encoder.finish()));

        // Output shape: [num_frames, bins_per_frame, 2] (real, imag pairs)
        let output_shape = vec![num_frames, bins_per_frame, 2];

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
    use crate::ops::window_function::{WindowFunction, WindowType};
    #[expect(unused_imports, reason = "conditional imports")]
    use std::sync::Arc;

    #[tokio::test]
    async fn test_stft_basic() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let signal = Tensor::from_vec_on(vec![0.0; 1024], vec![1024], device.clone())
            .await
            .unwrap();
        let window = WindowFunction::new(512, WindowType::Rectangular, device.clone())
            .unwrap()
            .execute()
            .unwrap();

        let output = STFT::new(signal, window, 512, 256)
            .unwrap()
            .execute()
            .unwrap();
        assert!(output.shape().len() == 3);
        assert_eq!(output.shape()[2], 2); // Complex pairs
    }
}

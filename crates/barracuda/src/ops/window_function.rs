// SPDX-License-Identifier: AGPL-3.0-or-later
//! `WindowFunction` - Various windowing functions for signal processing
//!
//! Implements Hann, Hamming, Blackman, Bartlett, and Rectangular windows.
//! Reduces spectral leakage in FFT.
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
use std::sync::Arc;

/// f64 is the canonical source — math is universal, precision is silicon.
static SHADER_F64: &str = include_str!("../shaders/audio/window_function_f64.wgsl");
static SHADER_F32: std::sync::LazyLock<String> =
    std::sync::LazyLock::new(|| SHADER_F64.to_string());

/// Window type for signal processing (reduces spectral leakage).
#[derive(Clone, Copy)]
pub enum WindowType {
    /// Hann (raised cosine) window.
    Hann,
    /// Hamming window.
    Hamming,
    /// Blackman window.
    Blackman,
    /// Bartlett (triangular) window.
    Bartlett,
    /// Rectangular (boxcar) window.
    Rectangular,
}

impl WindowType {
    fn to_u32(self) -> u32 {
        match self {
            WindowType::Hann => 0,
            WindowType::Hamming => 1,
            WindowType::Blackman => 2,
            WindowType::Bartlett => 3,
            WindowType::Rectangular => 4,
        }
    }
}

/// Window function operation (WGSL).
pub struct WindowFunction {
    length: usize,
    window_type: WindowType,
    device: Arc<crate::device::WgpuDevice>,
}

impl WindowFunction {
    /// Create a new window function operation.
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn new(
        length: usize,
        window_type: WindowType,
        device: Arc<crate::device::WgpuDevice>,
    ) -> Result<Self> {
        if length == 0 {
            return Err(BarracudaError::InvalidInput {
                message: "Window length must be greater than 0".to_string(),
            });
        }
        Ok(Self {
            length,
            window_type,
            device,
        })
    }

    /// Get the WGSL shader source
    fn wgsl_shader() -> &'static str {
        &SHADER_F32
    }

    /// Execute the window function and return the output tensor.
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = &self.device;

        // Create output buffer
        let output_buffer = device.create_buffer_f32(self.length)?;

        // Create uniform buffer for parameters
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            length: u32,
            window_type: u32,
        }

        let params = Params {
            length: self.length as u32,
            window_type: self.window_type.to_u32(),
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("WindowFunction Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        // Compile shader
        let shader_module =
            device.compile_shader(Self::wgsl_shader(), Some("WindowFunction Shader"));

        // Create bind group layout
        let bind_group_layout =
            device
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("WindowFunction Bind Group Layout"),
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
            label: Some("WindowFunction Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: output_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: params_buffer.as_entire_binding(),
                },
            ],
        });

        // Create compute pipeline
        let pipeline_layout =
            device
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("WindowFunction Pipeline Layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    immediate_size: 0,
                });

        let compute_pipeline =
            device
                .device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some("WindowFunction Pipeline"),
                    layout: Some(&pipeline_layout),
                    module: &shader_module,
                    entry_point: Some("main"),
                    cache: None,
                    compilation_options: Default::default(),
                });

        // Execute compute shader
        let mut encoder = device.create_encoder_guarded(&wgpu::CommandEncoderDescriptor {
            label: Some("WindowFunction Encoder"),
        });

        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("WindowFunction Pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(&compute_pipeline);
            compute_pass.set_bind_group(0, Some(&bind_group), &[]);
            // Deep Debt Evolution: Capability-based dispatch
            let caps = DeviceCapabilities::from_device(device);
            let optimal_wg_size = caps.optimal_workgroup_size(WorkloadType::ElementWise);
            let workgroups = (self.length as u32).div_ceil(optimal_wg_size);
            compute_pass.dispatch_workgroups(workgroups, 1, 1);
        }

        device.submit_commands(Some(encoder.finish()));

        // Return tensor without reading back (zero-copy)
        Ok(Tensor::from_buffer(
            output_buffer,
            vec![self.length],
            device.clone(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device_if_gpu_available;

    #[tokio::test]
    async fn test_window_hann() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let window = WindowFunction::new(512, WindowType::Hann, device)
            .unwrap()
            .execute()
            .unwrap();
        assert_eq!(window.shape(), &[512]);

        // Verify window values (would need readback to check exact values)
        let data = window.to_vec().unwrap();
        assert_eq!(data.len(), 512);
        // Hann window should be ~0 at edges and ~1 at center
        assert!(data[0].abs() < 0.1);
        assert!(data[256] > 0.9);
    }

    #[tokio::test]
    async fn test_window_hamming() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let window = WindowFunction::new(256, WindowType::Hamming, device)
            .unwrap()
            .execute()
            .unwrap();
        assert_eq!(window.shape(), &[256]);
    }

    #[tokio::test]
    async fn test_window_rectangular() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let window = WindowFunction::new(128, WindowType::Rectangular, device)
            .unwrap()
            .execute()
            .unwrap();
        let data = window.to_vec().unwrap();
        // Rectangular window should be all 1.0
        assert!(data.iter().all(|&x| (x - 1.0).abs() < 1e-5));
    }
}

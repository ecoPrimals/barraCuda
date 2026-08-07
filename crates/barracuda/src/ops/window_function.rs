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

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;
use std::sync::Arc;

/// f64 is the canonical source — math is universal, precision is silicon.
static SHADER_F64: &str = include_str!("../shaders/audio/window_function_f64.wgsl");

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
            Self::Hann => 0,
            Self::Hamming => 1,
            Self::Blackman => 2,
            Self::Bartlett => 3,
            Self::Rectangular => 4,
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
            return Err(BarracudaError::invalid_input(
                "Window length must be greater than 0",
            ));
        }
        Ok(Self {
            length,
            window_type,
            device,
        })
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

        ComputeDispatch::new(device, "WindowFunction")
            .shader(SHADER_F64, "main")
            .storage_rw(0, &output_buffer)
            .uniform(1, &params_buffer)
            .dispatch_1d(self.length as u32)
            .submit()?;

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

    #[tokio::test]
    async fn test_window_hann() {
        let device = crate::device::test_pool::get_test_device().await;
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
        let device = crate::device::test_pool::get_test_device().await;
        let window = WindowFunction::new(256, WindowType::Hamming, device)
            .unwrap()
            .execute()
            .unwrap();
        assert_eq!(window.shape(), &[256]);
    }

    #[tokio::test]
    async fn test_window_rectangular() {
        let device = crate::device::test_pool::get_test_device().await;
        let window = WindowFunction::new(128, WindowType::Rectangular, device)
            .unwrap()
            .execute()
            .unwrap();
        let data = window.to_vec().unwrap();
        // Rectangular window should be all 1.0
        assert!(data.iter().all(|&x| (x - 1.0).abs() < 1e-5));
    }
}

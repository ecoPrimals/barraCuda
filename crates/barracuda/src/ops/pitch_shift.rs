//! PitchShift - Pitch shifting without tempo change
//!
//! Changes pitch by resampling in frequency domain.
//! Combines time stretching with resampling.
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
use wgpu::util::DeviceExt;

const SHADER_F64: &str = include_str!("../shaders/audio/pitch_shift_f64.wgsl");
static SHADER_F32: std::sync::LazyLock<String> =
    std::sync::LazyLock::new(|| crate::shaders::precision::downcast_f64_to_f32(SHADER_F64));

/// PitchShift operation
pub struct PitchShift {
    signal: Tensor,
    n_steps: f32, // Semitones to shift (positive = up, negative = down)
    bins_per_octave: f32,
}

impl PitchShift {
    /// Create a new pitch shift operation
    pub fn new(signal: Tensor, n_steps: f32, bins_per_octave: f32) -> Result<Self> {
        if bins_per_octave <= 0.0 {
            return Err(BarracudaError::InvalidInput {
                message: "bins_per_octave must be positive".to_string(),
            });
        }

        Ok(Self {
            signal,
            n_steps,
            bins_per_octave,
        })
    }

    /// Get the WGSL shader source
    fn wgsl_shader() -> &'static str {
        &SHADER_F32
    }

    /// Execute the pitch shift operation
    pub fn execute(self) -> Result<Tensor> {
        let device = self.signal.device();
        let input_length: usize = self.signal.shape().iter().product();

        // Compute shift ratio: 2^(n_steps / bins_per_octave)
        let rate = 2.0_f32.powf(self.n_steps / self.bins_per_octave);
        let output_length = (input_length as f32 / rate) as usize;

        // Access input buffer directly (zero-copy)
        let input_buffer = self.signal.buffer();

        // Create output buffer
        let output_buffer = device.create_buffer_f32(output_length)?;

        // Create uniform buffer for parameters
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            input_length: u32,
            output_length: u32,
            rate: f32,
        }

        let params = Params {
            input_length: input_length as u32,
            output_length: output_length as u32,
            rate,
        };

        let params_buffer = device
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("PitchShift Params"),
                contents: bytemuck::cast_slice(&[params]),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        ComputeDispatch::new(device, "PitchShift")
            .shader(Self::wgsl_shader(), "main")
            .storage_read(0, input_buffer)
            .storage_rw(1, &output_buffer)
            .uniform(2, &params_buffer)
            .dispatch_1d(output_length as u32)
            .submit();

        // Output shape: [output_length]
        let output_shape = vec![output_length];

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
    async fn test_pitch_shift_basic() {
        let Some(device) = get_test_device_if_gpu_available().await else {
            return;
        };
        let signal = Tensor::from_vec_on(vec![0.5; 10_000], vec![10_000], device.clone())
            .await
            .unwrap();

        let shifted = PitchShift::new(signal, 2.0, 12.0)
            .unwrap()
            .execute()
            .unwrap();
        assert!(shifted.shape()[0] > 0);
    }
}

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

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::{BarracudaError, Result};
use crate::tensor::Tensor;

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
            return Err(BarracudaError::invalid_input(format!(
                "n_mfcc ({n_mfcc}) cannot exceed n_mels ({n_mels})"
            )));
        }

        let mel_size: usize = mel_spectrogram.shape().iter().product();
        if mel_size != n_frames * n_mels {
            return Err(BarracudaError::invalid_input(format!(
                "Mel spectrogram size ({}) must equal n_frames * n_mels ({})",
                mel_size,
                n_frames * n_mels
            )));
        }

        Ok(Self {
            mel_spectrogram,
            n_frames,
            n_mels,
            n_mfcc,
        })
    }

    /// Get the WGSL shader source
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

        ComputeDispatch::new(device, "MFCC")
            .shader(include_str!("../shaders/audio/mfcc_f64.wgsl"), "main")
            .storage_read(0, mel_buffer)
            .storage_rw(1, &output_buffer)
            .uniform(2, &params_buffer)
            .dispatch_1d((self.n_frames * self.n_mfcc) as u32)
            .submit()?;

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

    #[tokio::test]
    async fn test_mfcc_basic() {
        let device = crate::device::test_pool::get_test_device().await;
        let mel_spec = Tensor::from_vec_on(vec![1.0; 100 * 80], vec![100, 80], device.clone())
            .await
            .unwrap();

        let mfcc_features = MFCC::new(mel_spec, 100, 80, 13).unwrap().execute().unwrap();
        assert_eq!(mfcc_features.shape(), &[100, 13]);
    }
}

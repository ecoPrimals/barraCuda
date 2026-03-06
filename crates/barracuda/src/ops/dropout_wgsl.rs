// SPDX-License-Identifier: AGPL-3.0-or-later
//! Dropout - Random dropout for regularization - Pure WGSL
//!
//! Deep Debt Principles:
//! - Self-knowledge: Operation knows its probability and seed
//! - Zero hardcoding: All parameters passed at runtime
//! - Modern idiomatic Rust: Safe, zero unsafe code
//! - Complete implementation: Production-ready, no mocks
//! - Hardware-agnostic: Pure WGSL for universal compute

use crate::device::ComputeDispatch;
use crate::error::Result;
use crate::tensor::Tensor;

/// f64 is the canonical source — f32 derived via `downcast_f64_to_f32` when needed.
const SHADER_F64: &str = include_str!("../shaders/dropout/dropout_f64.wgsl");

static SHADER_F32: std::sync::LazyLock<String> =
    std::sync::LazyLock::new(|| crate::shaders::precision::downcast_f64_to_f32(SHADER_F64));

/// Dropout operation - Random dropout for regularization
pub struct Dropout {
    input: Tensor,
    probability: f32,
    seed: u32,
}

impl Dropout {
    /// Create a new dropout operation
    #[must_use]
    pub fn new(input: Tensor, probability: f32, seed: u32) -> Self {
        Self {
            input,
            probability,
            seed,
        }
    }

    /// Get the WGSL shader source
    fn wgsl_shader() -> &'static str {
        &SHADER_F32
    }

    /// Execute the dropout operation
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let size: usize = self.input.shape().iter().product();

        // Calculate scale factor (1 / (1 - p)) for inverted dropout
        let scale = if self.probability < 1.0 {
            1.0 / (1.0 - self.probability)
        } else {
            0.0
        };

        // Create buffers
        // Access input buffer directly (zero-copy)
        let input_buffer = self.input.buffer();

        let output_buffer = device.create_buffer_f32(size)?;

        // Create uniform buffer for parameters
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            size: u32,
            probability: f32,
            scale: f32,
            seed: u32,
        }

        let params = Params {
            size: size as u32,
            probability: self.probability,
            scale,
            seed: self.seed,
        };

        let params_buffer = device.create_uniform_buffer("Dropout Params", &params);

        ComputeDispatch::new(device, "dropout")
            .shader(Self::wgsl_shader(), "main")
            .storage_read(0, input_buffer)
            .storage_rw(1, &output_buffer)
            .uniform(2, &params_buffer)
            .dispatch_1d(size as u32)
            .submit()?;

        // Read back results
        let output_data = crate::utils::read_buffer(device, &output_buffer, size)?;

        Ok(Tensor::new(
            output_data,
            self.input.shape().to_vec(),
            device.clone(),
        ))
    }
}

impl Tensor {
    /// Apply dropout with given probability
    /// # Arguments
    /// * `probability` - Probability of dropping each element (0.0 to 1.0)
    /// * `seed` - Random seed for reproducibility
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn dropout_wgsl(self, probability: f32, seed: u32) -> Result<Self> {
        Dropout::new(self, probability, seed).execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    async fn get_test_device() -> Option<std::sync::Arc<crate::device::WgpuDevice>> {
        crate::device::test_pool::get_test_device_if_gpu_available().await
    }

    #[tokio::test]
    async fn test_dropout_deterministic() {
        let Some(device) = get_test_device().await else {
            return;
        };
        let data = vec![1.0; 100];
        let input = Tensor::new(data, vec![100], device.clone());

        let Ok(output1) = input.clone().dropout_wgsl(0.5, 42) else {
            if device.is_lost() {
                return;
            }
            panic!("dropout_wgsl failed on non-lost device");
        };
        let Ok(output2) = input.dropout_wgsl(0.5, 42) else {
            if device.is_lost() {
                return;
            }
            panic!("dropout_wgsl failed on non-lost device");
        };

        let (Ok(v1), Ok(v2)) = (output1.to_vec(), output2.to_vec()) else {
            if device.is_lost() {
                return;
            }
            panic!("readback failed on non-lost device");
        };
        assert_eq!(v1, v2);
    }

    #[tokio::test]
    async fn test_dropout_zero_probability() {
        let Some(device) = get_test_device().await else {
            return;
        };
        let data = vec![1.0, 2.0, 3.0];
        let input = Tensor::new(data.clone(), vec![3], device.clone());

        let Ok(output) = input.dropout_wgsl(0.0, 42) else {
            if device.is_lost() {
                return;
            }
            panic!("dropout_wgsl failed on non-lost device");
        };

        let result = output.to_vec().unwrap();
        assert_eq!(result[0], 1.0);
        assert_eq!(result[1], 2.0);
        assert_eq!(result[2], 3.0);
    }

    #[tokio::test]
    async fn test_dropout_full_probability() {
        let Some(device) = get_test_device().await else {
            return;
        };
        let data = vec![1.0, 2.0, 3.0];
        let input = Tensor::new(data, vec![3], device.clone());

        let Ok(output) = input.dropout_wgsl(1.0, 42) else {
            if device.is_lost() {
                return;
            }
            panic!("dropout_wgsl failed on non-lost device");
        };

        let result = output.to_vec().unwrap();
        assert_eq!(result[0], 0.0);
        assert_eq!(result[1], 0.0);
        assert_eq!(result[2], 0.0);
    }
}

// SPDX-License-Identifier: AGPL-3.0-or-later
//! `LogSumExp` - Pure WGSL (f64)
//!
//! Deep Debt Principles:
//! - Self-knowledge: Operation knows its computation
//! - Zero hardcoding: Hardware-agnostic implementation
//! - Modern idiomatic Rust: Safe, zero unsafe code
//! - Complete implementation: Production-ready, no mocks
//! - Hardware-agnostic: Pure WGSL for universal compute
//!
//! Computes log-sum-exp with numerical stability (f64 precision).
//! Used in softmax, log-likelihood computations.
//! Expects f64 input tensor (use `Tensor::from_data_pod` with &[f64]).

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::Result;
use crate::tensor::Tensor;

/// `LogSumExp` operation
pub struct LogSumExp {
    input: Tensor,
}

/// f64 is the canonical source — math is universal, precision is silicon.
static WGSL_LOGSUMEXP_REDUCE_F64: &str =
    include_str!("../shaders/reduce/logsumexp_reduce_f64.wgsl");
/// Batched logsumexp over rows [batch × width] (neuralSpring).
pub static WGSL_LOGSUMEXP_REDUCE: std::sync::LazyLock<String> = std::sync::LazyLock::new(|| {
    crate::shaders::precision::downcast_f64_to_f32_with_transcendentals(WGSL_LOGSUMEXP_REDUCE_F64)
});

impl LogSumExp {
    /// Create a new logsumexp operation
    #[must_use]
    pub fn new(input: Tensor) -> Self {
        Self { input }
    }

    /// WGSL kernel for logsumexp (f32 variant).
    pub const WGSL_LOGSUMEXP_F32: &str = include_str!("../shaders/math/logsumexp.wgsl");

    /// f64 version for universal math library portability.
    #[must_use]
    pub fn wgsl_shader_f64() -> &'static str {
        include_str!("../shaders/math/logsumexp_f64.wgsl")
    }

    /// Execute the logsumexp operation
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.input.device();
        let size: usize = self.input.shape().iter().product();

        // Access input buffer directly (zero-copy)
        let input_buffer = self.input.buffer();

        // Output is a single scalar (f64)
        let output_buffer = device.create_buffer_f64(1)?;

        // Create uniform buffer for parameters
        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Metadata {
            size: u32,
        }

        let metadata = Metadata { size: size as u32 };

        let metadata_buffer = device.create_uniform_buffer("LogSumExp Metadata", &metadata);

        ComputeDispatch::new(device, "LogSumExp")
            .shader(Self::wgsl_shader_f64(), "main")
            .f64()
            .storage_read(0, input_buffer)
            .storage_rw(1, &output_buffer)
            .uniform(2, &metadata_buffer)
            .dispatch_1d(size as u32)
            .submit()?;

        // Return tensor without reading back (zero-copy)
        Ok(Tensor::from_buffer(output_buffer, vec![1], device.clone()))
    }
}

impl Tensor {
    /// Compute log-sum-exp (numerically stable)
    /// # Errors
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn logsumexp(self) -> Result<Self> {
        LogSumExp::new(self).execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::test_pool::get_test_device_if_f64_gpu_available;

    #[tokio::test]
    async fn test_logsumexp_basic() {
        let Some(device) = get_test_device_if_f64_gpu_available().await else {
            return;
        };
        let input = Tensor::from_f64_data(&[1.0f64, 2.0, 3.0, 4.0], vec![4], device).unwrap();

        let output = input.logsumexp().unwrap();
        let result = output.to_f64_vec().unwrap();
        assert_eq!(result.len(), 1);
        assert!(result[0].is_finite());
        assert!(result[0] >= 4.0);
    }

    #[tokio::test]
    async fn test_logsumexp_edge_cases() {
        let Some(device) = get_test_device_if_f64_gpu_available().await else {
            return;
        };
        let input = Tensor::from_f64_data(&[5.0f64], vec![1], device.clone()).unwrap();
        let output = input.logsumexp().unwrap();
        let result = output.to_f64_vec().unwrap();
        assert_eq!(result.len(), 1);
        assert!((result[0] - 5.0).abs() < 0.01);

        let input = Tensor::from_f64_data(&[0.0f64, 0.0, 0.0], vec![3], device).unwrap();
        let output = input.logsumexp().unwrap();
        let result = output.to_f64_vec().unwrap();
        assert!(result[0].is_finite());
    }

    #[tokio::test]
    async fn test_logsumexp_large_values() {
        let Some(device) = get_test_device_if_f64_gpu_available().await else {
            return;
        };
        let input = Tensor::from_f64_data(&[100.0f64, 101.0, 102.0], vec![3], device).unwrap();
        let output = input.logsumexp().unwrap();
        let result = output.to_f64_vec().unwrap();
        assert!(result[0].is_finite());
        assert!(result[0] > 102.0);
    }
}

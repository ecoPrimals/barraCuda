// SPDX-License-Identifier: AGPL-3.0-or-later
//! `RANDOM_UNIFORM` - Uniform random sampling - Pure WGSL
//!
//! Deep Debt Principles:
//! - Self-knowledge: Operation knows its computation
//! - Zero hardcoding: Hardware-agnostic implementation
//! - Modern idiomatic Rust: Safe, zero unsafe code
//! - Complete implementation: Production-ready, no mocks
//! - Hardware-agnostic: Pure WGSL for universal compute

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::Result;
use crate::tensor::Tensor;
use std::sync::Arc;

/// Uniform random sampling (GPU accelerated)
pub struct RandomUniformGpu {
    device: Arc<crate::device::WgpuDevice>,
    n_samples: u32,
    bounds: Vec<f32>, // Interleaved [lo0, hi0, lo1, hi1, ...]
    seed: u32,
}

impl RandomUniformGpu {
    /// Create new uniform random sampler
    #[must_use]
    pub fn new(
        device: Arc<crate::device::WgpuDevice>,
        n_samples: u32,
        bounds: &[(f32, f32)],
        seed: u32,
    ) -> Self {
        let bounds_flat: Vec<f32> = bounds.iter().flat_map(|b| <[f32; 2]>::from(*b)).collect();
        Self {
            device,
            n_samples,
            bounds: bounds_flat,
            seed,
        }
    }

    /// Generate uniform random samples
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn generate(self) -> Result<Tensor> {
        let device = &self.device;
        let n_dims = self.bounds.len() / 2;
        let output_size = (self.n_samples as usize) * n_dims;

        let output_buffer = device.create_buffer_f32(output_size)?;

        // Create bounds buffer
        let bounds_buffer = device.create_buffer_f32_init("RandomUniform Bounds", &self.bounds);

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            n_samples: u32,
            n_dims: u32,
            seed: u32,
            _pad: u32,
        }

        let params = Params {
            n_samples: self.n_samples,
            n_dims: n_dims as u32,
            seed: self.seed,
            _pad: 0,
        };
        let params_buffer = device.create_uniform_buffer("RandomUniform Params", &params);

        ComputeDispatch::new(device, "RandomUniform")
            .shader(
                include_str!("../shaders/sample/random_uniform_f64.wgsl"),
                "main",
            )
            .storage_read(0, &bounds_buffer)
            .storage_rw(1, &output_buffer)
            .uniform(2, &params_buffer)
            .dispatch_1d(self.n_samples)
            .submit()?;

        Ok(Tensor::from_buffer(
            output_buffer,
            vec![self.n_samples as usize, n_dims],
            self.device.clone(),
        ))
    }
}

/// Generate uniform random samples on GPU
///
/// # Errors
///
/// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
/// readback fails (e.g. device lost or out of memory).
pub fn random_uniform_gpu(
    device: Arc<crate::device::WgpuDevice>,
    n_samples: u32,
    bounds: &[(f32, f32)],
    seed: u32,
) -> Result<Tensor> {
    RandomUniformGpu::new(device, n_samples, bounds, seed).generate()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_random_uniform_basic() {
        let device = crate::device::test_pool::get_test_device().await;
        let bounds = vec![(0.0, 1.0), (-1.0, 1.0)];
        let result = random_uniform_gpu(device, 100, &bounds, 42).unwrap();
        let data = result.to_vec().unwrap();

        // Should have 100 * 2 = 200 values
        assert_eq!(data.len(), 200);

        // Check bounds for each sample
        for i in 0..100 {
            let x = data[i * 2];
            let y = data[i * 2 + 1];
            assert!((0.0..=1.0).contains(&x), "x={x} out of [0,1]");
            assert!((-1.0..=1.0).contains(&y), "y={y} out of [-1,1]");
        }
    }

    #[tokio::test]
    async fn test_random_uniform_different_seeds() {
        let device = crate::device::test_pool::get_test_device().await;
        let bounds = vec![(0.0, 1.0)];

        let r1 = random_uniform_gpu(device.clone(), 10, &bounds, 42).unwrap();
        let r2 = random_uniform_gpu(device, 10, &bounds, 99).unwrap();

        let d1 = r1.to_vec().unwrap();
        let d2 = r2.to_vec().unwrap();

        // Different seeds should give different results
        let different = d1.iter().zip(d2.iter()).any(|(a, b)| (a - b).abs() > 1e-6);
        assert!(different, "Different seeds should give different results");
    }
}

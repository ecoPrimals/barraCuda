// SPDX-License-Identifier: AGPL-3.0-or-later
//! PRNG Xoshiro128** - High-quality pseudorandom f64 generator - Pure WGSL
//!
//! Deep Debt Principles:
//! - Self-knowledge: Operation knows its computation
//! - Zero hardcoding: Hardware-agnostic implementation
//! - Modern idiomatic Rust: Safe, zero unsafe code
//! - Complete implementation: Production-ready, no mocks
//! - Hardware-agnostic: Pure WGSL for universal compute
//!
//! f64 pipeline: seeds as `array<u32>` (1 per output, expanded to 4-stride),
//! output as `array<f64>` in [0, 1).

use crate::device::compute_pipeline::ComputeDispatch;
use crate::error::Result;
use crate::tensor::Tensor;

/// Xoshiro128** PRNG for GPU random number generation.
pub struct PrngXoshiro {
    seeds: Tensor,
    offset: u32,
}

impl PrngXoshiro {
    /// Creates a new PRNG with the given seeds tensor (u32) and offset.
    #[must_use]
    pub fn new(seeds: Tensor, offset: u32) -> Self {
        Self { seeds, offset }
    }

    /// Xoshiro128** stateful PRNG (neuralSpring): per-thread state, `n_samples` per thread.
    #[must_use]
    pub fn wgsl_xoshiro128ss() -> &'static str {
        include_str!("../shaders/misc/xoshiro128ss_f64.wgsl")
    }

    /// WGSL kernel for Xoshiro PRNG (f32 variant).
    pub const WGSL_PRNG_XOSHIRO_F32: &str = include_str!("../shaders/misc/prng_xoshiro.wgsl");

    /// f64 version for universal math library portability.
    #[must_use]
    pub fn wgsl_shader_f64() -> &'static str {
        include_str!("../shaders/misc/prng_xoshiro_f64.wgsl")
    }

    /// Executes the PRNG and returns random values in [0, 1).
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn execute(self) -> Result<Tensor> {
        let device = self.seeds.device();
        let seed_count: usize = self.seeds.shape().iter().product();

        if seed_count == 0 {
            return Ok(Tensor::new(vec![], vec![0], device.clone()));
        }

        // f64 shader expects 4 u32s per output (seed_base = idx * 4); expand 1→4 stride
        let seeds_data = device.read_buffer_u32(self.seeds.buffer(), seed_count)?;
        let expanded: Vec<u32> = (0..seed_count)
            .flat_map(|i| [seeds_data[i], 0u32, 0u32, 0u32])
            .collect();
        let seeds_buffer = device.create_buffer_u32_init("PRNG Xoshiro seeds expanded", &expanded);

        let output_buffer = device.create_buffer_f64(seed_count)?;

        #[repr(C)]
        #[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
        struct Params {
            size: u32,
            offset: u32,
        }

        let params = Params {
            size: seed_count as u32,
            offset: self.offset,
        };
        let params_buffer = device.create_uniform_buffer("PRNG Xoshiro Params", &params);

        ComputeDispatch::new(device, "PRNG Xoshiro")
            .shader(
                include_str!("../shaders/misc/prng_xoshiro_f64.wgsl"),
                "main",
            )
            .f64()
            .storage_read(0, &seeds_buffer)
            .storage_rw(1, &output_buffer)
            .uniform(2, &params_buffer)
            .dispatch_1d(seed_count as u32)
            .submit()?;

        Ok(Tensor::from_buffer(
            output_buffer,
            self.seeds.shape().to_vec(),
            device.clone(),
        ))
    }
}

impl Tensor {
    /// Generate random f64 values in [0, 1) using xoshiro128** PRNG.
    /// Seeds tensor must contain u32 data (use `Tensor::from_data_pod` with u32).
    ///
    /// # Errors
    ///
    /// Returns [`Err`] if buffer allocation, GPU dispatch, or buffer
    /// readback fails (e.g. device lost or out of memory).
    pub fn prng_xoshiro(self, offset: u32) -> Result<Self> {
        PrngXoshiro::new(self, offset).execute()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_prng_xoshiro() {
        let device = crate::device::test_pool::get_test_device().await;
        let seeds: Vec<u32> = vec![1, 2, 3, 4, 5, 100, 200, 300];
        let seeds_tensor = Tensor::from_data_pod(&seeds, vec![8], device).unwrap();
        let output = seeds_tensor.prng_xoshiro(0).unwrap();
        let result = output.to_f64_vec().unwrap();
        assert_eq!(result.len(), 8);
        assert!(
            result
                .iter()
                .all(|&x| (0.0..1.0).contains(&x) && x.is_finite())
        );
    }

    #[tokio::test]
    async fn test_prng_xoshiro_statistical_validation() {
        let device = crate::device::test_pool::get_test_device().await;
        let n = 10_000_u32;
        let seeds: Vec<u32> = (1..=n).collect();
        let seeds_tensor = Tensor::from_data_pod(&seeds, vec![n as usize], device).unwrap();
        let output = seeds_tensor.prng_xoshiro(0).unwrap();
        let result = output.to_f64_vec().unwrap();
        assert_eq!(result.len(), n as usize);

        assert!(
            result
                .iter()
                .all(|&x| (0.0..1.0).contains(&x) && x.is_finite())
        );

        let mean = result.iter().sum::<f64>() / n as f64;
        assert!(
            (mean - 0.5).abs() < 0.02,
            "GPU xoshiro U(0,1) mean {mean} outside tolerance"
        );

        let var = result.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        let expected_var = 1.0 / 12.0;
        assert!(
            (var - expected_var).abs() < 0.02,
            "GPU xoshiro U(0,1) variance {var} vs expected {expected_var}"
        );

        let n_bins = 10;
        let mut bins = vec![0u32; n_bins];
        for &x in &result {
            let bin = (x * n_bins as f64).min((n_bins - 1) as f64) as usize;
            bins[bin] += 1;
        }
        let expected_count = n as f64 / n_bins as f64;
        let chi2: f64 = bins
            .iter()
            .map(|&b| (b as f64 - expected_count).powi(2) / expected_count)
            .sum();
        assert!(
            chi2 < 30.0,
            "GPU xoshiro chi-squared {chi2} exceeds critical value (df=9, p=0.001~27.9)"
        );
    }

    #[tokio::test]
    async fn test_prng_xoshiro_seed_independence() {
        let device = crate::device::test_pool::get_test_device().await;
        let seeds_a: Vec<u32> = vec![42, 43, 44, 45];
        let seeds_b: Vec<u32> = vec![100, 200, 300, 400];
        let t_a = Tensor::from_data_pod(&seeds_a, vec![4], device.clone()).unwrap();
        let t_b = Tensor::from_data_pod(&seeds_b, vec![4], device).unwrap();
        let out_a = t_a.prng_xoshiro(0).unwrap().to_f64_vec().unwrap();
        let out_b = t_b.prng_xoshiro(0).unwrap().to_f64_vec().unwrap();
        assert_ne!(
            out_a, out_b,
            "different seeds must produce different output"
        );
    }
}

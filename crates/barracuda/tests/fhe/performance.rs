// SPDX-License-Identifier: AGPL-3.0-or-later
//! Performance regression tests for FHE shaders.
//!
//! Cold-start thresholds account for first-run shader compilation on
//! software renderers (llvmpipe).  Warm-path targets are documented in
//! comments but not asserted — profile with `cargo bench` instead.

#![expect(clippy::unwrap_used, reason = "tests")]
use super::helpers::*;
use barracuda::device::WgpuDevice;
use barracuda::ops::fhe_fast_poly_mul::FheFastPolyMul;
use barracuda::ops::fhe_ntt::FheNtt;
use barracuda::ops::fhe_poly_add::create_fhe_poly_tensor;
use std::sync::Arc;
use std::time::Duration;

/// Cold-start budget for NTT(N=4096): includes shader compilation on
/// llvmpipe.  Warm target is <200μs on discrete GPU.
const NTT_N4096_COLD_BUDGET: Duration = Duration::from_secs(10);

/// Cold-start budget for fast polynomial multiply(N=4096): includes
/// three shader compilations (NTT + pointwise + INTT) on llvmpipe.
/// Warm target is <500μs on discrete GPU.
const FAST_POLY_MUL_N4096_COLD_BUDGET: Duration = Duration::from_secs(20);

#[tokio::test]
async fn test_ntt_performance_n4096() {
    if !crate::common::run_gpu_resilient_async(|| async {
        let degree = 4096u32;
        let modulus = 65537u64;
        let root = find_root_of_unity(degree, modulus).expect("65537 supports N=4096");
        let input = random_polynomial(degree as usize, modulus);

        let device = Arc::new(
            WgpuDevice::new()
                .await
                .expect("Failed to create GPU device"),
        );
        let input_tensor = create_fhe_poly_tensor(&input, device.clone())
            .await
            .unwrap();

        let start = std::time::Instant::now();

        let ntt = FheNtt::new(input_tensor, degree, modulus, root).unwrap();
        let _result_tensor = ntt.execute().unwrap();

        let elapsed = start.elapsed();
        println!("NTT(N=4096) performance: {elapsed:?}");

        assert!(
            elapsed < NTT_N4096_COLD_BUDGET,
            "NTT exceeded cold-start budget ({elapsed:?} > {NTT_N4096_COLD_BUDGET:?})"
        );
    }) {
        return;
    }
}

#[tokio::test]
async fn test_fast_poly_mul_performance_n4096() {
    if !crate::common::run_gpu_resilient_async(|| async {
        let degree = 4096u32;
        let modulus = 65537u64;
        let root = find_root_of_unity(degree, modulus).expect("65537 supports N=4096");
        let a = random_polynomial(degree as usize, modulus);
        let b = random_polynomial(degree as usize, modulus);

        let device = Arc::new(
            WgpuDevice::new()
                .await
                .expect("Failed to create GPU device"),
        );
        let a_tensor = create_fhe_poly_tensor(&a, device.clone()).await.unwrap();
        let b_tensor = create_fhe_poly_tensor(&b, device.clone()).await.unwrap();

        let start = std::time::Instant::now();

        let fast_mul = FheFastPolyMul::new(a_tensor, b_tensor, degree, modulus, root).unwrap();
        let _result_tensor = fast_mul.execute().unwrap();

        let elapsed = start.elapsed();
        println!("Fast multiply(N=4096) performance: {elapsed:?}");

        assert!(
            elapsed < FAST_POLY_MUL_N4096_COLD_BUDGET,
            "Fast poly mul exceeded cold-start budget ({elapsed:?} > {FAST_POLY_MUL_N4096_COLD_BUDGET:?})"
        );
    }) {
        return;
    }
}

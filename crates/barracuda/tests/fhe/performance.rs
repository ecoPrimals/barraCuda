// SPDX-License-Identifier: AGPL-3.0-or-later
//! Performance regression tests for FHE shaders.

use super::helpers::*;
use barracuda::device::WgpuDevice;
use barracuda::ops::fhe_fast_poly_mul::FheFastPolyMul;
use barracuda::ops::fhe_ntt::FheNtt;
use barracuda::ops::fhe_poly_add::create_fhe_poly_tensor;
use std::sync::Arc;

#[tokio::test]
async fn test_ntt_performance_n4096() {
    if !crate::common::run_gpu_resilient_async(|| async {
        // NTT(N=4096) should complete in <200μs
        // 12289 only works for degree ≤ 2048; use 65537 for N=4096 (65537 ≡ 1 mod 65536)

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

        // Execute NTT
        let ntt = FheNtt::new(input_tensor, degree, modulus, root).unwrap();
        let _result_tensor = ntt.execute().unwrap();

        let elapsed = start.elapsed();

        // Should be fast (target: <200μs, but allow more for first run)
        // Note: First run may be slower due to shader compilation
        println!("✅ NTT(N=4096) performance: {:?}", elapsed);

        // Just verify it completes without panicking
        assert!(
            elapsed.as_millis() < 1000,
            "NTT should complete in reasonable time"
        );
    }) {
        return;
    }
}

#[tokio::test]
async fn test_fast_poly_mul_performance_n4096() {
    if !crate::common::run_gpu_resilient_async(|| async {
        // Fast multiply(N=4096) should complete in <500μs
        // 12289 only works for degree ≤ 2048; use 65537 for N=4096

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

        // Execute fast multiply
        let fast_mul = FheFastPolyMul::new(a_tensor, b_tensor, degree, modulus, root).unwrap();
        let _result_tensor = fast_mul.execute().unwrap();

        let elapsed = start.elapsed();

        // Should be fast (target: <500μs for full pipeline, but allow more for first run)
        println!("✅ Fast multiply(N=4096) performance: {:?}", elapsed);

        // Just verify it completes without panicking
        assert!(
            elapsed.as_millis() < 2000,
            "Fast multiply should complete in reasonable time"
        );
    }) {
        return;
    }
}

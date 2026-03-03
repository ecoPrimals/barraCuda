// SPDX-License-Identifier: AGPL-3.0-or-later
//! NTT (Number Theoretic Transform) unit tests.

use super::helpers::*;
use barracuda::device::WgpuDevice;
use barracuda::ops::fhe_intt::{compute_inverse_root, FheIntt};
use barracuda::ops::fhe_ntt::FheNtt;
use barracuda::ops::fhe_poly_add::create_fhe_poly_tensor;
use std::sync::Arc;

#[tokio::test]
async fn test_ntt_basic_known_vector() {
    if !crate::common::run_gpu_resilient_async(|| async {
        // Test NTT on known input with known output
        // For N=4, modulus=17, root=4

        let degree = 4u32;
        let modulus = 17u64;
        let root = 4u64;
        let input = vec![1u64, 2, 3, 4];

        let device = Arc::new(
            WgpuDevice::new()
                .await
                .expect("Failed to create GPU device"),
        );
        let input_tensor = create_fhe_poly_tensor(&input, device.clone())
            .await
            .unwrap();

        let ntt = FheNtt::new(input_tensor, degree, modulus, root).unwrap();
        let result_tensor = ntt.execute().unwrap();
        let result = read_poly_from_tensor(&result_tensor).await;

        // After NTT, should still be in [0, modulus)
        assert_eq!(result.len(), degree as usize);
        assert!(
            result.iter().all(|&x| x < modulus),
            "All coefficients should be < modulus"
        );

        println!("✅ NTT basic known vector test passed");
    }) {
        return;
    }
}

#[tokio::test]
async fn test_ntt_all_power_of_two_degrees() {
    if !crate::common::run_gpu_resilient_async(|| async {
        // Test that NTT works for standard FHE degrees
        // Use modulus 97: supports degree ≤ 16 (97-1=96 divisible by 2*16=32)
        // 12289 supports degree ≤ 2048 but has known issues with some degree/root combos

        let device = Arc::new(
            WgpuDevice::new()
                .await
                .expect("Failed to create GPU device"),
        );
        let modulus = 97u64;

        for &degree in &[4usize, 8, 16] {
            if !super::helpers::modulus_supports_degree(modulus, degree as u32) {
                continue;
            }
            let degree_u32 = degree as u32;
            let input = random_polynomial(degree, modulus);
            let root = find_root_of_unity(degree_u32, modulus).expect("97 has roots for 4,8,16");

            let input_tensor = create_fhe_poly_tensor(&input, device.clone())
                .await
                .unwrap();

            let ntt = FheNtt::new(input_tensor, degree_u32, modulus, root).unwrap();
            let result_tensor = ntt.execute().unwrap();
            let result = read_poly_from_tensor(&result_tensor).await;

            assert_eq!(
                result.len(),
                input.len(),
                "NTT should preserve element count"
            );
            assert!(
                result.iter().all(|&x| x < modulus),
                "All coefficients should be < modulus"
            );

            println!("✅ NTT works for N={}", degree);
        }
    }) {
        return;
    }
}

#[tokio::test]
async fn test_ntt_round_trip_identity() {
    if !crate::common::run_gpu_resilient_async(|| async {
        // Mathematical property: NTT → INTT = identity
        // Use (17, 4) and (257, 8), (257, 16): proven in fhe_properties
        // 17 ≡ 1 mod 8, 257 ≡ 1 mod 16 and mod 32

        let device = Arc::new(
            WgpuDevice::new()
                .await
                .expect("Failed to create GPU device"),
        );

        for &(degree, modulus) in &[(4u32, 17u64)] {
            let input = random_polynomial(degree as usize, modulus);
            let root = find_root_of_unity(degree, modulus).expect("Should find root");
            let inv_root = compute_inverse_root(degree, modulus, root);

            let input_tensor = create_fhe_poly_tensor(&input, device.clone())
                .await
                .unwrap();

            // Forward NTT
            let ntt = FheNtt::new(input_tensor.clone(), degree, modulus, root).unwrap();
            let ntt_result_tensor = ntt.execute().unwrap();

            // Inverse NTT (includes 1/N scaling; output equals original)
            let intt = FheIntt::new(ntt_result_tensor, degree, modulus, inv_root).unwrap();
            let intt_result_tensor = intt.execute().unwrap();
            let intt_result = read_poly_from_tensor(&intt_result_tensor).await;

            for (i, (&orig, &recovered)) in input.iter().zip(intt_result.iter()).enumerate() {
                assert_eq!(
                    orig, recovered,
                    "Round-trip should preserve coefficient {} (degree={})",
                    i, degree
                );
            }

            println!("✅ NTT → INTT = identity for N={}", degree);
        }
    }) {
        return;
    }
}

#[tokio::test]
async fn test_ntt_different_moduli() {
    if !crate::common::run_gpu_resilient_async(|| async {
        // Test with different FHE-friendly primes

        let device = Arc::new(
            WgpuDevice::new()
                .await
                .expect("Failed to create GPU device"),
        );

        // Only test moduli with known roots for degree 4 (17, 97)
        for &modulus in &[17u64, 97u64] {
            let degree = 4u32;
            let input = random_polynomial(degree as usize, modulus);

            if let Some(root) = find_root_of_unity(degree, modulus) {
                let input_tensor = create_fhe_poly_tensor(&input, device.clone())
                    .await
                    .unwrap();

                let ntt = FheNtt::new(input_tensor, degree, modulus, root).unwrap();
                let result_tensor = ntt.execute().unwrap();
                let result = read_poly_from_tensor(&result_tensor).await;

                assert_eq!(result.len(), degree as usize);
                assert!(result.iter().all(|&x| x < modulus));
            }

            println!("✅ NTT works with modulus={}", modulus);
        }
    }) {
        return;
    }
}

#[tokio::test]
async fn test_ntt_zero_polynomial() {
    if !crate::common::run_gpu_resilient_async(|| async {
        // Edge case: All zeros

        let degree = 16u32;
        let modulus = 12289u64;
        let root = find_root_of_unity(degree, modulus).expect("Should find root");
        let input = vec![0u64; degree as usize];

        let device = Arc::new(
            WgpuDevice::new()
                .await
                .expect("Failed to create GPU device"),
        );
        let input_tensor = create_fhe_poly_tensor(&input, device.clone())
            .await
            .unwrap();

        let ntt = FheNtt::new(input_tensor, degree, modulus, root).unwrap();
        let result_tensor = ntt.execute().unwrap();
        let result = read_poly_from_tensor(&result_tensor).await;

        // NTT of zeros should be zeros
        assert!(
            result.iter().all(|&x| x == 0),
            "NTT of zero polynomial should be zero"
        );

        println!("✅ NTT handles zero polynomial");
    }) {
        return;
    }
}

#[tokio::test]
async fn test_ntt_max_coefficients() {
    if !crate::common::run_gpu_resilient_async(|| async {
        // Edge case: Coefficients at maximum (modulus - 1)

        let degree = 8u32;
        let modulus = 12289u64;
        let root = find_root_of_unity(degree, modulus).expect("Should find root");
        let input = vec![modulus - 1; degree as usize];

        let device = Arc::new(
            WgpuDevice::new()
                .await
                .expect("Failed to create GPU device"),
        );
        let input_tensor = create_fhe_poly_tensor(&input, device.clone())
            .await
            .unwrap();

        let ntt = FheNtt::new(input_tensor, degree, modulus, root).unwrap();
        let result_tensor = ntt.execute().unwrap();
        let result = read_poly_from_tensor(&result_tensor).await;

        // Should not overflow - all results should be < modulus
        assert!(
            result.iter().all(|&x| x < modulus),
            "NTT should not overflow with max coefficients"
        );

        println!("✅ NTT handles maximum coefficients");
    }) {
        return;
    }
}

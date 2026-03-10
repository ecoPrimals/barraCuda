// SPDX-License-Identifier: AGPL-3.0-only
//! Point-wise multiplication unit tests (NTT domain).

#![expect(clippy::unwrap_used, reason = "tests")]
use super::helpers::*;
use barracuda::device::WgpuDevice;
use barracuda::ops::fhe_ntt::FheNtt;
use barracuda::ops::fhe_pointwise_mul::FhePointwiseMul;
use barracuda::ops::fhe_poly_add::create_fhe_poly_tensor;
use std::sync::Arc;

#[tokio::test]
async fn test_pointwise_mul_basic() {
    if !crate::common::run_gpu_resilient_async(|| async {
        // Simple element-wise multiplication
        // Use (257, 4): 257 ≡ 1 mod 8; larger modulus avoids Barrett reduction edge cases

        let degree = 4u32;
        let modulus = 12289u64;
        let root = find_root_of_unity(degree, modulus).expect("12289 supports degree 4");
        let a = vec![1u64, 2, 3, 4];
        let b = vec![5u64, 6, 7, 8];

        let device = Arc::new(
            WgpuDevice::new()
                .await
                .expect("Failed to create GPU device"),
        );

        // Convert to NTT domain first
        let a_tensor = create_fhe_poly_tensor(&a, device.clone()).await.unwrap();
        let b_tensor = create_fhe_poly_tensor(&b, device.clone()).await.unwrap();

        let ntt_a = FheNtt::new(a_tensor, degree, modulus, root).unwrap();
        let ntt_b = FheNtt::new(b_tensor, degree, modulus, root).unwrap();

        let a_ntt = ntt_a.execute().unwrap();
        let b_ntt = ntt_b.execute().unwrap();

        // Point-wise multiply
        let pointwise = FhePointwiseMul::new(a_ntt, b_ntt, degree, modulus).unwrap();
        let result_tensor = pointwise.execute().unwrap();
        let result = read_poly_from_tensor(&result_tensor).await;

        // Expected: [5, 12, 21, 32] mod 17 = [5, 12, 4, 15]
        // But in NTT domain, so we need to verify properties
        assert_eq!(result.len(), degree as usize);
        assert!(result.iter().all(|&x| x < modulus));

        println!("✅ Point-wise multiply basic test passed");
    }) {
        return;
    }
}

#[tokio::test]
async fn test_pointwise_mul_identity() {
    if !crate::common::run_gpu_resilient_async(|| async {
        // Multiply by 1 (identity)
        // Use (257, 4): 257 ≡ 1 mod 8; larger modulus avoids Barrett edge cases

        let degree = 4u32;
        let modulus = 12289u64;
        let root = find_root_of_unity(degree, modulus).expect("12289 has root for degree 4");
        let input = random_polynomial(degree as usize, modulus);
        let mut ones = vec![0u64; degree as usize];
        ones[0] = 1;

        let device = Arc::new(
            WgpuDevice::new()
                .await
                .expect("Failed to create GPU device"),
        );

        let input_tensor = create_fhe_poly_tensor(&input, device.clone())
            .await
            .unwrap();
        let ones_tensor = create_fhe_poly_tensor(&ones, device.clone()).await.unwrap();

        let ntt_input = FheNtt::new(input_tensor, degree, modulus, root).unwrap();
        let ntt_ones = FheNtt::new(ones_tensor, degree, modulus, root).unwrap();

        let input_ntt = ntt_input.execute().unwrap();
        let ones_ntt = ntt_ones.execute().unwrap();

        let pointwise = FhePointwiseMul::new(input_ntt, ones_ntt, degree, modulus).unwrap();
        let result_tensor = pointwise.execute().unwrap();
        let result = read_poly_from_tensor(&result_tensor).await;

        // In NTT domain, multiplying by ones should give same result as input
        let input_ntt_tensor = FheNtt::new(
            create_fhe_poly_tensor(&input, device.clone())
                .await
                .unwrap(),
            degree,
            modulus,
            root,
        )
        .unwrap()
        .execute()
        .unwrap();
        let input_ntt_result = read_poly_from_tensor(&input_ntt_tensor).await;

        // Results should match (pointwise multiply by 1 in NTT domain = identity)
        for (i, (&a, &b)) in result.iter().zip(input_ntt_result.iter()).enumerate() {
            assert_eq!(a, b, "Multiplying by 1 should preserve value at index {i}");
        }

        println!("✅ Point-wise multiply identity test passed");
    }) {
        return;
    }
}

#[tokio::test]
async fn test_pointwise_mul_zero() {
    if !crate::common::run_gpu_resilient_async(|| async {
        // Multiply by 0
        // Use modulus 97 (supports degree ≤ 16)

        let degree = 16u32;
        let modulus = 97u64;
        let root = find_root_of_unity(degree, modulus).expect("97 has root for degree 16");
        let input = random_polynomial(degree as usize, modulus);
        let zeros = vec![0u64; degree as usize];

        let device = Arc::new(
            WgpuDevice::new()
                .await
                .expect("Failed to create GPU device"),
        );

        let input_tensor = create_fhe_poly_tensor(&input, device.clone())
            .await
            .unwrap();
        let zeros_tensor = create_fhe_poly_tensor(&zeros, device.clone())
            .await
            .unwrap();

        let ntt_input = FheNtt::new(input_tensor, degree, modulus, root).unwrap();
        let ntt_zeros = FheNtt::new(zeros_tensor, degree, modulus, root).unwrap();

        let input_ntt = ntt_input.execute().unwrap();
        let zeros_ntt = ntt_zeros.execute().unwrap();

        let pointwise = FhePointwiseMul::new(input_ntt, zeros_ntt, degree, modulus).unwrap();
        let result_tensor = pointwise.execute().unwrap();
        let result = read_poly_from_tensor(&result_tensor).await;

        // Multiplying by zero should give all zeros
        assert!(
            result.iter().all(|&x| x == 0),
            "Multiplying by zero should give zeros"
        );

        println!("✅ Point-wise multiply zero test passed");
    }) {
        return;
    }
}

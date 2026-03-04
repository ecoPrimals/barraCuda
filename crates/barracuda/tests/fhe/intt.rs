// SPDX-License-Identifier: AGPL-3.0-or-later
//! INTT (Inverse Number Theoretic Transform) unit tests.

#![expect(clippy::unwrap_used, reason = "tests")]
use super::helpers::*;
use barracuda::device::WgpuDevice;
use barracuda::ops::fhe_intt::{compute_inverse_root, FheIntt};
use barracuda::ops::fhe_ntt::FheNtt;
use barracuda::ops::fhe_poly_add::create_fhe_poly_tensor;
use std::sync::Arc;

#[tokio::test]
async fn test_intt_basic() {
    if !crate::common::run_gpu_resilient_async(|| async {
        // INTT should be inverse of NTT
        // Use (17, 4): 17 ≡ 1 mod 8, minimal valid pair

        let degree = 4u32;
        let modulus = 17u64;
        let root = find_root_of_unity(degree, modulus).expect("17 has root for degree 4");
        let inv_root = compute_inverse_root(degree, modulus, root);
        let input = random_polynomial(degree as usize, modulus);

        let device = Arc::new(
            WgpuDevice::new()
                .await
                .expect("Failed to create GPU device"),
        );
        let input_tensor = create_fhe_poly_tensor(&input, device.clone())
            .await
            .unwrap();

        // Forward NTT
        let ntt = FheNtt::new(input_tensor.clone(), degree, modulus, root).unwrap();
        let ntt_result_tensor = ntt.execute().unwrap();

        // Inverse NTT (output equals original; INTT includes 1/N scaling)
        let intt = FheIntt::new(ntt_result_tensor, degree, modulus, inv_root).unwrap();
        let intt_result_tensor = intt.execute().unwrap();
        let intt_result = read_poly_from_tensor(&intt_result_tensor).await;

        assert_eq!(input.len(), intt_result.len());
        for (i, (&orig, &recovered)) in input.iter().zip(intt_result.iter()).enumerate() {
            assert_eq!(
                orig, recovered,
                "INTT should recover original coefficient {}",
                i
            );
        }

        println!("✅ INTT basic test passed");
    }) {
        return;
    }
}

#[tokio::test]
async fn test_intt_scaling() {
    if !crate::common::run_gpu_resilient_async(|| async {
        // INTT includes 1/N scaling; NTT(INTT) = identity
        // For input all 1s: NTT([1,1,...,1]) = [N,0,...,0], INTT gives back [1,1,...,1]
        // Use (17, 4): 17 ≡ 1 mod 8

        let degree = 4u32;
        let modulus = 17u64;
        let root = find_root_of_unity(degree, modulus).expect("17 has root for degree 4");
        let inv_root = compute_inverse_root(degree, modulus, root);

        // Create a polynomial with all coefficients = 1
        let input = vec![1u64; degree as usize];

        let device = Arc::new(
            WgpuDevice::new()
                .await
                .expect("Failed to create GPU device"),
        );
        let input_tensor = create_fhe_poly_tensor(&input, device.clone())
            .await
            .unwrap();

        // Forward NTT
        let ntt = FheNtt::new(input_tensor, degree, modulus, root).unwrap();
        let ntt_result_tensor = ntt.execute().unwrap();

        // Inverse NTT (should scale by N^(-1))
        let intt = FheIntt::new(ntt_result_tensor, degree, modulus, inv_root).unwrap();
        let intt_result_tensor = intt.execute().unwrap();
        let intt_result = read_poly_from_tensor(&intt_result_tensor).await;

        // INTT(NTT([1,1,...,1])) = [1,1,...,1]
        for &coeff in &intt_result {
            assert_eq!(coeff, 1, "INTT(NTT(ones)) should recover ones");
        }

        println!("✅ INTT scaling verified");
    }) {
        return;
    }
}

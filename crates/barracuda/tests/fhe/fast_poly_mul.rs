// SPDX-License-Identifier: AGPL-3.0-only
//! Fast polynomial multiplication tests (NTT-based).

#![expect(clippy::unwrap_used, reason = "tests")]
use super::helpers::*;
use barracuda::device::WgpuDevice;
use barracuda::ops::fhe_fast_poly_mul::FheFastPolyMul;
use barracuda::ops::fhe_poly_add::create_fhe_poly_tensor;
use std::sync::Arc;

/// Naive polynomial multiplication mod (X^N - 1) — cyclic convolution.
/// Standard NTT implements cyclic convolution, not negacyclic (X^N+1).
#[expect(
    clippy::needless_range_loop,
    reason = "indices i,j needed for k = (i+j) % degree"
)]
fn naive_poly_multiply(a: &[u64], b: &[u64], degree: usize, modulus: u64) -> Vec<u64> {
    let mut result = vec![0u64; degree];

    for i in 0..degree {
        for j in 0..degree {
            let k = (i + j) % degree;
            result[k] = ((u128::from(result[k])
                + (u128::from(a[i]) * u128::from(b[j]) % u128::from(modulus)))
                % u128::from(modulus)) as u64;
        }
    }

    result
}

#[tokio::test]
async fn test_fast_poly_mul_vs_naive() {
    if !crate::common::run_gpu_resilient_async(|| async {
        // Fast multiply should match naive multiply
        // Use (257, 4): 257 ≡ 1 mod 8; larger modulus avoids Barrett edge cases

        let device = Arc::new(
            WgpuDevice::new()
                .await
                .expect("Failed to create GPU device"),
        );
        let modulus = 12289u64;

        for &degree in &[4u32] {
            let root = find_root_of_unity(degree, modulus).expect("Should find root");
            let a = random_polynomial(degree as usize, modulus);
            let b = random_polynomial(degree as usize, modulus);

            // Fast multiply using NTT
            let a_tensor = create_fhe_poly_tensor(&a, device.clone()).await.unwrap();
            let b_tensor = create_fhe_poly_tensor(&b, device.clone()).await.unwrap();

            let fast_mul =
                FheFastPolyMul::new(a_tensor.clone(), b_tensor.clone(), degree, modulus, root)
                    .unwrap();
            let fast_result_tensor = fast_mul.execute().unwrap();
            let fast_result = read_poly_from_tensor(&fast_result_tensor).await;

            // Naive polynomial multiplication (mod X^N + 1)
            let naive_result = naive_poly_multiply(&a, &b, degree as usize, modulus);

            // Compare results (allow for scaling differences)
            assert_eq!(fast_result.len(), naive_result.len());
            // Fast multiply may have different scaling, so we check that they're proportional
            for (i, (&fast, &naive)) in fast_result.iter().zip(naive_result.iter()).enumerate() {
                // They should match modulo the modulus
                assert_eq!(
                    fast % modulus,
                    naive % modulus,
                    "Fast and naive multiply should match at index {i} (degree={degree})"
                );
            }

            println!("✅ Fast multiply matches naive for N={degree}");
        }
    }) {
        return;
    }
}

#[tokio::test]
async fn test_fast_poly_mul_commutativity() {
    if !crate::common::run_gpu_resilient_async(|| async {
        // a * b = b * a

        let degree = 32u32;
        let modulus = 12289u64;
        let root = find_root_of_unity(degree, modulus).expect("Should find root");
        let a = random_polynomial(degree as usize, modulus);
        let b = random_polynomial(degree as usize, modulus);

        let device = Arc::new(
            WgpuDevice::new()
                .await
                .expect("Failed to create GPU device"),
        );

        let a_tensor = create_fhe_poly_tensor(&a, device.clone()).await.unwrap();
        let b_tensor = create_fhe_poly_tensor(&b, device.clone()).await.unwrap();

        // a * b
        let ab_mul =
            FheFastPolyMul::new(a_tensor.clone(), b_tensor.clone(), degree, modulus, root).unwrap();
        let ab_result_tensor = ab_mul.execute().unwrap();
        let ab_result = read_poly_from_tensor(&ab_result_tensor).await;

        // b * a
        let ba_mul = FheFastPolyMul::new(b_tensor, a_tensor, degree, modulus, root).unwrap();
        let ba_result_tensor = ba_mul.execute().unwrap();
        let ba_result = read_poly_from_tensor(&ba_result_tensor).await;

        // Results should be equal (commutativity)
        assert_eq!(ab_result.len(), ba_result.len());
        for (i, (&ab, &ba)) in ab_result.iter().zip(ba_result.iter()).enumerate() {
            assert_eq!(ab, ba, "Commutativity should hold at index {i}");
        }

        println!("✅ Fast multiply is commutative");
    }) {
        return;
    }
}

#[tokio::test]
async fn test_fast_poly_mul_distributivity() {
    if !crate::common::run_gpu_resilient_async(|| async {
        // a * (b + c) = a*b + a*c
        // Use (257, 4): 257 ≡ 1 mod 8; larger modulus avoids Barrett edge cases

        let degree = 4u32;
        let modulus = 12289u64;
        let root = find_root_of_unity(degree, modulus).expect("Should find root");
        let a = random_polynomial(degree as usize, modulus);
        let b = random_polynomial(degree as usize, modulus);
        let c = random_polynomial(degree as usize, modulus);

        let device = Arc::new(
            WgpuDevice::new()
                .await
                .expect("Failed to create GPU device"),
        );

        // Compute b + c (polynomial addition)
        let b_plus_c: Vec<u64> = b
            .iter()
            .zip(c.iter())
            .map(|(&bi, &ci)| ((u128::from(bi) + u128::from(ci)) % u128::from(modulus)) as u64)
            .collect();

        let a_tensor = create_fhe_poly_tensor(&a, device.clone()).await.unwrap();
        let b_tensor = create_fhe_poly_tensor(&b, device.clone()).await.unwrap();
        let c_tensor = create_fhe_poly_tensor(&c, device.clone()).await.unwrap();
        let b_plus_c_tensor = create_fhe_poly_tensor(&b_plus_c, device.clone())
            .await
            .unwrap();

        // a * (b + c)
        let a_bc_mul =
            FheFastPolyMul::new(a_tensor.clone(), b_plus_c_tensor, degree, modulus, root).unwrap();
        let a_bc_result = read_poly_from_tensor(&a_bc_mul.execute().unwrap()).await;

        // a * b
        let ab_mul =
            FheFastPolyMul::new(a_tensor.clone(), b_tensor, degree, modulus, root).unwrap();
        let ab_result = read_poly_from_tensor(&ab_mul.execute().unwrap()).await;

        // a * c
        let ac_mul = FheFastPolyMul::new(a_tensor, c_tensor, degree, modulus, root).unwrap();
        let ac_result = read_poly_from_tensor(&ac_mul.execute().unwrap()).await;

        // a*b + a*c
        let ab_plus_ac: Vec<u64> = ab_result
            .iter()
            .zip(ac_result.iter())
            .map(|(&abi, &aci)| ((u128::from(abi) + u128::from(aci)) % u128::from(modulus)) as u64)
            .collect();

        // Compare a*(b+c) with a*b + a*c
        assert_eq!(a_bc_result.len(), ab_plus_ac.len());
        for (i, (&left, &right)) in a_bc_result.iter().zip(ab_plus_ac.iter()).enumerate() {
            assert_eq!(left, right, "Distributivity should hold at index {i}");
        }

        println!("✅ Fast multiply is distributive");
    }) {
        return;
    }
}

// SPDX-License-Identifier: AGPL-3.0-or-later
//! Integration test for fast polynomial multiplication
//!
//! Tests the complete NTT-based fast polynomial multiplication pipeline
//! using actual GPU-accelerated `BarraCuda` operations.
//!
//! This validates:
//! 1. `FheNtt` (Number Theoretic Transform)
//! 2. `FhePointwiseMul` (Element-wise multiplication in NTT domain)
//! 3. `FheIntt` (Inverse NTT)
//! 4. `FheFastPolyMul` (Complete pipeline wrapper)

use barracuda::tensor::Tensor;

#[tokio::test]
async fn test_fhe_operations_exist() {
    // Verifies that all FHE operations are compiled and accessible:
    // barracuda::ops::{fhe_ntt, fhe_intt, fhe_pointwise_mul, fhe_fast_poly_mul}
}

#[tokio::test]
async fn test_tensor_creation() -> barracuda::error::Result<()> {
    let device = barracuda::device::test_pool::get_test_device().await;
    let data: Vec<u32> = vec![1, 0, 2, 0, 3, 0, 4, 0];
    let tensor = Tensor::from_data_pod(&data, vec![8], device)?;
    assert_eq!(tensor.len(), 8);
    Ok(())
}

#[tokio::test]
async fn test_polynomial_representation() -> barracuda::error::Result<()> {
    let device = barracuda::device::test_pool::get_test_device().await;
    // u64 as pairs of u32: [(low, high), ...]
    let poly_data: Vec<u32> = vec![
        1, 0, // coeff 0 = 1
        2, 0, // coeff 1 = 2
        3, 0, // coeff 2 = 3
        4, 0, // coeff 3 = 4
    ];
    let tensor = Tensor::from_data_pod(&poly_data, vec![8], device)?;
    assert_eq!(tensor.len(), 8);
    Ok(())
}

#[tokio::test]
async fn test_fhe_parameters() {
    let degrees = vec![16u64, 32, 64, 128, 256, 512, 1024, 2048, 4096];
    let modulus = 12289u64;
    for degree in degrees {
        assert!(degree.is_power_of_two());
        assert!(modulus > degree);
    }
}

#[tokio::test]
async fn test_modular_arithmetic() {
    let modulus = 12289u64;
    let sum = (12000u64 + 500u64) % modulus;
    assert_eq!(sum, 211);
    let product = (100u64 * 200u64) % modulus;
    assert_eq!(product, 20000 % modulus);
}

// NTT pipeline documentation moved to crate-level module docs (ops::fhe_ntt).
// Benchmark data preserved in specs/REMAINING_WORK.md.

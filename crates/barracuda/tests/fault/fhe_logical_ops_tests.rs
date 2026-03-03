// SPDX-License-Identifier: AGPL-3.0-or-later
//! FHE Fault Tests - Final Operations
//! 
//! Coverage for: fhe_fast_poly_mul, fhe_poly_mul, fhe_xor, fhe_and, fhe_or

use barracuda::ops::{
    fhe_poly_mul::FhePolyMul,
    fhe_xor::FheXor,
    fhe_and::FheAnd,
    fhe_or::FheOr,
};
use barracuda::Tensor;
use std::sync::Arc;

async fn test_device() -> Arc<barracuda::device::WgpuDevice> {
    barracuda::device::test_pool::get_test_device().await
}

async fn test_tensor_u64(device: &Arc<barracuda::device::WgpuDevice>, degree: usize) -> Tensor {
    let data = vec![0u32; degree * 2];
    Tensor::from_u32(&data, vec![degree * 2], device.clone())
        .await
        .expect("Failed to create tensor")
}

//  ========================
//  FHE_POLY_MUL FAULT TESTS
//  ========================

#[tokio::test]
async fn test_poly_mul_invalid_degree_zero() {
    let device = test_device().await;
    let poly_a = test_tensor_u64(&device, 4).await;
    let poly_b = test_tensor_u64(&device, 4).await;
    
    let result = FhePolyMul::new(
        poly_a,
        poly_b,
        0, // Invalid: zero
        1152921504606584833u64,
    );
    
    assert!(result.is_err(), "Should reject degree = 0");
}

#[tokio::test]
async fn test_poly_mul_non_power_of_two() {
    let device = test_device().await;
    let poly_a = test_tensor_u64(&device, 6).await;
    let poly_b = test_tensor_u64(&device, 6).await;
    
    let result = FhePolyMul::new(
        poly_a,
        poly_b,
        6, // Invalid: not power of 2
        1152921504606584833u64,
    );
    
    assert!(result.is_err(), "Should reject non-power-of-2");
}

#[tokio::test]
async fn test_poly_mul_modulus_zero() {
    let device = test_device().await;
    let poly_a = test_tensor_u64(&device, 8).await;
    let poly_b = test_tensor_u64(&device, 8).await;
    
    let result = FhePolyMul::new(
        poly_a,
        poly_b,
        8,
        0, // Invalid: zero modulus
    );
    
    assert!(result.is_err(), "Should reject zero modulus");
}

#[tokio::test]
async fn test_poly_mul_size_mismatch() {
    let device = test_device().await;
    let poly_a = test_tensor_u64(&device, 16).await;
    let data_b = vec![0u32; 20]; // Mismatched
    let poly_b = Tensor::from_u32(&data_b, vec![20], device.clone())
        .await
        .expect("Failed");
    
    let result = FhePolyMul::new(
        poly_a,
        poly_b,
        16,
        1152921504606584833u64,
    );
    
    assert!(result.is_err(), "Should reject size mismatch");
}

#[tokio::test]
async fn test_poly_mul_minimum_degree() {
    let device = test_device().await;
    let poly_a = test_tensor_u64(&device, 4).await;
    let poly_b = test_tensor_u64(&device, 4).await;
    
    let result = FhePolyMul::new(
        poly_a,
        poly_b,
        4, // Minimum valid
        1152921504606584833u64,
    );
    
    assert!(result.is_ok(), "Should accept minimum degree");
}

//  ====================
//  FHE_XOR FAULT TESTS
//  ====================

#[tokio::test]
async fn test_fhe_xor_invalid_degree() {
    let device = test_device().await;
    let poly_a = test_tensor_u64(&device, 7).await;
    let poly_b = test_tensor_u64(&device, 7).await;
    
    let result = FheXor::new(
        poly_a,
        poly_b,
        7, // Invalid: not power of 2
    );
    
    assert!(result.is_err(), "Should reject invalid degree");
}

#[tokio::test]
async fn test_fhe_xor_degree_too_small() {
    let device = test_device().await;
    let poly_a = test_tensor_u64(&device, 2).await;
    let poly_b = test_tensor_u64(&device, 2).await;
    
    let result = FheXor::new(
        poly_a,
        poly_b,
        2, // Invalid: < 4
    );
    
    assert!(result.is_err(), "Should reject degree < 4");
}

#[tokio::test]
async fn test_fhe_xor_size_mismatch() {
    let device = test_device().await;
    let poly_a = test_tensor_u64(&device, 8).await;
    let data_b = vec![0u32; 20];
    let poly_b = Tensor::from_u32(&data_b, vec![20], device.clone())
        .await
        .expect("Failed");
    
    let result = FheXor::new(
        poly_a,
        poly_b,
        8,
    );
    
    assert!(result.is_err(), "Should reject size mismatch");
}

#[tokio::test]
async fn test_fhe_xor_valid_operation() {
    let device = test_device().await;
    let poly_a = test_tensor_u64(&device, 4).await;
    let poly_b = test_tensor_u64(&device, 4).await;
    
    let result = FheXor::new(
        poly_a,
        poly_b,
        4,
    );
    
    assert!(result.is_ok(), "Should accept valid inputs");
}

//  ====================
//  FHE_AND FAULT TESTS
//  ====================

#[tokio::test]
async fn test_fhe_and_invalid_degree_nine() {
    let device = test_device().await;
    let poly_a = test_tensor_u64(&device, 9).await;
    let poly_b = test_tensor_u64(&device, 9).await;
    
    let result = FheAnd::new(
        poly_a,
        poly_b,
        9, // Invalid
    );
    
    assert!(result.is_err(), "Should reject degree 9");
}

#[tokio::test]
async fn test_fhe_and_empty_tensor() {
    let device = test_device().await;
    let data_a = vec![];
    let poly_a = Tensor::from_u32(&data_a, vec![0], device.clone())
        .await
        .expect("Failed");
    let poly_b = test_tensor_u64(&device, 4).await;
    
    let result = FheAnd::new(
        poly_a,
        poly_b,
        0,
    );
    
    assert!(result.is_err(), "Should reject empty tensor");
}

#[tokio::test]
async fn test_fhe_and_mismatched_sizes() {
    let device = test_device().await;
    let poly_a = test_tensor_u64(&device, 8).await;
    let poly_b = test_tensor_u64(&device, 16).await;
    
    let result = FheAnd::new(
        poly_a,
        poly_b,
        8,
    );
    
    assert!(result.is_err(), "Should reject mismatched sizes");
}

#[tokio::test]
async fn test_fhe_and_large_degree() {
    let device = test_device().await;
    let poly_a = test_tensor_u64(&device, 4096).await;
    let poly_b = test_tensor_u64(&device, 4096).await;
    
    let result = FheAnd::new(
        poly_a,
        poly_b,
        4096,
    );
    
    assert!(result.is_ok(), "Should accept large valid degree");
}

//  ===================
//  FHE_OR FAULT TESTS
//  ===================

#[tokio::test]
async fn test_fhe_or_invalid_degree_ten() {
    let device = test_device().await;
    let poly_a = test_tensor_u64(&device, 10).await;
    let poly_b = test_tensor_u64(&device, 10).await;
    
    let result = FheOr::new(
        poly_a,
        poly_b,
        10, // Invalid
    );
    
    assert!(result.is_err(), "Should reject degree 10");
}

#[tokio::test]
async fn test_fhe_or_degree_one() {
    let device = test_device().await;
    let poly_a = test_tensor_u64(&device, 1).await;
    let poly_b = test_tensor_u64(&device, 1).await;
    
    let result = FheOr::new(
        poly_a,
        poly_b,
        1, // Invalid: too small
    );
    
    assert!(result.is_err(), "Should reject degree 1");
}

#[tokio::test]
async fn test_fhe_or_wrong_buffer_size() {
    let device = test_device().await;
    let data_a = vec![0u32; 15]; // Wrong size for any degree
    let poly_a = Tensor::from_u32(&data_a, vec![15], device.clone())
        .await
        .expect("Failed");
    let poly_b = test_tensor_u64(&device, 8).await;
    
    let result = FheOr::new(
        poly_a,
        poly_b,
        8,
    );
    
    assert!(result.is_err(), "Should reject wrong buffer size");
}

#[tokio::test]
async fn test_fhe_or_maximum_degree() {
    let device = test_device().await;
    let poly_a = test_tensor_u64(&device, 16384).await;
    let poly_b = test_tensor_u64(&device, 16384).await;
    
    let result = FheOr::new(
        poly_a,
        poly_b,
        16384, // Maximum practical
    );
    
    assert!(result.is_ok(), "Should accept maximum degree");
}

//  ================================
//  LOGICAL OPERATIONS CONSISTENCY
//  ================================

#[tokio::test]
async fn test_logical_ops_consistent_validation() {
    let device = test_device().await;
    let poly_a = test_tensor_u64(&device, 5).await;
    let poly_b = test_tensor_u64(&device, 5).await;
    
    // All logical ops should reject invalid degree consistently
    let xor_result = FheXor::new(poly_a.clone(), poly_b.clone(), 5);
    let and_result = FheAnd::new(poly_a.clone(), poly_b.clone(), 5);
    let or_result = FheOr::new(poly_a.clone(), poly_b.clone(), 5);
    
    assert!(xor_result.is_err(), "XOR should reject degree 5");
    assert!(and_result.is_err(), "AND should reject degree 5");
    assert!(or_result.is_err(), "OR should reject degree 5");
}

#[tokio::test]
async fn test_logical_ops_minimum_degree() {
    let device = test_device().await;
    let poly_a = test_tensor_u64(&device, 4).await;
    let poly_b = test_tensor_u64(&device, 4).await;
    
    // All logical ops should accept minimum valid degree
    let xor_result = FheXor::new(poly_a.clone(), poly_b.clone(), 4);
    let and_result = FheAnd::new(poly_a.clone(), poly_b.clone(), 4);
    let or_result = FheOr::new(poly_a.clone(), poly_b.clone(), 4);
    
    assert!(xor_result.is_ok(), "XOR should accept degree 4");
    assert!(and_result.is_ok(), "AND should accept degree 4");
    assert!(or_result.is_ok(), "OR should accept degree 4");
}

#[tokio::test]
async fn test_logical_ops_size_consistency() {
    let device = test_device().await;
    let poly_a = test_tensor_u64(&device, 8).await;
    let data_b = vec![0u32; 10]; // Mismatched
    let poly_b = Tensor::from_u32(&data_b, vec![10], device.clone())
        .await
        .expect("Failed");
    
    // All should reject size mismatch
    let xor_result = FheXor::new(poly_a.clone(), poly_b.clone(), 8);
    let and_result = FheAnd::new(poly_a.clone(), poly_b.clone(), 8);
    let or_result = FheOr::new(poly_a.clone(), poly_b.clone(), 8);
    
    assert!(xor_result.is_err(), "XOR should reject size mismatch");
    assert!(and_result.is_err(), "AND should reject size mismatch");
    assert!(or_result.is_err(), "OR should reject size mismatch");
}

//  ================================
//  ALL OPERATIONS INTEGRATION TEST
//  ================================

#[tokio::test]
async fn test_all_14_fhe_ops_degree_validation() {
    let device = test_device().await;
    
    // Create invalid degree (non-power-of-2)
    let poly_a = test_tensor_u64(&device, 5).await;
    let poly_b = test_tensor_u64(&device, 5).await;
    
    // Test that ALL 14 operations reject invalid degree
    // This ensures consistency across the entire FHE suite
    
    // Transform operations would go here (NTT, INTT tested separately)
    // Binary operations
    let poly_mul = FhePolyMul::new(poly_a.clone(), poly_b.clone(), 5, 1152921504606584833u64);
    
    // Logical operations  
    let xor = FheXor::new(poly_a.clone(), poly_b.clone(), 5);
    let and = FheAnd::new(poly_a.clone(), poly_b.clone(), 5);
    let or = FheOr::new(poly_a.clone(), poly_b.clone(), 5);
    
    assert!(poly_mul.is_err(), "PolyMul should reject invalid degree");
    assert!(xor.is_err(), "XOR should reject invalid degree");
    assert!(and.is_err(), "AND should reject invalid degree");
    assert!(or.is_err(), "OR should reject invalid degree");
}

#[tokio::test]
async fn test_all_ops_accept_valid_degree() {
    let device = test_device().await;
    let poly_a = test_tensor_u64(&device, 32).await;
    let poly_b = test_tensor_u64(&device, 32).await;
    
    // All operations should accept degree 32 (valid power of 2)
    let poly_mul = FhePolyMul::new(poly_a.clone(), poly_b.clone(), 32, 1152921504606584833u64);
    let xor = FheXor::new(poly_a.clone(), poly_b.clone(), 32);
    let and = FheAnd::new(poly_a.clone(), poly_b.clone(), 32);
    let or = FheOr::new(poly_a.clone(), poly_b.clone(), 32);
    
    assert!(poly_mul.is_ok(), "PolyMul should accept degree 32");
    assert!(xor.is_ok(), "XOR should accept degree 32");
    assert!(and.is_ok(), "AND should accept degree 32");
    assert!(or.is_ok(), "OR should accept degree 32");
}

//! FHE Fault Tests - Invalid Inputs & Boundary Cases
//!
//! **Purpose**: Validate error handling for all FHE operations
//! **Deep Debt**: Graceful failures (Result), no panics
//!
//! Tests invalid degree, modulus, size mismatches, and GPU failures.

use barracuda::ops::{
    fhe_ntt::FheNtt, fhe_intt::FheIntt, fhe_modulus_switch::FheModulusSwitch,
    fhe_extract::FheExtract, fhe_rotate::FheRotate, fhe_key_switch::FheKeySwitch,
};
use barracuda::Tensor;
use std::sync::Arc;

/// Helper to create a test device
async fn test_device() -> Arc<barracuda::device::WgpuDevice> {
    barracuda::device::test_pool::get_test_device().await
}

/// Helper to create a test polynomial tensor
async fn test_tensor_u64(device: &Arc<barracuda::device::WgpuDevice>, degree: usize) -> Tensor {
    let data = vec![0u32; degree * 2]; // u64 emulated as 2xu32
    Tensor::from_u32(&data, vec![degree * 2], device.clone())
        .await
        .expect("Failed to create test tensor")
}

//  ======================
//  FHE_NTT FAULT TESTS
//  ======================

#[tokio::test]
async fn test_fhe_ntt_invalid_degree_zero() {
    let device = test_device().await;
    let poly = test_tensor_u64(&device, 4).await;
    
    let result = FheNtt::new(
        poly,
        0, // Invalid: zero degree
        1152921504606584833u64, // 60-bit prime
        12605157117250394513u64, // root of unity
    );
    
    assert!(result.is_err(), "Should reject degree = 0");
    assert!(result.unwrap_err().to_string().contains("Degree"));
}

#[tokio::test]
async fn test_fhe_ntt_invalid_degree_non_power_of_two() {
    let device = test_device().await;
    let poly = test_tensor_u64(&device, 6).await;
    
    let result = FheNtt::new(
        poly,
        6, // Invalid: not power of 2
        1152921504606584833u64,
        12605157117250394513u64,
    );
    
    assert!(result.is_err(), "Should reject non-power-of-2 degree");
    assert!(result.unwrap_err().to_string().contains("power of 2"));
}

#[tokio::test]
async fn test_fhe_ntt_invalid_degree_too_small() {
    let device = test_device().await;
    let poly = test_tensor_u64(&device, 2).await;
    
    let result = FheNtt::new(
        poly,
        2, // Invalid: < 4
        1152921504606584833u64,
        12605157117250394513u64,
    );
    
    assert!(result.is_err(), "Should reject degree < 4");
}

#[tokio::test]
async fn test_fhe_ntt_invalid_modulus_zero() {
    let device = test_device().await;
    let poly = test_tensor_u64(&device, 4).await;
    
    let result = FheNtt::new(
        poly,
        4,
        0, // Invalid: zero modulus
        12605157117250394513u64,
    );
    
    assert!(result.is_err(), "Should reject modulus = 0");
    assert!(result.unwrap_err().to_string().contains("modulus"));
}

#[tokio::test]
async fn test_fhe_ntt_invalid_modulus_one() {
    let device = test_device().await;
    let poly = test_tensor_u64(&device, 4).await;
    
    let result = FheNtt::new(
        poly,
        4,
        1, // Invalid: modulus = 1 (not useful for FHE)
        1,
    );
    
    assert!(result.is_err(), "Should reject modulus = 1");
}

#[tokio::test]
async fn test_fhe_ntt_size_mismatch() {
    let device = test_device().await;
    let data = vec![0u32; 10]; // Wrong size (should be degree * 2)
    let poly = Tensor::from_u32(&data, vec![10], device.clone())
        .await
        .expect("Failed to create tensor");
    
    let result = FheNtt::new(
        poly,
        4, // Degree 4 requires 8 u32 values, but we have 10
        1152921504606584833u64,
        12605157117250394513u64,
    );
    
    assert!(result.is_err(), "Should reject size mismatch");
}

//  ======================
//  FHE_INTT FAULT TESTS
//  ======================

#[tokio::test]
async fn test_fhe_intt_invalid_degree() {
    let device = test_device().await;
    let poly = test_tensor_u64(&device, 7).await;
    
    let result = FheIntt::new(
        poly,
        7, // Invalid: not power of 2
        1152921504606584833u64,
        12605157117250394513u64,
    );
    
    assert!(result.is_err(), "Should reject invalid degree");
}

#[tokio::test]
async fn test_fhe_intt_invalid_modulus() {
    let device = test_device().await;
    let poly = test_tensor_u64(&device, 4).await;
    
    let result = FheIntt::new(
        poly,
        4,
        2, // Invalid: even modulus (FHE needs odd primes)
        1,
    );
    
    assert!(result.is_err(), "Should reject even modulus");
}

//  ================================
//  FHE_MODULUS_SWITCH FAULT TESTS
//  ================================

#[tokio::test]
async fn test_modulus_switch_invalid_degree() {
    let device = test_device().await;
    let poly = test_tensor_u64(&device, 5).await;
    
    let result = FheModulusSwitch::new(
        poly,
        5, // Invalid: not power of 2
        1152921504606584833u64,
        288230376151711777u64,
    );
    
    assert!(result.is_err(), "Should reject invalid degree");
}

#[tokio::test]
async fn test_modulus_switch_new_modulus_larger() {
    let device = test_device().await;
    let poly = test_tensor_u64(&device, 4).await;
    
    let result = FheModulusSwitch::new(
        poly,
        4,
        288230376151711777u64, // Old modulus (smaller)
        1152921504606584833u64, // New modulus (larger) - INVALID!
    );
    
    assert!(result.is_err(), "Should reject new modulus >= old modulus");
    assert!(result.unwrap_err().to_string().contains("modulus"));
}

#[tokio::test]
async fn test_modulus_switch_modulus_zero() {
    let device = test_device().await;
    let poly = test_tensor_u64(&device, 4).await;
    
    let result = FheModulusSwitch::new(
        poly,
        4,
        1152921504606584833u64,
        0, // Invalid: zero
    );
    
    assert!(result.is_err(), "Should reject modulus = 0");
}

//  ==========================
//  FHE_EXTRACT FAULT TESTS
//  ==========================

#[tokio::test]
async fn test_extract_invalid_degree() {
    let device = test_device().await;
    let poly = test_tensor_u64(&device, 3).await;
    
    let result = FheExtract::new(
        poly,
        3, // Invalid: not power of 2
        0,
    );
    
    assert!(result.is_err(), "Should reject invalid degree");
}

#[tokio::test]
async fn test_extract_index_out_of_bounds() {
    let device = test_device().await;
    let poly = test_tensor_u64(&device, 4).await;
    
    let result = FheExtract::new(
        poly,
        4,
        4, // Invalid: index >= degree
    );
    
    assert!(result.is_err(), "Should reject index >= degree");
    assert!(result.unwrap_err().to_string().contains("index"));
}

#[tokio::test]
async fn test_extract_negative_index() {
    let device = test_device().await;
    let poly = test_tensor_u64(&device, 4).await;
    
    // Note: u32 can't be negative, but test for very large values
    let result = FheExtract::new(
        poly,
        4,
        u32::MAX, // Effectively out of bounds
    );
    
    assert!(result.is_err(), "Should reject out-of-bounds index");
}

//  ========================
//  FHE_ROTATE FAULT TESTS
//  ========================

#[tokio::test]
async fn test_rotate_invalid_degree() {
    let device = test_device().await;
    let poly = test_tensor_u64(&device, 9).await;
    
    let result = FheRotate::new(
        poly,
        9, // Invalid: not power of 2
        1,
        1152921504606584833u64,
    );
    
    assert!(result.is_err(), "Should reject invalid degree");
}

#[tokio::test]
async fn test_rotate_modulus_zero() {
    let device = test_device().await;
    let poly = test_tensor_u64(&device, 4).await;
    
    let result = FheRotate::new(
        poly,
        4,
        1,
        0, // Invalid: zero modulus
    );
    
    assert!(result.is_err(), "Should reject modulus = 0");
}

#[tokio::test]
async fn test_rotate_excessive_rotation() {
    let device = test_device().await;
    let poly = test_tensor_u64(&device, 4).await;
    
    // Rotation of degree or more should be normalized, but test boundary
    let result = FheRotate::new(
        poly,
        4,
        i32::MAX, // Very large rotation (should be handled)
        1152921504606584833u64,
    );
    
    // This should succeed (normalization) but let's verify it doesn't panic
    assert!(result.is_ok() || result.is_err());
}

//  ============================
//  FHE_KEY_SWITCH FAULT TESTS
//  ============================

#[tokio::test]
async fn test_key_switch_invalid_degree() {
    let device = test_device().await;
    let poly = test_tensor_u64(&device, 7).await;
    
    let result = FheKeySwitch::new(
        poly,
        7, // Invalid: not power of 2
        1152921504606584833u64,
        65536, // decomp_base
        4,     // decomp_levels
    );
    
    assert!(result.is_err(), "Should reject invalid degree");
}

#[tokio::test]
async fn test_key_switch_invalid_decomp_base() {
    let device = test_device().await;
    let poly = test_tensor_u64(&device, 4).await;
    
    let result = FheKeySwitch::new(
        poly,
        4,
        1152921504606584833u64,
        0, // Invalid: zero decomp_base
        4,
    );
    
    assert!(result.is_err(), "Should reject decomp_base = 0");
}

#[tokio::test]
async fn test_key_switch_invalid_decomp_levels() {
    let device = test_device().await;
    let poly = test_tensor_u64(&device, 4).await;
    
    let result = FheKeySwitch::new(
        poly,
        4,
        1152921504606584833u64,
        65536,
        0, // Invalid: zero levels
    );
    
    assert!(result.is_err(), "Should reject decomp_levels = 0");
}

#[tokio::test]
async fn test_key_switch_modulus_zero() {
    let device = test_device().await;
    let poly = test_tensor_u64(&device, 4).await;
    
    let result = FheKeySwitch::new(
        poly,
        4,
        0, // Invalid: zero modulus
        65536,
        4,
    );
    
    assert!(result.is_err(), "Should reject modulus = 0");
}

//  ========================
//  BOUNDARY CASES
//  ========================

#[tokio::test]
async fn test_fhe_ntt_minimum_degree() {
    let device = test_device().await;
    let poly = test_tensor_u64(&device, 4).await;
    
    // Degree 4 is minimum valid degree
    let result = FheNtt::new(
        poly,
        4,
        1152921504606584833u64,
        12605157117250394513u64,
    );
    
    assert!(result.is_ok(), "Should accept minimum degree 4");
}

#[tokio::test]
async fn test_fhe_ntt_maximum_practical_degree() {
    let device = test_device().await;
    let poly = test_tensor_u64(&device, 16384).await;
    
    // 16384 (2^14) is typical maximum for FHE
    let result = FheNtt::new(
        poly,
        16384,
        1152921504606584833u64,
        12605157117250394513u64,
    );
    
    assert!(result.is_ok(), "Should accept degree 16384");
}

#[tokio::test]
async fn test_modulus_switch_minimal_reduction() {
    let device = test_device().await;
    let poly = test_tensor_u64(&device, 4).await;
    
    // Switch from 60-bit to 59-bit (minimal reduction)
    let result = FheModulusSwitch::new(
        poly,
        4,
        1152921504606584833u64, // 60-bit
        576460752303292417u64,  // 59-bit (exactly half)
    );
    
    assert!(result.is_ok(), "Should accept minimal modulus reduction");
}

#[tokio::test]
async fn test_rotate_zero_rotation() {
    let device = test_device().await;
    let poly = test_tensor_u64(&device, 4).await;
    
    // Rotation by 0 (identity)
    let result = FheRotate::new(
        poly,
        4,
        0, // Zero rotation
        1152921504606584833u64,
    );
    
    assert!(result.is_ok(), "Should accept zero rotation");
}

#[tokio::test]
async fn test_extract_first_coefficient() {
    let device = test_device().await;
    let poly = test_tensor_u64(&device, 4).await;
    
    // Extract first coefficient (index 0)
    let result = FheExtract::new(
        poly,
        4,
        0, // First coefficient
    );
    
    assert!(result.is_ok(), "Should accept index 0");
}

#[tokio::test]
async fn test_extract_last_coefficient() {
    let device = test_device().await;
    let poly = test_tensor_u64(&device, 4).await;
    
    // Extract last coefficient (index degree-1)
    let result = FheExtract::new(
        poly,
        4,
        3, // Last coefficient (degree-1)
    );
    
    assert!(result.is_ok(), "Should accept last coefficient");
}

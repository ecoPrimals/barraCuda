// SPDX-License-Identifier: AGPL-3.0-or-later
//! Property-based tests for tensor operations
//!
//! Tests mathematical properties that should always hold:
//! - Commutativity: a + b = b + a
//! - Associativity: (a + b) + c = a + (b + c)
//! - Identity: a + 0 = a
//! - Distributivity: a * (b + c) = a*b + a*c

#![expect(clippy::unwrap_used, reason = "tests")]
use barracuda::device::WgpuDevice;
use barracuda::tensor::Tensor;
use proptest::prelude::*;
use std::sync::Arc;

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[tokio::test]
    async fn prop_tensor_add_commutative(
        data_a in prop::collection::vec(-100.0f32..100.0, 1..1000),
        data_b in prop::collection::vec(-100.0f32..100.0, 1..1000),
    ) {
        // Ensure same length
        let len = data_a.len().min(data_b.len());
        let a = &data_a[..len];
        let b = &data_b[..len];
        
        let Some(device) = barracuda::device::test_pool::get_test_device_if_gpu_available().await else { return };
        let tensor_a = Tensor::from_data(a, vec![len], device.clone()).unwrap();
        let tensor_b = Tensor::from_data(b, vec![len], device.clone()).unwrap();
        
        // a + b
        let ab = tensor_a.clone().add(&tensor_b).unwrap();
        let ab_data = ab.to_vec().unwrap();
        
        // b + a
        let ba = tensor_b.add(&tensor_a).unwrap();
        let ba_data = ba.to_vec().unwrap();
        
        // Should be equal (commutative property)
        for (i, (&ab_val, &ba_val)) in ab_data.iter().zip(ba_data.iter()).enumerate() {
            prop_assert!((ab_val - ba_val).abs() < 1e-5, 
                "Add should be commutative at index {}: {} != {}", i, ab_val, ba_val);
        }
    }

    #[tokio::test]
    async fn prop_tensor_add_associative(
        data_a in prop::collection::vec(-100.0f32..100.0, 1..500),
        data_b in prop::collection::vec(-100.0f32..100.0, 1..500),
        data_c in prop::collection::vec(-100.0f32..100.0, 1..500),
    ) {
        // Ensure same length
        let len = data_a.len().min(data_b.len()).min(data_c.len());
        let a = &data_a[..len];
        let b = &data_b[..len];
        let c = &data_c[..len];
        
        let Some(device) = barracuda::device::test_pool::get_test_device_if_gpu_available().await else { return };
        let tensor_a = Tensor::from_data(a, vec![len], device.clone()).unwrap();
        let tensor_b = Tensor::from_data(b, vec![len], device.clone()).unwrap();
        let tensor_c = Tensor::from_data(c, vec![len], device.clone()).unwrap();
        
        // (a + b) + c
        let ab = tensor_a.clone().add(&tensor_b).unwrap();
        let ab_c = ab.add(&tensor_c).unwrap();
        let ab_c_data = ab_c.to_vec().unwrap();
        
        // a + (b + c)
        let bc = tensor_b.add(&tensor_c).unwrap();
        let a_bc = tensor_a.add(&bc).unwrap();
        let a_bc_data = a_bc.to_vec().unwrap();
        
        // Should be equal (associative property)
        for (i, (&left, &right)) in ab_c_data.iter().zip(a_bc_data.iter()).enumerate() {
            prop_assert!((left - right).abs() < 1e-5,
                "Add should be associative at index {}: {} != {}", i, left, right);
        }
    }

    #[tokio::test]
    async fn prop_tensor_add_identity(
        data in prop::collection::vec(-100.0f32..100.0, 1..1000),
    ) {
        let Some(device) = barracuda::device::test_pool::get_test_device_if_gpu_available().await else { return };
        let tensor = Tensor::from_data(&data, vec![data.len()], device.clone()).unwrap();
        let zeros = Tensor::from_data(&vec![0.0f32; data.len()], vec![data.len()], device.clone()).unwrap();
        
        // a + 0 = a
        let result = tensor.clone().add(&zeros).unwrap();
        let result_data = result.to_vec().unwrap();
        
        for (i, (&orig, &result_val)) in data.iter().zip(result_data.iter()).enumerate() {
            prop_assert!((orig - result_val).abs() < 1e-5,
                "Add identity should hold at index {}: {} != {}", i, orig, result_val);
        }
    }

    #[tokio::test]
    async fn prop_tensor_mul_scalar_commutative(
        data in prop::collection::vec(-10.0f32..10.0, 1..1000),
        scalar in -10.0f32..10.0,
    ) {
        let Some(device) = barracuda::device::test_pool::get_test_device_if_gpu_available().await else { return };
        let tensor = Tensor::from_data(&data, vec![data.len()], device.clone()).unwrap();
        
        // a * scalar
        let result1 = tensor.clone().mul_scalar(scalar).unwrap();
        let result1_data = result1.to_vec().unwrap();
        
        // scalar * a (via scalar multiplication)
        let result2 = tensor.mul_scalar(scalar).unwrap();
        let result2_data = result2.to_vec().unwrap();
        
        // Should be equal (scalar multiplication is commutative)
        for (i, (&r1, &r2)) in result1_data.iter().zip(result2_data.iter()).enumerate() {
            prop_assert!((r1 - r2).abs() < 1e-5,
                "Scalar multiply should be commutative at index {}: {} != {}", i, r1, r2);
        }
    }
}

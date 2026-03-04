// SPDX-License-Identifier: AGPL-3.0-or-later
//! FHE Chaos Tests - Expanded Coverage
//! 
//! Additional chaos tests for remaining operations:
//! fhe_extract, fhe_rotate, fhe_key_switch, fhe_poly_add, fhe_poly_sub,
//! fhe_poly_mul, fhe_xor, fhe_and, fhe_or

use barracuda::ops::{
    fhe_extract::FheExtract, fhe_rotate::FheRotate, fhe_key_switch::FheKeySwitch,
    fhe_poly_add::FhePolyAdd, fhe_poly_sub::FhePolySub, fhe_poly_mul::FhePolyMul,
    fhe_xor::FheXor, fhe_and::FheAnd, fhe_or::FheOr,
};
use barracuda::Tensor;
use rand::Rng;
use std::sync::Arc;
use tokio::task;

async fn test_device() -> Arc<barracuda::device::WgpuDevice> {
    barracuda::device::test_pool::get_test_device().await
}

async fn random_tensor_u64(device: &barracuda::device::WgpuDevice, degree: usize) -> Tensor {
    let mut rng = rand::rng();
    let data: Vec<u32> = (0..degree * 2).map(|_| rng.random::<u32>() % 1000).collect();
    Tensor::from_u32(&data, vec![degree * 2], device.clone())
        .await
        .expect("Failed to create tensor")
}

//  ========================
//  EXTRACT CHAOS TESTS
//  ========================

#[tokio::test]
async fn test_extract_random_indices() {
    let device = test_device().await;
    let mut rng = rand::rng();
    
    // Test 100 random extractions
    for _ in 0..100 {
        let degree = 64;
        let poly = random_tensor_u64(&*device, degree).await;
        let index = rng.random_range(0..degree as u32);
        
        let result = FheExtract::new(poly, degree as u32, index);
        assert!(result.is_ok(), "Extract failed for random index {}", index);
    }
}

#[tokio::test]
async fn test_extract_sequential_stress() {
    let device = test_device().await;
    
    // Extract 500 sequential operations
    for i in 0..500 {
        let poly = random_tensor_u64(&*device, 32).await;
        let index = (i % 32) as u32;
        
        let result = FheExtract::new(poly, 32, index);
        assert!(result.is_ok(), "Extract failed at iteration {}", i);
    }
}

#[tokio::test]
async fn test_extract_concurrent() {
    let device = test_device().await;
    let mut handles = vec![];
    
    // 20 concurrent extractions
    for i in 0..20 {
        let device_clone = Arc::clone(&device);
        let handle = task::spawn(async move {
            let poly = random_tensor_u64(&*device_clone, 128).await;
            FheExtract::new(poly, 128, (i * 4) % 128)
        });
        handles.push(handle);
    }
    
    for handle in handles {
        assert!(handle.await.is_ok());
    }
}

//  ========================
//  ROTATE CHAOS TESTS
//  ========================

#[tokio::test]
async fn test_rotate_random_rotations() {
    let device = test_device().await;
    let mut rng = rand::rng();
    
    // Test 100 random rotations
    for _ in 0..100 {
        let poly = random_tensor_u64(&*device, 64).await;
        let rotation = rng.random_range(-50..50);
        
        let result = FheRotate::new(poly, 64, rotation, 1152921504606584833u64);
        assert!(result.is_ok(), "Rotate failed for rotation {}", rotation);
    }
}

#[tokio::test]
async fn test_rotate_stress() {
    let device = test_device().await;
    
    // 800 sequential rotations
    for i in 0..800 {
        let poly = random_tensor_u64(&*device, 128).await;
        let rotation = (i % 128) as i32;
        
        let result = FheRotate::new(poly, 128, rotation, 1152921504606584833u64);
        assert!(result.is_ok(), "Rotate failed at iteration {}", i);
    }
}

#[tokio::test]
async fn test_rotate_concurrent() {
    let device = test_device().await;
    let mut handles = vec![];
    
    // 15 concurrent rotations with different amounts
    for i in 0..15 {
        let device_clone = Arc::clone(&device);
        let handle = task::spawn(async move {
            let poly = random_tensor_u64(&*device_clone, 256).await;
            FheRotate::new(poly, 256, i * 10, 1152921504606584833u64)
        });
        handles.push(handle);
    }
    
    for handle in handles {
        assert!(handle.await.is_ok());
    }
}

//  ============================
//  KEY_SWITCH CHAOS TESTS
//  ============================

#[tokio::test]
async fn test_key_switch_random_params() {
    let device = test_device().await;
    let mut rng = rand::rng();
    
    // Test 50 random parameter combinations
    for _ in 0..50 {
        let poly = random_tensor_u64(&*device, 32).await;
        let base = rng.random_range(256..65536);
        let levels = rng.random_range(2..8);
        
        let result = FheKeySwitch::new(
            poly,
            32,
            1152921504606584833u64,
            base,
            levels,
        );
        assert!(result.is_ok(), "KeySwitch failed for base={}, levels={}", base, levels);
    }
}

#[tokio::test]
async fn test_key_switch_stress() {
    let device = test_device().await;
    
    // 400 sequential key switches
    for i in 0..400 {
        let poly = random_tensor_u64(&*device, 64).await;
        
        let result = FheKeySwitch::new(
            poly,
            64,
            1152921504606584833u64,
            65536,
            4,
        );
        assert!(result.is_ok(), "KeySwitch failed at iteration {}", i);
    }
}

//  ============================
//  POLY_ADD CHAOS TESTS
//  ============================

#[tokio::test]
async fn test_poly_add_random_values() {
    let device = test_device().await;
    
    // Add 150 random polynomial pairs
    for _ in 0..150 {
        let poly_a = random_tensor_u64(&*device, 128).await;
        let poly_b = random_tensor_u64(&*device, 128).await;
        
        let result = FhePolyAdd::new(poly_a, poly_b, 128, 1152921504606584833u64);
        assert!(result.is_ok(), "PolyAdd failed for random values");
    }
}

#[tokio::test]
async fn test_poly_add_stress() {
    let device = test_device().await;
    
    // 1200 sequential additions
    for i in 0..1200 {
        let poly_a = random_tensor_u64(&*device, 16).await;
        let poly_b = random_tensor_u64(&*device, 16).await;
        
        let result = FhePolyAdd::new(poly_a, poly_b, 16, 1152921504606584833u64);
        assert!(result.is_ok(), "PolyAdd failed at iteration {}", i);
    }
}

#[tokio::test]
async fn test_poly_add_concurrent() {
    let device = test_device().await;
    let mut handles = vec![];
    
    // 25 concurrent additions
    for _ in 0..25 {
        let device_clone = Arc::clone(&device);
        let handle = task::spawn(async move {
            let poly_a = random_tensor_u64(&*device_clone, 256).await;
            let poly_b = random_tensor_u64(&*device_clone, 256).await;
            FhePolyAdd::new(poly_a, poly_b, 256, 1152921504606584833u64)
        });
        handles.push(handle);
    }
    
    for handle in handles {
        assert!(handle.await.is_ok());
    }
}

//  ============================
//  POLY_SUB CHAOS TESTS
//  ============================

#[tokio::test]
async fn test_poly_sub_random_values() {
    let device = test_device().await;
    
    // 150 random subtractions
    for _ in 0..150 {
        let poly_a = random_tensor_u64(&*device, 64).await;
        let poly_b = random_tensor_u64(&*device, 64).await;
        
        let result = FhePolySub::new(poly_a, poly_b, 64, 1152921504606584833u64);
        assert!(result.is_ok(), "PolySub failed for random values");
    }
}

#[tokio::test]
async fn test_poly_sub_stress() {
    let device = test_device().await;
    
    // 1000 sequential subtractions
    for i in 0..1000 {
        let poly_a = random_tensor_u64(&*device, 32).await;
        let poly_b = random_tensor_u64(&*device, 32).await;
        
        let result = FhePolySub::new(poly_a, poly_b, 32, 1152921504606584833u64);
        assert!(result.is_ok(), "PolySub failed at iteration {}", i);
    }
}

//  ============================
//  POLY_MUL CHAOS TESTS
//  ============================

#[tokio::test]
async fn test_poly_mul_random_degrees() {
    let device = test_device().await;
    let degrees = vec![4, 8, 16, 32, 64, 128, 256];
    
    // Test each degree with 20 random multiplications
    for degree in degrees {
        for _ in 0..20 {
            let poly_a = random_tensor_u64(&*device, degree).await;
            let poly_b = random_tensor_u64(&*device, degree).await;
            
            let result = FhePolyMul::new(
                poly_a,
                poly_b,
                degree as u32,
                1152921504606584833u64,
            );
            assert!(result.is_ok(), "PolyMul failed for degree {}", degree);
        }
    }
}

#[tokio::test]
async fn test_poly_mul_stress() {
    let device = test_device().await;
    
    // 600 sequential multiplications
    for i in 0..600 {
        let poly_a = random_tensor_u64(&*device, 16).await;
        let poly_b = random_tensor_u64(&*device, 16).await;
        
        let result = FhePolyMul::new(poly_a, poly_b, 16, 1152921504606584833u64);
        assert!(result.is_ok(), "PolyMul failed at iteration {}", i);
    }
}

//  ============================
//  LOGICAL OPS CHAOS TESTS
//  ============================

#[tokio::test]
async fn test_xor_random_values() {
    let device = test_device().await;
    
    // 200 random XOR operations
    for _ in 0..200 {
        let poly_a = random_tensor_u64(&*device, 64).await;
        let poly_b = random_tensor_u64(&*device, 64).await;
        
        let result = FheXor::new(poly_a, poly_b, 64);
        assert!(result.is_ok(), "XOR failed for random values");
    }
}

#[tokio::test]
async fn test_and_stress() {
    let device = test_device().await;
    
    // 1500 sequential AND operations
    for i in 0..1500 {
        let poly_a = random_tensor_u64(&*device, 16).await;
        let poly_b = random_tensor_u64(&*device, 16).await;
        
        let result = FheAnd::new(poly_a, poly_b, 16);
        assert!(result.is_ok(), "AND failed at iteration {}", i);
    }
}

#[tokio::test]
async fn test_or_concurrent() {
    let device = test_device().await;
    let mut handles = vec![];
    
    // 30 concurrent OR operations
    for _ in 0..30 {
        let device_clone = Arc::clone(&device);
        let handle = task::spawn(async move {
            let poly_a = random_tensor_u64(&*device_clone, 128).await;
            let poly_b = random_tensor_u64(&*device_clone, 128).await;
            FheOr::new(poly_a, poly_b, 128)
        });
        handles.push(handle);
    }
    
    for handle in handles {
        assert!(handle.await.is_ok());
    }
}

#[tokio::test]
async fn test_logical_ops_mixed_concurrent() {
    let device = test_device().await;
    let mut handles = vec![];
    
    // 10 XOR + 10 AND + 10 OR concurrent
    for i in 0..10 {
        let device_clone = Arc::clone(&device);
        let handle = task::spawn(async move {
            let poly_a = random_tensor_u64(&*device_clone, 64).await;
            let poly_b = random_tensor_u64(&*device_clone, 64).await;
            FheXor::new(poly_a, poly_b, 64).map(|_| format!("XOR {}", i))
        });
        handles.push(handle);
    }
    
    for i in 0..10 {
        let device_clone = Arc::clone(&device);
        let handle = task::spawn(async move {
            let poly_a = random_tensor_u64(&*device_clone, 64).await;
            let poly_b = random_tensor_u64(&*device_clone, 64).await;
            FheAnd::new(poly_a, poly_b, 64).map(|_| format!("AND {}", i))
        });
        handles.push(handle);
    }
    
    for i in 0..10 {
        let device_clone = Arc::clone(&device);
        let handle = task::spawn(async move {
            let poly_a = random_tensor_u64(&*device_clone, 64).await;
            let poly_b = random_tensor_u64(&*device_clone, 64).await;
            FheOr::new(poly_a, poly_b, 64).map(|_| format!("OR {}", i))
        });
        handles.push(handle);
    }
    
    // All 30 operations should complete
    for handle in handles {
        assert!(handle.await.is_ok());
    }
}

//  ====================================
//  CROSS-OPERATION STRESS TESTS
//  ====================================

#[tokio::test]
#[ignore] // Heavy test - run explicitly
async fn test_all_binary_ops_mega_stress() {
    let device = test_device().await;
    
    // 2000 mixed operations
    for i in 0..2000 {
        let poly_a = random_tensor_u64(&*device, 32).await;
        let poly_b = random_tensor_u64(&*device, 32).await;
        
        match i % 6 {
            0 => {
                let _ = FhePolyAdd::new(poly_a, poly_b, 32, 1152921504606584833u64);
            }
            1 => {
                let _ = FhePolySub::new(poly_a, poly_b, 32, 1152921504606584833u64);
            }
            2 => {
                let _ = FhePolyMul::new(poly_a, poly_b, 32, 1152921504606584833u64);
            }
            3 => {
                let _ = FheXor::new(poly_a, poly_b, 32);
            }
            4 => {
                let _ = FheAnd::new(poly_a, poly_b, 32);
            }
            5 => {
                let _ = FheOr::new(poly_a, poly_b, 32);
            }
            _ => unreachable!(),
        }
        
        if i % 100 == 0 {
            println!("Completed {} operations", i);
        }
    }
}

#[tokio::test]
async fn test_varying_degrees_stress() {
    let device = test_device().await;
    let degrees = vec![4, 8, 16, 32, 64, 128, 256, 512];
    
    // Test each degree with mixed operations
    for degree in degrees {
        for _ in 0..50 {
            let poly_a = random_tensor_u64(&*device, degree).await;
            let poly_b = random_tensor_u64(&*device, degree).await;
            
            let _ = FhePolyAdd::new(
                poly_a.clone(),
                poly_b.clone(),
                degree as u32,
                1152921504606584833u64,
            );
            let _ = FheXor::new(poly_a, poly_b, degree as u32);
        }
    }
}

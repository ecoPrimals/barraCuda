// SPDX-License-Identifier: AGPL-3.0-or-later
//! FHE Chaos Tests - Random inputs, stress testing, concurrent execution
//!
//! **Purpose**: Find edge case bugs through systematic randomization and stress
//! **Coverage**: Random dimensions, concurrent operations, stress tests  
//! **Deep Debt**: Discover failures through chaos, no hardcoded assumptions

use barracuda::ops::{
    fhe_ntt::FheNtt, fhe_intt::FheIntt, fhe_modulus_switch::FheModulusSwitch,
};
use barracuda::Tensor;
use rand::Rng;
use std::sync::Arc;
use tokio::task;

/// Helper to create test device
async fn test_device() -> Arc<barracuda::device::WgpuDevice> {
    barracuda::device::test_pool::get_test_device().await
}

/// Helper to create random polynomial tensor
async fn random_tensor_u64(device: &barracuda::device::WgpuDevice, degree: usize) -> Tensor {
    let mut rng = rand::thread_rng();
    let data: Vec<u32> = (0..degree * 2).map(|_| rng.gen::<u32>() % 1000).collect();
    Tensor::from_u32(&data, vec![degree * 2], device.clone())
        .await
        .expect("Failed to create tensor")
}

//  =============================
//  RANDOM INPUTS TESTS
//  =============================

#[tokio::test]
async fn test_fhe_ntt_random_degrees() {
    let device = test_device().await;
    let valid_degrees = vec![4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384];
    
    for degree in valid_degrees {
        let poly = random_tensor_u64(&*device, degree).await;
        
        let result = FheNtt::new(
            poly,
            degree as u32,
            1152921504606584833u64,
            12605157117250394513u64,
        );
        
        assert!(result.is_ok(), "Failed for degree {}", degree);
        
        // Execute to ensure GPU pipeline works
        if let Ok(op) = result {
            let exec_result = op.execute();
            assert!(exec_result.is_ok(), "Execution failed for degree {}", degree);
        }
    }
}

#[tokio::test]
async fn test_fhe_intt_random_values() {
    let device = test_device().await;
    
    // Run with 50 random polynomials
    for _i in 0..50 {
        let poly = random_tensor_u64(&*device, 64).await;
        
        let result = FheIntt::new(
            poly,
            64,
            1152921504606584833u64,
            12605157117250394513u64,
        );
        
        assert!(result.is_ok(), "Random INTT creation failed");
    }
}

#[tokio::test]
async fn test_modulus_switch_random_ratios() {
    let device = test_device().await;
    let mut rng = rand::thread_rng();
    
    // Test random modulus reduction ratios
    for _ in 0..20 {
        let poly = random_tensor_u64(&*device, 16).await;
        
        // Random large modulus
        let q_old = 1152921504606584833u64;
        // Random smaller modulus (50-90% of old)
        let reduction_factor = rng.gen_range(2..10);
        let q_new = q_old / reduction_factor;
        
        let result = FheModulusSwitch::new(
            poly,
            16,
            q_old,
            q_new,
        );
        
        assert!(result.is_ok(), "Modulus switch failed for ratio {}", reduction_factor);
    }
}

#[tokio::test]
async fn test_fhe_operations_random_valid_inputs() {
    let device = test_device().await;
    let mut rng = rand::thread_rng();
    
    // Test 100 random valid operation combinations
    for _ in 0..100 {
        let degree_pow = rng.gen_range(2..10); // 2^2 to 2^9 (4 to 512)
        let degree = 2usize.pow(degree_pow);
        
        let poly = random_tensor_u64(&*device, degree).await;
        
        // Random NTT
        let ntt_result = FheNtt::new(
            poly,
            degree as u32,
            1152921504606584833u64,
            12605157117250394513u64,
        );
        
        assert!(ntt_result.is_ok(), "Random NTT failed for degree {}", degree);
    }
}

//  =============================
//  STRESS TESTS  
//  =============================

#[tokio::test]
async fn test_fhe_ntt_sequential_stress() {
    let device = test_device().await;
    
    // Run 1000 NTT operations sequentially
    for i in 0..1000 {
        let poly = random_tensor_u64(&*device, 32).await;
        
        let result = FheNtt::new(
            poly,
            32,
            1152921504606584833u64,
            12605157117250394513u64,
        );
        
        assert!(result.is_ok(), "Failed at iteration {}", i);
        
        // Execute every 100th operation to verify GPU pipeline
        if i % 100 == 0 {
            if let Ok(op) = result {
                let exec_result = op.execute();
                assert!(exec_result.is_ok(), "Execution failed at iteration {}", i);
            }
        }
    }
}

#[tokio::test]
async fn test_modulus_switch_sequential_stress() {
    let device = test_device().await;
    
    // Run 500 modulus switch operations
    for i in 0..500 {
        let poly = random_tensor_u64(&*device, 64).await;
        
        let result = FheModulusSwitch::new(
            poly,
            64,
            1152921504606584833u64,
            288230376151711777u64,
        );
        
        assert!(result.is_ok(), "Modulus switch failed at iteration {}", i);
    }
}

#[tokio::test]
#[ignore] // Expensive test - run explicitly
async fn test_fhe_large_degree_stress() {
    let device = test_device().await;
    
    // Stress test with large degree (16384 - maximum practical for FHE)
    for i in 0..50 {
        let poly = random_tensor_u64(&*device, 16384).await;
        
        let result = FheNtt::new(
            poly,
            16384,
            1152921504606584833u64,
            12605157117250394513u64,
        );
        
        assert!(result.is_ok(), "Large degree NTT failed at iteration {}", i);
    }
}

//  =============================
//  CONCURRENT EXECUTION TESTS
//  =============================

#[tokio::test]
async fn test_fhe_ntt_concurrent_10() {
    let device = test_device().await;
    
    // Launch 10 concurrent NTT operations
    let mut handles = vec![];
    
    for i in 0..10 {
        let device_clone = Arc::clone(&device);
        let handle = task::spawn(async move {
            let poly = random_tensor_u64(&*device_clone, 64).await;
            
            let result = FheNtt::new(
                poly,
                64,
                1152921504606584833u64,
                12605157117250394513u64,
            );
            
            assert!(result.is_ok(), "Concurrent NTT {} failed", i);
            result
        });
        handles.push(handle);
    }
    
    // Wait for all to complete
    for (i, handle) in handles.into_iter().enumerate() {
        let result = handle.await;
        assert!(result.is_ok(), "Concurrent task {} panicked", i);
    }
}

#[tokio::test]
async fn test_fhe_operations_concurrent_mixed() {
    let device = test_device().await;
    
    // Launch mixed concurrent operations (NTT + INTT + ModulusSwitch)
    let mut handles = vec![];
    
    // 5 NTT operations
    for i in 0..5 {
        let device_clone = Arc::clone(&device);
        let handle = task::spawn(async move {
            let poly = random_tensor_u64(&*device_clone, 128).await;
            FheNtt::new(poly, 128, 1152921504606584833u64, 12605157117250394513u64)
                .map(|_| format!("NTT {}", i))
        });
        handles.push(handle);
    }
    
    // 5 INTT operations
    for i in 0..5 {
        let device_clone = Arc::clone(&device);
        let handle = task::spawn(async move {
            let poly = random_tensor_u64(&*device_clone, 128).await;
            FheIntt::new(poly, 128, 1152921504606584833u64, 12605157117250394513u64)
                .map(|_| format!("INTT {}", i))
        });
        handles.push(handle);
    }
    
    // 5 ModulusSwitch operations
    for i in 0..5 {
        let device_clone = Arc::clone(&device);
        let handle = task::spawn(async move {
            let poly = random_tensor_u64(&*device_clone, 128).await;
            FheModulusSwitch::new(poly, 128, 1152921504606584833u64, 288230376151711777u64)
                .map(|_| format!("ModSwitch {}", i))
        });
        handles.push(handle);
    }
    
    // Wait for all 15 operations
    for handle in handles {
        let result = handle.await;
        assert!(result.is_ok(), "Concurrent operation failed");
        let op_result = result.unwrap();
        assert!(op_result.is_ok(), "Operation creation failed: {:?}", op_result);
    }
}

#[tokio::test]
#[ignore] // Heavy concurrent test - run explicitly
async fn test_fhe_concurrent_stress_100() {
    let device = test_device().await;
    
    // Launch 100 concurrent operations (heavy stress)
    let mut handles = vec![];
    
    for i in 0..100 {
        let device_clone = Arc::clone(&device);
        let handle = task::spawn(async move {
            let poly = random_tensor_u64(&*device_clone, 256).await;
            FheNtt::new(poly, 256, 1152921504606584833u64, 12605157117250394513u64)
                .map(|_| i)
        });
        handles.push(handle);
    }
    
    // Collect all results
    let mut success_count = 0;
    for handle in handles {
        if let Ok(Ok(Ok(_))) = handle.await {
            success_count += 1;
        }
    }
    
    // Should succeed with high rate (allowing for some resource contention)
    assert!(success_count >= 90, "Too many failures: {}/100", success_count);
}

//  =============================
//  MEMORY PRESSURE TESTS
//  =============================

#[tokio::test]
#[ignore] // Memory-intensive - run explicitly
async fn test_fhe_memory_pressure() {
    let device = test_device().await;
    
    // Create many large tensors to test memory handling
    let mut tensors = vec![];
    
    for i in 0..100 {
        let poly = random_tensor_u64(&*device, 4096).await;
        tensors.push(poly);
        
        if i % 10 == 0 {
            // Verify we can still create operations
            let test_poly = random_tensor_u64(&*device, 256).await;
            let result = FheNtt::new(
                test_poly,
                256,
                1152921504606584833u64,
                12605157117250394513u64,
            );
            assert!(result.is_ok(), "Memory pressure test failed at iteration {}", i);
        }
    }
}

//  =============================
//  EDGE CASE COMBINATIONS
//  =============================

#[tokio::test]
async fn test_fhe_min_max_degree_combinations() {
    let device = test_device().await;
    
    // Test minimum and maximum degrees
    let degrees = vec![4, 16384]; // Min and max
    
    for degree in degrees {
        let poly = random_tensor_u64(&*device, degree).await;
        
        let result = FheNtt::new(
            poly,
            degree as u32,
            1152921504606584833u64,
            12605157117250394513u64,
        );
        
        assert!(result.is_ok(), "Edge degree {} failed", degree);
    }
}

#[tokio::test]
async fn test_modulus_switch_extreme_reductions() {
    let device = test_device().await;
    
    let poly = random_tensor_u64(&*device, 64).await;
    
    // Test 99% reduction (extreme but valid)
    let q_old = 1152921504606584833u64;
    let q_new = q_old / 100; // ~1% of original
    
    let result = FheModulusSwitch::new(
        poly,
        64,
        q_old,
        q_new,
    );
    
    assert!(result.is_ok(), "Extreme modulus reduction failed");
}

//! Tensor 3D corruption investigation test
//!
//! Minimal reproduction for Tensor::from_data 3D array corruption bug

use crate::device::WgpuDevice;
use crate::tensor::Tensor;
use std::sync::Arc;

#[tokio::test]
async fn test_tensor_3d_roundtrip_minimal() {
    let Some(device) = crate::device::test_pool::get_test_device_if_gpu_available().await else { return };
    
    println!("\n=== Testing 3D Tensor Roundtrip ===\n");
    
    // Test various 3D shapes
    for &(nx, ny, nz) in &[(2,2,2), (3,3,3), (4,4,4), (5,5,5)] {
        let size = nx * ny * nz;
        println!("Testing shape [{nx}, {ny}, {nz}] (size={size}):");
        
        // Create data: all 1.0s
        let data = vec![1.0f32; size];
        println!("  Input: {} elements, all 1.0", data.len());
        
        // Create tensor
        let tensor = Tensor::from_data(&data, vec![nx, ny, nz], device.clone()).unwrap();
        println!("  Tensor created: shape={:?}, len={}", tensor.shape(), tensor.len());
        
        // Read back
        let result = tensor.to_vec().unwrap();
        println!("  Output: {} elements", result.len());
        println!("  First 10 values: {:?}", &result[0..10.min(result.len())]);
        
        // Validate
        assert_eq!(result.len(), size, "Length mismatch");
        
        let mut errors = 0;
        for (i, &val) in result.iter().enumerate() {
            if (val - 1.0).abs() > 1e-5 {
                if errors < 5 {
                    println!("  ❌ Index {i}: expected 1.0, got {val}");
                }
                errors += 1;
            }
        }
        
        if errors == 0 {
            println!("  ✅ All values correct!\n");
        } else {
            println!("  ❌ {} corrupted values out of {}\n", errors, size);
            panic!("Tensor 3D corruption detected for shape [{nx}, {ny}, {nz}]");
        }
    }
    
    println!("=== All 3D shapes passed! ===");
}

#[tokio::test]
async fn test_tensor_3d_pattern() {
    let Some(device) = crate::device::test_pool::get_test_device_if_gpu_available().await else { return };
    
    println!("\n=== Testing 3D Tensor with Pattern ===\n");
    
    // 3x3x3 with incrementing values
    let data: Vec<f32> = (0..27).map(|i| i as f32).collect();
    println!("Input pattern: [0.0, 1.0, 2.0, ..., 26.0]");
    println!("First 10: {:?}", &data[0..10]);
    
    let tensor = Tensor::from_data(&data, vec![3, 3, 3], device.clone()).unwrap();
    let result = tensor.to_vec().unwrap();
    
    println!("Output pattern:");
    println!("First 10: {:?}", &result[0..10]);
    println!("Last 10:  {:?}", &result[17..27]);
    
    // Check if pattern matches
    let mut errors = 0;
    for (i, &val) in result.iter().enumerate() {
        if (val - i as f32).abs() > 1e-5 {
            if errors < 10 {
                println!("  ❌ Index {i}: expected {}.0, got {val}", i);
            }
            errors += 1;
        }
    }
    
    if errors > 0 {
        println!("\n❌ {} corrupted values", errors);
        panic!("Pattern corruption detected");
    }
    
    println!("\n✅ Pattern preserved correctly!");
}

#[tokio::test]
async fn test_tensor_2d_vs_3d() {
    let Some(device) = crate::device::test_pool::get_test_device_if_gpu_available().await else { return };
    
    println!("\n=== Comparing 2D vs 3D Tensors ===\n");
    
    // 2D tensor (known to work)
    let data_2d = vec![1.0; 9];
    let tensor_2d = Tensor::from_data(&data_2d, vec![3, 3], device.clone()).unwrap();
    let result_2d = tensor_2d.to_vec().unwrap();
    println!("2D [3,3]: {:?}", &result_2d);
    
    // 3D tensor (suspected corruption)
    let data_3d = vec![1.0; 27];
    let tensor_3d = Tensor::from_data(&data_3d, vec![3, 3, 3], device.clone()).unwrap();
    let result_3d = tensor_3d.to_vec().unwrap();
    println!("3D [3,3,3]: {:?}", &result_3d[0..10]);
    
    // Check 2D
    for (i, &val) in result_2d.iter().enumerate() {
        assert!((val - 1.0).abs() < 1e-5, "2D corrupted at {i}: {val}");
    }
    println!("✅ 2D tensor: All correct");
    
    // Check 3D
    let mut errors_3d = 0;
    for (i, &val) in result_3d.iter().enumerate() {
        if (val - 1.0).abs() > 1e-5 {
            errors_3d += 1;
        }
    }
    
    if errors_3d > 0 {
        println!("❌ 3D tensor: {} corrupted values", errors_3d);
        panic!("3D corruption confirmed");
    }
    
    println!("✅ 3D tensor: All correct");
}

//! Tests for Expand operation

use super::*;
use crate::device::test_pool::get_test_device_if_gpu_available;

#[tokio::test]
async fn test_expand_basic() {
    let Some(device) = get_test_device_if_gpu_available().await else {
        return;
    };
    // Broadcast from [3] to [9] (repeat 3 times)
    let input_data = vec![1.0, 2.0, 3.0];
    let input = Tensor::from_vec_on(input_data.clone(), vec![3], device)
        .await
        .unwrap();
    let result = input.expand_wgsl(vec![9]).unwrap().to_vec().unwrap();

    assert_eq!(result.len(), 9);
    // Should repeat pattern: [1,2,3,1,2,3,1,2,3]
    let expected = vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0];
    for (r, e) in result.iter().zip(expected.iter()) {
        assert!((r - e).abs() < 1e-6, "Expected {}, got {}", e, r);
    }
}

#[tokio::test]
async fn test_expand_single_element() {
    let Some(device) = get_test_device_if_gpu_available().await else {
        return;
    };
    // Broadcast single value to multiple
    let input_data = vec![5.0];
    let input = Tensor::from_vec_on(input_data, vec![1], device)
        .await
        .unwrap();
    let result = input.expand_wgsl(vec![10]).unwrap().to_vec().unwrap();

    assert_eq!(result.len(), 10);
    assert!(result.iter().all(|&x| (x - 5.0).abs() < 1e-6));
}

#[tokio::test]
async fn test_expand_no_change() {
    let Some(device) = get_test_device_if_gpu_available().await else {
        return;
    };
    // No expansion needed (already target size)
    let input_data = vec![1.0, 2.0, 3.0];
    let input = Tensor::from_vec_on(input_data.clone(), vec![3], device.clone())
        .await
        .unwrap();
    let result = input.expand_wgsl(vec![3]).unwrap();
    let output = result.to_vec().unwrap();
    assert_eq!(output, input_data);
}

#[tokio::test]
async fn test_expand_boundary() {
    let Some(dev) = get_test_device_if_gpu_available().await else {
        return;
    };

    // Large expansion factor
    let input = vec![2.78];
    let output = Tensor::from_vec_on(input, vec![1], dev.clone())
        .await
        .unwrap()
        .expand_wgsl(vec![1000])
        .unwrap()
        .to_vec()
        .unwrap();
    assert_eq!(output.len(), 1000);
    assert!(output.iter().all(|&x| (x - 2.78).abs() < 1e-5));

    // Smaller expansion
    let input = vec![99.0];
    let output = Tensor::from_vec_on(input, vec![1], dev.clone())
        .await
        .unwrap()
        .expand_wgsl(vec![5])
        .unwrap()
        .to_vec()
        .unwrap();
    assert_eq!(output.len(), 5);
    assert!(output.iter().all(|&x| (x - 99.0).abs() < 1e-5));
}

#[tokio::test]
async fn test_expand_large_batch() {
    let Some(dev) = get_test_device_if_gpu_available().await else {
        return;
    };

    // Single value to large tensor
    let input = vec![42.0];
    let output = Tensor::from_vec_on(input, vec![1], dev.clone())
        .await
        .unwrap()
        .expand_wgsl(vec![10000])
        .unwrap()
        .to_vec()
        .unwrap();
    assert_eq!(output.len(), 10000);
    assert!(output.iter().all(|&x| (x - 42.0).abs() < 1e-5));
}

#[tokio::test]
async fn test_expand_precision() {
    let Some(dev) = get_test_device_if_gpu_available().await else {
        return;
    };

    // Verify exact value preserved during broadcast
    let input = vec![1.23456];
    let output = Tensor::from_vec_on(input, vec![1], dev.clone())
        .await
        .unwrap()
        .expand_wgsl(vec![100])
        .unwrap()
        .to_vec()
        .unwrap();
    assert_eq!(output.len(), 100);

    // All values should match exactly
    for val in output.iter() {
        assert!((val - 1.23456).abs() < 1e-6);
    }
}

#[tokio::test]
async fn test_expand_2d_broadcast_second_dim() {
    let Some(dev) = get_test_device_if_gpu_available().await else {
        return;
    };
    // (3, 1) → (3, 5): broadcast second dim
    let input_data: Vec<f32> = vec![1.0, 2.0, 3.0];
    let input = Tensor::from_vec_on(input_data, vec![3, 1], dev.clone())
        .await
        .unwrap();
    let result = input.expand_wgsl(vec![3, 5]).unwrap();
    let output = result.to_vec().unwrap();

    assert_eq!(result.shape(), &vec![3, 5]);
    // Each row should be the same: [1,1,1,1,1], [2,2,2,2,2], [3,3,3,3,3]
    for i in 0..3 {
        let expected_val = (i + 1) as f32;
        for j in 0..5 {
            let idx = i * 5 + j;
            assert!(
                (output[idx] - expected_val).abs() < 1e-6,
                "Expected {} at [{}, {}], got {}",
                expected_val,
                i,
                j,
                output[idx]
            );
        }
    }
}

#[tokio::test]
async fn test_expand_2d_broadcast_first_dim() {
    let Some(dev) = get_test_device_if_gpu_available().await else {
        return;
    };
    // (1, 5) → (4, 5): broadcast first dim
    let input_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let input = Tensor::from_vec_on(input_data.clone(), vec![1, 5], dev.clone())
        .await
        .unwrap();
    let result = input.expand_wgsl(vec![4, 5]).unwrap();
    let output = result.to_vec().unwrap();

    assert_eq!(result.shape(), &vec![4, 5]);
    // All rows should be the same: [1,2,3,4,5]
    for i in 0..4 {
        for j in 0..5 {
            let idx = i * 5 + j;
            assert!(
                (output[idx] - input_data[j]).abs() < 1e-6,
                "Expected {} at [{}, {}], got {}",
                input_data[j],
                i,
                j,
                output[idx]
            );
        }
    }
}

#[tokio::test]
async fn test_expand_add_dimension() {
    let Some(dev) = get_test_device_if_gpu_available().await else {
        return;
    };
    // (3,) → (3, 5): add dimension then broadcast
    let input_data: Vec<f32> = vec![1.0, 2.0, 3.0];
    let input = Tensor::from_vec_on(input_data.clone(), vec![3], dev.clone())
        .await
        .unwrap();
    let result = input.expand_wgsl(vec![3, 5]).unwrap();
    let output = result.to_vec().unwrap();

    assert_eq!(result.shape(), &vec![3, 5]);
    // Each row should repeat the same value: [1,1,1,1,1], [2,2,2,2,2], [3,3,3,3,3]
    for i in 0..3 {
        let expected_val = input_data[i];
        for j in 0..5 {
            let idx = i * 5 + j;
            assert!(
                (output[idx] - expected_val).abs() < 1e-6,
                "Expected {} at [{}, {}], got {}",
                expected_val,
                i,
                j,
                output[idx]
            );
        }
    }
}

#[tokio::test]
async fn test_expand_3d_broadcast_middle_dim() {
    let Some(dev) = get_test_device_if_gpu_available().await else {
        return;
    };
    // (3, 1, 5) → (3, 4, 5): broadcast middle dim
    let input_data: Vec<f32> = (0..15).map(|i| i as f32).collect(); // 3*1*5 = 15
    let input = Tensor::from_vec_on(input_data, vec![3, 1, 5], dev.clone())
        .await
        .unwrap();
    let result = input.expand_wgsl(vec![3, 4, 5]).unwrap();
    let output = result.to_vec().unwrap();

    assert_eq!(result.shape(), &vec![3, 4, 5]);
    // For each of the 3 slices, the middle dimension should be broadcasted
    // Slice 0: values 0-4 repeated 4 times
    // Slice 1: values 5-9 repeated 4 times
    // Slice 2: values 10-14 repeated 4 times
    for i in 0..3 {
        for j in 0..4 {
            for k in 0..5 {
                let idx = i * 20 + j * 5 + k;
                let expected_val = (i * 5 + k) as f32;
                assert!(
                    (output[idx] - expected_val).abs() < 1e-6,
                    "Expected {} at [{}, {}, {}], got {}",
                    expected_val,
                    i,
                    j,
                    k,
                    output[idx]
                );
            }
        }
    }
}

#[tokio::test]
async fn test_expand_scalar_to_tensor() {
    let Some(dev) = get_test_device_if_gpu_available().await else {
        return;
    };
    // Scalar (1,) → (2, 3, 4)
    let input_data = vec![42.0];
    let input = Tensor::from_vec_on(input_data, vec![1], dev.clone())
        .await
        .unwrap();
    let result = input.expand_wgsl(vec![2, 3, 4]).unwrap();
    let output = result.to_vec().unwrap();

    assert_eq!(result.shape(), &vec![2, 3, 4]);
    assert_eq!(output.len(), 24);
    assert!(output.iter().all(|&x| (x - 42.0).abs() < 1e-6));
}

#[tokio::test]
async fn test_expand_incompatible_shapes() {
    let Some(dev) = get_test_device_if_gpu_available().await else {
        return;
    };
    // (3, 4) cannot broadcast to (3, 5) - both dims are > 1 and different
    let input_data: Vec<f32> = (0..12).map(|i| i as f32).collect();
    let input = Tensor::from_vec_on(input_data, vec![3, 4], dev.clone())
        .await
        .unwrap();

    assert!(input.expand_wgsl(vec![3, 5]).is_err());
}

#[tokio::test]
async fn test_expand_4d_broadcast() {
    let Some(dev) = get_test_device_if_gpu_available().await else {
        return;
    };
    // (1, 3, 1, 5) → (2, 3, 4, 5): broadcast first and third dims
    let input_data: Vec<f32> = (0..15).map(|i| i as f32).collect(); // 1*3*1*5 = 15
    let input = Tensor::from_vec_on(input_data, vec![1, 3, 1, 5], dev.clone())
        .await
        .unwrap();
    let result = input.expand_wgsl(vec![2, 3, 4, 5]).unwrap();
    let output = result.to_vec().unwrap();

    assert_eq!(result.shape(), &vec![2, 3, 4, 5]);
    // Verify broadcasting: first and third dims are broadcasted
    for i in 0..2 {
        for j in 0..3 {
            for k in 0..4 {
                for l in 0..5 {
                    let idx = i * 60 + j * 20 + k * 5 + l;
                    let expected_val = (j * 5 + l) as f32;
                    assert!(
                        (output[idx] - expected_val).abs() < 1e-6,
                        "Expected {} at [{}, {}, {}, {}], got {}",
                        expected_val,
                        i,
                        j,
                        k,
                        l,
                        output[idx]
                    );
                }
            }
        }
    }
}

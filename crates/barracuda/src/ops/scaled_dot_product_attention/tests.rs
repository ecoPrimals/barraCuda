// SPDX-License-Identifier: AGPL-3.0-only
//! Tests for Scaled Dot-Product Attention operation

use crate::device::WgpuDevice;
use crate::error::Result;
use crate::tensor::Tensor;
use std::sync::Arc;

async fn create_test_tensor(
    device: Arc<WgpuDevice>,
    shape: Vec<usize>,
    value: f32,
) -> Result<Tensor> {
    let size: usize = shape.iter().product();
    let data: Vec<f32> = vec![value; size];
    Tensor::from_vec_on(data, shape, device).await
}

#[tokio::test]
async fn test_scaled_dot_product_attention_basic() {
    let Some(dev) = crate::device::test_pool::get_test_device_if_gpu_available().await else {
        return;
    };

    // Small example: 1 batch, 1 head, 2 seq, 2 dim
    let query = create_test_tensor(dev.clone(), vec![1, 1, 2, 2], 0.5)
        .await
        .unwrap();
    let key = create_test_tensor(dev.clone(), vec![1, 1, 2, 2], 0.5)
        .await
        .unwrap();
    let value = create_test_tensor(dev.clone(), vec![1, 1, 2, 2], 1.0)
        .await
        .unwrap();

    let output = query.scaled_dot_product_attention(key, value).unwrap();

    assert_eq!(output.shape(), &[1, 1, 2, 2]);
}

#[tokio::test]
async fn test_scaled_dot_product_attention_multi_head() {
    let Some(dev) = crate::device::test_pool::get_test_device_if_gpu_available().await else {
        return;
    };

    // Multi-head: 2 batch, 4 heads, 8 seq, 16 dim
    let query = create_test_tensor(dev.clone(), vec![2, 4, 8, 16], 0.5)
        .await
        .unwrap();
    let key = create_test_tensor(dev.clone(), vec![2, 4, 8, 16], 0.5)
        .await
        .unwrap();
    let value = create_test_tensor(dev.clone(), vec![2, 4, 8, 16], 1.0)
        .await
        .unwrap();

    let output = query.scaled_dot_product_attention(key, value).unwrap();

    assert_eq!(output.shape(), &[2, 4, 8, 16]);
}

#[tokio::test]
async fn test_scaled_dot_product_attention_shape_validation() {
    let Some(dev) = crate::device::test_pool::get_test_device_if_gpu_available().await else {
        return;
    };

    let query = create_test_tensor(dev.clone(), vec![1, 1, 4, 4], 0.5)
        .await
        .unwrap();
    let key = create_test_tensor(dev.clone(), vec![1, 1, 4, 4], 0.5)
        .await
        .unwrap();
    let value = create_test_tensor(dev.clone(), vec![1, 1, 4, 5], 1.0)
        .await
        .unwrap(); // Wrong shape

    let result = query.scaled_dot_product_attention(key, value);
    assert!(result.is_err()); // Should fail shape validation
}

//! Tests for Scaled Dot-Product Attention

use super::*;
use crate::device::test_pool::get_test_device_if_gpu_available;

#[tokio::test]
async fn test_attention_basic() {
    let Some(device) = get_test_device_if_gpu_available().await else {
        return;
    };
    // Small test: [1 batch, 1 head, 4 seq, 8 dim]
    let query = Tensor::from_vec_on(vec![1.0; 32], vec![1, 1, 4, 8], device.clone())
        .await
        .unwrap();
    let key = Tensor::from_vec_on(vec![1.0; 32], vec![1, 1, 4, 8], device.clone())
        .await
        .unwrap();
    let value = Tensor::from_vec_on(vec![2.0; 32], vec![1, 1, 4, 8], device)
        .await
        .unwrap();

    let output = query.attention(&key, &value).unwrap();

    assert_eq!(output.shape(), &[1, 1, 4, 8]);
    let result = output.to_vec().unwrap();

    // With uniform Q,K, attention weights should be uniform (1/seq_len)
    // So output should be close to value (since all weighted equally)
    assert!(result.iter().all(|&x| (x - 2.0).abs() < 0.1));
}

#[tokio::test]
async fn test_attention_shape_validation() {
    let Some(device) = get_test_device_if_gpu_available().await else {
        return;
    };
    let query = Tensor::from_vec_on(vec![1.0; 32], vec![1, 1, 4, 8], device.clone())
        .await
        .unwrap();
    let key = Tensor::from_vec_on(vec![1.0; 16], vec![1, 1, 2, 8], device.clone()) // Wrong shape!
        .await
        .unwrap();
    let value = Tensor::from_vec_on(vec![1.0; 32], vec![1, 1, 4, 8], device)
        .await
        .unwrap();

    let result = query.attention(&key, &value);
    assert!(result.is_err());
}

#[tokio::test]
async fn test_attention_multi_head() {
    let Some(device) = get_test_device_if_gpu_available().await else {
        return;
    };
    // Test with multiple heads: [2 batch, 4 heads, 8 seq, 16 dim]
    let size = 2 * 4 * 8 * 16;
    let query = Tensor::from_vec_on(vec![0.5; size], vec![2, 4, 8, 16], device.clone())
        .await
        .unwrap();
    let key = query.clone();
    let value = Tensor::from_vec_on(vec![1.0; size], vec![2, 4, 8, 16], device)
        .await
        .unwrap();

    let output = query.attention(&key, &value).unwrap();

    assert_eq!(output.shape(), &[2, 4, 8, 16]);
    let result = output.to_vec().unwrap();
    assert_eq!(result.len(), size);
    assert!(result.iter().all(|&x| x.is_finite()));
}

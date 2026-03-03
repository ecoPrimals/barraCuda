//! Tests for Grouped Query Attention
//!
//! Validates GQA with grouped key/value heads (LLaMA-style).

use super::*;
use crate::device::WgpuDevice;
use crate::error::Result;
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
async fn test_grouped_query_attention_basic() {
    let Some(dev) = crate::device::test_pool::get_test_device_if_gpu_available().await else {
        return;
    };

    // 8 query heads, 2 KV heads (4 heads per group)
    let query = create_test_tensor(dev.clone(), vec![1, 8, 4, 4], 0.5)
        .await
        .unwrap();
    let key = create_test_tensor(dev.clone(), vec![1, 2, 4, 4], 0.5)
        .await
        .unwrap();
    let value = create_test_tensor(dev.clone(), vec![1, 2, 4, 4], 0.5)
        .await
        .unwrap();

    let output = query.grouped_query_attention(key, value).unwrap();

    assert_eq!(output.shape(), &[1, 8, 4, 4]);
}

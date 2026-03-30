// SPDX-License-Identifier: AGPL-3.0-or-later
//! Tests for `NonZero` operation

use super::NonZero;
use crate::device::test_pool::get_test_device;
use crate::tensor::Tensor;

#[tokio::test]
async fn test_nonzero_basic() {
    let device = get_test_device().await;
    let input = Tensor::from_vec_on(vec![0.0, 1.0, 0.0, 2.0, 0.0], vec![5], device.clone())
        .await
        .unwrap();

    let result = NonZero::new(input).unwrap().execute().unwrap();
    let indices = result.to_vec().unwrap();
    assert_eq!(indices.len(), 2);
    assert_eq!(indices[0] as u32, 1);
    assert_eq!(indices[1] as u32, 3);
}

#[tokio::test]
async fn test_nonzero_all_zero() {
    let device = get_test_device().await;
    let input = Tensor::from_vec_on(vec![0.0, 0.0, 0.0], vec![3], device.clone())
        .await
        .unwrap();

    let result = NonZero::new(input).unwrap().execute().unwrap();
    assert_eq!(result.shape(), &[0]);
}

#[tokio::test]
async fn test_nonzero_all_nonzero() {
    let device = get_test_device().await;
    let input = Tensor::from_vec_on(vec![1.0, 2.0, 3.0], vec![3], device.clone())
        .await
        .unwrap();

    let result = NonZero::new(input).unwrap().execute().unwrap();
    let indices = result.to_vec().unwrap();
    assert_eq!(indices.len(), 3);
}

#[tokio::test]
async fn test_nonzero_2d() {
    let device = get_test_device().await;
    let input = Tensor::from_vec_on(
        vec![0.0, 1.0, 0.0, 2.0, 0.0, 3.0],
        vec![2, 3],
        device.clone(),
    )
    .await
    .unwrap();

    let result = NonZero::new(input).unwrap().execute().unwrap();
    let indices = result.to_vec().unwrap();
    assert_eq!(indices.len(), 3);
}

#[tokio::test]
async fn test_nonzero_empty() {
    let device = get_test_device().await;
    let input = Tensor::from_vec_on(vec![], vec![0], device.clone())
        .await
        .unwrap();

    assert!(NonZero::new(input).is_err());
}

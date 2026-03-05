// SPDX-License-Identifier: AGPL-3.0-or-later
use crate::device::test_pool::get_test_device_if_gpu_available;
use crate::ops::unique::Unique;
use crate::tensor::Tensor;

#[tokio::test]
async fn test_unique_basic() {
    let Some(device) = get_test_device_if_gpu_available().await else {
        return;
    };
    let input = Tensor::from_vec_on(vec![1.0, 2.0, 1.0, 3.0, 2.0], vec![5], device.clone())
        .await
        .unwrap();

    let result = Unique::new(input).unwrap().execute().unwrap();
    let unique = result.to_vec().unwrap();
    assert!(unique.len() <= 5);
}

#[tokio::test]
async fn test_unique_all_same() {
    let Some(device) = get_test_device_if_gpu_available().await else {
        return;
    };
    let input = Tensor::from_vec_on(vec![5.0, 5.0, 5.0], vec![3], device.clone())
        .await
        .unwrap();

    let result = Unique::new(input).unwrap().execute().unwrap();
    let unique = result.to_vec().unwrap();
    assert_eq!(unique.len(), 1);
    assert_eq!(unique[0], 5.0);
}

#[tokio::test]
async fn test_unique_all_different() {
    let Some(device) = get_test_device_if_gpu_available().await else {
        return;
    };
    let input = Tensor::from_vec_on(vec![1.0, 2.0, 3.0], vec![3], device.clone())
        .await
        .unwrap();

    let result = Unique::new(input).unwrap().execute().unwrap();
    let unique = result.to_vec().unwrap();
    assert_eq!(unique.len(), 3);
}

#[tokio::test]
async fn test_unique_empty() {
    let Some(device) = get_test_device_if_gpu_available().await else {
        return;
    };
    let input = Tensor::from_vec_on(vec![], vec![0], device.clone())
        .await
        .unwrap();

    assert!(Unique::new(input).is_err());
}

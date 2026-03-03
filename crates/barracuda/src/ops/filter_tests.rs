use super::*;

#[tokio::test]
async fn test_filter_gt_basic() {
    let Some(device) = crate::device::test_pool::get_test_device_if_gpu_available().await else {
        return;
    };
    // [1.0, 5.0, 3.0, 7.0] > 4.0 → keep [5.0, 7.0], count=2
    let input = Tensor::from_data(&[1.0f32, 5.0, 3.0, 7.0], vec![4], device.clone()).unwrap();
    let result = input.filter(FilterOperation::GreaterThan, 4.0).unwrap();
    assert_eq!(result.count, 2, "Expected 2 elements > 4.0");
    let out = result.selected.to_vec().unwrap();
    // First `count` elements are valid
    let selected: Vec<f32> = out[..result.count].to_vec();
    assert!(
        selected.iter().all(|&v| v > 4.0),
        "All selected must be > 4.0: {selected:?}"
    );
}

#[tokio::test]
async fn test_filter_all_pass() {
    let Some(device) = crate::device::test_pool::get_test_device_if_gpu_available().await else {
        return;
    };
    let input = Tensor::from_data(&[1.0f32, 2.0, 3.0], vec![3], device).unwrap();
    let result = input.filter(FilterOperation::LessThan, 100.0).unwrap();
    assert_eq!(result.count, 3);
}

#[tokio::test]
async fn test_filter_none_pass() {
    let Some(device) = crate::device::test_pool::get_test_device_if_gpu_available().await else {
        return;
    };
    let input = Tensor::from_data(&[1.0f32, 2.0, 3.0], vec![3], device).unwrap();
    let result = input.filter(FilterOperation::GreaterThan, 100.0).unwrap();
    assert_eq!(result.count, 0);
}

#[tokio::test]
async fn test_filter_ge() {
    let Some(device) = crate::device::test_pool::get_test_device_if_gpu_available().await else {
        return;
    };
    // [1, 5, 5, 7] >= 5 → [5, 5, 7], count=3
    let input = Tensor::from_data(&[1.0f32, 5.0, 5.0, 7.0], vec![4], device).unwrap();
    let result = input.filter(FilterOperation::GreaterOrEqual, 5.0).unwrap();
    assert_eq!(result.count, 3);
}

#[tokio::test]
async fn test_filter_empty_input() {
    let Some(device) = crate::device::test_pool::get_test_device_if_gpu_available().await else {
        return;
    };
    let input = Tensor::from_data(&[] as &[f32], vec![0], device).unwrap();
    let result = input.filter(FilterOperation::GreaterThan, 0.0).unwrap();
    assert_eq!(result.count, 0);
}

#[tokio::test]
async fn test_filter_single_element() {
    let Some(device) = crate::device::test_pool::get_test_device_if_gpu_available().await else {
        return;
    };
    let input = Tensor::from_data(&[42.0f32], vec![1], device).unwrap();
    let result = input.filter(FilterOperation::GreaterThan, 10.0).unwrap();
    assert_eq!(result.count, 1);
    let out = result.selected.to_vec().unwrap();
    assert!((out[0] - 42.0).abs() < 1e-5);
}

#[tokio::test]
async fn test_filter_single_element_none_pass() {
    let Some(device) = crate::device::test_pool::get_test_device_if_gpu_available().await else {
        return;
    };
    let input = Tensor::from_data(&[5.0f32], vec![1], device).unwrap();
    let result = input.filter(FilterOperation::GreaterThan, 10.0).unwrap();
    assert_eq!(result.count, 0);
}

#[tokio::test]
async fn test_filter_large() {
    let Some(device) = crate::device::test_pool::get_test_device_if_gpu_available().await else {
        return;
    };
    // 1024 elements alternating positive/negative — keep positive (> 0)
    let data: Vec<f32> = (0..1024)
        .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
        .collect();
    let input = Tensor::from_data(&data, vec![1024], device).unwrap();
    let result = input.filter(FilterOperation::GreaterThan, 0.0).unwrap();
    assert_eq!(result.count, 512);
}

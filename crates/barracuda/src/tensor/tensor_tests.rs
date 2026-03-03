// SPDX-License-Identifier: AGPL-3.0-or-later
use super::*;

#[tokio::test]
async fn test_tensor_creation() {
    let tensor = Tensor::zeros(vec![2, 3]).await.unwrap();
    assert_eq!(tensor.shape(), &[2, 3]);
    assert_eq!(tensor.len(), 6);

    let data = tensor.to_vec().unwrap();
    assert_eq!(data, vec![0.0; 6]);
}

#[tokio::test]
async fn test_tensor_from_vec() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let tensor = Tensor::from_vec(data.clone(), vec![2, 3]).await.unwrap();

    assert_eq!(tensor.shape(), &[2, 3]);
    assert_eq!(tensor.to_vec().unwrap(), data);
}

#[tokio::test]
async fn test_tensor_reshape() {
    let tensor = Tensor::ones(vec![2, 3]).await.unwrap();
    let reshaped = tensor.reshape(vec![3, 2]).unwrap();

    assert_eq!(reshaped.shape(), &[3, 2]);
    assert_eq!(reshaped.len(), 6);
}

#[tokio::test]
async fn test_tensor_device() {
    let tensor = Tensor::zeros(vec![10]).await.unwrap();
    println!("Tensor on device: {}", tensor.device().name());
    assert!(!tensor.device().name().is_empty());
}

#[tokio::test]
async fn test_scalar_mul() {
    let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![4])
        .await
        .unwrap();
    let result = tensor.mul_scalar(2.0).unwrap();
    let data = result.to_vec().unwrap();

    assert_eq!(data, vec![2.0, 4.0, 6.0, 8.0]);
}

#[tokio::test]
async fn test_scalar_add() {
    let tensor = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![4])
        .await
        .unwrap();
    let result = tensor.add_scalar(10.0).unwrap();
    let data = result.to_vec().unwrap();

    assert_eq!(data, vec![11.0, 12.0, 13.0, 14.0]);
}

#[tokio::test]
async fn test_scalar_div() {
    let tensor = Tensor::from_vec(vec![10.0, 20.0, 30.0, 40.0], vec![4])
        .await
        .unwrap();
    let result = tensor.div_scalar(2.0).unwrap();
    let data = result.to_vec().unwrap();

    assert_eq!(data, vec![5.0, 10.0, 15.0, 20.0]);
}

#[tokio::test]
async fn test_randn_shape() {
    let tensor = Tensor::randn(vec![10, 20]).await.unwrap();
    assert_eq!(tensor.shape(), &[10, 20]);
    assert_eq!(tensor.len(), 200);

    let data = tensor.to_vec().unwrap();
    let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
    let variance: f32 = data.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / data.len() as f32;

    assert!(mean.abs() < 0.3, "Mean {} too far from 0", mean);
    assert!(
        (variance.sqrt() - 1.0).abs() < 0.3,
        "Std {} too far from 1",
        variance.sqrt()
    );
}

#[tokio::test]
async fn test_rand_shape() {
    let tensor = Tensor::rand(vec![10, 20]).await.unwrap();
    assert_eq!(tensor.shape(), &[10, 20]);
    assert_eq!(tensor.len(), 200);

    let data = tensor.to_vec().unwrap();
    for &val in &data {
        assert!((0.0..1.0).contains(&val), "Value {} out of range", val);
    }
}

#[tokio::test]
async fn test_rand_range() {
    let tensor = Tensor::rand_range(vec![100], -5.0, 5.0).await.unwrap();
    let data = tensor.to_vec().unwrap();

    for &val in &data {
        assert!((-5.0..5.0).contains(&val), "Value {} out of range", val);
    }

    let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
    assert!(mean.abs() < 1.0, "Mean {} too far from 0", mean);
}

#[tokio::test]
async fn test_randn_reproducible() {
    use rand::SeedableRng;

    let mut rng1 = rand::rngs::StdRng::seed_from_u64(42);
    let tensor1 = Tensor::randn_with_rng(vec![10], &mut rng1).await.unwrap();

    let mut rng2 = rand::rngs::StdRng::seed_from_u64(42);
    let tensor2 = Tensor::randn_with_rng(vec![10], &mut rng2).await.unwrap();

    assert_eq!(tensor1.to_vec().unwrap(), tensor2.to_vec().unwrap());
}

#[tokio::test]
async fn test_query_device() {
    let tensor = Tensor::randn(vec![10]).await.unwrap();
    let device = tensor.query_device();

    assert!(matches!(device, Device::CPU | Device::GPU | Device::Auto));
}

#[tokio::test]
async fn test_prefer_device() {
    let tensor = Tensor::randn(vec![10, 10]).await.unwrap();

    let gpu_tensor = tensor.prefer_device(Device::GPU);
    assert_eq!(gpu_tensor.shape(), tensor.shape());
    assert_eq!(gpu_tensor.len(), tensor.len());
}

#[tokio::test]
async fn test_with_hint() {
    let tensor = Tensor::randn(vec![5, 5]).await.unwrap();

    let small = tensor.with_hint(WorkloadHint::SmallWorkload);
    assert_eq!(small.shape(), tensor.shape());

    let large = tensor.with_hint(WorkloadHint::LargeMatrices);
    assert_eq!(large.shape(), tensor.shape());
}

/// Validates 3D tensor create→readback roundtrip for multiple shapes.
/// Subsumes the former `test_tensor_laplacian_context_debug` which was identical
/// for a single 3×3×3 shape.
#[tokio::test]
async fn test_tensor_3d_roundtrip() -> Result<()> {
    let Some(device) = crate::device::test_pool::get_test_device_if_gpu_available().await else {
        return Ok(());
    };

    for &(nx, ny, nz) in &[(2, 2, 2), (3, 3, 3), (4, 4, 4)] {
        let size = nx * ny * nz;
        let data = vec![1.0f32; size];
        let tensor = Tensor::from_data(&data, vec![nx, ny, nz], device.clone()).unwrap();
        let result = tensor.to_vec().unwrap();

        assert_eq!(result.len(), size, "Length mismatch for [{nx}, {ny}, {nz}]");
        for (i, &val) in result.iter().enumerate() {
            assert!(
                (val - 1.0).abs() < 1e-5,
                "[{nx},{ny},{nz}] idx {i}: expected 1.0, got {val}"
            );
        }
    }

    Ok(())
}

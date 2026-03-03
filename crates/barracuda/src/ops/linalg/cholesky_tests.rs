// SPDX-License-Identifier: AGPL-3.0-or-later
use super::*;
use crate::device::test_pool::{
    get_test_device_if_f64_gpu_available, get_test_device_if_gpu_available,
};

const CHOLESKY_F32_SHADER: &str = include_str!("../../shaders/linalg/cholesky.wgsl");
const CHOLESKY_F64_SHADER: &str = include_str!("../../shaders/linalg/cholesky_f64.wgsl");

#[test]
fn cholesky_f32_shader_source_valid() {
    assert!(!CHOLESKY_F32_SHADER.is_empty());
    assert!(CHOLESKY_F32_SHADER.contains("fn ") || CHOLESKY_F32_SHADER.contains("@compute"));
}

#[test]
fn cholesky_f64_shader_source_valid() {
    assert!(!CHOLESKY_F64_SHADER.is_empty());
    assert!(CHOLESKY_F64_SHADER.contains("fn ") || CHOLESKY_F64_SHADER.contains("@compute"));
}

#[tokio::test]
async fn test_cholesky_2x2() {
    let Some(device) = get_test_device_if_gpu_available().await else {
        return;
    };
    // Simple 2x2 SPD matrix: [[4, 2], [2, 3]]
    // Expected L: [[2, 0], [1, sqrt(2)]]
    // Verification: L·Lᵀ = [[4, 2], [2, 3]] ✓
    let input_data = vec![4.0, 2.0, 2.0, 3.0];
    let input = Tensor::from_vec_on(input_data, vec![2, 2], device)
        .await
        .unwrap();

    let l = input.cholesky().unwrap();
    let output = l.to_vec().unwrap();

    // L should be lower triangular
    assert_eq!(output.len(), 4);

    // Check L[0,0] ≈ 2.0
    assert!(
        (output[0] - 2.0).abs() < 1e-5,
        "L[0,0] should be 2.0, got {}",
        output[0]
    );

    // Check L[0,1] ≈ 0.0 (upper triangle)
    assert!(
        output[1].abs() < 1e-5,
        "L[0,1] should be 0.0, got {}",
        output[1]
    );

    // Check L[1,0] ≈ 1.0
    assert!(
        (output[2] - 1.0).abs() < 1e-5,
        "L[1,0] should be 1.0, got {}",
        output[2]
    );

    // Check L[1,1] ≈ sqrt(2)
    assert!(
        (output[3] - std::f32::consts::SQRT_2).abs() < 1e-3,
        "L[1,1] should be sqrt(2), got {}",
        output[3]
    );
}

#[tokio::test]
async fn test_cholesky_identity() {
    let Some(device) = get_test_device_if_gpu_available().await else {
        return;
    };
    // Identity matrix should have L = I
    let input_data = vec![1.0, 0.0, 0.0, 1.0];
    let input = Tensor::from_vec_on(input_data, vec![2, 2], device)
        .await
        .unwrap();

    let l = input.cholesky().unwrap();
    let output = l.to_vec().unwrap();

    // Should be identity
    assert!((output[0] - 1.0).abs() < 1e-5);
    assert!(output[1].abs() < 1e-5);
    assert!(output[2].abs() < 1e-5);
    assert!((output[3] - 1.0).abs() < 1e-5);
}

#[tokio::test]
async fn test_cholesky_3x3() {
    let Some(device) = get_test_device_if_gpu_available().await else {
        return;
    };
    // 3x3 SPD matrix
    let input_data = vec![4.0, 2.0, 1.0, 2.0, 3.0, 1.0, 1.0, 1.0, 3.0];
    let input = Tensor::from_vec_on(input_data, vec![3, 3], device)
        .await
        .unwrap();

    let l = input.cholesky().unwrap();
    let output = l.to_vec().unwrap();

    // Just verify it's lower triangular and not all zeros
    assert_eq!(output.len(), 9);

    // Upper triangle should be zero
    assert!(output[1].abs() < 1e-5); // L[0,1]
    assert!(output[2].abs() < 1e-5); // L[0,2]
    assert!(output[5].abs() < 1e-5); // L[1,2]

    // Diagonal should be positive
    assert!(output[0] > 0.0); // L[0,0]
    assert!(output[4] > 0.0); // L[1,1]
    assert!(output[8] > 0.0); // L[2,2]
}

#[tokio::test]
async fn test_cholesky_reconstruction() {
    let Some(device) = get_test_device_if_gpu_available().await else {
        return;
    };
    // Test that L·Lᵀ = A
    let input_data = vec![4.0, 2.0, 2.0, 3.0];
    let input = Tensor::from_vec_on(input_data.clone(), vec![2, 2], device.clone())
        .await
        .unwrap();

    let (l, l_t) = input.cholesky_with_transpose().unwrap();

    // Compute L·Lᵀ
    let reconstructed = l.matmul(&l_t).unwrap();
    let recon_data = reconstructed.to_vec().unwrap();

    // Should match original matrix
    for (i, (&orig, &recon)) in input_data.iter().zip(recon_data.iter()).enumerate() {
        assert!(
            (orig - recon).abs() < 1e-4,
            "Reconstruction error at index {}: expected {}, got {}",
            i,
            orig,
            recon
        );
    }
}

// =========================================================================
// F64 Tests — Science-grade precision
// =========================================================================

#[tokio::test]
async fn test_cholesky_f64_2x2() {
    let Some(device) = get_test_device_if_f64_gpu_available().await else {
        return;
    };
    // SPD matrix: [[4, 2], [2, 3]]
    // Expected L: [[2, 0], [1, sqrt(2)]]
    let input_data: Vec<f64> = vec![4.0, 2.0, 2.0, 3.0];

    let result = CholeskyF64::execute(device, &input_data, 2).unwrap();

    assert_eq!(result.len(), 4);

    // Check L[0,0] ≈ 2.0
    assert!(
        (result[0] - 2.0).abs() < 1e-12,
        "L[0,0] should be 2.0, got {}",
        result[0]
    );

    // Check L[0,1] ≈ 0.0 (upper triangle)
    assert!(
        result[1].abs() < 1e-12,
        "L[0,1] should be 0.0, got {}",
        result[1]
    );

    // Check L[1,0] ≈ 1.0
    assert!(
        (result[2] - 1.0).abs() < 1e-12,
        "L[1,0] should be 1.0, got {}",
        result[2]
    );

    // Check L[1,1] ≈ sqrt(2)
    let sqrt_2: f64 = std::f64::consts::SQRT_2;
    assert!(
        (result[3] - sqrt_2).abs() < 1e-12,
        "L[1,1] should be sqrt(2)={}, got {}",
        sqrt_2,
        result[3]
    );
}

#[tokio::test]
async fn test_cholesky_f64_reconstruction() {
    let Some(device) = get_test_device_if_f64_gpu_available().await else {
        return;
    };
    // Test that L·Lᵀ = A with f64 precision
    let a: Vec<f64> = vec![4.0, 2.0, 2.0, 3.0];
    let n = 2;

    let l = CholeskyF64::execute(device, &a, n).unwrap();

    // Manual L·Lᵀ multiplication
    let mut reconstruction = vec![0.0f64; 4];
    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0;
            for k in 0..n {
                sum += l[i * n + k] * l[j * n + k]; // L[i,k] * L[j,k] (Lᵀ[k,j] = L[j,k])
            }
            reconstruction[i * n + j] = sum;
        }
    }

    // Should match original with f64 precision
    for (i, (&orig, &recon)) in a.iter().zip(reconstruction.iter()).enumerate() {
        assert!(
            (orig - recon).abs() < 1e-12,
            "f64 reconstruction error at {}: expected {}, got {}",
            i,
            orig,
            recon
        );
    }
}

#[tokio::test]
async fn test_cholesky_f64_3x3() {
    let Some(device) = get_test_device_if_f64_gpu_available().await else {
        return;
    };
    // 3x3 SPD matrix (row-major)
    let a: Vec<f64> = vec![4.0, 2.0, 1.0, 2.0, 3.0, 1.0, 1.0, 1.0, 3.0];
    let n = 3;

    let l = CholeskyF64::execute(device, &a, n).unwrap();

    // Verify lower triangular
    assert!(l[1].abs() < 1e-12); // L[0,1]
    assert!(l[2].abs() < 1e-12); // L[0,2]
    assert!(l[5].abs() < 1e-12); // L[1,2]

    // Verify diagonal is positive
    assert!(l[0] > 0.0);
    assert!(l[4] > 0.0);
    assert!(l[8] > 0.0);

    // Verify L·Lᵀ = A
    let mut recon = vec![0.0f64; 9];
    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0;
            for k in 0..n {
                sum += l[i * n + k] * l[j * n + k];
            }
            recon[i * n + j] = sum;
        }
    }

    for (i, (&orig, &r)) in a.iter().zip(recon.iter()).enumerate() {
        assert!(
            (orig - r).abs() < 1e-10,
            "3x3 f64 reconstruction error at {}: expected {}, got {}",
            i,
            orig,
            r
        );
    }
}

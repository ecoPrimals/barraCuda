// SPDX-License-Identifier: AGPL-3.0-or-later
//! Triangular solve tests

use super::f64::TriangularSolveF64;
use crate::device::test_pool::{
    get_test_device_if_f64_gpu_available, get_test_device_if_gpu_available,
};
use crate::tensor::Tensor;

const TRIANGULAR_SOLVE_F32_SHADER: &str =
    include_str!("../../../shaders/linalg/triangular_solve.wgsl");
const TRIANGULAR_SOLVE_F64_SHADER: &str =
    include_str!("../../../shaders/linalg/triangular_solve_f64.wgsl");

#[test]
fn triangular_solve_f32_shader_source_valid() {
    assert!(!TRIANGULAR_SOLVE_F32_SHADER.is_empty());
    assert!(
        TRIANGULAR_SOLVE_F32_SHADER.contains("fn ")
            || TRIANGULAR_SOLVE_F32_SHADER.contains("@compute")
    );
}

#[test]
fn triangular_solve_f64_shader_source_valid() {
    assert!(!TRIANGULAR_SOLVE_F64_SHADER.is_empty());
    assert!(
        TRIANGULAR_SOLVE_F64_SHADER.contains("fn ")
            || TRIANGULAR_SOLVE_F64_SHADER.contains("@compute")
    );
}

#[tokio::test]
async fn test_forward_substitution_2x2() {
    let Some(device) = get_test_device_if_gpu_available().await else {
        return;
    };
    // Lower triangular matrix L = [[2, 0], [3, 4]]
    // Solve L·x = b where b = [6, 17]
    // Expected: x = [3, 2]
    let l_data = vec![2.0, 0.0, 3.0, 4.0];
    let b_data = vec![6.0, 17.0];

    let l = Tensor::from_vec_on(l_data, vec![2, 2], device.clone())
        .await
        .unwrap();
    let b = Tensor::from_vec_on(b_data, vec![2], device).await.unwrap();

    let x = l.solve_triangular_forward(&b).unwrap();
    let solution = x.to_vec().unwrap();

    assert_eq!(solution.len(), 2);
    assert!(
        (solution[0] - 3.0).abs() < 1e-5,
        "x[0] should be 3.0, got {}",
        solution[0]
    );
    assert!(
        (solution[1] - 2.0).abs() < 1e-5,
        "x[1] should be 2.0, got {}",
        solution[1]
    );
}

#[tokio::test]
async fn test_backward_substitution_2x2() {
    let Some(device) = get_test_device_if_gpu_available().await else {
        return;
    };
    let u_data = vec![2.0, 3.0, 0.0, 4.0];
    let b_data = vec![12.0, 8.0];

    let u = Tensor::from_vec_on(u_data, vec![2, 2], device.clone())
        .await
        .unwrap();
    let b = Tensor::from_vec_on(b_data, vec![2], device).await.unwrap();

    let x = u.solve_triangular_backward(&b).unwrap();
    let solution = x.to_vec().unwrap();

    assert_eq!(solution.len(), 2);
    assert!(
        (solution[0] - 3.0).abs() < 1e-5,
        "x[0] should be 3.0, got {}",
        solution[0]
    );
    assert!(
        (solution[1] - 2.0).abs() < 1e-5,
        "x[1] should be 2.0, got {}",
        solution[1]
    );
}

#[tokio::test]
async fn test_cholesky_solve_pipeline() {
    let Some(device) = get_test_device_if_gpu_available().await else {
        return;
    };
    let a_data = vec![4.0, 2.0, 2.0, 3.0];
    let b_expected = vec![6.0, 5.0];

    let a = Tensor::from_vec_on(a_data.clone(), vec![2, 2], device.clone())
        .await
        .unwrap();
    let b = Tensor::from_vec_on(b_expected.clone(), vec![2], device.clone())
        .await
        .unwrap();

    let l = a.cholesky().unwrap();
    let z = l.solve_triangular_forward(&b).unwrap();
    let l_t = l.transpose().unwrap();
    let x = l_t.solve_triangular_backward(&z).unwrap();

    let a_verify = Tensor::from_vec_on(a_data, vec![2, 2], device)
        .await
        .unwrap();
    let x_2d = x.unsqueeze(1).unwrap();
    let ax = a_verify.matmul(&x_2d).unwrap().squeeze().unwrap();
    let ax_data = ax.to_vec().unwrap();

    for (i, (&expected, &actual)) in b_expected.iter().zip(ax_data.iter()).enumerate() {
        assert!(
            (expected - actual).abs() < 1e-4,
            "A·x verification failed at index {i}: expected {expected}, got {actual}"
        );
    }
}

#[tokio::test]
async fn test_triangular_solve_f64_forward() {
    let Some(device) = get_test_device_if_f64_gpu_available().await else {
        return;
    };
    let l: Vec<f64> = vec![2.0, 0.0, 3.0, 4.0];
    let b: Vec<f64> = vec![6.0, 17.0];

    let x = TriangularSolveF64::forward(device, &l, &b, 2).unwrap();

    assert!(
        (x[0] - 3.0).abs() < 1e-12,
        "x[0] should be 3.0, got {}",
        x[0]
    );
    assert!(
        (x[1] - 2.0).abs() < 1e-12,
        "x[1] should be 2.0, got {}",
        x[1]
    );
}

#[tokio::test]
async fn test_triangular_solve_f64_backward() {
    let Some(device) = get_test_device_if_f64_gpu_available().await else {
        return;
    };
    let u: Vec<f64> = vec![2.0, 3.0, 0.0, 4.0];
    let b: Vec<f64> = vec![12.0, 8.0];

    let x = TriangularSolveF64::backward(device, &u, &b, 2).unwrap();

    assert!(
        (x[0] - 3.0).abs() < 1e-12,
        "x[0] should be 3.0, got {}",
        x[0]
    );
    assert!(
        (x[1] - 2.0).abs() < 1e-12,
        "x[1] should be 2.0, got {}",
        x[1]
    );
}

#[tokio::test]
async fn test_triangular_solve_f64_cholesky_pipeline() {
    let Some(device) = get_test_device_if_f64_gpu_available().await else {
        return;
    };
    let a: Vec<f64> = vec![4.0, 2.0, 2.0, 3.0];
    let b: Vec<f64> = vec![6.0, 5.0];
    let n = 2;

    use crate::ops::linalg::cholesky::CholeskyF64;
    let l = CholeskyF64::execute(device.clone(), &a, n).unwrap();
    let x = TriangularSolveF64::cholesky_solve(device, &l, &b, n).unwrap();

    let ax0 = a[0].mul_add(x[0], a[1] * x[1]);
    let ax1 = a[2].mul_add(x[0], a[3] * x[1]);

    assert!(
        (ax0 - b[0]).abs() < 1e-10,
        "A·x[0] should be {}, got {}",
        b[0],
        ax0
    );
    assert!(
        (ax1 - b[1]).abs() < 1e-10,
        "A·x[1] should be {}, got {}",
        b[1],
        ax1
    );
}

#[tokio::test]
async fn test_triangular_solve_f64_3x3() {
    let Some(device) = get_test_device_if_f64_gpu_available().await else {
        return;
    };
    let l: Vec<f64> = vec![2.0, 0.0, 0.0, 1.0, 3.0, 0.0, 4.0, 2.0, 5.0];
    let b: Vec<f64> = vec![4.0, 5.0, 25.0];

    let x = TriangularSolveF64::forward(device, &l, &b, 3).unwrap();

    assert!(
        (x[0] - 2.0).abs() < 1e-12,
        "x[0] should be 2.0, got {}",
        x[0]
    );
    assert!(
        (x[1] - 1.0).abs() < 1e-12,
        "x[1] should be 1.0, got {}",
        x[1]
    );
    assert!(
        (x[2] - 3.0).abs() < 1e-12,
        "x[2] should be 3.0, got {}",
        x[2]
    );
}

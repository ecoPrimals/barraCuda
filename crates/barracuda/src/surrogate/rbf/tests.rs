// SPDX-License-Identifier: AGPL-3.0-or-later
#![expect(clippy::unwrap_used, reason = "tests")]

use super::*;
use crate::device::WgpuDevice;
use crate::device::test_pool::{
    get_test_device_if_f64_gpu_available_sync, get_test_device_if_gpu_available_sync,
};
use crate::surrogate::kernels::RBFKernel;
use std::sync::Arc;

fn device() -> Option<Arc<WgpuDevice>> {
    get_test_device_if_f64_gpu_available_sync()
}

fn any_device() -> Option<Arc<WgpuDevice>> {
    get_test_device_if_gpu_available_sync()
}

/// Construct a minimal `RBFSurrogate` from parts for testing predict
/// validation paths without needing GPU compute for training.
fn fake_surrogate(device: Arc<WgpuDevice>, n_dim: usize) -> RBFSurrogate {
    let n_train = 3;
    RBFSurrogate::from_parts(
        device,
        RbfTrainingData {
            train_x: vec![0.0; n_train * n_dim],
            train_y: vec![0.0; n_train],
            n_train,
            n_dim,
        },
        RbfTrainedModel {
            weights: vec![1.0; n_train],
            poly_coeffs: vec![0.0; n_dim + 1],
            kernel: RBFKernel::ThinPlateSpline,
            smoothing: 1e-12,
        },
    )
}

#[test]
fn test_rbf_linear_1d() {
    let Some(dev) = device() else { return };
    let x_train = vec![vec![0.0], vec![1.0], vec![2.0]];
    let y_train = vec![0.0, 2.0, 4.0];
    let surrogate =
        RBFSurrogate::train(dev, &x_train, &y_train, RBFKernel::ThinPlateSpline, 1e-12).unwrap();

    // Should interpolate training points exactly
    let y_pred = surrogate.predict(&x_train).unwrap();
    for i in 0..3 {
        assert!(
            (y_pred[i] - y_train[i]).abs() < 1e-10,
            "Failed to interpolate training point {}: pred = {}, true = {}",
            i,
            y_pred[i],
            y_train[i]
        );
    }

    // Test interpolation
    let y_mid = surrogate.predict(&[vec![1.5]]).unwrap();
    assert!(
        (y_mid[0] - 3.0).abs() < 0.1,
        "Poor interpolation at x=1.5: {}",
        y_mid[0]
    );
}

#[test]
fn test_rbf_quadratic_1d() {
    // Approximate y = x²
    let x_train: Vec<Vec<f64>> = (0..5).map(|i| vec![i as f64]).collect();
    let y_train: Vec<f64> = (0..5).map(|i| (i * i) as f64).collect();

    let Some(dev) = device() else { return };
    let surrogate =
        RBFSurrogate::train(dev, &x_train, &y_train, RBFKernel::ThinPlateSpline, 1e-12).unwrap();

    // Should interpolate training points exactly
    let y_pred = surrogate.predict(&x_train).unwrap();
    for i in 0..5 {
        assert!(
            (y_pred[i] - y_train[i]).abs() < 1e-8,
            "Failed at training point x={}: pred = {}, true = {}",
            i,
            y_pred[i],
            y_train[i]
        );
    }
}

#[test]
fn test_rbf_2d() {
    // Simple 2D function: f(x,y) = x + y
    let x_train = vec![
        vec![0.0, 0.0],
        vec![1.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 1.0],
    ];
    let y_train = vec![0.0, 1.0, 1.0, 2.0];

    let Some(dev) = device() else { return };
    let surrogate =
        RBFSurrogate::train(dev, &x_train, &y_train, RBFKernel::ThinPlateSpline, 1e-12).unwrap();

    // Test center point
    let y_center = surrogate.predict(&[vec![0.5, 0.5]]).unwrap();
    assert!(
        (y_center[0] - 1.0).abs() < 0.1,
        "Poor interpolation at center: {}",
        y_center[0]
    );
}

#[test]
fn test_rbf_gaussian_kernel() {
    let x_train = vec![vec![0.0], vec![1.0], vec![2.0]];
    let y_train = vec![0.0, 1.0, 0.0]; // Peak at x=1

    let Some(dev) = device() else { return };
    let surrogate = RBFSurrogate::train(
        dev,
        &x_train,
        &y_train,
        RBFKernel::Gaussian { epsilon: 1.0 },
        1e-12,
    )
    .unwrap();

    // Should interpolate training points
    let y_pred = surrogate.predict(&x_train).unwrap();
    for i in 0..3 {
        assert!((y_pred[i] - y_train[i]).abs() < 1e-8);
    }
}

#[test]
fn test_rbf_empty_training_data() {
    let Some(dev) = any_device() else { return };
    let result = RBFSurrogate::train(dev, &[], &[], RBFKernel::ThinPlateSpline, 1e-12);
    assert!(result.is_err());
}

#[test]
fn test_rbf_mismatched_lengths() {
    let Some(dev) = any_device() else { return };
    let x_train = vec![vec![0.0], vec![1.0]];
    let y_train = vec![0.0, 1.0, 2.0];
    let result = RBFSurrogate::train(dev, &x_train, &y_train, RBFKernel::ThinPlateSpline, 1e-12);
    assert!(result.is_err());
}

#[test]
fn test_loo_cv_rmse() {
    // With smoothing, LOO-CV should give meaningful results
    let x_train = vec![vec![0.0], vec![0.5], vec![1.0], vec![1.5], vec![2.0]];
    // Noisy linear function: y ≈ 2x
    let y_train = vec![0.1, 1.1, 1.9, 3.1, 3.9];

    // Use moderate smoothing so LOO-CV is defined
    let Some(dev) = device() else { return };
    let surrogate =
        RBFSurrogate::train(dev, &x_train, &y_train, RBFKernel::ThinPlateSpline, 1e-6).unwrap();

    let loo_rmse = surrogate.loo_cv_rmse().unwrap();

    // Should be non-negative
    assert!(loo_rmse >= 0.0);

    // Should be small since data is nearly linear
    assert!(loo_rmse < 1.0, "LOO-CV RMSE too large: {loo_rmse}");
}

#[test]
fn test_loo_cv_errors() {
    let x_train = vec![vec![0.0], vec![1.0], vec![2.0]];
    let y_train = vec![0.0, 1.0, 4.0];

    let Some(dev) = device() else { return };
    let surrogate =
        RBFSurrogate::train(dev, &x_train, &y_train, RBFKernel::ThinPlateSpline, 1e-4).unwrap();

    let errors = surrogate.loo_cv_errors().unwrap();

    // Should have one error per training point
    assert_eq!(errors.len(), 3);

    // Errors should be finite
    for e in &errors {
        assert!(e.is_finite(), "Non-finite LOO error: {e}");
    }
}

#[test]
fn test_loo_cv_with_exact_interpolation() {
    // With very small smoothing (exact interpolation), LOO residuals
    // may be near zero or undefined (H_ii ≈ 1)
    let x_train = vec![vec![0.0], vec![1.0], vec![2.0]];
    let y_train = vec![0.0, 1.0, 4.0];

    let Some(dev) = device() else { return };
    let surrogate =
        RBFSurrogate::train(dev, &x_train, &y_train, RBFKernel::ThinPlateSpline, 1e-12).unwrap();

    // Should not panic
    let _ = surrogate.loo_cv_rmse();
}

#[test]
fn test_rbf_accessors() {
    // Need at least n_dim + 1 points for polynomial augmentation
    let x_train = vec![
        vec![0.0, 0.0],
        vec![1.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 1.0],
    ];
    let y_train = vec![0.0, 1.0, 1.0, 2.0];

    let Some(dev) = device() else { return };
    let surrogate =
        RBFSurrogate::train(dev, &x_train, &y_train, RBFKernel::ThinPlateSpline, 1e-12).unwrap();

    assert_eq!(surrogate.n_train(), 4);
    assert_eq!(surrogate.n_dim(), 2);
}

#[test]
fn test_loo_cv_hat_diagonal_correct() {
    // Test that H_ii < 1 when smoothing > 0 (the bug was H_ii = 1 always)
    // Use non-linear noisy data that won't be perfectly fit
    let x_train = vec![
        vec![0.0],
        vec![0.5],
        vec![1.0],
        vec![1.5],
        vec![2.0],
        vec![2.5],
        vec![3.0],
    ];
    // Noisy quadratic: y ≈ x² + noise
    let y_train = vec![0.1, 0.35, 0.9, 2.3, 4.1, 6.2, 9.1];

    let Some(dev) = device() else { return };
    let surrogate = RBFSurrogate::train(
        dev,
        &x_train,
        &y_train,
        RBFKernel::Gaussian { epsilon: 1.0 },
        0.5,
    )
    .unwrap();

    let errors = surrogate.loo_cv_errors().unwrap();

    // With the CORRECT hat matrix and high smoothing, predictions != targets
    // So residuals (y - ŷ) should be non-zero
    // And with H_ii < 1, the LOO residuals should also be non-zero

    // First verify we have some residuals (smoothed predictions differ from targets)
    let train_points: Vec<Vec<f64>> = x_train;
    let predictions = surrogate.predict(&train_points).unwrap();
    let max_residual: f64 = predictions
        .iter()
        .zip(y_train.iter())
        .map(|(p, y)| (p - y).abs())
        .fold(0.0, f64::max);

    // With high smoothing, the fit should be imperfect
    assert!(
        max_residual > 0.01,
        "Expected imperfect fit with high smoothing. Max residual: {max_residual}"
    );

    // LOO errors should be non-zero (the main test)
    let max_abs_error = errors.iter().map(|e| e.abs()).fold(0.0, f64::max);
    assert!(
        max_abs_error > 0.001,
        "LOO errors should be non-zero with smoothing. Max error: {max_abs_error}"
    );
}

#[test]
fn test_loo_cv_smoothing_effect() {
    // More smoothing should generally increase LOO-CV RMSE (underfitting)
    // Less smoothing should decrease LOO-CV RMSE (better fit, but risk overfitting)
    let x_train: Vec<Vec<f64>> = (0..10).map(|i| vec![i as f64 * 0.1]).collect();
    let y_train: Vec<f64> = x_train
        .iter()
        .map(|x| 2.0f64.mul_add(x[0], 0.1 * (x[0] * 10.0).sin()))
        .collect();

    let Some(dev) = device() else { return };
    let kernel = RBFKernel::Gaussian { epsilon: 1.0 };
    let surrogate_low = RBFSurrogate::train(dev.clone(), &x_train, &y_train, kernel, 1e-4).unwrap();
    let surrogate_high = RBFSurrogate::train(dev, &x_train, &y_train, kernel, 0.5).unwrap();

    let rmse_low = surrogate_low.loo_cv_rmse().unwrap();
    let rmse_high = surrogate_high.loo_cv_rmse().unwrap();

    // Both should be finite and positive
    assert!(rmse_low.is_finite() && rmse_low > 0.0);
    assert!(rmse_high.is_finite() && rmse_high > 0.0);

    // High smoothing should generally give higher RMSE (underfitting)
    // But this isn't strictly monotonic, so we just verify they're different
    assert!(
        (rmse_low - rmse_high).abs() > 1e-6,
        "LOO-CV should be sensitive to smoothing. low={rmse_low}, high={rmse_high}"
    );
}

// === Tests that work on any GPU (including llvmpipe) ===
// These exercise validation paths that return before GPU compute.

#[test]
fn test_predict_empty_eval_returns_empty() {
    let Some(dev) = any_device() else { return };
    let surrogate = fake_surrogate(dev, 2);
    let result = surrogate.predict(&[]).unwrap();
    assert!(result.is_empty());
}

#[test]
fn test_predict_dimension_mismatch() {
    let Some(dev) = any_device() else { return };
    let surrogate = fake_surrogate(dev, 2);
    let result = surrogate.predict(&[vec![1.0, 2.0, 3.0]]);
    assert!(result.is_err());
}

#[test]
fn test_accessors_from_parts() {
    let Some(dev) = any_device() else { return };
    let surrogate = fake_surrogate(dev, 3);
    assert_eq!(surrogate.n_train(), 3);
    assert_eq!(surrogate.n_dim(), 3);
}

#[test]
fn test_loo_cv_empty_training_data() {
    let Some(dev) = any_device() else { return };
    let surrogate = RBFSurrogate::from_parts(
        dev,
        RbfTrainingData {
            train_x: vec![],
            train_y: vec![],
            n_train: 0,
            n_dim: 1,
        },
        RbfTrainedModel {
            weights: vec![],
            poly_coeffs: vec![0.0, 0.0],
            kernel: RBFKernel::ThinPlateSpline,
            smoothing: 1e-6,
        },
    );
    let errors = surrogate.loo_cv_errors().unwrap();
    assert!(errors.is_empty());
}

#[test]
fn test_loo_optimal_smoothing_empty_grid() {
    let Some(dev) = any_device() else { return };
    let x = vec![vec![0.0], vec![1.0], vec![2.0]];
    let y = vec![0.0, 1.0, 4.0];
    let result = loo_cv_optimal_smoothing(dev, &x, &y, RBFKernel::ThinPlateSpline, Some(&[]));
    assert!(result.is_err());
}

#[test]
fn test_loo_optimal_smoothing() {
    let Some(dev) = device() else { return };
    let x: Vec<Vec<f64>> = (0..8).map(|i| vec![i as f64 * 0.5]).collect();
    let y: Vec<f64> = x.iter().map(|v| 2.0f64.mul_add(v[0], 0.1)).collect();
    let result = loo_cv_optimal_smoothing(dev, &x, &y, RBFKernel::ThinPlateSpline, None).unwrap();
    assert!(result.smoothing > 0.0);
    assert!(result.rmse >= 0.0);
    assert!(!result.grid_results.is_empty());
}

#[test]
fn test_loo_optimal_smoothing_custom_grid() {
    let Some(dev) = device() else { return };
    let x = vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0]];
    let y = vec![0.0, 1.0, 4.0, 9.0];
    let grid = [1e-8, 1e-4, 1e-2];
    let result =
        loo_cv_optimal_smoothing(dev, &x, &y, RBFKernel::ThinPlateSpline, Some(&grid)).unwrap();
    assert!(result.grid_results.len() <= 3);
    assert!(result.rmse.is_finite());
}

#[test]
fn test_rbf_training_data_struct() {
    let data = RbfTrainingData {
        train_x: vec![1.0, 2.0, 3.0, 4.0],
        train_y: vec![10.0, 20.0],
        n_train: 2,
        n_dim: 2,
    };
    assert_eq!(data.n_train, 2);
    assert_eq!(data.n_dim, 2);
    assert_eq!(data.train_x.len(), 4);
}

#[test]
fn test_rbf_trained_model_struct() {
    let model = RbfTrainedModel {
        weights: vec![1.0, 2.0],
        poly_coeffs: vec![0.5, 0.3, 0.1],
        kernel: RBFKernel::Cubic,
        smoothing: 1e-6,
    };
    assert_eq!(model.weights.len(), 2);
    assert_eq!(model.poly_coeffs.len(), 3);
    assert_eq!(model.kernel.name(), "Cubic");
}

#[test]
fn test_loo_smoothing_struct() {
    let s = LooSmoothing {
        smoothing: 1e-4,
        rmse: 0.01,
        grid_results: vec![(1e-6, 0.1), (1e-4, 0.01), (1e-2, 0.05)],
    };
    assert_eq!(s.smoothing, 1e-4);
    assert_eq!(s.rmse, 0.01);
    assert_eq!(s.grid_results.len(), 3);
}

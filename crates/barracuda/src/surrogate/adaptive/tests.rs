// SPDX-License-Identifier: AGPL-3.0-or-later
//! Tests for adaptive RBF surrogate training

#![expect(clippy::unwrap_used, reason = "tests")]

use super::*;
use crate::device::test_pool::{
    get_test_device_if_f64_gpu_available_sync, get_test_device_if_gpu_available_sync,
};

#[test]
fn test_adaptive_config_default() {
    let config = AdaptiveConfig::default();
    assert_eq!(config.f32_threshold, 200);
    assert!(!config.force_f64);
    assert!(config.parallel);
    assert!(config.prefer_gpu);
}

#[test]
fn test_adaptive_config_exact() {
    let config = AdaptiveConfig::exact();
    assert!(config.force_f64);
    assert!(!config.prefer_gpu);
}

#[test]
fn test_adaptive_config_threshold() {
    let config = AdaptiveConfig::with_threshold(50);
    assert_eq!(config.f32_threshold, 50);
    assert!(!config.force_f64);
}

#[test]
fn test_adaptive_config_cpu_only() {
    let config = AdaptiveConfig::cpu_only();
    assert!(!config.prefer_gpu);
    assert!(!config.force_f64);
}

// === CPU distance function tests (no GPU needed) ===

#[test]
fn test_compute_distances_f64_identity() {
    let x = vec![0.0, 1.0, 2.0];
    let d = compute_distances_f64(&x, &x, 3, 3, 1);
    assert_eq!(d.len(), 9);
    for i in 0..3 {
        assert!((d[i * 3 + i]).abs() < 1e-14, "self-distance should be 0");
    }
    assert!((d[1] - 1.0).abs() < 1e-14);
    assert!((d[2] - 2.0).abs() < 1e-14);
    assert!((d[5] - 1.0).abs() < 1e-14);
}

#[test]
fn test_compute_distances_f64_2d() {
    let x = vec![0.0, 0.0, 3.0, 4.0];
    let d = compute_distances_f64(&x, &x, 2, 2, 2);
    assert!((d[0]).abs() < 1e-14);
    assert!((d[1] - 5.0).abs() < 1e-14, "3-4-5 triangle");
    assert!((d[2] - 5.0).abs() < 1e-14);
    assert!((d[3]).abs() < 1e-14);
}

#[test]
fn test_compute_distances_f64_asymmetric() {
    let x1 = vec![0.0];
    let x2 = vec![3.0, 5.0];
    let d = compute_distances_f64(&x1, &x2, 1, 2, 1);
    assert_eq!(d.len(), 2);
    assert!((d[0] - 3.0).abs() < 1e-14);
    assert!((d[1] - 5.0).abs() < 1e-14);
}

#[test]
fn test_compute_distances_f32_promoted_matches_f64() {
    let n = 10;
    let x: Vec<f64> = (0..n).map(|i| i as f64 * 0.7).collect();
    let d_f64 = compute_distances_f64(&x, &x, n, n, 1);
    let d_f32 = compute_distances_f32_promoted(&x, &x, n, n, 1);
    assert_eq!(d_f64.len(), d_f32.len());
    let max_err = d_f64
        .iter()
        .zip(d_f32.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    assert!(max_err < 1e-5, "f32/f64 max distance error: {max_err}");
}

#[test]
fn test_compute_distances_f32_promoted_zero_distance() {
    let x = vec![1.0, 2.0, 3.0];
    let d = compute_distances_f32_promoted(&x, &x, 3, 3, 1);
    for i in 0..3 {
        assert!(d[i * 3 + i].abs() < 1e-6, "self-distance should be ~0");
    }
}

#[test]
fn test_compute_distances_f64_single_point() {
    let x = vec![42.0, -7.0];
    let d = compute_distances_f64(&x, &x, 1, 1, 2);
    assert_eq!(d.len(), 1);
    assert!(d[0].abs() < 1e-14);
}

#[test]
fn test_compute_distances_f32_promoted_high_dim() {
    let n_dim = 10;
    let x1: Vec<f64> = (0..n_dim).map(|i| i as f64).collect();
    let x2: Vec<f64> = (0..n_dim).map(|i| (i as f64) + 1.0).collect();
    let d = compute_distances_f32_promoted(&x1, &x2, 1, 1, n_dim);
    let expected = (n_dim as f64).sqrt();
    assert!(
        (d[0] - expected).abs() < 1e-3,
        "10D unit offset: got {}, expected {expected}",
        d[0]
    );
}

#[test]
fn test_diagnostics_debug_clone() {
    let d = TrainingDiagnostics {
        used_f32_distances: true,
        used_gpu: false,
        n_train: 10,
        n_dim: 2,
        system_size: 13,
        max_distance_error: Some(0.001),
    };
    let d2 = d.clone();
    assert_eq!(d2.n_train, 10);
    assert!(d2.max_distance_error.is_some());
    let _ = format!("{d:?}");
}

#[test]
fn test_adaptive_config_clone_debug() {
    let c = AdaptiveConfig::default();
    let c2 = c.clone();
    assert_eq!(c2.f32_threshold, 200);
    let _ = format!("{c:?}");
}

// === Error-path tests using any GPU (including llvmpipe) ===

#[test]
fn test_train_adaptive_empty_data_error() {
    let Some(device) = get_test_device_if_gpu_available_sync() else {
        return;
    };
    let config = AdaptiveConfig::default();
    assert!(train_adaptive(device, &[], &[], RBFKernel::ThinPlateSpline, 1e-12, &config).is_err());
}

#[test]
fn test_train_adaptive_length_mismatch_error() {
    let Some(device) = get_test_device_if_gpu_available_sync() else {
        return;
    };
    let config = AdaptiveConfig::default();
    let result = train_adaptive(
        device,
        &[vec![0.0], vec![1.0]],
        &[0.0],
        RBFKernel::ThinPlateSpline,
        1e-12,
        &config,
    );
    assert!(result.is_err());
}

#[test]
fn test_train_with_validation_empty_data_error() {
    let Some(device) = get_test_device_if_gpu_available_sync() else {
        return;
    };
    assert!(train_with_validation(device, &[], &[], RBFKernel::ThinPlateSpline, 1e-12).is_err());
}

#[test]
fn test_train_with_validation_length_mismatch_error() {
    let Some(device) = get_test_device_if_gpu_available_sync() else {
        return;
    };
    let result = train_with_validation(
        device,
        &[vec![0.0]],
        &[0.0, 1.0],
        RBFKernel::ThinPlateSpline,
        1e-12,
    );
    assert!(result.is_err());
}

#[test]
fn test_train_adaptive_small_dataset() {
    let Some(device) = get_test_device_if_f64_gpu_available_sync() else {
        return;
    };
    let x_train: Vec<Vec<f64>> = (0..5).map(|i| vec![i as f64]).collect();
    let y_train: Vec<f64> = x_train.iter().map(|x| x[0] * x[0]).collect();
    let config = AdaptiveConfig::default();
    let (surrogate, diag) = train_adaptive(
        device,
        &x_train,
        &y_train,
        RBFKernel::ThinPlateSpline,
        1e-12,
        &config,
    )
    .unwrap();

    assert!(!diag.used_f32_distances);
    assert_eq!(diag.n_train, 5);
    assert_eq!(diag.n_dim, 1);
    assert_eq!(diag.system_size, 7); // 5 + 1 + 1

    // Should interpolate training points
    let y_pred = surrogate.predict(&x_train).unwrap();
    for i in 0..5 {
        assert!(
            (y_pred[i] - y_train[i]).abs() < 1e-6,
            "Bad interpolation at {}: {} vs {}",
            i,
            y_pred[i],
            y_train[i]
        );
    }
}

#[test]
fn test_train_adaptive_uses_f32_above_threshold() {
    let Some(device) = get_test_device_if_f64_gpu_available_sync() else {
        return;
    };
    let n = 10;
    let x_train: Vec<Vec<f64>> = (0..n).map(|i| vec![i as f64 / n as f64]).collect();
    let y_train: Vec<f64> = x_train
        .iter()
        .map(|x| (x[0] * std::f64::consts::PI).sin())
        .collect();
    let config = AdaptiveConfig::with_threshold(5);
    let (surrogate, diag) = train_adaptive(
        device,
        &x_train,
        &y_train,
        RBFKernel::ThinPlateSpline,
        1e-10,
        &config,
    )
    .unwrap();

    assert!(diag.used_f32_distances);
    assert_eq!(diag.n_train, n);

    // Should still interpolate reasonably well
    let y_pred = surrogate.predict(&x_train).unwrap();
    for i in 0..n {
        assert!(
            (y_pred[i] - y_train[i]).abs() < 0.1,
            "Bad f32-path interpolation at {}: {} vs {}",
            i,
            y_pred[i],
            y_train[i]
        );
    }
}

#[test]
fn test_train_adaptive_force_f64() {
    let Some(device) = get_test_device_if_f64_gpu_available_sync() else {
        return;
    };
    let n = 10;
    let x_train: Vec<Vec<f64>> = (0..n).map(|i| vec![i as f64]).collect();
    let y_train: Vec<f64> = x_train.iter().map(|x| x[0]).collect();
    let config = AdaptiveConfig {
        f32_threshold: 5,
        force_f64: true,
        parallel: true,
        prefer_gpu: false,
    };
    let (_, diag) = train_adaptive(
        device,
        &x_train,
        &y_train,
        RBFKernel::ThinPlateSpline,
        1e-12,
        &config,
    )
    .unwrap();

    assert!(!diag.used_f32_distances); // f64 forced
}

#[test]
fn test_train_with_validation() {
    let Some(device) = get_test_device_if_f64_gpu_available_sync() else {
        return;
    };
    let x_train = vec![vec![0.0], vec![1.0], vec![2.0], vec![3.0], vec![4.0]];
    let y_train = vec![0.0, 1.0, 4.0, 9.0, 16.0];
    let (surrogate, diag) = train_with_validation(
        device,
        &x_train,
        &y_train,
        RBFKernel::ThinPlateSpline,
        1e-12,
    )
    .unwrap();

    assert!(!diag.used_f32_distances);
    assert!(diag.max_distance_error.is_some());

    // f32 vs f64 distance error should be very small for these values
    let max_err = diag.max_distance_error.unwrap();
    assert!(
        max_err < 1e-4,
        "f32/f64 distance error too large: {max_err}"
    );

    // Surrogate should work
    let y_pred = surrogate.predict(&x_train).unwrap();
    for i in 0..5 {
        assert!((y_pred[i] - y_train[i]).abs() < 1e-6);
    }
}

#[test]
fn test_f32_vs_f64_distances_accuracy() {
    // Verify that f32 distances are close to f64 for typical scientific data
    let n = 20;
    let n_dim = 3;
    let mut train_x = Vec::with_capacity(n * n_dim);
    for i in 0..n {
        for d in 0..n_dim {
            train_x.push((i as f64).mul_add(0.1, d as f64 * 0.3));
        }
    }

    let d_f64 = super::compute_distances_f64(&train_x, &train_x, n, n, n_dim);
    let d_f32 = super::compute_distances_f32_promoted(&train_x, &train_x, n, n, n_dim);

    let max_abs_error = d_f64
        .iter()
        .zip(d_f32.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);

    let max_rel_error = d_f64
        .iter()
        .zip(d_f32.iter())
        .filter(|(a, _)| **a > 1e-10)
        .map(|(a, b)| (a - b).abs() / a)
        .fold(0.0_f64, f64::max);

    assert!(
        max_abs_error < 1e-3,
        "Max absolute distance error: {max_abs_error}"
    );
    assert!(
        max_rel_error < 1e-5,
        "Max relative distance error: {max_rel_error}"
    );
}

#[test]
fn test_adaptive_2d_function() {
    let Some(device) = get_test_device_if_f64_gpu_available_sync() else {
        return;
    };
    let x_train = vec![
        vec![0.0, 0.0],
        vec![1.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 1.0],
        vec![0.5, 0.5],
    ];
    let y_train: Vec<f64> = x_train
        .iter()
        .map(|x: &Vec<f64>| x[0].mul_add(x[0], x[1] * x[1]))
        .collect();
    let config = AdaptiveConfig::default();
    let (surrogate, diag) = train_adaptive(
        device,
        &x_train,
        &y_train,
        RBFKernel::ThinPlateSpline,
        1e-12,
        &config,
    )
    .unwrap();

    assert_eq!(diag.n_dim, 2);

    // Test at center (should interpolate exactly)
    let y_center = surrogate.predict(&[vec![0.5, 0.5]]).unwrap();
    assert!(
        (y_center[0] - 0.5).abs() < 0.1,
        "2D interpolation error: {}",
        y_center[0]
    );
}

#[test]
fn test_adaptive_errors() {
    let Some(device) = get_test_device_if_f64_gpu_available_sync() else {
        return;
    };
    let config = AdaptiveConfig::default();
    assert!(
        train_adaptive(
            device.clone(),
            &[],
            &[],
            RBFKernel::ThinPlateSpline,
            1e-12,
            &config
        )
        .is_err()
    );
    assert!(
        train_adaptive(
            device.clone(),
            &[vec![0.0], vec![1.0]],
            &[0.0],
            RBFKernel::ThinPlateSpline,
            1e-12,
            &config
        )
        .is_err()
    );
    assert!(train_with_validation(device, &[], &[], RBFKernel::ThinPlateSpline, 1e-12).is_err());
}

#[test]
fn test_adaptive_gaussian_kernel() {
    let Some(device) = get_test_device_if_f64_gpu_available_sync() else {
        return;
    };
    let x_train: Vec<Vec<f64>> = (0..8).map(|i| vec![i as f64 * 0.5]).collect();
    let y_train: Vec<f64> = x_train.iter().map(|x| (-x[0] * x[0]).exp()).collect();
    let config = AdaptiveConfig::with_threshold(5);
    let (surrogate, diag) = train_adaptive(
        device,
        &x_train,
        &y_train,
        RBFKernel::Gaussian { epsilon: 1.0 },
        1e-10,
        &config,
    )
    .unwrap();

    assert!(diag.used_f32_distances); // n=8 >= threshold=5

    // Should interpolate training data
    let y_pred = surrogate.predict(&x_train).unwrap();
    for i in 0..8 {
        assert!(
            (y_pred[i] - y_train[i]).abs() < 0.01,
            "Gaussian kernel interpolation failed at {}: {} vs {}",
            i,
            y_pred[i],
            y_train[i]
        );
    }
}

#[test]
fn test_diagnostics_fields() {
    let Some(device) = get_test_device_if_f64_gpu_available_sync() else {
        return;
    };
    let x_train = vec![
        vec![0.0, 0.0],
        vec![1.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 1.0],
    ];
    let y_train = vec![0.0, 1.0, 1.0, 2.0];
    let config = AdaptiveConfig::default();
    let (_, diag) = train_adaptive(
        device,
        &x_train,
        &y_train,
        RBFKernel::ThinPlateSpline,
        1e-12,
        &config,
    )
    .unwrap();

    assert_eq!(diag.n_train, 4);
    assert_eq!(diag.n_dim, 2);
    assert_eq!(diag.system_size, 7); // 4 + 2 + 1
    assert!(diag.max_distance_error.is_none()); // Not validation mode
    assert!(!diag.used_gpu); // CPU path
}

// GPU tests require async runtime and f64-capable GPU
mod gpu_tests {
    use super::*;
    use crate::device::test_pool::get_test_device_if_f64_gpu_available;

    #[tokio::test]
    async fn test_train_adaptive_gpu_basic() {
        let Some(device) = get_test_device_if_f64_gpu_available().await else {
            return;
        };

        let x_train: Vec<Vec<f64>> = (0..10).map(|i| vec![i as f64 / 10.0]).collect();
        let y_train: Vec<f64> = x_train.iter().map(|x| x[0] * x[0]).collect();

        let (surrogate, diag) = train_adaptive_gpu(
            &x_train,
            &y_train,
            RBFKernel::ThinPlateSpline,
            1e-10,
            device,
        )
        .await
        .unwrap();

        assert!(diag.used_gpu);
        assert!(diag.used_f32_distances);
        assert_eq!(diag.n_train, 10);
        assert_eq!(diag.n_dim, 1);

        let y_pred = surrogate.predict(&x_train).unwrap();
        for i in 0..10 {
            assert!(
                (y_pred[i] - y_train[i]).abs() < 0.1,
                "GPU interpolation error at {}: {} vs {}",
                i,
                y_pred[i],
                y_train[i]
            );
        }
    }

    #[tokio::test]
    async fn test_train_adaptive_gpu_2d() {
        let Some(device) = get_test_device_if_f64_gpu_available().await else {
            return;
        };

        let x_train = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
            vec![0.5, 0.5],
            vec![0.25, 0.75],
            vec![0.75, 0.25],
        ];
        let y_train: Vec<f64> = x_train
            .iter()
            .map(|x: &Vec<f64>| x[0].mul_add(x[0], x[1] * x[1]))
            .collect();

        let (surrogate, diag) = train_adaptive_gpu(
            &x_train,
            &y_train,
            RBFKernel::Gaussian { epsilon: 1.0 },
            1e-6,
            device,
        )
        .await
        .unwrap();

        assert!(diag.used_gpu);
        assert_eq!(diag.n_train, 7);
        assert_eq!(diag.n_dim, 2);

        let y_pred = surrogate.predict(&[vec![0.5, 0.5]]).unwrap();
        let expected = 0.5f64.mul_add(0.5, 0.5 * 0.5);
        assert!(
            (y_pred[0] - expected).abs() < 0.5,
            "GPU 2D interpolation error: {} vs {}",
            y_pred[0],
            expected
        );
    }

    #[tokio::test]
    async fn test_train_adaptive_gpu_larger_dataset() {
        let Some(device) = get_test_device_if_f64_gpu_available().await else {
            return;
        };

        let n_per_dim = 7;
        let n_dim = 2;
        let mut x_train = Vec::new();
        for i in 0..n_per_dim {
            for j in 0..n_per_dim {
                x_train.push(vec![
                    i as f64 / (n_per_dim - 1) as f64,
                    j as f64 / (n_per_dim - 1) as f64,
                ]);
            }
        }
        let n = x_train.len();
        let y_train: Vec<f64> = x_train
            .iter()
            .map(|x| x.iter().map(|v| v * v).sum::<f64>())
            .collect();

        let (surrogate, diag) = train_adaptive_gpu(
            &x_train,
            &y_train,
            RBFKernel::Gaussian { epsilon: 2.0 },
            1e-4,
            device,
        )
        .await
        .unwrap();

        assert!(diag.used_gpu);
        assert_eq!(diag.n_train, n);
        assert_eq!(diag.n_dim, n_dim);
        assert_eq!(diag.system_size, n + n_dim + 1);

        let y_pred = surrogate.predict(&x_train[..5]).unwrap();
        assert_eq!(y_pred.len(), 5);
        for (i, &pred) in y_pred.iter().enumerate() {
            assert!(pred.is_finite(), "GPU prediction {i} is not finite: {pred}");
        }
    }

    #[tokio::test]
    async fn test_train_adaptive_gpu_errors() {
        let Some(device) = get_test_device_if_f64_gpu_available().await else {
            return;
        };

        let result =
            train_adaptive_gpu(&[], &[], RBFKernel::ThinPlateSpline, 1e-12, device.clone()).await;
        assert!(result.is_err());

        let result = train_adaptive_gpu(
            &[vec![0.0], vec![1.0]],
            &[0.0],
            RBFKernel::ThinPlateSpline,
            1e-12,
            device,
        )
        .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_train_adaptive_gpu_multiquadric_kernel() {
        let Some(device) = get_test_device_if_f64_gpu_available().await else {
            return;
        };

        let x_train = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
            vec![0.5, 0.5],
        ];
        let y_train: Vec<f64> = x_train.iter().map(|x| x[0] + x[1]).collect();

        let (surrogate, diag) = train_adaptive_gpu(
            &x_train,
            &y_train,
            RBFKernel::Multiquadric { epsilon: 1.0 },
            1e-8,
            device,
        )
        .await
        .unwrap();

        assert!(diag.used_gpu);
        assert_eq!(diag.n_train, 5);
        assert_eq!(diag.n_dim, 2);

        let y_pred = surrogate.predict(&[vec![0.5, 0.5]]).unwrap();
        assert!(y_pred[0].is_finite());
        assert!((y_pred[0] - 1.0).abs() < 0.5);
    }

    #[tokio::test]
    async fn test_train_adaptive_gpu_cubic_kernel() {
        let Some(device) = get_test_device_if_f64_gpu_available().await else {
            return;
        };

        let x_train: Vec<Vec<f64>> = (0..6).map(|i| vec![i as f64 * 0.2]).collect();
        let y_train: Vec<f64> = x_train.iter().map(|x| x[0] * x[0] * x[0]).collect();

        let (surrogate, diag) =
            train_adaptive_gpu(&x_train, &y_train, RBFKernel::Cubic, 1e-10, device)
                .await
                .unwrap();

        assert!(diag.used_gpu);
        assert_eq!(diag.n_train, 6);

        let y_pred = surrogate.predict(&x_train).unwrap();
        for i in 0..6 {
            assert!(
                (y_pred[i] - y_train[i]).abs() < 0.1,
                "Cubic interpolation at {}: {} vs {}",
                i,
                y_pred[i],
                y_train[i]
            );
        }
    }

    #[tokio::test]
    async fn test_train_adaptive_gpu_3d_data() {
        let Some(device) = get_test_device_if_f64_gpu_available().await else {
            return;
        };

        let x_train = vec![
            vec![0.0, 0.0, 0.0],
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
            vec![1.0, 1.0, 1.0],
            vec![0.5, 0.5, 0.5],
        ];
        let y_train: Vec<f64> = x_train
            .iter()
            .map(|x: &Vec<f64>| x[0].mul_add(x[0], x[1].mul_add(x[1], x[2] * x[2])))
            .collect();

        let (surrogate, diag) = train_adaptive_gpu(
            &x_train,
            &y_train,
            RBFKernel::InverseMultiquadric { epsilon: 1.0 },
            1e-8,
            device,
        )
        .await
        .unwrap();

        assert!(diag.used_gpu);
        assert_eq!(diag.n_train, 6);
        assert_eq!(diag.n_dim, 3);
        assert_eq!(diag.system_size, 10); // 6 + 3 + 1

        let y_pred = surrogate.predict(&[vec![0.5, 0.5, 0.5]]).unwrap();
        let expected = 0.75;
        assert!(y_pred[0].is_finite());
        assert!((y_pred[0] - expected).abs() < 0.5);
    }

    #[tokio::test]
    async fn test_train_adaptive_gpu_quintic_kernel() {
        let Some(device) = get_test_device_if_f64_gpu_available().await else {
            return;
        };

        let x_train = vec![vec![0.0], vec![0.5], vec![1.0]];
        let y_train = vec![0.0, 0.125, 1.0]; // x^5 at 0, 0.5, 1

        let (surrogate, diag) =
            train_adaptive_gpu(&x_train, &y_train, RBFKernel::Quintic, 1e-12, device)
                .await
                .unwrap();

        assert!(diag.used_gpu);
        assert_eq!(diag.n_train, 3);

        let y_pred = surrogate.predict(&x_train).unwrap();
        for i in 0..3 {
            assert!(
                (y_pred[i] - y_train[i]).abs() < 0.1,
                "Quintic at {}: {} vs {}",
                i,
                y_pred[i],
                y_train[i]
            );
        }
    }
}

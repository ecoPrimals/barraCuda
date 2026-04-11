// SPDX-License-Identifier: AGPL-3.0-or-later
//! hotSpring Evolution: Mixing and Grid unit tests.
//!
//! `LinearMixer`, `BroydenMixer`, `Gradient1D`, `Gradient2D`, `Laplacian2D`, Cylindrical.

#![expect(clippy::unwrap_used, reason = "tests")]
mod common;

use barracuda::ops::grid::{
    CylindricalGradient, CylindricalLaplacian, Gradient1D, Gradient2D, Laplacian2D,
};
use barracuda::ops::mixing::{BroydenMixer, LinearMixer, MixingParams};

mod mixing_unit {
    use super::*;

    #[tokio::test]
    async fn test_linear_mixer_alpha_half() {
        if !super::common::run_gpu_resilient_async(|| async {
            let device = barracuda::device::test_pool::get_test_device().await;
            let params = MixingParams {
                alpha: 0.5,
                ..Default::default()
            };
            let mixer = LinearMixer::new(device, 1024, params).unwrap();
            let x_old = vec![0.0; 1024];
            let x_computed = vec![2.0; 1024];
            let result = mixer.mix(&x_old, &x_computed).await.unwrap();
            for (i, val) in result.iter().enumerate() {
                assert!(
                    (val - 1.0).abs() < 1e-10,
                    "At index {i}: expected 1.0, got {val}"
                );
            }
        }) {
            return;
        }
    }

    #[tokio::test]
    async fn test_linear_mixer_alpha_conservative() {
        if !super::common::run_gpu_resilient_async(|| async {
            let device = barracuda::device::test_pool::get_test_device().await;
            let params = MixingParams {
                alpha: 0.3,
                ..Default::default()
            };
            let mixer = LinearMixer::new(device, 512, params).unwrap();
            let x_old = vec![10.0; 512];
            let x_computed = vec![20.0; 512];
            let result = mixer.mix(&x_old, &x_computed).await.unwrap();
            for (i, val) in result.iter().enumerate() {
                assert!(
                    (val - 13.0).abs() < 1e-10,
                    "At index {i}: expected 13.0, got {val}"
                );
            }
        }) {
            return;
        }
    }

    #[tokio::test]
    async fn test_linear_mixer_alpha_one() {
        if !super::common::run_gpu_resilient_async(|| async {
            let device = barracuda::device::test_pool::get_test_device().await;
            let params = MixingParams {
                alpha: 1.0,
                ..Default::default()
            };
            let mixer = LinearMixer::new(device, 256, params).unwrap();
            let x_old = vec![1.0; 256];
            let x_computed = vec![5.0; 256];
            let result = mixer.mix(&x_old, &x_computed).await.unwrap();
            for (i, val) in result.iter().enumerate() {
                assert!(
                    (val - 5.0).abs() < 1e-10,
                    "At index {i}: expected 5.0, got {val}"
                );
            }
        }) {
            return;
        }
    }

    #[tokio::test]
    async fn test_linear_mixer_alpha_zero() {
        if !super::common::run_gpu_resilient_async(|| async {
            let device = barracuda::device::test_pool::get_test_device().await;
            let params = MixingParams {
                alpha: 0.0,
                ..Default::default()
            };
            let mixer = LinearMixer::new(device, 128, params).unwrap();
            let x_old = vec![std::f64::consts::PI; 128];
            let x_computed = vec![99.0; 128];
            let result = mixer.mix(&x_old, &x_computed).await.unwrap();
            for (i, val) in result.iter().enumerate() {
                assert!(
                    (val - std::f64::consts::PI).abs() < 1e-10,
                    "At index {i}: expected PI, got {val}"
                );
            }
        }) {
            return;
        }
    }

    #[tokio::test]
    async fn test_linear_mixer_varying_values() {
        if !super::common::run_gpu_resilient_async(|| async {
            let device = barracuda::device::test_pool::get_test_device().await;
            let params = MixingParams {
                alpha: 0.4,
                ..Default::default()
            };
            let n = 1000;
            let mixer = LinearMixer::new(device, n, params).unwrap();
            let x_old: Vec<f64> = (0..n).map(|i| i as f64).collect();
            let x_computed: Vec<f64> = (0..n).map(|i| (i * 2) as f64).collect();
            let result = mixer.mix(&x_old, &x_computed).await.unwrap();
            for (i, val) in result.iter().enumerate() {
                let expected = 1.4 * i as f64;
                assert!(
                    (val - expected).abs() < 1e-9,
                    "At index {i}: expected {expected}, got {val}"
                );
            }
        }) {
            return;
        }
    }

    #[tokio::test]
    async fn test_broyden_mixer_creation() {
        if !super::common::run_gpu_resilient_async(|| async {
            let device = barracuda::device::test_pool::get_test_device().await;
            let params = MixingParams {
                alpha: 0.4,
                n_warmup: 3,
                ..Default::default()
            };
            let mixer = BroydenMixer::new(device, 1024, 5, params);
            assert!(mixer.is_ok(), "BroydenMixer creation should succeed");
        }) {
            return;
        }
    }

    #[tokio::test]
    async fn test_broyden_mixer_warmup() {
        if !super::common::run_gpu_resilient_async(|| async {
            let device = barracuda::device::test_pool::get_test_device().await;
            let params = MixingParams {
                alpha: 0.5,
                n_warmup: 5,
                ..Default::default()
            };
            let mut mixer = BroydenMixer::new(device, 100, 5, params).unwrap();
            let x_old = vec![1.0; 100];
            let x_computed = vec![2.0; 100];
            for _ in 0..3 {
                let result = mixer.mix(&x_old, &x_computed).await.unwrap();
                for val in &result {
                    assert!((val - 1.5).abs() < 1e-10);
                }
            }
        }) {
            return;
        }
    }

    #[tokio::test]
    async fn test_broyden_mixer_reset() {
        if !super::common::run_gpu_resilient_async(|| async {
            let device = barracuda::device::test_pool::get_test_device().await;
            let params = MixingParams::default();
            let mut mixer = BroydenMixer::new(device, 50, 3, params).unwrap();
            let x_old = vec![1.0; 50];
            let x_computed = vec![2.0; 50];
            for _ in 0..5 {
                let _ = mixer.mix(&x_old, &x_computed).await;
            }
            mixer.reset();
            let result = mixer.mix(&x_old, &x_computed).await.unwrap();
            for val in &result {
                assert!((val - 1.4).abs() < 1e-10);
            }
        }) {
            return;
        }
    }
}

mod grid_unit {
    use super::*;

    #[tokio::test]
    async fn test_gradient_1d_linear() {
        if !super::common::run_gpu_resilient_async(|| async {
            let device = barracuda::device::test_pool::get_test_device().await;
            let n = 100;
            let dx = 0.1;
            let grad = Gradient1D::new(device, n, dx).unwrap();
            let input: Vec<f64> = (0..n).map(|i| i as f64 * dx).collect();
            let result = grad.compute(&input).await.unwrap();
            for (i, &val) in result.iter().enumerate().take(n - 1).skip(1) {
                assert!(
                    (val - 1.0).abs() < 1e-10,
                    "At i={i}: expected 1.0, got {val}",
                );
            }
        }) {
            return;
        }
    }

    #[tokio::test]
    async fn test_gradient_1d_quadratic() {
        if !super::common::run_gpu_resilient_async(|| async {
            let device = barracuda::device::test_pool::get_test_device().await;
            let n = 100;
            let dx = 0.1;
            let grad = Gradient1D::new(device, n, dx).unwrap();
            let input: Vec<f64> = (0..n).map(|i| (i as f64 * dx).powi(2)).collect();
            let result = grad.compute(&input).await.unwrap();
            for (i, &val) in result.iter().enumerate().take(n - 1).skip(1) {
                let x = i as f64 * dx;
                let expected = 2.0 * x;
                let error = (val - expected).abs();
                assert!(
                    error < 0.02,
                    "At i={i}: expected {expected}, got {val}, error={error}",
                );
            }
        }) {
            return;
        }
    }

    #[tokio::test]
    async fn test_gradient_1d_cubic() {
        if !super::common::run_gpu_resilient_async(|| async {
            let device = barracuda::device::test_pool::get_test_device().await;
            let n = 200;
            let dx = 0.05;
            let grad = Gradient1D::new(device, n, dx).unwrap();
            let input: Vec<f64> = (0..n).map(|i| (i as f64 * dx).powi(3)).collect();
            let result = grad.compute(&input).await.unwrap();
            for (i, &val) in result.iter().enumerate().take(n - 2).skip(2) {
                let x = i as f64 * dx;
                let expected = 3.0 * x * x;
                let abs_error = (val - expected).abs();
                // Combined tolerance: absolute near x=0 (where expected ~ 0) and
                // relative elsewhere. FD truncation error is O(dx^2) ≈ 0.0025.
                let tol = 0.05f64.mul_add(expected.abs(), dx * dx * 10.0);
                assert!(
                    abs_error < tol,
                    "At i={i}: abs_error={abs_error}, tol={tol}"
                );
            }
        }) {
            return;
        }
    }

    #[tokio::test]
    async fn test_gradient_1d_sine() {
        if !super::common::run_gpu_resilient_async(|| async {
            let device = barracuda::device::test_pool::get_test_device().await;
            let n = 200;
            let dx = 0.05;
            let grad = Gradient1D::new(device, n, dx).unwrap();
            let input: Vec<f64> = (0..n).map(|i| (i as f64 * dx).sin()).collect();
            let result = grad.compute(&input).await.unwrap();
            for (i, &val) in result.iter().enumerate().take(n - 5).skip(5) {
                let x = i as f64 * dx;
                let expected = x.cos();
                let error = (val - expected).abs();
                assert!(error < 0.01, "At i={i}: expected {expected}, got {val}",);
            }
        }) {
            return;
        }
    }

    #[tokio::test]
    async fn test_gradient_2d_creation() {
        if !super::common::run_gpu_resilient_async(|| async {
            let device = barracuda::device::test_pool::get_test_device().await;
            let grad = Gradient2D::new(device, 64, 64, 0.1, 0.1);
            assert!(grad.is_ok());
            assert_eq!(grad.unwrap().shape(), (64, 64));
        }) {
            return;
        }
    }

    #[tokio::test]
    async fn test_laplacian_2d_creation() {
        if !super::common::run_gpu_resilient_async(|| async {
            let device = barracuda::device::test_pool::get_test_device().await;
            // Laplacian2D needs 5 storage buffers; some backends (llvmpipe) cap at 4.
            let limits = device.device().limits();
            if limits.max_storage_buffers_per_shader_stage < 5 {
                // Skip: Laplacian2D needs 5 storage buffers; some backends cap at 4.
                return;
            }
            let lap = Laplacian2D::new(device, 100, 100, 0.05, 0.05).unwrap();
            assert_eq!(lap.shape(), (100, 100));
        }) {
            return;
        }
    }

    #[tokio::test]
    async fn test_cylindrical_gradient_creation() {
        if !super::common::run_gpu_resilient_async(|| async {
            let device = barracuda::device::test_pool::get_test_device().await;
            let cyl = CylindricalGradient::new(device, 50, 100, 0.1, 0.1, -5.0);
            assert!(cyl.is_ok());
            assert_eq!(cyl.unwrap().shape(), (50, 100));
        }) {
            return;
        }
    }

    #[tokio::test]
    async fn test_cylindrical_laplacian_creation() {
        if !super::common::run_gpu_resilient_async(|| async {
            let device = barracuda::device::test_pool::get_test_device().await;
            let cyl = CylindricalLaplacian::new(device, 50, 100, 0.1, 0.1, -5.0);
            assert!(cyl.is_ok());
            assert_eq!(cyl.unwrap().shape(), (50, 100));
        }) {
            return;
        }
    }
}

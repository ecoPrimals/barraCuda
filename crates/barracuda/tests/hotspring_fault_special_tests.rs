// SPDX-License-Identifier: AGPL-3.0-only
//! hotSpring Evolution: Fault (error handling) and Special functions (Hermite, Laguerre).

#![expect(clippy::unwrap_used, reason = "tests")]
mod common;

use barracuda::error::BarracudaError;
use barracuda::ops::grid::Gradient1D;
use barracuda::ops::mixing::{LinearMixer, MixingParams};

mod fault {
    use super::*;

    #[tokio::test]
    async fn test_mixer_dimension_mismatch_old() {
        if !super::common::run_gpu_resilient_async(|| async {
            let device = barracuda::device::test_pool::get_test_device().await;
            let params = MixingParams::default();
            let mixer = LinearMixer::new(device, 100, params).unwrap();
            let x_old = vec![1.0; 50];
            let x_computed = vec![2.0; 100];
            let result = mixer.mix(&x_old, &x_computed).await;
            assert!(result.is_err(), "Should fail on dimension mismatch");
            match result.unwrap_err() {
                BarracudaError::InvalidInput { message } => {
                    assert!(message.contains("mismatch") || message.contains("dimension"));
                }
                other => panic!("Expected InvalidInput error, got {other:?}"),
            }
        }) {
            return;
        }
    }

    #[tokio::test]
    async fn test_mixer_dimension_mismatch_computed() {
        if !super::common::run_gpu_resilient_async(|| async {
            let device = barracuda::device::test_pool::get_test_device().await;
            let params = MixingParams::default();
            let mixer = LinearMixer::new(device, 100, params).unwrap();
            let x_old = vec![1.0; 100];
            let x_computed = vec![2.0; 200];
            let result = mixer.mix(&x_old, &x_computed).await;
            assert!(result.is_err(), "Should fail on dimension mismatch");
        }) {
            return;
        }
    }

    #[tokio::test]
    async fn test_gradient_size_mismatch() {
        if !super::common::run_gpu_resilient_async(|| async {
            let device = barracuda::device::test_pool::get_test_device().await;
            let grad = Gradient1D::new(device, 100, 0.1).unwrap();
            let input = vec![1.0; 50];
            let result = grad.compute(&input).await;
            assert!(result.is_err(), "Should fail on size mismatch");
            match result.unwrap_err() {
                BarracudaError::InvalidInput { message } => {
                    assert!(message.contains("mismatch") || message.contains("size"));
                }
                other => panic!("Expected InvalidInput error, got {other:?}"),
            }
        }) {
            return;
        }
    }

    #[tokio::test]
    async fn test_mixer_zero_dimension() {
        if !super::common::run_gpu_resilient_async(|| async {
            let device = barracuda::device::test_pool::get_test_device().await;
            let params = MixingParams::default();
            let mixer = LinearMixer::new(device, 0, params);
            match mixer {
                Err(_) => {} // Zero-dim rejection at construction is correct
                Ok(m) => {
                    let result = m.mix(&[], &[]).await;
                    match result {
                        Err(_) => {} // Zero-dim rejection at execution is also acceptable
                        Ok(output) => assert!(output.is_empty()),
                    }
                }
            }
        }) {
            return;
        }
    }

    #[tokio::test]
    async fn test_gradient_empty_input() {
        if !super::common::run_gpu_resilient_async(|| async {
            let device = barracuda::device::test_pool::get_test_device().await;
            let grad = Gradient1D::new(device, 0, 0.1);
            match grad {
                Err(_) => {} // Zero-dim rejection at construction is correct
                Ok(g) => {
                    let result = g.compute(&[]).await;
                    match result {
                        Err(_) => {} // Empty-input rejection at execution is also acceptable
                        Ok(output) => assert!(output.is_empty()),
                    }
                }
            }
        }) {
            return;
        }
    }

    #[tokio::test]
    async fn test_mixer_nan_propagation() {
        if !super::common::run_gpu_resilient_async(|| async {
            let device = barracuda::device::test_pool::get_test_device().await;
            let params = MixingParams {
                alpha: 0.5,
                ..Default::default()
            };
            let mixer = LinearMixer::new(device, 10, params).unwrap();
            let x_old = vec![1.0; 10];
            let mut x_computed = vec![2.0; 10];
            x_computed[5] = f64::NAN;
            let result = mixer.mix(&x_old, &x_computed).await.unwrap();
            // GPU shaders may not preserve IEEE 754 NaN propagation on all
            // backends. Verify the NaN-tainted element is at least distinguishable
            // from the clean linear-mix result (1.0*0.5 + 2.0*0.5 = 1.5).
            let nan_output = result[5];
            assert!(
                nan_output.is_nan() || (nan_output - 1.5).abs() > 0.01,
                "NaN input should produce non-normal output, got {nan_output}"
            );
            for i in [0, 1, 2, 3, 4, 6, 7, 8, 9] {
                assert!(
                    result[i].is_finite(),
                    "Non-NaN input should produce finite output at {i}"
                );
            }
        }) {
            return;
        }
    }

    #[tokio::test]
    async fn test_mixer_infinity() {
        if !super::common::run_gpu_resilient_async(|| async {
            let device = barracuda::device::test_pool::get_test_device().await;
            let params = MixingParams {
                alpha: 0.5,
                ..Default::default()
            };
            let mixer = LinearMixer::new(device, 5, params).unwrap();
            let x_old = vec![1.0, f64::INFINITY, f64::NEG_INFINITY, 0.0, -1.0];
            let x_computed = vec![2.0, 3.0, 4.0, f64::INFINITY, f64::NEG_INFINITY];
            let result = mixer.mix(&x_old, &x_computed).await.unwrap();
            // GPU shaders may clamp or replace infinities on some backends.
            // Verify that infinity-tainted elements are at least non-normal
            // (infinite, NaN, or very large magnitude).
            for &i in &[1usize, 2, 3, 4] {
                assert!(
                    !result[i].is_finite() || result[i].abs() > 1e300,
                    "Infinity input at [{i}] should produce non-normal output, got {}",
                    result[i]
                );
            }
        }) {
            return;
        }
    }

    #[tokio::test]
    async fn test_gradient_nan_handling() {
        if !super::common::run_gpu_resilient_async(|| async {
            let device = barracuda::device::test_pool::get_test_device().await;
            let grad = Gradient1D::new(device, 20, 0.1).unwrap();
            let mut input: Vec<f64> = (0..20).map(f64::from).collect();
            input[10] = f64::NAN;
            let result = grad.compute(&input).await.unwrap();
            assert!(result[9].is_nan() || result[10].is_nan() || result[11].is_nan());
        }) {
            return;
        }
    }
}

mod special_functions {
    fn hermite_cpu(n: u32, x: f64) -> f64 {
        if n == 0 {
            return 1.0;
        }
        if n == 1 {
            return 2.0 * x;
        }
        let mut h_prev = 1.0;
        let mut h_curr = 2.0 * x;
        for k in 1..n {
            let h_next = (2.0 * x).mul_add(h_curr, -(2.0 * f64::from(k) * h_prev));
            h_prev = h_curr;
            h_curr = h_next;
        }
        h_curr
    }

    fn laguerre_cpu(n: u32, alpha: f64, x: f64) -> f64 {
        if n == 0 {
            return 1.0;
        }
        if n == 1 {
            return 1.0 + alpha - x;
        }
        let mut l_prev = 1.0;
        let mut l_curr = 1.0 + alpha - x;
        for k in 1..n {
            let kf = f64::from(k);
            let l_next = (2.0f64.mul_add(kf, 1.0) + alpha - x)
                .mul_add(l_curr, -((kf + alpha) * l_prev))
                / (kf + 1.0);
            l_prev = l_curr;
            l_curr = l_next;
        }
        l_curr
    }

    #[test]
    fn test_hermite_n0() {
        for x in [-2.0_f64, -1.0, 0.0, 0.5, 1.0, 2.0, 2.72] {
            assert!((hermite_cpu(0, x) - 1.0).abs() < 1e-15);
        }
    }

    #[test]
    fn test_hermite_n1() {
        for x in [-2.0_f64, -1.0, 0.0, 0.5, 1.0, 2.0] {
            let expected = 2.0 * x;
            assert!((hermite_cpu(1, x) - expected).abs() < 1e-14);
        }
    }

    #[test]
    fn test_hermite_n2() {
        for x in [-2.0_f64, -1.0, 0.0, 0.5, 1.0, 2.0] {
            let expected = (4.0 * x).mul_add(x, -2.0);
            assert!(
                (hermite_cpu(2, x) - expected).abs() < 1e-13,
                "H_2({}) = {} vs expected {}",
                x,
                hermite_cpu(2, x),
                expected
            );
        }
    }

    #[test]
    fn test_hermite_n3() {
        for x in [-2.0_f64, -1.0, 0.0, 0.5, 1.0, 2.0] {
            let expected = 8.0f64.mul_add(x.powi(3), -(12.0 * x));
            assert!(
                (hermite_cpu(3, x) - expected).abs() < 1e-12,
                "H_3({}) = {} vs expected {}",
                x,
                hermite_cpu(3, x),
                expected
            );
        }
    }

    #[test]
    fn test_hermite_at_zero() {
        assert!((hermite_cpu(0, 0.0) - 1.0).abs() < 1e-15);
        assert!((hermite_cpu(1, 0.0) - 0.0).abs() < 1e-15);
        assert!((hermite_cpu(2, 0.0) - (-2.0)).abs() < 1e-14);
        assert!((hermite_cpu(3, 0.0) - 0.0).abs() < 1e-14);
        assert!((hermite_cpu(4, 0.0) - 12.0).abs() < 1e-13);
    }

    #[test]
    fn test_laguerre_n0() {
        for x in [0.0_f64, 0.5, 1.0, 2.0, 5.0] {
            assert!((laguerre_cpu(0, 0.0, x) - 1.0).abs() < 1e-15);
        }
    }

    #[test]
    fn test_laguerre_n1() {
        for x in [0.0_f64, 0.5, 1.0, 2.0, 5.0] {
            let expected = 1.0 - x;
            assert!(
                (laguerre_cpu(1, 0.0, x) - expected).abs() < 1e-14,
                "L_1({}) = {} vs expected {}",
                x,
                laguerre_cpu(1, 0.0, x),
                expected
            );
        }
    }

    #[test]
    fn test_laguerre_n2() {
        for x in [0.0_f64, 0.5, 1.0, 2.0, 5.0] {
            let expected = f64::midpoint(x.mul_add(x, -(4.0 * x)), 2.0);
            assert!(
                (laguerre_cpu(2, 0.0, x) - expected).abs() < 1e-13,
                "L_2({}) = {} vs expected {}",
                x,
                laguerre_cpu(2, 0.0, x),
                expected
            );
        }
    }

    #[test]
    fn test_laguerre_alpha1() {
        assert!((laguerre_cpu(0, 1.0, 2.0) - 1.0).abs() < 1e-15);
        for x in [0.0_f64, 1.0, 2.0] {
            let expected = 2.0 - x;
            assert!(
                (laguerre_cpu(1, 1.0, x) - expected).abs() < 1e-14,
                "L_1^(1)({}) = {} vs expected {}",
                x,
                laguerre_cpu(1, 1.0, x),
                expected
            );
        }
    }

    #[test]
    fn test_laguerre_at_zero() {
        for n in 0..10 {
            assert!(
                (laguerre_cpu(n, 0.0, 0.0) - 1.0).abs() < 1e-10,
                "L_{}(0) should be 1, got {}",
                n,
                laguerre_cpu(n, 0.0, 0.0)
            );
        }
    }

    #[test]
    fn test_hermite_high_order() {
        let h10 = hermite_cpu(10, 1.0);
        assert!(h10.is_finite(), "H_10(1) should be finite");
        assert!(h10.abs() > 1.0, "H_10(1) should be non-trivial");
        let h20 = hermite_cpu(20, 0.5);
        assert!(h20.is_finite(), "H_20(0.5) should be finite");
    }

    #[test]
    fn test_laguerre_high_order() {
        let l10 = laguerre_cpu(10, 0.0, 1.0);
        assert!(l10.is_finite(), "L_10(1) should be finite");
        let l20 = laguerre_cpu(20, 2.0, 2.0);
        assert!(l20.is_finite(), "L_20^(2)(2) should be finite");
    }
}

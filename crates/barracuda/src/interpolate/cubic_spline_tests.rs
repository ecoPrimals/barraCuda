// SPDX-License-Identifier: AGPL-3.0-or-later
#![expect(clippy::unwrap_used, reason = "tests")]

use super::*;
use std::f64::consts::PI;

const TIGHT: f64 = 1e-10;
const LOOSE: f64 = 1e-8;

#[test]
fn test_natural_spline_passes_through_points() {
    let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
    let y = vec![0.0, 1.0, 0.0, 1.0, 0.0];

    let spline = CubicSpline::natural(&x, &y).unwrap();

    for (xi, yi) in x.iter().zip(y.iter()) {
        let y_eval = spline.eval(*xi).unwrap();
        assert!(
            (y_eval - yi).abs() < 1e-10,
            "Spline should pass through data points"
        );
    }
}

#[test]
fn test_linear_data_gives_linear_spline() {
    let x = vec![0.0, 1.0, 2.0, 3.0];
    let y = vec![0.0, 1.0, 2.0, 3.0];

    let spline = CubicSpline::natural(&x, &y).unwrap();

    // For linear data, natural spline should be linear
    for &xi in &[0.5, 1.5, 2.5] {
        let y_eval = spline.eval(xi).unwrap();
        assert!(
            (y_eval - xi).abs() < 1e-10,
            "Linear data should give linear interpolation"
        );
    }

    // Second derivatives should be zero
    for y2i in spline.second_derivatives() {
        assert!(
            y2i.abs() < 1e-10,
            "Second derivatives should be zero for linear data"
        );
    }
}

#[test]
fn test_spline_continuity() {
    let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
    let y = vec![0.0, 1.0, 0.5, 1.5, 1.0];

    let spline = CubicSpline::natural(&x, &y).unwrap();

    // Check C² continuity at interior knots
    for i in 1..x.len() - 1 {
        let xi = x[i];
        let eps = 1e-6;

        let y_left = spline.eval(xi - eps).unwrap();
        let y_right = spline.eval(xi + eps).unwrap();
        let y_mid = spline.eval(xi).unwrap();

        // Value continuity
        assert!(
            (y_left - y_mid).abs() < 1e-4 && (y_right - y_mid).abs() < 1e-4,
            "Spline should be continuous"
        );

        // First derivative continuity
        let dy_left = spline.derivative(xi - eps).unwrap();
        let dy_right = spline.derivative(xi + eps).unwrap();
        assert!(
            (dy_left - dy_right).abs() < 1e-3,
            "First derivative should be continuous"
        );
    }
}

#[test]
fn test_clamped_spline_derivatives() {
    let x = vec![0.0, 1.0, 2.0];
    let y = vec![0.0, 1.0, 0.0];

    let dy_left = 2.0;
    let dy_right = -2.0;

    let spline = CubicSpline::clamped(&x, &y, dy_left, dy_right).unwrap();

    // Check derivatives at endpoints
    let dy0 = spline.derivative(x[0]).unwrap();
    let dy_n = spline.derivative(x[x.len() - 1]).unwrap();

    assert!(
        (dy0 - dy_left).abs() < 1e-6,
        "Left derivative should match: {dy0} vs {dy_left}"
    );
    assert!(
        (dy_n - dy_right).abs() < 1e-6,
        "Right derivative should match: {dy_n} vs {dy_right}"
    );
}

#[test]
fn test_sine_interpolation() {
    // Interpolate sin(x) on [0, 2π]
    let n = 10;
    let x: Vec<f64> = (0..=n).map(|i| 2.0 * PI * i as f64 / n as f64).collect();
    let y: Vec<f64> = x.iter().map(|&xi| xi.sin()).collect();

    let spline = CubicSpline::natural(&x, &y).unwrap();

    // Test at intermediate points
    for i in 0..100 {
        let xi = 2.0 * PI * i as f64 / 100.0;
        let y_spline = spline.eval(xi).unwrap();
        let y_exact = xi.sin();

        assert!(
            (y_spline - y_exact).abs() < 0.01,
            "Spline should approximate sin(x) well: {y_spline} vs {y_exact} at x={xi}"
        );
    }
}

#[test]
fn test_integration() {
    // Integrate linear function y = x from 0 to 2
    let x = vec![0.0, 1.0, 2.0];
    let y = vec![0.0, 1.0, 2.0];

    let spline = CubicSpline::natural(&x, &y).unwrap();

    let integral = spline.integrate(0.0, 2.0).unwrap();
    let expected = 2.0; // ∫₀² x dx = x²/2 |₀² = 2

    assert!(
        (integral - expected).abs() < 1e-10,
        "Integral should be 2: got {integral}"
    );
}

#[test]
fn test_invalid_input_too_few_points() {
    let x = vec![0.0];
    let y = vec![0.0];

    assert!(CubicSpline::natural(&x, &y).is_err());
}

#[test]
fn test_invalid_input_non_increasing_x() {
    let x = vec![0.0, 2.0, 1.0, 3.0];
    let y = vec![0.0, 1.0, 0.0, 1.0];

    assert!(CubicSpline::natural(&x, &y).is_err());
}

#[test]
fn test_invalid_input_length_mismatch() {
    let x = vec![0.0, 1.0, 2.0];
    let y = vec![0.0, 1.0];

    assert!(CubicSpline::natural(&x, &y).is_err());
}

#[test]
fn test_extrapolation() {
    let x = vec![0.0, 1.0, 2.0];
    let y = vec![0.0, 1.0, 0.0];

    let spline = CubicSpline::natural(&x, &y).unwrap();

    // Extrapolation should work (uses end segment)
    let y_left = spline.eval(-0.5);
    let y_right = spline.eval(2.5);

    assert!(y_left.is_ok());
    assert!(y_right.is_ok());
}

#[test]
fn test_two_points() {
    // Edge case: exactly 2 points (linear interpolation)
    let x = vec![0.0, 1.0];
    let y = vec![0.0, 2.0];

    let spline = CubicSpline::natural(&x, &y).unwrap();

    assert!((spline.eval(0.5).unwrap() - 1.0).abs() < 1e-10);
    assert!((spline.derivative(0.5).unwrap() - 2.0).abs() < 1e-10);
}

#[test]
fn test_not_a_knot_boundary() {
    let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
    let y = vec![0.0, 1.0, 0.5, 1.5, 1.0];

    let spline = CubicSpline::new(&x, &y, SplineBoundary::NotAKnot).unwrap();

    for (xi, yi) in x.iter().zip(y.iter()) {
        let y_eval = spline.eval(*xi).unwrap();
        assert!(
            (y_eval - yi).abs() < 1e-10,
            "Not-a-knot spline should pass through data points"
        );
    }
}

#[test]
fn test_not_a_knot_fallback_small_n() {
    let x = vec![0.0, 1.0, 2.0];
    let y = vec![0.0, 1.0, 0.0];

    let spline = CubicSpline::new(&x, &y, SplineBoundary::NotAKnot).unwrap();

    let y_eval = spline.eval(1.0).unwrap();
    assert!((y_eval - 1.0).abs() < 1e-10);
}

#[test]
fn test_eval_many() {
    let x = vec![0.0, 1.0, 2.0, 3.0];
    let y = vec![0.0, 1.0, 2.0, 3.0];

    let spline = CubicSpline::natural(&x, &y).unwrap();

    let x_eval = vec![0.5, 1.0, 1.5, 2.5];
    let results = spline.eval_many(&x_eval).unwrap();

    assert_eq!(results.len(), 4);
    for (&xi, &yi) in x_eval.iter().zip(results.iter()) {
        assert!(
            (yi - xi).abs() < 1e-10,
            "eval_many should match eval: {yi} vs {xi}"
        );
    }
}

#[test]
fn test_second_derivative_at_points() {
    let x = vec![0.0, 1.0, 2.0, 3.0];
    let y = vec![0.0, 1.0, 2.0, 3.0];

    let spline = CubicSpline::natural(&x, &y).unwrap();

    for &xi in &[0.5, 1.5, 2.5] {
        let d2 = spline.second_derivative(xi).unwrap();
        assert!(
            d2.abs() < 1e-8,
            "Second derivative of linear data should be ~0: got {d2}"
        );
    }
}

#[test]
fn test_partial_integration() {
    let x = vec![0.0, 1.0, 2.0, 3.0];
    let y = vec![0.0, 1.0, 2.0, 3.0];

    let spline = CubicSpline::natural(&x, &y).unwrap();

    // Integrate x from 0 to 1 = 0.5
    let integral = spline.integrate(0.0, 1.0).unwrap();
    assert!(
        (integral - 0.5).abs() < 1e-8,
        "Integral of x from 0 to 1 should be 0.5: got {integral}"
    );
}

#[test]
fn test_accessors() {
    let x = vec![0.0, 1.0, 2.0];
    let y = vec![0.0, 1.0, 0.0];

    let spline = CubicSpline::natural(&x, &y).unwrap();

    assert_eq!(spline.x_data(), &[0.0, 1.0, 2.0]);
    assert_eq!(spline.y_data(), &[0.0, 1.0, 0.0]);
    assert_eq!(spline.second_derivatives().len(), 3);
}

#[test]
fn test_quadratic_integration() {
    // y = x^2, integral from 0 to 2 = 8/3
    let x: Vec<f64> = (0..=10).map(|i| i as f64 * 0.2).collect();
    let y: Vec<f64> = x.iter().map(|&xi| xi * xi).collect();

    let spline = CubicSpline::natural(&x, &y).unwrap();

    let integral = spline.integrate(0.0, 2.0).unwrap();
    let expected = 8.0 / 3.0;
    assert!(
        (integral - expected).abs() < 0.01,
        "Integral of x^2 from 0 to 2 should be ~2.667: got {integral}"
    );
}

#[test]
fn test_integrate_reversed_limits_negates() {
    let x = vec![0.0, 1.0, 2.0];
    let y = vec![0.0, 1.0, 2.0];
    let spline = CubicSpline::natural(&x, &y).unwrap();
    let forward = spline.integrate(0.2, 1.7).unwrap();
    let backward = spline.integrate(1.7, 0.2).unwrap();
    assert!((forward + backward).abs() < LOOSE);
}

#[test]
fn test_integrate_spans_multiple_segments() {
    let x = vec![0.0, 1.0, 2.0, 3.0];
    let y = vec![0.0, 1.0, 0.0, 1.0];
    let spline = CubicSpline::natural(&x, &y).unwrap();
    let piecewise: f64 = spline.integrate(0.5, 1.5).unwrap() + spline.integrate(1.5, 2.5).unwrap();
    let whole = spline.integrate(0.5, 2.5).unwrap();
    assert!((piecewise - whole).abs() < LOOSE);
}

#[test]
fn test_cubic_spline_input_slice_and_vec_ref() {
    let x = [0.0_f64, 1.0, 2.0];
    let y_vec = vec![0.0_f64, 1.0, 4.0];
    let s1 = CubicSpline::natural(x.as_slice(), &y_vec).unwrap();
    let s2 = CubicSpline::natural(&x[..], y_vec.as_slice()).unwrap();
    let xi = 0.25;
    assert!((s1.eval(xi).unwrap() - s2.eval(xi).unwrap()).abs() < TIGHT);

    let xv = vec![0.0_f64, 1.0, 2.0];
    let s3 = CubicSpline::natural(&xv, &y_vec).unwrap();
    assert!((s3.eval(xi).unwrap() - s1.eval(xi).unwrap()).abs() < TIGHT);
}

#[test]
fn test_find_interval_extrapolation_branches() {
    let x = vec![1.0, 2.0, 4.0];
    let y = vec![10.0, 20.0, 40.0];
    let spline = CubicSpline::natural(&x, &y).unwrap();
    assert_eq!(spline.find_interval(0.5).unwrap(), 0);
    assert_eq!(spline.find_interval(10.0).unwrap(), 1);
}

#[cfg(feature = "gpu")]
#[tokio::test]
async fn test_eval_many_gpu_matches_cpu() {
    use crate::device::test_pool::test_prelude::*;
    let Some(device) = test_gpu_device().await else {
        return;
    };
    let x = vec![0.0, 1.0, 2.0, 3.0, 4.0];
    let y = vec![0.0, 1.0, 4.0, 9.0, 16.0];
    let spline = CubicSpline::natural(&x, &y).unwrap();
    let queries = [0.3, 1.7, 2.2, 3.9];
    let cpu = spline.eval_many(&queries).unwrap();
    let gpu = match spline.eval_many_gpu(&queries, &device) {
        Ok(v) => v,
        Err(e) if e.is_device_lost() => return,
        Err(e) => panic!("eval_many_gpu: {e}"),
    };
    assert_eq!(cpu.len(), gpu.len());
    for (c, g) in cpu.iter().zip(gpu.iter()) {
        assert!((c - g).abs() < LOOSE, "cpu {c} vs gpu {g}");
    }
}

// SPDX-License-Identifier: AGPL-3.0-or-later
use super::*;
use std::f64::consts::PI;

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
        "Left derivative should match: {} vs {}",
        dy0,
        dy_left
    );
    assert!(
        (dy_n - dy_right).abs() < 1e-6,
        "Right derivative should match: {} vs {}",
        dy_n,
        dy_right
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
            "Spline should approximate sin(x) well: {} vs {} at x={}",
            y_spline,
            y_exact,
            xi
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
        "Integral should be 2: got {}",
        integral
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

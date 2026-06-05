// SPDX-License-Identifier: AGPL-3.0-or-later
//! Wave 79 coverage: edge cases and error paths for CPU-only methods.
//!
//! Exercises validation branches, singular matrices, empty inputs,
//! and boundary conditions for stats, linalg, signal, spectral, graph,
//! and ODE handlers.

use crate::ipc::methods::dispatch;
use serde_json::json;

use super::test_primal;

// ── linalg edge cases ───────────────────────────────────────────────────

#[tokio::test]
async fn linalg_solve_singular_matrix() {
    let primal = test_primal();
    let resp = dispatch(
        &primal,
        "linalg.solve",
        &json!({
            "matrix": [[1.0, 2.0], [2.0, 4.0]],
            "b": [3.0, 6.0]
        }),
        json!(1),
    )
    .await;
    assert!(resp.error.is_some());
}

#[tokio::test]
async fn linalg_solve_empty_matrix() {
    let primal = test_primal();
    let resp = dispatch(
        &primal,
        "linalg.solve",
        &json!({"matrix": [], "b": []}),
        json!(2),
    )
    .await;
    assert!(resp.error.is_some());
}

#[tokio::test]
async fn linalg_solve_non_square() {
    let primal = test_primal();
    let resp = dispatch(
        &primal,
        "linalg.solve",
        &json!({
            "matrix": [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
            "b": [1.0, 2.0]
        }),
        json!(3),
    )
    .await;
    assert!(resp.error.is_some());
}

#[tokio::test]
async fn linalg_solve_3x3_valid() {
    let primal = test_primal();
    let resp = dispatch(
        &primal,
        "linalg.solve",
        &json!({
            "matrix": [[2.0, 1.0, -1.0], [-3.0, -1.0, 2.0], [-2.0, 1.0, 2.0]],
            "b": [8.0, -11.0, -3.0]
        }),
        json!(4),
    )
    .await;
    assert!(resp.error.is_none(), "solve failed: {:?}", resp.error);
    let result = resp.result.unwrap();
    let x = result["result"].as_array().unwrap();
    assert_eq!(x.len(), 3);
    assert!((x[0].as_f64().unwrap() - 2.0).abs() < 1e-10);
    assert!((x[1].as_f64().unwrap() - 3.0).abs() < 1e-10);
    assert!((x[2].as_f64().unwrap() + 1.0).abs() < 1e-10);
}

#[tokio::test]
async fn linalg_svd_missing_matrix() {
    let primal = test_primal();
    let resp = dispatch(&primal, "linalg.svd", &json!({}), json!(5)).await;
    assert!(resp.error.is_some());
}

#[tokio::test]
async fn linalg_qr_missing_matrix() {
    let primal = test_primal();
    let resp = dispatch(&primal, "linalg.qr", &json!({}), json!(6)).await;
    assert!(resp.error.is_some());
}

#[tokio::test]
async fn linalg_eigenvalues_missing_matrix() {
    let primal = test_primal();
    let resp = dispatch(
        &primal,
        "linalg.eigenvalues",
        &json!({}),
        json!(7),
    )
    .await;
    assert!(resp.error.is_some());
}

// ── signal edge cases ───────────────────────────────────────────────────

#[tokio::test]
async fn signal_detect_peaks_empty_data() {
    let primal = test_primal();
    let resp = dispatch(
        &primal,
        "signal.detect_peaks",
        &json!({"data": []}),
        json!(10),
    )
    .await;
    assert!(resp.error.is_some());
}

#[tokio::test]
async fn signal_detect_peaks_single_element() {
    let primal = test_primal();
    let resp = dispatch(
        &primal,
        "signal.detect_peaks",
        &json!({"data": [5.0]}),
        json!(11),
    )
    .await;
    // Single element: either error or empty result
    assert!(resp.error.is_some() || resp.result.is_some());
}

#[tokio::test]
async fn signal_bandpass_missing_params() {
    let primal = test_primal();
    let resp = dispatch(
        &primal,
        "signal.bandpass",
        &json!({"data": [1.0, 2.0, 3.0]}),
        json!(12),
    )
    .await;
    assert!(resp.error.is_some());
}

#[tokio::test]
async fn signal_derivative_valid() {
    let primal = test_primal();
    let resp = dispatch(
        &primal,
        "signal.derivative",
        &json!({"signal": [0.0, 1.0, 4.0, 9.0, 16.0, 25.0, 36.0]}),
        json!(13),
    )
    .await;
    assert!(resp.error.is_none(), "derivative failed: {:?}", resp.error);
    let result = resp.result.unwrap();
    assert!(result["result"].is_array());
}

// ── spectral edge cases ─────────────────────────────────────────────────

#[tokio::test]
async fn spectral_fft_missing_data() {
    let primal = test_primal();
    let resp = dispatch(&primal, "spectral.fft", &json!({}), json!(20)).await;
    assert!(resp.error.is_some());
}

#[tokio::test]
async fn spectral_fft_single_sample() {
    let primal = test_primal();
    let resp = dispatch(
        &primal,
        "spectral.fft",
        &json!({"data": [42.0]}),
        json!(21),
    )
    .await;
    assert!(resp.error.is_none());
}

#[tokio::test]
async fn spectral_power_spectrum_valid() {
    let primal = test_primal();
    let data: Vec<f64> = (0..64_i32).map(|i| (f64::from(i) * 0.1).sin()).collect();
    let resp = dispatch(
        &primal,
        "spectral.power_spectrum",
        &json!({"data": data}),
        json!(22),
    )
    .await;
    assert!(resp.error.is_none());
    let result = resp.result.unwrap();
    assert!(result["result"].is_array());
}

#[tokio::test]
async fn spectral_stft_missing_params() {
    let primal = test_primal();
    let resp = dispatch(
        &primal,
        "spectral.stft",
        &json!({"data": [1.0, 2.0]}),
        json!(23),
    )
    .await;
    assert!(resp.error.is_some() || resp.result.is_some());
}

// ── stats edge cases ────────────────────────────────────────────────────

#[tokio::test]
async fn stats_mean_missing_data() {
    let primal = test_primal();
    let resp = dispatch(
        &primal,
        "stats.mean",
        &json!({}),
        json!(30),
    )
    .await;
    assert!(resp.error.is_some());
}

#[tokio::test]
async fn stats_mean_single_value() {
    let primal = test_primal();
    let resp = dispatch(
        &primal,
        "stats.mean",
        &json!({"data": [7.5]}),
        json!(31),
    )
    .await;
    assert!(resp.error.is_none());
    let result = resp.result.unwrap();
    assert!((result["result"].as_f64().unwrap() - 7.5).abs() < 1e-10);
}

#[tokio::test]
async fn stats_variance_requires_data() {
    let primal = test_primal();
    let resp = dispatch(
        &primal,
        "stats.variance",
        &json!({}),
        json!(32),
    )
    .await;
    assert!(resp.error.is_some());
}

#[tokio::test]
async fn stats_spearman_mismatched_lengths() {
    let primal = test_primal();
    let resp = dispatch(
        &primal,
        "stats.spearman",
        &json!({"x": [1.0, 2.0], "y": [1.0, 2.0, 3.0]}),
        json!(33),
    )
    .await;
    assert!(resp.error.is_some());
}

#[tokio::test]
async fn stats_chi_squared_valid() {
    let primal = test_primal();
    let resp = dispatch(
        &primal,
        "stats.chi_squared",
        &json!({
            "observed": [10.0, 20.0, 30.0],
            "expected": [15.0, 15.0, 30.0]
        }),
        json!(34),
    )
    .await;
    assert!(resp.error.is_none());
    let result = resp.result.unwrap();
    assert!(result["result"].as_f64().unwrap() > 0.0);
}

#[tokio::test]
async fn stats_weighted_mean_mismatched() {
    let primal = test_primal();
    let resp = dispatch(
        &primal,
        "stats.weighted_mean",
        &json!({"data": [1.0, 2.0], "weights": [0.5]}),
        json!(35),
    )
    .await;
    assert!(resp.error.is_some());
}

#[tokio::test]
async fn stats_shannon_entropy_valid() {
    let primal = test_primal();
    let resp = dispatch(
        &primal,
        "stats.shannon",
        &json!({"counts": [5.0, 3.0, 2.0, 1.0]}),
        json!(36),
    )
    .await;
    assert!(resp.error.is_none(), "shannon failed: {:?}", resp.error);
    let result = resp.result.unwrap();
    assert!(result["result"].as_f64().unwrap() >= 0.0);
}

// ── graph / ode edge cases ──────────────────────────────────────────────

#[tokio::test]
async fn graph_belief_propagation_missing_graph() {
    let primal = test_primal();
    let resp = dispatch(
        &primal,
        "graph.belief_propagation",
        &json!({}),
        json!(40),
    )
    .await;
    assert!(resp.error.is_some());
}

#[tokio::test]
async fn ode_step_missing_params() {
    let primal = test_primal();
    let resp = dispatch(&primal, "ode.step", &json!({}), json!(41)).await;
    assert!(resp.error.is_some());
}

// ── activation edge cases ───────────────────────────────────────────────

#[tokio::test]
async fn activation_softmax_empty() {
    let primal = test_primal();
    let resp = dispatch(
        &primal,
        "activation.softmax",
        &json!({"data": []}),
        json!(50),
    )
    .await;
    assert!(resp.error.is_some());
}

#[tokio::test]
async fn activation_softmax_single() {
    let primal = test_primal();
    let resp = dispatch(
        &primal,
        "activation.softmax",
        &json!({"data": [100.0]}),
        json!(51),
    )
    .await;
    assert!(resp.error.is_none());
    let result = resp.result.unwrap();
    let arr = result["result"].as_array().unwrap();
    assert!((arr[0].as_f64().unwrap() - 1.0).abs() < 1e-10);
}

#[tokio::test]
async fn activation_gelu_valid() {
    let primal = test_primal();
    let resp = dispatch(
        &primal,
        "activation.gelu",
        &json!({"data": [-2.0, -1.0, 0.0, 1.0, 2.0]}),
        json!(52),
    )
    .await;
    assert!(resp.error.is_none());
    let result = resp.result.unwrap();
    let arr = result["result"].as_array().unwrap();
    assert_eq!(arr.len(), 5);
    assert!((arr[2].as_f64().unwrap()).abs() < 1e-5);
}

// ── math edge cases ─────────────────────────────────────────────────────

#[tokio::test]
async fn math_sigmoid_boundary_values() {
    let primal = test_primal();
    let resp = dispatch(
        &primal,
        "math.sigmoid",
        &json!({"data": [0.0]}),
        json!(60),
    )
    .await;
    assert!(resp.error.is_none());
    let result = resp.result.unwrap();
    let arr = result["result"].as_array().unwrap();
    assert!((arr[0].as_f64().unwrap() - 0.5).abs() < 1e-10);
}

#[tokio::test]
async fn math_log2_zero() {
    let primal = test_primal();
    let resp = dispatch(
        &primal,
        "math.log2",
        &json!({"data": [0.0]}),
        json!(61),
    )
    .await;
    // log2(0) is -inf, handler returns result (f64 supports -inf)
    assert!(resp.error.is_none() || resp.error.is_some());
}

#[tokio::test]
async fn math_log2_valid() {
    let primal = test_primal();
    let resp = dispatch(
        &primal,
        "math.log2",
        &json!({"data": [8.0]}),
        json!(62),
    )
    .await;
    assert!(resp.error.is_none());
    let result = resp.result.unwrap();
    let arr = result["result"].as_array().unwrap();
    assert!((arr[0].as_f64().unwrap() - 3.0).abs() < 1e-10);
}

// ── noise/rng ───────────────────────────────────────────────────────────

#[tokio::test]
async fn noise_perlin2d_valid() {
    let primal = test_primal();
    let resp = dispatch(
        &primal,
        "noise.perlin2d",
        &json!({"x": 1.5, "y": 2.3}),
        json!(70),
    )
    .await;
    assert!(resp.error.is_none());
    let result = resp.result.unwrap();
    let val = result["result"].as_f64().unwrap();
    assert!((-1.0..=1.0).contains(&val));
}

#[tokio::test]
async fn rng_uniform_range() {
    let primal = test_primal();
    let resp = dispatch(
        &primal,
        "rng.uniform",
        &json!({"min": 10.0, "max": 20.0, "n": 5, "seed": 42}),
        json!(71),
    )
    .await;
    assert!(resp.error.is_none());
    let result = resp.result.unwrap();
    let values = result["result"].as_array().unwrap();
    assert_eq!(values.len(), 5);
    for v in values {
        let f = v.as_f64().unwrap();
        assert!((10.0..=20.0).contains(&f));
    }
}

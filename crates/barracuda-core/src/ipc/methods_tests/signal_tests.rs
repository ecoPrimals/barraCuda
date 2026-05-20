// SPDX-License-Identifier: AGPL-3.0-or-later
//! Tests for signal.detect_peaks, signal.bandpass, signal.derivative IPC methods.

use super::*;

// ── signal.detect_peaks ──────────────────────────────────────────────────

#[test]
fn test_signal_detect_peaks_simple() {
    let signal = vec![0.0, 1.0, 0.5, 2.0, 0.3, 3.0, 0.1];
    let resp = signal_detect_peaks(
        &serde_json::json!({"signal": signal, "distance": 1}),
        serde_json::json!(1),
    );
    let result = resp.result.expect("detect_peaks should succeed");
    let count = result["count"].as_u64().unwrap();
    assert!(count >= 2, "should detect at least 2 peaks, got {count}");
}

#[test]
fn test_signal_detect_peaks_with_height() {
    let signal = vec![0.0, 1.0, 0.0, 5.0, 0.0, 2.0, 0.0];
    let resp = signal_detect_peaks(
        &serde_json::json!({"signal": signal, "distance": 1, "min_height": 3.0}),
        serde_json::json!(2),
    );
    let result = resp.result.expect("detect_peaks should succeed");
    let indices: Vec<u64> = result["indices"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap())
        .collect();
    assert_eq!(indices, vec![3], "only peak at index 3 has height >= 3.0");
}

#[test]
fn test_signal_detect_peaks_missing_signal() {
    let resp = signal_detect_peaks(&serde_json::json!({}), serde_json::json!(3));
    assert!(resp.error.is_some());
    assert_eq!(resp.error.unwrap().code, INVALID_PARAMS);
}

#[tokio::test]
async fn test_dispatch_signal_detect_peaks() {
    let primal = test_primal();
    let resp = dispatch(
        &primal,
        "signal.detect_peaks",
        &serde_json::json!({"signal": [0.0, 5.0, 0.0], "distance": 1}),
        serde_json::json!(4),
    )
    .await;
    let result = resp.result.expect("dispatch should succeed");
    assert_eq!(result["count"], 1);
}

// ── signal.bandpass ──────────────────────────────────────────────────────

#[test]
fn test_signal_bandpass_preserves_length() {
    let signal: Vec<f64> = (0..64).map(|i| (f64::from(i) * 0.1).sin()).collect();
    let resp = signal_bandpass(
        &serde_json::json!({
            "signal": signal,
            "sample_rate": 100.0,
            "low_hz": 1.0,
            "high_hz": 10.0
        }),
        serde_json::json!(10),
    );
    let result = resp.result.expect("bandpass should succeed");
    let filtered_len = result["result"].as_array().unwrap().len();
    assert_eq!(filtered_len, 64);
}

#[test]
fn test_signal_bandpass_invalid_params() {
    let resp = signal_bandpass(
        &serde_json::json!({
            "signal": [1.0, 2.0],
            "sample_rate": 100.0,
            "low_hz": 20.0,
            "high_hz": 5.0
        }),
        serde_json::json!(11),
    );
    assert!(resp.error.is_some(), "low_hz > high_hz should fail");
}

#[test]
fn test_signal_bandpass_missing_params() {
    let resp = signal_bandpass(
        &serde_json::json!({"signal": [1.0], "sample_rate": 100.0}),
        serde_json::json!(12),
    );
    assert!(resp.error.is_some());
}

// ── signal.derivative ────────────────────────────────────────────────────

#[test]
fn test_signal_derivative_length() {
    let signal: Vec<f64> = (0..20).map(f64::from).collect();
    let resp = signal_derivative(
        &serde_json::json!({"signal": signal}),
        serde_json::json!(20),
    );
    let result = resp.result.expect("derivative should succeed");
    let d: Vec<f64> = result["result"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    assert_eq!(d.len(), 20);
    assert_eq!(d[0], 0.0, "boundary samples should be zero");
    assert_eq!(d[1], 0.0, "boundary samples should be zero");
}

#[test]
fn test_signal_derivative_linear_signal() {
    let signal: Vec<f64> = (0..10).map(f64::from).collect();
    let resp = signal_derivative(
        &serde_json::json!({"signal": signal}),
        serde_json::json!(21),
    );
    let result = resp.result.expect("derivative should succeed");
    let d: Vec<f64> = result["result"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();
    for &val in &d[2..8] {
        assert!(
            (val - 1.0).abs() < 1e-10,
            "derivative of linear signal should be constant 1.0, got {val}"
        );
    }
}

#[tokio::test]
async fn test_dispatch_signal_derivative() {
    let primal = test_primal();
    let resp = dispatch(
        &primal,
        "signal.derivative",
        &serde_json::json!({"signal": [0.0, 1.0, 2.0, 3.0, 4.0]}),
        serde_json::json!(22),
    )
    .await;
    assert!(resp.result.is_some());
}

// SPDX-License-Identifier: AGPL-3.0-or-later
//! Coverage tests for `linalg.batched_tridiag_eigh` — QL eigendecomposition
//! over inline-data CPU path (groundSpring Exp 012 absorption).
#![expect(
    clippy::unwrap_used,
    reason = "test assertions: unwrap is idiomatic for test code"
)]

use super::super::linalg::linalg_batched_tridiag_eigh;
use serde_json::json;

fn call(params: serde_json::Value) -> serde_json::Value {
    let id = json!(1);
    let resp = linalg_batched_tridiag_eigh(&params, id);
    let s = serde_json::to_string(&resp).unwrap();
    serde_json::from_str(&s).unwrap()
}

#[test]
fn single_2x2_system() {
    let resp = call(json!({
        "diagonals": [2.0, 2.0],
        "subdiagonals": [1.0],
        "n": 2
    }));
    assert!(resp["error"].is_null(), "unexpected error: {resp}");
    let result = &resp["result"];
    let eigenvalues: Vec<f64> = serde_json::from_value(result["eigenvalues"].clone()).unwrap();
    assert_eq!(eigenvalues.len(), 2);
    let mut sorted = eigenvalues;
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert!(
        (sorted[0] - 1.0).abs() < 1e-10,
        "expected ~1.0, got {}",
        sorted[0]
    );
    assert!(
        (sorted[1] - 3.0).abs() < 1e-10,
        "expected ~3.0, got {}",
        sorted[1]
    );
    assert_eq!(result["n"], 2);
    assert_eq!(result["n_batches"], 1);
}

#[test]
fn batched_two_systems() {
    let resp = call(json!({
        "diagonals": [2.0, 2.0, 5.0, 5.0],
        "subdiagonals": [1.0, 0.0],
        "n": 2,
        "n_batches": 2
    }));
    assert!(resp["error"].is_null(), "unexpected error: {resp}");
    let eigenvalues: Vec<f64> =
        serde_json::from_value(resp["result"]["eigenvalues"].clone()).unwrap();
    assert_eq!(eigenvalues.len(), 4);

    let mut batch1: Vec<f64> = eigenvalues[0..2].to_vec();
    batch1.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert!((batch1[0] - 1.0).abs() < 1e-10);
    assert!((batch1[1] - 3.0).abs() < 1e-10);

    let mut batch2: Vec<f64> = eigenvalues[2..4].to_vec();
    batch2.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert!((batch2[0] - 5.0).abs() < 1e-10);
    assert!((batch2[1] - 5.0).abs() < 1e-10);
}

#[test]
fn single_element() {
    let resp = call(json!({
        "diagonals": [7.0],
        "subdiagonals": [],
        "n": 1
    }));
    assert!(resp["error"].is_null(), "unexpected error: {resp}");
    let eigenvalues: Vec<f64> =
        serde_json::from_value(resp["result"]["eigenvalues"].clone()).unwrap();
    assert_eq!(eigenvalues.len(), 1);
    assert!((eigenvalues[0] - 7.0).abs() < 1e-14);
}

#[test]
fn missing_diagonals() {
    let resp = call(json!({ "subdiagonals": [1.0], "n": 2 }));
    assert!(resp["error"].is_object());
    assert_eq!(resp["error"]["code"], -32602);
}

#[test]
fn missing_subdiagonals() {
    let resp = call(json!({ "diagonals": [2.0, 2.0], "n": 2 }));
    assert!(resp["error"].is_object());
    assert_eq!(resp["error"]["code"], -32602);
}

#[test]
fn missing_n() {
    let resp = call(json!({ "diagonals": [2.0, 2.0], "subdiagonals": [1.0] }));
    assert!(resp["error"].is_object());
    assert_eq!(resp["error"]["code"], -32602);
}

#[test]
fn n_zero_rejected() {
    let resp = call(json!({ "diagonals": [], "subdiagonals": [], "n": 0 }));
    assert!(resp["error"].is_object());
    assert_eq!(resp["error"]["code"], -32602);
}

#[test]
fn diagonals_length_mismatch() {
    let resp = call(json!({
        "diagonals": [1.0, 2.0, 3.0],
        "subdiagonals": [1.0],
        "n": 2,
        "n_batches": 1
    }));
    assert!(resp["error"].is_object());
    assert_eq!(resp["error"]["code"], -32602);
}

#[test]
fn subdiagonals_length_mismatch() {
    let resp = call(json!({
        "diagonals": [1.0, 2.0],
        "subdiagonals": [1.0, 2.0],
        "n": 2,
        "n_batches": 1
    }));
    assert!(resp["error"].is_object());
    assert_eq!(resp["error"]["code"], -32602);
}

#[test]
fn eigenvectors_returned() {
    let resp = call(json!({
        "diagonals": [2.0, 2.0],
        "subdiagonals": [1.0],
        "n": 2
    }));
    assert!(resp["error"].is_null());
    let eigvecs: Vec<f64> = serde_json::from_value(resp["result"]["eigenvectors"].clone()).unwrap();
    assert_eq!(
        eigvecs.len(),
        4,
        "2×2 system should produce 4 eigenvector elements"
    );
}

#[test]
fn three_by_three_toeplitz() {
    let resp = call(json!({
        "diagonals": [2.0, 2.0, 2.0],
        "subdiagonals": [1.0, 1.0],
        "n": 3
    }));
    assert!(resp["error"].is_null(), "unexpected error: {resp}");
    let eigenvalues: Vec<f64> =
        serde_json::from_value(resp["result"]["eigenvalues"].clone()).unwrap();
    assert_eq!(eigenvalues.len(), 3);
    let mut sorted = eigenvalues;
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let sqrt2 = std::f64::consts::SQRT_2;
    assert!((sorted[0] - (2.0 - sqrt2)).abs() < 1e-10);
    assert!((sorted[1] - 2.0).abs() < 1e-10);
    assert!((sorted[2] - (2.0 + sqrt2)).abs() < 1e-10);
}

#[test]
fn n_batches_defaults_to_one() {
    let resp = call(json!({
        "diagonals": [3.0, 3.0],
        "subdiagonals": [1.0],
        "n": 2
    }));
    assert!(resp["error"].is_null());
    assert_eq!(resp["result"]["n_batches"], 1);
}

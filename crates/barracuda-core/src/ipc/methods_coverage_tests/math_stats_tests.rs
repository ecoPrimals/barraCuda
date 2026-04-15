// SPDX-License-Identifier: AGPL-3.0-or-later
//! Coverage tests for math.*, stats.*, and health.readiness handlers.

use super::test_primal;
use crate::ipc::jsonrpc;
use crate::lifecycle::PrimalLifecycle;

use super::super::health::health_readiness;
use super::super::math::{math_log2, math_sigmoid, stats_mean, stats_std_dev, stats_weighted_mean};

// ── health.readiness after start ────────────────────────────────────

#[tokio::test]
async fn health_readiness_after_start() {
    let mut primal = test_primal();
    primal.start().await.unwrap();
    let resp = health_readiness(&primal, serde_json::json!(50));
    let result = resp.result.expect("health.readiness always succeeds");
    assert_eq!(result["status"], "ready");
}

// ── math.sigmoid ────────────────────────────────────────────────────

#[test]
fn math_sigmoid_missing_data() {
    let resp = math_sigmoid(&serde_json::json!({}), serde_json::json!(100));
    let err = resp.error.expect("missing data should fail");
    assert_eq!(err.code, jsonrpc::INVALID_PARAMS);
    assert!(err.message.contains("data"));
}

#[test]
fn math_sigmoid_happy_path() {
    let resp = math_sigmoid(
        &serde_json::json!({"data": [0.0, 1.0, -1.0]}),
        serde_json::json!(101),
    );
    assert!(resp.error.is_none());
    let result = resp.result.unwrap();
    let arr = result["result"].as_array().unwrap();
    assert_eq!(arr.len(), 3);
    assert!((arr[0].as_f64().unwrap() - 0.5).abs() < 1e-10);
}

// ── math.log2 ───────────────────────────────────────────────────────

#[test]
fn math_log2_missing_data() {
    let resp = math_log2(&serde_json::json!({}), serde_json::json!(102));
    assert!(resp.error.is_some());
}

#[test]
fn math_log2_happy_path() {
    let resp = math_log2(
        &serde_json::json!({"data": [1.0, 2.0, 8.0]}),
        serde_json::json!(103),
    );
    assert!(resp.error.is_none());
    let arr = resp.result.unwrap()["result"].as_array().unwrap().clone();
    assert!((arr[0].as_f64().unwrap()).abs() < 1e-10);
    assert!((arr[1].as_f64().unwrap() - 1.0).abs() < 1e-10);
    assert!((arr[2].as_f64().unwrap() - 3.0).abs() < 1e-10);
}

// ── stats.mean ──────────────────────────────────────────────────────

#[test]
fn stats_mean_missing_data() {
    let resp = stats_mean(&serde_json::json!({}), serde_json::json!(104));
    assert!(resp.error.is_some());
}

#[test]
fn stats_mean_happy_path() {
    let resp = stats_mean(
        &serde_json::json!({"data": [2.0, 4.0, 6.0]}),
        serde_json::json!(105),
    );
    assert!(resp.error.is_none());
    let result = resp.result.unwrap()["result"].as_f64().unwrap();
    assert!((result - 4.0).abs() < 1e-10);
}

// ── stats.std_dev ───────────────────────────────────────────────────

#[test]
fn stats_std_dev_missing_data() {
    let resp = stats_std_dev(&serde_json::json!({}), serde_json::json!(106));
    assert!(resp.error.is_some());
}

#[test]
fn stats_std_dev_happy_path() {
    let resp = stats_std_dev(
        &serde_json::json!({"data": [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0]}),
        serde_json::json!(107),
    );
    assert!(resp.error.is_none());
    let result = resp.result.unwrap()["result"].as_f64().unwrap();
    assert!(result > 0.0);
}

// ── stats.weighted_mean ─────────────────────────────────────────────

#[test]
fn stats_weighted_mean_missing_values() {
    let resp = stats_weighted_mean(
        &serde_json::json!({"weights": [1.0]}),
        serde_json::json!(108),
    );
    assert!(resp.error.is_some());
}

#[test]
fn stats_weighted_mean_missing_weights() {
    let resp = stats_weighted_mean(
        &serde_json::json!({"values": [1.0]}),
        serde_json::json!(109),
    );
    assert!(resp.error.is_some());
}

#[test]
fn stats_weighted_mean_length_mismatch() {
    let resp = stats_weighted_mean(
        &serde_json::json!({"values": [1.0, 2.0], "weights": [1.0]}),
        serde_json::json!(110),
    );
    let err = resp.error.expect("length mismatch");
    assert!(err.message.contains("!="));
}

#[test]
fn stats_weighted_mean_zero_weight() {
    let resp = stats_weighted_mean(
        &serde_json::json!({"values": [1.0], "weights": [0.0]}),
        serde_json::json!(111),
    );
    let err = resp.error.expect("zero weight");
    assert!(err.message.contains("zero"));
}

#[test]
fn stats_weighted_mean_happy_path() {
    let resp = stats_weighted_mean(
        &serde_json::json!({"values": [10.0, 20.0], "weights": [1.0, 3.0]}),
        serde_json::json!(112),
    );
    assert!(resp.error.is_none());
    let result = resp.result.unwrap()["result"].as_f64().unwrap();
    assert!((result - 17.5).abs() < 1e-10);
}

// ── stats.std_dev edge cases ────────────────────────────────────────

#[test]
fn stats_std_dev_empty_data() {
    let resp = stats_std_dev(&serde_json::json!({"data": []}), serde_json::json!(600));
    let err = resp.error.expect("empty data should fail");
    assert_eq!(err.code, jsonrpc::INTERNAL_ERROR);
    assert!(err.message.contains("std_dev failed"));
}

#[test]
fn stats_std_dev_single_element() {
    let resp = stats_std_dev(&serde_json::json!({"data": [42.0]}), serde_json::json!(601));
    let err = resp
        .error
        .expect("single element should fail for sample std_dev");
    assert_eq!(err.code, jsonrpc::INTERNAL_ERROR);
    assert!(err.message.contains("std_dev failed"));
}

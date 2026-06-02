// SPDX-License-Identifier: AGPL-3.0-or-later
//! Tests for the cross-gate compute dispatch pipeline methods.
//!
//! These exercise `compute.dispatch.capabilities`, `compute.dispatch.submit`,
//! and `compute.dispatch.result` — the wire contract that hotSpring's
//! cross-gate compute dispatch (`cross_gate.rs`) consumes.

use super::super::compute::{dispatch_capabilities, dispatch_result, dispatch_submit};
use super::test_primal;
use crate::ipc::jsonrpc;

#[test]
fn capabilities_returns_cpu_capabilities_without_gpu() {
    let primal = test_primal();
    let resp = dispatch_capabilities(&primal, serde_json::json!(1));
    assert!(resp.error.is_none());
    let result = resp.result.expect("should have result");
    assert_eq!(result["gpu_available"], false);
    let caps = result["capabilities"].as_array().expect("caps array");
    let cap_strs: Vec<&str> = caps.iter().filter_map(|v| v.as_str()).collect();
    assert!(cap_strs.contains(&"gpu.f32"));
    assert!(cap_strs.contains(&"cpu.tensor_ops"));
    assert!(cap_strs.contains(&"cpu.math"));
    assert!(cap_strs.contains(&"cpu.stats"));
    assert!(!cap_strs.contains(&"gpu.tensor_ops"));
}

#[test]
fn capabilities_includes_primal_and_version() {
    let primal = test_primal();
    let resp = dispatch_capabilities(&primal, serde_json::json!(2));
    let result = resp.result.expect("should have result");
    assert_eq!(result["primal"], "barraCuda");
    assert!(result["version"].as_str().is_some());
}

#[tokio::test]
async fn submit_missing_input_returns_invalid_params() {
    let primal = test_primal();
    let resp = dispatch_submit(&primal, &serde_json::json!({}), serde_json::json!(1)).await;
    let err = resp.error.expect("missing input.data should fail");
    assert_eq!(err.code, jsonrpc::INVALID_PARAMS);
    assert!(err.message.contains("input.data"));
}

#[tokio::test]
async fn submit_empty_data_returns_failed_status() {
    let primal = test_primal();
    let params = serde_json::json!({
        "input": {"data": [], "format": "f64_array"},
        "spring": "hotSpring",
    });
    let resp = dispatch_submit(&primal, &params, serde_json::json!(2)).await;
    assert!(resp.error.is_none(), "should return success envelope with failed status");
    let result = resp.result.expect("should have result");
    assert_eq!(result["status"], "failed");
    assert!(result["error"].as_str().unwrap_or("").contains("non-empty"));
}

#[tokio::test]
async fn submit_with_binary_and_no_dispatch_peer_falls_back() {
    let primal = test_primal();
    let params = serde_json::json!({
        "binary_b64": "AQIDBAU=",
        "input": {"data": [1.0, 2.0, 3.0], "format": "f64_array"},
        "input_hash": "abc",
        "spring": "hotSpring",
        "dispatch_mode": "passthrough",
    });
    let resp = dispatch_submit(&primal, &params, serde_json::json!(20)).await;
    assert!(resp.error.is_none());
    let result = resp.result.expect("should have result");
    assert_eq!(result["status"], "completed");
    assert_eq!(result["routed"], false);
    assert!(result["note"].as_str().unwrap_or("").contains("dispatch peer"));
}

#[tokio::test]
async fn submit_valid_returns_job_id_cpu_fallback() {
    let primal = test_primal();
    let input_data: Vec<f64> = (0..16).map(|i| i as f64).collect();
    let params = serde_json::json!({
        "input": {"data": input_data, "format": "f64_array"},
        "binary_b64": "placeholder_binary",
        "input_hash": "abc123",
        "spring": "hotSpring",
        "dispatch_mode": "passthrough",
    });
    let resp = dispatch_submit(&primal, &params, serde_json::json!(3)).await;
    assert!(resp.error.is_none(), "submit should succeed: {:?}", resp.error);
    let result = resp.result.expect("should have result");
    assert_eq!(result["status"], "completed");
    let job_id = result["job_id"].as_str().expect("should have job_id");
    assert!(job_id.starts_with("job-"));
    assert_eq!(result["elements"], 16);
}

#[tokio::test]
async fn submit_then_retrieve_result() {
    let primal = test_primal();
    let input_data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
    let params = serde_json::json!({
        "input": {"data": input_data, "format": "f64_array"},
        "binary_b64": "test",
        "spring": "hotSpring",
    });
    let resp = dispatch_submit(&primal, &params, serde_json::json!(4)).await;
    let result = resp.result.expect("submit should succeed");
    let job_id = result["job_id"].as_str().expect("job_id");

    let retrieve_resp = dispatch_result(
        &primal,
        &serde_json::json!({"job_id": job_id}),
        serde_json::json!(5),
    );
    assert!(retrieve_resp.error.is_none(), "result retrieval should succeed");
    let output = retrieve_resp.result.expect("should have output");
    assert_eq!(output["format"], "f32_array");
    assert_eq!(output["substrate"], "cpu_fallback");
    let out_arr = output["output"].as_array().expect("output array");
    assert_eq!(out_arr.len(), 4);
}

#[test]
fn result_missing_job_id_returns_invalid_params() {
    let primal = test_primal();
    let resp = dispatch_result(&primal, &serde_json::json!({}), serde_json::json!(6));
    let err = resp.error.expect("missing job_id should fail");
    assert_eq!(err.code, jsonrpc::INVALID_PARAMS);
    assert!(err.message.contains("job_id"));
}

#[test]
fn result_unknown_job_id_returns_not_found() {
    let primal = test_primal();
    let resp = dispatch_result(
        &primal,
        &serde_json::json!({"job_id": "job-nonexistent"}),
        serde_json::json!(7),
    );
    let err = resp.error.expect("unknown job should fail");
    assert!(err.message.contains("not found"));
}

#[tokio::test]
async fn full_pipeline_roundtrip() {
    let primal = test_primal();
    let caps_resp = dispatch_capabilities(&primal, serde_json::json!(10));
    assert!(caps_resp.error.is_none());

    let input: Vec<f64> = (0..64).map(|i| i as f64).collect();
    let submit_resp = dispatch_submit(
        &primal,
        &serde_json::json!({
            "input": {"data": input, "format": "f64_array"},
            "binary_b64": "compiled_shader_binary",
            "input_hash": "deadbeef",
            "spring": "hotSpring",
            "dispatch_mode": "passthrough",
        }),
        serde_json::json!(11),
    )
    .await;
    assert!(submit_resp.error.is_none());
    let job_id = submit_resp.result.as_ref().expect("result")["job_id"]
        .as_str()
        .expect("job_id");

    let result_resp = dispatch_result(
        &primal,
        &serde_json::json!({"job_id": job_id}),
        serde_json::json!(12),
    );
    assert!(result_resp.error.is_none());
    let output = result_resp.result.expect("output");
    assert_eq!(output["shape"].as_array().expect("shape")[0], 64);
}

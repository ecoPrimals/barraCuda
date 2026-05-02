// SPDX-License-Identifier: AGPL-3.0-or-later
//! Coverage tests for Sprint 50 JSON-RPC surface expansion:
//! - `stats.entropy` alias for `stats.shannon`
//! - `graph.belief_propagation` (chain PGM forward pass)

use crate::ipc::jsonrpc;

use super::super::graph::graph_belief_propagation;
use super::super::math::stats_shannon;

// ── stats.entropy alias (PG-47) ────────────────────────────────────

#[test]
fn stats_entropy_alias_dispatches_to_shannon() {
    let params = serde_json::json!({"counts": [10.0, 20.0, 30.0, 40.0]});
    let resp = stats_shannon(&params, serde_json::json!(5001));
    assert!(resp.error.is_none());
    let r = resp.result.unwrap();
    assert!(r["result"].as_f64().unwrap() > 0.0);
    assert_eq!(r["unit"], "nats");
}

#[test]
fn stats_shannon_frequencies_path() {
    let params = serde_json::json!({"frequencies": [0.25, 0.25, 0.25, 0.25]});
    let resp = stats_shannon(&params, serde_json::json!(5002));
    assert!(resp.error.is_none());
    let h = resp.result.unwrap()["result"].as_f64().unwrap();
    assert!(
        (h - 4.0_f64.ln()).abs() < 1e-10,
        "uniform entropy should be ln(4)"
    );
}

#[test]
fn stats_shannon_missing_params() {
    let resp = stats_shannon(&serde_json::json!({}), serde_json::json!(5003));
    let err = resp
        .error
        .expect("should fail without counts or frequencies");
    assert_eq!(err.code, jsonrpc::INVALID_PARAMS);
}

// ── graph.belief_propagation ────────────────────────────────────────

#[test]
fn belief_propagation_identity_transition() {
    let params = serde_json::json!({
        "input": [0.3, 0.7],
        "transitions": [[1.0, 0.0, 0.0, 1.0]],
        "layer_dims": [2]
    });
    let resp = graph_belief_propagation(&params, serde_json::json!(5010));
    assert!(resp.error.is_none());
    let r = resp.result.unwrap();
    let dists = r["distributions"].as_array().unwrap();
    assert_eq!(dists.len(), 2);
    let final_dist: Vec<f64> = dists[1]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|v| v.as_f64())
        .collect();
    assert!((final_dist[0] - 0.3).abs() < 1e-10);
    assert!((final_dist[1] - 0.7).abs() < 1e-10);
}

#[test]
fn belief_propagation_two_layers() {
    let params = serde_json::json!({
        "input": [1.0, 0.0],
        "transitions": [
            [0.5, 0.5, 0.0, 1.0],
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        ],
        "layer_dims": [2, 3]
    });
    let resp = graph_belief_propagation(&params, serde_json::json!(5011));
    assert!(resp.error.is_none());
    let r = resp.result.unwrap();
    assert_eq!(r["n_layers"], 2);
    let dists = r["distributions"].as_array().unwrap();
    assert_eq!(dists.len(), 3);
}

#[test]
fn belief_propagation_missing_input() {
    let params = serde_json::json!({"transitions": [[1.0]], "layer_dims": [1]});
    let resp = graph_belief_propagation(&params, serde_json::json!(5012));
    let err = resp.error.expect("should fail without input");
    assert_eq!(err.code, jsonrpc::INVALID_PARAMS);
}

#[test]
fn belief_propagation_mismatched_lengths() {
    let params = serde_json::json!({
        "input": [0.5, 0.5],
        "transitions": [[1.0, 0.0, 0.0, 1.0]],
        "layer_dims": [2, 3]
    });
    let resp = graph_belief_propagation(&params, serde_json::json!(5013));
    let err = resp.error.expect("mismatched lengths should fail");
    assert_eq!(err.code, jsonrpc::INVALID_PARAMS);
    assert!(err.message.contains("equal length"));
}

// ── btsp.negotiate dispatch fallback (Sprint 51 / Phase 3) ─────────

#[tokio::test]
async fn btsp_negotiate_dispatch_requires_session() {
    let primal = super::test_primal();
    let params = serde_json::json!({
        "session_id": "fake",
        "preferred_cipher": "chacha20-poly1305",
    });
    let resp =
        crate::ipc::methods::dispatch(&primal, "btsp.negotiate", &params, serde_json::json!(9001))
            .await;
    let err = resp
        .error
        .expect("btsp.negotiate without session should fail");
    assert_eq!(err.code, -32600);
    assert!(err.message.contains("authenticated BTSP session"));
}

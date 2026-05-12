// SPDX-License-Identifier: AGPL-3.0-or-later
//! Coverage tests for `precision.route` — the precision routing advisory.
//!
//! Without GPU hardware, the handler returns domain minimum tier requirements.
//! These tests validate param parsing, error paths, and the no-GPU advisory
//! contract for all 15 physics domains.

use crate::ipc::jsonrpc;

use super::super::precision::precision_route;
use super::test_primal;

// ── Parameter validation ──────────────────────────────────────────────

#[test]
fn missing_domain_returns_invalid_params() {
    let primal = test_primal();
    let resp = precision_route(&primal, &serde_json::json!({}), serde_json::json!(1));
    let err = resp.error.expect("must fail without domain");
    assert_eq!(err.code, jsonrpc::INVALID_PARAMS);
    assert!(err.message.contains("domain"));
}

#[test]
fn null_domain_returns_invalid_params() {
    let primal = test_primal();
    let resp = precision_route(
        &primal,
        &serde_json::json!({"domain": null}),
        serde_json::json!(2),
    );
    let err = resp.error.expect("null domain must fail");
    assert_eq!(err.code, jsonrpc::INVALID_PARAMS);
}

#[test]
fn numeric_domain_returns_invalid_params() {
    let primal = test_primal();
    let resp = precision_route(
        &primal,
        &serde_json::json!({"domain": 42}),
        serde_json::json!(3),
    );
    let err = resp.error.expect("numeric domain must fail");
    assert_eq!(err.code, jsonrpc::INVALID_PARAMS);
}

#[test]
fn unknown_domain_returns_invalid_params() {
    let primal = test_primal();
    let resp = precision_route(
        &primal,
        &serde_json::json!({"domain": "quantum_gravity"}),
        serde_json::json!(4),
    );
    let err = resp.error.expect("unknown domain must fail");
    assert_eq!(err.code, jsonrpc::INVALID_PARAMS);
    assert!(
        err.message.contains("quantum_gravity"),
        "error should echo the invalid domain"
    );
    assert!(
        err.message.contains("lattice_qcd"),
        "error should list valid domains"
    );
}

// ── No-GPU advisory responses (all 15 domains) ───────────────────────

fn route_no_gpu(domain: &str) -> serde_json::Value {
    let primal = test_primal();
    let resp = precision_route(
        &primal,
        &serde_json::json!({"domain": domain}),
        serde_json::json!(100),
    );
    assert!(resp.error.is_none(), "domain {domain} must succeed");
    resp.result.unwrap()
}

#[test]
fn lattice_qcd_no_gpu() {
    let r = route_no_gpu("lattice_qcd");
    assert_eq!(r["recommended_tier"], "F32");
    assert_eq!(r["fma_safe"], true);
    assert_eq!(r["hardware_hint"], "compute");
    assert_eq!(r["requires_compiler"], false);
    assert_eq!(r["needs_sovereign_compile"], false);
    assert!(r["adapter"].is_null());
}

#[test]
fn gradient_flow_no_gpu() {
    let r = route_no_gpu("gradient_flow");
    assert_eq!(r["recommended_tier"], "DF64");
    assert_eq!(r["requires_compiler"], true);
    assert_eq!(r["hardware_hint"], "compute");
}

#[test]
fn dielectric_no_gpu() {
    let r = route_no_gpu("dielectric");
    assert_eq!(r["recommended_tier"], "F64");
    assert_eq!(r["fma_safe"], false, "dielectric is FMA-sensitive");
    assert_eq!(r["hardware_hint"], "compute");
}

#[test]
fn kinetic_fluid_no_gpu() {
    let r = route_no_gpu("kinetic_fluid");
    assert_eq!(r["recommended_tier"], "F32");
    assert_eq!(r["fma_safe"], true);
}

#[test]
fn eigensolve_no_gpu() {
    let r = route_no_gpu("eigensolve");
    assert_eq!(r["recommended_tier"], "F64");
    assert_eq!(r["fma_safe"], false, "eigensolve is FMA-sensitive");
}

#[test]
fn molecular_dynamics_no_gpu() {
    let r = route_no_gpu("molecular_dynamics");
    assert_eq!(r["recommended_tier"], "F32");
}

#[test]
fn nuclear_eos_no_gpu() {
    let r = route_no_gpu("nuclear_eos");
    assert_eq!(r["recommended_tier"], "DF64");
    assert_eq!(r["requires_compiler"], true);
}

#[test]
fn population_pk_no_gpu() {
    let r = route_no_gpu("population_pk");
    assert_eq!(r["recommended_tier"], "DF64");
}

#[test]
fn bioinformatics_no_gpu() {
    let r = route_no_gpu("bioinformatics");
    assert_eq!(r["recommended_tier"], "F32");
    assert_eq!(r["requires_compiler"], false);
}

#[test]
fn hydrology_no_gpu() {
    let r = route_no_gpu("hydrology");
    assert_eq!(r["recommended_tier"], "F32");
}

#[test]
fn statistics_no_gpu() {
    let r = route_no_gpu("statistics");
    assert_eq!(r["recommended_tier"], "F32");
}

#[test]
fn general_no_gpu() {
    let r = route_no_gpu("general");
    assert_eq!(r["recommended_tier"], "F32");
}

#[test]
fn inference_no_gpu() {
    let r = route_no_gpu("inference");
    assert_eq!(r["recommended_tier"], "Q4");
    assert_eq!(r["hardware_hint"], "compute");
}

#[test]
fn training_no_gpu() {
    let r = route_no_gpu("training");
    assert_eq!(r["recommended_tier"], "BF16");
    assert_eq!(r["hardware_hint"], "tensor_core");
    assert_eq!(r["requires_compiler"], true);
}

#[test]
fn hashing_no_gpu() {
    let r = route_no_gpu("hashing");
    assert_eq!(r["recommended_tier"], "Binary");
    assert_eq!(r["requires_compiler"], false);
    assert_eq!(r["hardware_hint"], "compute");
}

// ── Response structure completeness ───────────────────────────────────

#[test]
fn response_has_all_required_fields() {
    let r = route_no_gpu("lattice_qcd");
    assert!(
        r.get("recommended_tier").is_some(),
        "missing recommended_tier"
    );
    assert!(r.get("fma_safe").is_some(), "missing fma_safe");
    assert!(
        r.get("requires_compiler").is_some(),
        "missing requires_compiler"
    );
    assert!(r.get("hardware_hint").is_some(), "missing hardware_hint");
    assert!(r.get("rationale").is_some(), "missing rationale");
    assert!(
        r.get("needs_sovereign_compile").is_some(),
        "missing needs_sovereign_compile"
    );
    assert!(r.get("adapter").is_some(), "missing adapter");
}

// ── Dispatch integration test ─────────────────────────────────────────

#[tokio::test]
async fn precision_route_dispatches_from_method_name() {
    use crate::ipc::methods::dispatch;
    let primal = test_primal();
    let resp = dispatch(
        &primal,
        "precision.route",
        &serde_json::json!({"domain": "lattice_qcd"}),
        serde_json::json!(200),
    )
    .await;
    assert!(resp.error.is_none(), "dispatch must succeed");
    let r = resp.result.unwrap();
    assert_eq!(r["recommended_tier"], "F32");
}

#[tokio::test]
async fn precision_route_dispatches_with_namespace_prefix() {
    use crate::ipc::methods::dispatch;
    let primal = test_primal();
    let resp = dispatch(
        &primal,
        "barracuda.precision.route",
        &serde_json::json!({"domain": "general"}),
        serde_json::json!(201),
    )
    .await;
    assert!(resp.error.is_none(), "namespaced dispatch must succeed");
}

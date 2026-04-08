// SPDX-License-Identifier: AGPL-3.0-or-later
//! Tests for primal.info, identity.get, primal.capabilities, and Wire Standard L2.

use super::*;

// ── primal.info and primal.capabilities (no GPU needed) ─────────────────

#[test]
fn test_primal_info() {
    let primal = test_primal();
    let resp = info(&primal, serde_json::json!(100));
    let result = resp.result.expect("primal.info should always succeed");
    assert_eq!(result["primal"], "barraCuda");
    assert_eq!(result["protocol"], "json-rpc-2.0");
    assert_eq!(result["namespace"], "barracuda");
    assert_eq!(result["license"], "AGPL-3.0-or-later");
    assert!(result["version"].is_string());
}

#[test]
fn test_primal_capabilities_no_gpu() {
    let primal = test_primal();
    let resp = capabilities(&primal, serde_json::json!(101));
    let result = resp
        .result
        .expect("primal.capabilities should always succeed");
    assert_eq!(result["primal"], "barracuda");
    assert!(result["version"].is_string());
    assert!(result["methods"].is_array());
    assert!(result["provided_capabilities"].is_array());
    assert!(result["provides"].is_array());
    assert!(result["requires"].is_array());
    assert!(result["domains"].is_array());
    assert_eq!(result["protocol"], "jsonrpc-2.0");
    assert!(result["transport"].is_array());
    assert_eq!(result["hardware"]["gpu_available"], false);
    assert_eq!(result["hardware"]["f64_shaders"], false);
    assert_eq!(result["hardware"]["spirv_passthrough"], false);
}

// ── identity.get (Wire Standard L2) ─────────────────────────────────────

#[test]
fn test_identity_get() {
    let resp = identity(serde_json::json!(102));
    let result = resp.result.expect("identity.get should always succeed");
    assert_eq!(result["primal"], "barracuda");
    assert!(result["version"].is_string());
    assert_eq!(result["domain"], crate::PRIMAL_DOMAIN);
    assert_eq!(result["license"], "AGPL-3.0-or-later");
}

#[tokio::test]
async fn test_dispatch_identity_get() {
    let primal = test_primal();
    let resp = dispatch(
        &primal,
        "identity.get",
        &serde_json::json!({}),
        serde_json::json!(103),
    )
    .await;
    let result = resp.result.expect("identity.get via dispatch");
    assert_eq!(result["primal"], "barracuda");
    assert_eq!(result["domain"], crate::PRIMAL_DOMAIN);
}

// ── Wire Standard L2 compliance structural checks ───────────────────────

#[test]
fn test_wire_standard_l2_capabilities_envelope() {
    let primal = test_primal();
    let resp = capabilities(&primal, serde_json::json!(104));
    let result = resp.result.expect("capabilities.list must succeed");

    assert!(
        result["primal"].is_string(),
        "Wire Standard L2: primal field required"
    );
    assert!(
        result["version"].is_string(),
        "Wire Standard L2: version field required"
    );
    let methods = result["methods"]
        .as_array()
        .expect("Wire Standard L2: methods must be array");
    assert!(
        !methods.is_empty(),
        "Wire Standard L2: methods must not be empty"
    );
    for m in methods {
        let name = m.as_str().expect("methods must be strings");
        assert!(
            name.contains('.'),
            "Wire Standard L2: method {name} must follow domain.operation"
        );
    }
}

#[test]
fn test_wire_standard_l2_provided_capabilities_structure() {
    let primal = test_primal();
    let resp = capabilities(&primal, serde_json::json!(105));
    let result = resp.result.unwrap();

    let groups = result["provided_capabilities"]
        .as_array()
        .expect("provided_capabilities must be array");
    assert!(!groups.is_empty(), "at least one capability group");
    for group in groups {
        assert!(group["type"].is_string(), "group must have type");
        assert!(group["methods"].is_array(), "group must have methods array");
        assert!(group["version"].is_string(), "group must have version");
    }
}

#[test]
fn test_wire_standard_l2_methods_all_callable() {
    let primal = test_primal();
    let resp = capabilities(&primal, serde_json::json!(106));
    let result = resp.result.unwrap();
    let advertised: Vec<&str> = result["methods"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|v| v.as_str())
        .collect();
    for method in &advertised {
        assert!(
            REGISTERED_METHODS.contains(method),
            "advertised method {method} must be in REGISTERED_METHODS"
        );
    }
    for method in REGISTERED_METHODS {
        assert!(
            advertised.contains(method),
            "REGISTERED_METHODS entry {method} must be advertised"
        );
    }
}

// ── primal.info and primal.capabilities via dispatch ────────────────────

#[tokio::test]
async fn test_dispatch_primal_info() {
    let primal = test_primal();
    let resp = dispatch(
        &primal,
        "primal.info",
        &serde_json::json!({}),
        serde_json::json!(110),
    )
    .await;
    assert!(
        resp.result.is_some(),
        "primal.info should succeed via dispatch"
    );
    assert_eq!(resp.result.unwrap()["primal"], "barraCuda");
}

#[tokio::test]
async fn test_dispatch_primal_capabilities() {
    let primal = test_primal();
    let resp = dispatch(
        &primal,
        "primal.capabilities",
        &serde_json::json!({}),
        serde_json::json!(111),
    )
    .await;
    assert!(
        resp.result.is_some(),
        "primal.capabilities should succeed via dispatch"
    );
}

// SPDX-License-Identifier: AGPL-3.0-or-later
use super::super::jsonrpc::{INTERNAL_ERROR, METHOD_NOT_FOUND};
use super::compute::{compute_dispatch, parse_shape};
use super::device::{list as device_list, probe as device_probe};
use super::dispatch;
use super::fhe::{fhe_ntt, fhe_pointwise_mul};
use super::health::{
    health_check, health_liveness, health_readiness, tolerances_get, validate_gpu_stack,
};
use super::primal::{capabilities, info};
use super::tensor::{tensor_create, tensor_matmul};
use super::{REGISTERED_METHODS, normalize_method};
use crate::BarraCudaPrimal;

fn test_primal() -> BarraCudaPrimal {
    BarraCudaPrimal::new()
}

// ── parse_shape helper ──────────────────────────────────────────────────

#[test]
fn parse_shape_valid() {
    let arr = vec![
        serde_json::json!(2),
        serde_json::json!(3),
        serde_json::json!(4),
    ];
    let shape = parse_shape(&arr).expect("valid shape");
    assert_eq!(shape, vec![2, 3, 4]);
}

#[test]
fn parse_shape_single_element() {
    let arr = vec![serde_json::json!(128)];
    assert_eq!(parse_shape(&arr), Some(vec![128]));
}

#[test]
fn parse_shape_empty() {
    let arr: Vec<serde_json::Value> = vec![];
    assert_eq!(parse_shape(&arr), Some(vec![]));
}

#[test]
fn parse_shape_with_non_numeric() {
    let arr = vec![serde_json::json!(2), serde_json::json!("bad")];
    let shape = parse_shape(&arr);
    assert!(
        shape.is_none() || shape.as_ref().is_some_and(|s| s.len() < 2),
        "non-numeric values should be filtered out"
    );
}

// ── normalize_method and REGISTERED_METHODS ─────────────────────────────

#[test]
fn normalize_strips_legacy_namespace() {
    let legacy = format!("{}.device.list", crate::PRIMAL_NAMESPACE);
    assert_eq!(normalize_method(&legacy), "device.list");
}

#[test]
fn normalize_passes_through_standard_names() {
    assert_eq!(normalize_method("device.list"), "device.list");
    assert_eq!(normalize_method("health.check"), "health.check");
}

#[test]
fn normalize_passes_through_foreign_prefix() {
    assert_eq!(
        normalize_method("other_primal.device.list"),
        "other_primal.device.list"
    );
}

#[test]
fn normalize_empty() {
    assert_eq!(normalize_method(""), "");
}

#[test]
fn registered_methods_count() {
    assert_eq!(REGISTERED_METHODS.len(), 30);
}

#[test]
fn registered_methods_semantic_format() {
    for method in REGISTERED_METHODS {
        assert!(
            method.contains('.'),
            "method {method} should use domain.operation format"
        );
        assert!(
            !method.starts_with(&format!("{}.", crate::PRIMAL_NAMESPACE)),
            "method {method} should NOT have primal namespace prefix"
        );
    }
}

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
    assert!(result["provides"].is_array());
    assert!(result["requires"].is_array());
    assert!(result["domains"].is_array());
    assert!(result["methods"].is_array());
    assert_eq!(result["hardware"]["gpu_available"], false);
    assert_eq!(result["hardware"]["f64_shaders"], false);
    assert_eq!(result["hardware"]["spirv_passthrough"], false);
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

// ── device.list and device.probe ────────────────────────────────────────

#[tokio::test]
async fn test_device_list_no_gpu() {
    let primal = test_primal();
    let resp = device_list(&primal, serde_json::json!(1)).await;
    assert!(resp.result.is_some());
    let result = resp.result.unwrap();
    assert!(result["devices"].as_array().unwrap().is_empty());
}

#[tokio::test]
async fn test_device_probe_no_gpu() {
    let primal = test_primal();
    let resp = device_probe(&primal, serde_json::json!(120)).await;
    let result = resp.result.expect("device.probe always returns success");
    assert_eq!(result["available"], false);
    assert!(result["reason"].is_string());
}

// ── health and tolerances ───────────────────────────────────────────────

#[test]
fn test_health_liveness() {
    let resp = health_liveness(serde_json::json!(200));
    let result = resp.result.expect("health.liveness always succeeds");
    assert_eq!(result["status"], "alive");
}

#[test]
fn test_health_readiness_not_started() {
    let primal = test_primal();
    let resp = health_readiness(&primal, serde_json::json!(201));
    let result = resp.result.expect("health.readiness always succeeds");
    assert_eq!(result["status"], "not_ready");
    assert_eq!(result["gpu_available"], false);
}

#[tokio::test]
async fn test_health_check() {
    let primal = test_primal();
    let resp = health_check(&primal, serde_json::json!(2)).await;
    assert!(resp.result.is_some());
    let result = resp.result.unwrap();
    assert_eq!(result["name"], "barraCuda");
}

// ── health alias dispatch tests ─────────────────────────────────────────

#[tokio::test]
async fn test_dispatch_health_liveness() {
    let primal = test_primal();
    let resp = dispatch(
        &primal,
        "health.liveness",
        &serde_json::json!({}),
        serde_json::json!(210),
    )
    .await;
    assert_eq!(resp.result.unwrap()["status"], "alive");
}

#[tokio::test]
async fn test_dispatch_ping_alias() {
    let primal = test_primal();
    let resp = dispatch(
        &primal,
        "ping",
        &serde_json::json!({}),
        serde_json::json!(211),
    )
    .await;
    assert_eq!(resp.result.unwrap()["status"], "alive");
}

#[tokio::test]
async fn test_dispatch_health_alias() {
    let primal = test_primal();
    let resp = dispatch(
        &primal,
        "health",
        &serde_json::json!({}),
        serde_json::json!(212),
    )
    .await;
    assert_eq!(resp.result.unwrap()["status"], "alive");
}

#[tokio::test]
async fn test_dispatch_health_readiness() {
    let primal = test_primal();
    let resp = dispatch(
        &primal,
        "health.readiness",
        &serde_json::json!({}),
        serde_json::json!(213),
    )
    .await;
    let result = resp.result.unwrap();
    assert!(result["status"].is_string());
}

#[tokio::test]
async fn test_dispatch_status_alias() {
    let primal = test_primal();
    let resp = dispatch(
        &primal,
        "status",
        &serde_json::json!({}),
        serde_json::json!(214),
    )
    .await;
    assert!(resp.result.is_some(), "status alias for health.check");
}

#[tokio::test]
async fn test_dispatch_check_alias() {
    let primal = test_primal();
    let resp = dispatch(
        &primal,
        "check",
        &serde_json::json!({}),
        serde_json::json!(215),
    )
    .await;
    assert!(resp.result.is_some(), "check alias for health.check");
}

#[tokio::test]
async fn test_dispatch_capabilities_list() {
    let primal = test_primal();
    let resp = dispatch(
        &primal,
        "capabilities.list",
        &serde_json::json!({}),
        serde_json::json!(216),
    )
    .await;
    assert!(resp.result.is_some(), "capabilities.list canonical");
    assert!(resp.result.unwrap()["methods"].is_array());
}

#[tokio::test]
async fn test_dispatch_capability_list_alias() {
    let primal = test_primal();
    let resp = dispatch(
        &primal,
        "capability.list",
        &serde_json::json!({}),
        serde_json::json!(217),
    )
    .await;
    assert!(resp.result.is_some(), "capability.list alias");
}

#[test]
fn test_tolerances_default() {
    let resp = tolerances_get(&serde_json::json!({}), serde_json::json!(3));
    let result = resp.result.unwrap();
    assert_eq!(result["name"], "default");
}

#[test]
fn test_tolerances_fhe() {
    let resp = tolerances_get(&serde_json::json!({"name": "fhe"}), serde_json::json!(4));
    let result = resp.result.unwrap();
    assert_eq!(result["abs_tol"], 0.0);
    assert_eq!(result["rel_tol"], 0.0);
}

#[test]
fn test_tolerances_double_alias() {
    let resp = tolerances_get(
        &serde_json::json!({"name": "double"}),
        serde_json::json!(130),
    );
    let result = resp.result.unwrap();
    assert_eq!(result["name"], "double");
    assert!(result["abs_tol"].as_f64().unwrap() > 0.0);
}

#[test]
fn test_tolerances_emulated_double_alias() {
    let resp = tolerances_get(
        &serde_json::json!({"name": "emulated_double"}),
        serde_json::json!(131),
    );
    let result = resp.result.unwrap();
    assert_eq!(result["name"], "emulated_double");
}

#[test]
fn test_tolerances_float_alias() {
    let resp = tolerances_get(
        &serde_json::json!({"name": "float"}),
        serde_json::json!(132),
    );
    let result = resp.result.unwrap();
    assert_eq!(result["name"], "float");
}

#[test]
fn test_tolerances_unknown_returns_defaults() {
    let resp = tolerances_get(
        &serde_json::json!({"name": "some_unknown_precision"}),
        serde_json::json!(133),
    );
    let result = resp.result.unwrap();
    assert!(result["abs_tol"].as_f64().unwrap() > 0.0);
    assert!(result["rel_tol"].as_f64().unwrap() > 0.0);
}

// ── dispatch routing ────────────────────────────────────────────────────

#[tokio::test]
async fn test_dispatch_routing() {
    let primal = test_primal();
    let resp = dispatch(
        &primal,
        "device.list",
        &serde_json::json!({}),
        serde_json::json!(5),
    )
    .await;
    assert!(resp.result.is_some());

    let resp = dispatch(
        &primal,
        "nonexistent.method",
        &serde_json::json!({}),
        serde_json::json!(6),
    )
    .await;
    assert!(resp.error.is_some());
    assert_eq!(resp.error.unwrap().code, METHOD_NOT_FOUND);
}

#[tokio::test]
async fn test_dispatch_wrong_namespace() {
    let primal = test_primal();
    let resp = dispatch(
        &primal,
        "other_primal.device.list",
        &serde_json::json!({}),
        serde_json::json!(140),
    )
    .await;
    assert!(resp.error.is_some());
    assert_eq!(resp.error.unwrap().code, METHOD_NOT_FOUND);
}

// ── validate ────────────────────────────────────────────────────────────

#[tokio::test]
async fn test_validate_no_gpu() {
    let primal = test_primal();
    let resp = validate_gpu_stack(&primal, serde_json::json!(10)).await;
    assert!(resp.error.is_some());
    let err = resp.error.unwrap();
    assert_eq!(err.code, INTERNAL_ERROR);
}

// ── compute.dispatch error paths ────────────────────────────────────────

#[tokio::test]
async fn test_compute_dispatch_no_gpu() {
    let primal = test_primal();
    let resp = compute_dispatch(
        &primal,
        &serde_json::json!({"op": "zeros", "shape": [4]}),
        serde_json::json!(11),
    )
    .await;
    assert!(resp.error.is_some());
    assert_eq!(resp.error.unwrap().code, INTERNAL_ERROR);
}

#[tokio::test]
async fn test_compute_dispatch_missing_op() {
    let primal = test_primal();
    let resp = compute_dispatch(&primal, &serde_json::json!({}), serde_json::json!(12)).await;
    assert!(resp.error.is_some());
}

#[tokio::test]
async fn test_compute_dispatch_ones_no_gpu() {
    let primal = test_primal();
    let resp = compute_dispatch(
        &primal,
        &serde_json::json!({"op": "ones", "shape": [2, 3]}),
        serde_json::json!(150),
    )
    .await;
    assert!(resp.error.is_some());
    assert_eq!(resp.error.unwrap().code, INTERNAL_ERROR);
}

#[tokio::test]
async fn test_compute_dispatch_read_nonexistent_tensor() {
    let primal = test_primal();
    let resp = compute_dispatch(
        &primal,
        &serde_json::json!({"op": "read", "tensor_id": "nonexistent"}),
        serde_json::json!(151),
    )
    .await;
    let err = resp.error.expect("nonexistent tensor returns error");
    assert_eq!(err.code, super::super::jsonrpc::INVALID_PARAMS);
    assert!(err.message.contains("Tensor not found"));
}

#[tokio::test]
async fn test_compute_dispatch_unknown_op_no_gpu() {
    let primal = test_primal();
    let resp = compute_dispatch(
        &primal,
        &serde_json::json!({"op": "unknown_operation"}),
        serde_json::json!(152),
    )
    .await;
    assert!(resp.error.is_some());
}

// ── tensor error paths ──────────────────────────────────────────────────

#[tokio::test]
async fn test_tensor_create_no_gpu() {
    let primal = test_primal();
    let resp = tensor_create(
        &primal,
        &serde_json::json!({"shape": [2, 3]}),
        serde_json::json!(13),
    )
    .await;
    assert!(resp.error.is_some());
    assert_eq!(resp.error.unwrap().code, INTERNAL_ERROR);
}

#[tokio::test]
async fn test_tensor_create_missing_shape() {
    let primal = test_primal();
    let resp = tensor_create(&primal, &serde_json::json!({}), serde_json::json!(14)).await;
    assert!(resp.error.is_some());
}

#[tokio::test]
async fn test_tensor_matmul_tensors_not_found() {
    let primal = test_primal();
    let resp = tensor_matmul(
        &primal,
        &serde_json::json!({"lhs_id": "a", "rhs_id": "b"}),
        serde_json::json!(15),
    )
    .await;
    let err = resp.error.expect("nonexistent tensors return error");
    assert_eq!(err.code, super::super::jsonrpc::INVALID_PARAMS);
    assert!(err.message.contains("Tensor not found"));
}

#[tokio::test]
async fn test_tensor_matmul_missing_lhs() {
    let primal = test_primal();
    let resp = tensor_matmul(
        &primal,
        &serde_json::json!({"rhs_id": "b"}),
        serde_json::json!(160),
    )
    .await;
    assert!(resp.error.is_some());
}

#[tokio::test]
async fn test_tensor_matmul_missing_rhs() {
    let primal = test_primal();
    let resp = tensor_matmul(
        &primal,
        &serde_json::json!({"lhs_id": "a"}),
        serde_json::json!(161),
    )
    .await;
    assert!(resp.error.is_some());
}

// ── FHE error paths ─────────────────────────────────────────────────────

#[tokio::test]
async fn test_fhe_ntt_no_gpu() {
    let primal = test_primal();
    let resp = fhe_ntt(
        &primal,
        &serde_json::json!({"modulus": 17, "degree": 4, "root_of_unity": 4, "coefficients": [1, 2, 3, 4]}),
        serde_json::json!(17),
    )
    .await;
    assert!(resp.error.is_some());
    assert_eq!(resp.error.unwrap().code, INTERNAL_ERROR);
}

#[tokio::test]
async fn test_fhe_ntt_missing_modulus() {
    let primal = test_primal();
    let resp = fhe_ntt(
        &primal,
        &serde_json::json!({"degree": 4, "root_of_unity": 4, "coefficients": [1, 2, 3, 4]}),
        serde_json::json!(170),
    )
    .await;
    assert!(resp.error.is_some());
}

#[tokio::test]
async fn test_fhe_ntt_missing_degree() {
    let primal = test_primal();
    let resp = fhe_ntt(
        &primal,
        &serde_json::json!({"modulus": 17, "root_of_unity": 4, "coefficients": [1, 2, 3, 4]}),
        serde_json::json!(171),
    )
    .await;
    assert!(resp.error.is_some());
}

#[tokio::test]
async fn test_fhe_ntt_missing_root_of_unity() {
    let primal = test_primal();
    let resp = fhe_ntt(
        &primal,
        &serde_json::json!({"modulus": 17, "degree": 4, "coefficients": [1, 2, 3, 4]}),
        serde_json::json!(172),
    )
    .await;
    assert!(resp.error.is_some());
}

#[tokio::test]
async fn test_fhe_ntt_missing_coefficients() {
    let primal = test_primal();
    let resp = fhe_ntt(
        &primal,
        &serde_json::json!({"modulus": 17, "degree": 4, "root_of_unity": 4}),
        serde_json::json!(173),
    )
    .await;
    assert!(resp.error.is_some());
}

#[tokio::test]
async fn test_fhe_ntt_missing_params() {
    let primal = test_primal();
    let resp = fhe_ntt(&primal, &serde_json::json!({}), serde_json::json!(18)).await;
    assert!(resp.error.is_some());
}

#[tokio::test]
async fn test_fhe_pointwise_mul_no_gpu() {
    let primal = test_primal();
    let resp = fhe_pointwise_mul(
        &primal,
        &serde_json::json!({"modulus": 17, "degree": 4, "a": [1, 2, 3, 4], "b": [5, 6, 7, 8]}),
        serde_json::json!(19),
    )
    .await;
    assert!(resp.error.is_some());
    assert_eq!(resp.error.unwrap().code, INTERNAL_ERROR);
}

#[tokio::test]
async fn test_fhe_pointwise_mul_missing_a() {
    let primal = test_primal();
    let resp = fhe_pointwise_mul(
        &primal,
        &serde_json::json!({"modulus": 17, "degree": 4, "b": [5, 6, 7, 8]}),
        serde_json::json!(180),
    )
    .await;
    assert!(resp.error.is_some());
}

#[tokio::test]
async fn test_fhe_pointwise_mul_missing_b() {
    let primal = test_primal();
    let resp = fhe_pointwise_mul(
        &primal,
        &serde_json::json!({"modulus": 17, "degree": 4, "a": [1, 2, 3, 4]}),
        serde_json::json!(181),
    )
    .await;
    assert!(resp.error.is_some());
}

#[tokio::test]
async fn test_fhe_pointwise_mul_missing_params() {
    let primal = test_primal();
    let resp = fhe_pointwise_mul(&primal, &serde_json::json!({}), serde_json::json!(20)).await;
    assert!(resp.error.is_some());
}

// ── all routes ──────────────────────────────────────────────────────────

#[tokio::test]
async fn test_all_dispatch_routes_exist() {
    let primal = test_primal();
    for method in REGISTERED_METHODS {
        let resp = dispatch(
            &primal,
            method,
            &serde_json::json!({}),
            serde_json::json!(99),
        )
        .await;
        if let Some(err) = &resp.error {
            assert_ne!(err.code, METHOD_NOT_FOUND, "Method {method} not routed");
        }
    }
}

// ── tolerances comprehensive ────────────────────────────────────────────

#[test]
#[expect(clippy::float_cmp, reason = "exact tolerance comparison in test")]
fn test_tolerances_all_precisions() {
    for (name, abs_tol) in [("fhe", 0.0), ("f64", 1e-12), ("f32", 1e-5), ("df64", 1e-10)] {
        let resp = tolerances_get(&serde_json::json!({"name": name}), serde_json::json!(name));
        let result = resp.result.unwrap();
        assert_eq!(result["abs_tol"].as_f64().unwrap(), abs_tol);
    }
}

// ── tensor store ────────────────────────────────────────────────────────

#[test]
fn test_tensor_store() {
    let primal = test_primal();
    assert_eq!(primal.tensor_count(), 0);
    assert!(primal.get_tensor("nonexistent").is_none());
}

// ── dispatch via JSON-RPC text protocol (all routes) ────────────────────

#[tokio::test]
async fn test_dispatch_device_probe() {
    let primal = test_primal();
    let resp = dispatch(
        &primal,
        "device.probe",
        &serde_json::json!({}),
        serde_json::json!(200),
    )
    .await;
    let result = resp.result.expect("device.probe always succeeds");
    assert_eq!(result["available"], false);
}

#[tokio::test]
async fn test_dispatch_health_check() {
    let primal = test_primal();
    let resp = dispatch(
        &primal,
        "health.check",
        &serde_json::json!({}),
        serde_json::json!(201),
    )
    .await;
    let result = resp.result.expect("health.check succeeds without GPU");
    assert_eq!(result["name"], "barraCuda");
}

#[tokio::test]
async fn test_dispatch_tolerances_get() {
    let primal = test_primal();
    let resp = dispatch(
        &primal,
        "tolerances.get",
        &serde_json::json!({"name": "f32"}),
        serde_json::json!(202),
    )
    .await;
    let result = resp.result.expect("tolerances.get always succeeds");
    assert_eq!(result["name"], "f32");
}

#[tokio::test]
async fn test_dispatch_validate_gpu_stack() {
    let primal = test_primal();
    let resp = dispatch(
        &primal,
        "validate.gpu_stack",
        &serde_json::json!({}),
        serde_json::json!(203),
    )
    .await;
    assert!(resp.error.is_some(), "validate without GPU returns error");
}

#[tokio::test]
async fn test_dispatch_compute_dispatch() {
    let primal = test_primal();
    let resp = dispatch(
        &primal,
        "compute.dispatch",
        &serde_json::json!({"op": "zeros", "shape": [4]}),
        serde_json::json!(204),
    )
    .await;
    assert!(resp.error.is_some());
}

#[tokio::test]
async fn test_dispatch_tensor_create() {
    let primal = test_primal();
    let resp = dispatch(
        &primal,
        "tensor.create",
        &serde_json::json!({"shape": [2, 2]}),
        serde_json::json!(205),
    )
    .await;
    assert!(resp.error.is_some());
}

#[tokio::test]
async fn test_dispatch_tensor_matmul() {
    let primal = test_primal();
    let resp = dispatch(
        &primal,
        "tensor.matmul",
        &serde_json::json!({"lhs_id": "a", "rhs_id": "b"}),
        serde_json::json!(206),
    )
    .await;
    assert!(resp.error.is_some());
}

#[tokio::test]
async fn test_dispatch_fhe_ntt() {
    let primal = test_primal();
    let resp = dispatch(
        &primal,
        "fhe.ntt",
        &serde_json::json!({"modulus": 17, "degree": 4, "root_of_unity": 4, "coefficients": [1,2,3,4]}),
        serde_json::json!(207),
    )
    .await;
    assert!(resp.error.is_some());
}

#[tokio::test]
async fn test_dispatch_fhe_pointwise_mul() {
    let primal = test_primal();
    let resp = dispatch(
        &primal,
        "fhe.pointwise_mul",
        &serde_json::json!({"modulus": 17, "degree": 4, "a": [1,2,3,4], "b": [5,6,7,8]}),
        serde_json::json!(208),
    )
    .await;
    assert!(resp.error.is_some());
}

// ── normalize_method edge cases ─────────────────────────────────────────

#[test]
fn normalize_just_namespace_no_dot() {
    assert_eq!(
        normalize_method(crate::PRIMAL_NAMESPACE),
        crate::PRIMAL_NAMESPACE
    );
}

#[test]
fn normalize_namespace_with_dot() {
    let input = format!("{}.", crate::PRIMAL_NAMESPACE);
    assert_eq!(normalize_method(&input), "");
}

#[test]
fn normalize_legacy_prefix_accepted() {
    let legacy = format!("{}.device.list", crate::PRIMAL_NAMESPACE);
    let resp_method = normalize_method(&legacy);
    assert_eq!(resp_method, "device.list");
}

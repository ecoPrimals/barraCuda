// SPDX-License-Identifier: AGPL-3.0-only
use super::*;

fn test_primal() -> BarraCudaPrimal {
    BarraCudaPrimal::new()
}

#[tokio::test]
async fn test_device_list_no_gpu() {
    let primal = test_primal();
    let resp = device_list(&primal, serde_json::json!(1)).await;
    assert!(resp.result.is_some());
    let result = resp.result.unwrap();
    assert!(result["devices"].as_array().unwrap().is_empty());
}

#[tokio::test]
async fn test_health_check() {
    let primal = test_primal();
    let resp = health_check(&primal, serde_json::json!(2)).await;
    assert!(resp.result.is_some());
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

#[tokio::test]
async fn test_dispatch_routing() {
    let primal = test_primal();
    let resp = dispatch(
        &primal,
        "barracuda.device.list",
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
async fn test_validate_no_gpu() {
    let primal = test_primal();
    let resp = validate_gpu_stack(&primal, serde_json::json!(10)).await;
    assert!(resp.error.is_some());
}

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
}

#[tokio::test]
async fn test_compute_dispatch_missing_op() {
    let primal = test_primal();
    let resp = compute_dispatch(&primal, &serde_json::json!({}), serde_json::json!(12)).await;
    assert!(resp.error.is_some()); // INTERNAL_ERROR (no GPU) or INVALID_PARAMS
}

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
}

#[tokio::test]
async fn test_tensor_create_missing_shape() {
    let primal = test_primal();
    let resp = tensor_create(&primal, &serde_json::json!({}), serde_json::json!(14)).await;
    assert!(resp.error.is_some()); // INTERNAL_ERROR (no GPU) or INVALID_PARAMS
}

#[tokio::test]
async fn test_tensor_matmul_no_gpu() {
    let primal = test_primal();
    let resp = tensor_matmul(
        &primal,
        &serde_json::json!({"lhs_id": "a", "rhs_id": "b"}),
        serde_json::json!(15),
    )
    .await;
    assert!(resp.error.is_some());
}

#[tokio::test]
async fn test_tensor_matmul_missing_params() {
    let primal = test_primal();
    let resp = tensor_matmul(&primal, &serde_json::json!({}), serde_json::json!(16)).await;
    assert!(resp.error.is_some()); // INTERNAL_ERROR (no GPU) or INVALID_PARAMS
}

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
}

#[tokio::test]
async fn test_fhe_ntt_missing_params() {
    let primal = test_primal();
    let resp = fhe_ntt(&primal, &serde_json::json!({}), serde_json::json!(18)).await;
    assert!(resp.error.is_some()); // INTERNAL_ERROR (no GPU) or INVALID_PARAMS
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
}

#[tokio::test]
async fn test_fhe_pointwise_mul_missing_params() {
    let primal = test_primal();
    let resp = fhe_pointwise_mul(&primal, &serde_json::json!({}), serde_json::json!(20)).await;
    assert!(resp.error.is_some()); // INTERNAL_ERROR (no GPU) or INVALID_PARAMS
}

#[tokio::test]
async fn test_all_dispatch_routes_exist() {
    let primal = test_primal();
    let methods = [
        "barracuda.device.list",
        "barracuda.device.probe",
        "barracuda.health.check",
        "barracuda.tolerances.get",
        "barracuda.validate.gpu_stack",
        "barracuda.compute.dispatch",
        "barracuda.tensor.create",
        "barracuda.tensor.matmul",
        "barracuda.fhe.ntt",
        "barracuda.fhe.pointwise_mul",
    ];
    for method in methods {
        let resp = dispatch(
            &primal,
            method,
            &serde_json::json!({}),
            serde_json::json!(99),
        )
        .await;
        // All should NOT return METHOD_NOT_FOUND
        if let Some(err) = &resp.error {
            assert_ne!(err.code, METHOD_NOT_FOUND, "Method {method} not routed");
        }
    }
}

#[test]
#[expect(clippy::float_cmp, reason = "tests")]
fn test_tolerances_all_precisions() {
    for (name, abs_tol) in [("fhe", 0.0), ("f64", 1e-12), ("f32", 1e-5), ("df64", 1e-10)] {
        let resp = tolerances_get(&serde_json::json!({"name": name}), serde_json::json!(name));
        let result = resp.result.unwrap();
        assert_eq!(result["abs_tol"].as_f64().unwrap(), abs_tol);
    }
}

#[test]
fn test_tensor_store() {
    let primal = test_primal();
    assert_eq!(primal.tensor_count(), 0);
    assert!(primal.get_tensor("nonexistent").is_none());
}

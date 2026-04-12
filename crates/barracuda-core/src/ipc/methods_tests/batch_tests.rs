// SPDX-License-Identifier: AGPL-3.0-or-later
//! Tests for `tensor.batch.submit` IPC method — error paths and validation.

use super::*;

#[tokio::test]
async fn test_batch_missing_ops() {
    let primal = test_primal();
    let resp = tensor_batch_submit(&primal, &serde_json::json!({}), serde_json::json!(1)).await;
    assert!(resp.error.is_some());
    assert_eq!(resp.error.unwrap().code, INVALID_PARAMS);
}

#[tokio::test]
async fn test_batch_empty_ops() {
    let primal = test_primal();
    let resp = tensor_batch_submit(
        &primal,
        &serde_json::json!({"ops": []}),
        serde_json::json!(2),
    )
    .await;
    assert!(resp.error.is_some());
    assert_eq!(resp.error.unwrap().code, INVALID_PARAMS);
}

#[tokio::test]
async fn test_batch_no_gpu() {
    let primal = test_primal();
    let resp = tensor_batch_submit(
        &primal,
        &serde_json::json!({
            "ops": [{"op": "create", "alias": "x", "shape": [4], "data": [1,2,3,4]}]
        }),
        serde_json::json!(3),
    )
    .await;
    assert!(resp.error.is_some());
    assert_eq!(resp.error.unwrap().code, INTERNAL_ERROR);
}

#[tokio::test]
async fn test_batch_unknown_op() {
    let primal = test_primal();
    let resp = tensor_batch_submit(
        &primal,
        &serde_json::json!({
            "ops": [{"op": "quantum_entangle", "alias": "x"}]
        }),
        serde_json::json!(4),
    )
    .await;
    let err = resp.error.unwrap();
    assert_eq!(err.code, INVALID_PARAMS);
    assert!(err.message.contains("quantum_entangle"));
}

#[tokio::test]
async fn test_batch_missing_alias_ref() {
    let primal = test_primal();
    let resp = tensor_batch_submit(
        &primal,
        &serde_json::json!({
            "ops": [
                {"op": "relu", "alias": "y", "input": "nonexistent"}
            ]
        }),
        serde_json::json!(5),
    )
    .await;
    let err = resp.error.unwrap();
    assert_eq!(err.code, INVALID_PARAMS);
    assert!(err.message.contains("nonexistent"));
}

#[tokio::test]
async fn test_batch_create_shape_data_mismatch() {
    let primal = test_primal();
    let resp = tensor_batch_submit(
        &primal,
        &serde_json::json!({
            "ops": [
                {"op": "create", "alias": "x", "shape": [2, 3], "data": [1.0]}
            ]
        }),
        serde_json::json!(6),
    )
    .await;
    let err = resp.error.unwrap();
    assert_eq!(err.code, INVALID_PARAMS);
    assert!(err.message.contains("data length"));
}

#[tokio::test]
async fn test_batch_readback_unknown_alias() {
    let primal = test_primal();
    let resp = tensor_batch_submit(
        &primal,
        &serde_json::json!({
            "ops": [
                {"op": "readback", "alias": "out", "input": "missing"}
            ]
        }),
        serde_json::json!(7),
    )
    .await;
    let err = resp.error.unwrap();
    assert_eq!(err.code, INVALID_PARAMS);
    assert!(err.message.contains("missing"));
}

#[tokio::test]
async fn test_batch_readback_missing_input() {
    let primal = test_primal();
    let resp = tensor_batch_submit(
        &primal,
        &serde_json::json!({
            "ops": [
                {"op": "readback", "alias": "out"}
            ]
        }),
        serde_json::json!(8),
    )
    .await;
    let err = resp.error.unwrap();
    assert_eq!(err.code, INVALID_PARAMS);
}

#[tokio::test]
async fn test_batch_dispatch_routed() {
    let primal = test_primal();
    let resp = dispatch(
        &primal,
        "tensor.batch.submit",
        &serde_json::json!({"ops": []}),
        serde_json::json!(9),
    )
    .await;
    assert!(resp.error.is_some(), "empty ops should error");
}

#[tokio::test]
async fn test_batch_dispatch_with_legacy_prefix() {
    let primal = test_primal();
    let resp = dispatch(
        &primal,
        "barracuda.tensor.batch.submit",
        &serde_json::json!({"ops": []}),
        serde_json::json!(10),
    )
    .await;
    assert!(
        resp.error.is_some(),
        "empty ops should error via legacy prefix too"
    );
}

#[tokio::test]
async fn test_batch_op_missing_op_field() {
    let primal = test_primal();
    let resp = tensor_batch_submit(
        &primal,
        &serde_json::json!({"ops": [{"alias": "x", "data": [1]}]}),
        serde_json::json!(11),
    )
    .await;
    let err = resp.error.unwrap();
    assert_eq!(err.code, INVALID_PARAMS);
    assert!(err.message.contains("missing 'op'"));
}

#[tokio::test]
async fn test_batch_create_missing_shape() {
    let primal = test_primal();
    let resp = tensor_batch_submit(
        &primal,
        &serde_json::json!({"ops": [{"op": "create", "alias": "x", "data": [1]}]}),
        serde_json::json!(12),
    )
    .await;
    let err = resp.error.unwrap();
    assert_eq!(err.code, INVALID_PARAMS);
    assert!(err.message.contains("create requires 'shape'"));
}

#[tokio::test]
async fn test_batch_binary_missing_operands() {
    let primal = test_primal();
    let resp = tensor_batch_submit(
        &primal,
        &serde_json::json!({"ops": [{"op": "add", "alias": "z"}]}),
        serde_json::json!(13),
    )
    .await;
    let err = resp.error.unwrap();
    assert_eq!(err.code, INVALID_PARAMS);
    assert!(err.message.contains("requires 'a'"));
}

#[tokio::test]
async fn test_batch_scale_missing_input() {
    let primal = test_primal();
    let resp = tensor_batch_submit(
        &primal,
        &serde_json::json!({"ops": [{"op": "scale", "alias": "s"}]}),
        serde_json::json!(14),
    )
    .await;
    let err = resp.error.unwrap();
    assert_eq!(err.code, INVALID_PARAMS);
    assert!(err.message.contains("requires 'input'"));
}

#[tokio::test]
async fn test_batch_fma_missing_operand() {
    let primal = test_primal();
    let resp = tensor_batch_submit(
        &primal,
        &serde_json::json!({"ops": [
            {"op": "create", "alias": "x", "shape": [1], "data": [1.0]},
            {"op": "fma", "alias": "z", "a": "x", "b": "x"}
        ]}),
        serde_json::json!(15),
    )
    .await;
    let err = resp.error.unwrap();
    assert_eq!(err.code, INVALID_PARAMS);
    assert!(err.message.contains("fma requires 'c'"));
}

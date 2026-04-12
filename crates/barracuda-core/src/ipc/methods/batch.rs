// SPDX-License-Identifier: AGPL-3.0-or-later
//! `tensor.batch.submit` — Fused multi-op pipeline over IPC.
//!
//! Wraps [`barracuda::session::TensorSession`] so that springs can execute
//! entire tensor pipelines (e.g. "matmul → softmax → readback") in a single
//! JSON-RPC round-trip. All ops share one GPU command submission.
//!
//! ## Wire Contract
//!
//! **Request** (`tensor.batch.submit`):
//! ```json
//! {
//!   "ops": [
//!     {"op": "create", "alias": "x", "data": [1,2,3,4], "shape": [1,4]},
//!     {"op": "create", "alias": "w", "data": [1,0,0,1], "shape": [4,1]},
//!     {"op": "matmul", "alias": "h", "a": "x", "b": "w"},
//!     {"op": "relu",   "alias": "out", "input": "h"},
//!     {"op": "readback", "alias": "final", "input": "out"}
//!   ]
//! }
//! ```
//!
//! **Response**:
//! ```json
//! {
//!   "status": "completed",
//!   "outputs": {
//!     "x":     {"result_id": "t_...", "shape": [1,4], "elements": 4},
//!     "h":     {"result_id": "t_...", "shape": [1,1], "elements": 1},
//!     "out":   {"result_id": "t_...", "shape": [1,1], "elements": 1},
//!     "final": {"data": [1.0], "shape": [1,1]}
//!   },
//!   "ops_executed": 5
//! }
//! ```
//!
//! **Supported ops**: `create`, `add`, `mul`, `fma`, `scale`, `matmul`,
//! `relu`, `gelu`, `softmax`, `layer_norm`, `reshape`, `readback`.

use super::super::jsonrpc::{INTERNAL_ERROR, INVALID_PARAMS, JsonRpcResponse};
use super::compute::parse_shape;
use crate::BarraCudaPrimal;
use serde_json::Value;
use std::collections::HashMap;

/// `tensor.batch.submit` — execute a fused pipeline in a single GPU submission.
pub(super) async fn tensor_batch_submit(
    primal: &BarraCudaPrimal,
    params: &Value,
    id: Value,
) -> JsonRpcResponse {
    let Some(ops) = params.get("ops").and_then(|v| v.as_array()) else {
        return JsonRpcResponse::error(
            id,
            INVALID_PARAMS,
            "Missing required param: ops (array of batch operations)",
        );
    };

    if ops.is_empty() {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "ops array must not be empty");
    }

    const VALID_OPS: &[&str] = &[
        "create",
        "add",
        "mul",
        "fma",
        "scale",
        "matmul",
        "relu",
        "gelu",
        "softmax",
        "layer_norm",
        "reshape",
        "readback",
    ];

    if let Err(msg) = validate_batch_ops(ops, VALID_OPS) {
        return JsonRpcResponse::error(id, INVALID_PARAMS, msg);
    }

    let Some(dev) = primal.device() else {
        return JsonRpcResponse::error(id, INTERNAL_ERROR, "No GPU device available");
    };

    let mut session = barracuda::session::TensorSession::with_device(dev);
    let mut aliases: HashMap<String, barracuda::session::SessionTensor> = HashMap::new();
    let mut outputs: serde_json::Map<String, Value> = serde_json::Map::new();
    let mut readbacks: Vec<(String, String)> = Vec::new();

    for (i, op_val) in ops.iter().enumerate() {
        let op_name = op_val["op"].as_str().expect("validated above");
        let alias = op_val
            .get("alias")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();

        let result = match op_name {
            "create" => batch_create(&mut session, op_val, i),
            "add" => batch_binary(&mut session, &aliases, op_val, i, BinaryOp::Add),
            "mul" => batch_binary(&mut session, &aliases, op_val, i, BinaryOp::Mul),
            "fma" => batch_fma(&mut session, &aliases, op_val, i),
            "scale" => batch_scale(&mut session, &aliases, op_val, i),
            "matmul" => batch_binary(&mut session, &aliases, op_val, i, BinaryOp::MatMul),
            "relu" => batch_unary(&mut session, &aliases, op_val, i, UnaryOp::ReLU),
            "gelu" => batch_unary(&mut session, &aliases, op_val, i, UnaryOp::Gelu),
            "softmax" => batch_unary(&mut session, &aliases, op_val, i, UnaryOp::Softmax),
            "layer_norm" => batch_layer_norm(&mut session, &aliases, op_val, i),
            "reshape" => batch_reshape(&mut session, &aliases, op_val, i),
            "readback" => {
                let input_alias = op_val["input"].as_str().expect("validated above");
                readbacks.push((alias, input_alias.to_string()));
                continue;
            }
            _ => unreachable!("validated above"),
        };

        match result {
            Ok(tensor) => {
                if !alias.is_empty() {
                    aliases.insert(alias, tensor);
                }
            }
            Err(msg) => return JsonRpcResponse::error(id, INVALID_PARAMS, msg),
        }
    }

    if let Err(e) = session.run() {
        return JsonRpcResponse::error(id, INTERNAL_ERROR, format!("Batch execution failed: {e}"));
    }

    for (alias, tensor) in &aliases {
        let shape: Vec<usize> = tensor.shape().to_vec();
        let elements: usize = shape.iter().product();
        let stored_tensor = match tensor.to_tensor() {
            Ok(t) => t,
            Err(e) => {
                return JsonRpcResponse::error(
                    id,
                    INTERNAL_ERROR,
                    format!("Failed to export tensor '{alias}': {e}"),
                );
            }
        };
        let result_id = primal.store_tensor(stored_tensor);
        outputs.insert(
            alias.clone(),
            serde_json::json!({
                "result_id": result_id,
                "shape": shape,
                "elements": elements,
            }),
        );
    }

    for (readback_alias, input_alias) in &readbacks {
        let Some(tensor) = aliases.get(input_alias) else {
            return JsonRpcResponse::error(
                id,
                INTERNAL_ERROR,
                format!("readback: alias '{input_alias}' lost after execution"),
            );
        };
        let shape: Vec<usize> = tensor.shape().to_vec();
        match tensor.to_vec() {
            Ok(data) => {
                let key = if readback_alias.is_empty() {
                    input_alias.clone()
                } else {
                    readback_alias.clone()
                };
                outputs.insert(
                    key,
                    serde_json::json!({
                        "data": data,
                        "shape": shape,
                    }),
                );
            }
            Err(e) => {
                return JsonRpcResponse::error(
                    id,
                    INTERNAL_ERROR,
                    format!("Readback of '{input_alias}' failed: {e}"),
                );
            }
        }
    }

    JsonRpcResponse::success(
        id,
        serde_json::json!({
            "status": "completed",
            "outputs": outputs,
            "ops_executed": ops.len(),
        }),
    )
}

// ── Pre-validation (runs before device check so callers get INVALID_PARAMS) ──

fn validate_batch_ops(ops: &[Value], valid_ops: &[&str]) -> Result<(), String> {
    let mut defined_aliases = std::collections::HashSet::new();

    for (i, op_val) in ops.iter().enumerate() {
        let op_name = op_val
            .get("op")
            .and_then(|v| v.as_str())
            .ok_or_else(|| format!("ops[{i}]: missing 'op' field"))?;

        if !valid_ops.contains(&op_name) {
            return Err(format!("ops[{i}]: unknown op '{op_name}'"));
        }

        if let Some(alias) = op_val.get("alias").and_then(|v| v.as_str()) {
            if !alias.is_empty() {
                defined_aliases.insert(alias.to_string());
            }
        }

        match op_name {
            "create" => {
                if op_val.get("shape").and_then(|v| v.as_array()).is_none() {
                    return Err(format!("ops[{i}]: create requires 'shape'"));
                }
                if let (Some(shape_arr), Some(data_arr)) = (
                    op_val.get("shape").and_then(|v| v.as_array()),
                    op_val.get("data").and_then(|v| v.as_array()),
                ) {
                    let elements: u64 = shape_arr.iter().filter_map(|v| v.as_u64()).product();
                    let data_len = data_arr.len() as u64;
                    if data_len != elements {
                        return Err(format!(
                            "ops[{i}]: data length {data_len} ≠ shape product {elements}"
                        ));
                    }
                }
            }
            "readback" => {
                let input = op_val
                    .get("input")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| format!("ops[{i}]: readback requires 'input' alias"))?;
                if !defined_aliases.contains(input) {
                    return Err(format!("ops[{i}]: unknown alias '{input}'"));
                }
            }
            "add" | "mul" | "matmul" => {
                for key in ["a", "b"] {
                    let ref_name = op_val
                        .get(key)
                        .and_then(|v| v.as_str())
                        .ok_or_else(|| format!("ops[{i}]: {op_name} requires '{key}' alias"))?;
                    if !defined_aliases.contains(ref_name) {
                        return Err(format!("ops[{i}]: unknown alias '{ref_name}'"));
                    }
                }
            }
            "fma" => {
                for key in ["a", "b", "c"] {
                    let ref_name = op_val
                        .get(key)
                        .and_then(|v| v.as_str())
                        .ok_or_else(|| format!("ops[{i}]: fma requires '{key}' alias"))?;
                    if !defined_aliases.contains(ref_name) {
                        return Err(format!("ops[{i}]: unknown alias '{ref_name}'"));
                    }
                }
            }
            "scale" | "relu" | "gelu" | "softmax" | "layer_norm" | "reshape" => {
                let ref_name = op_val
                    .get("input")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| format!("ops[{i}]: {op_name} requires 'input' alias"))?;
                if !defined_aliases.contains(ref_name) {
                    return Err(format!("ops[{i}]: unknown alias '{ref_name}'"));
                }
            }
            _ => {}
        }
    }
    Ok(())
}

// ── Op helpers ──────────────────────────────────────────────────────────────

enum BinaryOp {
    Add,
    Mul,
    MatMul,
}

enum UnaryOp {
    ReLU,
    Gelu,
    Softmax,
}

fn resolve<'a>(
    aliases: &'a HashMap<String, barracuda::session::SessionTensor>,
    val: &Value,
    key: &str,
    idx: usize,
) -> Result<&'a barracuda::session::SessionTensor, String> {
    let name = val
        .get(key)
        .and_then(|v| v.as_str())
        .ok_or_else(|| format!("ops[{idx}]: missing '{key}' alias"))?;
    aliases
        .get(name)
        .ok_or_else(|| format!("ops[{idx}]: unknown alias '{name}'"))
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "f64→f32 narrowing is intentional for JSON numeric inputs"
)]
fn batch_create(
    session: &mut barracuda::session::TensorSession,
    val: &Value,
    idx: usize,
) -> Result<barracuda::session::SessionTensor, String> {
    let shape_arr = val
        .get("shape")
        .and_then(|v| v.as_array())
        .ok_or_else(|| format!("ops[{idx}]: create requires 'shape'"))?;
    let shape = parse_shape(shape_arr)
        .ok_or_else(|| format!("ops[{idx}]: shape dimension exceeds platform usize"))?;
    let elements: usize = shape.iter().product();

    let data: Vec<f32> = val.get("data").and_then(|v| v.as_array()).map_or_else(
        || vec![0.0; elements],
        |arr| {
            arr.iter()
                .filter_map(|v| v.as_f64().map(|n| n as f32))
                .collect()
        },
    );

    if data.len() != elements {
        return Err(format!(
            "ops[{idx}]: data length {} ≠ shape product {elements}",
            data.len()
        ));
    }

    session
        .tensor_with_shape(&data, &shape)
        .map_err(|e| format!("ops[{idx}]: create failed: {e}"))
}

fn batch_binary(
    session: &mut barracuda::session::TensorSession,
    aliases: &HashMap<String, barracuda::session::SessionTensor>,
    val: &Value,
    idx: usize,
    op: BinaryOp,
) -> Result<barracuda::session::SessionTensor, String> {
    let a = resolve(aliases, val, "a", idx)?;
    let b = resolve(aliases, val, "b", idx)?;
    match op {
        BinaryOp::Add => session.add(a, b),
        BinaryOp::Mul => session.mul(a, b),
        BinaryOp::MatMul => session.matmul(a, b),
    }
    .map_err(|e| format!("ops[{idx}]: {e}"))
}

fn batch_fma(
    session: &mut barracuda::session::TensorSession,
    aliases: &HashMap<String, barracuda::session::SessionTensor>,
    val: &Value,
    idx: usize,
) -> Result<barracuda::session::SessionTensor, String> {
    let a = resolve(aliases, val, "a", idx)?;
    let b = resolve(aliases, val, "b", idx)?;
    let c = resolve(aliases, val, "c", idx)?;
    session
        .fma(a, b, c)
        .map_err(|e| format!("ops[{idx}]: fma failed: {e}"))
}

#[expect(
    clippy::cast_possible_truncation,
    reason = "f64→f32 narrowing is intentional for JSON numeric inputs"
)]
fn batch_scale(
    session: &mut barracuda::session::TensorSession,
    aliases: &HashMap<String, barracuda::session::SessionTensor>,
    val: &Value,
    idx: usize,
) -> Result<barracuda::session::SessionTensor, String> {
    let a = resolve(aliases, val, "input", idx)?;
    let scalar = val
        .get("scalar")
        .and_then(|v| v.as_f64())
        .ok_or_else(|| format!("ops[{idx}]: scale requires 'scalar'"))? as f32;
    session
        .scale(a, scalar)
        .map_err(|e| format!("ops[{idx}]: scale failed: {e}"))
}

fn batch_unary(
    session: &mut barracuda::session::TensorSession,
    aliases: &HashMap<String, barracuda::session::SessionTensor>,
    val: &Value,
    idx: usize,
    op: UnaryOp,
) -> Result<barracuda::session::SessionTensor, String> {
    let a = resolve(aliases, val, "input", idx)?;
    match op {
        UnaryOp::ReLU => session.relu(a),
        UnaryOp::Gelu => session.gelu(a),
        UnaryOp::Softmax => session.softmax(a),
    }
    .map_err(|e| format!("ops[{idx}]: {e}"))
}

fn batch_layer_norm(
    session: &mut barracuda::session::TensorSession,
    aliases: &HashMap<String, barracuda::session::SessionTensor>,
    val: &Value,
    idx: usize,
) -> Result<barracuda::session::SessionTensor, String> {
    let a = resolve(aliases, val, "input", idx)?;
    let feature_size = val
        .get("feature_size")
        .and_then(|v| v.as_u64())
        .ok_or_else(|| format!("ops[{idx}]: layer_norm requires 'feature_size'"))?
        as usize;
    session
        .layer_norm(a, feature_size)
        .map_err(|e| format!("ops[{idx}]: layer_norm failed: {e}"))
}

fn batch_reshape(
    session: &mut barracuda::session::TensorSession,
    aliases: &HashMap<String, barracuda::session::SessionTensor>,
    val: &Value,
    idx: usize,
) -> Result<barracuda::session::SessionTensor, String> {
    let a = resolve(aliases, val, "input", idx)?;
    let shape_arr = val
        .get("shape")
        .and_then(|v| v.as_array())
        .ok_or_else(|| format!("ops[{idx}]: reshape requires 'shape'"))?;
    let shape = parse_shape(shape_arr)
        .ok_or_else(|| format!("ops[{idx}]: shape dimension exceeds platform usize"))?;
    session
        .reshape(a, shape)
        .map_err(|e| format!("ops[{idx}]: reshape failed: {e}"))
}

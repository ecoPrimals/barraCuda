// SPDX-License-Identifier: AGPL-3.0-or-later
//! Graph / PGM handlers for JSON-RPC IPC.
//!
//! Wires barraCuda's graph-theoretic primitives to the IPC surface —
//! Laplacian construction, belief propagation on chain PGMs, etc.

use super::super::jsonrpc::{INVALID_PARAMS, JsonRpcResponse};
use super::params::extract_f64_array;
use serde_json::Value;

/// `linalg.graph_laplacian` — compute graph Laplacian L = D - A.
///
/// Params: `adjacency` (flat row-major array), `n` (dimension).
pub(super) fn linalg_graph_laplacian(params: &Value, id: Value) -> JsonRpcResponse {
    let Some(adjacency) = extract_f64_array(params, "adjacency") else {
        return JsonRpcResponse::error(
            id,
            INVALID_PARAMS,
            "Missing required param: adjacency (array)",
        );
    };
    let Some(n) = params.get("n").and_then(|v| v.as_u64()) else {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Missing required param: n (integer)");
    };
    #[expect(clippy::cast_possible_truncation, reason = "n is a matrix dimension")]
    let n = n as usize;
    if n == 0 || adjacency.len() != n * n {
        return JsonRpcResponse::error(
            id,
            INVALID_PARAMS,
            format!("adjacency length {} != n*n ({})", adjacency.len(), n * n),
        );
    }
    let laplacian = barracuda::linalg::graph_laplacian(&adjacency, n);
    let rows: Vec<Vec<f64>> = laplacian.chunks(n).map(<[f64]>::to_vec).collect();
    JsonRpcResponse::success(id, serde_json::json!({ "result": rows, "n": n }))
}

/// `graph.belief_propagation` — chain PGM forward pass (HMM-like).
///
/// Params: `input` (probability distribution), `transitions` (array of flat row-major
/// transition matrices), `layer_dims` (array of output dimensions per layer).
pub(super) fn graph_belief_propagation(params: &Value, id: Value) -> JsonRpcResponse {
    let Some(input) = extract_f64_array(params, "input") else {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Missing required param: input (array)");
    };
    let Some(transitions_val) = params.get("transitions").and_then(|v| v.as_array()) else {
        return JsonRpcResponse::error(
            id,
            INVALID_PARAMS,
            "Missing required param: transitions (array of arrays)",
        );
    };
    let Some(dims_val) = params.get("layer_dims").and_then(|v| v.as_array()) else {
        return JsonRpcResponse::error(
            id,
            INVALID_PARAMS,
            "Missing required param: layer_dims (array of integers)",
        );
    };
    let transitions: Vec<Vec<f64>> = transitions_val
        .iter()
        .filter_map(|t| {
            t.as_array()
                .map(|a| a.iter().filter_map(|v| v.as_f64()).collect())
        })
        .collect();
    #[expect(
        clippy::cast_possible_truncation,
        reason = "layer dims are small integers"
    )]
    let layer_dims: Vec<usize> = dims_val
        .iter()
        .filter_map(|v| v.as_u64().map(|n| n as usize))
        .collect();
    if transitions.len() != layer_dims.len() {
        return JsonRpcResponse::error(
            id,
            INVALID_PARAMS,
            "transitions and layer_dims must have equal length",
        );
    }
    let trans_refs: Vec<&[f64]> = transitions.iter().map(Vec::as_slice).collect();
    let distributions =
        barracuda::linalg::belief_propagation_chain(&input, &trans_refs, &layer_dims);
    JsonRpcResponse::success(
        id,
        serde_json::json!({ "distributions": distributions, "n_layers": layer_dims.len() }),
    )
}

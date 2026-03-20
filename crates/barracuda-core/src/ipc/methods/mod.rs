// SPDX-License-Identifier: AGPL-3.0-or-later
//! JSON-RPC method handlers for barraCuda IPC endpoints.
//!
//! Each handler follows the semantic naming standard from wateringHole:
//! `{namespace}.{domain}.{operation}`. Method names are derived from
//! [`PRIMAL_NAMESPACE`](crate::PRIMAL_NAMESPACE) at startup — no
//! hardcoded primal names on the wire.

mod compute;
mod device;
mod fhe;
mod health;
mod primal;
mod tensor;

use super::jsonrpc::{JsonRpcResponse, METHOD_NOT_FOUND};
use crate::BarraCudaPrimal;
use serde_json::Value;
use std::sync::LazyLock;

/// Domain-qualified method suffixes (namespace-agnostic).
///
/// Single source of truth for which operations this primal supports.
/// Full wire names are built at startup by prepending [`PRIMAL_NAMESPACE`](crate::PRIMAL_NAMESPACE).
const METHOD_SUFFIXES: &[&str] = &[
    "primal.info",
    "primal.capabilities",
    "device.list",
    "device.probe",
    "health.check",
    "tolerances.get",
    "validate.gpu_stack",
    "compute.dispatch",
    "tensor.create",
    "tensor.matmul",
    "fhe.ntt",
    "fhe.pointwise_mul",
];

/// All JSON-RPC method names this primal supports, fully qualified.
///
/// Built from [`PRIMAL_NAMESPACE`](crate::PRIMAL_NAMESPACE) + `METHOD_SUFFIXES`
/// so renaming the namespace automatically updates all wire names.
pub static REGISTERED_METHODS: LazyLock<Vec<String>> = LazyLock::new(|| {
    METHOD_SUFFIXES
        .iter()
        .map(|suffix| format!("{}.{suffix}", crate::PRIMAL_NAMESPACE))
        .collect()
});

/// Strip the namespace prefix from a fully-qualified method name.
pub fn method_suffix(method: &str) -> Option<&str> {
    method
        .strip_prefix(crate::PRIMAL_NAMESPACE)
        .and_then(|s| s.strip_prefix('.'))
}

/// Route a JSON-RPC method call to the appropriate handler.
pub async fn dispatch(
    primal: &BarraCudaPrimal,
    method: &str,
    params: &Value,
    id: Value,
) -> JsonRpcResponse {
    match method_suffix(method) {
        Some("primal.info") => primal::info(primal, id),
        Some("primal.capabilities") => primal::capabilities(primal, id),
        Some("device.list") => device::list(primal, id).await,
        Some("device.probe") => device::probe(primal, id).await,
        Some("health.check") => health::health_check(primal, id).await,
        Some("tolerances.get") => health::tolerances_get(params, id),
        Some("validate.gpu_stack") => health::validate_gpu_stack(primal, id).await,
        Some("compute.dispatch") => compute::compute_dispatch(primal, params, id).await,
        Some("tensor.create") => tensor::tensor_create(primal, params, id).await,
        Some("tensor.matmul") => tensor::tensor_matmul(primal, params, id).await,
        Some("fhe.ntt") => fhe::fhe_ntt(primal, params, id).await,
        Some("fhe.pointwise_mul") => fhe::fhe_pointwise_mul(primal, params, id).await,
        _ => JsonRpcResponse::error(id, METHOD_NOT_FOUND, format!("Unknown method: {method}")),
    }
}

#[cfg(test)]
#[path = "../methods_tests.rs"]
mod tests;

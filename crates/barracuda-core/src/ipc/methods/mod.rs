// SPDX-License-Identifier: AGPL-3.0-or-later
//! JSON-RPC method handlers for barraCuda IPC endpoints.
//!
//! Method names follow the wateringHole semantic naming standard:
//! `{domain}.{operation}` — no primal prefix on the wire. Primals are
//! identified by their transport endpoint, not by method name prefixes.
//!
//! For backward compatibility, the legacy `{namespace}.{domain}.{operation}`
//! form is also accepted and silently normalized.

mod compute;
mod device;
mod fhe;
mod health;
mod primal;
mod tensor;

use super::jsonrpc::{JsonRpcResponse, METHOD_NOT_FOUND};
use crate::BarraCudaPrimal;
use serde_json::Value;

/// Semantic method names this primal supports (wateringHole `{domain}.{operation}`).
///
/// Single source of truth for which operations this primal provides.
/// Wire names are these exact strings — no primal prefix.
pub const REGISTERED_METHODS: &[&str] = &[
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

/// Normalize a method name: accepts both `{domain}.{operation}` (standard)
/// and legacy `{namespace}.{domain}.{operation}` (backward-compatible).
pub fn normalize_method(method: &str) -> &str {
    method
        .strip_prefix(crate::PRIMAL_NAMESPACE)
        .and_then(|s| s.strip_prefix('.'))
        .unwrap_or(method)
}

/// Route a JSON-RPC method call to the appropriate handler.
pub async fn dispatch(
    primal: &BarraCudaPrimal,
    method: &str,
    params: &Value,
    id: Value,
) -> JsonRpcResponse {
    match normalize_method(method) {
        "primal.info" => primal::info(primal, id),
        "primal.capabilities" => primal::capabilities(primal, id),
        "device.list" => device::list(primal, id).await,
        "device.probe" => device::probe(primal, id).await,
        "health.check" => health::health_check(primal, id).await,
        "tolerances.get" => health::tolerances_get(params, id),
        "validate.gpu_stack" => health::validate_gpu_stack(primal, id).await,
        "compute.dispatch" => compute::compute_dispatch(primal, params, id).await,
        "tensor.create" => tensor::tensor_create(primal, params, id).await,
        "tensor.matmul" => tensor::tensor_matmul(primal, params, id).await,
        "fhe.ntt" => fhe::fhe_ntt(primal, params, id).await,
        "fhe.pointwise_mul" => fhe::fhe_pointwise_mul(primal, params, id).await,
        _ => JsonRpcResponse::error(id, METHOD_NOT_FOUND, format!("Unknown method: {method}")),
    }
}

#[cfg(test)]
#[path = "../methods_tests.rs"]
mod tests;

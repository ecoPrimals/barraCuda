// SPDX-License-Identifier: AGPL-3.0-or-later
//! Health, tolerances, and GPU stack validation handlers.
//!
//! Per wateringHole `SEMANTIC_METHOD_NAMING_STANDARD.md` v2.2.0, the `health.*`
//! endpoints are non-negotiable ecosystem probes. Every primal MUST respond to:
//!
//! | Canonical Name     | Required Aliases            | Response shape            |
//! |--------------------|-----------------------------|---------------------------|
//! | `health.liveness`  | `ping`, `health`            | `{"status": "alive"}`     |
//! | `health.readiness` | (none)                      | `{"status": "ready", …}`  |
//! | `health.check`     | `status`, `check`           | `{"status": "healthy", …}`|

use super::super::jsonrpc::{INTERNAL_ERROR, JsonRpcResponse};
use crate::BarraCudaPrimal;
use serde_json::Value;

/// `health.liveness` — fast liveness probe (always alive if process is up).
///
/// Springs, orchestrators, and load balancers use this to determine whether the
/// primal process is alive. Responds unconditionally; does not probe hardware.
pub(super) fn health_liveness(id: Value) -> JsonRpcResponse {
    JsonRpcResponse::success(id, serde_json::json!({"status": "alive"}))
}

/// `health.readiness` — can the primal serve compute requests?
///
/// Returns `"ready"` if the primal is running and can accept work (even in
/// degraded/CPU-only mode). Returns `"not_ready"` during startup or after
/// stop.
pub(super) fn health_readiness(primal: &BarraCudaPrimal, id: Value) -> JsonRpcResponse {
    use crate::health::PrimalHealth;
    let ready = primal.is_ready();
    let has_gpu = primal.device().is_some();
    JsonRpcResponse::success(
        id,
        serde_json::json!({
            "status": if ready { "ready" } else { "not_ready" },
            "gpu_available": has_gpu,
        }),
    )
}

/// `health.check` — full health report (may probe hardware).
pub(super) async fn health_check(primal: &BarraCudaPrimal, id: Value) -> JsonRpcResponse {
    use crate::health::PrimalHealth;
    match primal.health_check().await {
        Ok(report) => JsonRpcResponse::success(
            id,
            serde_json::json!({
                "name": report.name,
                "version": report.version,
                "status": report.status.to_string(),
            }),
        ),
        Err(e) => JsonRpcResponse::error(id, INTERNAL_ERROR, e.to_string()),
    }
}

/// `barracuda.tolerances.get` — Get numerical tolerances for a named operation.
pub(super) fn tolerances_get(params: &Value, id: Value) -> JsonRpcResponse {
    let name = params
        .get("name")
        .and_then(|v| v.as_str())
        .unwrap_or("default");

    let (abs_tol, rel_tol) = match name {
        "fhe" => (0.0, 0.0),
        "f64" | "double" => (1e-12, 1e-10),
        "f32" | "float" => (1e-5, 1e-4),
        "df64" | "emulated_double" => (1e-10, 1e-8),
        _ => (1e-6, 1e-5),
    };

    JsonRpcResponse::success(
        id,
        serde_json::json!({
            "name": name,
            "abs_tol": abs_tol,
            "rel_tol": rel_tol,
        }),
    )
}

/// `barracuda.validate.gpu_stack` — Run GPU validation suite.
///
/// Uses 2×2 identity matrix: minimal size that validates matmul path without
/// unnecessary GPU memory/transfer overhead. Aligned with tarpc validate_gpu_stack.
pub(super) async fn validate_gpu_stack(primal: &BarraCudaPrimal, id: Value) -> JsonRpcResponse {
    let Some(dev) = primal.device() else {
        return JsonRpcResponse::error(id, INTERNAL_ERROR, "No GPU device available");
    };

    let mut results = Vec::new();

    // Matmul validation: 2×2 identity (minimal, fast, validates GPU stack)
    let dev_arc = std::sync::Arc::new(dev);
    let matmul_pass = {
        let eye = vec![1.0, 0.0, 0.0, 1.0];
        let input = vec![1.0, 2.0, 3.0, 4.0];
        match (
            barracuda::tensor::Tensor::from_data(&eye, vec![2, 2], dev_arc.clone()),
            barracuda::tensor::Tensor::from_data(&input, vec![2, 2], dev_arc.clone()),
        ) {
            (Ok(eye_t), Ok(inp)) => inp
                .matmul(&eye_t)
                .and_then(|out| out.to_vec())
                .is_ok_and(|v| v.iter().zip(&input).all(|(a, b)| (a - b).abs() < 1e-4)),
            _ => false,
        }
    };
    results.push(serde_json::json!({"test": "matmul_identity", "pass": matmul_pass}));

    // Tensor round-trip validation
    let roundtrip_pass = {
        let data = vec![std::f32::consts::PI, 2.71, 1.41, 0.57];
        barracuda::tensor::Tensor::from_data(&data, vec![4], dev_arc)
            .and_then(|t| t.to_vec())
            .is_ok_and(|v| v.iter().zip(&data).all(|(a, b)| (a - b).abs() < 1e-6))
    };
    results.push(serde_json::json!({"test": "tensor_roundtrip", "pass": roundtrip_pass}));

    let all_pass = results.iter().all(|r| r["pass"].as_bool().unwrap_or(false));

    JsonRpcResponse::success(
        id,
        serde_json::json!({
            "gpu_available": true,
            "status": if all_pass { "pass" } else { "partial_fail" },
            "tests": results,
        }),
    )
}

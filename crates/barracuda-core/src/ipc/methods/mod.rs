// SPDX-License-Identifier: AGPL-3.0-or-later
//! JSON-RPC method handlers for barraCuda IPC endpoints.
//!
//! Method names follow the wateringHole semantic naming standard:
//! `{domain}.{operation}` — no primal prefix on the wire. Primals are
//! identified by their transport endpoint, not by method name prefixes.
//!
//! For backward compatibility, the legacy `{namespace}.{domain}.{operation}`
//! form is also accepted and silently normalized.

mod batch;
mod compute;
mod device;
mod fhe;
mod health;
mod math;
mod primal;
mod tensor;

use super::jsonrpc::{JsonRpcResponse, METHOD_NOT_FOUND};
use crate::BarraCudaPrimal;
use serde_json::Value;

/// Semantic method names this primal supports (wateringHole `{domain}.{operation}`).
///
/// Single source of truth for which operations this primal provides.
/// Wire names are these exact strings — no primal prefix.
///
/// Per `SEMANTIC_METHOD_NAMING_STANDARD.md` v2.2.0, `health.liveness`,
/// `health.readiness`, `health.check`, and `capabilities.list` are
/// non-negotiable ecosystem probes registered as canonical names.
pub const REGISTERED_METHODS: &[&str] = &[
    // ── Ecosystem probes (non-negotiable) ─────────────────────────────
    "health.liveness",
    "health.readiness",
    "health.check",
    "capabilities.list",
    // ── Primal identity ───────────────────────────────────────────────
    "identity.get",
    "primal.info",
    "primal.capabilities",
    // ── Device ────────────────────────────────────────────────────────
    "device.list",
    "device.probe",
    "tolerances.get",
    "validate.gpu_stack",
    // ── Compute ───────────────────────────────────────────────────────
    "compute.dispatch",
    // ── Math & activation (CPU) ───────────────────────────────────────
    "math.sigmoid",
    "math.log2",
    "activation.fitts",
    "activation.hick",
    // ── Statistics (CPU) ──────────────────────────────────────────────
    "stats.mean",
    "stats.std_dev",
    "stats.weighted_mean",
    // ── Noise & RNG (CPU) ─────────────────────────────────────────────
    "noise.perlin2d",
    "noise.perlin3d",
    "rng.uniform",
    // ── Tensor (GPU) ──────────────────────────────────────────────────
    "tensor.create",
    "tensor.matmul",
    "tensor.add",
    "tensor.scale",
    "tensor.clamp",
    "tensor.reduce",
    "tensor.sigmoid",
    // ── Batch pipeline (GPU) ─────────────────────────────────────────
    "tensor.batch.submit",
    // ── FHE ───────────────────────────────────────────────────────────
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
///
/// Canonical names and required aliases per wateringHole standards:
/// - `health.liveness` + aliases `ping`, `health`
/// - `health.readiness` (no aliases)
/// - `health.check` + aliases `status`, `check`
/// - `capabilities.list` + alias `capability.list`
/// - `primal.capabilities` (legacy alias for `capabilities.list`)
pub async fn dispatch(
    primal: &BarraCudaPrimal,
    method: &str,
    params: &Value,
    id: Value,
) -> JsonRpcResponse {
    match normalize_method(method) {
        // Ecosystem probes
        "health.liveness" | "ping" | "health" => health::health_liveness(id),
        "health.readiness" => health::health_readiness(primal, id),
        "health.check" | "status" | "check" => health::health_check(primal, id).await,
        "identity.get" => primal::identity(id),
        "primal.info" => primal::info(primal, id),
        "primal.capabilities" | "capabilities.list" | "capability.list" => {
            primal::capabilities(primal, id)
        }
        // Device
        "device.list" => device::list(primal, id).await,
        "device.probe" => device::probe(primal, id).await,
        "tolerances.get" => health::tolerances_get(params, id),
        "validate.gpu_stack" => health::validate_gpu_stack(primal, id).await,
        // Compute
        "compute.dispatch" => compute::compute_dispatch(primal, params, id).await,
        // Math & activation (CPU)
        "math.sigmoid" => math::math_sigmoid(params, id),
        "math.log2" => math::math_log2(params, id),
        "activation.fitts" => math::activation_fitts(params, id),
        "activation.hick" => math::activation_hick(params, id),
        // Statistics (CPU)
        "stats.mean" => math::stats_mean(params, id),
        "stats.std_dev" => math::stats_std_dev(params, id),
        "stats.weighted_mean" => math::stats_weighted_mean(params, id),
        // Noise & RNG (CPU)
        "noise.perlin2d" => math::noise_perlin2d(params, id),
        "noise.perlin3d" => math::noise_perlin3d(params, id),
        "rng.uniform" => math::rng_uniform(params, id),
        // Tensor (GPU)
        "tensor.create" => tensor::tensor_create(primal, params, id).await,
        "tensor.matmul" => tensor::tensor_matmul(primal, params, id).await,
        "tensor.add" => tensor::tensor_add(primal, params, id).await,
        "tensor.scale" => tensor::tensor_scale(primal, params, id).await,
        "tensor.clamp" => tensor::tensor_clamp(primal, params, id).await,
        "tensor.reduce" => tensor::tensor_reduce(primal, params, id).await,
        "tensor.sigmoid" => tensor::tensor_sigmoid(primal, params, id).await,
        // Batch pipeline
        "tensor.batch.submit" => batch::tensor_batch_submit(primal, params, id).await,
        // FHE
        "fhe.ntt" => fhe::fhe_ntt(primal, params, id).await,
        "fhe.pointwise_mul" => fhe::fhe_pointwise_mul(primal, params, id).await,
        _ => JsonRpcResponse::error(id, METHOD_NOT_FOUND, format!("Unknown method: {method}")),
    }
}

#[cfg(test)]
#[path = "../methods_tests/mod.rs"]
mod tests;

#[cfg(test)]
#[path = "../methods_coverage_tests.rs"]
mod coverage_tests;

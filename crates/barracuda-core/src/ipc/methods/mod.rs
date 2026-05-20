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
mod graph;
mod health;
mod linalg;
mod math;
mod ml;
mod nautilus;
mod params;
mod precision;
mod primal;
mod spectral;
mod tensor;

use std::sync::LazyLock;

use super::jsonrpc::{JsonRpcResponse, METHOD_NOT_FOUND};
use super::method_gate::{CallerContext, MethodGate};
use crate::BarraCudaPrimal;
use serde_json::Value;

static GATE: LazyLock<MethodGate> = LazyLock::new(MethodGate::from_env);

/// Semantic method names this primal supports (wateringHole `{domain}.{operation}`).
///
/// Single source of truth for which operations this primal provides.
/// Wire names are these exact strings — no primal prefix.
///
/// Per `SEMANTIC_METHOD_NAMING_STANDARD.md` v2.2.0, `health.liveness`,
/// `health.readiness`, `health.check`, and `capabilities.list` are
/// non-negotiable ecosystem probes registered as canonical names.
pub(crate) const REGISTERED_METHODS: &[&str] = &[
    // ── Ecosystem probes (non-negotiable) ─────────────────────────────
    "health.liveness",
    "health.readiness",
    "health.check",
    "health.version",
    "capabilities.list",
    // ── Auth / gate introspection (JH-0) ───────────────────────────────
    "auth.check",
    "auth.mode",
    "auth.peer_info",
    // ── Primal identity & lifecycle ────────────────────────────────────
    "identity.get",
    "primal.info",
    "primal.capabilities",
    "primal.announce",
    // ── Device ────────────────────────────────────────────────────────
    "device.list",
    "device.probe",
    "tolerances.get",
    "validate.gpu_stack",
    // ── Precision routing advisory ────────────────────────────────────
    "precision.route",
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
    "stats.variance",
    "stats.correlation",
    "stats.pearson",
    "stats.spearman",
    "stats.covariance",
    "stats.weighted_mean",
    "stats.chi_squared",
    "stats.anova_oneway",
    "stats.shannon",
    "stats.entropy",
    "stats.fit_linear",
    "stats.empirical_spectral_density",
    "stats.simpson",
    "stats.bray_curtis",
    "stats.hill",
    "stats.fit_quadratic",
    "stats.fit_exponential",
    "stats.fit_logarithmic",
    "stats.rarefaction_curve",
    "stats.gamma_fit",
    "stats.gamma_cdf",
    "signal.detect_peaks",
    "signal.bandpass",
    "signal.derivative",
    // ── Linear algebra (CPU inline-data) ──────────────────────────────
    "linalg.solve",
    "linalg.eigenvalues",
    "stats.eigh",
    "linalg.svd",
    "linalg.qr",
    "linalg.graph_laplacian",
    // ── Numerical (CPU inline-data) ───────────────────────────────────
    "ode.step",
    // ── Graph / PGM (CPU inline-data) ─────────────────────────────────
    "graph.belief_propagation",
    // ── Spectral (CPU inline-data) ────────────────────────────────────
    "spectral.fft",
    "spectral.power_spectrum",
    "spectral.stft",
    // ── Activation (CPU inline-data) ──────────────────────────────────
    "activation.softmax",
    "activation.gelu",
    // ── ML (CPU inline-data) ──────────────────────────────────────────
    "ml.mlp_forward",
    "ml.mlp_train",
    "ml.attention",
    "ml.esn_predict",
    // ── Nautilus (server sessions, Path B) ─────────────────────────────
    "nautilus.create",
    "nautilus.observe",
    "nautilus.train",
    "nautilus.predict",
    "nautilus.export",
    "nautilus.import",
    // ── Noise & RNG (CPU) ─────────────────────────────────────────────
    "noise.perlin2d",
    "noise.perlin3d",
    "rng.uniform",
    // ── Tensor (GPU + inline) ─────────────────────────────────────────
    "tensor.create",
    "tensor.matmul",
    "tensor.matmul_inline",
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
    // ── BTSP Phase 3 ─────────────────────────────────────────────────
    "btsp.negotiate",
    "btsp.capabilities",
];

/// Normalize a method name: accepts both `{domain}.{operation}` (standard)
/// and legacy `{namespace}.{domain}.{operation}` (backward-compatible).
pub(crate) fn normalize_method(method: &str) -> &str {
    method
        .strip_prefix(crate::PRIMAL_NAMESPACE)
        .and_then(|s| s.strip_prefix('.'))
        .unwrap_or(method)
}

/// Extract a [`CallerContext`] from request params.
///
/// If params contain `"_auth": {"bearer": "..."}`, the token is extracted.
/// Otherwise returns a default loopback context (no token, no peer creds).
fn extract_caller_context(params: &Value) -> CallerContext {
    let bearer = params
        .get("_auth")
        .and_then(|a| a.get("bearer"))
        .and_then(|b| b.as_str())
        .map(String::from);
    CallerContext {
        bearer_token: bearer,
        peer: None,
        origin: super::method_gate::ConnectionOrigin::Loopback,
    }
}

/// Route a JSON-RPC method call to the appropriate handler.
///
/// Canonical names and required aliases per wateringHole standards:
/// - `health.liveness` + aliases `ping`, `health`
/// - `health.readiness` (no aliases)
/// - `health.check` + aliases `status`, `check`
/// - `capabilities.list` + alias `capability.list`
/// - `primal.capabilities` (legacy alias for `capabilities.list`)
///
/// Pre-dispatch: the [`MethodGate`] checks caller authorization per
/// `METHOD_GATE_STANDARD.md` v1.0 (JH-0). Default mode: permissive.
pub async fn dispatch(
    primal: &BarraCudaPrimal,
    method: &str,
    params: &Value,
    id: Value,
) -> JsonRpcResponse {
    let caller = extract_caller_context(params);
    let normalized = normalize_method(method);

    // Auth introspection methods — handled by the gate itself.
    match normalized {
        "auth.check" => return GATE.handle_auth_check(&caller, id),
        "auth.mode" => return GATE.handle_auth_mode(id),
        "auth.peer_info" => return GATE.handle_auth_peer_info(&caller, id),
        _ => {}
    }

    // Pre-dispatch gate check.
    if let Err(denied) = GATE.check(normalized, &caller, &id) {
        return denied;
    }

    match normalized {
        // Ecosystem probes
        "health.liveness" | "ping" | "health" => health::health_liveness(id),
        "health.readiness" => health::health_readiness(primal, id),
        "health.check" | "status" | "check" => health::health_check(primal, id).await,
        "health.version" => health::health_version(id),
        "identity.get" => primal::identity(id),
        "primal.info" => primal::info(primal, id),
        "primal.capabilities" | "capabilities.list" | "capability.list" => {
            primal::capabilities(primal, id)
        }
        "primal.announce" => primal::announce(primal, id),
        // Device
        "device.list" => device::list(primal, id).await,
        "device.probe" => device::probe(primal, id).await,
        "tolerances.get" => health::tolerances_get(params, id),
        "validate.gpu_stack" => health::validate_gpu_stack(primal, id).await,
        // Precision routing advisory
        "precision.route" => precision::precision_route(primal, params, id),
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
        "stats.variance" => math::stats_variance(params, id),
        "stats.correlation" | "stats.pearson" => math::stats_correlation(params, id),
        "stats.spearman" => math::stats_spearman(params, id),
        "stats.covariance" => math::stats_covariance(params, id),
        "stats.weighted_mean" => math::stats_weighted_mean(params, id),
        "stats.chi_squared" => math::stats_chi_squared(params, id),
        "stats.anova_oneway" => math::stats_anova_oneway(params, id),
        "stats.shannon" | "stats.entropy" => math::stats_shannon(params, id),
        "stats.fit_linear" => math::stats_fit_linear(params, id),
        "stats.empirical_spectral_density" => math::stats_empirical_spectral_density(params, id),
        "stats.simpson" => math::stats_simpson(params, id),
        "stats.bray_curtis" => math::stats_bray_curtis(params, id),
        "stats.hill" => math::stats_hill(params, id),
        "stats.fit_quadratic" => math::stats_fit_quadratic(params, id),
        "stats.fit_exponential" => math::stats_fit_exponential(params, id),
        "stats.fit_logarithmic" => math::stats_fit_logarithmic(params, id),
        "stats.rarefaction_curve" => math::stats_rarefaction_curve(params, id),
        "stats.gamma_fit" => math::stats_gamma_fit(params, id),
        "stats.gamma_cdf" => math::stats_gamma_cdf(params, id),
        // Signal processing
        "signal.detect_peaks" => math::signal_detect_peaks(params, id),
        "signal.bandpass" => math::signal_bandpass(params, id),
        "signal.derivative" => math::signal_derivative(params, id),
        // Linear algebra (CPU inline-data)
        "linalg.solve" => linalg::linalg_solve(params, id),
        "linalg.eigenvalues" | "stats.eigh" => linalg::linalg_eigenvalues(params, id),
        "linalg.svd" => linalg::linalg_svd(params, id),
        "linalg.qr" => linalg::linalg_qr(params, id),
        "linalg.graph_laplacian" => graph::linalg_graph_laplacian(params, id),
        // Numerical (CPU inline-data)
        "ode.step" => math::ode_step(params, id),
        // Graph / PGM (CPU inline-data)
        "graph.belief_propagation" => graph::graph_belief_propagation(params, id),
        // Spectral (CPU inline-data)
        "spectral.fft" => spectral::spectral_fft(params, id),
        "spectral.power_spectrum" => spectral::spectral_power_spectrum(params, id),
        "spectral.stft" => spectral::spectral_stft(params, id),
        // Activation (CPU inline-data)
        "activation.softmax" => math::activation_softmax(params, id),
        "activation.gelu" => math::activation_gelu(params, id),
        // ML (CPU inline-data)
        "ml.mlp_forward" => ml::ml_mlp_forward(params, id),
        "ml.mlp_train" => ml::ml_mlp_train(params, id),
        "ml.attention" => ml::ml_attention(params, id),
        "ml.esn_predict" => ml::ml_esn_predict(params, id),
        // Nautilus (server sessions, Path B)
        "nautilus.create" => nautilus::nautilus_create(params, id),
        "nautilus.observe" => nautilus::nautilus_observe(params, id),
        "nautilus.train" => nautilus::nautilus_train(params, id),
        "nautilus.predict" => nautilus::nautilus_predict(params, id),
        "nautilus.export" => nautilus::nautilus_export(params, id),
        "nautilus.import" => nautilus::nautilus_import(params, id),
        // Noise & RNG (CPU)
        "noise.perlin2d" => math::noise_perlin2d(params, id),
        "noise.perlin3d" => math::noise_perlin3d(params, id),
        "rng.uniform" => math::rng_uniform(params, id),
        // Tensor (GPU)
        "tensor.create" => tensor::tensor_create(primal, params, id).await,
        "tensor.matmul" => tensor::tensor_matmul(primal, params, id).await,
        "tensor.matmul_inline" => tensor::tensor_matmul_inline(params, id),
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
        // BTSP Phase 3 — negotiate handled at transport layer; reaching
        // dispatch means no authenticated session.
        "btsp.negotiate" => JsonRpcResponse::error(
            id,
            -32600,
            "btsp.negotiate requires an authenticated BTSP session (Phase 1 handshake first)",
        ),
        "btsp.capabilities" => health::btsp_capabilities(id),
        _ => JsonRpcResponse::error(id, METHOD_NOT_FOUND, format!("Unknown method: {method}")),
    }
}

#[cfg(test)]
#[path = "../methods_tests/mod.rs"]
mod tests;

#[cfg(test)]
#[path = "../methods_coverage_tests/mod.rs"]
mod coverage_tests;

// SPDX-License-Identifier: AGPL-3.0-or-later
//! Primal identity and capability advertisement handlers.
//!
//! All advertised capabilities, provides, and domains are derived from the
//! IPC dispatch table — zero hardcoded domain lists. The primal only has
//! self-knowledge and discovers others at runtime.

use super::super::jsonrpc::JsonRpcResponse;
use super::REGISTERED_METHODS;
use crate::BarraCudaPrimal;
use serde_json::Value;

/// `barracuda.primal.info` — Primal identity for runtime discovery.
///
/// Other primals call this method (not hardcoded names) to identify barraCuda.
pub(super) fn info(_primal: &BarraCudaPrimal, id: Value) -> JsonRpcResponse {
    JsonRpcResponse::success(
        id,
        serde_json::json!({
            "primal": crate::PRIMAL_NAME,
            "version": env!("CARGO_PKG_VERSION"),
            "protocol": "json-rpc-2.0",
            "namespace": crate::PRIMAL_NAMESPACE,
            "license": "AGPL-3.0-or-later",
        }),
    )
}

/// `capabilities.list` / `primal.capabilities` — Advertise capabilities.
///
/// Wire Standard L2 compliant: `{primal, version, methods}` envelope with
/// structured capability groups and hardware state. All lists derived from
/// the dispatch table via [`crate::discovery`] — zero hardcoded domain lists.
pub(super) fn capabilities(primal: &BarraCudaPrimal, id: Value) -> JsonRpcResponse {
    let has_gpu = primal.device().is_some();
    let has_sovereign = primal.has_sovereign_dispatch();
    let has_f64 = primal.compute_device().is_some_and(|d| d.has_f64_shaders());
    let has_spirv = primal.device().is_some_and(|d| d.has_spirv_passthrough());

    let version = env!("CARGO_PKG_VERSION");
    let provides_list: Vec<_> = crate::discovery::provides()
        .iter()
        .map(|id_str| serde_json::json!({ "id": id_str, "version": version }))
        .collect();

    let provided_capabilities = crate::discovery::provided_capability_groups(version);

    JsonRpcResponse::success(
        id,
        serde_json::json!({
            "primal": crate::PRIMAL_NAMESPACE,
            "version": version,
            "methods": REGISTERED_METHODS,
            "provided_capabilities": provided_capabilities,
            "consumed_capabilities": ["shader.compile.cpu", "shader.validate", "compute.dispatch", "compute.dispatch.submit"],
            "provides": provides_list,
            "requires": [
                { "id": "shader.compile", "version": ">=0.1.0", "optional": true },
            ],
            "domains": crate::discovery::capabilities(),
            "hardware": {
                "gpu_available": has_gpu,
                "sovereign_ipc": has_sovereign,
                "f64_shaders": has_f64,
                "spirv_passthrough": has_spirv,
            },
            "protocol": "jsonrpc-2.0",
            "transport": ["uds", "tcp"],
        }),
    )
}

/// `primal.announce` — Atomic self-registration payload for biomeOS composition.
///
/// Returns the identity, capabilities, signal tier, cost hints, and latency
/// estimates so biomeOS Neural API can register this primal into the composition
/// graph and compute routing weights without additional round-trips.
///
/// Schema aligned with biomeOS v3.68+ Neural API (Wave 43).
/// All composition metadata derived from [`crate::discovery::composition_hints`].
pub(super) fn announce(primal: &BarraCudaPrimal, id: Value) -> JsonRpcResponse {
    let version = env!("CARGO_PKG_VERSION");
    let has_gpu = primal.device().is_some();
    let socket = crate::ipc::transport::discovery_socket_path();
    let hints = crate::discovery::composition_hints();

    JsonRpcResponse::success(
        id,
        serde_json::json!({
            "primal": crate::PRIMAL_NAME,
            "namespace": crate::PRIMAL_NAMESPACE,
            "version": version,
            "domain": crate::PRIMAL_DOMAIN,
            "methods": REGISTERED_METHODS,
            "capabilities": hints.capabilities,
            "signal_tiers": hints.signal_tiers,
            "socket": socket,
            "cost_hints": hints.cost_hints,
            "latency_estimates": hints.latency_estimates,
            "hardware": {
                "gpu_available": has_gpu,
            },
            "transport": ["uds", "tcp"],
            "license": "AGPL-3.0-or-later",
        }),
    )
}

/// `identity.get` — Lightweight primal identity for observability.
///
/// Wire Standard L2: returns `{primal, version, domain, license}`.
pub(super) fn identity(id: Value) -> JsonRpcResponse {
    JsonRpcResponse::success(
        id,
        serde_json::json!({
            "primal": crate::PRIMAL_NAMESPACE,
            "version": env!("CARGO_PKG_VERSION"),
            "domain": crate::PRIMAL_DOMAIN,
            "license": "AGPL-3.0-or-later",
        }),
    )
}

/// `method.describe` — Runtime method introspection for distributed compositions.
///
/// Returns the parameter schema, description, access level, and examples for a
/// given method name. Enables callers to self-correct param format mismatches
/// without hardcoded knowledge of each primal's API surface.
pub(super) fn method_describe(params: &Value, id: Value) -> JsonRpcResponse {
    let method_name = params
        .get("method")
        .and_then(|v| v.as_str())
        .or_else(|| params.get("name").and_then(|v| v.as_str()));

    let Some(name) = method_name else {
        return JsonRpcResponse::error(
            id,
            -32602,
            "Missing required param: \"method\" (string)",
        );
    };

    let normalized = super::normalize_method(name);

    if !REGISTERED_METHODS.contains(&normalized) {
        return JsonRpcResponse::error(
            id,
            -32601,
            format!("Unknown method: {normalized}"),
        );
    }

    let descriptor = method_descriptor(normalized);
    JsonRpcResponse::success(id, descriptor)
}

/// Structured descriptor for a method — params, description, access, examples.
fn method_descriptor(method: &str) -> Value {
    let (description, params_schema, access) = match method {
        // Ecosystem probes
        "health.liveness" => ("Returns alive status", serde_json::json!({}), "public"),
        "health.readiness" => ("Returns readiness with GPU/lifecycle state", serde_json::json!({}), "public"),
        "health.check" => ("Full health report with device info", serde_json::json!({}), "public"),
        "health.version" => ("Version and build metadata", serde_json::json!({}), "public"),
        "capabilities.list" => ("Full capability advertisement", serde_json::json!({}), "public"),
        // Identity
        "identity.get" => ("Lightweight primal identity", serde_json::json!({}), "public"),
        "primal.info" => ("Full primal info with capabilities", serde_json::json!({}), "public"),
        "primal.capabilities" => ("Alias for capabilities.list", serde_json::json!({}), "public"),
        "primal.announce" => ("Composition registration payload for biomeOS", serde_json::json!({}), "public"),
        // Device
        "device.list" => ("List available compute devices", serde_json::json!({}), "public"),
        "device.probe" => ("Probe GPU capabilities", serde_json::json!({}), "public"),
        "tolerances.get" => ("Get precision tolerances for an operation", serde_json::json!({"op": "string (operation name)"}), "public"),
        "validate.gpu_stack" => ("Validate GPU driver and shader stack", serde_json::json!({}), "public"),
        // Precision
        "precision.route" => ("Advisory: route computation to best-precision path", serde_json::json!({"op": "string", "precision": "string? (f32|f64)"}), "public"),
        // Compute
        "compute.dispatch" => ("Dispatch a compute operation", serde_json::json!({"op": "string", "data": "array<f64>", "shape": "array<u32>?"}), "protected"),
        "compute.dispatch.capabilities" => ("List dispatchable operations", serde_json::json!({}), "public"),
        "compute.dispatch.submit" => ("Submit compute to peer", serde_json::json!({"op": "string", "data": "array<f64>"}), "protected"),
        "compute.dispatch.result" => ("Retrieve dispatch result", serde_json::json!({"job_id": "string"}), "protected"),
        // Math
        "math.sigmoid" => ("Sigmoid activation on scalar or array", serde_json::json!({"x": "f64 | array<f64>"}), "public"),
        "math.log2" => ("Base-2 logarithm", serde_json::json!({"x": "f64 | array<f64>"}), "public"),
        "activation.fitts" => ("Fitts' law index of difficulty", serde_json::json!({"distance": "f64", "width": "f64"}), "public"),
        "activation.hick" => ("Hick-Hyman reaction time", serde_json::json!({"n": "u32 (number of choices)"}), "public"),
        "activation.softmax" => ("Softmax over array", serde_json::json!({"data": "array<f64>"}), "public"),
        "activation.gelu" => ("GELU activation", serde_json::json!({"data": "array<f64>"}), "public"),
        // Stats
        "stats.mean" => ("Arithmetic mean", serde_json::json!({"data": "array<f64>"}), "public"),
        "stats.std_dev" => ("Standard deviation", serde_json::json!({"data": "array<f64>"}), "public"),
        "stats.variance" => ("Variance", serde_json::json!({"data": "array<f64>"}), "public"),
        "stats.correlation" | "stats.pearson" => ("Pearson correlation coefficient", serde_json::json!({"x": "array<f64>", "y": "array<f64>"}), "public"),
        "stats.spearman" => ("Spearman rank correlation", serde_json::json!({"x": "array<f64>", "y": "array<f64>"}), "public"),
        "stats.covariance" => ("Sample covariance", serde_json::json!({"x": "array<f64>", "y": "array<f64>"}), "public"),
        "stats.weighted_mean" => ("Weighted mean", serde_json::json!({"data": "array<f64>", "weights": "array<f64>"}), "public"),
        "stats.chi_squared" => ("Chi-squared test statistic", serde_json::json!({"observed": "array<f64>", "expected": "array<f64>"}), "public"),
        "stats.anova_oneway" => ("One-way ANOVA F-statistic", serde_json::json!({"groups": "array<array<f64>>"}), "public"),
        "stats.shannon" | "stats.entropy" => ("Shannon entropy", serde_json::json!({"data": "array<f64>"}), "public"),
        "stats.fit_linear" => ("Linear regression fit", serde_json::json!({"x": "array<f64>", "y": "array<f64>"}), "public"),
        "stats.empirical_spectral_density" => ("Empirical spectral density estimate", serde_json::json!({"data": "array<f64>"}), "public"),
        "stats.simpson" => ("Simpson diversity index", serde_json::json!({"data": "array<f64>"}), "public"),
        "stats.bray_curtis" => ("Bray-Curtis dissimilarity", serde_json::json!({"a": "array<f64>", "b": "array<f64>"}), "public"),
        "stats.hill" => ("Hill diversity number", serde_json::json!({"data": "array<f64>", "q": "f64"}), "public"),
        "stats.fit_quadratic" => ("Quadratic regression fit", serde_json::json!({"x": "array<f64>", "y": "array<f64>"}), "public"),
        "stats.fit_exponential" => ("Exponential regression fit", serde_json::json!({"x": "array<f64>", "y": "array<f64>"}), "public"),
        "stats.fit_logarithmic" => ("Logarithmic regression fit", serde_json::json!({"x": "array<f64>", "y": "array<f64>"}), "public"),
        "stats.rarefaction_curve" => ("Rarefaction curve computation", serde_json::json!({"data": "array<f64>", "steps": "u32?"}), "public"),
        "stats.gamma_fit" => ("Fit gamma distribution", serde_json::json!({"data": "array<f64>"}), "public"),
        "stats.gamma_cdf" => ("Gamma CDF evaluation", serde_json::json!({"x": "f64", "shape": "f64", "rate": "f64"}), "public"),
        // Signal
        "signal.detect_peaks" => ("Peak detection in time series", serde_json::json!({"data": "array<f64>", "threshold": "f64?"}), "public"),
        "signal.bandpass" => ("Bandpass filter", serde_json::json!({"data": "array<f64>", "low": "f64", "high": "f64", "sample_rate": "f64"}), "public"),
        "signal.derivative" => ("Numerical derivative", serde_json::json!({"data": "array<f64>", "dt": "f64?"}), "public"),
        // Linalg
        "linalg.solve" => ("Solve linear system Ax=b", serde_json::json!({"a": "array<array<f64>>", "b": "array<f64>"}), "public"),
        "linalg.eigenvalues" => ("Eigenvalue decomposition", serde_json::json!({"matrix": "array<array<f64>>"}), "public"),
        "linalg.svd" => ("Singular value decomposition", serde_json::json!({"matrix": "array<array<f64>>"}), "public"),
        "linalg.qr" => ("QR decomposition", serde_json::json!({"matrix": "array<array<f64>>"}), "public"),
        "linalg.graph_laplacian" => ("Graph Laplacian from adjacency matrix", serde_json::json!({"adjacency": "array<array<f64>>"}), "public"),
        // Numerical
        "ode.step" => ("Single ODE integration step", serde_json::json!({"state": "array<f64>", "dt": "f64", "method": "string? (euler|rk4)"}), "public"),
        // Graph
        "graph.belief_propagation" => ("Belief propagation on factor graph", serde_json::json!({"factors": "array", "messages": "array?", "iterations": "u32?"}), "public"),
        // Spectral
        "spectral.fft" => ("Fast Fourier transform", serde_json::json!({"data": "array<f64>"}), "public"),
        "spectral.power_spectrum" => ("Power spectral density", serde_json::json!({"data": "array<f64>"}), "public"),
        "spectral.stft" => ("Short-time Fourier transform", serde_json::json!({"data": "array<f64>", "window_size": "u32", "hop_size": "u32?"}), "public"),
        // ML
        "ml.mlp_forward" => ("Forward pass through trained MLP", serde_json::json!({"model": "object (SimpleMlp)", "input": "array<f64>"}), "public"),
        "ml.mlp_train" => ("Train single-layer perceptron", serde_json::json!({"input_dim": "u32", "output_dim": "u32", "data": "array<array<f64>>", "labels": "array<array<f64>>", "epochs": "u32?", "lr": "f64?"}), "public"),
        "ml.mlp_infer" => ("Batch inference on trained model", serde_json::json!({"model": "object (SimpleMlp)", "inputs": "array<array<f64>>"}), "public"),
        "ml.mlp_save" => ("Save model to binary format", serde_json::json!({"model": "object (SimpleMlp)", "path": "string"}), "public"),
        "ml.mlp_load" => ("Load model from file", serde_json::json!({"path": "string"}), "public"),
        "ml.perceptron_train" => ("End-to-end perceptron training pipeline", serde_json::json!({"input_dim": "u32", "output_dim": "u32", "data": "array<array<f64>>", "labels": "array<array<f64>>", "epochs": "u32?", "lr": "f64?"}), "public"),
        "ml.attention" => ("Scaled dot-product attention", serde_json::json!({"q": "array<array<f64>>", "k": "array<array<f64>>", "v": "array<array<f64>>"}), "public"),
        "ml.esn_predict" => ("Echo State Network prediction", serde_json::json!({"input": "array<f64>", "reservoir_size": "u32?"}), "public"),
        // Nautilus
        "nautilus.create" => ("Create nautilus session", serde_json::json!({"config": "object?"}), "public"),
        "nautilus.observe" => ("Feed observation data", serde_json::json!({"session_id": "string", "data": "array<f64>"}), "public"),
        "nautilus.train" => ("Train nautilus model", serde_json::json!({"session_id": "string"}), "public"),
        "nautilus.predict" => ("Predict from nautilus model", serde_json::json!({"session_id": "string", "horizon": "u32?"}), "public"),
        "nautilus.export" => ("Export trained model", serde_json::json!({"session_id": "string"}), "public"),
        "nautilus.import" => ("Import model into session", serde_json::json!({"model": "object"}), "public"),
        // Noise & RNG
        "noise.perlin2d" => ("2D Perlin noise", serde_json::json!({"x": "f64", "y": "f64", "seed": "u32?"}), "public"),
        "noise.perlin3d" => ("3D Perlin noise", serde_json::json!({"x": "f64", "y": "f64", "z": "f64", "seed": "u32?"}), "public"),
        "rng.uniform" => ("Uniform random numbers", serde_json::json!({"count": "u32", "min": "f64?", "max": "f64?"}), "public"),
        // Tensor
        "tensor.create" => ("Create GPU tensor from data", serde_json::json!({"data": "array<f64>", "shape": "array<u32>"}), "protected"),
        "tensor.matmul" => ("GPU matrix multiplication (tensor handles)", serde_json::json!({"a": "string (handle)", "b": "string (handle)"}), "protected"),
        "tensor.matmul_inline" => ("CPU matrix multiplication (inline data)", serde_json::json!({"a": "array<array<f64>>", "b": "array<array<f64>>"}), "public"),
        "tensor.add" => ("Element-wise tensor addition", serde_json::json!({"a": "string (handle)", "b": "string (handle)"}), "protected"),
        "tensor.scale" => ("Scalar multiplication", serde_json::json!({"tensor": "string (handle)", "scalar": "f64"}), "protected"),
        "tensor.clamp" => ("Clamp tensor values", serde_json::json!({"tensor": "string (handle)", "min": "f64", "max": "f64"}), "protected"),
        "tensor.reduce" => ("Reduction operation (sum, mean, max, min)", serde_json::json!({"tensor": "string (handle)", "op": "string (sum|mean|max|min)"}), "protected"),
        "tensor.sigmoid" => ("Sigmoid activation on tensor", serde_json::json!({"tensor": "string (handle)"}), "protected"),
        "tensor.batch.submit" => ("Submit batch of tensor operations", serde_json::json!({"ops": "array<object>"}), "protected"),
        // FHE
        "fhe.ntt" => ("Number-theoretic transform for FHE", serde_json::json!({"data": "array<u64>", "modulus": "u64"}), "protected"),
        "fhe.pointwise_mul" => ("Pointwise polynomial multiplication", serde_json::json!({"a": "array<u64>", "b": "array<u64>", "modulus": "u64"}), "protected"),
        // Mesh
        "mesh.trust_verify" => ("Verify BTSP trust with peer", serde_json::json!({"peer_id": "string", "token": "string?"}), "public"),
        "mesh.health" => ("Cross-gate mesh health probe", serde_json::json!({}), "public"),
        // Auth
        "auth.check" => ("Check caller authentication status", serde_json::json!({}), "public"),
        "auth.mode" => ("Report current auth mode", serde_json::json!({}), "public"),
        "auth.peer_info" => ("Report peer connection info", serde_json::json!({}), "public"),
        // BTSP
        "btsp.negotiate" => ("Initiate BTSP Phase 1 handshake", serde_json::json!({"peer_id": "string"}), "protected"),
        "btsp.capabilities" => ("List supported BTSP capabilities", serde_json::json!({}), "public"),
        // Method introspection
        "method.describe" => ("Describe a method's params, access level, and purpose", serde_json::json!({"method": "string (method name)"}), "public"),
        _ => ("Unknown method", serde_json::json!({}), "unknown"),
    };

    serde_json::json!({
        "method": method,
        "description": description,
        "params": params_schema,
        "access": access,
        "primal": crate::PRIMAL_NAMESPACE,
        "version": env!("CARGO_PKG_VERSION"),
    })
}

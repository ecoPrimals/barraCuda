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
            "protocol": crate::PROTOCOL_ID,
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
            "protocol": crate::PROTOCOL_ID,
            "transport": ["uds", "tcp"],
        }),
    )
}

/// `protocols.list` — Advertise supported protocols and negotiation capability.
///
/// G65 single-socket: clients can send `PROTOCOLS: tarpc,jsonrpc\n` on the
/// primary socket to negotiate the wire protocol. Returns both G65 negotiation
/// info and C2 dual-socket endpoints for backward compatibility.
pub(super) fn protocols_list(id: Value) -> JsonRpcResponse {
    let version = env!("CARGO_PKG_VERSION");
    let primary_endpoint = crate::ipc::transport::discovery_socket_path();

    let supported = crate::ipc::IpcProtocol::supported();
    let supported_names: Vec<&str> = supported.iter().map(|p| p.negotiation_name()).collect();

    let jsonrpc_proto = serde_json::json!({
        "name": "jsonrpc",
        "endpoint": format!("unix://{primary_endpoint}"),
        "enabled": true,
        "priority": 2,
    });

    #[cfg(unix)]
    let protocols = {
        let tarpc_path = crate::ipc::transport::default_tarpc_socket_path();
        vec![
            serde_json::json!({
                "name": "tarpc",
                "version": "0.37",
                "endpoint": format!("unix://{}", tarpc_path.display()),
                "enabled": true,
                "priority": 1,
            }),
            jsonrpc_proto,
        ]
    };

    #[cfg(not(unix))]
    let protocols = vec![jsonrpc_proto];

    JsonRpcResponse::success(
        id,
        serde_json::json!({
            "primal": crate::PRIMAL_NAME,
            "version": version,
            "protocols": protocols,
            "negotiation": {
                "g65": true,
                "supported": supported_names,
                "endpoint": format!("unix://{primary_endpoint}"),
                "header": "PROTOCOLS: tarpc,jsonrpc",
            },
            "dual_socket": cfg!(unix),
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
        return JsonRpcResponse::error(id, -32602, "Missing required param: \"method\" (string)");
    };

    let normalized = super::normalize_method(name);

    if !REGISTERED_METHODS.contains(&normalized) {
        return JsonRpcResponse::error(id, -32601, format!("Unknown method: {normalized}"));
    }

    let descriptor = method_descriptor(normalized);
    JsonRpcResponse::success(id, descriptor)
}

fn describe_health(method: &str) -> Option<(&str, Value, &str)> {
    match method {
        "health.liveness" => Some(("Returns alive status", serde_json::json!({}), "public")),
        "health.readiness" => Some((
            "Returns readiness with GPU/lifecycle state",
            serde_json::json!({}),
            "public",
        )),
        "health.check" => Some((
            "Full health report with device info",
            serde_json::json!({}),
            "public",
        )),
        "health.version" => Some((
            "Version and build metadata",
            serde_json::json!({}),
            "public",
        )),
        "capabilities.list" => Some((
            "Full capability advertisement",
            serde_json::json!({}),
            "public",
        )),
        "protocols.list" => Some((
            "Supported protocols with G65 negotiation and C2 endpoints",
            serde_json::json!({}),
            "public",
        )),
        _ => None,
    }
}

fn describe_identity(method: &str) -> Option<(&str, Value, &str)> {
    match method {
        "identity.get" => Some((
            "Lightweight primal identity",
            serde_json::json!({}),
            "public",
        )),
        "primal.info" => Some((
            "Full primal info with capabilities",
            serde_json::json!({}),
            "public",
        )),
        "primal.capabilities" => Some((
            "Alias for capabilities.list",
            serde_json::json!({}),
            "public",
        )),
        "primal.announce" => Some((
            "Composition registration payload for biomeOS",
            serde_json::json!({}),
            "public",
        )),
        _ => None,
    }
}

fn describe_device(method: &str) -> Option<(&str, Value, &str)> {
    match method {
        "device.list" => Some((
            "List available compute devices",
            serde_json::json!({}),
            "public",
        )),
        "device.pool" => Some((
            "Multi-GPU pool status and per-device diagnostics",
            serde_json::json!({}),
            "public",
        )),
        "device.probe" => Some(("Probe GPU capabilities", serde_json::json!({}), "public")),
        "device.video_codecs" => Some((
            "Probe available video codec backends (NVENC, VAAPI, software)",
            serde_json::json!({}),
            "public",
        )),
        "tolerances.get" => Some((
            "Get precision tolerances for an operation",
            serde_json::json!({"op": "string (operation name)"}),
            "public",
        )),
        "validate.gpu_stack" => Some((
            "Validate GPU driver and shader stack",
            serde_json::json!({}),
            "public",
        )),
        "precision.route" => Some((
            "Advisory: route computation to best-precision path",
            serde_json::json!({"op": "string", "precision": "string? (f32|f64)"}),
            "public",
        )),
        _ => None,
    }
}

fn describe_compute(method: &str) -> Option<(&str, Value, &str)> {
    match method {
        "compute.dispatch" => Some((
            "Dispatch a compute operation",
            serde_json::json!({"op": "string", "data": "array<f64>", "shape": "array<u32>?"}),
            "protected",
        )),
        "compute.dispatch.capabilities" => Some((
            "List dispatchable operations",
            serde_json::json!({}),
            "public",
        )),
        "compute.dispatch.submit" => Some((
            "Submit compute to peer",
            serde_json::json!({"op": "string", "data": "array<f64>"}),
            "protected",
        )),
        "compute.dispatch.result" => Some((
            "Retrieve dispatch result",
            serde_json::json!({"job_id": "string"}),
            "protected",
        )),
        "math.sigmoid" => Some((
            "Sigmoid activation on scalar or array",
            serde_json::json!({"x": "f64 | array<f64>"}),
            "public",
        )),
        "math.log2" => Some((
            "Base-2 logarithm",
            serde_json::json!({"x": "f64 | array<f64>"}),
            "public",
        )),
        "activation.fitts" => Some((
            "Fitts' law index of difficulty",
            serde_json::json!({"distance": "f64", "width": "f64"}),
            "public",
        )),
        "activation.hick" => Some((
            "Hick-Hyman reaction time",
            serde_json::json!({"n": "u32 (number of choices)"}),
            "public",
        )),
        "activation.softmax" => Some((
            "Softmax over array",
            serde_json::json!({"data": "array<f64>"}),
            "public",
        )),
        "activation.gelu" => Some((
            "GELU activation",
            serde_json::json!({"data": "array<f64>"}),
            "public",
        )),
        _ => None,
    }
}

fn describe_stats(method: &str) -> Option<(&str, Value, &str)> {
    match method {
        "stats.mean" => Some((
            "Arithmetic mean",
            serde_json::json!({"data": "array<f64>"}),
            "public",
        )),
        "stats.std_dev" => Some((
            "Standard deviation",
            serde_json::json!({"data": "array<f64>"}),
            "public",
        )),
        "stats.variance" => Some((
            "Variance",
            serde_json::json!({"data": "array<f64>"}),
            "public",
        )),
        "stats.correlation" | "stats.pearson" => Some((
            "Pearson correlation coefficient",
            serde_json::json!({"x": "array<f64>", "y": "array<f64>"}),
            "public",
        )),
        "stats.spearman" => Some((
            "Spearman rank correlation",
            serde_json::json!({"x": "array<f64>", "y": "array<f64>"}),
            "public",
        )),
        "stats.covariance" => Some((
            "Sample covariance",
            serde_json::json!({"x": "array<f64>", "y": "array<f64>"}),
            "public",
        )),
        "stats.weighted_mean" => Some((
            "Weighted mean",
            serde_json::json!({"data": "array<f64>", "weights": "array<f64>"}),
            "public",
        )),
        "stats.chi_squared" => Some((
            "Chi-squared test statistic",
            serde_json::json!({"observed": "array<f64>", "expected": "array<f64>"}),
            "public",
        )),
        "stats.anova_oneway" => Some((
            "One-way ANOVA F-statistic",
            serde_json::json!({"groups": "array<array<f64>>"}),
            "public",
        )),
        "stats.shannon" | "stats.entropy" => Some((
            "Shannon entropy",
            serde_json::json!({"data": "array<f64>"}),
            "public",
        )),
        "stats.fit_linear" => Some((
            "Linear regression fit",
            serde_json::json!({"x": "array<f64>", "y": "array<f64>"}),
            "public",
        )),
        "stats.empirical_spectral_density" => Some((
            "Empirical spectral density estimate",
            serde_json::json!({"data": "array<f64>"}),
            "public",
        )),
        "stats.simpson" => Some((
            "Simpson diversity index",
            serde_json::json!({"data": "array<f64>"}),
            "public",
        )),
        "stats.bray_curtis" => Some((
            "Bray-Curtis dissimilarity",
            serde_json::json!({"a": "array<f64>", "b": "array<f64>"}),
            "public",
        )),
        "stats.hill" => Some((
            "Hill diversity number",
            serde_json::json!({"data": "array<f64>", "q": "f64"}),
            "public",
        )),
        "stats.fit_quadratic" => Some((
            "Quadratic regression fit",
            serde_json::json!({"x": "array<f64>", "y": "array<f64>"}),
            "public",
        )),
        "stats.fit_exponential" => Some((
            "Exponential regression fit",
            serde_json::json!({"x": "array<f64>", "y": "array<f64>"}),
            "public",
        )),
        "stats.fit_logarithmic" => Some((
            "Logarithmic regression fit",
            serde_json::json!({"x": "array<f64>", "y": "array<f64>"}),
            "public",
        )),
        "stats.rarefaction_curve" => Some((
            "Rarefaction curve computation",
            serde_json::json!({"data": "array<f64>", "steps": "u32?"}),
            "public",
        )),
        "stats.gamma_fit" => Some((
            "Fit gamma distribution",
            serde_json::json!({"data": "array<f64>"}),
            "public",
        )),
        "stats.gamma_cdf" => Some((
            "Gamma CDF evaluation",
            serde_json::json!({"x": "f64", "shape": "f64", "rate": "f64"}),
            "public",
        )),
        _ => None,
    }
}

fn describe_signal(method: &str) -> Option<(&str, Value, &str)> {
    match method {
        "signal.detect_peaks" => Some((
            "Peak detection in time series",
            serde_json::json!({"data": "array<f64>", "threshold": "f64?"}),
            "public",
        )),
        "signal.bandpass" => Some((
            "Bandpass filter",
            serde_json::json!({"data": "array<f64>", "low": "f64", "high": "f64", "sample_rate": "f64"}),
            "public",
        )),
        "signal.derivative" => Some((
            "Numerical derivative",
            serde_json::json!({"data": "array<f64>", "dt": "f64?"}),
            "public",
        )),
        _ => None,
    }
}

fn describe_linalg(method: &str) -> Option<(&str, Value, &str)> {
    match method {
        "linalg.solve" => Some((
            "Solve linear system Ax=b",
            serde_json::json!({"a": "array<array<f64>>", "b": "array<f64>"}),
            "public",
        )),
        "linalg.eigenvalues" => Some((
            "Eigenvalue decomposition",
            serde_json::json!({"matrix": "array<array<f64>>"}),
            "public",
        )),
        "linalg.batched_tridiag_eigh" => Some((
            "Batched tridiagonal symmetric eigendecomposition (QL with Wilkinson shifts)",
            serde_json::json!({"diagonals": "array<f64>", "subdiagonals": "array<f64>", "n": "u64", "n_batches": "u64 (optional, default 1)"}),
            "public",
        )),
        "linalg.svd" => Some((
            "Singular value decomposition",
            serde_json::json!({"matrix": "array<array<f64>>"}),
            "public",
        )),
        "linalg.qr" => Some((
            "QR decomposition",
            serde_json::json!({"matrix": "array<array<f64>>"}),
            "public",
        )),
        "linalg.graph_laplacian" => Some((
            "Graph Laplacian from adjacency matrix",
            serde_json::json!({"adjacency": "array<array<f64>>"}),
            "public",
        )),
        _ => None,
    }
}

fn describe_domain(method: &str) -> Option<(&str, Value, &str)> {
    match method {
        "ode.step" => Some((
            "Single ODE integration step",
            serde_json::json!({"state": "array<f64>", "dt": "f64", "method": "string? (euler|rk4)"}),
            "public",
        )),
        "graph.belief_propagation" => Some((
            "Belief propagation on factor graph",
            serde_json::json!({"factors": "array", "messages": "array?", "iterations": "u32?"}),
            "public",
        )),
        "spectral.fft" => Some((
            "Fast Fourier transform",
            serde_json::json!({"data": "array<f64>"}),
            "public",
        )),
        "spectral.power_spectrum" => Some((
            "Power spectral density",
            serde_json::json!({"data": "array<f64>"}),
            "public",
        )),
        "spectral.stft" => Some((
            "Short-time Fourier transform",
            serde_json::json!({"data": "array<f64>", "window_size": "u32", "hop_size": "u32?"}),
            "public",
        )),
        "ml.mlp_forward" => Some((
            "Forward pass through trained MLP",
            serde_json::json!({"model": "object (SimpleMlp)", "input": "array<f64>"}),
            "public",
        )),
        "ml.mlp_train" => Some((
            "Train single-layer perceptron",
            serde_json::json!({"input_dim": "u32", "output_dim": "u32", "data": "array<array<f64>>", "labels": "array<array<f64>>", "epochs": "u32?", "lr": "f64?"}),
            "public",
        )),
        "ml.mlp_infer" => Some((
            "Batch inference on trained model",
            serde_json::json!({"model": "object (SimpleMlp)", "inputs": "array<array<f64>>"}),
            "public",
        )),
        "ml.mlp_save" => Some((
            "Save model to binary format",
            serde_json::json!({"model": "object (SimpleMlp)", "path": "string"}),
            "public",
        )),
        "ml.mlp_load" => Some((
            "Load model from file",
            serde_json::json!({"path": "string"}),
            "public",
        )),
        "ml.perceptron_train" => Some((
            "End-to-end perceptron training pipeline",
            serde_json::json!({"input_dim": "u32", "output_dim": "u32", "data": "array<array<f64>>", "labels": "array<array<f64>>", "epochs": "u32?", "lr": "f64?"}),
            "public",
        )),
        "ml.attention" => Some((
            "Scaled dot-product attention",
            serde_json::json!({"q": "array<array<f64>>", "k": "array<array<f64>>", "v": "array<array<f64>>"}),
            "public",
        )),
        "ml.esn_predict" => Some((
            "Echo State Network prediction",
            serde_json::json!({"input": "array<f64>", "reservoir_size": "u32?"}),
            "public",
        )),
        "nautilus.create" => Some((
            "Create nautilus session",
            serde_json::json!({"config": "object?"}),
            "public",
        )),
        "nautilus.observe" => Some((
            "Feed observation data",
            serde_json::json!({"session_id": "string", "data": "array<f64>"}),
            "public",
        )),
        "nautilus.train" => Some((
            "Train nautilus model",
            serde_json::json!({"session_id": "string"}),
            "public",
        )),
        "nautilus.predict" => Some((
            "Predict from nautilus model",
            serde_json::json!({"session_id": "string", "horizon": "u32?"}),
            "public",
        )),
        "nautilus.export" => Some((
            "Export trained model",
            serde_json::json!({"session_id": "string"}),
            "public",
        )),
        "nautilus.import" => Some((
            "Import model into session",
            serde_json::json!({"model": "object"}),
            "public",
        )),
        "noise.perlin2d" => Some((
            "2D Perlin noise",
            serde_json::json!({"x": "f64", "y": "f64", "seed": "u32?"}),
            "public",
        )),
        "noise.perlin3d" => Some((
            "3D Perlin noise",
            serde_json::json!({"x": "f64", "y": "f64", "z": "f64", "seed": "u32?"}),
            "public",
        )),
        "rng.uniform" => Some((
            "Uniform random numbers",
            serde_json::json!({"count": "u32", "min": "f64?", "max": "f64?"}),
            "public",
        )),
        _ => None,
    }
}

fn describe_tensor(method: &str) -> Option<(&str, Value, &str)> {
    match method {
        "tensor.create" => Some((
            "Create GPU tensor from data",
            serde_json::json!({"data": "array<f64>", "shape": "array<u32>"}),
            "protected",
        )),
        "tensor.matmul" => Some((
            "GPU matrix multiplication (tensor handles)",
            serde_json::json!({"a": "string (handle)", "b": "string (handle)"}),
            "protected",
        )),
        "tensor.matmul_inline" => Some((
            "CPU matrix multiplication (inline data)",
            serde_json::json!({"a": "array<array<f64>>", "b": "array<array<f64>>"}),
            "public",
        )),
        "tensor.add" => Some((
            "Element-wise tensor addition",
            serde_json::json!({"a": "string (handle)", "b": "string (handle)"}),
            "protected",
        )),
        "tensor.scale" => Some((
            "Scalar multiplication",
            serde_json::json!({"tensor": "string (handle)", "scalar": "f64"}),
            "protected",
        )),
        "tensor.clamp" => Some((
            "Clamp tensor values",
            serde_json::json!({"tensor": "string (handle)", "min": "f64", "max": "f64"}),
            "protected",
        )),
        "tensor.reduce" => Some((
            "Reduction operation (sum, mean, max, min)",
            serde_json::json!({"tensor": "string (handle)", "op": "string (sum|mean|max|min)"}),
            "protected",
        )),
        "tensor.sigmoid" => Some((
            "Sigmoid activation on tensor",
            serde_json::json!({"tensor": "string (handle)"}),
            "protected",
        )),
        "tensor.batch.submit" => Some((
            "Submit batch of tensor operations",
            serde_json::json!({"ops": "array<object>"}),
            "protected",
        )),
        "fhe.ntt" => Some((
            "Number-theoretic transform for FHE",
            serde_json::json!({"data": "array<u64>", "modulus": "u64"}),
            "protected",
        )),
        "fhe.pointwise_mul" => Some((
            "Pointwise polynomial multiplication",
            serde_json::json!({"a": "array<u64>", "b": "array<u64>", "modulus": "u64"}),
            "protected",
        )),
        _ => None,
    }
}

fn describe_auth(method: &str) -> Option<(&str, Value, &str)> {
    match method {
        "mesh.trust_verify" => Some((
            "Verify BTSP trust with peer",
            serde_json::json!({"peer_id": "string", "token": "string?"}),
            "public",
        )),
        "mesh.health" => Some((
            "Cross-gate mesh health probe",
            serde_json::json!({}),
            "public",
        )),
        "auth.check" => Some((
            "Check caller authentication status",
            serde_json::json!({}),
            "public",
        )),
        "auth.mode" => Some(("Report current auth mode", serde_json::json!({}), "public")),
        "auth.peer_info" => Some((
            "Report peer connection info",
            serde_json::json!({}),
            "public",
        )),
        "btsp.negotiate" => Some((
            "Initiate BTSP Phase 1 handshake",
            serde_json::json!({"peer_id": "string"}),
            "protected",
        )),
        "btsp.capabilities" => Some((
            "List supported BTSP capabilities",
            serde_json::json!({}),
            "public",
        )),
        "method.describe" => Some((
            "Describe a method's params, access level, and purpose",
            serde_json::json!({"method": "string (method name)"}),
            "public",
        )),
        _ => None,
    }
}

/// Structured descriptor for a method — params, description, access, examples.
fn method_descriptor(method: &str) -> Value {
    let (description, params_schema, access) = describe_health(method)
        .or_else(|| describe_identity(method))
        .or_else(|| describe_device(method))
        .or_else(|| describe_compute(method))
        .or_else(|| describe_stats(method))
        .or_else(|| describe_signal(method))
        .or_else(|| describe_linalg(method))
        .or_else(|| describe_domain(method))
        .or_else(|| describe_tensor(method))
        .or_else(|| describe_auth(method))
        .unwrap_or_else(|| ("Unknown method", serde_json::json!({}), "unknown"));

    serde_json::json!({
        "method": method,
        "description": description,
        "params": params_schema,
        "access": access,
        "primal": crate::PRIMAL_NAMESPACE,
        "version": env!("CARGO_PKG_VERSION"),
    })
}

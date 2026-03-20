// SPDX-License-Identifier: AGPL-3.0-or-later
//! Primal identity and capability advertisement handlers.

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

/// `barracuda.primal.capabilities` — Advertise capabilities for discovery.
///
/// Returns the set of capabilities this primal provides at runtime. Other
/// primals use this to discover what barraCuda can do (capability-based
/// routing) rather than relying on hardcoded primal names.
pub(super) fn capabilities(primal: &BarraCudaPrimal, id: Value) -> JsonRpcResponse {
    let has_gpu = primal.device().is_some();
    let has_f64 = primal.device().is_some_and(|d| d.has_f64_shaders());
    let has_spirv = primal.device().is_some_and(|d| d.has_spirv_passthrough());

    let version = env!("CARGO_PKG_VERSION");
    JsonRpcResponse::success(
        id,
        serde_json::json!({
            "provides": [
                { "id": "gpu.compute", "version": version },
                { "id": "tensor.ops", "version": version },
                { "id": "gpu.dispatch", "version": version },
            ],
            "requires": [
                { "id": "shader.compile", "version": ">=0.1.0", "optional": true },
            ],
            "domains": [
                "gpu_compute",
                "tensor_ops",
                "fhe",
                "molecular_dynamics",
                "lattice_qcd",
                "statistics",
                "hydrology",
                "bio",
            ],
            "methods": &*REGISTERED_METHODS,
            "hardware": {
                "gpu_available": has_gpu,
                "f64_shaders": has_f64,
                "spirv_passthrough": has_spirv,
            },
        }),
    )
}

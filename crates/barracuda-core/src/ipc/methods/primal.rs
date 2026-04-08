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
    let has_f64 = primal.device().is_some_and(|d| d.has_f64_shaders());
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
            "consumed_capabilities": ["shader.compile.cpu", "shader.validate", "compute.dispatch"],
            "provides": provides_list,
            "requires": [
                { "id": "shader.compile", "version": ">=0.1.0", "optional": true },
            ],
            "domains": crate::discovery::capabilities(),
            "hardware": {
                "gpu_available": has_gpu,
                "f64_shaders": has_f64,
                "spirv_passthrough": has_spirv,
            },
            "protocol": "jsonrpc-2.0",
            "transport": ["uds", "tcp"],
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
            "domain": "compute",
            "license": "AGPL-3.0-or-later",
        }),
    )
}

// SPDX-License-Identifier: AGPL-3.0-or-later
//! Mesh trust and cross-gate validation handlers.
//!
//! These methods support cross-gate security validation. A remote peer
//! (e.g. eastGate) connects over TCP, completes BTSP handshake, then calls
//! `mesh.trust_verify` to confirm the session is authenticated end-to-end.

use super::super::jsonrpc::JsonRpcResponse;
use serde_json::Value;

/// `mesh.trust_verify` — Confirm BTSP trust relationship with this peer.
///
/// Called by remote gates after BTSP handshake completes. Returns trust
/// metadata: whether the session is authenticated, cipher in use, and
/// the primal's identity. Enables cross-gate trust validation without
/// requiring a separate trust service.
///
/// Wire contract:
/// ```json
/// {"method": "mesh.trust_verify", "params": {"nonce": "optional-challenge-nonce"}}
/// ```
///
/// Response:
/// ```json
/// {"trusted": true, "primal": "barraCuda", "gate": "strandGate", "cipher": "chacha20-poly1305", ...}
/// ```
pub(super) fn mesh_trust_verify(params: &Value, id: Value) -> JsonRpcResponse {
    let caller_nonce = params
        .get("nonce")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    let bearer = params
        .get("_auth")
        .and_then(|a| a.get("bearer"))
        .and_then(|b| b.as_str());

    let authenticated = bearer.is_some();

    let mut response = serde_json::json!({
        "trusted": authenticated,
        "primal": crate::PRIMAL_NAME,
        "version": env!("CARGO_PKG_VERSION"),
        "gate": "strandGate",
        "capabilities": ["math", "compute", "ml", "tensor", "stats"],
        "btsp_phase3": true,
        "cipher_suites": ["chacha20-poly1305", "hmac_plain", "null"],
    });

    if !caller_nonce.is_empty() {
        let echo = format!("ack:{caller_nonce}");
        response["nonce_echo"] = Value::String(echo);
    }

    if !authenticated {
        response["reason"] = Value::String(
            "No bearer token in request — BTSP handshake may not have completed".into(),
        );
    }

    JsonRpcResponse::success(id, response)
}

/// `mesh.health` — Cross-gate mesh health probe.
///
/// Returns the status of mesh-critical services (security provider, discovery)
/// and peer connectivity metadata. Used by partner gates to confirm this node
/// is operational in the mesh.
pub(super) fn mesh_health(id: Value) -> JsonRpcResponse {
    let socket_dir = crate::ipc::transport_config::resolve_socket_dir();
    let security_live = has_socket_in(&socket_dir, "beardog");
    let discovery_live = has_socket_in(&socket_dir, "songbird");

    let status = if security_live && discovery_live {
        "healthy"
    } else if security_live || discovery_live {
        "degraded"
    } else {
        "offline"
    };

    JsonRpcResponse::success(
        id,
        serde_json::json!({
            "status": status,
            "gate": "strandGate",
            "primal": crate::PRIMAL_NAME,
            "services": {
                "security_provider": security_live,
                "discovery": discovery_live,
            },
            "federation_port": 7700,
        }),
    )
}

fn has_socket_in(dir: &std::path::Path, prefix: &str) -> bool {
    dir.read_dir()
        .map(|entries| {
            entries.filter_map(Result::ok).any(|e| {
                let name = e.file_name();
                let path = std::path::Path::new(&name);
                let matches_prefix = path
                    .to_str()
                    .is_some_and(|n| n.starts_with(prefix));
                let is_sock = path
                    .extension()
                    .is_some_and(|ext| ext.eq_ignore_ascii_case("sock"));
                matches_prefix && is_sock
            })
        })
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn trust_verify_no_auth() {
        let resp = mesh_trust_verify(&json!({}), json!(1));
        let result = resp.result.unwrap();
        assert_eq!(result["trusted"], false);
        assert_eq!(result["primal"], "barraCuda");
        assert_eq!(result["gate"], "strandGate");
        assert!(result["reason"].as_str().unwrap().contains("bearer"));
    }

    #[test]
    fn trust_verify_with_auth() {
        let resp = mesh_trust_verify(
            &json!({"_auth": {"bearer": "test-token-abc"}}),
            json!(2),
        );
        let result = resp.result.unwrap();
        assert_eq!(result["trusted"], true);
        assert!(result.get("reason").is_none());
    }

    #[test]
    fn trust_verify_nonce_echo() {
        let resp = mesh_trust_verify(
            &json!({"nonce": "eastgate-challenge-42", "_auth": {"bearer": "tok"}}),
            json!(3),
        );
        let result = resp.result.unwrap();
        assert_eq!(result["nonce_echo"], "ack:eastgate-challenge-42");
    }

    #[test]
    fn mesh_health_returns_status() {
        let resp = mesh_health(json!(4));
        let result = resp.result.unwrap();
        let status = result["status"].as_str().unwrap();
        assert!(
            ["healthy", "degraded", "offline"].contains(&status),
            "unexpected status: {status}"
        );
        assert_eq!(result["gate"], "strandGate");
        assert_eq!(result["federation_port"], 7700);
        assert!(result["services"].is_object());
    }
}

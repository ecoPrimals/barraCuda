// SPDX-License-Identifier: AGPL-3.0-or-later
//! Outbound `primal.announce` push to biomeOS Neural API.
//!
//! After socket bind, barraCuda pushes its identity, capabilities, signal tier,
//! cost hints, and latency estimates to the Neural API so biomeOS can compute
//! routing weights for `capability.call` dispatch.
//!
//! Socket discovery follows WAVE42 tiered lookup:
//! 1. `$NEURAL_API_SOCKET` — explicit override
//! 2. `$XDG_RUNTIME_DIR/biomeos/neural-api-{family}.sock`
//! 3. `/tmp/biomeos/neural-api-{family}.sock`
//!
//! Failure is non-fatal: barraCuda operates standalone when biomeOS is absent.

use crate::ipc::methods::REGISTERED_METHODS;
use std::path::PathBuf;

const DEFAULT_FAMILY: &str = "ecoPrimal";

/// Resolve the biomeOS Neural API socket path via tiered lookup.
///
/// Testable variant accepting an environment reader function.
fn resolve_neural_api_socket_with(reader: &dyn Fn(&str) -> Option<String>) -> Option<PathBuf> {
    if let Some(explicit) = reader("NEURAL_API_SOCKET") {
        let path = PathBuf::from(&explicit);
        if path.exists() {
            return Some(path);
        }
        tracing::debug!("NEURAL_API_SOCKET={explicit} does not exist");
    }

    let family = reader("ECOPRIMALS_FAMILY_ID")
        .or_else(|| reader("BIOMEOS_FAMILY_ID"))
        .or_else(|| reader("FAMILY_ID"))
        .unwrap_or_else(|| DEFAULT_FAMILY.to_string());

    let socket_name = format!("neural-api-{family}.sock");

    if let Some(xdg) = reader("XDG_RUNTIME_DIR") {
        let path = PathBuf::from(xdg).join("biomeos").join(&socket_name);
        if path.exists() {
            return Some(path);
        }
    }

    let fallback = PathBuf::from("/tmp/biomeos").join(&socket_name);
    if fallback.exists() {
        return Some(fallback);
    }

    None
}

/// Resolve the Neural API socket from real environment variables.
#[must_use]
pub fn resolve_neural_api_socket() -> Option<PathBuf> {
    resolve_neural_api_socket_with(&|key| std::env::var(key).ok())
}

/// Build the `primal.announce` JSON-RPC request payload.
///
/// Schema aligned with biomeOS v3.68+ Neural API (Wave 43/44).
fn build_announce_payload(own_socket: &str, version: &str) -> serde_json::Value {
    let methods: Vec<&str> = REGISTERED_METHODS.to_vec();

    serde_json::json!({
        "jsonrpc": "2.0",
        "method": "primal.announce",
        "params": {
            "primal": crate::PRIMAL_NAME,
            "namespace": crate::PRIMAL_NAMESPACE,
            "version": version,
            "domain": crate::PRIMAL_DOMAIN,
            "pid": std::process::id(),
            "socket": own_socket,
            "capabilities": ["math", "shader", "compute"],
            "methods": methods,
            "signal_tiers": ["node"],
            "cost_hints": {
                "math": 20.0,
                "shader": 50.0,
                "compute": 80.0,
            },
            "latency_estimates": {
                "math": 10,
                "shader": 100,
                "compute": 200,
            },
            "attestation": null,
        },
        "id": 1
    })
}

/// Send a JSON-RPC request over a Unix domain socket (newline-delimited).
///
/// Returns the parsed response or an IO error. Applies a 5-second read timeout.
#[cfg(unix)]
async fn send_jsonrpc_uds(
    socket_path: &std::path::Path,
    request: &serde_json::Value,
) -> Result<serde_json::Value, std::io::Error> {
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt};
    use tokio::net::UnixStream;

    let stream = UnixStream::connect(socket_path).await?;
    let (reader, mut writer) = stream.into_split();

    let mut line = serde_json::to_string(request)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    line.push('\n');

    writer.write_all(line.as_bytes()).await?;
    writer.shutdown().await?;

    let response_line = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        tokio::io::BufReader::new(reader).lines().next_line(),
    )
    .await
    .map_err(|_| std::io::Error::new(std::io::ErrorKind::TimedOut, "neural-api read timeout"))?
    .and_then(|opt| opt.ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "no response")
    }))?;

    serde_json::from_str(&response_line)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
}

/// Push `primal.announce` to the biomeOS Neural API.
///
/// Fire-and-forget: logs success or warns on failure. Non-fatal — barraCuda
/// operates standalone when biomeOS is absent.
#[cfg(unix)]
pub async fn announce_to_neural_api(own_socket: &str, version: &str) {
    let Some(neural_socket) = resolve_neural_api_socket() else {
        tracing::info!(mode = "standalone", "Neural API socket not found — skipping primal.announce");
        return;
    };

    tracing::info!(
        neural_socket = %neural_socket.display(),
        "pushing primal.announce to Neural API"
    );

    let payload = build_announce_payload(own_socket, version);

    match send_jsonrpc_uds(&neural_socket, &payload).await {
        Ok(response) => {
            if response.get("error").is_some() {
                tracing::warn!(
                    response = %response,
                    "Neural API returned error for primal.announce"
                );
            } else {
                tracing::info!("primal.announce accepted by Neural API");
            }
        }
        Err(e) => {
            tracing::debug!(
                error = %e,
                "Neural API announce failed (non-fatal, standalone mode)"
            );
        }
    }
}

#[cfg(not(unix))]
pub async fn announce_to_neural_api(_own_socket: &str, _version: &str) {
    tracing::debug!("primal.announce: Unix sockets not supported on this platform");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolve_neural_api_socket_explicit() {
        let reader = |key: &str| -> Option<String> {
            match key {
                "NEURAL_API_SOCKET" => Some("/nonexistent/path.sock".into()),
                _ => None,
            }
        };
        // Explicit path doesn't exist, should return None
        assert!(resolve_neural_api_socket_with(&reader).is_none());
    }

    #[test]
    fn test_resolve_neural_api_socket_xdg_fallback() {
        let reader = |key: &str| -> Option<String> {
            match key {
                "XDG_RUNTIME_DIR" => Some("/tmp/test-xdg".into()),
                _ => None,
            }
        };
        // Neither XDG path nor /tmp fallback exist — returns None
        assert!(resolve_neural_api_socket_with(&reader).is_none());
    }

    #[test]
    fn test_resolve_neural_api_socket_family_override() {
        let reader = |key: &str| -> Option<String> {
            match key {
                "ECOPRIMALS_FAMILY_ID" => Some("myFamily".into()),
                "XDG_RUNTIME_DIR" => Some("/tmp/test-xdg".into()),
                _ => None,
            }
        };
        // Socket doesn't exist, but verify resolution doesn't panic
        assert!(resolve_neural_api_socket_with(&reader).is_none());
    }

    #[test]
    fn test_build_announce_payload_structure() {
        let payload = build_announce_payload("/run/user/1000/biomeos/math.sock", "0.4.0");

        assert_eq!(payload["jsonrpc"], "2.0");
        assert_eq!(payload["method"], "primal.announce");

        let params = &payload["params"];
        assert_eq!(params["primal"], "barraCuda");
        assert_eq!(params["namespace"], "barracuda");
        assert_eq!(params["domain"], "math");
        assert_eq!(params["version"], "0.4.0");
        assert_eq!(params["socket"], "/run/user/1000/biomeos/math.sock");
        assert!(params["pid"].is_number());

        let caps = params["capabilities"].as_array().unwrap();
        assert_eq!(caps.len(), 3);

        let tiers = params["signal_tiers"].as_array().unwrap();
        assert_eq!(tiers[0], "node");

        assert_eq!(params["cost_hints"]["math"], 20.0);
        assert_eq!(params["cost_hints"]["shader"], 50.0);
        assert_eq!(params["cost_hints"]["compute"], 80.0);

        assert_eq!(params["latency_estimates"]["math"], 10);
        assert_eq!(params["latency_estimates"]["shader"], 100);
        assert_eq!(params["latency_estimates"]["compute"], 200);

        let methods = params["methods"].as_array().unwrap();
        assert!(methods.len() >= 87);
    }
}

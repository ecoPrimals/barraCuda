// SPDX-License-Identifier: AGPL-3.0-or-later
//! BTSP security-provider discovery helpers.
//!
//! Resolution chain (per `NUCLEUS_TWO_TIER_CRYPTO_MODEL.md`):
//! 1. `$BEARDOG_SOCKET` or `$BTSP_PROVIDER_SOCKET` env var (composition-injected)
//! 2. `$BIOMEOS_SOCKET_DIR/{SECURITY_DOMAIN}-{family_id}.sock`
//! 3. `$BIOMEOS_SOCKET_DIR/{SECURITY_DOMAIN}.sock`
//! 4. Discovery files in `$BIOMEOS_SOCKET_DIR/*.json` advertising
//!    `btsp.session.create` capability
//! 5. Songbird `DISCOVERY_SOCKET` `ipc.resolve` fallback

use super::btsp::{read_ndjson_line, write_ndjson_line};
use super::transport::{resolve_family_id, resolve_socket_dir};

const SECURITY_DOMAIN: &str = "crypto";

/// Discover the security-domain socket for BTSP handshake delegation.
pub(super) fn discover_security_provider() -> Option<std::path::PathBuf> {
    for var in ["BEARDOG_SOCKET", "BTSP_PROVIDER_SOCKET"] {
        if let Ok(path) = std::env::var(var) {
            let p = std::path::PathBuf::from(&path);
            if p.exists() {
                return Some(p);
            }
            tracing::debug!("{var}={path} set but socket does not exist, falling through");
        }
    }

    let sock_dir = resolve_socket_dir();

    if let Some(fid) = resolve_family_id() {
        let scoped = sock_dir.join(format!("{SECURITY_DOMAIN}-{fid}.sock"));
        if scoped.exists() {
            return Some(scoped);
        }
    }
    let unscoped = sock_dir.join(format!("{SECURITY_DOMAIN}.sock"));
    if unscoped.exists() {
        return Some(unscoped);
    }

    discover_by_capability(&sock_dir, "btsp.session.create")
}

/// Resolve a capability via Songbird's `DISCOVERY_SOCKET` using `ipc.resolve`.
pub(super) async fn resolve_via_discovery_socket(capability: &str) -> Option<std::path::PathBuf> {
    let discovery_path = std::env::var("DISCOVERY_SOCKET").ok()?;
    let discovery_path = std::path::Path::new(&discovery_path);
    if !discovery_path.exists() {
        tracing::debug!(
            "DISCOVERY_SOCKET={} set but socket does not exist",
            discovery_path.display()
        );
        return None;
    }

    let stream = tokio::net::UnixStream::connect(discovery_path).await.ok()?;
    let mut reader = tokio::io::BufReader::new(stream);

    let request = serde_json::json!({
        "jsonrpc": "2.0",
        "method": "ipc.resolve",
        "params": { "capability": capability },
        "id": 1
    });
    write_ndjson_line(reader.get_mut(), &request).await.ok()?;
    let response_line = read_ndjson_line(&mut reader).await.ok()?;
    let response: serde_json::Value = serde_json::from_str(&response_line).ok()?;

    let result = response.get("result")?;
    let unix_addr = result
        .get("unix")
        .or_else(|| result.get("socket"))
        .and_then(|v| v.as_str())?;
    let sock = std::path::PathBuf::from(unix_addr.trim_start_matches("unix://"));
    sock.exists().then_some(sock)
}

fn discover_by_capability(sock_dir: &std::path::Path, method: &str) -> Option<std::path::PathBuf> {
    let entries = std::fs::read_dir(sock_dir).ok()?;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().is_some_and(|ext| ext == "json") {
            if let Some(sock) = check_discovery_file_for_method(&path, method) {
                return Some(sock);
            }
        }
    }
    None
}

fn check_discovery_file_for_method(
    path: &std::path::Path,
    method: &str,
) -> Option<std::path::PathBuf> {
    let content = std::fs::read_to_string(path).ok()?;
    let info: serde_json::Value = serde_json::from_str(&content).ok()?;
    let methods = info.get("methods")?.as_array()?;
    let has_method = methods
        .iter()
        .any(|m| m.as_str().is_some_and(|s| s == method));
    if !has_method {
        return None;
    }
    let unix_addr = info
        .get("transports")
        .and_then(|t| t.get("unix"))
        .and_then(|v| v.as_str())
        .and_then(|s| s.strip_prefix("unix://"))?;
    let sock = std::path::PathBuf::from(unix_addr);
    sock.exists().then_some(sock)
}

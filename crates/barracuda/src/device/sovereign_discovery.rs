// SPDX-License-Identifier: AGPL-3.0-or-later
//! Capability-based discovery for sovereign compute dispatch endpoints.
//!
//! Scans ecosystem manifests at runtime to find primals advertising
//! `compute.dispatch`. No hardcoded primal names — discovery is entirely
//! capability-driven per `CAPABILITY_BASED_DISCOVERY_STANDARD.md`.

use super::coral_compiler::DEFAULT_ECOPRIMALS_DISCOVERY_DIR;

/// Capability string for compute dispatch discovery.
pub(super) const DISPATCH_CAPABILITY: &str = "compute.dispatch";

/// Environment variable for explicit compute dispatch endpoint override.
pub(super) const DISPATCH_ADDR_ENV: &str = "BARRACUDA_DISPATCH_ADDR";

/// Canonical discovery subdirectory name.
const DISCOVERY_SUBDIR: &str = "discovery";

/// Detect whether a compute dispatch endpoint is available at runtime.
///
/// Discovery chain (first match wins):
/// 1. `BARRACUDA_DISPATCH_ADDR` env var
/// 2. Capability scan of `$XDG_RUNTIME_DIR/ecoPrimals/*.json` for
///    `"compute.dispatch"` in the `provides` or `capabilities` array
#[cfg(feature = "sovereign-dispatch")]
pub(super) fn detect_dispatch_addr() -> Option<String> {
    if let Ok(addr) = std::env::var(DISPATCH_ADDR_ENV) {
        let addr = addr.trim().to_owned();
        if !addr.is_empty() {
            return Some(addr);
        }
    }

    let runtime_dir = std::env::var("XDG_RUNTIME_DIR").ok()?;
    let eco_dir = std::env::var("ECOPRIMALS_DISCOVERY_DIR")
        .unwrap_or_else(|_| DEFAULT_ECOPRIMALS_DISCOVERY_DIR.to_owned());
    let base_dir = std::path::PathBuf::from(runtime_dir).join(eco_dir);

    for dir in [&base_dir, &base_dir.join(DISCOVERY_SUBDIR)] {
        if let Some(addr) = scan_dispatch_capability(dir) {
            return Some(addr);
        }
    }
    None
}

/// Scan a directory for any primal advertising `compute.dispatch`.
#[cfg(feature = "sovereign-dispatch")]
fn scan_dispatch_capability(dir: &std::path::Path) -> Option<String> {
    let entries = std::fs::read_dir(dir).ok()?;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().is_some_and(|e| e == "json") {
            if let Some(addr) = read_dispatch_transport(&path) {
                return Some(addr);
            }
        }
    }
    None
}

/// Query the dispatch primal's GPU architecture via `compute.dispatch.capabilities`.
///
/// Returns `Some(arch)` (e.g. `"sm_89"`) when the dispatch primal is
/// reachable and reports a target arch, `None` otherwise.
#[cfg(feature = "sovereign-dispatch")]
pub(super) fn query_dispatch_arch(addr: &str) -> Option<String> {
    let handle = tokio::runtime::Handle::try_current().ok()?;
    let addr = addr.to_owned();
    tokio::task::block_in_place(|| {
        handle.block_on(async {
            let host_port = addr.trim_start_matches("http://");
            let body = serde_json::json!({
                "jsonrpc": "2.0",
                "method": "compute.dispatch.capabilities",
                "params": [],
                "id": 1,
            });
            let body_str = serde_json::to_string(&body).ok()?;

            let http_request = format!(
                "POST / HTTP/1.1\r\nHost: {host_port}\r\n\
                 Content-Type: application/json\r\n\
                 Content-Length: {}\r\n\
                 Connection: close\r\n\r\n{body_str}",
                body_str.len()
            );

            let timeout = tokio::time::Duration::from_secs(5);
            let mut stream =
                tokio::time::timeout(timeout, tokio::net::TcpStream::connect(host_port))
                    .await
                    .ok()?
                    .ok()?;

            tokio::io::AsyncWriteExt::write_all(&mut stream, http_request.as_bytes())
                .await
                .ok()?;
            let mut buf = Vec::new();
            tokio::io::AsyncReadExt::read_to_end(&mut stream, &mut buf)
                .await
                .ok()?;

            let response_str = String::from_utf8_lossy(&buf);
            let json_start = response_str.find('{')?;
            let rpc: serde_json::Value = serde_json::from_str(&response_str[json_start..]).ok()?;

            let result = rpc.get("result")?;
            let arch = result.get("arch").and_then(serde_json::Value::as_str);
            arch.map(str::to_owned)
        })
    })
}

/// Read a primal manifest and extract transport if it provides dispatch.
#[cfg(feature = "sovereign-dispatch")]
fn read_dispatch_transport(path: &std::path::Path) -> Option<String> {
    let content = std::fs::read_to_string(path).ok()?;
    let info: serde_json::Value = serde_json::from_str(&content).ok()?;

    let has_dispatch = info
        .get("provides")
        .or_else(|| info.get("capabilities"))
        .and_then(|v| v.as_array())
        .is_some_and(|caps| {
            caps.iter().any(|c| {
                c.as_str()
                    .is_some_and(|s| s.starts_with(DISPATCH_CAPABILITY))
            })
        });

    if !has_dispatch {
        return None;
    }

    let jsonrpc = info.get("transports")?.get("jsonrpc")?;
    if let Some(s) = jsonrpc.as_str() {
        return Some(s.to_owned());
    }
    jsonrpc
        .get("tcp")
        .and_then(|v| v.as_str())
        .map(str::to_owned)
}

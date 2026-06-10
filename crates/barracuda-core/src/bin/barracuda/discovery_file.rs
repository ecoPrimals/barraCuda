// SPDX-License-Identifier: AGPL-3.0-or-later
//! Discovery file management, PID file, and systemd notification helpers.
//!
//! These lifecycle helpers run only in the binary context (server/service mode).
//! Extracted from `main.rs` to keep the binary under 800 lines.

use barracuda_core::env_keys;

/// Write a discovery file so peer primals can find barraCuda.
///
/// Capabilities, provides, and methods are derived from the actual registered
/// IPC methods — no hardcoded values. Per wateringHole capability-based discovery.
///
/// Supports both TCP (`host:port`) and Unix socket (`unix:///path`) transports.
/// The discovery file is co-located with the socket (derived from `unix_path`'s
/// parent) to avoid `/tmp` pollution. Falls back to `resolve_socket_dir()`.
pub fn write_discovery_file(
    tcp_addr: Option<&str>,
    tarpc_addr: Option<&str>,
    unix_path: Option<&std::path::Path>,
) {
    let dir = unix_path
        .and_then(|p| p.parent())
        .map(std::path::Path::to_path_buf)
        .or_else(discovery_dir);
    let Some(dir) = dir else { return };
    if std::fs::create_dir_all(&dir).is_err() {
        return;
    }
    let filename = format!("{}-core.json", barracuda_core::PRIMAL_NAMESPACE);
    let path = dir.join(&filename);

    let mut transports = serde_json::Map::new();
    if let Some(addr) = tcp_addr {
        transports.insert(
            "jsonrpc".into(),
            serde_json::Value::String(addr.to_string()),
        );
    }
    if let Some(tarpc) = tarpc_addr {
        transports.insert("tarpc".into(), serde_json::Value::String(tarpc.to_string()));
    }
    if let Some(sock) = unix_path {
        transports.insert(
            "unix".into(),
            serde_json::Value::String(format!("unix://{}", sock.display())),
        );
        if !transports.contains_key("jsonrpc") {
            transports.insert(
                "jsonrpc".into(),
                serde_json::Value::String(format!("unix://{}", sock.display())),
            );
        }
    }

    let capabilities: Vec<serde_json::Value> = barracuda_core::discovery::capabilities()
        .into_iter()
        .map(serde_json::Value::String)
        .collect();
    let provides: Vec<serde_json::Value> = barracuda_core::discovery::provides()
        .into_iter()
        .map(serde_json::Value::String)
        .collect();
    let methods: Vec<serde_json::Value> = barracuda_core::discovery::registered_methods()
        .iter()
        .map(|m| serde_json::Value::String(m.clone()))
        .collect();

    let discovery = serde_json::json!({
        "primal": barracuda_core::PRIMAL_NAME,
        "pid": std::process::id(),
        "transports": transports,
        "capabilities": capabilities,
        "provides": provides,
        "methods": methods,
        "requires": [{ "id": "shader.compile", "optional": true }],
    });

    match serde_json::to_string_pretty(&discovery) {
        Ok(json) => match std::fs::write(&path, json) {
            Ok(()) => tracing::info!(path = %path.display(), "wrote discovery file"),
            Err(e) => tracing::warn!(error = %e, "failed to write discovery file"),
        },
        Err(e) => tracing::warn!(error = %e, "failed to serialize discovery file"),
    }
}

/// Remove the discovery file on shutdown.
///
/// Derives location from the socket path (consistent with [`write_discovery_file`]).
pub fn remove_discovery_file(unix_path: Option<&std::path::Path>) {
    let dir = unix_path
        .and_then(|p| p.parent())
        .map(std::path::Path::to_path_buf)
        .or_else(discovery_dir);
    if let Some(dir) = dir {
        let filename = format!("{}-core.json", barracuda_core::PRIMAL_NAMESPACE);
        let path = dir.join(&filename);
        if path.exists() {
            let _ = std::fs::remove_file(&path);
        }
    }
}

/// The shared discovery directory for all ecoPrimals.
pub fn discovery_dir() -> Option<std::path::PathBuf> {
    let dir = barracuda_core::ipc::transport::resolve_socket_dir();
    if dir.as_os_str().is_empty() {
        None
    } else {
        Some(dir)
    }
}

/// Write PID file at `$XDG_RUNTIME_DIR/{namespace}/{namespace}.pid` (best-effort).
/// Returns a guard that removes the file on drop.
pub fn write_pid_file() -> Option<PidFileGuard> {
    let ns = barracuda_core::PRIMAL_NAMESPACE;
    let dir = std::env::var(env_keys::XDG_RUNTIME_DIR).ok()?;
    let dir = std::path::PathBuf::from(dir).join(ns);
    std::fs::create_dir_all(&dir).ok()?;
    let path = dir.join(format!("{ns}.pid"));
    std::fs::write(&path, std::process::id().to_string()).ok()?;
    tracing::debug!(path = %path.display(), "wrote PID file");
    Some(PidFileGuard { path })
}

/// RAII guard that removes the PID file on drop.
pub struct PidFileGuard {
    path: std::path::PathBuf,
}

impl Drop for PidFileGuard {
    fn drop(&mut self) {
        if self.path.exists() {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

/// Send READY=1 to `NOTIFY_SOCKET` for systemd Type=notify (best-effort).
#[cfg(unix)]
pub fn notify_systemd_ready() {
    let Ok(socket_path) = std::env::var(env_keys::NOTIFY_SOCKET) else {
        return;
    };
    let sock = match std::os::unix::net::UnixDatagram::unbound() {
        Ok(s) => s,
        Err(e) => {
            tracing::warn!(error = %e, "failed to create notify socket");
            return;
        }
    };
    let msg = b"READY=1\n";
    let addr: std::path::PathBuf = if socket_path.starts_with('@') {
        use std::ffi::OsStr;
        use std::os::unix::ffi::OsStrExt;
        let bytes: Vec<u8> = std::iter::once(0)
            .chain(socket_path.trim_start_matches('@').bytes())
            .collect();
        std::path::PathBuf::from(OsStr::from_bytes(&bytes))
    } else {
        std::path::PathBuf::from(&socket_path)
    };
    if let Err(e) = sock.send_to(msg, &addr) {
        tracing::warn!(error = %e, "failed to send sd_notify READY");
    } else {
        tracing::debug!("sent sd_notify READY");
    }
}

/// Spawn the Neural API announce task with the standard delay.
///
/// Consolidates the announce pattern used by both `server` and `service` modes.
pub fn spawn_neural_announce(addr: String, delay_ms: u64) {
    tokio::spawn(async move {
        tokio::time::sleep(std::time::Duration::from_millis(delay_ms)).await;
        barracuda_core::ipc::neural_announce::announce_to_neural_api(
            &addr,
            env!("CARGO_PKG_VERSION"),
        )
        .await;
    });
}

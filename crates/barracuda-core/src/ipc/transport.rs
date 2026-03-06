// SPDX-License-Identifier: AGPL-3.0-or-later
//! Transport layer for barraCuda IPC.
//!
//! Supports Unix domain sockets (primary) and TCP (fallback), per
//! wateringHole `UNIVERSAL_IPC_STANDARD_V3.md`.

use super::jsonrpc::{JsonRpcRequest, JsonRpcResponse};
use super::methods;
use crate::BarraCudaPrimal;
use barracuda::error::Result;
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::TcpListener;

/// Maximum tarpc frame length.
///
/// Configurable via `BARRACUDA_IPC_MAX_FRAME_BYTES`.  Defaults to 256 MiB —
/// large enough for any tensor payload, bounded enough for DoS protection.
/// The previous default of `usize::MAX` offered no transport-layer protection.
fn tarpc_max_frame_length() -> usize {
    const DEFAULT: usize = 256 * 1024 * 1024; // 256 MiB
    std::env::var("BARRACUDA_IPC_MAX_FRAME_BYTES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT)
}

/// Maximum concurrent tarpc connections.
///
/// Configurable via `BARRACUDA_IPC_MAX_CONNECTIONS`.  Defaults to 10.
fn tarpc_max_concurrent_connections() -> usize {
    const DEFAULT: usize = 10;
    std::env::var("BARRACUDA_IPC_MAX_CONNECTIONS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT)
}

/// Default TCP bind host when no environment or CLI override is provided.
///
/// `127.0.0.1` = localhost-only. This is the secure default: the primal
/// listens only on the loopback interface. External access requires explicit
/// configuration via `BARRACUDA_IPC_HOST` or `--bind`.
const DEFAULT_BIND_HOST: &str = "127.0.0.1";

/// Resolve the TCP bind address from the primal's own configuration.
///
/// Resolution chain (first match wins):
/// 1. `explicit` — CLI `--bind` argument
/// 2. `BARRACUDA_IPC_BIND` — full `host:port` from environment
/// 3. `BARRACUDA_IPC_HOST` + `BARRACUDA_IPC_PORT` — composed from environment
/// 4. `{DEFAULT_BIND_HOST}:0` — localhost, ephemeral port
///
/// The primal only has self-knowledge. It does not embed assumptions about
/// network topology or other primals — it simply resolves its own bind
/// address from its own configuration sources.
pub fn resolve_bind_address(explicit: Option<&str>) -> String {
    if let Some(addr) = explicit {
        return addr.to_string();
    }
    if let Ok(addr) = std::env::var("BARRACUDA_IPC_BIND") {
        return addr;
    }
    let host =
        std::env::var("BARRACUDA_IPC_HOST").unwrap_or_else(|_| DEFAULT_BIND_HOST.to_string());
    match std::env::var("BARRACUDA_IPC_PORT") {
        Ok(port) => format!("{host}:{port}"),
        Err(_) => format!("{host}:0"),
    }
}

/// IPC server for barraCuda primal.
///
/// Serves both JSON-RPC 2.0 (text, newline-delimited) and tarpc (binary,
/// serde-transport). JSON-RPC is the primary protocol for external consumers;
/// tarpc is the high-throughput protocol for primal-to-primal calls.
pub struct IpcServer {
    primal: Arc<BarraCudaPrimal>,
}

impl IpcServer {
    /// Create a new IPC server wrapping the primal.
    pub fn new(primal: Arc<BarraCudaPrimal>) -> Self {
        Self { primal }
    }

    /// Serve the tarpc binary RPC endpoint on the given TCP address.
    ///
    /// This runs alongside `serve_tcp` on a different port. The tarpc
    /// transport uses `serde-transport` with JSON framing for maximum
    /// throughput between Rust primals.
    pub async fn serve_tarpc(&self, addr: &str) -> Result<()> {
        use crate::rpc::{BarraCudaServer, BarraCudaService};
        use futures::prelude::*;
        use tarpc::server::{self, Channel};

        let mut listener =
            tarpc::serde_transport::tcp::listen(addr, tarpc::tokio_serde::formats::Json::default)
                .await
                .map_err(|e: std::io::Error| {
                    barracuda::error::BarracudaError::Internal(e.to_string())
                })?;
        let local_addr = listener.local_addr();
        tracing::info!("barraCuda tarpc listening on tcp://{local_addr}");

        listener
            .config_mut()
            .max_frame_length(tarpc_max_frame_length());

        let server = BarraCudaServer::new(Arc::clone(&self.primal));

        listener
            .filter_map(|r| future::ready(r.ok()))
            .map(server::BaseChannel::with_defaults)
            .map(move |channel| {
                let srv = server.clone();
                channel.execute(srv.serve()).for_each(|response| {
                    tokio::spawn(response);
                    async {}
                })
            })
            .buffer_unordered(tarpc_max_concurrent_connections())
            .for_each(|()| async {})
            .await;

        Ok(())
    }

    /// Start listening on TCP (JSON-RPC 2.0) with graceful shutdown on
    /// SIGINT/SIGTERM per wateringHole `UNIBIN_ARCHITECTURE_STANDARD.md`.
    pub async fn serve_tcp(&self, addr: &str) -> Result<()> {
        let listener = TcpListener::bind(addr).await?;
        let local_addr = listener.local_addr()?;
        tracing::info!("barraCuda IPC listening on tcp://{local_addr}");

        loop {
            tokio::select! {
                result = listener.accept() => {
                    let (stream, peer) = result?;
                    tracing::debug!("IPC connection from {peer}");
                    let primal = Arc::clone(&self.primal);
                    tokio::spawn(async move {
                        handle_stream(primal, stream).await;
                    });
                }
                () = shutdown_signal() => {
                    tracing::info!("Shutdown signal received, stopping TCP server");
                    break;
                }
            }
        }
        Ok(())
    }

    /// Start listening on a Unix domain socket with graceful shutdown.
    #[cfg(unix)]
    pub async fn serve_unix(&self, path: &std::path::Path) -> Result<()> {
        if path.exists() {
            std::fs::remove_file(path)?;
        }
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let listener = tokio::net::UnixListener::bind(path)?;
        tracing::info!("barraCuda IPC listening on unix://{}", path.display());

        loop {
            tokio::select! {
                result = listener.accept() => {
                    let (stream, _) = result?;
                    let primal = Arc::clone(&self.primal);
                    tokio::spawn(async move {
                        let (reader, mut writer) = stream.into_split();
                        let mut lines = BufReader::new(reader).lines();
                        while let Ok(Some(line)) = lines.next_line().await {
                            let Some(response) = handle_line(&primal, &line).await else {
                                continue;
                            };
                            let mut json = serde_json::to_string(&response).unwrap_or_else(|_| {
                                SERIALIZATION_ERROR.to_string()
                            });
                            json.push('\n');
                            if writer.write_all(json.as_bytes()).await.is_err() {
                                break;
                            }
                        }
                    });
                }
                () = shutdown_signal() => {
                    tracing::info!("Shutdown signal received, stopping Unix server");
                    break;
                }
            }
        }

        if path.exists() {
            std::fs::remove_file(path).ok();
        }
        Ok(())
    }

    /// Resolve the default IPC socket path per wateringHole standards.
    ///
    /// Prefers `$XDG_RUNTIME_DIR/barracuda/barracuda.sock`, falls back to
    /// `$TMPDIR/barracuda/barracuda.sock` via `std::env::temp_dir()`.
    #[cfg(unix)]
    pub fn default_socket_path() -> std::path::PathBuf {
        let base = std::env::var("XDG_RUNTIME_DIR")
            .map_or_else(|_| std::env::temp_dir(), std::path::PathBuf::from);
        base.join("barracuda").join("barracuda.sock")
    }
}

const SERIALIZATION_ERROR: &str =
    r#"{"jsonrpc":"2.0","error":{"code":-32603,"message":"Serialization error"},"id":null}"#;

/// Handle a single TCP connection (newline-delimited JSON-RPC).
async fn handle_stream(primal: Arc<BarraCudaPrimal>, stream: tokio::net::TcpStream) {
    let (reader, mut writer) = stream.into_split();
    let mut lines = BufReader::new(reader).lines();

    while let Ok(Some(line)) = lines.next_line().await {
        let Some(response) = handle_line(&primal, &line).await else {
            continue;
        };
        let mut json =
            serde_json::to_string(&response).unwrap_or_else(|_| SERIALIZATION_ERROR.to_string());
        json.push('\n');
        if writer.write_all(json.as_bytes()).await.is_err() {
            break;
        }
    }
}

/// Wait for SIGINT (Ctrl-C) or SIGTERM for graceful shutdown.
async fn shutdown_signal() {
    let ctrl_c = tokio::signal::ctrl_c();

    #[cfg(unix)]
    {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut sigterm) => {
                tokio::select! {
                    _ = ctrl_c => {}
                    _ = sigterm.recv() => {}
                }
            }
            Err(e) => {
                tracing::warn!("failed to register SIGTERM handler: {e}");
                let _ = ctrl_c.await;
            }
        }
    }

    #[cfg(not(unix))]
    {
        ctrl_c.await.ok();
    }
}

/// Parse a single line of JSON-RPC and dispatch to the method handler.
///
/// Returns `None` for notifications (requests without `id`), per JSON-RPC 2.0
/// spec: "The Server MUST NOT reply to a Notification".
async fn handle_line(primal: &BarraCudaPrimal, line: &str) -> Option<JsonRpcResponse> {
    let request: JsonRpcRequest = match serde_json::from_str(line) {
        Ok(r) => r,
        Err(_) => return Some(JsonRpcResponse::parse_error()),
    };

    if let Err(err_resp) = request.validate() {
        return Some(err_resp);
    }

    let id = match request.id {
        Some(ref v) if !v.is_null() => v.clone(),
        _ => return None,
    };
    Some(methods::dispatch(primal, &request.method, &request.params, id).await)
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "suppressed")]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_handle_valid_request() {
        let primal = BarraCudaPrimal::new();
        let line = r#"{"jsonrpc":"2.0","method":"barracuda.device.list","params":{},"id":1}"#;
        let resp = handle_line(&primal, line).await.expect("non-notification");
        assert!(resp.result.is_some());
        assert_eq!(resp.id, serde_json::json!(1));
    }

    #[tokio::test]
    async fn test_handle_parse_error() {
        let primal = BarraCudaPrimal::new();
        let resp = handle_line(&primal, "not json")
            .await
            .expect("parse error returns response");
        assert!(resp.error.is_some());
        assert_eq!(resp.error.unwrap().code, super::super::jsonrpc::PARSE_ERROR);
    }

    #[tokio::test]
    async fn test_handle_unknown_method() {
        let primal = BarraCudaPrimal::new();
        let line = r#"{"jsonrpc":"2.0","method":"nonexistent","params":{},"id":2}"#;
        let resp = handle_line(&primal, line).await.expect("non-notification");
        assert!(resp.error.is_some());
    }

    #[tokio::test]
    async fn test_notification_no_response() {
        let primal = BarraCudaPrimal::new();
        let line = r#"{"jsonrpc":"2.0","method":"barracuda.device.list","params":{}}"#;
        assert!(
            handle_line(&primal, line).await.is_none(),
            "notification must not produce response"
        );
    }

    #[tokio::test]
    async fn test_notification_null_id_no_response() {
        let primal = BarraCudaPrimal::new();
        let line = r#"{"jsonrpc":"2.0","method":"barracuda.device.list","params":{},"id":null}"#;
        assert!(
            handle_line(&primal, line).await.is_none(),
            "null id is a notification"
        );
    }
}

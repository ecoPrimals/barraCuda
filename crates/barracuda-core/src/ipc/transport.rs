// SPDX-License-Identifier: AGPL-3.0-or-later
//! Transport layer for barraCuda IPC.
//!
//! Transport-agnostic JSON-RPC 2.0 handler over any `AsyncRead + AsyncWrite`.
//! Supports Unix domain sockets (primary) and TCP (fallback), per
//! wateringHole `UNIVERSAL_IPC_STANDARD_V3.md` and `ECOBIN_ARCHITECTURE_STANDARD.md`.

use super::jsonrpc::{JsonRpcRequest, JsonRpcResponse};
use super::methods;
use crate::BarraCudaPrimal;
use barracuda::error::Result;
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncRead, AsyncWrite, AsyncWriteExt, BufReader};
use tokio::net::TcpListener;

const DEFAULT_MAX_FRAME_BYTES: usize = 256 * 1024 * 1024;
const DEFAULT_MAX_CONNECTIONS: usize = 10;

static MAX_FRAME_BYTES: std::sync::LazyLock<usize> = std::sync::LazyLock::new(|| {
    std::env::var("BARRACUDA_MAX_FRAME_BYTES")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_MAX_FRAME_BYTES)
});

static MAX_CONNECTIONS: std::sync::LazyLock<usize> = std::sync::LazyLock::new(|| {
    std::env::var("BARRACUDA_MAX_CONNECTIONS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_MAX_CONNECTIONS)
});

fn max_frame_bytes() -> usize {
    *MAX_FRAME_BYTES
}

fn max_connections() -> usize {
    *MAX_CONNECTIONS
}

/// Default TCP bind host when no environment or CLI override is provided.
///
/// `127.0.0.1` = localhost-only. This is the secure default: the primal
/// listens only on the loopback interface. External access requires explicit
/// configuration via `BARRACUDA_IPC_HOST` or `--bind`.
const DEFAULT_BIND_HOST: &str = "127.0.0.1";

/// Default family ID when `BIOMEOS_FAMILY_ID` is not set.
const DEFAULT_FAMILY_ID: &str = "default";

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
    std::env::var("BARRACUDA_IPC_PORT")
        .map_or_else(|_| format!("{host}:0"), |port| format!("{host}:{port}"))
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
    pub const fn new(primal: Arc<BarraCudaPrimal>) -> Self {
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

        listener.config_mut().max_frame_length(max_frame_bytes());

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
            .buffer_unordered(max_connections())
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
    ///
    /// If `on_ready` is provided, it is invoked after the listener is bound
    /// (e.g. for systemd Type=notify via `sd_notify`).
    #[cfg(unix)]
    pub async fn serve_unix<F>(&self, path: &std::path::Path, on_ready: Option<F>) -> Result<()>
    where
        F: FnOnce() + Send,
    {
        if path.exists() {
            std::fs::remove_file(path)?;
        }
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let listener = tokio::net::UnixListener::bind(path)?;
        tracing::info!("barraCuda IPC listening on unix://{}", path.display());

        if let Some(f) = on_ready {
            f();
        }

        loop {
            tokio::select! {
                result = listener.accept() => {
                    let (stream, _) = result?;
                    let primal = Arc::clone(&self.primal);
                    tokio::spawn(async move {
                        let (reader, writer) = stream.into_split();
                        handle_connection(primal, reader, writer).await;
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

    /// Resolve the default IPC socket path per wateringHole `PRIMAL_IPC_PROTOCOL`.
    ///
    /// Path: `$XDG_RUNTIME_DIR/biomeos/barracuda-{family_id}.sock`
    /// Family ID: `$BIOMEOS_FAMILY_ID` or `"default"`.
    /// Fallback base: `$TMPDIR/biomeos/` via `std::env::temp_dir()`.
    #[cfg(unix)]
    pub fn default_socket_path() -> std::path::PathBuf {
        let family_id =
            std::env::var("BIOMEOS_FAMILY_ID").unwrap_or_else(|_| DEFAULT_FAMILY_ID.to_owned());
        let base = std::env::var("XDG_RUNTIME_DIR")
            .map_or_else(|_| std::env::temp_dir(), std::path::PathBuf::from);
        base.join("biomeos")
            .join(format!("barracuda-{family_id}.sock"))
    }
}

const SERIALIZATION_ERROR: &str =
    r#"{"jsonrpc":"2.0","error":{"code":-32603,"message":"Serialization error"},"id":null}"#;

/// Transport-agnostic JSON-RPC 2.0 connection handler.
///
/// Works over any `AsyncRead + AsyncWrite` stream — TCP, Unix socket,
/// or future transports (named pipes, abstract sockets). This is the
/// ecoBin v2.0 transport-agnostic protocol handler: add a new transport
/// by binding a listener and passing accepted connections here.
async fn handle_connection<R, W>(primal: Arc<BarraCudaPrimal>, reader: R, mut writer: W)
where
    R: AsyncRead + Unpin + Send,
    W: AsyncWrite + Unpin + Send,
{
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

/// Handle a single TCP connection (newline-delimited JSON-RPC).
async fn handle_stream(primal: Arc<BarraCudaPrimal>, stream: tokio::net::TcpStream) {
    let (reader, writer) = stream.into_split();
    handle_connection(primal, reader, writer).await;
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
#[expect(
    clippy::unwrap_used,
    reason = "test assertions: unwrap is idiomatic for test code"
)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_handle_valid_request() {
        let primal = BarraCudaPrimal::new();
        let line = r#"{"jsonrpc":"2.0","method":"device.list","params":{},"id":1}"#;
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
        let line = r#"{"jsonrpc":"2.0","method":"device.list","params":{}}"#;
        assert!(
            handle_line(&primal, line).await.is_none(),
            "notification must not produce response"
        );
    }

    #[tokio::test]
    async fn test_notification_null_id_no_response() {
        let primal = BarraCudaPrimal::new();
        let line = r#"{"jsonrpc":"2.0","method":"device.list","params":{},"id":null}"#;
        assert!(
            handle_line(&primal, line).await.is_none(),
            "null id is a notification"
        );
    }

    // ─── resolve_bind_address tests ───

    #[test]
    fn resolve_explicit_addr() {
        assert_eq!(resolve_bind_address(Some("0.0.0.0:8080")), "0.0.0.0:8080");
    }

    #[test]
    fn resolve_explicit_always_wins() {
        // Explicit always takes priority regardless of env state
        let addr = resolve_bind_address(Some("10.0.0.1:9000"));
        assert_eq!(addr, "10.0.0.1:9000");
    }

    // ─── handle_connection integration tests ───

    #[tokio::test]
    async fn handle_connection_valid_request() {
        let primal = Arc::new(BarraCudaPrimal::new());
        let request = r#"{"jsonrpc":"2.0","method":"health.check","params":{},"id":1}"#;
        let input = format!("{request}\n");

        let (reader, _) = tokio::io::duplex(4096);
        let (_, writer) = tokio::io::duplex(4096);

        // Use a proper in-memory stream pair
        let (client_reader, mut server_writer) = tokio::io::duplex(4096);
        let (mut server_reader_buf, client_writer) = tokio::io::duplex(4096);

        server_writer.write_all(input.as_bytes()).await.unwrap();
        server_writer.shutdown().await.unwrap();
        drop(reader);
        drop(writer);

        handle_connection(primal, client_reader, client_writer).await;

        let mut response = String::new();
        tokio::io::AsyncReadExt::read_to_string(&mut server_reader_buf, &mut response)
            .await
            .unwrap();
        assert!(response.contains("\"jsonrpc\":\"2.0\""));
        assert!(response.contains("\"result\""));
    }

    #[tokio::test]
    async fn handle_connection_parse_error() {
        let primal = Arc::new(BarraCudaPrimal::new());
        let input = "not valid json\n";

        let (client_reader, mut server_writer) = tokio::io::duplex(4096);
        let (mut server_reader_buf, client_writer) = tokio::io::duplex(4096);

        server_writer.write_all(input.as_bytes()).await.unwrap();
        server_writer.shutdown().await.unwrap();

        handle_connection(primal, client_reader, client_writer).await;

        let mut response = String::new();
        tokio::io::AsyncReadExt::read_to_string(&mut server_reader_buf, &mut response)
            .await
            .unwrap();
        assert!(
            response.contains("-32700"),
            "should contain parse error code"
        );
    }

    #[tokio::test]
    async fn handle_connection_notification_no_reply() {
        let primal = Arc::new(BarraCudaPrimal::new());
        let input = "{\"jsonrpc\":\"2.0\",\"method\":\"device.list\",\"params\":{}}\n";

        let (client_reader, mut server_writer) = tokio::io::duplex(4096);
        let (mut server_reader_buf, client_writer) = tokio::io::duplex(4096);

        server_writer.write_all(input.as_bytes()).await.unwrap();
        server_writer.shutdown().await.unwrap();

        handle_connection(primal, client_reader, client_writer).await;

        let mut response = String::new();
        tokio::io::AsyncReadExt::read_to_string(&mut server_reader_buf, &mut response)
            .await
            .unwrap();
        assert!(response.is_empty(), "notification must produce no response");
    }
}

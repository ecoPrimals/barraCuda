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

/// Maximum tarpc frame length. `usize::MAX` allows arbitrarily large
/// payloads — tighten this if DoS protection is needed at the transport layer.
const TARPC_MAX_FRAME_LENGTH: usize = usize::MAX;

/// Maximum concurrent tarpc connections processed simultaneously.
const TARPC_MAX_CONCURRENT_CONNECTIONS: usize = 10;

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
            .max_frame_length(TARPC_MAX_FRAME_LENGTH);

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
            .buffer_unordered(TARPC_MAX_CONCURRENT_CONNECTIONS)
            .for_each(|()| async {})
            .await;

        Ok(())
    }

    /// Start listening on TCP (JSON-RPC 2.0). Returns the bound address.
    pub async fn serve_tcp(&self, addr: &str) -> Result<()> {
        let listener = TcpListener::bind(addr).await?;
        let local_addr = listener.local_addr()?;
        tracing::info!("barraCuda IPC listening on tcp://{local_addr}");

        loop {
            let (stream, peer) = listener.accept().await?;
            tracing::debug!("IPC connection from {peer}");
            let primal = Arc::clone(&self.primal);

            tokio::spawn(async move {
                let (reader, mut writer) = stream.into_split();
                let mut lines = BufReader::new(reader).lines();

                while let Ok(Some(line)) = lines.next_line().await {
                    let response = handle_line(&primal, &line).await;
                    let mut json = serde_json::to_string(&response).unwrap_or_else(|_| {
                        r#"{"jsonrpc":"2.0","error":{"code":-32603,"message":"Serialization error"},"id":null}"#.to_string()
                    });
                    json.push('\n');

                    if writer.write_all(json.as_bytes()).await.is_err() {
                        break;
                    }
                }
            });
        }
    }

    /// Start listening on a Unix domain socket.
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
            let (stream, _) = listener.accept().await?;
            let primal = Arc::clone(&self.primal);

            tokio::spawn(async move {
                let (reader, mut writer) = stream.into_split();
                let mut lines = BufReader::new(reader).lines();

                while let Ok(Some(line)) = lines.next_line().await {
                    let response = handle_line(&primal, &line).await;
                    let mut json = serde_json::to_string(&response).unwrap_or_else(|_| {
                        r#"{"jsonrpc":"2.0","error":{"code":-32603,"message":"Serialization error"},"id":null}"#.to_string()
                    });
                    json.push('\n');

                    if writer.write_all(json.as_bytes()).await.is_err() {
                        break;
                    }
                }
            });
        }
    }
}

/// Parse a single line of JSON-RPC and dispatch to the method handler.
async fn handle_line(primal: &BarraCudaPrimal, line: &str) -> JsonRpcResponse {
    let request: JsonRpcRequest = match serde_json::from_str(line) {
        Ok(r) => r,
        Err(_) => return JsonRpcResponse::parse_error(),
    };

    if let Err(err_resp) = request.validate() {
        return err_resp;
    }

    let id = request.id.clone().unwrap_or(serde_json::Value::Null);
    methods::dispatch(primal, &request.method, &request.params, id).await
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_handle_valid_request() {
        let primal = BarraCudaPrimal::new();
        let line = r#"{"jsonrpc":"2.0","method":"barracuda.device.list","params":{},"id":1}"#;
        let resp = handle_line(&primal, line).await;
        assert!(resp.result.is_some());
        assert_eq!(resp.id, serde_json::json!(1));
    }

    #[tokio::test]
    async fn test_handle_parse_error() {
        let primal = BarraCudaPrimal::new();
        let resp = handle_line(&primal, "not json").await;
        assert!(resp.error.is_some());
        assert_eq!(resp.error.unwrap().code, super::super::jsonrpc::PARSE_ERROR);
    }

    #[tokio::test]
    async fn test_handle_unknown_method() {
        let primal = BarraCudaPrimal::new();
        let line = r#"{"jsonrpc":"2.0","method":"nonexistent","params":{},"id":2}"#;
        let resp = handle_line(&primal, line).await;
        assert!(resp.error.is_some());
    }
}

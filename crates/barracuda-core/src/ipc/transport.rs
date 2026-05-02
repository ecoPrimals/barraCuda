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
pub const DEFAULT_BIND_HOST: &str = "127.0.0.1";

/// Resolve the TCP bind host from the environment.
///
/// Checks `BARRACUDA_IPC_HOST`, falling back to [`DEFAULT_BIND_HOST`].
/// Used when `--port` supplies only a port number and the host must come
/// from the environment (e.g. Docker containers that need `0.0.0.0` to
/// accept cross-container probes).
pub fn resolve_bind_host() -> String {
    std::env::var("BARRACUDA_IPC_HOST").unwrap_or_else(|_| DEFAULT_BIND_HOST.to_string())
}

/// Default family ID when no `FAMILY_ID` env var is set.
const DEFAULT_FAMILY_ID: &str = "default";

/// Default ecosystem socket namespace per `PRIMAL_IPC_PROTOCOL.md`.
///
/// All primals place Unix sockets under `$XDG_RUNTIME_DIR/{namespace}/`.
/// Override at runtime with the `BIOMEOS_SOCKET_DIR` environment variable.
pub(crate) const DEFAULT_ECOSYSTEM_SOCKET_DIR: &str = "biomeos";

/// Resolve the family ID per `PRIMAL_SELF_KNOWLEDGE_STANDARD.md` §4.
///
/// Precedence: `BARRACUDA_FAMILY_ID` → `FAMILY_ID` → `BIOMEOS_FAMILY_ID` (legacy).
/// Returns `None` when unset or `"default"`.
pub fn resolve_family_id() -> Option<String> {
    const KEYS: &[&str] = &["BARRACUDA_FAMILY_ID", "FAMILY_ID", "BIOMEOS_FAMILY_ID"];
    for key in KEYS {
        if let Ok(val) = std::env::var(key) {
            if !val.is_empty() && val != DEFAULT_FAMILY_ID {
                return Some(val);
            }
        }
    }
    None
}

/// Resolve the socket directory per `PRIMAL_SELF_KNOWLEDGE_STANDARD.md` §3.
///
/// Resolution: `BIOMEOS_SOCKET_DIR` → `$XDG_RUNTIME_DIR/biomeos` → `$TMPDIR/biomeos`.
pub fn resolve_socket_dir() -> std::path::PathBuf {
    if let Ok(dir) = std::env::var("BIOMEOS_SOCKET_DIR") {
        return std::path::PathBuf::from(dir);
    }
    let base = std::env::var("XDG_RUNTIME_DIR")
        .map_or_else(|_| std::env::temp_dir(), std::path::PathBuf::from);
    base.join(DEFAULT_ECOSYSTEM_SOCKET_DIR)
}

/// Validate the `BIOMEOS_INSECURE` guard per `BTSP_PROTOCOL_STANDARD.md` §Compliance.
///
/// When `FAMILY_ID` is set (non-default), `BIOMEOS_INSECURE=1` MUST NOT also be
/// set. You cannot claim a family AND skip authentication.
///
/// # Errors
///
/// Returns [`crate::error::BarracudaCoreError::Lifecycle`] when both `FAMILY_ID`
/// and `BIOMEOS_INSECURE` are set.
pub fn validate_insecure_guard() -> crate::error::Result<()> {
    let family_id = resolve_family_id();
    let insecure = std::env::var("BIOMEOS_INSECURE")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);
    if let Some(ref fid) = family_id {
        if insecure {
            return Err(crate::error::BarracudaCoreError::lifecycle(format!(
                "FAMILY_ID={fid} but BIOMEOS_INSECURE=1 — cannot claim a family \
                 and skip authentication. Unset one or the other. \
                 See BTSP_PROTOCOL_STANDARD.md §Compliance."
            )));
        }
    }
    Ok(())
}

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
    ///
    /// On `AddrInUse`, logs a warning and returns `Ok(())` — the tarpc
    /// endpoint is optional and should not crash the binary when the port
    /// is occupied by another primal.
    pub async fn serve_tarpc(&self, addr: &str) -> Result<()> {
        use crate::rpc::{BarraCudaServer, BarraCudaService};
        use futures::prelude::*;
        use tarpc::server::{self, Channel};

        let mut listener = match tarpc::serde_transport::tcp::listen(
            addr,
            tarpc::tokio_serde::formats::Json::default,
        )
        .await
        {
            Ok(l) => l,
            Err(e) if e.kind() == std::io::ErrorKind::AddrInUse => {
                tracing::warn!(
                    addr,
                    "tarpc TCP skipped: address already in use \
                     (another primal may occupy this port). \
                     Primary transport is unaffected."
                );
                return Ok(());
            }
            Err(e) => return Err(e.into()),
        };
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

    /// Serve the tarpc binary RPC endpoint on a Unix domain socket.
    ///
    /// Transport parity with `serve_tarpc` (TCP) — both JSON-RPC and tarpc
    /// are now available on Unix sockets for local composition without TCP
    /// overhead. Uses `serde_transport::new` over a `UnixStream` split.
    #[cfg(unix)]
    pub async fn serve_tarpc_unix(&self, path: &std::path::Path) -> Result<()> {
        use crate::rpc::{BarraCudaServer, BarraCudaService};
        use futures::StreamExt;
        use tarpc::server::{self, Channel};
        use tokio_serde::formats::Json;

        if path.exists() {
            std::fs::remove_file(path)?;
        }
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let listener = tokio::net::UnixListener::bind(path)?;
        tracing::info!("barraCuda tarpc listening on unix://{}", path.display());

        let server = BarraCudaServer::new(Arc::clone(&self.primal));

        loop {
            tokio::select! {
                result = listener.accept() => {
                    let (mut stream, _) = result?;
                    let srv = server.clone();
                    tokio::spawn(async move {
                        let outcome = super::btsp::guard_connection(&mut stream).await;
                        if !outcome.should_accept() {
                            tracing::warn!("BTSP handshake rejected tarpc connection: {outcome:?}");
                            return;
                        }
                        if let Some(session) = outcome.session() {
                            if session.cipher.requires_key() {
                                tracing::warn!(
                                    cipher = ?session.cipher,
                                    "tarpc does not support BTSP-encrypted frames — \
                                     rejecting keyed-cipher connection (use JSON-RPC for \
                                     encrypted transport)"
                                );
                                return;
                            }
                        }
                        let transport = tarpc::serde_transport::new(
                            tokio_util::codec::LengthDelimitedCodec::builder()
                                .max_frame_length(max_frame_bytes())
                                .new_framed(stream),
                            Json::default(),
                        );
                        let channel = server::BaseChannel::with_defaults(transport);
                        channel
                            .execute(srv.serve())
                            .for_each(|response| {
                                tokio::spawn(response);
                                async {}
                            })
                            .await;
                    });
                }
                () = shutdown_signal() => {
                    tracing::info!("Shutdown signal received, stopping tarpc Unix server");
                    break;
                }
            }
        }

        if path.exists() {
            std::fs::remove_file(path).ok();
        }
        Ok(())
    }

    /// Try to bind a TCP listener without starting the accept loop.
    ///
    /// Returns the bound listener and local address on success, or `None`
    /// when the port is already occupied (logs a warning). This separates
    /// bind from serve so callers can write the discovery file only after
    /// confirming the bind succeeded — avoiding phantom endpoints (LD-05).
    pub async fn try_bind_tcp(addr: &str) -> Option<(TcpListener, std::net::SocketAddr)> {
        match TcpListener::bind(addr).await {
            Ok(listener) => match listener.local_addr() {
                Ok(local) => Some((listener, local)),
                Err(e) => {
                    tracing::warn!(addr, error = %e, "TCP bind succeeded but local_addr failed");
                    None
                }
            },
            Err(e) if e.kind() == std::io::ErrorKind::AddrInUse => {
                tracing::warn!(
                    addr,
                    "TCP sidecar skipped: address already in use \
                     (another primal may occupy this port). \
                     UDS primary transport is unaffected."
                );
                None
            }
            Err(e) => {
                tracing::warn!(addr, error = %e, "TCP sidecar bind failed");
                None
            }
        }
    }

    /// Start listening on TCP (JSON-RPC 2.0) with graceful shutdown on
    /// SIGINT/SIGTERM per wateringHole `UNIBIN_ARCHITECTURE_STANDARD.md`.
    pub async fn serve_tcp(&self, addr: &str) -> Result<()> {
        let listener = TcpListener::bind(addr).await?;
        let local_addr = listener.local_addr()?;
        tracing::info!("barraCuda IPC listening on tcp://{local_addr}");
        self.serve_tcp_listener(listener).await
    }

    /// Run the JSON-RPC accept loop on a pre-bound TCP listener.
    ///
    /// Use [`IpcServer::try_bind_tcp`] to obtain the listener, then call this to start
    /// serving. This two-step pattern allows writing the discovery file
    /// between bind and serve — only advertising TCP if the bind succeeded.
    pub async fn serve_tcp_listener(&self, listener: TcpListener) -> Result<()> {
        if let Ok(addr) = listener.local_addr() {
            tracing::info!("barraCuda IPC listening on tcp://{addr}");
        }

        loop {
            tokio::select! {
                result = listener.accept() => {
                    let (mut stream, peer) = result?;
                    tracing::debug!("IPC connection from {peer}");
                    let primal = Arc::clone(&self.primal);
                    tokio::spawn(async move {
                        let outcome = super::btsp::guard_connection(&mut stream).await;
                        if !outcome.should_accept() {
                            tracing::warn!("BTSP handshake rejected connection from {peer}: {outcome:?}");
                            return;
                        }
                        let session = outcome.session().cloned();
                        let replay = outcome.consumed_line().map(String::from);
                        if let Some(ref session) = session {
                            if session.cipher.requires_key() {
                                let (reader, writer) = stream.into_split();
                                handle_btsp_connection(primal, reader, writer, session).await;
                                return;
                            }
                        }
                        handle_stream(primal, stream, session, replay).await;
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
                    let (mut stream, _) = result?;
                    let primal = Arc::clone(&self.primal);
                    tokio::spawn(async move {
                        let outcome = super::btsp::guard_connection(&mut stream).await;
                        if !outcome.should_accept() {
                            tracing::warn!("BTSP handshake rejected connection: {outcome:?}");
                            return;
                        }
                        let session = outcome.session().cloned();
                        let replay = outcome.consumed_line().map(String::from);
                        let (reader, writer) = stream.into_split();
                        if let Some(ref session) = session {
                            if session.cipher.requires_key() {
                                handle_btsp_connection(primal, reader, writer, session).await;
                                return;
                            }
                        }
                        handle_connection(primal, reader, writer, session, replay).await;
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
    /// Domain-based: `$BIOMEOS_SOCKET_DIR/math.sock`
    /// With family ID: `$BIOMEOS_SOCKET_DIR/math-{family_id}.sock`
    ///
    /// Per `PRIMAL_SELF_KNOWLEDGE_STANDARD.md` §3: primals bind using their
    /// capability domain stem, not their primal name.
    #[cfg(unix)]
    pub fn default_socket_path() -> std::path::PathBuf {
        let dir = resolve_socket_dir();
        let domain = crate::PRIMAL_DOMAIN;
        let sock_name = match resolve_family_id() {
            Some(family_id) => format!("{domain}-{family_id}.sock"),
            None => format!("{domain}.sock"),
        };
        dir.join(sock_name)
    }

    /// Create a legacy primal-named symlink for backward compatibility.
    ///
    /// Per `PRIMAL_SELF_KNOWLEDGE_STANDARD.md` §3 "Legacy compatibility":
    /// during migration, primals MAY symlink `{primal}.sock → {domain}.sock`
    /// so consumers using identity-based discovery can still find the socket.
    #[cfg(unix)]
    pub fn create_legacy_symlink(socket_path: &std::path::Path) {
        let dir = resolve_socket_dir();
        let ns = crate::PRIMAL_NAMESPACE;
        let legacy_name = match resolve_family_id() {
            Some(family_id) => format!("{ns}-{family_id}.sock"),
            None => format!("{ns}.sock"),
        };
        let legacy_path = dir.join(legacy_name);
        if legacy_path.exists() {
            std::fs::remove_file(&legacy_path).ok();
        }
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(socket_path, &legacy_path).ok();
        }
    }

    /// Remove the legacy primal-named symlink on shutdown.
    #[cfg(unix)]
    pub fn remove_legacy_symlink() {
        let dir = resolve_socket_dir();
        let ns = crate::PRIMAL_NAMESPACE;
        let legacy_name = match resolve_family_id() {
            Some(family_id) => format!("{ns}-{family_id}.sock"),
            None => format!("{ns}.sock"),
        };
        let legacy_path = dir.join(legacy_name);
        if legacy_path.is_symlink() {
            std::fs::remove_file(&legacy_path).ok();
        }
    }

    /// Resolve the default TCP port from environment.
    ///
    /// Per `plasmidBin/ports.env`, each primal has a canonical TCP port assigned
    /// via `{PRIMAL}_PORT` env var. When set, the server binds TCP alongside UDS.
    pub fn default_tcp_port() -> Option<u16> {
        std::env::var("BARRACUDA_PORT")
            .ok()
            .and_then(|v| v.parse().ok())
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
///
/// Supports both single requests and JSON-RPC 2.0 batch requests (JSON
/// arrays). Per spec: an empty batch returns a parse error; notifications
/// within a batch produce no response entries; if all requests are
/// notifications, no response is sent.
///
/// `replay` is an optional first line consumed during the BTSP handshake
/// guard. When `FAMILY_ID` is set and a legacy (non-BTSP) client connects,
/// the guard reads the first line looking for a `ClientHello`. If the line
/// is a plain JSON-RPC request, the guard returns it here for replay so
/// the request is not silently dropped (LD-10 fix).
async fn handle_connection<R, W>(
    primal: Arc<BarraCudaPrimal>,
    reader: R,
    mut writer: W,
    session: Option<super::btsp::BtspSession>,
    replay: Option<String>,
) where
    R: AsyncRead + Unpin + Send,
    W: AsyncWrite + Unpin + Send,
{
    if let Some(ref line) = replay {
        if dispatch_line(&primal, line, &mut writer).await.is_err() {
            return;
        }
    }
    let mut buf_reader = BufReader::new(reader);
    let mut lines = (&mut buf_reader).lines();
    while let Ok(Some(line)) = lines.next_line().await {
        if let Some(upgraded) = try_negotiate_upgrade(session.as_ref(), &line, &mut writer).await {
            tracing::info!("Phase 3 negotiate: switching to encrypted framing");
            handle_btsp_connection(primal, buf_reader.into_inner(), writer, &upgraded).await;
            return;
        }
        if dispatch_line(&primal, &line, &mut writer).await.is_err() {
            break;
        }
    }
}

/// If `line` is a `btsp.negotiate` request and the session supports it,
/// write the response and return the upgraded `BtspSession`. Returns `None`
/// for non-negotiate lines or when the cipher stays NULL.
async fn try_negotiate_upgrade<W>(
    session: Option<&super::btsp::BtspSession>,
    line: &str,
    writer: &mut W,
) -> Option<super::btsp::BtspSession>
where
    W: AsyncWrite + Unpin + Send,
{
    let session = session.as_ref()?;
    let parsed: serde_json::Value = serde_json::from_str(line.trim()).ok()?;
    if parsed.get("method")?.as_str()? != "btsp.negotiate" {
        return None;
    }
    let id = parsed.get("id").cloned().unwrap_or(serde_json::Value::Null);
    let params = parsed
        .get("params")
        .cloned()
        .unwrap_or(serde_json::Value::Null);
    let (resp, upgraded) = match super::btsp::negotiate_phase3(session, &params) {
        Ok(nr) => {
            let keyed = nr.session.cipher.requires_key();
            (
                JsonRpcResponse::success(id, nr.response),
                keyed.then_some(nr.session),
            )
        }
        Err(reason) => (JsonRpcResponse::error(id, -32602, reason), None),
    };
    let mut json = serde_json::to_string(&resp).unwrap_or_else(|_| SERIALIZATION_ERROR.to_string());
    json.push('\n');
    if writer.write_all(json.as_bytes()).await.is_err() {
        return None;
    }
    if upgraded.is_some() {
        let _ = writer.flush().await;
    }
    upgraded
}

/// Dispatch a single NDJSON line (single request or batch array) and write
/// the response. Returns `Err(())` on write failure (caller should close).
async fn dispatch_line<W>(
    primal: &Arc<BarraCudaPrimal>,
    line: &str,
    writer: &mut W,
) -> std::result::Result<(), ()>
where
    W: AsyncWrite + Unpin + Send,
{
    let trimmed = line.trim_start();
    if trimmed.starts_with('[') {
        let Some(batch_json) = handle_batch(primal, trimmed).await else {
            return Ok(());
        };
        let mut out = batch_json;
        out.push('\n');
        writer.write_all(out.as_bytes()).await.map_err(|_| ())?;
    } else {
        let Some(response) = handle_line(primal, line).await else {
            return Ok(());
        };
        let mut json =
            serde_json::to_string(&response).unwrap_or_else(|_| SERIALIZATION_ERROR.to_string());
        json.push('\n');
        writer.write_all(json.as_bytes()).await.map_err(|_| ())?;
    }
    Ok(())
}

/// Handle a single TCP connection (newline-delimited JSON-RPC).
async fn handle_stream(
    primal: Arc<BarraCudaPrimal>,
    stream: tokio::net::TcpStream,
    session: Option<super::btsp::BtspSession>,
    replay: Option<String>,
) {
    let (reader, writer) = stream.into_split();
    handle_connection(primal, reader, writer, session, replay).await;
}

/// Handle a BTSP Phase 3 connection using length-prefixed encrypted frames.
///
/// Each frame is decrypted, dispatched as JSON-RPC, and the response
/// encrypted back. Per `BTSP_PROTOCOL_STANDARD.md` §Wire Framing.
async fn handle_btsp_connection<R, W>(
    primal: Arc<BarraCudaPrimal>,
    reader: R,
    writer: W,
    session: &super::btsp::BtspSession,
) where
    R: AsyncRead + Unpin + Send,
    W: AsyncWrite + Unpin + Send,
{
    use super::btsp_frame::{
        BtspFrameReader, BtspFrameWriter, read_frame_as_line, write_line_as_frame,
    };

    let mut frame_reader = BtspFrameReader::new(reader, session);
    let mut frame_writer = BtspFrameWriter::new(writer, session);

    while let Some(line) = read_frame_as_line(&mut frame_reader).await {
        let trimmed = line.trim_start();
        if trimmed.starts_with('[') {
            let Some(batch_json) = handle_batch(&primal, trimmed).await else {
                continue;
            };
            if write_line_as_frame(&mut frame_writer, &batch_json)
                .await
                .is_err()
            {
                break;
            }
        } else {
            let Some(response) = handle_line(&primal, &line).await else {
                continue;
            };
            let json = serde_json::to_string(&response)
                .unwrap_or_else(|_| SERIALIZATION_ERROR.to_string());
            if write_line_as_frame(&mut frame_writer, &json).await.is_err() {
                break;
            }
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

/// Validate and dispatch a parsed `JsonRpcRequest`, moving the `id` to avoid
/// cloning the `serde_json::Value` on every request.
///
/// Returns `None` for notifications (requests without `id`), per JSON-RPC 2.0
/// spec: "The Server MUST NOT reply to a Notification".
async fn handle_request(
    primal: &BarraCudaPrimal,
    request: JsonRpcRequest,
) -> Option<JsonRpcResponse> {
    if let Err(err_resp) = request.validate() {
        return Some(err_resp);
    }

    let JsonRpcRequest {
        id, method, params, ..
    } = request;
    let id = match id {
        Some(v) if !v.is_null() => v,
        _ => return None,
    };
    Some(methods::dispatch(primal, &method, &params, id).await)
}

/// Parse a single line of JSON-RPC and dispatch to the method handler.
async fn handle_line(primal: &BarraCudaPrimal, line: &str) -> Option<JsonRpcResponse> {
    let request: JsonRpcRequest = match serde_json::from_str(line) {
        Ok(r) => r,
        Err(_) => return Some(JsonRpcResponse::parse_error()),
    };
    handle_request(primal, request).await
}

/// Handle a JSON-RPC 2.0 batch request (JSON array on a single line).
///
/// Per JSON-RPC 2.0 spec §6:
/// - An empty array is an invalid request.
/// - Each element is processed independently; notifications produce no entry.
/// - If all elements are notifications, no response is returned (`None`).
/// - Non-array elements within the batch produce individual parse errors.
///
/// Uses `serde_json::from_value` to avoid stringify/re-parse per element.
async fn handle_batch(primal: &BarraCudaPrimal, line: &str) -> Option<String> {
    let items: Vec<serde_json::Value> = match serde_json::from_str(line) {
        Ok(v) => v,
        Err(_) => {
            return Some(
                serde_json::to_string(&JsonRpcResponse::parse_error())
                    .unwrap_or_else(|_| SERIALIZATION_ERROR.to_string()),
            );
        }
    };

    if items.is_empty() {
        return Some(
            serde_json::to_string(&JsonRpcResponse::error(
                serde_json::Value::Null,
                super::jsonrpc::INVALID_REQUEST,
                "empty batch",
            ))
            .unwrap_or_else(|_| SERIALIZATION_ERROR.to_string()),
        );
    }

    let mut responses = Vec::new();
    for item in items {
        match serde_json::from_value::<JsonRpcRequest>(item) {
            Ok(request) => {
                if let Some(resp) = handle_request(primal, request).await {
                    responses.push(resp);
                }
            }
            Err(_) => responses.push(JsonRpcResponse::parse_error()),
        }
    }

    if responses.is_empty() {
        return None;
    }

    Some(serde_json::to_string(&responses).unwrap_or_else(|_| SERIALIZATION_ERROR.to_string()))
}

#[cfg(test)]
#[path = "transport_tests.rs"]
mod tests;

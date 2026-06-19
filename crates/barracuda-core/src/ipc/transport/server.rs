// SPDX-License-Identifier: AGPL-3.0-or-later
use super::super::transport_config::{resolve_family_id, resolve_socket_dir};
use super::connection::{handle_btsp_connection, handle_connection, handle_stream};
use crate::BarraCudaPrimal;
use crate::env_keys;
use barracuda::error::Result;
use std::sync::Arc;
use tokio::net::TcpListener;

const DEFAULT_MAX_FRAME_BYTES: usize = 256 * 1024 * 1024;
const DEFAULT_MAX_CONNECTIONS: usize = 10;

static MAX_FRAME_BYTES: std::sync::LazyLock<usize> = std::sync::LazyLock::new(|| {
    std::env::var(env_keys::BARRACUDA_MAX_FRAME_BYTES)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_MAX_FRAME_BYTES)
});

static MAX_CONNECTIONS: std::sync::LazyLock<usize> = std::sync::LazyLock::new(|| {
    std::env::var(env_keys::BARRACUDA_MAX_CONNECTIONS)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_MAX_CONNECTIONS)
});

pub(super) fn max_frame_bytes() -> usize {
    *MAX_FRAME_BYTES
}

pub(super) fn max_connections() -> usize {
    *MAX_CONNECTIONS
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
    #[cfg(feature = "tarpc-transport")]
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
    #[cfg(all(unix, feature = "tarpc-transport"))]
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
        if path.exists() {
            let _ = std::fs::remove_file(path);
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
                        let outcome = super::super::btsp::guard_connection(&mut stream).await;
                        if !outcome.should_accept() {
                            tracing::warn!("BTSP handshake rejected tarpc connection: {outcome:?}");
                            return;
                        }
                        if let Some(session) = outcome.session()
                            && session.cipher.requires_key() {
                                tracing::warn!(
                                    cipher = ?session.cipher,
                                    "tarpc does not support BTSP-encrypted frames — \
                                     rejecting keyed-cipher connection (use JSON-RPC for \
                                     encrypted transport)"
                                );
                                return;
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
                        let outcome = super::super::btsp::guard_connection(&mut stream).await;
                        if !outcome.should_accept() {
                            tracing::warn!("BTSP handshake rejected connection from {peer}: {outcome:?}");
                            return;
                        }
                        let session = outcome.session().cloned();
                        let replay = outcome.consumed_line().map(String::from);
                        if let Some(ref session) = session
                            && session.cipher.requires_key() {
                                let (reader, writer) = stream.into_split();
                                handle_btsp_connection(primal, reader, writer, session).await;
                                return;
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
        // Remove stale socket files AND broken symlinks before bind().
        // `path.exists()` follows symlinks (returns false for broken links),
        // but `symlink_metadata()` detects any filesystem entry at the path.
        if path.symlink_metadata().is_ok() {
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
                        let outcome = super::super::btsp::guard_connection(&mut stream).await;
                        if !outcome.should_accept() {
                            tracing::warn!("BTSP handshake rejected connection: {outcome:?}");
                            return;
                        }
                        let session = outcome.session().cloned();
                        let replay = outcome.consumed_line().map(String::from);
                        let (reader, writer) = stream.into_split();
                        if let Some(ref session) = session
                            && session.cipher.requires_key() {
                                handle_btsp_connection(primal, reader, writer, session).await;
                                return;
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
    ///
    /// Skips creation when the legacy path would resolve to the socket path
    /// itself (e.g., `--socket barracuda-eastgate.sock` with `FAMILY_ID=eastgate`).
    #[cfg(unix)]
    pub fn create_legacy_symlink(socket_path: &std::path::Path) {
        let dir = socket_path
            .parent()
            .map_or_else(resolve_socket_dir, std::path::Path::to_path_buf);
        let ns = crate::PRIMAL_NAMESPACE;
        let legacy_name = match resolve_family_id() {
            Some(family_id) => format!("{ns}-{family_id}.sock"),
            None => format!("{ns}.sock"),
        };
        let legacy_path = dir.join(&legacy_name);

        // Guard: skip if legacy path would be the same file as the socket.
        // This prevents self-referencing symlinks when the operator passes
        // `--socket <dir>/barracuda-<family>.sock` explicitly.
        if let (Ok(canon_legacy_dir), Ok(canon_sock)) = (
            legacy_path.parent().map_or_else(
                || std::path::PathBuf::from(".").canonicalize(),
                |p| p.canonicalize(),
            ),
            socket_path.parent().map_or_else(
                || std::path::PathBuf::from(".").canonicalize(),
                |p| p.canonicalize(),
            ),
        ) {
            let legacy_file = legacy_path.file_name();
            let sock_file = socket_path.file_name();
            if canon_legacy_dir == canon_sock && legacy_file == sock_file {
                tracing::debug!(
                    "skip legacy symlink: would self-reference {}",
                    legacy_path.display()
                );
                return;
            }
        }

        if legacy_path.exists() || legacy_path.is_symlink() {
            std::fs::remove_file(&legacy_path).ok();
        }
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink(socket_path, &legacy_path).ok();
        }
    }

    /// Remove the legacy primal-named symlink on shutdown.
    ///
    /// Derives the symlink location from the socket path's parent directory,
    /// consistent with [`Self::create_legacy_symlink`].
    #[cfg(unix)]
    pub fn remove_legacy_symlink(socket_path: &std::path::Path) {
        let dir = socket_path
            .parent()
            .map_or_else(resolve_socket_dir, std::path::Path::to_path_buf);
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
        std::env::var(env_keys::BARRACUDA_PORT)
            .ok()
            .and_then(|v| v.parse().ok())
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

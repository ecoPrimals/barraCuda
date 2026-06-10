// SPDX-License-Identifier: AGPL-3.0-or-later
//! barraCuda `UniBin` — single binary, multiple modes.
//!
//! Per wateringHole `UNIBIN_ARCHITECTURE_STANDARD.md` and
//! `GENOMEBIN_ARCHITECTURE_STANDARD.md`:
//! - One binary named after the primal
//! - Subcommands: `server`, `service`, `doctor`, `validate`, `version`

mod commands;
mod discovery_file;

use barracuda_core::BarraCudaPrimal;
use barracuda_core::lifecycle::PrimalLifecycle;
use clap::{Parser, Subcommand};
use std::sync::Arc;

/// Delay before emitting `primal.announce` to the Neural API.
///
/// Allows the server socket to fully bind and accept connections before
/// broadcasting availability to the mesh. Without this, fast consumers
/// may connect before the listener is ready.
const ANNOUNCE_DELAY_MS: u64 = 100;

#[derive(Parser)]
#[command(name = "barracuda")]
#[command(about = "barraCuda — sovereign GPU compute engine")]
#[command(version = env!("CARGO_PKG_VERSION"))]
#[command(
    long_about = "BARrier-free Rust Abstracted Cross-platform Unified Dimensional Algebra.\n\nVendor-agnostic GPU compute via WGSL. One source, any backend, identical results."
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Start the barraCuda IPC server.
    ///
    /// Per ecoBin standard, Unix domain sockets are the primary transport
    /// on Unix platforms. TCP is used as fallback or when `--bind`/`--port`
    /// is explicitly provided. Use `--no-unix` to force TCP-only mode.
    Server {
        /// TCP port for newline-delimited JSON-RPC.
        /// Per UniBin v1.1: `--port` is the universal entry point for
        /// orchestration. Springs and launchers compose primals via
        /// `{binary} server --port {port}`. Host resolved from
        /// `BARRACUDA_IPC_HOST` (default `127.0.0.1`).
        #[arg(long)]
        port: Option<u16>,

        /// TCP bind address for JSON-RPC (full `host:port` form).
        /// Overrides `--port` when both are provided.
        /// Resolved in order: `--bind`, `--port`, `BARRACUDA_IPC_BIND`,
        /// `BARRACUDA_IPC_HOST`:`BARRACUDA_IPC_PORT`, or `127.0.0.1:0` (ephemeral).
        /// When `--bind` or `--port` is provided, TCP becomes the primary transport.
        #[arg(long)]
        bind: Option<String>,

        /// TCP bind address for tarpc binary RPC (default: disabled).
        #[arg(long)]
        tarpc_bind: Option<String>,

        /// Unix socket path for tarpc binary RPC (default: disabled).
        /// When set, serves tarpc over a Unix socket alongside TCP and JSON-RPC.
        #[cfg(unix)]
        #[arg(long)]
        tarpc_unix: Option<String>,

        /// Unix socket path override. Defaults to
        /// `$BIOMEOS_SOCKET_DIR/math.sock` (or `math-{family_id}.sock`
        /// when `FAMILY_ID` is set). Legacy `barracuda.sock` symlink created.
        #[cfg(unix)]
        #[arg(long, visible_alias = "socket", num_args = 0..=1, default_missing_value = "__default__")]
        unix: Option<String>,

        /// Disable Unix socket transport (force TCP-only).
        #[cfg(unix)]
        #[arg(long)]
        no_unix: bool,

        /// Skip GPU/wgpu adapter probe — start immediately in cpu-shader-only
        /// mode. Eliminates ~30s startup delay on GPU-less hosts (broken DRM,
        /// containers, VPS). Also available via `BARRACUDA_NO_GPU_PROBE=true`.
        #[arg(long)]
        no_gpu_probe: bool,
    },

    /// Start the barraCuda IPC server in service mode (systemd/init).
    ///
    /// Per genomeBin standard: Unix socket by default (Unix), no interactive
    /// output, graceful shutdown on SIGTERM/SIGINT, optional PID file,
    /// systemd Type=notify support via `NOTIFY_SOCKET`.
    Service,

    /// Health check and diagnostics.
    Doctor,

    /// Run GPU validation suite.
    Validate {
        /// Run extended validation (FHE + QCD canary).
        #[arg(long)]
        extended: bool,
    },

    /// Invoke a JSON-RPC method against a running barraCuda server.
    ///
    /// Discovers the server via `$XDG_RUNTIME_DIR/biomeos/barracuda-core.json`,
    /// `BARRACUDA_IPC_BIND`, or falls back to `--addr`.
    Client {
        /// JSON-RPC method name (e.g. `device.list`).
        method: String,

        /// JSON params (as a JSON string). Defaults to `{}`.
        #[arg(long, default_value = "{}")]
        params: String,

        /// Explicit server address (`host:port`). Overrides discovery.
        #[arg(long)]
        addr: Option<String>,
    },

    /// Print version and build info.
    Version,
}

#[tokio::main]
async fn main() -> Result<(), barracuda_core::error::BarracudaCoreError> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Server {
            port,
            bind,
            tarpc_bind,
            #[cfg(unix)]
            tarpc_unix,
            #[cfg(unix)]
            unix,
            #[cfg(unix)]
            no_unix,
            no_gpu_probe,
        } => {
            if no_gpu_probe {
                barracuda_core::set_no_gpu_probe();
            }

            let (effective_bind, effective_unix) =
                resolve_transport_override(bind, port, unix.as_deref());

            run_server(
                effective_bind,
                tarpc_bind,
                #[cfg(unix)]
                tarpc_unix,
                #[cfg(unix)]
                effective_unix,
                #[cfg(unix)]
                no_unix,
            )
            .await?;
        }
        Commands::Service => run_service_mode().await?,
        Commands::Doctor => commands::run_doctor().await?,
        Commands::Validate { extended } => commands::run_validate(extended).await?,
        Commands::Client {
            method,
            params,
            addr,
        } => commands::run_client(&method, &params, addr.as_deref()).await?,
        Commands::Version => commands::print_version(),
    }

    Ok(())
}

/// Resolve `TRANSPORT_ENDPOINT` env override for launcher-injected transport.
#[cfg(unix)]
fn resolve_transport_override(
    bind: Option<String>,
    port: Option<u16>,
    unix: Option<&str>,
) -> (Option<String>, Option<String>) {
    use barracuda_core::ipc::transport::TransportEndpoint;

    if let Ok(raw) = std::env::var("TRANSPORT_ENDPOINT") {
        match serde_json::from_str::<TransportEndpoint>(&raw) {
            Ok(TransportEndpoint::Uds { path }) => {
                tracing::info!("TRANSPORT_ENDPOINT: UDS at {path}");
                return (None, Some(path));
            }
            Ok(TransportEndpoint::Tcp { host, port: p }) => {
                tracing::info!("TRANSPORT_ENDPOINT: TCP at {host}:{p}");
                return (Some(format!("{host}:{p}")), unix.map(String::from));
            }
            Ok(TransportEndpoint::MeshRelay { peer_id, .. }) => {
                tracing::warn!(
                    "TRANSPORT_ENDPOINT: mesh_relay ({peer_id}) not directly bindable, ignoring"
                );
            }
            Err(e) => {
                tracing::warn!("TRANSPORT_ENDPOINT parse error: {e}, falling back to CLI flags");
            }
        }
    }

    let effective_bind = bind.or_else(|| {
        port.map(|p| {
            format!(
                "{}:{p}",
                barracuda_core::ipc::transport::resolve_bind_host()
            )
        })
    });
    (effective_bind, unix.map(String::from))
}

#[cfg(not(unix))]
fn resolve_transport_override(
    bind: Option<String>,
    port: Option<u16>,
    _unix: Option<&str>,
) -> (Option<String>, Option<String>) {
    use barracuda_core::ipc::transport::TransportEndpoint;

    if let Ok(raw) = std::env::var("TRANSPORT_ENDPOINT") {
        if let Ok(TransportEndpoint::Tcp { host, port: p }) =
            serde_json::from_str::<TransportEndpoint>(&raw)
        {
            tracing::info!("TRANSPORT_ENDPOINT: TCP at {host}:{p}");
            return (Some(format!("{host}:{p}")), None);
        }
    }

    let effective_bind = bind.or_else(|| {
        port.map(|p| {
            format!(
                "{}:{p}",
                barracuda_core::ipc::transport::resolve_bind_host()
            )
        })
    });
    (effective_bind, None)
}

async fn run_server(
    bind: Option<String>,
    tarpc_bind: Option<String>,
    #[cfg(unix)] tarpc_unix: Option<String>,
    #[cfg(unix)] unix: Option<String>,
    #[cfg(unix)] no_unix: bool,
) -> Result<(), barracuda_core::error::BarracudaCoreError> {
    barracuda_core::ipc::transport::validate_insecure_guard()?;
    let mut primal = BarraCudaPrimal::new();
    primal.start().await.map_err(|e| {
        barracuda_core::error::BarracudaCoreError::lifecycle(format!("Failed to start: {e}"))
    })?;
    let primal = Arc::new(primal);

    let server = barracuda_core::ipc::IpcServer::new(Arc::clone(&primal));

    #[cfg(feature = "tarpc-transport")]
    if let Some(ref tarpc_addr) = tarpc_bind {
        let tarpc_server = barracuda_core::ipc::IpcServer::new(Arc::clone(&primal));
        let tarpc_addr = tarpc_addr.clone();
        tokio::spawn(async move {
            match tarpc_server.serve_tarpc(&tarpc_addr).await {
                Ok(()) => tracing::info!("tarpc TCP server stopped"),
                Err(e) => tracing::error!("tarpc TCP server error: {e}"),
            }
        });
    }

    #[cfg(all(unix, feature = "tarpc-transport"))]
    if let Some(ref tarpc_sock) = tarpc_unix {
        let tarpc_server = barracuda_core::ipc::IpcServer::new(Arc::clone(&primal));
        let tarpc_path = std::path::PathBuf::from(tarpc_sock);
        tokio::spawn(async move {
            if let Err(e) = tarpc_server.serve_tarpc_unix(&tarpc_path).await {
                tracing::error!("tarpc Unix server error: {e}");
            }
        });
    }

    #[cfg(unix)]
    {
        let use_unix = !no_unix;
        let explicit_unix = unix.is_some();

        if use_unix || explicit_unix {
            let sock_path = match &unix {
                Some(p) if p != "__default__" => std::path::PathBuf::from(p),
                _ => barracuda_core::ipc::IpcServer::default_socket_path(),
            };

            if let Some(ref tcp) = bind {
                if let Some((listener, local_addr)) =
                    barracuda_core::ipc::IpcServer::try_bind_tcp(tcp).await
                {
                    let effective_tcp = local_addr.to_string();
                    discovery_file::write_discovery_file(
                        Some(&effective_tcp),
                        tarpc_bind.as_deref(),
                        Some(&sock_path),
                    );
                    let tcp_server = barracuda_core::ipc::IpcServer::new(Arc::clone(&primal));
                    tokio::spawn(async move {
                        if let Err(e) = tcp_server.serve_tcp_listener(listener).await {
                            tracing::error!("TCP server error: {e}");
                        }
                    });
                } else {
                    discovery_file::write_discovery_file(
                        None,
                        tarpc_bind.as_deref(),
                        Some(&sock_path),
                    );
                }
            } else {
                discovery_file::write_discovery_file(
                    None,
                    tarpc_bind.as_deref(),
                    Some(&sock_path),
                );
            }

            barracuda_core::ipc::IpcServer::create_legacy_symlink(&sock_path);
            barracuda_core::discovery::register_with_discovery(&format!(
                "unix://{}",
                sock_path.display()
            ))
            .await;

            let announce_socket = sock_path.to_string_lossy().to_string();
            discovery_file::spawn_neural_announce(announce_socket, ANNOUNCE_DELAY_MS);

            server.serve_unix(&sock_path, None::<fn()>).await?;
            barracuda_core::ipc::IpcServer::remove_legacy_symlink(&sock_path);
            discovery_file::remove_discovery_file(Some(&sock_path));
            return Ok(());
        }
    }

    // TCP-only fallback (--no-unix or non-Unix platform).
    let bind_addr = bind.unwrap_or_else(|| {
        barracuda_core::ipc::IpcServer::default_tcp_port().map_or_else(
            || barracuda_core::ipc::transport::resolve_bind_address(None),
            |p| {
                format!(
                    "{}:{p}",
                    barracuda_core::ipc::transport::resolve_bind_host()
                )
            },
        )
    });
    if let Some((listener, local_addr)) =
        barracuda_core::ipc::IpcServer::try_bind_tcp(&bind_addr).await
    {
        let effective_addr = local_addr.to_string();
        discovery_file::write_discovery_file(Some(&effective_addr), tarpc_bind.as_deref(), None);
        #[cfg(unix)]
        barracuda_core::discovery::register_with_discovery(&format!("tcp://{effective_addr}"))
            .await;

        let announce_addr = format!("tcp://{effective_addr}");
        discovery_file::spawn_neural_announce(announce_addr, ANNOUNCE_DELAY_MS);

        server.serve_tcp_listener(listener).await.map_err(|e| {
            barracuda_core::error::BarracudaCoreError::lifecycle(format!(
                "TCP server error on {effective_addr}: {e}"
            ))
        })?;
        discovery_file::remove_discovery_file(None);
    } else {
        return Err(barracuda_core::error::BarracudaCoreError::lifecycle(
            format!(
                "TCP bind failed on {bind_addr}: address already in use. \
             If another primal occupies this port, use a different \
             --port/BARRACUDA_IPC_PORT or run in UDS mode (default on Unix)."
            ),
        ));
    }
    Ok(())
}

/// Run the server in service mode (systemd/init).
///
/// Per genomeBin: Unix socket default, PID file, `NOTIFY_SOCKET`, no banner.
async fn run_service_mode() -> Result<(), barracuda_core::error::BarracudaCoreError> {
    barracuda_core::ipc::transport::validate_insecure_guard()?;
    let _pid_guard = discovery_file::write_pid_file();

    let mut primal = BarraCudaPrimal::new();
    primal.start().await.map_err(|e| {
        barracuda_core::error::BarracudaCoreError::lifecycle(format!("Failed to start: {e}"))
    })?;
    let primal = Arc::new(primal);

    let server = barracuda_core::ipc::IpcServer::new(Arc::clone(&primal));

    #[cfg(unix)]
    {
        let sock_path = barracuda_core::ipc::IpcServer::default_socket_path();
        discovery_file::write_discovery_file(None, None, Some(&sock_path));
        barracuda_core::ipc::IpcServer::create_legacy_symlink(&sock_path);
        barracuda_core::discovery::register_with_discovery(&format!(
            "unix://{}",
            sock_path.display()
        ))
        .await;

        let announce_socket = sock_path.to_string_lossy().to_string();
        discovery_file::spawn_neural_announce(announce_socket, ANNOUNCE_DELAY_MS);

        let on_ready = || discovery_file::notify_systemd_ready();
        server.serve_unix(&sock_path, Some(on_ready)).await?;
        barracuda_core::ipc::IpcServer::remove_legacy_symlink(&sock_path);
        discovery_file::remove_discovery_file(Some(&sock_path));
    }

    #[cfg(not(unix))]
    {
        let bind_addr = barracuda_core::ipc::transport::resolve_bind_address(None);
        discovery_file::write_discovery_file(Some(&bind_addr), None, None);
        server.serve_tcp(&bind_addr).await?;
        discovery_file::remove_discovery_file(None);
    }

    Ok(())
}

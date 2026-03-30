// SPDX-License-Identifier: AGPL-3.0-or-later
//! barraCuda `UniBin` — single binary, multiple modes.
//!
//! Per wateringHole `UNIBIN_ARCHITECTURE_STANDARD.md` and
//! `GENOMEBIN_ARCHITECTURE_STANDARD.md`:
//! - One binary named after the primal
//! - Subcommands: `server`, `service`, `doctor`, `validate`, `version`

use barracuda_core::BarraCudaPrimal;
use barracuda_core::health::PrimalHealth;
use barracuda_core::lifecycle::PrimalLifecycle;
use clap::{Parser, Subcommand};
use std::sync::Arc;

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
        /// `{binary} server --port {port}`. Binds to `127.0.0.1:{port}`.
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
        /// `$XDG_RUNTIME_DIR/biomeos/barracuda.sock`.
        #[cfg(unix)]
        #[arg(long, num_args = 0..=1, default_missing_value = "__default__")]
        unix: Option<String>,

        /// Disable Unix socket transport (force TCP-only).
        #[cfg(unix)]
        #[arg(long)]
        no_unix: bool,
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
        /// JSON-RPC method name (e.g. `barracuda.device.list`).
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
        } => {
            let effective_bind = bind.or_else(|| port.map(|p| format!("127.0.0.1:{p}")));
            run_server(
                effective_bind,
                tarpc_bind,
                #[cfg(unix)]
                tarpc_unix,
                #[cfg(unix)]
                unix,
                #[cfg(unix)]
                no_unix,
            )
            .await?;
        }
        Commands::Service => run_service_mode().await?,
        Commands::Doctor => run_doctor().await?,
        Commands::Validate { extended } => run_validate(extended).await?,
        Commands::Client {
            method,
            params,
            addr,
        } => run_client(&method, &params, addr.as_deref()).await?,
        Commands::Version => print_version(),
    }

    Ok(())
}

async fn run_server(
    bind: Option<String>,
    tarpc_bind: Option<String>,
    #[cfg(unix)] tarpc_unix: Option<String>,
    #[cfg(unix)] unix: Option<String>,
    #[cfg(unix)] no_unix: bool,
) -> Result<(), barracuda_core::error::BarracudaCoreError> {
    let mut primal = BarraCudaPrimal::new();
    primal.start().await.map_err(|e| {
        barracuda_core::error::BarracudaCoreError::lifecycle(format!("Failed to start: {e}"))
    })?;
    let primal = Arc::new(primal);

    let server = barracuda_core::ipc::IpcServer::new(Arc::clone(&primal));

    if let Some(ref tarpc_addr) = tarpc_bind {
        let tarpc_server = barracuda_core::ipc::IpcServer::new(Arc::clone(&primal));
        let tarpc_addr = tarpc_addr.clone();
        tokio::spawn(async move {
            if let Err(e) = tarpc_server.serve_tarpc(&tarpc_addr).await {
                tracing::error!("tarpc TCP server error: {e}");
            }
        });
    }

    #[cfg(unix)]
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

            // Per plasmidBin convention: also bind TCP alongside UDS when a
            // port is available (--port, --bind, or BARRACUDA_PORT env var).
            let tcp_addr = bind.or_else(|| {
                barracuda_core::ipc::IpcServer::default_tcp_port().map(|p| format!("127.0.0.1:{p}"))
            });
            if let Some(tcp) = tcp_addr {
                let tcp_server = barracuda_core::ipc::IpcServer::new(Arc::clone(&primal));
                write_discovery_file(Some(&tcp), tarpc_bind.as_deref(), Some(&sock_path));
                tokio::spawn(async move {
                    if let Err(e) = tcp_server.serve_tcp(&tcp).await {
                        tracing::error!("TCP server error: {e}");
                    }
                });
            } else {
                write_discovery_file(None, tarpc_bind.as_deref(), Some(&sock_path));
            }

            server.serve_unix(&sock_path, None::<fn()>).await?;
            remove_discovery_file();
            return Ok(());
        }
    }

    let bind_addr = barracuda_core::ipc::transport::resolve_bind_address(bind.as_deref());
    write_discovery_file(Some(&bind_addr), tarpc_bind.as_deref(), None);
    server.serve_tcp(&bind_addr).await?;
    remove_discovery_file();
    Ok(())
}

async fn run_doctor() -> Result<(), barracuda_core::error::BarracudaCoreError> {
    let mut primal = BarraCudaPrimal::new();
    primal.start().await.map_err(|e| {
        barracuda_core::error::BarracudaCoreError::lifecycle(format!("Failed to start: {e}"))
    })?;

    let report = primal.health_check().await.map_err(|e| {
        barracuda_core::error::BarracudaCoreError::health(format!("Health check failed: {e}"))
    })?;

    println!("barraCuda Doctor");
    println!("================");
    println!("Name:    {}", report.name);
    println!("Version: {}", report.version);
    println!("Status:  {}", report.status);

    if let Some(dev) = primal.device() {
        let info = dev.adapter_info();
        println!("\nGPU Device:");
        println!("  Adapter:  {}", info.name);
        println!("  Type:     {:?}", info.device_type);
        println!("  Backend:  {:?}", info.backend);
        println!("  Driver:   {} {}", info.driver, info.driver_info);

        let limits = dev.device().limits();
        println!("\n  Limits:");
        println!(
            "    Max buffer size:       {} MiB",
            limits.max_buffer_size / (1024 * 1024)
        );
        println!(
            "    Max storage buffers:   {}",
            limits.max_storage_buffers_per_shader_stage
        );
        println!(
            "    Max workgroup size X:  {}",
            limits.max_compute_workgroup_size_x
        );
    } else {
        println!("\nNo GPU device available (CPU-only mode)");
    }
    Ok(())
}

async fn run_validate(extended: bool) -> Result<(), barracuda_core::error::BarracudaCoreError> {
    let mut primal = BarraCudaPrimal::new();
    primal.start().await.map_err(|e| {
        barracuda_core::error::BarracudaCoreError::lifecycle(format!("Failed to start: {e}"))
    })?;

    if primal.device().is_none() {
        println!("No GPU device available. Cannot run validation.");
        std::process::exit(1);
    }

    println!("barraCuda GPU Validation");
    println!("========================");
    println!(
        "Device: {}",
        primal
            .device()
            .map_or_else(|| "none".to_string(), |d| d.adapter_info().name.clone())
    );
    println!("Mode: {}", if extended { "extended" } else { "standard" });
    println!();
    println!("GPU device available and responsive.");
    println!("For full FHE/QCD validation, use: cargo run --bin validate_gpu --features gpu");
    Ok(())
}

async fn run_client(
    method: &str,
    params: &str,
    addr: Option<&str>,
) -> Result<(), barracuda_core::error::BarracudaCoreError> {
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt};

    let server_addr = resolve_client_addr(addr)?;

    let params: serde_json::Value = serde_json::from_str(params)?;

    let request = serde_json::json!({
        "jsonrpc": "2.0",
        "method": method,
        "params": params,
        "id": 1,
    });

    let mut line = serde_json::to_string(&request)?;
    line.push('\n');

    let response_line = if let Some(sock_path) = server_addr.strip_prefix("unix://") {
        #[cfg(unix)]
        {
            let stream = tokio::net::UnixStream::connect(sock_path)
                .await
                .map_err(|e| {
                    barracuda_core::error::BarracudaCoreError::ipc(format!(
                        "connect to unix://{sock_path}: {e}"
                    ))
                })?;
            let (reader, mut writer) = stream.into_split();
            writer.write_all(line.as_bytes()).await?;
            writer.shutdown().await?;
            tokio::io::BufReader::new(reader).lines().next_line().await
        }

        #[cfg(not(unix))]
        {
            let _ = sock_path;
            return Err(barracuda_core::error::BarracudaCoreError::ipc(
                "Unix sockets not supported on this platform",
            ));
        }
    } else {
        let stream = tokio::net::TcpStream::connect(&server_addr)
            .await
            .map_err(|e| {
                barracuda_core::error::BarracudaCoreError::ipc(format!(
                    "connect to {server_addr}: {e}"
                ))
            })?;
        let (reader, mut writer) = stream.into_split();
        writer.write_all(line.as_bytes()).await?;
        writer.shutdown().await?;
        tokio::io::BufReader::new(reader).lines().next_line().await
    };

    if let Ok(Some(resp)) = response_line {
        let response: serde_json::Value =
            serde_json::from_str(&resp).unwrap_or_else(|_| serde_json::json!({"raw": resp}));
        println!(
            "{}",
            serde_json::to_string_pretty(&response).unwrap_or(resp)
        );
    } else {
        eprintln!("No response from server");
        std::process::exit(1);
    }
    Ok(())
}

fn print_version() {
    println!("barraCuda {}", env!("CARGO_PKG_VERSION"));
    println!("License: AGPL-3.0-or-later");
    println!("MSRV:    {}", env!("CARGO_PKG_RUST_VERSION"));
    println!("Arch:    {}", std::env::consts::ARCH);
    println!("OS:      {}", std::env::consts::OS);
}

/// Write a discovery file so peer primals can find barraCuda.
///
/// Capabilities, provides, and methods are derived from the actual registered
/// IPC methods — no hardcoded values. Per wateringHole capability-based discovery.
///
/// Supports both TCP (`host:port`) and Unix socket (`unix:///path`) transports.
/// File path: `$XDG_RUNTIME_DIR/biomeos/barracuda-core.json`
fn write_discovery_file(
    tcp_addr: Option<&str>,
    tarpc_addr: Option<&str>,
    unix_path: Option<&std::path::Path>,
) {
    let Some(dir) = discovery_dir() else { return };
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
fn remove_discovery_file() {
    if let Some(dir) = discovery_dir() {
        let filename = format!("{}-core.json", barracuda_core::PRIMAL_NAMESPACE);
        let path = dir.join(&filename);
        if path.exists() {
            let _ = std::fs::remove_file(&path);
        }
    }
}

/// Run the server in service mode (systemd/init).
///
/// Per genomeBin: Unix socket default, PID file, `NOTIFY_SOCKET`, no banner.
async fn run_service_mode() -> Result<(), barracuda_core::error::BarracudaCoreError> {
    let _pid_guard = write_pid_file();

    let mut primal = BarraCudaPrimal::new();
    primal.start().await.map_err(|e| {
        barracuda_core::error::BarracudaCoreError::lifecycle(format!("Failed to start: {e}"))
    })?;
    let primal = Arc::new(primal);

    let server = barracuda_core::ipc::IpcServer::new(Arc::clone(&primal));

    #[cfg(unix)]
    {
        let sock_path = barracuda_core::ipc::IpcServer::default_socket_path();
        write_discovery_file(None, None, Some(&sock_path));
        let on_ready = || notify_systemd_ready();
        server.serve_unix(&sock_path, Some(on_ready)).await?;
        remove_discovery_file();
    }

    #[cfg(not(unix))]
    {
        let bind_addr = barracuda_core::ipc::transport::resolve_bind_address(None);
        write_discovery_file(Some(&bind_addr), None, None);
        server.serve_tcp(&bind_addr).await?;
        remove_discovery_file();
    }

    Ok(())
}

/// Write PID file at `$XDG_RUNTIME_DIR/{namespace}/{namespace}.pid` (best-effort).
/// Returns a guard that removes the file on drop.
fn write_pid_file() -> Option<PidFileGuard> {
    let ns = barracuda_core::PRIMAL_NAMESPACE;
    let dir = std::env::var("XDG_RUNTIME_DIR").ok()?;
    let dir = std::path::PathBuf::from(dir).join(ns);
    std::fs::create_dir_all(&dir).ok()?;
    let path = dir.join(format!("{ns}.pid"));
    std::fs::write(&path, std::process::id().to_string()).ok()?;
    tracing::debug!(path = %path.display(), "wrote PID file");
    Some(PidFileGuard { path })
}

struct PidFileGuard {
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
fn notify_systemd_ready() {
    let Ok(socket_path) = std::env::var("NOTIFY_SOCKET") else {
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
    // Abstract sockets (Linux): @name → \0name
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

/// The shared discovery directory for all ecoPrimals.
fn discovery_dir() -> Option<std::path::PathBuf> {
    std::env::var("XDG_RUNTIME_DIR")
        .ok()
        .map(|d| std::path::PathBuf::from(d).join("biomeos"))
}

/// Resolve the server address for the `client` subcommand.
///
/// Resolution chain (first match wins):
/// 1. Explicit `--addr` CLI argument
/// 2. `BARRACUDA_IPC_BIND` environment variable
/// 3. Discovery file at `$XDG_RUNTIME_DIR/biomeos/barracuda-core.json`
fn resolve_client_addr(
    explicit: Option<&str>,
) -> std::result::Result<String, barracuda_core::error::BarracudaCoreError> {
    if let Some(addr) = explicit {
        return Ok(addr.to_string());
    }

    if let Ok(addr) = std::env::var("BARRACUDA_IPC_BIND") {
        return Ok(addr);
    }

    if let Some(dir) = discovery_dir() {
        let filename = format!("{}-core.json", barracuda_core::PRIMAL_NAMESPACE);
        let path = dir.join(&filename);
        if let Ok(content) = std::fs::read_to_string(&path) {
            if let Ok(info) = serde_json::from_str::<serde_json::Value>(&content) {
                let transports = info.get("transports");

                // Prefer TCP for the client subcommand (TCP is simpler for CLI)
                if let Some(addr) = transports
                    .and_then(|t| t.get("jsonrpc"))
                    .and_then(|v| v.as_str())
                {
                    if !addr.starts_with("unix://") {
                        return Ok(addr.to_string());
                    }
                }

                // Fall back to Unix socket if available (strip unix:// prefix)
                if let Some(unix_addr) = transports
                    .and_then(|t| t.get("unix"))
                    .and_then(|v| v.as_str())
                {
                    return Ok(unix_addr.to_string());
                }

                // Accept unix:// in jsonrpc transport too
                if let Some(addr) = transports
                    .and_then(|t| t.get("jsonrpc"))
                    .and_then(|v| v.as_str())
                {
                    return Ok(addr.to_string());
                }
            }
        }
    }

    Err(barracuda_core::error::BarracudaCoreError::ipc(
        "cannot discover barraCuda server; use --addr or start `barracuda server`",
    ))
}

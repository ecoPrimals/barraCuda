// SPDX-License-Identifier: AGPL-3.0-or-later
//! Subcommand implementations for the barraCuda binary.
//!
//! Extracted from `main.rs` to keep the binary under 800 lines per deep debt
//! targets. Each subcommand is a self-contained async function.

use barracuda_core::BarraCudaPrimal;
use barracuda_core::env_keys;
use barracuda_core::health::PrimalHealth;
use barracuda_core::lifecycle::PrimalLifecycle;

/// `barracuda doctor` — health check and GPU diagnostics.
pub async fn run_doctor() -> Result<(), barracuda_core::error::BarracudaCoreError> {
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

/// `barracuda validate` — GPU validation suite.
pub async fn run_validate(extended: bool) -> Result<(), barracuda_core::error::BarracudaCoreError> {
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

/// `barracuda client` — invoke a JSON-RPC method against a running server.
pub async fn run_client(
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

/// `barracuda version` — print version and build info.
pub fn print_version() {
    println!("barraCuda {}", env!("CARGO_PKG_VERSION"));
    println!("License: AGPL-3.0-or-later");
    println!("MSRV:    {}", env!("CARGO_PKG_RUST_VERSION"));
    println!("Arch:    {}", std::env::consts::ARCH);
    println!("OS:      {}", std::env::consts::OS);
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

    if let Ok(addr) = std::env::var(env_keys::BARRACUDA_IPC_BIND) {
        return Ok(addr);
    }

    if let Some(dir) = super::discovery_file::discovery_dir() {
        let filename = format!("{}-core.json", barracuda_core::PRIMAL_NAMESPACE);
        let path = dir.join(&filename);
        if let Ok(content) = std::fs::read_to_string(&path) {
            if let Ok(info) = serde_json::from_str::<serde_json::Value>(&content) {
                let transports = info.get("transports");

                if let Some(addr) = transports
                    .and_then(|t| t.get("jsonrpc"))
                    .and_then(|v| v.as_str())
                {
                    if !addr.starts_with("unix://") {
                        return Ok(addr.to_string());
                    }
                }

                if let Some(unix_addr) = transports
                    .and_then(|t| t.get("unix"))
                    .and_then(|v| v.as_str())
                {
                    return Ok(unix_addr.to_string());
                }

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

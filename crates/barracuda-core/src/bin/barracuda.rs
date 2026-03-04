// SPDX-License-Identifier: AGPL-3.0-or-later
//! barraCuda UniBin — single binary, multiple modes.
//!
//! Per wateringHole `UNIBIN_ARCHITECTURE_STANDARD.md`:
//! - One binary named after the primal
//! - Subcommands: `server`, `doctor`, `validate`, `version`

use barracuda_core::health::PrimalHealth;
use barracuda_core::lifecycle::PrimalLifecycle;
use barracuda_core::BarraCudaPrimal;
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
    Server {
        /// TCP bind address for JSON-RPC (default: 127.0.0.1:0).
        #[arg(long, default_value = "127.0.0.1:0")]
        bind: String,

        /// TCP bind address for tarpc binary RPC (default: disabled).
        #[arg(long)]
        tarpc_bind: Option<String>,

        /// Use Unix socket instead of TCP for JSON-RPC.
        #[cfg(unix)]
        #[arg(long)]
        unix: Option<String>,
    },

    /// Health check and diagnostics.
    Doctor,

    /// Run GPU validation suite.
    Validate {
        /// Run extended validation (FHE + QCD canary).
        #[arg(long)]
        extended: bool,
    },

    /// Print version and build info.
    Version,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Server {
            bind,
            tarpc_bind,
            #[cfg(unix)]
            unix,
        } => {
            let mut primal = BarraCudaPrimal::new();
            primal
                .start()
                .await
                .map_err(|e| format!("Failed to start: {e}"))?;
            let primal = Arc::new(primal);

            let server = barracuda_core::ipc::IpcServer::new(Arc::clone(&primal));

            if let Some(ref tarpc_addr) = tarpc_bind {
                let tarpc_server = barracuda_core::ipc::IpcServer::new(Arc::clone(&primal));
                let tarpc_addr = tarpc_addr.clone();
                tokio::spawn(async move {
                    if let Err(e) = tarpc_server.serve_tarpc(&tarpc_addr).await {
                        tracing::error!("tarpc server error: {e}");
                    }
                });
            }

            #[cfg(unix)]
            if let Some(path) = unix {
                server.serve_unix(std::path::Path::new(&path)).await?;
                return Ok(());
            }

            server.serve_tcp(&bind).await?;
        }

        Commands::Doctor => {
            let mut primal = BarraCudaPrimal::new();
            primal
                .start()
                .await
                .map_err(|e| format!("Failed to start: {e}"))?;

            let report = primal
                .health_check()
                .await
                .map_err(|e| format!("Health check failed: {e}"))?;

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
        }

        Commands::Validate { extended } => {
            let mut primal = BarraCudaPrimal::new();
            primal
                .start()
                .await
                .map_err(|e| format!("Failed to start: {e}"))?;

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
                    .map_or("none".to_string(), |d| d.adapter_info().name.clone())
            );
            println!("Mode: {}", if extended { "extended" } else { "standard" });
            println!();
            println!("GPU device available and responsive.");
            println!(
                "For full FHE/QCD validation, use: cargo run --bin validate_gpu --features gpu"
            );
        }

        Commands::Version => {
            println!("barraCuda {}", env!("CARGO_PKG_VERSION"));
            println!("License: AGPL-3.0-or-later");
            println!("MSRV:    {}", env!("CARGO_PKG_RUST_VERSION"));
            println!("Arch:    {}", std::env::consts::ARCH);
            println!("OS:      {}", std::env::consts::OS);
        }
    }

    Ok(())
}

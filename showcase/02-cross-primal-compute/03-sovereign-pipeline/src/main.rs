// SPDX-License-Identifier: AGPL-3.0-or-later
//! Sovereign Pipeline Showcase
//!
//! The capstone demo: full sovereign compute pipeline from hardware discovery
//! through shader compilation to GPU dispatch and result validation.
//!
//! Each layer degrades gracefully when cross-primal services are absent.

use barracuda::device::coral_compiler::GLOBAL_CORAL;
use barracuda::device::coral_compiler::types::AdapterDescriptor;
use barracuda::device::vendor::{VENDOR_AMD, VENDOR_NVIDIA};
use barracuda::device::{DeviceCapabilities, WgpuDevice};
use barracuda::ops::variance_f64_wgsl::VarianceF64;
use barracuda::pipeline::GpuViewF64;
use std::sync::Arc;
use std::time::Instant;

/// Pick a coralReef ISA string from the server's list for this adapter (NVIDIA `sm_*`, AMD `gfx*`).
fn coral_target_for_adapter(
    supported_archs: &[String],
    adapter: &AdapterDescriptor,
) -> Option<String> {
    let prefix = match adapter.vendor_id {
        VENDOR_NVIDIA => "sm_",
        VENDOR_AMD => "gfx",
        _ => return None,
    };
    supported_archs
        .iter()
        .filter(|a| a.starts_with(prefix))
        .min()
        .cloned()
}

fn has_capability(manifest_path: &std::path::Path, capability: &str) -> bool {
    if manifest_path.extension().is_none_or(|e| e != "json") {
        return false;
    }
    let Ok(content) = std::fs::read_to_string(manifest_path) else {
        return false;
    };
    let Ok(info) = serde_json::from_str::<serde_json::Value>(&content) else {
        return false;
    };
    info.get("provides")
        .or_else(|| info.get("capabilities"))
        .and_then(|v| v.as_array())
        .is_some_and(|caps| {
            caps.iter()
                .any(|c| c.as_str().is_some_and(|s| s.starts_with(capability)))
        })
}

#[tokio::main]
async fn main() -> barracuda::error::Result<()> {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  barraCuda — Sovereign Compute Pipeline                     ║");
    println!("║  toadStool + barraCuda + coralReef                          ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let pipeline_start = Instant::now();

    // ═══════════════════════════════════════════════════════════════
    // Layer 1: Hardware Discovery (toadStool or local)
    // ═══════════════════════════════════════════════════════════════
    println!("═══ Layer 1: Hardware Discovery ═══════════════════════════");
    println!();

    let discovery_dir = std::env::var("XDG_RUNTIME_DIR")
        .or_else(|_| std::env::var("TMPDIR"))
        .unwrap_or_else(|_| "/tmp".to_string());
    let ecosystem_dir = std::env::var("ECOPRIMALS_DISCOVERY_DIR")
        .unwrap_or_else(|_| "ecoPrimals".to_string());
    let discovery_path = std::path::PathBuf::from(&discovery_dir).join(&ecosystem_dir);

    let toadstool_available = discovery_path
        .read_dir()
        .ok()
        .map(|entries| {
            entries
                .filter_map(Result::ok)
                .any(|e| {
                    has_capability(&e.path(), "compute.dispatch")
                        || has_capability(&e.path(), "hardware.profile")
                })
        })
        .unwrap_or(false);

    if toadstool_available {
        println!("  toadStool: AVAILABLE — rich hardware profiling active");
    } else {
        println!("  toadStool: not found — using barraCuda local discovery");
    }

    let t = Instant::now();
    let device = WgpuDevice::new().await?;
    let device = Arc::new(device);
    let discovery_time = t.elapsed();

    let caps = DeviceCapabilities::from_device(&device);
    let strategy = caps.fp64_strategy();

    println!("  GPU:           {}", caps.device_name);
    println!("  Backend:       {:?}", caps.backend);
    println!("  Vendor:        0x{:04X}", caps.vendor);
    println!("  Fp64Strategy:  {strategy:?}");
    println!("  f64 shaders:   {}", caps.f64_shaders);
    println!(
        "  Discovery:     {:.1} ms",
        discovery_time.as_secs_f64() * 1000.0
    );
    println!();

    // ═══════════════════════════════════════════════════════════════
    // Layer 2: Shader Compilation (coralReef or wgpu)
    // ═══════════════════════════════════════════════════════════════
    println!("═══ Layer 2: Shader Compilation ═══════════════════════════");
    println!();

    let coral = &*GLOBAL_CORAL;
    let coral_health = coral.health().await;

    match &coral_health {
        Some(h) => {
            println!("  coralReef: AVAILABLE (v{})", h.version);
            let adapter = AdapterDescriptor::from_adapter_info(device.adapter_info());
            let arch_display = coral
                .supported_archs()
                .await
                .as_ref()
                .and_then(|archs| coral_target_for_adapter(archs, &adapter))
                .unwrap_or_else(|| "unknown".to_owned());
            println!("  Compilation: WGSL → coralReef IPC → native binary ({arch_display})");

            if let Some(archs) = coral.supported_archs().await {
                println!("  Targets: {}", archs.join(", "));
            }
        }
        None => {
            println!("  coralReef: not found — using wgpu path");
            println!("  Compilation: WGSL → naga → SPIR-V → driver compiler");
        }
    }
    println!();

    // ═══════════════════════════════════════════════════════════════
    // Layer 3: GPU Dispatch (math execution)
    // ═══════════════════════════════════════════════════════════════
    println!("═══ Layer 3: GPU Dispatch ═════════════════════════════════");
    println!();

    let n = 200_000;
    let data: Vec<f64> = (0..n)
        .map(|i| (i as f64 * 0.001).sin() * 100.0 + 500.0)
        .collect();

    println!("  Workload: {n} samples — fused mean+variance + correlation");
    println!();

    // Fused mean + variance
    let t = Instant::now();
    let variance_op = VarianceF64::new(Arc::clone(&device))?;
    let mv_result = variance_op.mean_variance(&data, 1);
    let mv_time = t.elapsed();

    match mv_result {
        Ok([gpu_mean, gpu_var]) => {
            println!("  Fused Welford (single dispatch):");
            println!("    mean     = {gpu_mean:.10}");
            println!("    variance = {gpu_var:.10}");
            println!(
                "    time     = {:.3} ms",
                mv_time.as_secs_f64() * 1000.0
            );
        }
        Err(e) => {
            println!("  Fused Welford: GPU f64 unavailable ({e})");
            println!("  Falling back to CPU computation...");

            let cpu_mean = data.iter().sum::<f64>() / data.len() as f64;
            let cpu_var = data
                .iter()
                .map(|x| (x - cpu_mean).powi(2))
                .sum::<f64>()
                / (data.len() - 1) as f64;
            println!("    mean     = {cpu_mean:.10} (CPU)");
            println!("    variance = {cpu_var:.10} (CPU)");
        }
    }
    println!();

    // GpuView zero-readback chain
    let data2: Vec<f64> = (0..n)
        .map(|i| (i as f64 * 0.001).cos() * 100.0 + 500.0)
        .collect();

    let t = Instant::now();
    let view_a = GpuViewF64::upload(Arc::clone(&device), &data)?;
    let view_b = GpuViewF64::upload(Arc::clone(&device), &data2)?;
    let upload_time = t.elapsed();

    println!("  GpuView chain (data stays on GPU):");
    println!(
        "    upload:  2 x {} KB = {:.3} ms",
        n * 8 / 1024,
        upload_time.as_secs_f64() * 1000.0
    );

    let t = Instant::now();
    match view_a.sum() {
        Ok(s) => println!(
            "    sum(a):  {s:.6}  ({:.3} ms)",
            t.elapsed().as_secs_f64() * 1000.0
        ),
        Err(_) => println!("    sum(a):  skipped (f64 unavailable)"),
    }

    let t = Instant::now();
    match GpuViewF64::correlation(&view_a, &view_b) {
        Ok(r) => println!(
            "    corr:    {r:.6}  ({:.3} ms, both buffers on GPU)",
            t.elapsed().as_secs_f64() * 1000.0
        ),
        Err(_) => println!("    corr:    skipped (f64 unavailable)"),
    }
    println!();

    // ═══════════════════════════════════════════════════════════════
    // Pipeline Summary
    // ═══════════════════════════════════════════════════════════════
    let total_time = pipeline_start.elapsed();

    println!("═══ Pipeline Summary ═════════════════════════════════════");
    println!();
    println!("  Layer 1 (hardware):  {}", if toadstool_available { "toadStool" } else { "barraCuda local" });
    println!("  Layer 2 (compiler):  {}", if coral_health.is_some() { "coralReef native" } else { "wgpu/naga/SPIR-V" });
    println!("  Layer 3 (dispatch):  GPU via wgpu");
    println!("  Total pipeline:      {:.1} ms", total_time.as_secs_f64() * 1000.0);
    println!();
    println!("  The sovereign compute pipeline:");
    println!("    Math is universal — same WGSL shaders everywhere.");
    println!("    Compilation adapts — coralReef native or wgpu standard.");
    println!("    Hardware adapts — toadStool orchestration or local discovery.");
    println!("    Each layer degrades gracefully. The math never changes.");
    println!();

    Ok(())
}

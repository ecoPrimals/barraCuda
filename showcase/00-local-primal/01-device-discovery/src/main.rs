// SPDX-License-Identifier: AGPL-3.0-or-later
//! Device Discovery Showcase
//!
//! Demonstrates barraCuda's runtime GPU detection, capability scoring,
//! and precision routing — zero hardcoding, fully capability-based.

use barracuda::device::{DeviceCapabilities, WgpuDevice, WorkloadType};
use std::sync::Arc;

#[tokio::main]
async fn main() -> barracuda::error::Result<()> {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  barraCuda — Device Discovery Showcase                      ║");
    println!("║  Runtime GPU detection, zero hardcoding                     ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // --- Phase 1: Discover GPU ---
    println!("─── Phase 1: GPU Discovery ───────────────────────────────────");
    println!();
    let device = WgpuDevice::new().await?;
    let device = Arc::new(device);
    println!("  GPU discovered: {}", device.adapter_info().name);
    println!("  Backend:        {:?}", device.adapter_info().backend);
    println!("  Driver:         {}", device.adapter_info().driver);
    println!("  Device type:    {:?}", device.adapter_info().device_type);
    println!();

    // --- Phase 2: Capability Scoring ---
    println!("─── Phase 2: Capability Scoring ──────────────────────────────");
    println!();
    let caps = DeviceCapabilities::from_device(&device);
    println!("{caps}");

    // --- Phase 3: Workload-Aware Sizing ---
    println!("─── Phase 3: Optimal Workgroup Sizes ─────────────────────────");
    println!();
    let workloads = [
        ("Element-wise (add, mul, relu)", WorkloadType::ElementWise),
        ("Matrix multiplication", WorkloadType::MatMul),
        ("Reduction (sum, max, mean)", WorkloadType::Reduction),
        ("FHE operations", WorkloadType::FHE),
        ("Convolution", WorkloadType::Convolution),
    ];

    println!("  {:40} {:>6} {:>10} {:>14}", "Workload", "1D", "2D", "3D");
    println!("  {:40} {:>6} {:>10} {:>14}", "─".repeat(40), "──────", "──────────", "──────────────");
    for (name, workload) in workloads {
        let s1 = caps.optimal_workgroup_size(workload);
        let s2 = caps.optimal_workgroup_size_2d(workload);
        let s3 = caps.optimal_workgroup_size_3d(workload);
        println!(
            "  {:40} {:>6} {:>4}x{:<5} {:>4}x{:>2}x{:<4}",
            name, s1, s2.0, s2.1, s3.0, s3.1, s3.2
        );
    }
    println!();

    // --- Phase 4: Precision Routing ---
    println!("─── Phase 4: Precision Routing ───────────────────────────────");
    println!();
    let strategy = caps.fp64_strategy();
    let routing = caps.precision_routing();
    println!("  Capability-based precision:");
    println!("    Device:            {}", caps.device_name);
    println!("    Vendor:            0x{:04X}", caps.vendor);
    println!("    PrecisionRouting:  {routing:?}");
    println!("    Fp64Strategy:      {strategy:?}");
    println!();

    match strategy {
        barracuda::device::Fp64Strategy::Native => {
            println!("  This GPU has full native f64 (1:2 rate or better).");
            println!("  All shaders use native f64 — maximum precision.");
        }
        barracuda::device::Fp64Strategy::Hybrid => {
            println!("  This GPU has limited f64 (1:64 consumer rate).");
            println!("  Bulk math routes through DF64 (f32-pair, ~48-bit mantissa).");
            println!("  Reductions use native f64 for final accumulation.");
        }
        barracuda::device::Fp64Strategy::Concurrent => {
            println!("  DF64 + native f64 run concurrently for cross-validation.");
        }
        barracuda::device::Fp64Strategy::Sovereign => {
            println!("  Sovereign path: coralReef compiles WGSL to native binary.");
        }
    }
    println!();

    // --- Phase 5: Matrix Capacity ---
    println!("─── Phase 5: Matrix Capacity ─────────────────────────────────");
    println!();
    let sizes = [(1024, 1024, 1024), (4096, 4096, 4096), (8192, 8192, 8192)];
    for (m, n, k) in sizes {
        let supported = caps.supports_large_matmul(m, n, k);
        let memory_mb = (m * k + k * n + m * n) * 4 / (1024 * 1024);
        let status = if supported { "supported" } else { "too large " };
        println!("  {m:>5}x{n:<5} x {k:>5}  (~{memory_mb:>4} MB)  [{status}]");
    }
    println!();

    // --- Summary ---
    println!("─── Summary ────────────────────────────────────────────────");
    println!();
    println!("  All values discovered at runtime — zero hardcoding.");
    println!("  Vendor-specific optimizations selected by capability scoring.");
    println!("  Precision routing adapts to actual GPU f64 throughput.");
    println!();

    Ok(())
}

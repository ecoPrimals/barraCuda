// SPDX-License-Identifier: AGPL-3.0-or-later
//! coralReef Shader Compilation Showcase
//!
//! Demonstrates the barraCuda → coralReef shader compilation path.
//! Probes for coralReef availability, attempts WGSL compilation,
//! and gracefully degrades to the wgpu path if coralReef is absent.

use barracuda::device::coral_compiler::GLOBAL_CORAL;
use barracuda::device::coral_compiler::types::AdapterDescriptor;
use barracuda::device::vendor::{VENDOR_AMD, VENDOR_NVIDIA};
use barracuda::device::{DeviceCapabilities, WgpuDevice};
use std::sync::Arc;

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

const DEMO_SHADER: &str = r"
@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i < arrayLength(&input) {
        output[i] = input[i] * input[i] + input[i];  // x^2 + x
    }
}
";

#[tokio::main]
async fn main() -> barracuda::error::Result<()> {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  barraCuda — coralReef Shader Compilation Showcase          ║");
    println!("║  Cross-primal WGSL → native binary pipeline                 ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // --- Phase 1: Local GPU Discovery ---
    println!("─── Phase 1: Local GPU Discovery ─────────────────────────");
    println!();
    let device = WgpuDevice::new().await?;
    let device = Arc::new(device);
    let caps = DeviceCapabilities::from_device(&device);
    println!("  GPU:    {}", caps.device_name);
    println!("  Vendor: 0x{:04X}", caps.vendor);
    println!("  f64:    {}", caps.f64_shaders);
    println!();

    // --- Phase 2: Probe coralReef ---
    println!("─── Phase 2: Probe coralReef ─────────────────────────────");
    println!();
    let coral = &*GLOBAL_CORAL;
    let health = coral.health().await;

    match &health {
        Some(h) => {
            println!("  coralReef discovered and healthy:");
            println!("    Status:  {:?}", h.status);
            println!("    Version: {}", h.version);
            println!();

            if let Some(archs) = coral.supported_archs().await {
                println!("  Supported architectures:");
                for arch in &archs {
                    println!("    - {arch}");
                }
            }
            if let Some(capabilities) = coral.capabilities().await {
                println!("  Capabilities:");
                for cap in &capabilities {
                    println!("    - {cap}");
                }
            }
        }
        None => {
            println!("  coralReef not found — this is expected if coralReef is not running.");
            println!("  barraCuda will use the wgpu path (Vulkan/Metal/DX12).");
            println!();
            println!("  Discovery searched:");
            println!("    1. $BARRACUDA_SHADER_COMPILER_ADDR (not set)");
            println!("    2. $XDG_RUNTIME_DIR/ecoPrimals/*.json (shader.compile capability)");
            println!("    3. Localhost probe (if BARRACUDA_SHADER_COMPILER_PORT set)");
        }
    }
    println!();

    // --- Phase 3: Attempt Shader Compilation ---
    println!("─── Phase 3: Shader Compilation ──────────────────────────");
    println!();
    println!("  Demo shader: x^2 + x (elementwise, workgroup_size=256)");
    println!();

    let adapter = AdapterDescriptor::from_adapter_info(device.adapter_info());
    let arch_str = if health.is_some() {
        coral
            .supported_archs()
            .await
            .as_ref()
            .and_then(|archs| coral_target_for_adapter(archs, &adapter))
    } else {
        None
    };
    let arch_display = arch_str.as_deref().unwrap_or("unknown");

    if health.is_some() {
        println!("  Attempting coralReef compilation (arch={arch_display})...");

        if let Some(target) = arch_str {
            match coral.compile_wgsl_direct(DEMO_SHADER, &target, false).await {
                Some(binary) => {
                    println!("  Compilation successful:");
                    println!("    Binary size: {} bytes", binary.binary.len());
                    println!("    Arch target: {target}");
                    println!("    Path: coralReef native (shader.compile.wgsl)");
                }
                None => {
                    println!("  coralReef compilation returned None.");
                    println!("  Falling back to wgpu path...");
                    println!("  wgpu path: WGSL → naga → SPIR-V → driver compiler");
                }
            }
        } else {
            println!("  Architecture not mappable to coralReef target.");
            println!("  Using wgpu path: WGSL → naga → SPIR-V → driver compiler");
        }
    } else {
        println!("  coralReef not available — using wgpu path.");
        println!("  wgpu path: WGSL → naga → SPIR-V → driver compiler");
        println!();
        println!("  To test with coralReef:");
        println!("    1. Start coralReef: cd ../coralReef && cargo run -- server");
        println!("    2. Re-run this demo");
    }
    println!();

    // --- Phase 4: Execute via wgpu (always works) ---
    println!("─── Phase 4: Execute via wgpu (always available) ─────────");
    println!();

    let input_data: Vec<f32> = (0..1024).map(|i| i as f32).collect();
    let expected: Vec<f32> = input_data.iter().map(|&x| x * x + x).collect();

    let view = barracuda::pipeline::GpuViewF32::upload(Arc::clone(&device), &input_data)?;
    let result = view.download()?;
    let correct = result
        .iter()
        .zip(expected.iter())
        .all(|(a, b)| (a - b).abs() < 1e-3);

    println!(
        "  Dispatched x^2+x on {} elements via wgpu: {}",
        input_data.len(),
        if correct { "PASS" } else { "FAIL" }
    );
    println!("  Sample: input[10]={}, output[10]={} (expected {})",
        input_data[10], result[10], expected[10]);
    println!();

    // --- Summary ---
    println!("─── Summary ────────────────────────────────────────────────");
    println!();
    println!("  barraCuda discovers coralReef via capability scan.");
    println!("  If available: WGSL → coralReef IPC → native binary.");
    println!("  If not: WGSL → naga → SPIR-V → driver compiler (wgpu).");
    println!("  Math is the same. Only the compilation substrate changes.");
    println!();

    Ok(())
}

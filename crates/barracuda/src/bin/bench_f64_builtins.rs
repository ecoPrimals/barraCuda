// SPDX-License-Identifier: AGPL-3.0-or-later
//! F64 Built-in Capability Survey
//!
//! Probes every f64 WGSL built-in on the selected GPU and reports which are
//! natively supported vs need the `math_f64.wgsl` software fallback.
//!
//! # Background
//!
//! WGSL → naga → SPIR-V → Vulkan bypasses the proprietary FP64 "consumer lock"
//! that CUDA/OpenCL enforce. Both NVIDIA and AMD GPUs expose `VK_KHR_shader_float64`
//! via Vulkan even when CUDA would restrict it. This means we can access native
//! f64 built-ins directly — but individual transcendentals (exp, log, sin, cos)
//! may be broken on specific open-source driver stacks (NVK/NAK as of Mesa ≤25.2).
//!
//! By probing each function in isolation (so one crash does not hide others) we
//! build the exact capability matrix for each device. `ShaderTemplate` uses this
//! to emit native calls where safe, and the `math_f64.wgsl` library otherwise.
//!
//! # Usage
//!
//!   cargo run --release --bin `bench_f64_builtins`
//!   `BARRACUDA_GPU_ADAPTER=amd`  cargo run --release --bin `bench_f64_builtins`
//!   `BARRACUDA_GPU_ADAPTER=0`    cargo run --release --bin `bench_f64_builtins`
//!   `BARRACUDA_GPU_ADAPTER=1`    cargo run --release --bin `bench_f64_builtins`

use barracuda::device::WgpuDevice;
use barracuda::device::capabilities::DeviceCapabilities;
use barracuda::device::probe::probe_f64_builtins;
use std::sync::Arc;

#[tokio::main]
async fn main() {
    println!("=================================================================");
    println!(" BarraCuda F64 Built-in Capability Survey");
    println!(" WGSL → naga → SPIR-V → Vulkan path (bypasses proprietary lock)");
    println!("=================================================================\n");

    let adapters = WgpuDevice::enumerate_adapters().await;
    println!("Available GPU adapters:");
    for (i, info) in adapters.iter().enumerate() {
        println!(
            "  [{i}] {name}  backend={backend:?}  type={dtype:?}",
            name = info.name,
            backend = info.backend,
            dtype = info.device_type,
        );
    }
    println!();

    let selected = std::env::var("BARRACUDA_GPU_ADAPTER").unwrap_or_else(|_| "auto".to_string());
    println!("BARRACUDA_GPU_ADAPTER = {selected}\n");

    let device = match WgpuDevice::from_env().await {
        Ok(d) => Arc::new(d),
        Err(e) => {
            eprintln!("Failed to create device: {e}");
            std::process::exit(1);
        }
    };

    println!("Active device:  {}", device.name());
    println!("Device type:    {:?}", device.device_type());
    println!("SHADER_F64:     {}", device.has_f64_shaders());

    if !device.has_f64_shaders() {
        eprintln!(
            "\nERROR: SHADER_F64 not enabled on this device.\n\
             Cannot probe f64 built-ins — try a different adapter."
        );
        std::process::exit(1);
    }

    let caps = DeviceCapabilities::from_device(&device);
    println!("{caps}");

    println!("─────────────────────────────────────────────────────────────────");
    println!("Probing f64 built-ins (each in isolated shader to catch crashes)…");
    println!("─────────────────────────────────────────────────────────────────\n");

    let caps = probe_f64_builtins(&device).await;

    // Pretty-print the matrix
    let sym = |b: bool| if b { "NATIVE  " } else { "fallback" };
    println!("{:<12} Status  Notes", "Function");
    println!("{}", "─".repeat(60));
    println!("exp(f64)     {}  crashes NVK ≤Mesa 25.2", sym(caps.exp));
    println!("log(f64)     {}  crashes NVK ≤Mesa 25.2", sym(caps.log));
    println!("exp2(f64)    {}  2^x", sym(caps.exp2));
    println!("log2(f64)    {}  log base 2", sym(caps.log2));
    println!(
        "sin(f64)     {}  MUFU path on NVIDIA (may promote to f32)",
        sym(caps.sin)
    );
    println!("cos(f64)     {}  MUFU path on NVIDIA", sym(caps.cos));
    println!(
        "sqrt(f64)    {}  DSQRT hardware instruction",
        sym(caps.sqrt)
    );
    println!("fma(f64,…)   {}  DFMA hardware instruction", sym(caps.fma));
    println!(
        "abs/min/max  {}  bit-level operations",
        sym(caps.abs_min_max)
    );
    println!("{}", "─".repeat(60));
    println!();

    let total = caps.native_count();
    println!(
        "Native built-ins: {total}/9  {}",
        match total {
            9 => "— full native f64 ✓",
            7..=8 => "— nearly full, minor fallbacks",
            4..=6 => "— partial, fallback for transcendentals",
            _ => "— minimal, software lib required for most ops",
        }
    );
    println!();

    if caps.needs_exp_log_workaround() {
        println!("⚠  exp/log workaround REQUIRED for this device.");
        println!("   ShaderTemplate will patch exp() → exp_f64() and log() → log_f64().");
        println!("   NAK fix target: contribute exp/log to nak/src/from_nir.rs");
    } else {
        println!("✓  No exp/log workaround needed — native transcendentals work.");
        println!("   ShaderTemplate will use native WGSL calls directly.");
    }

    println!();
    println!("ShaderTemplate optimisation plan based on these results:");

    if caps.sqrt && caps.fma && caps.abs_min_max {
        println!("  → sqrt_f64(), fma(), abs/min/max: REMOVE software implementations");
        println!("    Replace with native sqrt(), fma(), abs(), min(), max() in all shaders");
    }
    if caps.exp && caps.log {
        println!("  → exp_f64(), log_f64(): REMOVE, replace with native exp(), log()");
    } else {
        println!("  → exp_f64(), log_f64(): KEEP as software fallback (native broken)");
    }
    if caps.sin && caps.cos {
        println!("  → sin_f64(), cos_f64(): REMOVE, replace with native sin(), cos()");
        println!("    Note: NVIDIA MUFU sin/cos may have reduced precision — validate");
    } else {
        println!("  → sin_f64(), cos_f64(): KEEP as software fallback");
    }
    if caps.exp2 && caps.log2 {
        println!("  → exp2_f64(), log2_f64(): REMOVE, replace with native exp2(), log2()");
    }

    println!();
    println!("Cross-GPU capability matrix (run on each device and compare):");
    println!("  RTX 3090  (SM86/Ampere, proprietary): run with BARRACUDA_GPU_ADAPTER=0");
    println!("  RX 6950 XT (RDNA2/NAVI21, RADV/ACO): run with BARRACUDA_GPU_ADAPTER=1");
    println!("  Titan V   (SM70/Volta, NVK/NAK):     run on hotSpring machine");
    println!("  RTX 4070  (SM89/Ada, proprietary):   run on hotSpring machine");
}

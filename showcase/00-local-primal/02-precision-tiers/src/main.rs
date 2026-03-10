// SPDX-License-Identifier: AGPL-3.0-only
//! Precision Tiers Showcase
//!
//! Demonstrates the 3-tier precision model (F32 / F64 / DF64) by computing
//! mean+variance on the same dataset at each tier, comparing error bounds
//! against a CPU f64 reference.

use barracuda::device::{DeviceCapabilities, GpuDriverProfile, WgpuDevice};
use barracuda::ops::variance_f64_wgsl::VarianceF64;
use barracuda::pipeline::GpuViewF32;
use std::sync::Arc;
use std::time::Instant;

fn cpu_reference(data: &[f64]) -> (f64, f64) {
    let n = data.len() as f64;
    let mean = data.iter().sum::<f64>() / n;
    let var = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    (mean, var)
}

fn f32_mean_variance(data: &[f32]) -> (f64, f64) {
    let n = data.len() as f64;
    let mean = data.iter().map(|x| f64::from(*x)).sum::<f64>() / n;
    let var = data
        .iter()
        .map(|x| (f64::from(*x) - mean).powi(2))
        .sum::<f64>()
        / (n - 1.0);
    (mean, var)
}

#[tokio::main]
async fn main() -> barracuda::error::Result<()> {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  barraCuda — Precision Tiers Showcase                       ║");
    println!("║  F32 vs F64 vs DF64: same math, different precision         ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let device = WgpuDevice::new().await?;
    let device = Arc::new(device);
    let caps = DeviceCapabilities::from_device(&device);
    let profile = GpuDriverProfile::from_device(&device);
    let strategy = profile.fp64_strategy();

    println!("  GPU:           {}", device.adapter_info().name);
    println!("  Fp64Strategy:  {strategy:?}");
    println!("  f64 shaders:   {}", caps.f64_shaders);
    println!();

    // Generate test data: values near 1e6 with small differences (stresses precision)
    let n = 100_000;
    let data_f64: Vec<f64> = (0..n)
        .map(|i| 1_000_000.0 + (i as f64) * 0.001)
        .collect();
    let data_f32: Vec<f32> = data_f64.iter().map(|&x| x as f32).collect();

    println!("─── Dataset ────────────────────────────────────────────────");
    println!("  N = {n} values near 1,000,000 with 0.001 spacing");
    println!("  This stresses precision: mean ~ 1e6, variance ~ small");
    println!();

    // CPU f64 reference (ground truth)
    let (ref_mean, ref_var) = cpu_reference(&data_f64);
    println!("─── CPU f64 Reference (ground truth) ─────────────────────");
    println!("  mean     = {ref_mean:.15}");
    println!("  variance = {ref_var:.15}");
    println!();

    // Tier 1: F32 (CPU, demonstrates precision loss)
    println!("─── Tier 1: F32 (23-bit mantissa) ────────────────────────");
    let t = Instant::now();
    let (f32_mean, f32_var) = f32_mean_variance(&data_f32);
    let f32_time = t.elapsed();
    let f32_mean_err = ((f32_mean - ref_mean) / ref_mean).abs();
    let f32_var_err = ((f32_var - ref_var) / ref_var).abs();
    println!("  mean     = {f32_mean:.15}");
    println!("  variance = {f32_var:.15}");
    println!("  mean relative error: {f32_mean_err:.2e}");
    println!("  var  relative error: {f32_var_err:.2e}");
    println!("  time: {:.3} ms", f32_time.as_secs_f64() * 1000.0);
    println!();

    // Tier 2: GPU F32 via GpuView
    println!("─── Tier 2: GPU F32 (via GpuViewF32 upload) ──────────────");
    let t = Instant::now();
    let view = GpuViewF32::upload(Arc::clone(&device), &data_f32)?;
    let gpu_f32_data = view.download()?;
    let (gpu_f32_mean, gpu_f32_var) = f32_mean_variance(&gpu_f32_data);
    let gpu_f32_time = t.elapsed();
    let gpu_f32_mean_err = ((gpu_f32_mean - ref_mean) / ref_mean).abs();
    let gpu_f32_var_err = ((gpu_f32_var - ref_var) / ref_var).abs();
    println!("  mean     = {gpu_f32_mean:.15}");
    println!("  variance = {gpu_f32_var:.15}");
    println!("  mean relative error: {gpu_f32_mean_err:.2e}");
    println!("  var  relative error: {gpu_f32_var_err:.2e}");
    println!("  time: {:.3} ms (includes upload + download)", gpu_f32_time.as_secs_f64() * 1000.0);
    println!();

    // Tier 3: GPU F64 via fused Welford (if available)
    println!("─── Tier 3: GPU F64 (fused Welford, 52-bit mantissa) ─────");
    let variance_op = VarianceF64::new(Arc::clone(&device))?;
    let t = Instant::now();
    match variance_op.mean_variance(&data_f64, 1) {
        Ok([gpu_mean, gpu_var]) => {
            let gpu_f64_time = t.elapsed();
            let f64_mean_err = ((gpu_mean - ref_mean) / ref_mean).abs();
            let f64_var_err = ((gpu_var - ref_var) / ref_var).abs();
            println!("  mean     = {gpu_mean:.15}");
            println!("  variance = {gpu_var:.15}");
            println!("  mean relative error: {f64_mean_err:.2e}");
            println!("  var  relative error: {f64_var_err:.2e}");
            println!("  time: {:.3} ms", gpu_f64_time.as_secs_f64() * 1000.0);
        }
        Err(e) => {
            println!("  Skipped: GPU f64 not available on this device");
            println!("  Reason: {e}");
        }
    }
    println!();

    // Summary table
    println!("─── Comparison ─────────────────────────────────────────────");
    println!();
    println!("  {:12} {:>18} {:>18}", "Tier", "Mean Rel Error", "Var Rel Error");
    println!("  {:12} {:>18} {:>18}", "────────────", "──────────────────", "──────────────────");
    println!("  {:12} {:>18.2e} {:>18.2e}", "F32 (23-bit)", f32_mean_err, f32_var_err);
    println!("  {:12} {:>18.2e} {:>18.2e}", "GPU F32", gpu_f32_mean_err, gpu_f32_var_err);
    println!("  {:12} {:>18}", "GPU F64", "see above (or N/A)");
    println!("  {:12} {:>18}", "CPU f64 ref", "0 (ground truth)");
    println!();
    println!("  F32 loses ~7 digits of precision near 1e6.");
    println!("  F64/DF64 preserves ~15 digits — critical for scientific computing.");
    println!();

    Ok(())
}

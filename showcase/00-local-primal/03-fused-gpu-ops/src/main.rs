// SPDX-License-Identifier: AGPL-3.0-or-later
//! Fused GPU Operations Showcase
//!
//! Demonstrates single-dispatch fused operations (Welford mean+variance,
//! 5-accumulator correlation) and GpuView zero-readback chains.

use barracuda::device::WgpuDevice;
use barracuda::ops::correlation_f64_wgsl::CorrelationF64;
use barracuda::ops::variance_f64_wgsl::VarianceF64;
use barracuda::pipeline::GpuViewF64;
use std::sync::Arc;
use std::time::Instant;

fn cpu_mean_variance(data: &[f64]) -> (f64, f64) {
    let n = data.len() as f64;
    let mean = data.iter().sum::<f64>() / n;
    let var = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1.0);
    (mean, var)
}

fn cpu_pearson(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len() as f64;
    let mx = x.iter().sum::<f64>() / n;
    let my = y.iter().sum::<f64>() / n;
    let cov: f64 = x.iter().zip(y).map(|(a, b)| (a - mx) * (b - my)).sum();
    let sx: f64 = x.iter().map(|a| (a - mx).powi(2)).sum::<f64>().sqrt();
    let sy: f64 = y.iter().map(|b| (b - my).powi(2)).sum::<f64>().sqrt();
    cov / (sx * sy)
}

#[tokio::main]
async fn main() -> barracuda::error::Result<()> {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  barraCuda — Fused GPU Operations Showcase                  ║");
    println!("║  Single-dispatch stats + GpuView zero-readback chains       ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    let device = WgpuDevice::new().await?;
    let device = Arc::new(device);
    println!("  GPU: {}", device.adapter_info().name);
    println!();

    let n = 500_000;
    let x: Vec<f64> = (0..n).map(|i| (i as f64 * 0.01).sin()).collect();
    let y: Vec<f64> = (0..n)
        .map(|i| (i as f64 * 0.01).sin() * 0.95 + (i as f64 * 0.03).cos() * 0.05)
        .collect();
    println!("  Dataset: {n} samples (sin wave with correlated noise)");
    println!();

    // --- Fused Mean + Variance (Welford) ---
    println!("─── Fused Mean + Variance (single Welford dispatch) ──────");
    println!();

    let t = Instant::now();
    let (cpu_mean, cpu_var) = cpu_mean_variance(&x);
    let cpu_time = t.elapsed();
    println!("  CPU reference:  mean={cpu_mean:.12}, var={cpu_var:.12}");
    println!("  CPU time:       {:.3} ms", cpu_time.as_secs_f64() * 1000.0);

    let variance_op = VarianceF64::new(Arc::clone(&device))?;
    let t = Instant::now();
    match variance_op.mean_variance(&x, 1) {
        Ok([gpu_mean, gpu_var]) => {
            let gpu_time = t.elapsed();
            println!(
                "  GPU fused:      mean={gpu_mean:.12}, var={gpu_var:.12}"
            );
            println!(
                "  GPU time:       {:.3} ms (single dispatch)",
                gpu_time.as_secs_f64() * 1000.0
            );
            let mean_err = ((gpu_mean - cpu_mean) / cpu_mean).abs();
            let var_err = ((gpu_var - cpu_var) / cpu_var).abs();
            println!("  Relative error: mean={mean_err:.2e}, var={var_err:.2e}");
        }
        Err(e) => println!("  GPU f64 unavailable: {e}"),
    }
    println!();

    // --- Fused Correlation (5-accumulator) ---
    println!("─── Fused Correlation (5-accumulator single dispatch) ─────");
    println!();

    let t = Instant::now();
    let cpu_r = cpu_pearson(&x, &y);
    let cpu_corr_time = t.elapsed();
    println!("  CPU Pearson r:  {cpu_r:.12}");
    println!("  CPU time:       {:.3} ms", cpu_corr_time.as_secs_f64() * 1000.0);

    let corr_op = CorrelationF64::new(Arc::clone(&device))?;
    let t = Instant::now();
    match corr_op.correlation_full(&x, &y) {
        Ok(result) => {
            let gpu_corr_time = t.elapsed();
            println!("  GPU fused:");
            println!("    Pearson r = {:.12}", result.pearson_r);
            println!("    mean_x    = {:.12}", result.mean_x);
            println!("    mean_y    = {:.12}", result.mean_y);
            println!("    var_x     = {:.12}", result.var_x);
            println!("    var_y     = {:.12}", result.var_y);
            println!(
                "  GPU time:       {:.3} ms (single dispatch, 5 accumulators)",
                gpu_corr_time.as_secs_f64() * 1000.0
            );
            let r_err = ((result.pearson_r - cpu_r) / cpu_r).abs();
            println!("  Relative error: r={r_err:.2e}");
        }
        Err(e) => println!("  GPU f64 unavailable: {e}"),
    }
    println!();

    // --- GpuView: Zero-Readback Chain ---
    println!("─── GpuView: Zero-Readback Chain ─────────────────────────");
    println!();
    println!("  Data uploaded once, stays on GPU for multiple operations:");
    println!();

    let t = Instant::now();
    let view_x = GpuViewF64::upload(Arc::clone(&device), &x)?;
    let view_y = GpuViewF64::upload(Arc::clone(&device), &y)?;
    let upload_time = t.elapsed();
    println!(
        "  1. Upload:     {n} x f64 x 2 = {} KB  ({:.3} ms)",
        n * 16 / 1024,
        upload_time.as_secs_f64() * 1000.0
    );

    let t = Instant::now();
    match view_x.mean_variance(1) {
        Ok([mean, var]) => {
            let mv_time = t.elapsed();
            println!(
                "  2. mean_var:   mean={mean:.8}, var={var:.8}  ({:.3} ms, no readback)",
                mv_time.as_secs_f64() * 1000.0
            );
        }
        Err(e) => println!("  2. mean_var:   unavailable ({e})"),
    }

    let t = Instant::now();
    match view_x.sum() {
        Ok(total) => {
            let sum_time = t.elapsed();
            println!(
                "  3. sum:        {total:.8}  ({:.3} ms, no readback)",
                sum_time.as_secs_f64() * 1000.0
            );
        }
        Err(e) => println!("  3. sum:        unavailable ({e})"),
    }

    let t = Instant::now();
    match GpuViewF64::correlation(&view_x, &view_y) {
        Ok(r) => {
            let corr_time = t.elapsed();
            println!(
                "  4. correlation: r={r:.8}  ({:.3} ms, both buffers already on GPU)",
                corr_time.as_secs_f64() * 1000.0
            );
        }
        Err(e) => println!("  4. correlation: unavailable ({e})"),
    }
    println!();

    println!("─── Summary ────────────────────────────────────────────────");
    println!();
    println!("  Fused ops:  1 dispatch instead of 2-5 naive dispatches");
    println!("  GpuView:    Data stays on GPU — zero intermediate readbacks");
    println!("  Pipeline:   upload -> stats -> correlate -> download");
    println!("  Result:     Minimal host-device traffic for statistical chains");
    println!();

    Ok(())
}

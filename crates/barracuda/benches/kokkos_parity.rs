// SPDX-License-Identifier: AGPL-3.0-or-later
//! Kokkos parity benchmarks — barraCuda vs `sarkas_gpu` baseline.
//!
//! GPU optimization strategies applied:
//! - Pipeline cache warmup (10 iterations discarded)
//! - Persistent GPU-resident buffers (no host↔device round-trip per iteration)
//! - Median + P95 reporting (avoids outlier skew from driver scheduling)
//! - Single device creation amortized across all benchmarks
//!
//! Run with: `cargo bench --bench kokkos_parity`

#![expect(
    clippy::unwrap_used,
    reason = "benchmarks use unwrap for concise GPU setup"
)]

use barracuda::device::WgpuDevice;
use barracuda::ops::variance_f64_wgsl::VarianceF64;
use std::hint::black_box;
use std::sync::Arc;
use std::time::Instant;
use wgpu::util::DeviceExt;

const WARMUP_ITERS: usize = 10;
const BENCH_ITERS: usize = 100;

fn sorted_durations(times: &mut [std::time::Duration]) -> (f64, f64, f64) {
    times.sort();
    let median_ms = times[times.len() / 2].as_secs_f64() * 1000.0;
    let p95_ms = times[times.len() * 95 / 100].as_secs_f64() * 1000.0;
    let min_ms = times[0].as_secs_f64() * 1000.0;
    (min_ms, median_ms, p95_ms)
}

fn bench_mean_variance_upload(device: &Arc<WgpuDevice>, n: usize) {
    let var = VarianceF64::new(Arc::clone(device)).unwrap();
    let data: Vec<f64> = (0..n).map(|i| (i as f64) / (n as f64)).collect();

    for _ in 0..WARMUP_ITERS {
        let _ = black_box(var.mean_variance(&data, 0).unwrap());
    }

    let mut times = Vec::with_capacity(BENCH_ITERS);
    for _ in 0..BENCH_ITERS {
        let start = Instant::now();
        let result = var.mean_variance(&data, 0).unwrap();
        let elapsed = start.elapsed();
        black_box(&result);
        times.push(elapsed);
    }

    let (min, median, p95) = sorted_durations(&mut times);
    println!(
        "mean_variance_f64 (upload path)  n={n:>8}  min={min:.3}ms  median={median:.3}ms  p95={p95:.3}ms"
    );
}

fn bench_mean_variance_resident(device: &Arc<WgpuDevice>, n: usize) {
    let var = VarianceF64::new(Arc::clone(device)).unwrap();
    let data: Vec<f64> = (0..n).map(|i| (i as f64) / (n as f64)).collect();

    let input_buf = device
        .device()
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("bench_input"),
            contents: bytemuck::cast_slice(&data),
            usage: wgpu::BufferUsages::STORAGE,
        });

    for _ in 0..WARMUP_ITERS {
        let _ = black_box(var.mean_variance_buffer(&input_buf, n, 0).unwrap());
    }

    let mut times = Vec::with_capacity(BENCH_ITERS);
    for _ in 0..BENCH_ITERS {
        let start = Instant::now();
        let result = var.mean_variance_buffer(&input_buf, n, 0).unwrap();
        let elapsed = start.elapsed();
        black_box(&result);
        times.push(elapsed);
    }

    let (min, median, p95) = sorted_durations(&mut times);
    println!(
        "mean_variance_f64 (GPU-resident)  n={n:>8}  min={min:.3}ms  median={median:.3}ms  p95={p95:.3}ms"
    );
}

fn main() {
    let device = pollster::block_on(WgpuDevice::new());
    match device {
        Ok(dev) => {
            let dev = Arc::new(dev);
            println!("barraCuda Kokkos Parity Benchmarks");
            println!("==================================");
            println!("GPU:    {}", dev.adapter_info().name);
            println!(
                "Driver: {} {}",
                dev.adapter_info().driver,
                dev.adapter_info().driver_info
            );
            println!("Warmup: {WARMUP_ITERS} iters, Measure: {BENCH_ITERS} iters");
            println!();

            for &n in &[1_000, 10_000, 100_000, 1_000_000] {
                bench_mean_variance_upload(&dev, n);
            }
            println!();
            for &n in &[1_000, 10_000, 100_000, 1_000_000] {
                bench_mean_variance_resident(&dev, n);
            }
        }
        Err(e) => {
            eprintln!("No GPU available for benchmarks: {e}");
            std::process::exit(1);
        }
    }
}

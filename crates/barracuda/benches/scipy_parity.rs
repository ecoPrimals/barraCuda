// SPDX-License-Identifier: AGPL-3.0-or-later
//! SciPy parity benchmarks — barraCuda scientific ops vs SciPy/NumPy baselines.
//!
//! Benchmarks operations that map to common SciPy/NumPy workflows:
//! - f64 reduction: sum, mean, variance (→ numpy.sum, numpy.mean, numpy.var)
//! - Pairwise distance: cdist (→ scipy.spatial.distance.cdist)
//! - Special functions: Bessel J0/J1, Beta (→ scipy.special)
//!
//! Reference SciPy numbers (NumPy 1.26, Intel Xeon, single-threaded):
//!   numpy.var(N=1M):     ~1.2 ms
//!   scipy.cdist(1K×1K):  ~50 ms (Euclidean, D=3)
//!   scipy.special.j0(N): ~0.8 ms (N=100K)
//!
//! Run with: `cargo bench --bench scipy_parity`

#![expect(
    clippy::unwrap_used,
    reason = "benchmarks use unwrap for concise GPU setup"
)]

use barracuda::device::WgpuDevice;
use barracuda::ops::sum_reduce_f64::SumReduceF64;
use barracuda::ops::variance_f64_wgsl::VarianceF64;
use std::hint::black_box;
use std::sync::Arc;
use std::time::Instant;

const WARMUP_ITERS: usize = 10;
const BENCH_ITERS: usize = 100;

fn sorted_durations(times: &mut [std::time::Duration]) -> (f64, f64, f64) {
    times.sort();
    let median_ms = times[times.len() / 2].as_secs_f64() * 1000.0;
    let p95_ms = times[times.len() * 95 / 100].as_secs_f64() * 1000.0;
    let min_ms = times[0].as_secs_f64() * 1000.0;
    (min_ms, median_ms, p95_ms)
}

// ── numpy.sum (f64 reduction) ────────────────────────────────────────────────

fn bench_sum_f64(device: &Arc<WgpuDevice>, n: usize) {
    let data: Vec<f64> = (0..n).map(|i| (i as f64) / (n as f64)).collect();

    for _ in 0..WARMUP_ITERS {
        let _ = black_box(SumReduceF64::sum(Arc::clone(device), &data).unwrap());
    }

    let mut times = Vec::with_capacity(BENCH_ITERS);
    for _ in 0..BENCH_ITERS {
        let start = Instant::now();
        let result = SumReduceF64::sum(Arc::clone(device), &data).unwrap();
        let elapsed = start.elapsed();
        black_box(&result);
        times.push(elapsed);
    }

    let (min, median, p95) = sorted_durations(&mut times);
    println!(
        "sum_f64 (upload)         n={n:>8}  min={min:.3}ms  median={median:.3}ms  p95={p95:.3}ms"
    );
}

// ── numpy.var (f64 Welford variance) ──────────────────────────────────────────

fn bench_variance_f64(device: &Arc<WgpuDevice>, n: usize) {
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
        "variance_f64 (Welford)   n={n:>8}  min={min:.3}ms  median={median:.3}ms  p95={p95:.3}ms"
    );
}

// ── scipy.spatial.distance.cdist (pairwise Euclidean) ─────────────────────────

fn bench_cdist(device: &Arc<WgpuDevice>, m: usize, dim: usize) {
    use barracuda::ops::cdist_wgsl::Cdist;
    use barracuda::ops::cdist_wgsl::DistanceMetric;
    use barracuda::tensor::Tensor;

    let a_data: Vec<f32> = (0..m * dim)
        .map(|i| ((i * 7 + 13) % 1000) as f32 / 1000.0)
        .collect();
    let b_data: Vec<f32> = (0..m * dim)
        .map(|i| ((i * 11 + 17) % 1000) as f32 / 1000.0)
        .collect();

    let a = Tensor::from_vec_on_sync(a_data.clone(), vec![m, dim], Arc::clone(device)).unwrap();
    let b = Tensor::from_vec_on_sync(b_data.clone(), vec![m, dim], Arc::clone(device)).unwrap();

    for _ in 0..WARMUP_ITERS {
        let a_t =
            Tensor::from_vec_on_sync(a_data.clone(), vec![m, dim], Arc::clone(device)).unwrap();
        let b_t =
            Tensor::from_vec_on_sync(b_data.clone(), vec![m, dim], Arc::clone(device)).unwrap();
        let _ = black_box(
            Cdist::new(a_t, b_t, DistanceMetric::Euclidean)
                .execute()
                .unwrap(),
        );
    }

    let mut times = Vec::with_capacity(BENCH_ITERS);
    for _ in 0..BENCH_ITERS {
        let a_t =
            Tensor::from_vec_on_sync(a_data.clone(), vec![m, dim], Arc::clone(device)).unwrap();
        let b_t =
            Tensor::from_vec_on_sync(b_data.clone(), vec![m, dim], Arc::clone(device)).unwrap();
        let start = Instant::now();
        let result = Cdist::new(a_t, b_t, DistanceMetric::Euclidean)
            .execute()
            .unwrap();
        let elapsed = start.elapsed();
        black_box(&result);
        times.push(elapsed);
    }
    drop((a, b));

    let (min, median, p95) = sorted_durations(&mut times);
    let pairs = (m * m) as f64;
    let pairs_per_sec = pairs / (median / 1000.0);
    println!(
        "cdist Euclidean (f32)    M={m:>5} D={dim}  min={min:.3}ms  median={median:.3}ms  \
         p95={p95:.3}ms  [{pairs_per_sec:.0} pairs/s]"
    );
}

fn main() {
    let device = barracuda::runtime::tokio_block_on(WgpuDevice::new());
    match device {
        Ok(dev) => {
            let dev = Arc::new(dev);
            println!("barraCuda SciPy Parity Benchmarks");
            println!("=================================");
            println!("GPU:    {}", dev.adapter_info().name);
            println!(
                "Driver: {} {}",
                dev.adapter_info().driver,
                dev.adapter_info().driver_info
            );
            println!("Warmup: {WARMUP_ITERS} iters, Measure: {BENCH_ITERS} iters");

            let has_f64 = dev.device().features().contains(wgpu::Features::SHADER_F64);

            if has_f64 {
                println!("\n── numpy.sum (f64 GPU reduction) ──");
                for &n in &[10_000, 100_000, 1_000_000] {
                    bench_sum_f64(&dev, n);
                }

                println!("\n── numpy.var (f64 fused Welford) ──");
                for &n in &[10_000, 100_000, 1_000_000] {
                    bench_variance_f64(&dev, n);
                }
            } else {
                println!("\n[SKIP] f64 reductions — GPU lacks SHADER_F64");
            }

            println!("\n── scipy.spatial.distance.cdist (Euclidean) ──");
            for &m in &[256, 512, 1_000] {
                bench_cdist(&dev, m, 3);
            }

            println!("\n── Reference: SciPy (NumPy 1.26, single-thread CPU) ──");
            println!("  numpy.var(1M):           ~1.2 ms");
            println!("  scipy.cdist(1K×1K, D=3): ~50 ms (Euclidean)");
        }
        Err(e) => {
            eprintln!("No GPU available for benchmarks: {e}");
            std::process::exit(1);
        }
    }
}

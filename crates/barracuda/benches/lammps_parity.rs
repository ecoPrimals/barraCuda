// SPDX-License-Identifier: AGPL-3.0-or-later
//! LAMMPS parity benchmarks — barraCuda MD force kernels vs LAMMPS-class timings.
//!
//! Benchmarks fundamental pair-force kernels at LAMMPS-comparable particle counts:
//! - Lennard-Jones (f64, uniform σ/ε)
//! - Yukawa (f64, screened Coulomb with PBC)
//!
//! GPU optimization strategies (same as kokkos_parity):
//! - Pipeline cache warmup (10 iterations discarded)
//! - Median + P95 reporting (avoids outlier skew from driver scheduling)
//! - Single device creation amortized across all benchmarks
//!
//! Reference LAMMPS numbers for a comparable LJ system (Argon, N=4000,
//! `pair_style lj/cut 2.5`, reduced units on a V100):
//!   ~2630 timesteps/s (LAMMPS GPU) vs ~4000 timesteps/s (Kokkos/CUDA)
//!
//! Run with: `cargo bench --bench lammps_parity`

#![expect(
    clippy::unwrap_used,
    reason = "benchmarks use unwrap for concise GPU setup"
)]

use barracuda::device::WgpuDevice;
use barracuda::ops::md::forces::{LennardJonesF64, YukawaForceF64};
use barracuda::tensor::Tensor;
use std::hint::black_box;
use std::sync::Arc;
use std::time::Instant;

const WARMUP_ITERS: usize = 10;
const BENCH_ITERS: usize = 50;

fn sorted_durations(times: &mut [std::time::Duration]) -> (f64, f64, f64) {
    times.sort();
    let median_ms = times[times.len() / 2].as_secs_f64() * 1000.0;
    let p95_ms = times[times.len() * 95 / 100].as_secs_f64() * 1000.0;
    let min_ms = times[0].as_secs_f64() * 1000.0;
    (min_ms, median_ms, p95_ms)
}

/// Generate a simple cubic lattice of N particles in a box.
fn cubic_lattice(n: usize, box_side: f64) -> Vec<f64> {
    let n_per_side = (n as f64).cbrt().ceil() as usize;
    let spacing = box_side / n_per_side as f64;
    let mut positions = Vec::with_capacity(n * 3);
    for ix in 0..n_per_side {
        for iy in 0..n_per_side {
            for iz in 0..n_per_side {
                if positions.len() / 3 >= n {
                    return positions;
                }
                positions.push((ix as f64).mul_add(spacing, spacing * 0.5));
                positions.push((iy as f64).mul_add(spacing, spacing * 0.5));
                positions.push((iz as f64).mul_add(spacing, spacing * 0.5));
            }
        }
    }
    positions
}

fn bench_lj_f64(device: &Arc<WgpuDevice>, n: usize) {
    let sigma = 1.0;
    let epsilon = 1.0;
    let cutoff = 2.5;
    let density = 0.8442; // reduced LJ density (Argon liquid)
    let box_side = (n as f64 / density).cbrt();
    let positions = cubic_lattice(n, box_side);

    for _ in 0..WARMUP_ITERS {
        let _ = black_box(
            LennardJonesF64::compute_uniform(
                Arc::clone(device),
                &positions,
                sigma,
                epsilon,
                cutoff,
            )
            .unwrap(),
        );
    }

    let mut times = Vec::with_capacity(BENCH_ITERS);
    for _ in 0..BENCH_ITERS {
        let start = Instant::now();
        let result = LennardJonesF64::compute_uniform(
            Arc::clone(device),
            &positions,
            sigma,
            epsilon,
            cutoff,
        )
        .unwrap();
        let elapsed = start.elapsed();
        black_box(&result);
        times.push(elapsed);
    }

    let (min, median, p95) = sorted_durations(&mut times);
    let force_evals_per_sec = (n as f64) / (median / 1000.0);
    println!(
        "LJ f64 (all-pairs)       n={n:>6}  min={min:.3}ms  median={median:.3}ms  \
         p95={p95:.3}ms  [{force_evals_per_sec:.0} particles/s]"
    );
}

fn bench_yukawa_f64(device: &Arc<WgpuDevice>, n: usize) {
    let kappa = 2.0;
    let prefactor = 1.0;
    let cutoff = 4.0;
    let box_side = (n as f64 / 0.5).cbrt(); // moderate OCP density
    let positions = cubic_lattice(n, box_side);

    let pos_tensor = Tensor::from_f64_data(&positions, vec![n, 3], Arc::clone(device)).unwrap();

    for _ in 0..WARMUP_ITERS {
        let yukawa =
            YukawaForceF64::new(pos_tensor.clone(), kappa, prefactor, cutoff, box_side, None)
                .unwrap();
        let _ = black_box(yukawa.execute().unwrap());
    }

    let mut times = Vec::with_capacity(BENCH_ITERS);
    for _ in 0..BENCH_ITERS {
        let yukawa =
            YukawaForceF64::new(pos_tensor.clone(), kappa, prefactor, cutoff, box_side, None)
                .unwrap();
        let start = Instant::now();
        let result = yukawa.execute().unwrap();
        let elapsed = start.elapsed();
        black_box(&result);
        times.push(elapsed);
    }

    let (min, median, p95) = sorted_durations(&mut times);
    let force_evals_per_sec = (n as f64) / (median / 1000.0);
    println!(
        "Yukawa f64 (all-pairs)   n={n:>6}  min={min:.3}ms  median={median:.3}ms  \
         p95={p95:.3}ms  [{force_evals_per_sec:.0} particles/s]"
    );
}

fn main() {
    let device = barracuda::runtime::tokio_block_on(WgpuDevice::new());
    match device {
        Ok(dev) => {
            let dev = Arc::new(dev);
            println!("barraCuda LAMMPS Parity Benchmarks");
            println!("==================================");
            println!("GPU:    {}", dev.adapter_info().name);
            println!(
                "Driver: {} {}",
                dev.adapter_info().driver,
                dev.adapter_info().driver_info
            );
            println!("Warmup: {WARMUP_ITERS} iters, Measure: {BENCH_ITERS} iters");

            if !dev.device().features().contains(wgpu::Features::SHADER_F64) {
                eprintln!("GPU does not support SHADER_F64 — f64 force benchmarks require f64");
                std::process::exit(1);
            }

            println!("\n── Lennard-Jones (LAMMPS pair_style lj/cut) ──");
            for &n in &[256, 1_000, 4_000] {
                bench_lj_f64(&dev, n);
            }

            println!("\n── Yukawa (LAMMPS pair_style yukawa) ──");
            for &n in &[256, 1_000, 4_000] {
                bench_yukawa_f64(&dev, n);
            }

            println!("\n── Reference: LAMMPS GPU (V100, Argon LJ, N=4000) ──");
            println!("  ~2630 timesteps/s (LAMMPS GPU package)");
            println!("  ~4000 timesteps/s (Kokkos/CUDA)");
        }
        Err(e) => {
            eprintln!("No GPU available for benchmarks: {e}");
            std::process::exit(1);
        }
    }
}

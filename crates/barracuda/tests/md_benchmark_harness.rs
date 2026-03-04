// SPDX-License-Identifier: AGPL-3.0-or-later

//! MD Benchmark Harness — Tier 1 timing for Yukawa OCP and PPPM.
//!
//! Records wall time, energy drift, and throughput for comparison against
//! LAMMPS/Kokkos per the cross-spring Kokkos validation notice.
//!
//! Run: `cargo test --test md_benchmark_harness -- --nocapture`

#![expect(clippy::unwrap_used, reason = "integration test")]

use barracuda::device::test_pool;
use barracuda::ops::md::forces::yukawa_celllist_f64::CellListParams;
use barracuda::ops::md::forces::{YukawaCellListF64, YukawaForceF64};
use barracuda::tensor::Tensor;
use std::time::Instant;

/// Generate a cubic FCC-like lattice of N particles in a box of side L.
fn cubic_lattice(n_side: usize, box_side: f64) -> Vec<f64> {
    let spacing = box_side / n_side as f64;
    let mut positions = Vec::with_capacity(n_side * n_side * n_side * 3);
    for ix in 0..n_side {
        for iy in 0..n_side {
            for iz in 0..n_side {
                positions.push((ix as f64 + 0.5) * spacing);
                positions.push((iy as f64 + 0.5) * spacing);
                positions.push((iz as f64 + 0.5) * spacing);
            }
        }
    }
    positions
}

/// Benchmark report for a single MD primitive.
#[derive(Debug)]
struct BenchmarkResult {
    name: String,
    n_particles: usize,
    wall_time_ms: f64,
    throughput_pairs_per_sec: f64,
}

impl std::fmt::Display for BenchmarkResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}: N={}, wall={:.2}ms, throughput={:.2e} pairs/s",
            self.name, self.n_particles, self.wall_time_ms, self.throughput_pairs_per_sec
        )
    }
}

#[tokio::test]
async fn bench_yukawa_allpairs_256() {
    let Some(device) = test_pool::get_test_device_if_f64_gpu_available().await else {
        println!("Skipping: No f64 GPU available");
        return;
    };

    let n_side = 6; // 216 particles
    let box_side = 10.0;
    let positions = cubic_lattice(n_side, box_side);
    let n = positions.len() / 3;

    let pos_tensor = Tensor::from_f64_data(&positions, vec![n, 3], device.clone()).unwrap();

    let kappa = 2.0;
    let prefactor = 1.0;
    let cutoff = box_side / 2.0;

    let start = Instant::now();
    let yukawa = YukawaForceF64::new(pos_tensor, kappa, prefactor, cutoff, box_side, None).unwrap();
    let (_forces, _pe) = yukawa.execute().unwrap();
    let elapsed = start.elapsed();

    let n_pairs = (n * (n - 1)) / 2;
    let result = BenchmarkResult {
        name: "Yukawa all-pairs f64".into(),
        n_particles: n,
        wall_time_ms: elapsed.as_secs_f64() * 1000.0,
        throughput_pairs_per_sec: n_pairs as f64 / elapsed.as_secs_f64(),
    };
    println!("[BENCH] {result}");
}

#[tokio::test]
async fn bench_yukawa_celllist_512() {
    let Some(device) = test_pool::get_test_device_if_f64_gpu_available().await else {
        println!("Skipping: No f64 GPU available");
        return;
    };

    let n_side = 8; // 512 particles
    let box_side = 12.0;
    let positions = cubic_lattice(n_side, box_side);
    let n = positions.len() / 3;

    let params = CellListParams {
        box_size: [box_side, box_side, box_side],
        n_cells: [4, 4, 4],
        cutoff: box_side / 3.0,
        kappa: 2.0,
        prefactor: 1.0,
        epsilon: 1e-10,
    };

    let op = YukawaCellListF64::new(device).unwrap();

    let start = Instant::now();
    let (_forces, _pe) = op.compute_forces(&positions, &params).unwrap();
    let elapsed = start.elapsed();

    let result = BenchmarkResult {
        name: "Yukawa cell-list f64".into(),
        n_particles: n,
        wall_time_ms: elapsed.as_secs_f64() * 1000.0,
        throughput_pairs_per_sec: (n * 27 * (n / 64)) as f64 / elapsed.as_secs_f64(),
    };
    println!("[BENCH] {result}");
}

#[tokio::test]
async fn bench_pppm_1000() {
    use barracuda::ops::md::electrostatics::{PppmAccuracy, PppmParams};

    let Some(device) = test_pool::get_test_device_if_f64_gpu_available().await else {
        println!("Skipping: No f64 GPU available");
        return;
    };

    let n = 1000;
    let box_side = 20.0;
    let params = PppmParams::auto(n, box_side, PppmAccuracy::Medium);

    let mut rng_seed = 42u64;
    let mut positions = Vec::with_capacity(n * 3);
    let mut charges = Vec::with_capacity(n);
    for _ in 0..n {
        for _ in 0..3 {
            barracuda::ops::lattice::constants::lcg_step(&mut rng_seed);
            positions.push((rng_seed as f64 / u64::MAX as f64) * box_side);
        }
        charges.push(if positions.len() % 6 < 3 { 1.0 } else { -1.0 });
    }

    let pppm =
        barracuda::ops::md::electrostatics::PppmGpu::from_device(&device, params.clone()).await;
    let Ok(pppm) = pppm else {
        println!("Skipping PPPM bench: GPU PPPM init failed");
        return;
    };

    let start = Instant::now();
    let result = pppm.compute(&positions, &charges).await;
    let elapsed = start.elapsed();

    match result {
        Ok((_forces, energy)) => {
            println!(
                "[BENCH] PPPM f64: N={n}, wall={:.2}ms, energy={:.6}",
                elapsed.as_secs_f64() * 1000.0,
                energy
            );
        }
        Err(e) => {
            println!("[BENCH] PPPM f64: N={n}, FAILED: {e}");
        }
    }
}

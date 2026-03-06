// SPDX-License-Identifier: AGPL-3.0-or-later
//! Workgroup-size diagnostic for batched eigensolve across GPU drivers
//!
//! Measures performance of the warp-packed Jacobi batched eigensolve.
//! Key insight from hotSpring GPU sovereignty analysis (Feb 18, 2026):
//!
//! - wg1  — @`workgroup_size(1,1,1)`: wastes 31/32 SIMD lanes on NVIDIA
//! - wp32 — @`workgroup_size(32,1,1)`: fills full warp with independent matrices
//!
//! Measured impact:
//!   NVK/NAK (Titan V):       152ms → 69ms  (2.2x speedup)
//!   NVIDIA proprietary:       neutral (scheduler already handles wg1 well)
//!
//! The remaining 149x NAK compiler gap (after warp-packing) decomposes into:
//!
//!   1. No SM70 instruction scheduling  (~3-4x)
//!   2. No dual-issue exploitation       (~2x)
//!   3. Limited loop unrolling           (~1.5-2x)
//!   4. Missing f64 FMA selection        (~1.3-1.5x)
//!   5. Generic shared-mem scheduling    (~1.5-2x)
//!
//! NAK is written in Rust — see DEBT.md W-003 for the contribution roadmap.
//!
//! Usage:
//!   cargo run --release --bin `bench_wgsize_nvk`
//!   `BARRACUDA_GPU_ADAPTER=titan` cargo run --release --bin `bench_wgsize_nvk`
//!   `BARRACUDA_GPU_ADAPTER=0`     cargo run --release --bin `bench_wgsize_nvk`

use barracuda::device::WgpuDevice;
use barracuda::device::capabilities::{DriverKind, GpuDriverProfile};
use barracuda::ops::linalg::BatchedEighGpu;
use std::f64::consts::PI;
use std::sync::Arc;
use std::time::Instant;

#[tokio::main]
async fn main() {
    println!("==========================================================");
    println!(" BarraCuda Workgroup-Size Eigensolve Diagnostic");
    println!(" hotSpring GPU Sovereignty Analysis — Feb 18, 2026");
    println!("==========================================================\n");

    // ── List all visible adapters ─────────────────────────────────────────
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

    // ── Create device via env-aware selector (honours BARRACUDA_GPU_ADAPTER) ──
    let device = match WgpuDevice::from_env().await {
        Ok(d) => Arc::new(d),
        Err(e) => {
            eprintln!("Failed to create device: {e}");
            std::process::exit(1);
        }
    };

    println!("Active device: {}", device.name());
    println!("Device type:   {:?}", device.device_type());
    println!("SHADER_F64:    {}", device.has_f64_shaders());

    if !device.has_f64_shaders() {
        eprintln!(
            "\nERROR: This device does not expose SHADER_F64 capability.\n\
             The f64 eigensolve shader cannot be compiled on this adapter.\n\
             Try a different adapter: BARRACUDA_GPU_ADAPTER=amd or BARRACUDA_GPU_ADAPTER=1\n\
             Full adapter list shown above."
        );
        std::process::exit(1);
    }

    // ── Print driver profile ──────────────────────────────────────────────
    let profile = GpuDriverProfile::from_device(&device);
    println!("{profile}");

    println!("Expected behavior:");
    match profile.driver {
        DriverKind::Nvk => {
            println!("  NVK/NAK: warp-packing gives ~2.2x speedup (batch=512, dim=30).");
            println!("  NAK wastes 31/32 SIMD lanes with wg1. wp32 fills the full warp.");
            println!("  Remaining gap (~9x vs proprietary) is NAK compiler deficiency.");
            println!("  Evolution: contribute SM70 latency tables + f64 FMA to Mesa NAK.");
        }
        DriverKind::NvidiaProprietary => {
            println!("  Proprietary PTXAS handles wg1 efficiently — warp-packing is neutral.");
            println!("  Results should be ~same as wg1 baseline.");
        }
        DriverKind::Radv => {
            println!("  RADV/ACO (RDNA2/3): ACO uses wave32 mode for compute by default.");
            println!("  wg_size=32 is empirically optimal (RX 6950 XT: 67ms vs 117ms for wg64).");
            println!("  AMD RX 6950 XT beats RTX 3090 on f64: 1:4 ratio vs 1:64 throttling.");
            println!("  ACO/RADV faster than proprietary for f64-heavy workloads on RDNA2.");
        }
        _ => {
            println!("  Unknown driver — results are informational.");
        }
    }
    println!();

    // ── Benchmark configurations ──────────────────────────────────────────
    // (n, batch, sweeps, label)
    let configs: &[(usize, usize, u32, &str)] = &[
        (20, 512, 200, ""),
        (30, 512, 200, ""),
        (12, 512, 200, "HFB"),        // hotSpring HFB n=12, batch=40 analogue
        (30, 512, 5, "dispatch-dom"), // dispatch-dominated: shows overhead floor
    ];

    const RUNS: u32 = 5;

    println!("──────────────────────────────────────────────────────────");
    println!(
        "{:<14} {:>6} {:>8} {:>12}   Notes",
        "Config", "Batch", "Sweeps", "Time(ms)"
    );
    println!("──────────────────────────────────────────────────────────");

    for &(n, batch, sweeps, label) in configs {
        let data = random_spd_batch(n, batch);

        // Warm-up: trigger pipeline compilation
        let _ =
            BatchedEighGpu::execute_single_dispatch(device.clone(), &data, n, batch, sweeps, 1e-10);

        // Timed runs
        let start = Instant::now();
        for _ in 0..RUNS {
            let _ = BatchedEighGpu::execute_single_dispatch(
                device.clone(),
                &data,
                n,
                batch,
                sweeps,
                1e-10,
            );
        }
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0 / f64::from(RUNS);
        let config_str = format!("n={n} s={sweeps}");

        println!("{config_str:<14} {batch:>6} {sweeps:>8} {elapsed_ms:>12.2}   {label}");
    }

    println!("──────────────────────────────────────────────────────────");
    println!();
    println!("All runs use warp-packed dispatch (wp32 = @workgroup_size(32,1,1)).");
    println!("Each thread owns one independent matrix — no barriers, no shared memory.");
    println!("Dispatch: batch.div_ceil(32) workgroups.");
    println!();
    println!("To reproduce hotSpring's 2.2x wg1→wp32 comparison:");
    println!("  1. Run this binary on an NVK device (Titan V via nouveau).");
    println!("  2. Temporarily change @workgroup_size(32,1,1) to @workgroup_size(1,1,1)");
    println!("     in crates/barracuda/src/shaders/linalg/batched_eigh_single_dispatch_f64.wgsl");
    println!("  3. Compare — expect ~2.2x slower on NVK, neutral on proprietary.");
    println!();
    println!("NAK contribution roadmap (DEBT.md W-003):");
    println!("  Phase 1: SM70 latency tables in nak/src/calc_instr_deps.rs");
    println!("  Phase 2: f64 FMA selection in nak/src/from_nir.rs");
    println!("  Phase 3: Loop unrolling for bounded nested loops (Jacobi pattern)");
    println!("  Phase 4: Dual-issue exploitation for Volta (SM70)");
    println!("  Upstream: https://gitlab.freedesktop.org/mesa/mesa");
}

/// Generate a batch of random symmetric positive-definite matrices for benchmarking.
///
/// Construction: A = Qᵀ D Q where Q is built from Givens rotations and
/// D = diag(1, 2, …, n). All eigenvalues are exactly 1..n.
fn random_spd_batch(n: usize, batch: usize) -> Vec<f64> {
    let mut data = vec![0.0f64; batch * n * n];
    let mut seed: u64 = 0xDEAD_BEEF_C0DE_1337;

    let mut next = || -> f64 {
        seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        (seed >> 11) as f64 / (1u64 << 53) as f64
    };

    for b in 0..batch {
        let base = b * n * n;

        // Start with identity, apply random Givens rotations to build Q
        let mut q = vec![0.0f64; n * n];
        for i in 0..n {
            q[i * n + i] = 1.0;
        }
        for i in 0..n {
            for j in (i + 1)..n {
                let theta = next() * PI * 0.5;
                let (c, s) = (theta.cos(), theta.sin());
                for k in 0..n {
                    let qki = q[k * n + i];
                    let qkj = q[k * n + j];
                    q[k * n + i] = c * qki - s * qkj;
                    q[k * n + j] = s * qki + c * qkj;
                }
            }
        }

        // A = Qᵀ D Q  where D[i,i] = i+1
        let mut a = vec![0.0f64; n * n];
        for i in 0..n {
            let di = (i + 1) as f64;
            for row in 0..n {
                for col in 0..n {
                    // A[row,col] += Q[i,row] * di * Q[i,col]
                    a[row * n + col] += q[i * n + row] * di * q[i * n + col];
                }
            }
        }

        data[base..base + n * n].copy_from_slice(&a);
    }

    data
}

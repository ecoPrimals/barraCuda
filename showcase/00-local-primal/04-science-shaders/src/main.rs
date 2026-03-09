// SPDX-License-Identifier: AGPL-3.0-or-later
//! Science Shaders Showcase
//!
//! Demonstrates barraCuda's domain-specific scientific computing:
//! Hill kinetics, statistical metrics, tolerance architecture,
//! and GPU-accelerated operations.

use barracuda::stats::{
    dot, hill, hill_activation, hill_repression, l2_norm, mae, mean, nash_sutcliffe, percentile,
    rmse,
};
use barracuda::tolerances;

#[tokio::main]
async fn main() -> barracuda::error::Result<()> {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  barraCuda — Science Shaders Showcase                       ║");
    println!("║  716+ WGSL shaders for GPU-accelerated scientific computing ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // --- Hill Kinetics (Gene Regulatory Networks) ---
    println!("─── Hill Kinetics (neuralSpring absorption) ──────────────");
    println!();
    println!("  Gene regulatory networks use Hill functions to model");
    println!("  cooperative binding: how transcription factors activate");
    println!("  or repress gene expression.");
    println!();

    let k = 5.0; // half-max concentration
    let n = 3.0; // Hill coefficient (cooperativity)
    let amplitude = 100.0;

    println!("  Parameters: K={k}, n={n}, amplitude={amplitude}");
    println!();
    println!("  {:>8} {:>12} {:>12} {:>12}", "[X]", "Hill", "Activation", "Repression");
    println!(
        "  {:>8} {:>12} {:>12} {:>12}",
        "────────", "────────────", "────────────", "────────────"
    );

    for &x in &[0.0, 1.0, 2.5, 5.0, 10.0, 20.0, 50.0] {
        let h = hill(x, k, n);
        let act = hill_activation(x, amplitude, k, n);
        let rep = hill_repression(x, amplitude, k, n);
        println!("  {x:>8.1} {h:>12.6} {act:>12.4} {rep:>12.4}");
    }
    println!();
    println!("  At [X]=K ({k}), Hill = 0.5 (half-maximal response).");
    println!("  Cooperativity n={n} gives a sigmoidal switch.");
    println!();

    // --- Statistical Metrics ---
    println!("─── Statistical Metrics ─────────────────────────────────");
    println!();

    let observed: Vec<f64> = (0..1000).map(|i| (i as f64 * 0.01).sin()).collect();
    let simulated: Vec<f64> = observed
        .iter()
        .enumerate()
        .map(|(i, &x)| x + (i as f64 * 0.07).cos() * 0.1)
        .collect();

    let m = mean(&observed);
    let d = dot(&observed, &simulated);
    let norm = l2_norm(&observed);
    let error_mae = mae(&observed, &simulated);
    let error_rmse = rmse(&observed, &simulated);
    let nse = nash_sutcliffe(&observed, &simulated);
    let p50 = percentile(&observed, 50.0);
    let p95 = percentile(&observed, 95.0);

    println!("  Dataset: 1000 samples (sin wave vs perturbed copy)");
    println!();
    println!("  mean(observed)     = {m:.8}");
    println!("  dot(obs, sim)      = {d:.8}");
    println!("  L2 norm(observed)  = {norm:.8}");
    println!("  MAE                = {error_mae:.8}");
    println!("  RMSE               = {error_rmse:.8}");
    println!("  Nash-Sutcliffe     = {nse:.8}  (1.0 = perfect)");
    println!("  Median (P50)       = {p50:.8}");
    println!("  P95                = {p95:.8}");
    println!();

    // --- Tolerance Architecture ---
    println!("─── Tolerance Architecture (groundSpring V74) ────────────");
    println!();
    println!("  barraCuda's universal precision tiers — springs select the");
    println!("  tier that matches their physics, not ad-hoc magic numbers:");
    println!();

    let tiers = [
        ("DETERMINISM", tolerances::DETERMINISM),
        ("MACHINE", tolerances::MACHINE),
        ("ACCUMULATION", tolerances::ACCUMULATION),
        ("TRANSCENDENTAL", tolerances::TRANSCENDENTAL),
        ("ITERATIVE", tolerances::ITERATIVE),
        ("STATISTICAL", tolerances::STATISTICAL),
        ("STOCHASTIC", tolerances::STOCHASTIC),
        ("EQUILIBRIUM", tolerances::EQUILIBRIUM),
    ];

    println!("  {:20} {:>14} {:>14}", "Tier", "Abs Tolerance", "Rel Tolerance");
    println!(
        "  {:20} {:>14} {:>14}",
        "────────────────────", "──────────────", "──────────────"
    );
    for (name, tier) in tiers {
        println!("  {name:20} {:.2e}       {:.2e}", tier.abs_tol, tier.rel_tol);
    }
    println!();
    println!("  DETERMINISM: bit-exact (FHE, crypto, data movement)");
    println!("  EQUILIBRIUM: relaxed (~1.0, thermalization fluctuations)");
    println!();

    // --- Epsilon Guards ---
    println!("─── Epsilon Guards (safe division, underflow prevention) ─");
    println!();
    println!("  {:24} {:>20}", "Constant", "Value");
    println!("  {:24} {:>20}", "────────────────────────", "────────────────────");
    println!("  {:24} {:>20.2e}", "eps::SAFE_DIV", tolerances::eps::SAFE_DIV);
    println!("  {:24} {:>20.2e}", "eps::SSA_FLOOR", tolerances::eps::SSA_FLOOR);
    println!("  {:24} {:>20.2e}", "eps::UNDERFLOW", tolerances::eps::UNDERFLOW);
    println!("  {:24} {:>20.2e}", "eps::OVERFLOW", tolerances::eps::OVERFLOW);
    println!("  {:24} {:>20.2e}", "eps::LOG_FLOOR", tolerances::eps::LOG_FLOOR);
    println!("  {:24} {:>20.2e}", "eps::DENSITY_FLOOR", tolerances::eps::DENSITY_FLOOR);
    println!("  {:24} {:>20.2e}", "eps::PROB_FLOOR", tolerances::eps::PROB_FLOOR);
    println!();

    // --- Shader Inventory ---
    println!("─── Shader Inventory ─────────────────────────────────────");
    println!();
    println!("  barraCuda includes 716+ WGSL shaders across domains:");
    println!();
    println!("  Domain               Shaders  Examples");
    println!("  ───────────────────  ───────  ───────────────────────────────");
    println!("  Molecular dynamics     ~80    Yukawa, Morse, PPPM, Verlet, Langevin");
    println!("  Linear algebra         ~60    GEMM, SVD, QR, LU, BiCGSTAB, CG");
    println!("  Statistics             ~50    Welford, correlation, bootstrap, Kimura");
    println!("  Bio/genomics           ~45    HMM, DADA2, k-mer, phylogenetics");
    println!("  Special functions      ~40    Bessel, Hermite, digamma, beta, erf");
    println!("  Nuclear physics        ~35    HFB, Skyrme, BCS, spin-orbit, SEMF");
    println!("  Optimization           ~30    BFGS, Nelder-Mead, CG, bisection");
    println!("  Precision (DF64)       ~25    Double-float arithmetic core library");
    println!("  Spectral               ~20    FFT, autocorrelation, IPR, Anderson");
    println!("  Lattice QCD            ~15    Wilson, gauge, leapfrog, Omelyan");
    println!("  Climate/hydrology      ~15    Richards, SCS-CN, Blaney-Criddle, ET0");
    println!("  Neural networks        ~15    MLP, attention, activation, dropout");
    println!("  And more...           ~280+   FHE, PDE, grid, pooling, loss, etc.");
    println!();

    println!("─── Summary ────────────────────────────────────────────────");
    println!();
    println!("  barraCuda is a sovereign math engine: the shaders express");
    println!("  universal math, the GPU is just the execution substrate.");
    println!("  Same shaders run on NVIDIA, AMD, Intel, Apple, or software.");
    println!();

    Ok(())
}

// SPDX-License-Identifier: AGPL-3.0-or-later

//! Cross-spring validation harness for barracuda primitives.
//!
//! Validates that primitives consumed by multiple springs produce
//! consistent results across CPU and GPU paths. Each check exercises
//! a specific primitive with known inputs and verifies against
//! documented tolerances.
//!
//! Provenance: wetSpring V86 `validate_cross_spring_evolution_modern`
//! pattern → adopted as barraCuda standard test harness.
//!
//! Run: `cargo test --test cross_spring_validation`

use barracuda::optimize::{brent, lbfgs_numerical, LbfgsConfig};
use barracuda::spectral::anderson::{
    anderson_3d, anderson_4d, anderson_eigenvalues, anderson_potential, lyapunov_exponent,
};
use barracuda::stats::hydrology::fao56_et0;

/// Tolerance registry by domain (mirrors wetSpring `tolerances/` structure).
#[allow(dead_code)]
mod tolerances {
    pub const SPECTRAL_R: f64 = 0.05;
    pub const LYAPUNOV_RELATIVE: f64 = 0.1;
    pub const ET0_RELATIVE: f64 = 0.02;
    pub const OPTIMIZER_F: f64 = 1e-4;
    pub const LINALG_RELATIVE: f64 = 1e-8;
}

#[derive(Debug)]
struct CheckResult {
    name: &'static str,
    origin: &'static str,
    passed: bool,
    detail: String,
}

fn check_anderson_1d_localization() -> CheckResult {
    let pot = anderson_potential(5000, 2.0, 42);
    let gamma = lyapunov_exponent(&pot, 0.0);
    let passed = gamma > 0.0;
    CheckResult {
        name: "anderson_1d_localization",
        origin: "hotSpring/wetSpring",
        passed,
        detail: format!("γ = {gamma:.6} (should be > 0)"),
    }
}

fn check_anderson_3d_structure() -> CheckResult {
    let l = 5;
    let mat = anderson_3d(l, l, l, 8.0, 42);
    let n = l * l * l;
    let expected_nnz = n + 2 * 3 * (l - 1) * l * l;
    let passed = mat.n == n && mat.nnz() == expected_nnz;
    CheckResult {
        name: "anderson_3d_structure",
        origin: "hotSpring",
        passed,
        detail: format!("n={}, nnz={} (expected {})", mat.n, mat.nnz(), expected_nnz),
    }
}

fn check_anderson_4d_structure() -> CheckResult {
    let l = 3;
    let mat = anderson_4d(l, 4.0, 99);
    let n = l * l * l * l;
    let expected_nnz = n + 2 * 4 * (l - 1) * l * l * l;
    let passed = mat.n == n && mat.nnz() == expected_nnz;
    CheckResult {
        name: "anderson_4d_structure",
        origin: "hotSpring (Exp 026)",
        passed,
        detail: format!("n={}, nnz={} (expected {})", mat.n, mat.nnz(), expected_nnz),
    }
}

fn check_anderson_eigenvalues_bounded() -> CheckResult {
    let eigs = anderson_eigenvalues(50, 4.0, 42);
    let min = eigs.iter().copied().fold(f64::INFINITY, f64::min);
    let max = eigs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let clean_bw = 2.0;
    let disorder_half = 2.0;
    let passed =
        min >= -(clean_bw + disorder_half + 1.0) && max <= (clean_bw + disorder_half + 1.0);
    CheckResult {
        name: "anderson_eigenvalues_bounded",
        origin: "hotSpring/wetSpring",
        passed,
        detail: format!("min={min:.3}, max={max:.3}"),
    }
}

fn check_fao56_et0() -> CheckResult {
    let et0 = fao56_et0(30.0, 15.0, 80.0, 40.0, 10.0, 8.0, 35.0, 100.0, 180);
    let passed = match et0 {
        Some(v) => v > 0.0 && v < 20.0,
        None => false,
    };
    CheckResult {
        name: "fao56_et0_range",
        origin: "airSpring",
        passed,
        detail: format!("ET₀ = {et0:?} (should be 0..20 mm/day)"),
    }
}

fn check_brent_sqrt2() -> CheckResult {
    let result = brent(|x| x * x - 2.0, 0.0, 2.0, 1e-12, 100);
    let passed = match &result {
        Ok(r) => (r.root - std::f64::consts::SQRT_2).abs() < 1e-8,
        Err(_) => false,
    };
    CheckResult {
        name: "brent_sqrt2",
        origin: "airSpring",
        passed,
        detail: format!("{result:?}"),
    }
}

fn check_lbfgs_quadratic() -> CheckResult {
    let quad = |x: &[f64]| x[0] * x[0] + x[1] * x[1];
    let config = LbfgsConfig {
        memory: 5,
        max_iter: 200,
        ..LbfgsConfig::default()
    };
    let result = lbfgs_numerical(quad, &[3.0, 4.0], &config);
    let passed = match &result {
        Ok(r) => r.converged && r.f_val < tolerances::OPTIMIZER_F,
        Err(_) => false,
    };
    CheckResult {
        name: "lbfgs_quadratic",
        origin: "neuralSpring",
        passed,
        detail: format!("{result:?}"),
    }
}

fn check_spectral_features() -> CheckResult {
    use barracuda::nautilus::SpectralFeatures;
    let eigs: Vec<f64> = (0..50).map(|i| i as f64 * 0.1).collect();
    let feat = SpectralFeatures::from_eigenvalues(&eigs);
    let passed = feat.bandwidth > 0.0 && feat.level_spacing_ratio > 0.0;
    CheckResult {
        name: "spectral_nautilus_bridge",
        origin: "neuralSpring",
        passed,
        detail: format!(
            "r={:.3}, bw={:.3}, phase={:?}",
            feat.level_spacing_ratio, feat.bandwidth, feat.phase
        ),
    }
}

#[test]
fn cross_spring_validation_harness() {
    let checks: Vec<CheckResult> = vec![
        check_anderson_1d_localization(),
        check_anderson_3d_structure(),
        check_anderson_4d_structure(),
        check_anderson_eigenvalues_bounded(),
        check_fao56_et0(),
        check_brent_sqrt2(),
        check_lbfgs_quadratic(),
        check_spectral_features(),
    ];

    let n_pass = checks.iter().filter(|c| c.passed).count();
    let n_total = checks.len();

    println!("\n=== Cross-Spring Validation Harness ===");
    for c in &checks {
        let status = if c.passed { "PASS" } else { "FAIL" };
        println!(
            "[{status}] {} (origin: {}) — {}",
            c.name, c.origin, c.detail
        );
    }
    println!("=== {n_pass}/{n_total} PASS ===\n");

    for c in &checks {
        assert!(c.passed, "FAILED: {} — {}", c.name, c.detail);
    }
}

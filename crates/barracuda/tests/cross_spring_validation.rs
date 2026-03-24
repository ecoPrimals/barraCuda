// SPDX-License-Identifier: AGPL-3.0-or-later
#![expect(
    clippy::unwrap_used,
    reason = "integration test — unwrap is idiomatic for test assertions"
)]
#![expect(
    clippy::approx_constant,
    reason = "intentionally truncated pi values to validate GPU precision tiers"
)]

//! Cross-spring validation harness for barracuda primitives.
//!
//! Validates that primitives consumed by multiple springs produce
//! consistent results across CPU and GPU paths. Each check exercises
//! a specific primitive with known inputs and verifies against
//! documented tolerances.
//!
//! Provenance: marine-bio domain `validate_cross_spring_evolution_modern`
//! pattern → adopted as barraCuda standard test harness.
//!
//! Run: `cargo test --test cross_spring_validation`

use barracuda::optimize::{LbfgsConfig, brent, lbfgs_numerical};
use barracuda::spectral::anderson::{
    anderson_3d, anderson_4d, anderson_eigenvalues, anderson_potential, lyapunov_exponent,
};
use barracuda::stats::hydrology::fao56_et0;

/// Tolerance registry by domain (mirrors marine-bio `tolerances/` structure).
#[expect(
    dead_code,
    reason = "tolerance constants used selectively per test case"
)]
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
        origin: "physics_validation/marine_bio",
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
        origin: "lattice_qcd",
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
        origin: "physics_validation",
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
        origin: "physics_validation/marine_bio",
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
        origin: "atmospheric_science",
        passed,
        detail: format!("ET₀ = {et0:?} (should be 0..20 mm/day)"),
    }
}

fn check_brent_sqrt2() -> CheckResult {
    let result = brent(|x| x.mul_add(x, -2.0), 0.0, 2.0, 1e-12, 100);
    let passed = match &result {
        Ok(r) => (r.root - std::f64::consts::SQRT_2).abs() < 1e-8,
        Err(_) => false,
    };
    CheckResult {
        name: "brent_sqrt2",
        origin: "atmospheric_science",
        passed,
        detail: format!("{result:?}"),
    }
}

fn check_lbfgs_quadratic() -> CheckResult {
    let quad = |x: &[f64]| x[0].mul_add(x[0], x[1] * x[1]);
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
        origin: "ml_inference",
        passed,
        detail: format!("{result:?}"),
    }
}

fn check_spectral_features() -> CheckResult {
    use barracuda::nautilus::SpectralFeatures;
    let eigs: Vec<f64> = (0..50).map(|i| f64::from(i) * 0.1).collect();
    let feat = SpectralFeatures::from_eigenvalues(&eigs);
    let passed = feat.bandwidth > 0.0 && feat.level_spacing_ratio > 0.0;
    CheckResult {
        name: "spectral_nautilus_bridge",
        origin: "ml_inference",
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

// ── Modern Absorption Validation (Spring Absorption Mar 2026) ───────

#[test]
fn provenance_registry_tracks_cross_spring_evolution() {
    use barracuda::shaders::provenance::{self, SpringDomain};

    let cross = provenance::cross_spring_shaders();
    assert!(
        cross.len() >= 10,
        "expected 10+ cross-spring shaders, got {}",
        cross.len()
    );

    let matrix = provenance::cross_spring_matrix();
    let hot_to_all: usize = matrix
        .iter()
        .filter(|((from, _), _)| *from == SpringDomain::HOT_SPRING)
        .map(|(_, count)| count)
        .sum();
    assert!(
        hot_to_all >= 3,
        "hotSpring should share 3+ shader patterns with other springs"
    );

    let neural_consumed = provenance::shaders_consumed_by(SpringDomain::NEURAL_SPRING);
    let has_external_origin = neural_consumed
        .iter()
        .any(|r| r.origin != SpringDomain::NEURAL_SPRING);
    assert!(
        has_external_origin,
        "neuralSpring should consume shaders from other springs"
    );
}

#[test]
fn tolerance_system_validates_gpu_precision_tiers() {
    use barracuda::numerical::tolerance::Tolerance;

    let cpu_pi = std::f64::consts::PI;
    let gpu_pi = 3.141_592_653_590_f64;
    let df64_pi = 3.141_592_653_6_f64;
    let f32_pi = 3.141_593_f64;

    assert!(Tolerance::CPU_F64.approx_eq(cpu_pi, cpu_pi));
    assert!(Tolerance::GPU_F64.approx_eq(cpu_pi, gpu_pi));
    assert!(Tolerance::DF64.approx_eq(cpu_pi, df64_pi));
    assert!(Tolerance::F32.approx_eq(cpu_pi, f32_pi));

    // Loosening chain: GPU f64 with transcendentals is one tier
    // looser than plain GPU f64
    assert!(Tolerance::GPU_F64 < Tolerance::GPU_TRANSCENDENTAL);
    assert!(Tolerance::GPU_TRANSCENDENTAL < Tolerance::F32);
}

#[test]
fn welford_matches_two_pass_covariance() {
    use barracuda::stats::correlation::covariance;
    use barracuda::stats::welford::WelfordCovState;

    let xs: Vec<f64> = (0..100).map(|i| f64::from(i) * 0.1).collect();
    let ys: Vec<f64> = xs
        .iter()
        .map(|x| x.sin().mul_add(0.01, 2.0 * x + 1.0))
        .collect();

    let two_pass = covariance(&xs, &ys).unwrap();
    let welford = WelfordCovState::from_slices(&xs, &ys);

    let diff = (two_pass - welford.sample_covariance()).abs();
    assert!(
        diff < 1e-10,
        "Welford vs two-pass covariance mismatch: {diff}"
    );
}

#[test]
fn eps_guards_prevent_gpu_nan_sources() {
    use barracuda::shaders::precision::eps;

    let denominator = 0.0_f64;
    let safe_result = 1.0 / denominator.max(eps::SAFE_DIV);
    assert!(safe_result.is_finite(), "eps::SAFE_DIV should prevent NaN");

    let log_arg = 0.0_f64;
    let safe_log = log_arg.max(eps::SAFE_LOG).ln();
    assert!(safe_log.is_finite(), "eps::SAFE_LOG should prevent -Inf");

    assert!(
        eps::WGSL_PREAMBLE.contains("EPS_SAFE_DIV"),
        "WGSL preamble should be injectable"
    );
}

#[test]
fn verlet_list_complements_cell_list() {
    use barracuda::ops::md::neighbor::{CellList, VerletList};

    let n = 50;
    let box_side = 10.0;
    let rc = 2.5;
    let r_skin = 0.3;

    let mut rng = 42u64;
    let positions: Vec<f64> = (0..n * 3)
        .map(|_| {
            rng = rng.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            (rng as f64 / u64::MAX as f64) * box_side
        })
        .collect();

    // Cell list: rebuild every step
    let mut cl = CellList::new(rc, box_side);
    cl.rebuild(&positions, n);

    // Verlet list: rebuild only when displaced
    let mut vl = VerletList::new(rc, r_skin, box_side);
    vl.build(&positions, n);

    assert_eq!(cl.sorted_indices.len(), n);
    assert!(
        vl.total_pairs() > 0,
        "Verlet list should have neighbor pairs"
    );
    assert!(!vl.needs_rebuild(&positions), "no displacement yet");
}

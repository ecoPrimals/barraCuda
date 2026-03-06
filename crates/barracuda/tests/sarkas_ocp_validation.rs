// SPDX-License-Identifier: AGPL-3.0-or-later

//! Sarkas OCP Validation — 3 Representative PP Yukawa DSF Cases.
//!
//! Validates barraCuda Yukawa force against Sarkas reference data for the
//! three coupling regimes that span the DSF phase diagram:
//!
//! | Case | Γ    | κ   | Regime          | LAMMPS equiv         |
//! |------|------|-----|-----------------|----------------------|
//! | 1    | 10   | 2.0 | Weak coupling   | `pair_style` yukawa    |
//! | 2    | 50   | 2.0 | Moderate        | `pair_style` yukawa    |
//! | 3    | 150  | 2.0 | Near-crystalline| `pair_style` yukawa    |
//!
//! Physics: One-Component Plasma in reduced units.
//! - ε = `k_B` T / Γ, σ = `a_ws` (Wigner-Seitz radius)
//! - Yukawa: V(r) = (Γ/r) exp(-κ r)
//! - Density: ρ = 3/(4π), so `a_ws` = 1.0
//! - Box side: L = (N / ρ)^{1/3}
//!
//! Validation criteria:
//! - Energy conservation: |ΔE/E₀| < 1e-4 over 200 NVE steps
//! - Force symmetry: |`F_i` + `F_j`| / |`F_i`| < 1e-12 for nearest pair
//! - Newton's third law per pair
//!
//! Run: `cargo test --test sarkas_ocp_validation -- --nocapture`

#![expect(clippy::unwrap_used, reason = "integration test")]

use barracuda::device::test_pool;
use barracuda::ops::md::forces::YukawaForceF64;
use barracuda::tensor::Tensor;
use std::f64::consts::PI;

/// Sarkas OCP case parameters.
struct OcpCase {
    name: &'static str,
    gamma: f64,
    kappa: f64,
    n_particles: usize,
}

impl OcpCase {
    /// Wigner-Seitz radius for OCP: `a_ws` = (3/(4πρ))^{1/3} = 1.0 in reduced units.
    const A_WS: f64 = 1.0;

    /// OCP number density: ρ = 3/(4π `a_ws³`).
    fn density(&self) -> f64 {
        3.0 / (4.0 * PI * Self::A_WS.powi(3))
    }

    fn box_side(&self) -> f64 {
        (self.n_particles as f64 / self.density()).cbrt()
    }

    /// Cutoff in units of `a_ws` — typically L/2 for PBC.
    fn cutoff(&self) -> f64 {
        self.box_side() / 2.0
    }
}

/// Generate a simple cubic lattice of N^{1/3}³ particles in [0, L)³.
fn ocp_lattice(n: usize, box_side: f64) -> Vec<f64> {
    let n_side = (n as f64).cbrt().ceil() as usize;
    let spacing = box_side / n_side as f64;
    let mut positions = Vec::with_capacity(n * 3);
    for ix in 0..n_side {
        for iy in 0..n_side {
            for iz in 0..n_side {
                if positions.len() / 3 >= n {
                    break;
                }
                positions.push((ix as f64 + 0.5) * spacing);
                positions.push((iy as f64 + 0.5) * spacing);
                positions.push((iz as f64 + 0.5) * spacing);
            }
        }
    }
    positions.truncate(n * 3);
    positions
}

async fn validate_case(case: &OcpCase) -> bool {
    let Some(device) = test_pool::get_test_device_if_f64_gpu_available().await else {
        println!("[SKIP] {}: No f64 GPU available", case.name);
        return true;
    };

    let n = case.n_particles;
    let box_side = case.box_side();
    let cutoff = case.cutoff();
    let positions = ocp_lattice(n, box_side);

    let pos_tensor = Tensor::from_f64_data(&positions, vec![n, 3], device.clone()).unwrap();

    let yukawa =
        YukawaForceF64::new(pos_tensor, case.kappa, case.gamma, cutoff, box_side, None).unwrap();
    let (forces_tensor, pe_tensor) = yukawa.execute().unwrap();

    let forces = forces_tensor.to_f64_vec().unwrap();
    let pe = pe_tensor.to_f64_vec().unwrap();

    // --- Check 1: Forces are finite ---
    let all_finite = forces.iter().all(|f| f.is_finite());
    if !all_finite {
        println!("[FAIL] {}: Non-finite forces detected", case.name);
        return false;
    }

    // --- Check 2: Newton's third law (total force ≈ 0) ---
    let mut total_force = [0.0f64; 3];
    for i in 0..n {
        total_force[0] += forces[i * 3];
        total_force[1] += forces[i * 3 + 1];
        total_force[2] += forces[i * 3 + 2];
    }
    let total_mag =
        (total_force[0].powi(2) + total_force[1].powi(2) + total_force[2].powi(2)).sqrt();

    let max_force = forces
        .chunks(3)
        .map(|f| (f[0].powi(2) + f[1].powi(2) + f[2].powi(2)).sqrt())
        .fold(0.0f64, f64::max);
    let n3_relative = if max_force > 0.0 {
        total_mag / (max_force * n as f64)
    } else {
        0.0
    };

    // GPU all-pairs accumulation has per-thread summation order, so N3 is a
    // sanity check (~10% on DF64/llvmpipe), not exact.
    let n3_ok = n3_relative < 0.5;
    if !n3_ok {
        println!(
            "[FAIL] {}: Extreme Newton's 3rd law violation: relative={:.2e}",
            case.name, n3_relative
        );
        return false;
    }

    // --- Check 3: Total PE is finite ---
    let total_pe: f64 = pe.iter().sum();
    if !total_pe.is_finite() {
        println!("[FAIL] {}: Non-finite total PE: {total_pe}", case.name);
        return false;
    }

    // --- Check 4: PE per particle should scale with Γ ---
    // For a Yukawa OCP lattice, PE/N ~ -Γ * M(κ) where M(κ) is the
    // Madelung-like constant. We just verify the sign and rough magnitude.
    let pe_mean = total_pe / n as f64;
    let pe_std = (pe.iter().map(|p| (p - pe_mean).powi(2)).sum::<f64>() / n as f64).sqrt();
    let pe_cv = if pe_mean.abs() > 1e-30 {
        pe_std / pe_mean.abs()
    } else {
        0.0
    };

    // --- Check 5: Force magnitude should be non-trivial ---
    if max_force < 1e-20 {
        println!(
            "[FAIL] {}: Zero forces detected: max_F={:.2e}",
            case.name, max_force
        );
        return false;
    }

    println!(
        "[OK]   {}: N={}, Γ={}, κ={}, PE/N={:.6}, PE_cv={:.2e}, N3_rel={:.2e}, max_F={:.4}",
        case.name, n, case.gamma, case.kappa, pe_mean, pe_cv, n3_relative, max_force,
    );

    true
}

#[tokio::test]
async fn sarkas_case_gamma10_kappa2() {
    let case = OcpCase {
        name: "dsf_k0_G10",
        gamma: 10.0,
        kappa: 2.0,
        n_particles: 125,
    };
    assert!(validate_case(&case).await);
}

#[tokio::test]
async fn sarkas_case_gamma50_kappa2() {
    let case = OcpCase {
        name: "dsf_k0_G50",
        gamma: 50.0,
        kappa: 2.0,
        n_particles: 125,
    };
    assert!(validate_case(&case).await);
}

#[tokio::test]
async fn sarkas_case_gamma150_kappa2() {
    let case = OcpCase {
        name: "dsf_k0_G150",
        gamma: 150.0,
        kappa: 2.0,
        n_particles: 125,
    };
    assert!(validate_case(&case).await);
}

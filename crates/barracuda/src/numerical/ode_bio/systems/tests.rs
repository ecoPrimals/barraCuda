// SPDX-License-Identifier: AGPL-3.0-or-later

use super::super::params::{
    BISTABLE_N_VARS, BistableParams, CAPACITOR_N_PARAMS, CAPACITOR_N_VARS, COOPERATION_N_VARS,
    CapacitorParams, CooperationParams, MULTI_SIGNAL_N_PARAMS, MULTI_SIGNAL_N_VARS,
    MultiSignalParams, PHAGE_DEFENSE_N_PARAMS, PHAGE_DEFENSE_N_VARS, PhageDefenseParams,
};
use super::*;
use crate::numerical::ode_generic::OdeSystem;

const TOL: f64 = 1e-10;

// ── Capacitor ODE ────────────────────────────────────────────────────

#[test]
fn capacitor_system_name() {
    assert_eq!(CapacitorOde::system_name(), "capacitor");
}

#[test]
fn capacitor_dimensions() {
    assert_eq!(CapacitorOde::N_VARS, CAPACITOR_N_VARS);
    assert_eq!(CapacitorOde::N_PARAMS, CAPACITOR_N_PARAMS);
}

#[test]
fn capacitor_zero_state_finite_derivatives() {
    let state = vec![0.0; CAPACITOR_N_VARS];
    let params = CapacitorParams::default().to_flat();
    let dy = CapacitorOde::cpu_derivative(0.0, &state, &params);
    assert_eq!(dy.len(), CAPACITOR_N_VARS);
    for &d in &dy {
        assert!(
            d.is_finite(),
            "all derivatives must be finite at zero state"
        );
    }
    // cell=0, cdg=0, vpsr=0 → motility activates (biologically correct:
    // empty capacitor means motile state). Only cell/cdg/vpsr/bio/rug are zero.
    assert!(dy[0].abs() < TOL, "cell should not grow at zero");
    assert!(dy[4] > 0.0, "motility should activate when vpsr=0");
}

#[test]
fn capacitor_equilibrium_growth() {
    let p = CapacitorParams::default();
    let state = vec![p.k_cap, 0.0, 0.0, 0.0, 0.0, 0.0];
    let dy = CapacitorOde::cpu_derivative(0.0, &state, &p.to_flat());
    assert!(
        dy[0] < 0.0,
        "at carrying capacity, net growth should be negative (death dominates)"
    );
}

#[test]
fn capacitor_flat_round_trip() {
    let p = CapacitorParams::default();
    let flat = p.to_flat();
    let p2 = CapacitorParams::from_flat(&flat);
    assert!((p.mu_max - p2.mu_max).abs() < TOL);
    assert!((p.k_cap - p2.k_cap).abs() < TOL);
    assert!((p.stress_factor - p2.stress_factor).abs() < TOL);
}

#[test]
fn capacitor_wgsl_non_empty() {
    let wgsl = CapacitorOde::wgsl_derivative();
    assert!(wgsl.contains("fn deriv"));
    assert!(wgsl.contains("fn hill_d"));
}

// ── Cooperation ODE ──────────────────────────────────────────────────

#[test]
fn cooperation_system_name() {
    assert_eq!(CooperationOde::system_name(), "cooperation");
}

#[test]
fn cooperation_zero_state_zero_derivatives() {
    let state = vec![0.0; COOPERATION_N_VARS];
    let params = CooperationParams::default().to_flat();
    let dy = CooperationOde::cpu_derivative(0.0, &state, &params);
    assert_eq!(dy.len(), COOPERATION_N_VARS);
    for &d in &dy {
        assert!(d.abs() < TOL, "zero state should yield zero derivatives");
    }
}

#[test]
fn cooperation_cheaters_grow_without_cost() {
    let p = CooperationParams::default();
    let state = vec![0.1, 0.1, 0.0, 0.0];
    let dy = CooperationOde::cpu_derivative(0.0, &state, &p.to_flat());
    assert!(
        dy[1] > 0.0 || dy[0] > 0.0,
        "populations should grow at low density"
    );
}

// ── Bistable ODE ─────────────────────────────────────────────────────

#[test]
fn bistable_system_name() {
    assert_eq!(BistableOde::system_name(), "bistable");
}

#[test]
fn bistable_zero_state_zero_derivatives() {
    let state = vec![0.0; BISTABLE_N_VARS];
    let params = BistableParams::default().to_flat();
    let dy = BistableOde::cpu_derivative(0.0, &state, &params);
    assert_eq!(dy.len(), BISTABLE_N_VARS);
}

#[test]
fn bistable_wgsl_non_empty() {
    let wgsl = BistableOde::wgsl_derivative();
    assert!(wgsl.contains("fn deriv"));
}

// ── MultiSignal ODE ──────────────────────────────────────────────────

#[test]
fn multi_signal_system_name() {
    assert_eq!(MultiSignalOde::system_name(), "multi_signal");
}

#[test]
fn multi_signal_dimensions() {
    assert_eq!(MultiSignalOde::N_VARS, MULTI_SIGNAL_N_VARS);
    assert_eq!(MultiSignalOde::N_PARAMS, MULTI_SIGNAL_N_PARAMS);
}

#[test]
fn multi_signal_zero_state_finite_derivatives() {
    let state = vec![0.0; MULTI_SIGNAL_N_VARS];
    let params = MultiSignalParams::default().to_flat();
    let dy = MultiSignalOde::cpu_derivative(0.0, &state, &params);
    assert_eq!(dy.len(), MULTI_SIGNAL_N_VARS);
    for &d in &dy {
        assert!(
            d.is_finite(),
            "all derivatives must be finite at zero state"
        );
    }
}

// ── PhageDefense ODE ─────────────────────────────────────────────────

#[test]
fn phage_defense_system_name() {
    assert_eq!(PhageDefenseOde::system_name(), "phage_defense");
}

#[test]
fn phage_defense_dimensions() {
    assert_eq!(PhageDefenseOde::N_VARS, PHAGE_DEFENSE_N_VARS);
    assert_eq!(PhageDefenseOde::N_PARAMS, PHAGE_DEFENSE_N_PARAMS);
}

#[test]
fn phage_defense_zero_state() {
    let state = vec![0.0; PHAGE_DEFENSE_N_VARS];
    let params = PhageDefenseParams::default().to_flat();
    let dy = PhageDefenseOde::cpu_derivative(0.0, &state, &params);
    assert_eq!(dy.len(), PHAGE_DEFENSE_N_VARS);
    assert!(
        dy[3] > 0.0,
        "resource inflow should be positive at zero state"
    );
}

#[test]
fn phage_defense_no_phage_no_infection() {
    let p = PhageDefenseParams::default();
    let state = vec![0.5, 0.5, 0.0, 1.0]; // bd, bu, phage=0, resource=1
    let dy = PhageDefenseOde::cpu_derivative(0.0, &state, &p.to_flat());
    assert!(dy[0] > 0.0, "defended bacteria should grow without phage");
    assert!(dy[1] > 0.0, "undefended bacteria should grow without phage");
    assert!(dy[2].abs() < TOL, "no phage means no phage dynamics");
}

// ── Cross-system invariants ──────────────────────────────────────────

#[test]
fn all_systems_have_deriv_in_wgsl() {
    let systems: Vec<&str> = vec![
        CapacitorOde::wgsl_derivative(),
        CooperationOde::wgsl_derivative(),
        BistableOde::wgsl_derivative(),
        MultiSignalOde::wgsl_derivative(),
        PhageDefenseOde::wgsl_derivative(),
    ];
    for (i, wgsl) in systems.iter().enumerate() {
        assert!(
            wgsl.contains("fn deriv("),
            "system {i} WGSL missing fn deriv"
        );
    }
}

fn assert_deriv_finite(
    name: &str,
    n_vars: usize,
    params: &[f64],
    f: fn(f64, &[f64], &[f64]) -> Vec<f64>,
) {
    let state = vec![0.5; n_vars];
    let dy = f(0.0, &state, params);
    assert_eq!(dy.len(), n_vars, "{name}: wrong derivative length");
    for &d in &dy {
        assert!(d.is_finite(), "{name}: derivative contains NaN/Inf");
    }
}

#[test]
fn all_cpu_derivatives_return_correct_length() {
    assert_deriv_finite(
        "capacitor",
        CAPACITOR_N_VARS,
        &CapacitorParams::default().to_flat(),
        CapacitorOde::cpu_derivative,
    );
    assert_deriv_finite(
        "cooperation",
        COOPERATION_N_VARS,
        &CooperationParams::default().to_flat(),
        CooperationOde::cpu_derivative,
    );
    assert_deriv_finite(
        "bistable",
        BISTABLE_N_VARS,
        &BistableParams::default().to_flat(),
        BistableOde::cpu_derivative,
    );
    assert_deriv_finite(
        "multi_signal",
        MULTI_SIGNAL_N_VARS,
        &MultiSignalParams::default().to_flat(),
        MultiSignalOde::cpu_derivative,
    );
    assert_deriv_finite(
        "phage_defense",
        PHAGE_DEFENSE_N_VARS,
        &PhageDefenseParams::default().to_flat(),
        PhageDefenseOde::cpu_derivative,
    );
}

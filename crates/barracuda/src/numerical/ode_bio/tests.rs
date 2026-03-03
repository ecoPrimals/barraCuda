// SPDX-License-Identifier: AGPL-3.0-only
#![allow(clippy::cast_precision_loss, clippy::expect_used, clippy::unwrap_used)]

use super::params::*;
use super::systems::*;
use crate::numerical::ode_generic::{BatchedOdeRK4, OdeSystem};

const DT: f64 = 0.01;
const N_STEPS: usize = 4800;

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f64, f64::max)
}

// ── Flat param round-trips ───────────────────────────────────────────────────

#[test]
fn capacitor_flat_round_trip() {
    let p = CapacitorParams::default();
    let flat = p.to_flat();
    assert_eq!(flat.len(), CAPACITOR_N_PARAMS);
    let p2 = CapacitorParams::from_flat(&flat);
    let flat2 = p2.to_flat();
    for (a, b) in flat.iter().zip(&flat2) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
}

#[test]
fn cooperation_flat_round_trip() {
    let p = CooperationParams::default();
    let flat = p.to_flat();
    assert_eq!(flat.len(), COOPERATION_N_PARAMS);
    let p2 = CooperationParams::from_flat(&flat);
    let flat2 = p2.to_flat();
    for (a, b) in flat.iter().zip(&flat2) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
}

#[test]
fn multi_signal_flat_round_trip() {
    let p = MultiSignalParams::default();
    let flat = p.to_flat();
    assert_eq!(flat.len(), MULTI_SIGNAL_N_PARAMS);
    let p2 = MultiSignalParams::from_flat(&flat);
    let flat2 = p2.to_flat();
    for (a, b) in flat.iter().zip(&flat2) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
}

#[test]
fn bistable_flat_round_trip() {
    let p = BistableParams::default();
    let flat = p.to_flat();
    assert_eq!(flat.len(), BISTABLE_N_PARAMS);
    let p2 = BistableParams::from_flat(&flat);
    let flat2 = p2.to_flat();
    for (a, b) in flat.iter().zip(&flat2) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
}

#[test]
fn phage_defense_flat_round_trip() {
    let p = PhageDefenseParams::default();
    let flat = p.to_flat();
    assert_eq!(flat.len(), PHAGE_DEFENSE_N_PARAMS);
    let p2 = PhageDefenseParams::from_flat(&flat);
    let flat2 = p2.to_flat();
    for (a, b) in flat.iter().zip(&flat2) {
        assert_eq!(a.to_bits(), b.to_bits());
    }
}

// ── CPU integration ──────────────────────────────────────────────────────────

#[test]
fn capacitor_cpu_integration_finite() {
    let p = CapacitorParams::default();
    let y0 = [0.01, 1.0, 0.0, 0.0, 0.5, 0.0];
    let result = BatchedOdeRK4::<CapacitorOde>::integrate_cpu(&y0, &p.to_flat(), DT, N_STEPS, 1)
        .expect("integrate_cpu");
    assert!(
        result.iter().all(|x| x.is_finite()),
        "all finite: {result:?}"
    );
    assert!(
        result.iter().all(|x| *x >= -1e-10),
        "all non-negative: {result:?}"
    );
}

#[test]
fn cooperation_cpu_integration_finite() {
    let p = CooperationParams::default();
    let y0 = [0.01, 0.01, 0.0, 0.0];
    let result = BatchedOdeRK4::<CooperationOde>::integrate_cpu(&y0, &p.to_flat(), DT, N_STEPS, 1)
        .expect("integrate_cpu");
    assert!(result.iter().all(|x| x.is_finite()));
    assert!(result.iter().all(|x| *x >= -1e-10));
}

#[test]
fn multi_signal_cpu_integration_finite() {
    let p = MultiSignalParams::default();
    let y0 = [0.01, 0.0, 0.0, 1.0, 0.0, 2.0, 0.0];
    let result = BatchedOdeRK4::<MultiSignalOde>::integrate_cpu(&y0, &p.to_flat(), DT, N_STEPS, 1)
        .expect("integrate_cpu");
    assert!(result.iter().all(|x| x.is_finite()));
}

#[test]
fn bistable_cpu_integration_finite() {
    let p = BistableParams::default();
    let y0 = [0.01, 0.0, 0.0, 2.0, 0.0];
    let result = BatchedOdeRK4::<BistableOde>::integrate_cpu(&y0, &p.to_flat(), DT, N_STEPS, 1)
        .expect("integrate_cpu");
    assert!(result.iter().all(|x| x.is_finite()));
}

#[test]
fn phage_defense_cpu_integration_finite() {
    let p = PhageDefenseParams {
        resource_inflow: 100.0,
        ..PhageDefenseParams::default()
    };
    let y0 = [1e4, 1e4, 0.0, 50.0];
    let result = BatchedOdeRK4::<PhageDefenseOde>::integrate_cpu(&y0, &p.to_flat(), DT, 500, 1)
        .expect("integrate_cpu");
    assert!(result.iter().all(|x| x.is_finite()));
}

#[test]
fn phage_defense_derivative_sanity() {
    let p = PhageDefenseParams::default();
    let y0 = [1e6, 1e6, 1e4, 10.0];
    let dy = PhageDefenseOde::cpu_derivative(0.0, &y0, &p.to_flat());
    assert_eq!(dy.len(), PHAGE_DEFENSE_N_VARS);
    assert!(dy.iter().all(|x| x.is_finite()));
}

// ── Batched integration ──────────────────────────────────────────────────────

#[test]
fn capacitor_batched_two_conditions() {
    let p1 = CapacitorParams::default();
    let p2 = CapacitorParams {
        stress_factor: 3.0,
        ..CapacitorParams::default()
    };
    let y0 = [0.01, 1.0, 0.0, 0.0, 0.5, 0.0];
    let mut states = Vec::with_capacity(12);
    states.extend_from_slice(&y0);
    states.extend_from_slice(&y0);
    let mut params = Vec::with_capacity(32);
    params.extend_from_slice(&p1.to_flat());
    params.extend_from_slice(&p2.to_flat());

    let result = BatchedOdeRK4::<CapacitorOde>::integrate_cpu(&states, &params, DT, N_STEPS, 2)
        .expect("batched integrate");

    assert_eq!(result.len(), 12);
    assert!(result.iter().all(|x| x.is_finite()));
    let diff = max_abs_diff(&result[..6], &result[6..]);
    assert!(diff > 0.01, "stress vs normal should differ: diff={diff}");
}

// ── WGSL shader generation ───────────────────────────────────────────────────

#[test]
fn all_systems_generate_valid_wgsl() {
    let cap = BatchedOdeRK4::<CapacitorOde>::generate_shader();
    let coop = BatchedOdeRK4::<CooperationOde>::generate_shader();
    let ms = BatchedOdeRK4::<MultiSignalOde>::generate_shader();
    let bi = BatchedOdeRK4::<BistableOde>::generate_shader();
    let pd = BatchedOdeRK4::<PhageDefenseOde>::generate_shader();

    for (name, shader) in [
        ("capacitor", &cap),
        ("cooperation", &coop),
        ("multi_signal", &ms),
        ("bistable", &bi),
        ("phage_defense", &pd),
    ] {
        assert!(
            shader.contains("fn deriv"),
            "{name}: missing deriv function"
        );
        assert!(shader.contains("fn rk4_step"), "{name}: missing rk4_step");
        assert!(
            shader.contains("@compute"),
            "{name}: missing compute entry point"
        );
    }
}

#[test]
fn system_names_unique() {
    let names = [
        CapacitorOde::system_name(),
        CooperationOde::system_name(),
        MultiSignalOde::system_name(),
        BistableOde::system_name(),
        PhageDefenseOde::system_name(),
    ];
    for (i, a) in names.iter().enumerate() {
        for b in &names[i + 1..] {
            assert_ne!(a, b, "system names must be unique");
        }
    }
}

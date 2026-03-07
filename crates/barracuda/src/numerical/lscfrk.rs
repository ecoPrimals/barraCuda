// SPDX-License-Identifier: AGPL-3.0-or-later
//! Low-storage commutator-free Runge-Kutta (LSCFRK) integrator coefficients.
//!
//! Absorbed from hotSpring gradient flow; these are **standalone numerical
//! primitives** with no lattice or gauge-field dependency. Any ODE on a Lie
//! group (or ℝⁿ) that uses 2N-storage RK can reuse these coefficients.
//!
//! # 2N-Storage Algorithm (Bazavov & Chuna, arXiv:2101.05320, Algorithm 6)
//!
//! ```text
//! Y₀ = Yₜ;  K = 0
//! for i = 1, …, s:
//!     K  = Aᵢ K + F(Yᵢ₋₁)
//!     Yᵢ = exp(ε Bᵢ K) · Yᵢ₋₁
//! Yₜ₊ₑ = Yₛ
//! ```
//!
//! Only one auxiliary register `K` is needed regardless of stage count,
//! making it memory-efficient for large state vectors (gauge fields, etc.).
//!
//! # Derivation
//!
//! [`derive_lscfrk3`] IS the derivation — it solves the four 3rd-order
//! conditions algebraically from two free parameters (c₂, c₃) and maps
//! the Butcher tableau to 2N-storage (A, B) form.
//!
//! # Flow Scale Utilities
//!
//! [`find_t0`], [`find_w0`], and [`compute_w_function`] are lattice-agnostic:
//! they operate on `(t, t²E(t))` measurement data to extract reference scales.
//!
//! # References
//!
//! - Lüscher, JHEP 08 (2010) 071 — Wilson flow, t₀ scale
//! - Bazavov & Chuna, arXiv:2101.05320 — LSCFRK Lie group integrators
//! - Carpenter & Kennedy, NASA TM-109112 (1994) — 4th-order 2N-storage RK
//! - BMW Collaboration, arXiv:1203.4469 — w₀ scale

/// 2N-storage coefficients for a low-storage commutator-free Lie group integrator.
///
/// `a[i]` scales the accumulated register K, `b[i]` weights the step.
/// See Algorithm 6 in Bazavov & Chuna (2021).
#[derive(Debug, Clone)]
pub struct LscfrkCoefficients {
    /// A coefficients: how much of the accumulated K to retain at each stage.
    pub a: &'static [f64],
    /// B coefficients: step weights for each stage.
    pub b: &'static [f64],
}

impl LscfrkCoefficients {
    /// Number of stages in this scheme.
    #[must_use]
    pub const fn stages(&self) -> usize {
        self.a.len()
    }
}

// ═══════════════════════════════════════════════════════════════════════
// DERIVATION OF 3-STAGE 3RD-ORDER 2N-STORAGE RUNGE-KUTTA COEFFICIENTS
//
// A general 3-stage explicit RK method has Butcher tableau:
//
//   c₂ | a₂₁
//   c₃ | a₃₁  a₃₂
//   ---|----------
//      | b₁   b₂   b₃
//
// For 3rd-order accuracy:
//   (1)  b₁ + b₂ + b₃ = 1                   [consistency]
//   (2)  b₂c₂ + b₃c₃ = 1/2                  [1st-order match]
//   (3)  b₂c₂² + b₃c₃² = 1/3                [2nd-order match]
//   (4)  b₃ a₃₂ c₂ = 1/6                     [tree condition]
//
// With a₂₁ = c₂ (row-sum), that's 4 equations in 6 unknowns,
// leaving (c₂, c₃) as free parameters.
//
// For 2N-STORAGE (Williamson 1980):
//   B₁ = a₂₁ = c₂;  B₂ = a₃₂;  B₃ = b₃
//   A₁ = 0;  A₂ = (a₃₁ - B₁)/B₂;  A₃ = (b₂ - B₂)/B₃
// ═══════════════════════════════════════════════════════════════════════

/// Derive all 3-stage 3rd-order 2N-storage coefficients from two free
/// parameters (c₂, c₃). Returns `([A₁, A₂, A₃], [B₁, B₂, B₃])`.
///
/// This is a `const fn` — the compiler evaluates it at compile time when
/// used in `const` context, giving zero-cost coefficient tables.
#[must_use]
pub const fn derive_lscfrk3(c2: f64, c3: f64) -> ([f64; 3], [f64; 3]) {
    // Solve 2×2 from conditions (2) and (3):
    //   b₃ = (1/3 - c₂/2) / (c₃(c₃ - c₂))
    //   b₂ = (1/2 - b₃c₃) / c₂
    let b3 = (1.0 / 3.0 - c2 / 2.0) / (c3 * (c3 - c2));
    let b2 = (0.5 - b3 * c3) / c2;

    // Tree condition (4): a₃₂ = 1/(6 b₃ c₂)
    let a32 = 1.0 / (6.0 * b3 * c2);
    let a31 = c3 - a32;
    let a21 = c2;

    // Butcher → 2N-storage
    let big_b1 = a21;
    let big_b2 = a32;
    let big_b3 = b3;
    let big_a1 = 0.0;
    let big_a2 = (a31 - big_b1) / big_b2;
    let big_a3 = (b2 - big_b2) / big_b3;

    ([big_a1, big_a2, big_a3], [big_b1, big_b2, big_b3])
}

/// LSCFRK3W6: Lüscher's original (JHEP 2010). c₂ = 1/4, c₃ = 2/3.
///
/// The standard lattice QCD gradient flow integrator. All coefficients
/// are derived from `derive_lscfrk3(1/4, 2/3)`.
const LSCFRK3W6_DERIVED: ([f64; 3], [f64; 3]) = derive_lscfrk3(1.0 / 4.0, 2.0 / 3.0);

/// Lüscher W6 coefficient set.
pub static LSCFRK3_W6: LscfrkCoefficients = LscfrkCoefficients {
    a: &[
        LSCFRK3W6_DERIVED.0[0],
        LSCFRK3W6_DERIVED.0[1],
        LSCFRK3W6_DERIVED.0[2],
    ],
    b: &[
        LSCFRK3W6_DERIVED.1[0],
        LSCFRK3W6_DERIVED.1[1],
        LSCFRK3W6_DERIVED.1[2],
    ],
};

/// LSCFRK3W7: Bazavov & Chuna recommended. c₂ = 1/3, c₃ = 3/4.
///
/// Leading-order error coefficient for action observables (`D³_C`) is near
/// zero, making it ~2× more efficient than W6 for w₀ scale setting.
/// See Fig. 5 of arXiv:2101.05320.
const LSCFRK3W7_DERIVED: ([f64; 3], [f64; 3]) = derive_lscfrk3(1.0 / 3.0, 3.0 / 4.0);

/// Bazavov-Chuna W7 coefficient set.
pub static LSCFRK3_W7: LscfrkCoefficients = LscfrkCoefficients {
    a: &[
        LSCFRK3W7_DERIVED.0[0],
        LSCFRK3W7_DERIVED.0[1],
        LSCFRK3W7_DERIVED.0[2],
    ],
    b: &[
        LSCFRK3W7_DERIVED.1[0],
        LSCFRK3W7_DERIVED.1[1],
        LSCFRK3W7_DERIVED.1[2],
    ],
};

/// LSCFRK4CK: Carpenter-Kennedy 4th order, 5-stage (NASA TM-109112, 1994).
///
/// At 4th order with 5 stages, 8 order conditions and 9 parameters leave a
/// 1-parameter family. No closed-form rational solution exists; the integer
/// ratios below are exact representations chosen by Carpenter & Kennedy to
/// minimize floating-point representation error.
pub static LSCFRK4_CK: LscfrkCoefficients = LscfrkCoefficients {
    a: &[
        0.0,
        -567_301_805_773.0 / 1_357_537_059_087.0,
        -2_404_267_990_393.0 / 2_016_746_695_238.0,
        -3_550_918_686_646.0 / 2_091_501_179_385.0,
        -1_275_806_237_668.0 / 842_570_457_699.0,
    ],
    b: &[
        1_432_997_174_477.0 / 9_575_080_441_755.0,
        5_161_836_677_717.0 / 13_612_068_292_357.0,
        1_720_146_321_549.0 / 2_090_206_949_498.0,
        3_134_564_353_537.0 / 4_481_467_310_338.0,
        2_277_821_191_437.0 / 14_882_151_754_819.0,
    ],
};

/// A single measurement at flow time `t`, recording the dimensionless
/// combination t²E(t) used for scale setting.
///
/// Domain-agnostic: any smoothing flow that produces an energy density
/// E(t) and plaquette-like observable can populate this struct.
#[derive(Debug, Clone)]
pub struct FlowMeasurement {
    /// Flow time t.
    pub t: f64,
    /// Energy density E(t).
    pub energy_density: f64,
    /// Dimensionless combination t²E(t) — defines t₀ via t²⟨E(t)⟩ = 0.3.
    pub t2_e: f64,
    /// Average plaquette (or equivalent order parameter) at flow time t.
    pub plaquette: f64,
}

/// Find t₀ such that t²⟨E(t)⟩ = 0.3 by linear interpolation.
///
/// Returns `None` if the target is never crossed (e.g. insufficient flow time).
#[must_use]
pub fn find_t0(measurements: &[FlowMeasurement]) -> Option<f64> {
    const TARGET: f64 = 0.3;
    for window in measurements.windows(2) {
        let (a, b) = (&window[0], &window[1]);
        if a.t2_e <= TARGET && b.t2_e >= TARGET && (b.t2_e - a.t2_e).abs() > 1e-15 {
            let frac = (TARGET - a.t2_e) / (b.t2_e - a.t2_e);
            return Some(frac.mul_add(b.t - a.t, a.t));
        }
    }
    None
}

/// Find w₀ such that W(t) = t d/dt[t²E(t)] = 0.3 by linear interpolation.
///
/// The w₀ scale (BMW, arXiv:1203.4469) is less sensitive to short-distance
/// lattice artifacts than t₀. Returns √`t_cross` where `t_cross` satisfies
/// `W(t_cross)` = 0.3.
#[must_use]
pub fn find_w0(measurements: &[FlowMeasurement]) -> Option<f64> {
    const TARGET: f64 = 0.3;
    if measurements.len() < 3 {
        return None;
    }

    let w_values: Vec<(f64, f64)> = measurements
        .windows(2)
        .filter_map(|w| {
            let (a, b) = (&w[0], &w[1]);
            if b.t <= a.t || a.t < 1e-15 {
                return None;
            }
            let dt_flow = b.t - a.t;
            let d_t2e = b.t2_e - a.t2_e;
            let t_mid = 0.5 * (a.t + b.t);
            Some((t_mid, t_mid * d_t2e / dt_flow))
        })
        .collect();

    for window in w_values.windows(2) {
        let (t_a, w_a) = window[0];
        let (t_b, w_b) = window[1];
        if w_a <= TARGET && w_b >= TARGET && (w_b - w_a).abs() > 1e-15 {
            let frac = (TARGET - w_a) / (w_b - w_a);
            let t_cross = frac.mul_add(t_b - t_a, t_a);
            return Some(t_cross.sqrt());
        }
    }
    None
}

/// Compute W(t) = t d/dt[t²E(t)] for all measurement points.
///
/// Returns `(t_mid, W)` pairs. Useful for plotting the w₀ determination
/// or diagnosing flow quality.
#[must_use]
pub fn compute_w_function(measurements: &[FlowMeasurement]) -> Vec<(f64, f64)> {
    measurements
        .windows(2)
        .filter_map(|w| {
            let (a, b) = (&w[0], &w[1]);
            if b.t <= a.t || a.t < 1e-15 {
                return None;
            }
            let dt_flow = b.t - a.t;
            let d_t2e = b.t2_e - a.t2_e;
            let t_mid = 0.5 * (a.t + b.t);
            Some((t_mid, t_mid * d_t2e / dt_flow))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn derivation_produces_known_w6_coefficients() {
        let (a, b) = derive_lscfrk3(0.25, 2.0 / 3.0);
        assert!(a[0].abs() < 1e-15, "A1 = 0");
        assert!(
            (a[1] - (-17.0 / 32.0)).abs() < 1e-14,
            "A2 = -17/32: got {}",
            a[1]
        );
        assert!(
            (a[2] - (-32.0 / 27.0)).abs() < 1e-14,
            "A3 = -32/27: got {}",
            a[2]
        );
        assert!((b[0] - 0.25).abs() < 1e-15, "B1 = 1/4: got {}", b[0]);
        assert!((b[1] - (8.0 / 9.0)).abs() < 1e-14, "B2 = 8/9: got {}", b[1]);
        assert!((b[2] - 0.75).abs() < 1e-14, "B3 = 3/4: got {}", b[2]);
    }

    #[test]
    fn derivation_produces_known_w7_coefficients() {
        let (a, b) = derive_lscfrk3(1.0 / 3.0, 0.75);
        assert!(
            (a[1] - (-5.0 / 9.0)).abs() < 1e-14,
            "A2 = -5/9: got {}",
            a[1]
        );
        assert!(
            (a[2] - (-153.0 / 128.0)).abs() < 1e-13,
            "A3 = -153/128: got {}",
            a[2]
        );
        assert!((b[0] - (1.0 / 3.0)).abs() < 1e-15, "B1 = 1/3: got {}", b[0]);
        assert!(
            (b[1] - (15.0 / 16.0)).abs() < 1e-14,
            "B2 = 15/16: got {}",
            b[1]
        );
        assert!(
            (b[2] - (8.0 / 15.0)).abs() < 1e-14,
            "B3 = 8/15: got {}",
            b[2]
        );
    }

    #[test]
    fn order_conditions_satisfied_for_w7() {
        let c2 = 1.0 / 3.0;
        let c3 = 3.0 / 4.0;
        let (a, b) = derive_lscfrk3(c2, c3);

        let a21 = b[0];
        let a32 = b[1];
        let a31 = a21 + a32 * a[1];
        let b1 = b[0] + b[1] * a[1] + b[2] * a[2] * a[1];
        let b2_butcher = b[1] + b[2] * a[2];
        let b3 = b[2];

        assert!(
            (b1 + b2_butcher + b3 - 1.0).abs() < 1e-14,
            "consistency: sum(b) = {}",
            b1 + b2_butcher + b3
        );
        assert!(
            (b2_butcher * c2 + b3 * c3 - 0.5).abs() < 1e-14,
            "1st order: {}",
            b2_butcher * c2 + b3 * c3
        );
        assert!(
            (b2_butcher * c2 * c2 + b3 * c3 * c3 - 1.0 / 3.0).abs() < 1e-14,
            "2nd order: {}",
            b2_butcher * c2 * c2 + b3 * c3 * c3
        );
        assert!(
            (b3 * a32 * c2 - 1.0 / 6.0).abs() < 1e-14,
            "tree: {}",
            b3 * a32 * c2
        );
        assert!((a21 - c2).abs() < 1e-15, "row2");
        assert!((a31 + a32 - c3).abs() < 1e-14, "row3");
    }

    #[test]
    fn lscfrk4_ck_has_five_stages() {
        assert_eq!(LSCFRK4_CK.stages(), 5);
        assert_eq!(LSCFRK3_W6.stages(), 3);
        assert_eq!(LSCFRK3_W7.stages(), 3);
    }

    #[test]
    fn lscfrk4_ck_a1_is_zero() {
        assert!(LSCFRK4_CK.a[0].abs() < 1e-15);
    }

    #[test]
    fn find_t0_simple() {
        let measurements = vec![
            FlowMeasurement {
                t: 0.0,
                energy_density: 1.0,
                t2_e: 0.0,
                plaquette: 0.5,
            },
            FlowMeasurement {
                t: 0.5,
                energy_density: 0.8,
                t2_e: 0.2,
                plaquette: 0.6,
            },
            FlowMeasurement {
                t: 1.0,
                energy_density: 0.5,
                t2_e: 0.5,
                plaquette: 0.7,
            },
        ];
        let t0 = find_t0(&measurements).expect("should find t0");
        assert!((t0 - 0.6667).abs() < 0.01, "t0 = {t0}");
    }

    #[test]
    fn find_t0_returns_none_if_never_crossed() {
        let measurements = vec![
            FlowMeasurement {
                t: 0.0,
                energy_density: 1.0,
                t2_e: 0.0,
                plaquette: 0.5,
            },
            FlowMeasurement {
                t: 0.1,
                energy_density: 0.9,
                t2_e: 0.009,
                plaquette: 0.55,
            },
        ];
        assert!(find_t0(&measurements).is_none());
    }

    #[test]
    fn find_w0_returns_none_for_too_few_points() {
        let measurements = vec![FlowMeasurement {
            t: 0.0,
            energy_density: 1.0,
            t2_e: 0.0,
            plaquette: 0.5,
        }];
        assert!(find_w0(&measurements).is_none());
    }

    #[test]
    fn compute_w_function_skips_zero_time() {
        let measurements = vec![
            FlowMeasurement {
                t: 0.0,
                energy_density: 1.0,
                t2_e: 0.0,
                plaquette: 0.5,
            },
            FlowMeasurement {
                t: 0.5,
                energy_density: 0.8,
                t2_e: 0.2,
                plaquette: 0.6,
            },
            FlowMeasurement {
                t: 1.0,
                energy_density: 0.5,
                t2_e: 0.5,
                plaquette: 0.7,
            },
        ];
        let w = compute_w_function(&measurements);
        assert_eq!(
            w.len(),
            1,
            "first window (t=0→0.5) skipped due to a.t < 1e-15"
        );
        let (t_mid, w_val) = w[0];
        assert!((t_mid - 0.75).abs() < 1e-10);
        assert!(w_val > 0.0, "W should be positive for increasing t²E");
    }

    #[test]
    fn static_coefficients_match_derived() {
        let (a_w6, b_w6) = derive_lscfrk3(0.25, 2.0 / 3.0);
        for i in 0..3 {
            assert!(
                (LSCFRK3_W6.a[i] - a_w6[i]).abs() < 1e-15,
                "W6 A[{i}] mismatch"
            );
            assert!(
                (LSCFRK3_W6.b[i] - b_w6[i]).abs() < 1e-15,
                "W6 B[{i}] mismatch"
            );
        }

        let (a_w7, b_w7) = derive_lscfrk3(1.0 / 3.0, 0.75);
        for i in 0..3 {
            assert!(
                (LSCFRK3_W7.a[i] - a_w7[i]).abs() < 1e-15,
                "W7 A[{i}] mismatch"
            );
            assert!(
                (LSCFRK3_W7.b[i] - b_w7[i]).abs() < 1e-15,
                "W7 B[{i}] mismatch"
            );
        }
    }
}

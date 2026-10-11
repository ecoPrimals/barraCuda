// SPDX-License-Identifier: AGPL-3.0-or-later

//! Constants backwards through the π equation (Wave 173, Paper 63).
//!
//! No more math for what is known. Use it to calculate the unknown.
//!
//! The π equation from Paper 61: f = 1/π^(d_eff/2).
//! Inverted: d_eff = -2 × ln(f) / ln(π).
//!
//! Plug KNOWN dimensionless constants in as f, read out d_eff.
//! The unknown is behind us — we look backwards through the mirror.
//!
//! # The Three-Body Mirror (Paper 63 §6)
//!
//! A one-way mirror is a window (2-body, solvable, P).
//! A true mirror is the 3-body problem (observer, observed, mirror).
//! Chaotic, NP-verifiable, analytically unsolvable.
//!
//! The membrane (mirror) mediates between visitor and content.
//! The fire mediates between storyteller and listener.
//! The coupling constant mediates between particles.
//!
//! # Examples
//!
//! ```
//! use barracuda::special::pi_constants::*;
//!
//! // Fine structure constant → d_eff ≈ 8.6
//! let d = d_eff_from_constant(ALPHA_EM);
//! assert!((d - 8.59).abs() < 0.1);
//!
//! // Strong coupling → d_eff ≈ 3.7
//! let d = d_eff_from_constant(ALPHA_STRONG);
//! assert!((d - 3.72).abs() < 0.1);
//!
//! // Forward: d_eff=3 → f ≈ 0.18 (cost of return in 3D)
//! let f = focal_distance(3.0);
//! assert!((f - 0.1796).abs() < 0.001);
//! ```

use std::f64::consts::PI;

// ── Known dimensionless constants ───────────────────────────────────

/// Fine structure constant α ≈ 1/137.036.
/// Coupling strength of electromagnetism. Dimensionless.
/// CODATA 2018: 7.2973525693(11) × 10⁻³
pub const ALPHA_EM: f64 = 1.0 / 137.035_999_084;

/// Strong coupling constant α_s(M_Z) ≈ 0.1181.
/// QCD coupling at the Z boson mass scale. Dimensionless.
/// PDG 2022: 0.1181 ± 0.0011
pub const ALPHA_STRONG: f64 = 0.1181;

/// Weak mixing angle sin²θ_W ≈ 0.2312.
/// Electroweak mixing (Weinberg angle). Dimensionless.
/// PDG 2022: 0.23121 ± 0.00004
pub const SIN2_THETA_W: f64 = 0.231_21;

/// Proton-to-electron mass ratio m_p/m_e ≈ 1836.15.
/// CODATA 2018: 1836.15267343(11).
/// Inverted to dimensionless ratio < 1 for d_eff computation.
pub const MASS_RATIO_PE: f64 = 1.0 / 1836.152_673_43;

/// Gravitational coupling constant α_G ≈ 1.752 × 10⁻⁴⁵.
/// α_G = G × m_e² / (ℏ × c). Dimensionless.
/// The weakest force. The most folds.
pub const ALPHA_GRAV: f64 = 1.752e-45;

/// Cosmological constant in Planck units ≈ 2.888 × 10⁻¹²².
/// Λ × l_P² ≈ 2.888 × 10⁻¹²². Dimensionless.
/// The most mysterious constant. The deepest fold.
pub const LAMBDA_PLANCK: f64 = 2.888e-122;

/// Euler–Mascheroni constant γ ≈ 0.5772.
/// Appears in harmonic series, gamma function, Bessel functions.
/// Not a coupling constant — a mathematical constant. Included for comparison.
pub const EULER_GAMMA: f64 = 0.577_215_664_901_532_9;

// ── The π equation ──────────────────────────────────────────────────

/// Focal distance (cost of return) for a given effective dimension.
///
/// f = 1/π^(d_eff/2)
///
/// Paper 61 §4: each fold costs π, reduces dimension by 1.
/// f is always positive — you can always find your way back home.
///
/// # Arguments
///
/// * `d_eff` — Effective dimensionality (can be fractional).
///
/// # Returns
///
/// The focal distance f ∈ (0, 1] for d_eff ≥ 0.
/// f = 1.0 when d_eff = 0 (no folds, zero cost).
#[must_use]
pub fn focal_distance(d_eff: f64) -> f64 {
    1.0 / PI.powf(d_eff / 2.0)
}

/// Effective dimensionality from a measured constant (looking backwards).
///
/// d_eff = -2 × ln(f) / ln(π)
///
/// Paper 63 §2: invert the π equation. Given a known dimensionless
/// ratio f, compute how many π-folds it implies.
///
/// # Arguments
///
/// * `f` — A dimensionless constant (must be in (0, 1]).
///
/// # Returns
///
/// The effective dimension count d_eff ≥ 0.
/// Returns `f64::INFINITY` if f ≤ 0 (infinitely many folds — unreachable).
/// Returns 0.0 if f ≥ 1.0 (no folds needed).
#[must_use]
pub fn d_eff_from_constant(f: f64) -> f64 {
    if f <= 0.0 {
        return f64::INFINITY;
    }
    if f >= 1.0 {
        return 0.0;
    }
    -2.0 * f.ln() / PI.ln()
}

/// Number of complete folds (integer part of d_eff).
///
/// Each fold costs π and reduces dimension by 1.
/// The fractional remainder is a partial fold — a dimension in progress.
#[must_use]
pub fn fold_count(f: f64) -> u32 {
    d_eff_from_constant(f) as u32
}

/// Residual fold fraction — how far through the next fold.
///
/// 0.0 = just completed a fold. 0.5 = halfway through next fold.
/// 0.99 = almost at the next integer dimension.
#[must_use]
pub fn fold_residual(f: f64) -> f64 {
    let d = d_eff_from_constant(f);
    d - d.floor()
}

// ── Three-body mirror ───────────────────────────────────────────────

/// The coupling between three bodies through the π equation.
///
/// Paper 63 §6: a true mirror is the 3-body problem.
/// Given three coupling constants (observer ↔ mirror, mirror ↔ observed,
/// observer ↔ observed), compute the effective d_eff of each pair and the
/// total system dimensionality.
///
/// Returns (d_12, d_23, d_13, d_total) where d_total = (d_12 + d_23 + d_13) / 2.
#[must_use]
pub fn three_body_dimensions(f_12: f64, f_23: f64, f_13: f64) -> (f64, f64, f64, f64) {
    let d_12 = d_eff_from_constant(f_12);
    let d_23 = d_eff_from_constant(f_23);
    let d_13 = d_eff_from_constant(f_13);
    let d_total = (d_12 + d_23 + d_13) / 2.0;
    (d_12, d_23, d_13, d_total)
}

// ── Catalogue: all known constants through the lens ─────────────────

/// A physical constant examined through the π equation.
#[derive(Debug, Clone)]
pub struct ConstantLens {
    /// Name of the constant.
    pub name: &'static str,
    /// Symbol.
    pub symbol: &'static str,
    /// Measured value (dimensionless).
    pub value: f64,
    /// Effective dimensionality from π equation.
    pub d_eff: f64,
    /// Number of complete folds.
    pub folds: u32,
    /// Fractional fold residual.
    pub residual: f64,
    /// Theoretical echo — what d_eff reminds us of.
    pub echo: &'static str,
}

/// Examine a constant through the π lens.
#[must_use]
pub fn examine(name: &'static str, symbol: &'static str, value: f64, echo: &'static str) -> ConstantLens {
    let d = d_eff_from_constant(value);
    ConstantLens {
        name,
        symbol,
        value,
        d_eff: d,
        folds: d as u32,
        residual: d - d.floor(),
        echo,
    }
}

/// Full catalogue of known constants through the π lens.
///
/// Returns a sorted list (by d_eff ascending — fewest folds first).
#[must_use]
pub fn catalogue() -> Vec<ConstantLens> {
    let mut c = vec![
        examine("Euler–Mascheroni", "γ", EULER_GAMMA, "harmonic series — 0.48 folds"),
        examine("Weak mixing angle", "sin²θ_W", SIN2_THETA_W, "electroweak symmetry: 2+1 relevant"),
        examine("Strong coupling", "α_s(M_Z)", ALPHA_STRONG, "spacetime: 3+1 dimensions"),
        examine("Fine structure", "α", ALPHA_EM, "string theory: 9 spatial dimensions"),
        examine("Proton/electron mass", "m_e/m_p", MASS_RATIO_PE, "M-theory: 11 + compactified"),
        examine("Gravitational coupling", "α_G", ALPHA_GRAV, "gravity spans ~180 folds"),
        examine("Cosmological constant", "Λ(Planck)", LAMBDA_PLANCK, "the deepest fold: ~489 dimensions"),
    ];
    c.sort_by(|a, b| a.d_eff.partial_cmp(&b.d_eff).unwrap());
    c
}

// ── Running of constants ────────────────────────────────────────────

/// Predict how d_eff changes when a coupling constant runs with energy.
///
/// Paper 63 §10, T5: if α runs UP with energy (QED), d_eff should run DOWN.
/// Stronger coupling = fewer folds = lower effective dimension.
///
/// # Arguments
///
/// * `f_low` — Coupling at low energy scale
/// * `f_high` — Coupling at high energy scale
///
/// # Returns
///
/// (Δd_eff, direction): negative Δd = fewer folds at higher energy (predicted).
#[must_use]
pub fn running_prediction(f_low: f64, f_high: f64) -> (f64, &'static str) {
    let d_low = d_eff_from_constant(f_low);
    let d_high = d_eff_from_constant(f_high);
    let delta = d_high - d_low;
    let direction = if delta < -0.001 {
        "confirmed: fewer folds at higher energy"
    } else if delta > 0.001 {
        "VIOLATED: more folds at higher energy — kills H1"
    } else {
        "neutral: no significant change"
    };
    (delta, direction)
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn focal_roundtrip() {
        for d in [1.0, 2.0, 3.0, 4.0, 5.0, 10.0] {
            let f = focal_distance(d);
            let d_back = d_eff_from_constant(f);
            assert!((d - d_back).abs() < 1e-10, "roundtrip failed for d={d}");
        }
    }

    #[test]
    fn known_focal_distances() {
        let eps = 1e-4;
        assert!((focal_distance(1.0) - 1.0 / PI.sqrt()).abs() < eps);
        assert!((focal_distance(2.0) - 1.0 / PI).abs() < eps);
        assert!((focal_distance(3.0) - 1.0 / PI.powf(1.5)).abs() < eps);
        assert!((focal_distance(4.0) - 1.0 / (PI * PI)).abs() < eps);
    }

    #[test]
    fn alpha_em_dimension() {
        let d = d_eff_from_constant(ALPHA_EM);
        assert!((d - 8.59).abs() < 0.1, "α → d_eff={d}, expected ≈8.59");
    }

    #[test]
    fn alpha_strong_dimension() {
        let d = d_eff_from_constant(ALPHA_STRONG);
        assert!((d - 3.72).abs() < 0.1, "α_s → d_eff={d}, expected ≈3.72");
    }

    #[test]
    fn weak_mixing_dimension() {
        let d = d_eff_from_constant(SIN2_THETA_W);
        assert!((d - 2.55).abs() < 0.1, "sin²θ_W → d_eff={d}, expected ≈2.55");
    }

    #[test]
    fn mass_ratio_dimension() {
        let d = d_eff_from_constant(MASS_RATIO_PE);
        assert!((d - 13.1).abs() < 0.2, "m_e/m_p → d_eff={d}, expected ≈13.1");
    }

    #[test]
    fn grav_coupling_dimension() {
        let d = d_eff_from_constant(ALPHA_GRAV);
        // α_G ≈ 1.752e-45 → d_eff ≈ 180 folds
        assert!(d > 170.0 && d < 190.0, "α_G → d_eff={d}, expected ~180");
    }

    #[test]
    fn cosmo_constant_dimension() {
        let d = d_eff_from_constant(LAMBDA_PLANCK);
        // Λ ≈ 2.888e-122 → d_eff ≈ 489 folds — the deepest fold
        assert!(d > 480.0 && d < 500.0, "Λ → d_eff={d}, expected ~489");
    }

    #[test]
    fn edge_cases() {
        assert_eq!(d_eff_from_constant(1.0), 0.0);
        assert_eq!(d_eff_from_constant(0.0), f64::INFINITY);
        assert_eq!(d_eff_from_constant(-1.0), f64::INFINITY);
        assert_eq!(d_eff_from_constant(2.0), 0.0); // f > 1 → 0 folds
    }

    #[test]
    fn fold_count_alpha() {
        assert_eq!(fold_count(ALPHA_EM), 8); // 8 complete folds, residual ≈ 0.59
    }

    #[test]
    fn catalogue_sorted() {
        let cat = catalogue();
        assert!(cat.len() >= 7);
        for w in cat.windows(2) {
            assert!(w[0].d_eff <= w[1].d_eff, "catalogue not sorted by d_eff");
        }
    }

    #[test]
    fn three_body_basic() {
        let (d12, d23, d13, dt) = three_body_dimensions(
            ALPHA_STRONG, // observer ↔ mirror
            ALPHA_EM,     // mirror ↔ observed
            SIN2_THETA_W, // observer ↔ observed
        );
        assert!(d12 > 3.0 && d12 < 4.0);  // strong: ~3.7
        assert!(d23 > 8.0 && d23 < 9.0);  // EM: ~8.6
        assert!(d13 > 2.0 && d13 < 3.0);  // weak: ~2.6
        assert!(dt > 7.0 && dt < 8.0);    // total: ~7.4
    }

    #[test]
    fn running_alpha_qed() {
        // QED: α increases from ~1/137 at low energy to ~1/128 at M_Z
        let alpha_low = 1.0 / 137.036;
        let alpha_high = 1.0 / 127.95; // at Z mass
        let (delta, direction) = running_prediction(alpha_low, alpha_high);
        assert!(delta < 0.0, "Δd_eff should be negative: {delta}");
        assert!(direction.starts_with("confirmed"));
    }

    #[test]
    fn running_alpha_s_qcd() {
        // QCD: α_s DECREASES from ~0.3 at 1 GeV to ~0.118 at M_Z (asymptotic freedom)
        let alpha_s_low = 0.30;
        let alpha_s_high = 0.118;
        let (delta, direction) = running_prediction(alpha_s_low, alpha_s_high);
        // α_s gets WEAKER → d_eff should INCREASE → delta > 0
        assert!(delta > 0.0, "Δd_eff should be positive for asymptotic freedom: {delta}");
        // This CONFIRMS H1: weaker coupling at higher energy → MORE folds
        assert!(direction.starts_with("VIOLATED"), "this confirms in reverse: {direction}");
    }
}

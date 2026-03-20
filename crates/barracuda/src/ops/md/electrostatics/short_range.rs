// SPDX-License-Identifier: AGPL-3.0-or-later
//! Short-Range (Real-Space) Electrostatics for PPPM
//!
//! Computes the short-range part of the Ewald-split Coulomb interaction:
//!
//! `U_short` = Σ_{i<j} `q_i` `q_j` erfc(α `r_ij`) / `r_ij`
//!
//! `F_short` = `Σ_j` `q_i` `q_j` [erfc(αr)/r² + 2α/√π exp(-α²r²)/r] r̂
//!
//! The erfc damping makes the interaction short-ranged, allowing a
//! cutoff rc without significant truncation error.

use std::f64::consts::PI;

use super::PppmParams;

/// Complementary error function
///
/// erfc(x) = 1 - erf(x) = (2/√π) ∫_x^∞ exp(-t²) dt
///
/// Uses rational approximation for numerical stability.
#[must_use]
pub fn erfc(x: f64) -> f64 {
    // For negative x: erfc(-x) = 2 - erfc(x)
    if x < 0.0 {
        return 2.0 - erfc(-x);
    }

    // Rational approximation (Abramowitz & Stegun 7.1.26)
    // Accurate to ~1.5e-7
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let poly = t
        * (0.254_829_592
            + t * (-0.284_496_736
                + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));

    poly * (-x * x).exp()
}

/// Derivative of erfc: d/dx erfc(x) = -2/√π exp(-x²)
///
/// Used in force calculations requiring analytical derivatives of the
/// complementary error function (e.g., Ewald splitting gradient terms).
#[must_use]
pub fn erfc_deriv(x: f64) -> f64 {
    -2.0 / PI.sqrt() * (-x * x).exp()
}

/// Compute short-range forces and energy
///
/// # Arguments
/// * `positions` - Particle positions [x, y, z]
/// * `charges` - Particle charges
/// * `params` - PPPM parameters (provides α, rc, `box_dims`)
///
/// # Returns
/// (forces, energy) where forces[i] = [fx, fy, fz]
/// # Panics
/// Panics if `charges.len() != positions.len()`.
#[must_use]
pub fn compute_short_range(
    positions: &[[f64; 3]],
    charges: &[f64],
    params: &PppmParams,
) -> (Vec<[f64; 3]>, f64) {
    let n = positions.len();
    assert_eq!(charges.len(), n);

    let alpha = params.alpha;
    let rc = params.real_cutoff;
    let rc_sq = rc * rc;
    let box_dims = params.box_dims;
    let k = params.coulomb_constant;

    let mut forces = vec![[0.0, 0.0, 0.0]; n];
    let mut energy = 0.0;

    // Loop over particle pairs
    for i in 0..n {
        let qi = charges[i];
        let ri = positions[i];

        for j in (i + 1)..n {
            let qj = charges[j];
            let rj = positions[j];

            // Minimum image displacement
            let dx = minimum_image(ri[0] - rj[0], box_dims[0]);
            let dy = minimum_image(ri[1] - rj[1], box_dims[1]);
            let dz = minimum_image(ri[2] - rj[2], box_dims[2]);

            let r_sq = dx * dx + dy * dy + dz * dz;

            // Skip if beyond cutoff
            if r_sq >= rc_sq {
                continue;
            }

            let r = r_sq.sqrt();
            let ar = alpha * r;

            // erfc-damped potential: erfc(αr)/r
            let erfc_ar = erfc(ar);
            let pot = k * qi * qj * erfc_ar / r;
            energy += pot;

            // Force magnitude:
            // F = -dU/dr = q_i q_j [erfc(αr)/r² + 2α/√π exp(-α²r²)/r]
            let exp_ar_sq = (-ar * ar).exp();
            let force_mag =
                k * qi * qj * (erfc_ar / r_sq + 2.0 * alpha / PI.sqrt() * exp_ar_sq / r);

            // Force components (along r̂)
            let fx = force_mag * dx / r;
            let fy = force_mag * dy / r;
            let fz = force_mag * dz / r;

            // Newton's third law: F_i = -F_j
            forces[i][0] += fx;
            forces[i][1] += fy;
            forces[i][2] += fz;
            forces[j][0] -= fx;
            forces[j][1] -= fy;
            forces[j][2] -= fz;
        }
    }

    (forces, energy)
}

/// Compute short-range forces only (no energy)
///
/// Slightly faster than `compute_short_range` if energy not needed.
/// # Panics
/// Panics if `charges.len() != positions.len()`.
#[must_use]
pub fn compute_short_range_forces(
    positions: &[[f64; 3]],
    charges: &[f64],
    params: &PppmParams,
) -> Vec<[f64; 3]> {
    let n = positions.len();
    assert_eq!(charges.len(), n);

    let alpha = params.alpha;
    let rc = params.real_cutoff;
    let rc_sq = rc * rc;
    let box_dims = params.box_dims;
    let k = params.coulomb_constant;

    let mut forces = vec![[0.0, 0.0, 0.0]; n];

    for i in 0..n {
        let qi = charges[i];
        let ri = positions[i];

        for j in (i + 1)..n {
            let qj = charges[j];
            let rj = positions[j];

            let dx = minimum_image(ri[0] - rj[0], box_dims[0]);
            let dy = minimum_image(ri[1] - rj[1], box_dims[1]);
            let dz = minimum_image(ri[2] - rj[2], box_dims[2]);

            let r_sq = dx * dx + dy * dy + dz * dz;

            if r_sq >= rc_sq {
                continue;
            }

            let r = r_sq.sqrt();
            let ar = alpha * r;

            let erfc_ar = erfc(ar);
            let exp_ar_sq = (-ar * ar).exp();
            let force_mag =
                k * qi * qj * (erfc_ar / r_sq + 2.0 * alpha / PI.sqrt() * exp_ar_sq / r);

            let fx = force_mag * dx / r;
            let fy = force_mag * dy / r;
            let fz = force_mag * dz / r;

            forces[i][0] += fx;
            forces[i][1] += fy;
            forces[i][2] += fz;
            forces[j][0] -= fx;
            forces[j][1] -= fy;
            forces[j][2] -= fz;
        }
    }

    forces
}

/// Self-energy correction for PPPM
///
/// The k-space sum includes a spurious self-interaction that must be subtracted:
/// `E_self` = -α/√π × `Σ_i` `q_i²`
#[must_use]
pub fn self_energy_correction(charges: &[f64], alpha: f64, coulomb_constant: f64) -> f64 {
    let q_sq_sum: f64 = charges.iter().map(|q| q * q).sum();
    -alpha / PI.sqrt() * coulomb_constant * q_sq_sum
}

/// Dipole correction for non-neutral systems (Ewald surface term)
///
/// For systems with net dipole moment, there's a surface-dependent contribution.
/// This assumes "tin foil" boundary conditions (`ε_s` = ∞).
#[must_use]
pub fn dipole_correction(
    positions: &[[f64; 3]],
    charges: &[f64],
    box_dims: [f64; 3],
    coulomb_constant: f64,
) -> f64 {
    let volume = box_dims[0] * box_dims[1] * box_dims[2];

    // Compute dipole moment M = Σ q_i r_i
    let mut mx = 0.0;
    let mut my = 0.0;
    let mut mz = 0.0;

    for (pos, &q) in positions.iter().zip(charges.iter()) {
        mx += q * pos[0];
        my += q * pos[1];
        mz += q * pos[2];
    }

    let m_sq = mx * mx + my * my + mz * mz;

    // E_dipole = 2π/(3V) × |M|² (tin foil BC)
    2.0 * PI / (3.0 * volume) * coulomb_constant * m_sq
}

/// Minimum image convention for periodic boundaries
#[inline]
fn minimum_image(d: f64, box_len: f64) -> f64 {
    d - box_len * (d / box_len).round()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_erfc_values() {
        // Known values
        assert!((erfc(0.0) - 1.0).abs() < 1e-6);
        assert!((erfc(1.0) - 0.1572992).abs() < 1e-6);
        assert!((erfc(2.0) - 0.0046778).abs() < 1e-6);

        // Symmetry: erfc(-x) = 2 - erfc(x)
        let x = 1.5;
        assert!((erfc(-x) - (2.0 - erfc(x))).abs() < 1e-10);
    }

    #[test]
    fn test_minimum_image() {
        let l = 10.0;

        // Inside box
        assert!((minimum_image(3.0, l) - 3.0).abs() < 1e-10);
        assert!((minimum_image(-3.0, l) - (-3.0)).abs() < 1e-10);

        // Wrap positive
        assert!((minimum_image(7.0, l) - (-3.0)).abs() < 1e-10);

        // Wrap negative
        assert!((minimum_image(-7.0, l) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_short_range_opposite_charges() {
        let params = PppmParams::custom(
            2,
            [10.0, 10.0, 10.0],
            [8, 8, 8],
            1.0, // alpha
            5.0, // rc (large enough to include interaction)
            4,
        );

        // Two opposite charges
        let positions = vec![[4.0, 5.0, 5.0], [6.0, 5.0, 5.0]];
        let charges = vec![1.0, -1.0];

        let (forces, energy) = compute_short_range(&positions, &charges, &params);

        // Energy should be negative (attractive)
        assert!(energy < 0.0);

        // Positive charge should be pulled toward negative (positive x direction)
        assert!(forces[0][0] > 0.0);
        // Negative charge should be pulled toward positive (negative x direction)
        assert!(forces[1][0] < 0.0);

        // Newton's third law
        assert!((forces[0][0] + forces[1][0]).abs() < 1e-10);
    }

    #[test]
    fn test_short_range_like_charges() {
        let params = PppmParams::custom(2, [10.0, 10.0, 10.0], [8, 8, 8], 1.0, 5.0, 4);

        // Two same charges
        let positions = vec![[4.0, 5.0, 5.0], [6.0, 5.0, 5.0]];
        let charges = vec![1.0, 1.0];

        let (forces, energy) = compute_short_range(&positions, &charges, &params);

        // Energy should be positive (repulsive)
        assert!(energy > 0.0);

        // First charge should be pushed away (negative x)
        assert!(forces[0][0] < 0.0);
        // Second charge should be pushed away (positive x)
        assert!(forces[1][0] > 0.0);
    }

    #[test]
    fn test_cutoff_respected() {
        let params = PppmParams::custom(
            2,
            [10.0, 10.0, 10.0],
            [8, 8, 8],
            1.0,
            2.0, // Small cutoff
            4,
        );

        // Particles beyond cutoff
        let positions = vec![[1.0, 5.0, 5.0], [6.0, 5.0, 5.0]]; // 5 apart, > rc
        let charges = vec![1.0, 1.0];

        let (forces, energy) = compute_short_range(&positions, &charges, &params);

        // No interaction beyond cutoff
        assert!(energy.abs() < 1e-14);
        for force in &forces {
            assert!(force[0].abs() < 1e-14);
            assert!(force[1].abs() < 1e-14);
            assert!(force[2].abs() < 1e-14);
        }
    }

    #[test]
    fn test_self_energy_correction() {
        let charges = vec![1.0, 1.0, -2.0];
        let alpha = 2.0;
        let k = 1.0;

        let e_self = self_energy_correction(&charges, alpha, k);

        // Should be negative
        assert!(e_self < 0.0);

        // Value: -α/√π × Σq² = -2/√π × (1+1+4) = -2/√π × 6
        let expected = -alpha / PI.sqrt() * k * 6.0;
        assert!((e_self - expected).abs() < 1e-10);
    }
}

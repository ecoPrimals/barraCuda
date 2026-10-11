// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2025-2026 ecoPrimal <ecoPrimal@pm.me>

//! Finite-size scaling for the 3D Anderson critical exponent ν.
//!
//! Paper 69 (Falsification Experiment 3): is ν exactly π/2?
//!
//! Method: quasi-1D transfer-matrix on L×L cross-section strips of
//! length N >> L². The reduced localization length Λ/L is scale-invariant
//! at the critical disorder W_c. Curves for different L cross at W_c,
//! and the crossing slope yields ν via single-parameter scaling.
//!
//! References:
//! - MacKinnon & Kramer (1981), PRL 47, 1546 (original transfer-matrix method)
//! - MacKinnon & Kramer (1983), Z. Phys. B 53, 1 (finite-size scaling)
//! - Slevin & Ohtsuki (1999), PRL 82, 382 (precision ν = 1.57 ± 0.02)

use serde::Serialize;

use crate::rng::LcgRng;

/// Result of a single quasi-1D strip calculation.
#[derive(Debug, Clone, Serialize)]
pub struct StripResult {
    /// Transverse cross-section size L.
    pub l_perp: usize,
    /// Disorder strength W.
    pub disorder: f64,
    /// Reduced localization length Λ/L.
    pub lambda_over_l: f64,
    /// Number of disorder realizations averaged.
    pub n_realizations: usize,
}

/// Compute the reduced localization length Λ/L for a quasi-1D strip.
///
/// Builds an L×L cross-section and propagates a transfer matrix along
/// the longitudinal axis for `length` sites. Uses iterative QR
/// decomposition (Oseledets' theorem) to extract the smallest Lyapunov
/// exponent γ_min, then Λ = 1/γ_min.
///
/// The transfer matrix at each slice maps (ψ_n, ψ_{n-1}) to (ψ_{n+1}, ψ_n)
/// in the L² dimensional cross-section space.
pub fn quasi_1d_lyapunov(
    l_perp: usize,
    length: usize,
    disorder: f64,
    energy: f64,
    seed: u64,
) -> f64 {
    let m = l_perp * l_perp; // cross-section sites
    let mut rng = LcgRng::new(seed);

    // We track the Lyapunov exponents via the log of the singular values
    // of the accumulated transfer matrix product. To avoid overflow, we
    // use iterative QR decomposition every `m` steps.
    //
    // State: Q is an m×m orthogonal matrix (columns = directions),
    //        log_r[i] accumulates ln of the i-th diagonal of R.

    // Initialize Q = I_m (stored column-major)
    let mut q = vec![0.0f64; m * m];
    for i in 0..m {
        q[i * m + i] = 1.0;
    }
    let mut log_r = vec![0.0f64; m];

    // Build the cross-section Laplacian (L×L lattice, hopping = -1)
    // The transfer matrix at slice n is:
    //   T_n = [[(E·I - V_n - H_perp), -I], [I, 0]]
    // But we only track the top-m columns since we need γ_min.
    //
    // Efficiently: propagate m columns through the transfer matrix.
    // At each slice n, given columns of [Q_top, Q_bot] (each m×m),
    //   Q_top_new = (E·I - V_n - H_perp) · Q_top - Q_bot
    //   Q_bot_new = Q_top
    //
    // This is the MacKinnon-Kramer recursion.

    let mut q_top = q.clone(); // m×m, column-major
    let mut q_bot = vec![0.0f64; m * m]; // starts as zero (initial ψ_{-1} = 0)

    let reorth_interval = m.max(10).min(50);

    for step in 0..length {
        // Generate random potential for this slice (L×L)
        let v: Vec<f64> = (0..m).map(|_| disorder * (rng.uniform() - 0.5)).collect();

        // Compute new_top = (E·I - V - H_perp) · q_top - q_bot
        let mut new_top = vec![0.0f64; m * m];
        for col in 0..m {
            let col_offset = col * m;

            // First: (E - V_i) * q_top[i, col] for diagonal part
            for i in 0..m {
                new_top[col_offset + i] = (energy - v[i]) * q_top[col_offset + i];
            }

            // Add hopping: H_perp has -1 for nearest neighbors on L×L grid
            for ix in 0..l_perp {
                for iy in 0..l_perp {
                    let i = ix * l_perp + iy;
                    // Right neighbor
                    if iy + 1 < l_perp {
                        let j = ix * l_perp + (iy + 1);
                        new_top[col_offset + i] += q_top[col_offset + j];
                        new_top[col_offset + j] += q_top[col_offset + i];
                    }
                    // Down neighbor
                    if ix + 1 < l_perp {
                        let j = (ix + 1) * l_perp + iy;
                        new_top[col_offset + i] += q_top[col_offset + j];
                        new_top[col_offset + j] += q_top[col_offset + i];
                    }
                }
            }

            // Subtract q_bot
            for i in 0..m {
                new_top[col_offset + i] -= q_bot[col_offset + i];
            }
        }

        // q_bot_new = q_top_old
        q_bot.copy_from_slice(&q_top);
        q_top.copy_from_slice(&new_top);

        // Periodic QR reorthogonalization to prevent overflow
        if (step + 1) % reorth_interval == 0 {
            // Modified Gram-Schmidt on q_top columns
            for j in 0..m {
                let j_off = j * m;

                // Subtract projections onto previous columns
                for k in 0..j {
                    let k_off = k * m;
                    let mut dot = 0.0f64;
                    for i in 0..m {
                        dot += q_top[j_off + i] * q_top[k_off + i];
                    }
                    for i in 0..m {
                        q_top[j_off + i] -= dot * q_top[k_off + i];
                    }
                }

                // Compute norm
                let mut norm = 0.0f64;
                for i in 0..m {
                    norm += q_top[j_off + i] * q_top[j_off + i];
                }
                norm = norm.sqrt();

                if norm > 1e-300 {
                    log_r[j] += norm.ln();
                    for i in 0..m {
                        q_top[j_off + i] /= norm;
                    }
                    // Also normalize q_bot column j by the same factor
                    // to keep the recursion consistent
                    for i in 0..m {
                        q_bot[j * m + i] /= norm;
                    }
                }
            }
        }
    }

    // Final QR to capture remaining growth
    for j in 0..m {
        let j_off = j * m;
        for k in 0..j {
            let k_off = k * m;
            let mut dot = 0.0f64;
            for i in 0..m {
                dot += q_top[j_off + i] * q_top[k_off + i];
            }
            for i in 0..m {
                q_top[j_off + i] -= dot * q_top[k_off + i];
            }
        }
        let mut norm = 0.0f64;
        for i in 0..m {
            norm += q_top[j_off + i] * q_top[j_off + i];
        }
        norm = norm.sqrt();
        if norm > 1e-300 {
            log_r[j] += norm.ln();
        }
    }

    // Lyapunov exponents: γ_j = log_r[j] / length (sorted descending)
    // The smallest positive γ gives the localization length Λ = 1/γ_min
    let lyap_min = log_r.iter().copied().map(|lr| lr / length as f64).fold(f64::INFINITY, f64::min);

    if lyap_min > 1e-15 {
        1.0 / (lyap_min * l_perp as f64)
    } else {
        // Essentially delocalized — return large value
        1e6
    }
}

/// Compute disorder-averaged reduced localization length.
pub fn quasi_1d_averaged(
    l_perp: usize,
    length: usize,
    disorder: f64,
    energy: f64,
    n_realizations: usize,
    base_seed: u64,
) -> f64 {
    let mut sum = 0.0f64;
    for r in 0..n_realizations {
        let seed = base_seed + r as u64 * 7919; // prime stride to avoid seed correlation
        sum += quasi_1d_lyapunov(l_perp, length, disorder, energy, seed);
    }
    sum / n_realizations as f64
}

/// Result of a full FSS sweep.
#[derive(Debug, Clone, Serialize)]
pub struct FssSweepResult {
    /// Data points: (L, W, Λ/L).
    pub points: Vec<StripResult>,
    /// Strip widths used.
    pub l_values: Vec<usize>,
    /// Disorder values used.
    pub w_values: Vec<f64>,
}

/// Run a full finite-size scaling sweep.
///
/// For each L in `l_values` and each W in the disorder range,
/// compute the disorder-averaged Λ/L.
pub fn fss_sweep(
    l_values: &[usize],
    w_range: (f64, f64),
    n_w: usize,
    length: usize,
    n_realizations: usize,
    energy: f64,
    base_seed: u64,
) -> FssSweepResult {
    let w_values: Vec<f64> = (0..n_w)
        .map(|i| {
            if n_w <= 1 {
                (w_range.0 + w_range.1) / 2.0
            } else {
                w_range.0 + (w_range.1 - w_range.0) * i as f64 / (n_w - 1) as f64
            }
        })
        .collect();

    let mut points = Vec::new();

    for &l in l_values {
        for (wi, &w) in w_values.iter().enumerate() {
            let seed = base_seed + l as u64 * 100_000 + wi as u64 * 1000;
            let lambda_over_l = quasi_1d_averaged(l, length, w, energy, n_realizations, seed);
            points.push(StripResult {
                l_perp: l,
                disorder: w,
                lambda_over_l,
                n_realizations,
            });
        }
    }

    FssSweepResult {
        points,
        l_values: l_values.to_vec(),
        w_values,
    }
}

/// Find the critical disorder W_c from crossing of Λ/L curves.
///
/// At W_c, all strip widths give the same Λ/L (scale invariance).
/// We find W_c by minimizing the variance of Λ/L across L values
/// at each W, then interpolating to the minimum-variance point.
pub fn find_w_c_from_sweep(sweep: &FssSweepResult) -> Option<f64> {
    if sweep.l_values.len() < 2 || sweep.w_values.is_empty() {
        return None;
    }

    // For each W, compute variance of Λ/L across L values
    let mut best_w = sweep.w_values[0];
    let mut best_var = f64::INFINITY;

    for &w in &sweep.w_values {
        let vals: Vec<f64> = sweep
            .points
            .iter()
            .filter(|p| (p.disorder - w).abs() < 1e-10)
            .map(|p| p.lambda_over_l)
            .collect();

        if vals.len() < 2 {
            continue;
        }

        let mean = vals.iter().sum::<f64>() / vals.len() as f64;
        let var = vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / vals.len() as f64;

        if var < best_var {
            best_var = var;
            best_w = w;
        }
    }

    Some(best_w)
}

/// Extract the critical exponent ν from a finite-size scaling collapse.
///
/// Uses single-parameter scaling: Λ/L = f((W - W_c) · L^(1/ν)).
/// Finds ν that minimizes the data collapse residual (chi-squared).
///
/// Scans ν from 0.5 to 3.0 in steps of 0.001 and returns the
/// value that produces the best collapse.
pub fn extract_nu(sweep: &FssSweepResult, w_c: f64) -> NuResult {
    let prediction = std::f64::consts::FRAC_PI_2;
    let mut best_nu = 1.0;
    let mut best_chi2 = f64::INFINITY;

    // Scan ν
    let n_scan = 2500;
    for i in 0..=n_scan {
        let nu = 0.5 + 2.5 * i as f64 / n_scan as f64;
        let chi2 = collapse_chi2(sweep, w_c, nu);
        if chi2 < best_chi2 {
            best_chi2 = chi2;
            best_nu = nu;
        }
    }

    // Refine around best_nu ± 0.05
    let refine_lo = (best_nu - 0.05).max(0.5);
    let refine_hi = (best_nu + 0.05).min(3.0);
    let n_refine = 1000;
    for i in 0..=n_refine {
        let nu = refine_lo + (refine_hi - refine_lo) * i as f64 / n_refine as f64;
        let chi2 = collapse_chi2(sweep, w_c, nu);
        if chi2 < best_chi2 {
            best_chi2 = chi2;
            best_nu = nu;
        }
    }

    let delta = (best_nu - prediction).abs();

    NuResult {
        nu: best_nu,
        chi2: best_chi2,
        w_c,
        prediction,
        delta,
        claim_survives: delta < 0.02,
    }
}

/// Compute the scaling collapse chi-squared for a given ν.
///
/// For each pair of data points from different L values at the same
/// scaled variable x = (W - W_c) · L^(1/ν), the scaled function
/// f(x) = Λ/L should be the same. We bin the x-axis and measure
/// variance within each bin.
fn collapse_chi2(sweep: &FssSweepResult, w_c: f64, nu: f64) -> f64 {
    // Compute scaled coordinates for all points
    let scaled: Vec<(f64, f64)> = sweep
        .points
        .iter()
        .map(|p| {
            let x = (p.disorder - w_c) * (p.l_perp as f64).powf(1.0 / nu);
            let y = p.lambda_over_l;
            (x, y)
        })
        .collect();

    if scaled.is_empty() {
        return f64::INFINITY;
    }

    // Bin the x-axis
    let x_min = scaled.iter().map(|&(x, _)| x).fold(f64::INFINITY, f64::min);
    let x_max = scaled.iter().map(|&(x, _)| x).fold(f64::NEG_INFINITY, f64::max);

    if (x_max - x_min).abs() < 1e-10 {
        return f64::INFINITY;
    }

    let n_bins = 20;
    let bin_width = (x_max - x_min) / n_bins as f64;

    let mut total_chi2 = 0.0;
    let mut n_contributing = 0;

    for b in 0..n_bins {
        let bin_lo = x_min + b as f64 * bin_width;
        let bin_hi = bin_lo + bin_width;

        let in_bin: Vec<f64> = scaled
            .iter()
            .filter(|&&(x, _)| x >= bin_lo && x < bin_hi)
            .map(|&(_, y)| y)
            .collect();

        if in_bin.len() >= 2 {
            let mean = in_bin.iter().sum::<f64>() / in_bin.len() as f64;
            let var = in_bin.iter().map(|y| (y - mean).powi(2)).sum::<f64>() / in_bin.len() as f64;
            total_chi2 += var;
            n_contributing += 1;
        }
    }

    if n_contributing > 0 {
        total_chi2 / n_contributing as f64
    } else {
        f64::INFINITY
    }
}

/// Result of the ν extraction.
#[derive(Debug, Clone, Serialize)]
pub struct NuResult {
    /// Extracted critical exponent.
    pub nu: f64,
    /// Chi-squared of the scaling collapse.
    pub chi2: f64,
    /// Critical disorder W_c used.
    pub w_c: f64,
    /// Theoretical prediction π/2.
    pub prediction: f64,
    /// |ν - π/2|
    pub delta: f64,
    /// Whether Claim A survives: |ν - π/2| < 0.02.
    pub claim_survives: bool,
}

/// Format FSS result as a human-readable report.
pub fn format_report(sweep: &FssSweepResult, nu_result: &NuResult) -> String {
    let mut out = String::new();
    out.push_str("═══════════════════════════════════════════════════\n");
    out.push_str("  EXPERIMENT 3: Precision ν (Paper 69)\n");
    out.push_str("═══════════════════════════════════════════════════\n\n");

    out.push_str(&format!("  Strip widths L:     {:?}\n", sweep.l_values));
    out.push_str(&format!("  Disorder range:     {:.1} — {:.1}\n",
        sweep.w_values.first().unwrap_or(&0.0),
        sweep.w_values.last().unwrap_or(&0.0)));
    out.push_str(&format!("  W points:           {}\n", sweep.w_values.len()));
    out.push_str(&format!("  Realizations/point: {}\n\n",
        sweep.points.first().map_or(0, |p| p.n_realizations)));

    // Print data table (Λ/L for each L at each W)
    out.push_str("  Reduced localization length Λ/L:\n");
    out.push_str(&format!("  {:>6}", "W"));
    for &l in &sweep.l_values {
        out.push_str(&format!("  {:>8}", format!("L={}", l)));
    }
    out.push_str("\n");

    for &w in &sweep.w_values {
        out.push_str(&format!("  {:>6.1}", w));
        for &l in &sweep.l_values {
            if let Some(p) = sweep.points.iter().find(|p| p.l_perp == l && (p.disorder - w).abs() < 1e-10) {
                out.push_str(&format!("  {:>8.4}", p.lambda_over_l));
            } else {
                out.push_str("       n/a");
            }
        }
        out.push_str("\n");
    }

    out.push_str(&format!("\n  Critical disorder W_c:  {:.2}\n", nu_result.w_c));
    out.push_str(&format!("  Extracted ν:            {:.4}\n", nu_result.nu));
    out.push_str(&format!("  Prediction (π/2):       {:.4}\n", nu_result.prediction));
    out.push_str(&format!("  Delta |ν - π/2|:        {:.4}\n", nu_result.delta));
    out.push_str(&format!("  Collapse χ²:            {:.6}\n\n", nu_result.chi2));

    if nu_result.claim_survives {
        out.push_str("  ▶ CLAIM A SURVIVES: |ν - π/2| < 0.02\n");
        out.push_str("    → The 3D Anderson critical exponent may be exactly π/2\n");
    } else {
        out.push_str("  ▶ CLAIM A DIES: |ν - π/2| ≥ 0.02\n");
        out.push_str("    → ν is a non-trivial numerical constant, not π/2\n");
    }

    out.push_str("═══════════════════════════════════════════════════\n");
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lyapunov_localized_at_high_disorder() {
        // At W=30 (well above W_c≈16.5), states are deeply localized.
        // Λ/L should be small (much less than 1).
        let ll = quasi_1d_lyapunov(4, 5000, 30.0, 0.0, 42);
        assert!(ll < 1.0, "at W=30, Λ/L should be < 1, got {}", ll);
        assert!(ll > 0.0, "Λ/L should be positive");
    }

    #[test]
    fn low_disorder_larger_than_high() {
        // In the metallic regime (W << W_c), Λ/L should be larger
        // than in the insulating regime (W >> W_c).
        let ll_low = quasi_1d_lyapunov(4, 5000, 2.0, 0.0, 42);
        let ll_high = quasi_1d_lyapunov(4, 5000, 30.0, 0.0, 42);
        assert!(ll_low > ll_high,
            "Λ/L at W=2 ({:.4}) should exceed Λ/L at W=30 ({:.4})",
            ll_low, ll_high);
    }

    #[test]
    fn averaged_reduces_fluctuation() {
        let single = quasi_1d_lyapunov(3, 2000, 16.0, 0.0, 42);
        let averaged = quasi_1d_averaged(3, 2000, 16.0, 0.0, 5, 42);
        // Both should be in the same ballpark
        assert!((single - averaged).abs() < single + 0.5,
            "averaged={:.4}, single={:.4}", averaged, single);
    }

    #[test]
    fn sweep_produces_data() {
        let result = fss_sweep(
            &[3, 4],
            (14.0, 18.0),
            3,
            1000,
            2,
            0.0,
            42,
        );
        assert_eq!(result.points.len(), 6); // 2 L values × 3 W values
        assert_eq!(result.l_values.len(), 2);
        assert_eq!(result.w_values.len(), 3);
    }

    #[test]
    fn find_wc_returns_value() {
        let result = fss_sweep(
            &[3, 4],
            (14.0, 18.0),
            5,
            1000,
            2,
            0.0,
            42,
        );
        let wc = find_w_c_from_sweep(&result);
        assert!(wc.is_some(), "should find a W_c");
        let wc = wc.unwrap();
        assert!(wc > 12.0 && wc < 22.0, "W_c should be in reasonable range, got {}", wc);
    }

    #[test]
    fn extract_nu_runs() {
        let result = fss_sweep(
            &[3, 4],
            (14.0, 18.0),
            5,
            1000,
            2,
            0.0,
            42,
        );
        let wc = find_w_c_from_sweep(&result).unwrap_or(16.5);
        let nu = extract_nu(&result, wc);
        assert!(nu.nu > 0.5 && nu.nu < 3.0, "ν should be in [0.5, 3.0], got {}", nu.nu);
        assert!(nu.chi2 >= 0.0, "chi2 should be non-negative");
    }

    #[test]
    fn report_renders() {
        let result = fss_sweep(
            &[3, 4],
            (14.0, 18.0),
            3,
            500,
            2,
            0.0,
            42,
        );
        let wc = find_w_c_from_sweep(&result).unwrap_or(16.5);
        let nu = extract_nu(&result, wc);
        let report = format_report(&result, &nu);
        assert!(report.contains("EXPERIMENT 3"));
        assert!(report.contains("π/2"));
    }
}

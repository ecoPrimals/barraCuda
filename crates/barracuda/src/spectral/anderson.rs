// SPDX-License-Identifier: AGPL-3.0-only

//! Anderson localization models: 1D, 2D, 3D discrete Schrödinger operators
//!
//! **Why this file is large (~700 lines)**: Single coherent domain—Anderson
//! localization. Contains Hamiltonian construction (1D–4D, correlated disorder),
//! Lyapunov exponent (transfer matrix), level-spacing statistics, Wegner block
//! renormalization, and LcgRng. All algorithms serve the same physics; splitting
//! would fragment a well-defined algorithm family.
//! with random potential, plus transfer-matrix Lyapunov exponent.
//!
//! 1D/2D: all states localized (Abrahams et al. 1979).
//! 3D: genuine metal-insulator transition at W_c ≈ 16.5.
//!
//! Provenance: hotSpring v0.6.0 (Kachkovskiy spectral theory)

use super::sparse::SpectralCsrMatrix;

/// Construct the 1D Anderson Hamiltonian on N sites with periodic boundary
/// conditions: H = -Δ + V, where V_i ~ Uniform[-W/2, W/2].
///
/// Returns (diagonal, off_diagonal) for the tridiagonal representation.
/// The hopping is t = 1, so the clean bandwidth is [-2, 2].
///
/// # Provenance
/// Anderson (1958), Phys. Rev. 109, 1492
pub fn anderson_hamiltonian(n: usize, disorder: f64, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut rng = LcgRng::new(seed);

    let diagonal: Vec<f64> = (0..n).map(|_| disorder * (rng.uniform() - 0.5)).collect();
    let off_diag = vec![-1.0; n - 1];

    (diagonal, off_diag)
}

/// Generate the random potential for the Anderson model (for use with
/// transfer matrix methods that need the raw potential, not the tridiag form).
pub fn anderson_potential(n: usize, disorder: f64, seed: u64) -> Vec<f64> {
    let mut rng = LcgRng::new(seed);
    (0..n).map(|_| disorder * (rng.uniform() - 0.5)).collect()
}

/// Compute the Lyapunov exponent γ(E) via the transfer matrix method.
///
/// For the 1D Schrödinger equation ψ_{n+1} + ψ_{n-1} + V_n ψ_n = E ψ_n,
/// the transfer matrix at site n is:
///   T_n = [[E − V_n, −1], [1, 0]]
///
/// The Lyapunov exponent γ = lim_{N→∞} (1/N) ln ‖T_N ⋯ T_1‖ measures
/// the exponential growth rate. γ > 0 implies localization.
///
/// Uses iterative renormalization to prevent overflow.
///
/// # Known results
/// - Anderson model, E=0, small W: γ(0) ≈ W²/96 (Kappus-Wegner 1981)
/// - Almost-Mathieu, irrational α: γ(E) = max(0, ln|λ|) a.e. (Herman 1983, Avila 2015)
pub fn lyapunov_exponent(potential: &[f64], energy: f64) -> f64 {
    let n = potential.len();
    if n == 0 {
        return 0.0;
    }

    let mut v_prev = 0.0f64;
    let mut v_curr = 1.0f64;
    let mut log_growth = 0.0f64;

    for &v_i in potential {
        let v_next = (energy - v_i) * v_curr - v_prev;
        v_prev = v_curr;
        v_curr = v_next;

        let norm = v_curr.hypot(v_prev);
        if norm > 0.0 {
            log_growth += norm.ln();
            v_curr /= norm;
            v_prev /= norm;
        }
    }

    log_growth / n as f64
}

/// Compute Lyapunov exponent averaged over multiple disorder realizations.
pub fn lyapunov_averaged(
    n_sites: usize,
    disorder: f64,
    energy: f64,
    n_realizations: usize,
    base_seed: u64,
) -> f64 {
    let mut sum = 0.0;
    for r in 0..n_realizations {
        let pot = anderson_potential(n_sites, disorder, base_seed + r as u64 * 1000);
        sum += lyapunov_exponent(&pot, energy);
    }
    sum / n_realizations as f64
}

/// Construct the 2D Anderson Hamiltonian on an Lx × Ly square lattice
/// with open boundary conditions.
///
/// H = -Δ₂D + V, where Δ₂D is the 2D discrete Laplacian (4 nearest
/// neighbors) and V_i ~ Uniform[-W/2, W/2].
///
/// The clean bandwidth is [-4, 4] (hopping t=1, coordination number z=4).
/// With disorder W, spectrum lies in [-4-W/2, 4+W/2].
///
/// Returns a SpectralCsrMatrix of dimension N = Lx × Ly.
///
/// # Provenance
/// Abrahams, Anderson, Licciardello, Ramakrishnan (1979) "Scaling Theory
/// of Localization in an Open and Topologically Disordered System"
/// Phys. Rev. Lett. 42, 673
pub fn anderson_2d(lx: usize, ly: usize, disorder: f64, seed: u64) -> SpectralCsrMatrix {
    let n = lx * ly;
    let mut rng = LcgRng::new(seed);

    let mut row_ptr = Vec::with_capacity(n + 1);
    let mut col_idx = Vec::new();
    let mut values = Vec::new();

    let idx = |ix: usize, iy: usize| -> usize { ix * ly + iy };

    row_ptr.push(0);

    for ix in 0..lx {
        for iy in 0..ly {
            let i = idx(ix, iy);
            let v_i = disorder * (rng.uniform() - 0.5);

            let mut entries: Vec<(usize, f64)> = Vec::new();

            if ix > 0 {
                entries.push((idx(ix - 1, iy), -1.0));
            }
            if iy > 0 {
                entries.push((idx(ix, iy - 1), -1.0));
            }
            entries.push((i, v_i));
            if iy + 1 < ly {
                entries.push((idx(ix, iy + 1), -1.0));
            }
            if ix + 1 < lx {
                entries.push((idx(ix + 1, iy), -1.0));
            }

            entries.sort_by_key(|&(c, _)| c);
            for (c, v) in entries {
                col_idx.push(c);
                values.push(v);
            }
            row_ptr.push(col_idx.len());
        }
    }

    SpectralCsrMatrix {
        n,
        row_ptr,
        col_idx,
        values,
    }
}

/// Construct the clean 2D tight-binding Hamiltonian (no disorder).
pub fn clean_2d_lattice(lx: usize, ly: usize) -> SpectralCsrMatrix {
    anderson_2d(lx, ly, 0.0, 0)
}

/// Construct the 3D Anderson Hamiltonian on an Lx × Ly × Lz cubic lattice
/// with open boundary conditions.
///
/// H = -Δ₃D + V, where Δ₃D is the 3D discrete Laplacian (6 nearest
/// neighbors) and V_i ~ Uniform[-W/2, W/2].
///
/// The clean bandwidth is [-6, 6] (hopping t=1, coordination number z=6).
/// With disorder W, spectrum lies in [-6-W/2, 6+W/2].
///
/// In 3D, a genuine **Anderson metal-insulator transition** exists at
/// critical disorder W_c ≈ 16.5 (band center, orthogonal class):
/// - W < W_c: extended states near band center, localized at band edges
///   → **mobility edge** separates extended from localized states
/// - W > W_c: all states localized
///
/// This is qualitatively different from 1D/2D where all states are
/// localized for any nonzero disorder.
///
/// Returns a SpectralCsrMatrix of dimension N = Lx × Ly × Lz.
///
/// # Provenance
/// Anderson (1958) Phys. Rev. 109, 1492
/// Abrahams, Anderson, Licciardello, Ramakrishnan (1979) Phys. Rev. Lett. 42, 673
/// Slevin & Ohtsuki (1999) Phys. Rev. Lett. 82, 382 [W_c ≈ 16.5]
pub fn anderson_3d(lx: usize, ly: usize, lz: usize, disorder: f64, seed: u64) -> SpectralCsrMatrix {
    let n = lx * ly * lz;
    let mut rng = LcgRng::new(seed);

    let mut row_ptr = Vec::with_capacity(n + 1);
    let mut col_idx = Vec::new();
    let mut values = Vec::new();

    let idx = |ix: usize, iy: usize, iz: usize| -> usize { (ix * ly + iy) * lz + iz };

    row_ptr.push(0);

    for ix in 0..lx {
        for iy in 0..ly {
            for iz in 0..lz {
                let v_i = disorder * (rng.uniform() - 0.5);

                let mut entries: Vec<(usize, f64)> = Vec::new();

                if ix > 0 {
                    entries.push((idx(ix - 1, iy, iz), -1.0));
                }
                if iy > 0 {
                    entries.push((idx(ix, iy - 1, iz), -1.0));
                }
                if iz > 0 {
                    entries.push((idx(ix, iy, iz - 1), -1.0));
                }
                entries.push((idx(ix, iy, iz), v_i));
                if iz + 1 < lz {
                    entries.push((idx(ix, iy, iz + 1), -1.0));
                }
                if iy + 1 < ly {
                    entries.push((idx(ix, iy + 1, iz), -1.0));
                }
                if ix + 1 < lx {
                    entries.push((idx(ix + 1, iy, iz), -1.0));
                }

                entries.sort_by_key(|&(c, _)| c);
                for (c, v) in entries {
                    col_idx.push(c);
                    values.push(v);
                }
                row_ptr.push(col_idx.len());
            }
        }
    }

    SpectralCsrMatrix {
        n,
        row_ptr,
        col_idx,
        values,
    }
}

/// Construct the clean 3D tight-binding Hamiltonian (no disorder).
pub fn clean_3d_lattice(l: usize) -> SpectralCsrMatrix {
    anderson_3d(l, l, l, 0.0, 0)
}

/// Construct the 3D Anderson Hamiltonian with spatially correlated disorder.
///
/// Applies exponential-kernel smoothing to the random potential with
/// correlation length `xi_corr` lattice spacings, then rescales to
/// preserve the target disorder variance W²/12.
///
/// When `xi_corr < 0.01` the smoothing is skipped and the result matches
/// [`anderson_3d`].
///
/// # Provenance
/// Adapted from wetSpring `validate_correlated_disorder.rs` (Feb 2026).
/// Motivated by Méndez-Bermúdez et al. (2014), J. Phys. A 47, 125101.
pub fn anderson_3d_correlated(
    l: usize,
    disorder: f64,
    xi_corr: f64,
    seed: u64,
) -> SpectralCsrMatrix {
    let n = l * l * l;
    let mut rng = LcgRng::new(seed);

    let raw: Vec<f64> = (0..n).map(|_| disorder * (rng.uniform() - 0.5)).collect();

    let potential = if xi_corr < 0.01 {
        raw
    } else {
        let idx_to_xyz = |i: usize| -> (usize, usize, usize) {
            let iz = i % l;
            let iy = (i / l) % l;
            let ix = i / (l * l);
            (ix, iy, iz)
        };

        let reach = (3.0 * xi_corr).ceil() as i64;
        let mut smoothed = vec![0.0; n];
        for (i, s_val) in smoothed.iter_mut().enumerate() {
            let (ix, iy, iz) = idx_to_xyz(i);
            let mut sum = 0.0;
            let mut norm = 0.0;
            for dx in -reach..=reach {
                for dy in -reach..=reach {
                    for dz in -reach..=reach {
                        let jx = ix as i64 + dx;
                        let jy = iy as i64 + dy;
                        let jz = iz as i64 + dz;
                        if (0..l as i64).contains(&jx)
                            && (0..l as i64).contains(&jy)
                            && (0..l as i64).contains(&jz)
                        {
                            let j = (jx as usize * l + jy as usize) * l + jz as usize;
                            let r = ((dx * dx + dy * dy + dz * dz) as f64).sqrt();
                            let kernel = (-r / xi_corr).exp();
                            sum += kernel * raw[j];
                            norm += kernel;
                        }
                    }
                }
            }
            *s_val = sum / norm;
        }

        let var: f64 = {
            let mean = smoothed.iter().sum::<f64>() / n as f64;
            smoothed.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64
        };
        let target_var = disorder * disorder / 12.0;
        let scale = if var > 1e-30 {
            (target_var / var).sqrt()
        } else {
            1.0
        };
        smoothed.iter().map(|v| v * scale).collect()
    };

    let idx = |ix: usize, iy: usize, iz: usize| -> usize { (ix * l + iy) * l + iz };

    let mut row_ptr = Vec::with_capacity(n + 1);
    let mut col_idx = Vec::new();
    let mut values = Vec::new();
    row_ptr.push(0);

    for ix in 0..l {
        for iy in 0..l {
            for iz in 0..l {
                let site = idx(ix, iy, iz);
                let mut entries: Vec<(usize, f64)> = Vec::new();

                if ix > 0 {
                    entries.push((idx(ix - 1, iy, iz), -1.0));
                }
                if iy > 0 {
                    entries.push((idx(ix, iy - 1, iz), -1.0));
                }
                if iz > 0 {
                    entries.push((idx(ix, iy, iz - 1), -1.0));
                }
                entries.push((site, potential[site]));
                if iz + 1 < l {
                    entries.push((idx(ix, iy, iz + 1), -1.0));
                }
                if iy + 1 < l {
                    entries.push((idx(ix, iy + 1, iz), -1.0));
                }
                if ix + 1 < l {
                    entries.push((idx(ix + 1, iy, iz), -1.0));
                }

                entries.sort_by_key(|&(c, _)| c);
                for (c, v) in entries {
                    col_idx.push(c);
                    values.push(v);
                }
                row_ptr.push(col_idx.len());
            }
        }
    }

    SpectralCsrMatrix {
        n,
        row_ptr,
        col_idx,
        values,
    }
}

/// Result of a single disorder-averaged level spacing ratio measurement.
#[derive(Debug, Clone, Copy)]
pub struct AndersonSweepPoint {
    /// Disorder strength W.
    pub w: f64,
    /// Mean level spacing ratio ⟨r⟩ over realizations.
    pub r_mean: f64,
    /// Standard error of the mean.
    pub r_stderr: f64,
}

/// Compute disorder-averaged level spacing ratio ⟨r⟩(W) across a sweep
/// of disorder strengths.
///
/// For each W in `n_w` evenly spaced values from `w_min` to `w_max`,
/// builds `n_realizations` Anderson 3D Hamiltonians of side length `l`,
/// diagonalizes via Lanczos, and computes the mean level spacing ratio
/// with standard error.
///
/// # Provenance
/// Adapted from wetSpring `validate_finite_size_scaling_v2.rs` (Feb 2026).
/// Oganesyan & Huse (2007), Phys. Rev. B 75, 155111.
pub fn anderson_sweep_averaged(
    l: usize,
    w_min: f64,
    w_max: f64,
    n_w: usize,
    n_realizations: usize,
    base_seed: u64,
) -> Vec<AndersonSweepPoint> {
    use super::lanczos::{lanczos, lanczos_eigenvalues};
    use super::stats::level_spacing_ratio;

    let n = l * l * l;
    let mut results = Vec::with_capacity(n_w);

    for wi in 0..n_w {
        let w = if n_w <= 1 {
            w_min
        } else {
            w_min + (w_max - w_min) * wi as f64 / (n_w - 1) as f64
        };

        let mut r_values = Vec::with_capacity(n_realizations);
        for r in 0..n_realizations {
            let seed = base_seed + (wi * 1000 + r * 100 + l * 10) as u64;
            let mat = anderson_3d(l, l, l, w, seed);
            let tri = lanczos(&mat, n, seed);
            let eigs = lanczos_eigenvalues(&tri);
            r_values.push(level_spacing_ratio(&eigs));
        }

        let mean = r_values.iter().sum::<f64>() / n_realizations as f64;
        let variance = if n_realizations > 1 {
            r_values.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (n_realizations - 1) as f64
        } else {
            0.0
        };
        let stderr = (variance / n_realizations as f64).sqrt();

        results.push(AndersonSweepPoint {
            w,
            r_mean: mean,
            r_stderr: stderr,
        });
    }

    results
}

/// Find the critical disorder W_c by linear interpolation of the ⟨r⟩(W)
/// sweep through the GOE–Poisson midpoint.
///
/// Returns `None` if the midpoint is never crossed.
///
/// The standard midpoint is (r_GOE + r_Poisson) / 2 ≈ (0.5307 + 0.3863) / 2 ≈ 0.4585.
pub fn find_w_c(sweep: &[AndersonSweepPoint], midpoint: f64) -> Option<f64> {
    let mut last = None;
    for i in 1..sweep.len() {
        let (w0, r0) = (sweep[i - 1].w, sweep[i - 1].r_mean);
        let (w1, r1) = (sweep[i].w, sweep[i].r_mean);
        if r0 > midpoint && r1 <= midpoint {
            let t = (midpoint - r0) / (r1 - r0);
            last = Some(w0 + t * (w1 - w0));
        }
    }
    last
}

/// Convenience: build a 1D Anderson Hamiltonian and return all eigenvalues.
///
/// Combines `anderson_hamiltonian` + `find_all_eigenvalues` into a single call
/// for the common case of studying spectral statistics.
pub fn anderson_eigenvalues(n: usize, disorder: f64, seed: u64) -> Vec<f64> {
    let (diag, off) = anderson_hamiltonian(n, disorder, seed);
    super::tridiag::find_all_eigenvalues(&diag, &off)
}

/// Construct the 4D Anderson Hamiltonian on an L⁴ hypercubic lattice
/// with open boundary conditions.
///
/// H = -Δ₄D + V, where Δ₄D is the 4D discrete Laplacian (8 nearest
/// neighbors) and V_i ~ Uniform[-W/2, W/2].
///
/// The clean bandwidth is [-8, 8] (hopping t=1, coordination number z=8).
/// With disorder W, spectrum lies in [-8-W/2, 8+W/2].
///
/// In 4D the upper critical dimension d_c=4 coincides with the lattice
/// dimension, producing logarithmic corrections to the metal-insulator
/// transition. Wegner's block renormalization group can be studied directly
/// by coarse-graining blocks of 2⁴ = 16 sites.
///
/// Returns a SpectralCsrMatrix of dimension N = L⁴.
///
/// # Provenance
/// hotSpring Exp 026 (4D Anderson + Wegner block proxy)
/// Wegner (1976) Z. Phys. B 25, 327
pub fn anderson_4d(l: usize, disorder: f64, seed: u64) -> SpectralCsrMatrix {
    let n = l * l * l * l;
    let mut rng = LcgRng::new(seed);

    let mut row_ptr = Vec::with_capacity(n + 1);
    let mut col_idx = Vec::new();
    let mut values = Vec::new();

    let idx =
        |ix: usize, iy: usize, iz: usize, iw: usize| -> usize { ((ix * l + iy) * l + iz) * l + iw };

    row_ptr.push(0);

    for ix in 0..l {
        for iy in 0..l {
            for iz in 0..l {
                for iw in 0..l {
                    let v_i = disorder * (rng.uniform() - 0.5);
                    let mut entries: Vec<(usize, f64)> = Vec::new();

                    if ix > 0 {
                        entries.push((idx(ix - 1, iy, iz, iw), -1.0));
                    }
                    if iy > 0 {
                        entries.push((idx(ix, iy - 1, iz, iw), -1.0));
                    }
                    if iz > 0 {
                        entries.push((idx(ix, iy, iz - 1, iw), -1.0));
                    }
                    if iw > 0 {
                        entries.push((idx(ix, iy, iz, iw - 1), -1.0));
                    }
                    entries.push((idx(ix, iy, iz, iw), v_i));
                    if iw + 1 < l {
                        entries.push((idx(ix, iy, iz, iw + 1), -1.0));
                    }
                    if iz + 1 < l {
                        entries.push((idx(ix, iy, iz + 1, iw), -1.0));
                    }
                    if iy + 1 < l {
                        entries.push((idx(ix, iy + 1, iz, iw), -1.0));
                    }
                    if ix + 1 < l {
                        entries.push((idx(ix + 1, iy, iz, iw), -1.0));
                    }

                    entries.sort_by_key(|&(c, _)| c);
                    for (c, v) in entries {
                        col_idx.push(c);
                        values.push(v);
                    }
                    row_ptr.push(col_idx.len());
                }
            }
        }
    }

    SpectralCsrMatrix {
        n,
        row_ptr,
        col_idx,
        values,
    }
}

/// Construct the clean 4D tight-binding Hamiltonian (no disorder).
pub fn clean_4d_lattice(l: usize) -> SpectralCsrMatrix {
    anderson_4d(l, 0.0, 0)
}

/// Wegner block renormalization for 4D Anderson model.
///
/// Coarse-grains a 4D lattice of side L into blocks of 2⁴ = 16 sites each,
/// returning the effective disorder of the renormalized Hamiltonian.
///
/// The effective on-site energy for each block is the mean of its 16 sites'
/// diagonal elements. The effective hopping between adjacent blocks is the
/// mean of the 2⁴⁻¹ = 8 inter-block bonds per face.
///
/// Returns a SpectralCsrMatrix of dimension (L/2)⁴.
///
/// # Panics
/// Panics if `l` is not even.
pub fn wegner_block_4d(original: &SpectralCsrMatrix, l: usize) -> SpectralCsrMatrix {
    assert!(
        l >= 2 && l.is_multiple_of(2),
        "L must be even and >= 2 for Wegner blocking"
    );
    let l2 = l / 2;
    let n_coarse = l2 * l2 * l2 * l2;

    let fine_idx =
        |ix: usize, iy: usize, iz: usize, iw: usize| -> usize { ((ix * l + iy) * l + iz) * l + iw };
    let coarse_idx = |ix: usize, iy: usize, iz: usize, iw: usize| -> usize {
        ((ix * l2 + iy) * l2 + iz) * l2 + iw
    };

    let mut block_diag = vec![0.0; n_coarse];
    for bx in 0..l2 {
        for by in 0..l2 {
            for bz in 0..l2 {
                for bw in 0..l2 {
                    let mut sum = 0.0;
                    for dx in 0..2 {
                        for dy in 0..2 {
                            for dz in 0..2 {
                                for dw in 0..2 {
                                    let site = fine_idx(
                                        bx * 2 + dx,
                                        by * 2 + dy,
                                        bz * 2 + dz,
                                        bw * 2 + dw,
                                    );
                                    for k in original.row_ptr[site]..original.row_ptr[site + 1] {
                                        if original.col_idx[k] == site {
                                            sum += original.values[k];
                                        }
                                    }
                                }
                            }
                        }
                    }
                    block_diag[coarse_idx(bx, by, bz, bw)] = sum / 16.0;
                }
            }
        }
    }

    let mut row_ptr = Vec::with_capacity(n_coarse + 1);
    let mut col_idx_out = Vec::new();
    let mut values_out = Vec::new();
    row_ptr.push(0);

    for bx in 0..l2 {
        for by in 0..l2 {
            for bz in 0..l2 {
                for bw in 0..l2 {
                    let ci = coarse_idx(bx, by, bz, bw);
                    let mut entries: Vec<(usize, f64)> = Vec::new();

                    if bx > 0 {
                        entries.push((coarse_idx(bx - 1, by, bz, bw), -1.0));
                    }
                    if by > 0 {
                        entries.push((coarse_idx(bx, by - 1, bz, bw), -1.0));
                    }
                    if bz > 0 {
                        entries.push((coarse_idx(bx, by, bz - 1, bw), -1.0));
                    }
                    if bw > 0 {
                        entries.push((coarse_idx(bx, by, bz, bw - 1), -1.0));
                    }
                    entries.push((ci, block_diag[ci]));
                    if bw + 1 < l2 {
                        entries.push((coarse_idx(bx, by, bz, bw + 1), -1.0));
                    }
                    if bz + 1 < l2 {
                        entries.push((coarse_idx(bx, by, bz + 1, bw), -1.0));
                    }
                    if by + 1 < l2 {
                        entries.push((coarse_idx(bx, by + 1, bz, bw), -1.0));
                    }
                    if bx + 1 < l2 {
                        entries.push((coarse_idx(bx + 1, by, bz, bw), -1.0));
                    }

                    entries.sort_by_key(|&(c, _)| c);
                    for (c, v) in entries {
                        col_idx_out.push(c);
                        values_out.push(v);
                    }
                    row_ptr.push(col_idx_out.len());
                }
            }
        }
    }

    SpectralCsrMatrix {
        n: n_coarse,
        row_ptr,
        col_idx: col_idx_out,
        values: values_out,
    }
}

/// LCG RNG for reproducible disorder; used by Anderson models and Lanczos.
pub(crate) struct LcgRng(u64);

impl LcgRng {
    pub(crate) fn new(seed: u64) -> Self {
        Self(seed.wrapping_add(1))
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0
    }

    pub(crate) fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

#[cfg(test)]
#[path = "anderson_tests.rs"]
mod tests;

// SPDX-License-Identifier: AGPL-3.0-or-later
use super::*;
use crate::spectral::GOLDEN_RATIO;

#[test]
fn lyapunov_positive_anderson() {
    let pot = anderson_potential(10_000, 2.0, 42);
    let gamma = lyapunov_exponent(&pot, 0.0);
    assert!(
        gamma > 0.0,
        "γ should be positive for Anderson, got {gamma}"
    );
}

#[test]
fn lyapunov_herman_formula() {
    let n = 50_000;
    let pot: Vec<f64> = (0..n)
        .map(|i| 2.0 * 2.0 * (std::f64::consts::TAU * GOLDEN_RATIO * i as f64).cos())
        .collect();
    let gamma = lyapunov_exponent(&pot, 0.0);
    let expected = (2.0f64).ln();
    assert!(
        (gamma - expected).abs() < 0.05,
        "Herman: γ={gamma:.4}, expected ln(2)={expected:.4}"
    );
}

#[test]
fn anderson_2d_nnz_correct() {
    let l = 10;
    let mat = anderson_2d(l, l, 1.0, 42);
    assert_eq!(mat.n, l * l);
    let expected_nnz = l * l + 2 * (l - 1) * l + 2 * l * (l - 1);
    assert_eq!(
        mat.nnz(),
        expected_nnz,
        "nnz={}, expected={expected_nnz}",
        mat.nnz()
    );
}

#[test]
fn anderson_3d_nnz_correct() {
    let l = 6;
    let mat = anderson_3d(l, l, l, 1.0, 42);
    let n = l * l * l;
    assert_eq!(mat.n, n);
    let expected_nnz = n + 2 * ((l - 1) * l * l + l * (l - 1) * l + l * l * (l - 1));
    assert_eq!(
        mat.nnz(),
        expected_nnz,
        "3D nnz={}, expected={expected_nnz}",
        mat.nnz()
    );
}

#[test]
fn correlated_3d_same_structure_as_uncorrelated() {
    let l = 6;
    let mat_c = anderson_3d_correlated(l, 4.0, 2.0, 42);
    let mat_u = anderson_3d(l, l, l, 4.0, 42);
    assert_eq!(mat_c.n, mat_u.n);
    assert_eq!(mat_c.nnz(), mat_u.nnz());
}

#[test]
fn correlated_3d_small_xi_matches_uncorrelated() {
    let l = 4;
    let mat = anderson_3d_correlated(l, 4.0, 0.001, 42);
    let mat_u = anderson_3d(l, l, l, 4.0, 42);
    // Same RNG sequence, so diagonal entries should match
    for i in 0..mat.n {
        let diag_c = mat.values[mat.row_ptr[i]..mat.row_ptr[i + 1]]
            .iter()
            .zip(&mat.col_idx[mat.row_ptr[i]..mat.row_ptr[i + 1]])
            .find(|(_, &c)| c == i)
            .map(|(&v, _)| v)
            .unwrap();
        let diag_u = mat_u.values[mat_u.row_ptr[i]..mat_u.row_ptr[i + 1]]
            .iter()
            .zip(&mat_u.col_idx[mat_u.row_ptr[i]..mat_u.row_ptr[i + 1]])
            .find(|(_, &c)| c == i)
            .map(|(&v, _)| v)
            .unwrap();
        assert!(
            (diag_c - diag_u).abs() < 1e-10,
            "site {i}: correlated={diag_c}, uncorrelated={diag_u}"
        );
    }
}

#[test]
fn correlated_3d_smooths_potential() {
    let l = 8;
    let mat_u = anderson_3d(l, l, l, 10.0, 99);
    let mat_c = anderson_3d_correlated(l, 10.0, 3.0, 99);

    // Correlated potential should have lower variance of neighbor differences
    fn neighbor_var(mat: &SpectralCsrMatrix) -> f64 {
        let mut diffs = Vec::new();
        for i in 0..mat.n {
            let diag = mat.values[mat.row_ptr[i]..mat.row_ptr[i + 1]]
                .iter()
                .zip(&mat.col_idx[mat.row_ptr[i]..mat.row_ptr[i + 1]])
                .find(|(_, &c)| c == i)
                .map(|(&v, _)| v)
                .unwrap_or(0.0);
            for idx in mat.row_ptr[i]..mat.row_ptr[i + 1] {
                let j = mat.col_idx[idx];
                if j != i {
                    let diag_j = mat.values[mat.row_ptr[j]..mat.row_ptr[j + 1]]
                        .iter()
                        .zip(&mat.col_idx[mat.row_ptr[j]..mat.row_ptr[j + 1]])
                        .find(|(_, &c)| c == j)
                        .map(|(&v, _)| v)
                        .unwrap_or(0.0);
                    diffs.push((diag - diag_j).powi(2));
                }
            }
        }
        diffs.iter().sum::<f64>() / diffs.len() as f64
    }

    let var_u = neighbor_var(&mat_u);
    let var_c = neighbor_var(&mat_c);
    assert!(
        var_c < var_u,
        "correlated neighbor variance {var_c:.4} should be less than uncorrelated {var_u:.4}"
    );
}

#[test]
fn sweep_averaged_produces_decreasing_r() {
    // Very small lattice for speed; r should decrease with increasing W
    let sweep = anderson_sweep_averaged(4, 5.0, 25.0, 3, 2, 42);
    assert_eq!(sweep.len(), 3);
    // At W=5, r should be higher (more GOE-like) than at W=25
    assert!(
        sweep[0].r_mean > sweep[2].r_mean,
        "r at W={:.0} ({:.3}) should exceed r at W={:.0} ({:.3})",
        sweep[0].w,
        sweep[0].r_mean,
        sweep[2].w,
        sweep[2].r_mean
    );
}

#[test]
fn find_w_c_linear_interpolation() {
    let sweep = vec![
        AndersonSweepPoint {
            w: 10.0,
            r_mean: 0.50,
            r_stderr: 0.01,
        },
        AndersonSweepPoint {
            w: 15.0,
            r_mean: 0.46,
            r_stderr: 0.01,
        },
        AndersonSweepPoint {
            w: 20.0,
            r_mean: 0.40,
            r_stderr: 0.01,
        },
    ];
    let midpoint = 0.4585; // (GOE + Poisson) / 2
    let w_c = find_w_c(&sweep, midpoint).unwrap();
    assert!(w_c > 10.0 && w_c < 20.0, "W_c={w_c}");
}

#[test]
fn anderson_4d_dimension_and_nnz() {
    let l = 4;
    let mat = anderson_4d(l, 1.0, 42);
    let n = l * l * l * l;
    assert_eq!(mat.n, n);
    // 8 neighbors in 4D (open BC), diagonal per site
    // bonds along each axis: (l-1)*l^3, times 4 axes, times 2 (symmetric)
    let expected_nnz = n + 2 * 4 * (l - 1) * l * l * l;
    assert_eq!(
        mat.nnz(),
        expected_nnz,
        "4D nnz={}, expected={expected_nnz}",
        mat.nnz()
    );
}

#[test]
fn anderson_4d_symmetric() {
    let l = 3;
    let mat = anderson_4d(l, 5.0, 99);
    for i in 0..mat.n {
        for k in mat.row_ptr[i]..mat.row_ptr[i + 1] {
            let j = mat.col_idx[k];
            let v_ij = mat.values[k];
            if i == j {
                continue;
            }
            let found = (mat.row_ptr[j]..mat.row_ptr[j + 1])
                .find(|&kk| mat.col_idx[kk] == i)
                .map(|kk| mat.values[kk]);
            assert_eq!(
                found,
                Some(v_ij),
                "Missing symmetric entry ({j},{i}) for ({i},{j})={v_ij}"
            );
        }
    }
}

#[test]
fn clean_4d_bandwidth() {
    let l = 3;
    let mat = clean_4d_lattice(l);
    // All diagonal entries should be 0 (no disorder)
    for i in 0..mat.n {
        for k in mat.row_ptr[i]..mat.row_ptr[i + 1] {
            if mat.col_idx[k] == i {
                assert!(
                    mat.values[k].abs() < 1e-15,
                    "clean 4D diagonal [{i}] = {}, expected 0",
                    mat.values[k]
                );
            }
        }
    }
}

#[test]
fn wegner_block_4d_coarsens() {
    let l = 4;
    let mat = anderson_4d(l, 8.0, 42);
    let coarse = wegner_block_4d(&mat, l);
    let l2 = l / 2;
    assert_eq!(coarse.n, l2 * l2 * l2 * l2);
    assert!(coarse.nnz() > 0);
}

#[test]
fn wegner_block_4d_clean_zero_diag() {
    let l = 4;
    let mat = clean_4d_lattice(l);
    let coarse = wegner_block_4d(&mat, l);
    for i in 0..coarse.n {
        for k in coarse.row_ptr[i]..coarse.row_ptr[i + 1] {
            if coarse.col_idx[k] == i {
                assert!(
                    coarse.values[k].abs() < 1e-14,
                    "Wegner clean coarse diagonal [{i}] = {}, expected ~0",
                    coarse.values[k]
                );
            }
        }
    }
}

// SPDX-License-Identifier: AGPL-3.0-or-later
//! Linear algebra handlers for JSON-RPC IPC.
//!
//! Inline-data CPU paths for composition graphs (small N).
//! GPU paths available via `tensor.create` + GPU linalg for large systems.

use super::super::jsonrpc::{INTERNAL_ERROR, INVALID_PARAMS, JsonRpcResponse};
use super::params::{extract_f64_array, extract_matrix};
use serde_json::Value;

/// `linalg.solve` — solve A·x = b via Gaussian elimination with partial pivoting.
///
/// Inline-data CPU path for composition graphs (small N). GPU path
/// available via `tensor.create` + GPU `LinSolve` for large systems.
pub(super) fn linalg_solve(params: &Value, id: Value) -> JsonRpcResponse {
    let Some(matrix) = extract_matrix(params, "matrix") else {
        return JsonRpcResponse::error(
            id,
            INVALID_PARAMS,
            "Missing required param: matrix (2D array)",
        );
    };
    let Some(b) = extract_f64_array(params, "b") else {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Missing required param: b (array)");
    };
    let n = matrix.len();
    if n == 0 {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Matrix must be non-empty");
    }
    if matrix.iter().any(|row| row.len() != n) || b.len() != n {
        return JsonRpcResponse::error(
            id,
            INVALID_PARAMS,
            "Matrix must be square and b must match dimension",
        );
    }
    let mut a: Vec<Vec<f64>> = matrix;
    let mut x = b;
    for col in 0..n {
        let pivot = (col..n).max_by(|&i, &j| {
            a[i][col]
                .abs()
                .partial_cmp(&a[j][col].abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        if let Some(pivot_row) = pivot {
            a.swap(col, pivot_row);
            x.swap(col, pivot_row);
        }
        let diag = a[col][col];
        if diag.abs() < 1e-15 {
            return JsonRpcResponse::error(id, INTERNAL_ERROR, "Singular or near-singular matrix");
        }
        for row in (col + 1)..n {
            let factor = a[row][col] / diag;
            let pivot_row_slice: Vec<f64> = a[col][col..n].to_vec();
            for (k, &pivot_val) in pivot_row_slice.iter().enumerate() {
                a[row][col + k] = factor.mul_add(-pivot_val, a[row][col + k]);
            }
            let x_col = x[col];
            x[row] = (-factor).mul_add(x_col, x[row]);
        }
    }
    for col in (0..n).rev() {
        for row in 0..col {
            let factor = a[row][col] / a[col][col];
            let x_col = x[col];
            x[row] = factor.mul_add(-x_col, x[row]);
        }
        x[col] /= a[col][col];
    }
    JsonRpcResponse::success(id, serde_json::json!({ "result": x }))
}

/// `linalg.eigenvalues` / `stats.eigh` — eigendecomposition of a symmetric matrix via Jacobi.
///
/// Returns both eigenvalues and eigenvectors. The eigenvector matrix V is
/// column-major: column k is the eigenvector for eigenvalue k.
pub(super) fn linalg_eigenvalues(params: &Value, id: Value) -> JsonRpcResponse {
    let Some(matrix) = extract_matrix(params, "matrix") else {
        return JsonRpcResponse::error(
            id,
            INVALID_PARAMS,
            "Missing required param: matrix (2D array)",
        );
    };
    let n = matrix.len();
    if n == 0 {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Matrix must be non-empty");
    }
    if matrix.iter().any(|row| row.len() != n) {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Matrix must be square");
    }
    let mut a: Vec<f64> = matrix.into_iter().flatten().collect();
    let mut v = vec![0.0; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    let max_iter = 100 * n * n;
    for _ in 0..max_iter {
        let mut max_off = 0.0_f64;
        let mut p = 0;
        let mut q = 1;
        for i in 0..n {
            for j in (i + 1)..n {
                if a[i * n + j].abs() > max_off {
                    max_off = a[i * n + j].abs();
                    p = i;
                    q = j;
                }
            }
        }
        if max_off < 1e-12 {
            break;
        }
        let app = a[p * n + p];
        let aqq = a[q * n + q];
        let apq = a[p * n + q];
        let theta = if (app - aqq).abs() < 1e-15 {
            std::f64::consts::FRAC_PI_4
        } else {
            0.5 * (2.0 * apq / (app - aqq)).atan()
        };
        let (sin_t, cos_t) = theta.sin_cos();
        let mut new_a = a.clone();
        for i in 0..n {
            new_a[i * n + p] = cos_t.mul_add(a[i * n + p], sin_t * a[i * n + q]);
            new_a[i * n + q] = (-sin_t).mul_add(a[i * n + p], cos_t * a[i * n + q]);
        }
        a.clone_from(&new_a);
        for j in 0..n {
            new_a[p * n + j] = cos_t.mul_add(a[p * n + j], sin_t * a[q * n + j]);
            new_a[q * n + j] = (-sin_t).mul_add(a[p * n + j], cos_t * a[q * n + j]);
        }
        a = new_a;
        let mut new_v = v.clone();
        for i in 0..n {
            new_v[i * n + p] = cos_t.mul_add(v[i * n + p], sin_t * v[i * n + q]);
            new_v[i * n + q] = (-sin_t).mul_add(v[i * n + p], cos_t * v[i * n + q]);
        }
        v = new_v;
    }
    let eigenvalues: Vec<f64> = (0..n).map(|i| a[i * n + i]).collect();
    let eigenvectors: Vec<Vec<f64>> = (0..n)
        .map(|col| (0..n).map(|row| v[row * n + col]).collect())
        .collect();
    JsonRpcResponse::success(
        id,
        serde_json::json!({ "eigenvalues": eigenvalues, "eigenvectors": eigenvectors, "result": eigenvalues }),
    )
}

/// `linalg.svd` — singular value decomposition A = UΣV^T.
///
/// Inline-data CPU path. Params: `matrix` (2D), `rows`, `cols`.
pub(super) fn linalg_svd(params: &Value, id: Value) -> JsonRpcResponse {
    let Some(matrix) = extract_matrix(params, "matrix") else {
        return JsonRpcResponse::error(
            id,
            INVALID_PARAMS,
            "Missing required param: matrix (2D array)",
        );
    };
    let m = matrix.len();
    if m == 0 {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Matrix must be non-empty");
    }
    let n = matrix[0].len();
    if n == 0 || matrix.iter().any(|row| row.len() != n) {
        return JsonRpcResponse::error(
            id,
            INVALID_PARAMS,
            "All rows must have equal non-zero length",
        );
    }
    let flat: Vec<f64> = matrix.into_iter().flatten().collect();
    match barracuda::ops::linalg::svd::svd_decompose(&flat, m, n) {
        Ok(svd) => {
            let u_2d: Vec<Vec<f64>> = svd.u.chunks(m).map(<[f64]>::to_vec).collect();
            let vt_2d: Vec<Vec<f64>> = svd.vt.chunks(n).map(<[f64]>::to_vec).collect();
            JsonRpcResponse::success(
                id,
                serde_json::json!({ "result": &svd.s, "u": u_2d, "s": svd.s, "vt": vt_2d, "m": m, "n": n }),
            )
        }
        Err(e) => JsonRpcResponse::error(id, INTERNAL_ERROR, format!("SVD failed: {e}")),
    }
}

/// `linalg.batched_tridiag_eigh` — batched tridiagonal symmetric eigendecomposition.
///
/// Solves `n_batches` independent tridiagonal eigenproblems via QL with Wilkinson
/// shifts. CPU inline-data path (GPU dispatch available via `tensor.create` for
/// large batch counts). Absorbed from groundSpring Exp 012.
///
/// Params: `diagonals` (flat `n_batches × n`), `subdiagonals` (flat `n_batches × (n-1)`),
/// `n` (matrix dimension), `n_batches` (number of independent systems).
pub(super) fn linalg_batched_tridiag_eigh(params: &Value, id: Value) -> JsonRpcResponse {
    let Some(diagonals) = extract_f64_array(params, "diagonals") else {
        return JsonRpcResponse::error(
            id,
            INVALID_PARAMS,
            "Missing required param: diagonals (flat f64 array)",
        );
    };
    let Some(subdiagonals) = extract_f64_array(params, "subdiagonals") else {
        return JsonRpcResponse::error(
            id,
            INVALID_PARAMS,
            "Missing required param: subdiagonals (flat f64 array)",
        );
    };
    let Some(n) = params.get("n").and_then(|v| v.as_u64()).map(|v| v as usize) else {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Missing required param: n (integer)");
    };
    let n_batches = params
        .get("n_batches")
        .and_then(|v| v.as_u64())
        .map_or(1, |v| v as usize);

    if n == 0 {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "n must be positive");
    }
    if diagonals.len() != n_batches * n {
        return JsonRpcResponse::error(
            id,
            INVALID_PARAMS,
            format!(
                "diagonals length {} != n_batches({n_batches}) × n({n})",
                diagonals.len()
            ),
        );
    }
    let sub_len = n.saturating_sub(1);
    if subdiagonals.len() != n_batches * sub_len {
        return JsonRpcResponse::error(
            id,
            INVALID_PARAMS,
            format!(
                "subdiagonals length {} != n_batches({n_batches}) × (n-1)({})",
                subdiagonals.len(),
                sub_len
            ),
        );
    }

    let mut all_eigenvalues = Vec::with_capacity(n_batches * n);
    let mut all_eigenvectors = Vec::with_capacity(n_batches * n * n);

    for batch in 0..n_batches {
        let diag_slice = &diagonals[batch * n..(batch + 1) * n];
        let sub_slice = &subdiagonals[batch * sub_len..(batch + 1) * sub_len];

        let (eigenvalues, eigenvectors) = barracuda::special::tridiagonal_ql(diag_slice, sub_slice);

        all_eigenvalues.extend_from_slice(&eigenvalues);
        all_eigenvectors.extend_from_slice(&eigenvectors);
    }

    JsonRpcResponse::success(
        id,
        serde_json::json!({
            "eigenvalues": all_eigenvalues,
            "eigenvectors": all_eigenvectors,
            "n": n,
            "n_batches": n_batches,
        }),
    )
}

/// `linalg.qr` — QR decomposition A = QR (Householder reflections).
///
/// Inline-data CPU path. Params: `matrix` (2D). Requires m >= n.
pub(super) fn linalg_qr(params: &Value, id: Value) -> JsonRpcResponse {
    let Some(matrix) = extract_matrix(params, "matrix") else {
        return JsonRpcResponse::error(
            id,
            INVALID_PARAMS,
            "Missing required param: matrix (2D array)",
        );
    };
    let m = matrix.len();
    if m == 0 {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Matrix must be non-empty");
    }
    let n = matrix[0].len();
    if n == 0 || matrix.iter().any(|row| row.len() != n) {
        return JsonRpcResponse::error(
            id,
            INVALID_PARAMS,
            "All rows must have equal non-zero length",
        );
    }
    let flat: Vec<f64> = matrix.into_iter().flatten().collect();
    match barracuda::ops::linalg::qr::qr_decompose(&flat, m, n) {
        Ok(qr) => {
            let q_2d: Vec<Vec<f64>> = qr.q.chunks(m).map(<[f64]>::to_vec).collect();
            let r_2d: Vec<Vec<f64>> = qr.r.chunks(n).map(<[f64]>::to_vec).collect();
            JsonRpcResponse::success(
                id,
                serde_json::json!({ "result": &q_2d, "q": q_2d, "r": r_2d, "m": m, "n": n }),
            )
        }
        Err(e) => JsonRpcResponse::error(id, INTERNAL_ERROR, format!("QR failed: {e}")),
    }
}

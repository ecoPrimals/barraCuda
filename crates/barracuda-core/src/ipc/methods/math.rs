// SPDX-License-Identifier: AGPL-3.0-or-later
//! Math, activation, noise, and RNG handlers for JSON-RPC IPC.
//!
//! Wires barraCuda's CPU-side math primitives to the IPC surface per
//! `SEMANTIC_METHOD_NAMING_STANDARD.md`. These are lightweight CPU ops
//! suitable for composition graph nodes — GPU tensor ops live in `tensor.rs`.

use super::super::jsonrpc::{INTERNAL_ERROR, INVALID_PARAMS, JsonRpcResponse};
use serde_json::Value;

fn extract_f64_array(params: &Value, key: &str) -> Option<Vec<f64>> {
    params
        .get(key)
        .and_then(|v| v.as_array())
        .map(|arr| arr.iter().filter_map(|v| v.as_f64()).collect())
}

fn extract_f64(params: &Value, key: &str) -> Option<f64> {
    params.get(key).and_then(|v| v.as_f64())
}

/// `math.sigmoid` — element-wise sigmoid on f64 array.
pub(super) fn math_sigmoid(params: &Value, id: Value) -> JsonRpcResponse {
    let Some(data) = extract_f64_array(params, "data") else {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Missing required param: data (array)");
    };
    let result: Vec<f64> = data
        .iter()
        .map(|&x| barracuda::activations::sigmoid(x))
        .collect();
    JsonRpcResponse::success(id, serde_json::json!({ "result": result }))
}

/// `math.log2` — element-wise log base 2 on f64 array.
pub(super) fn math_log2(params: &Value, id: Value) -> JsonRpcResponse {
    let Some(data) = extract_f64_array(params, "data") else {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Missing required param: data (array)");
    };
    let result: Vec<f64> = data.iter().map(|x| x.log2()).collect();
    JsonRpcResponse::success(id, serde_json::json!({ "result": result }))
}

/// `stats.mean` — arithmetic mean of f64 array.
pub(super) fn stats_mean(params: &Value, id: Value) -> JsonRpcResponse {
    let Some(data) = extract_f64_array(params, "data") else {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Missing required param: data (array)");
    };
    let result = barracuda::stats::metrics::mean(&data);
    JsonRpcResponse::success(id, serde_json::json!({ "result": result }))
}

/// `stats.std_dev` — sample standard deviation (Bessel's correction, N-1 denominator).
///
/// Convention: uses **sample** standard deviation `sqrt(Σ(xᵢ - x̄)² / (N-1))`,
/// the unbiased estimator. For population std_dev, divide result by
/// `sqrt(N / (N-1))`. The `convention` field in the response confirms this.
pub(super) fn stats_std_dev(params: &Value, id: Value) -> JsonRpcResponse {
    let Some(data) = extract_f64_array(params, "data") else {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Missing required param: data (array)");
    };
    match barracuda::stats::correlation::std_dev(&data) {
        Ok(result) => JsonRpcResponse::success(
            id,
            serde_json::json!({ "result": result, "convention": "sample", "denominator": "N-1" }),
        ),
        Err(e) => JsonRpcResponse::error(id, INTERNAL_ERROR, format!("std_dev failed: {e}")),
    }
}

/// `stats.variance` — sample variance (Bessel's correction, N-1 denominator).
pub(super) fn stats_variance(params: &Value, id: Value) -> JsonRpcResponse {
    let Some(data) = extract_f64_array(params, "data") else {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Missing required param: data (array)");
    };
    match barracuda::stats::correlation::variance(&data) {
        Ok(result) => JsonRpcResponse::success(
            id,
            serde_json::json!({ "result": result, "convention": "sample", "denominator": "N-1" }),
        ),
        Err(e) => JsonRpcResponse::error(id, INTERNAL_ERROR, format!("variance failed: {e}")),
    }
}

/// `stats.correlation` — Pearson product-moment correlation coefficient.
pub(super) fn stats_correlation(params: &Value, id: Value) -> JsonRpcResponse {
    let Some(x) = extract_f64_array(params, "x") else {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Missing required param: x (array)");
    };
    let Some(y) = extract_f64_array(params, "y") else {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Missing required param: y (array)");
    };
    match barracuda::stats::correlation::pearson_correlation(&x, &y) {
        Ok(result) => JsonRpcResponse::success(id, serde_json::json!({ "result": result })),
        Err(e) => JsonRpcResponse::error(id, INTERNAL_ERROR, format!("correlation failed: {e}")),
    }
}

/// `stats.weighted_mean` — weighted arithmetic mean.
pub(super) fn stats_weighted_mean(params: &Value, id: Value) -> JsonRpcResponse {
    let Some(values) = extract_f64_array(params, "values") else {
        return JsonRpcResponse::error(
            id,
            INVALID_PARAMS,
            "Missing required param: values (array)",
        );
    };
    let Some(weights) = extract_f64_array(params, "weights") else {
        return JsonRpcResponse::error(
            id,
            INVALID_PARAMS,
            "Missing required param: weights (array)",
        );
    };
    if values.len() != weights.len() {
        return JsonRpcResponse::error(
            id,
            INVALID_PARAMS,
            format!(
                "values length {} != weights length {}",
                values.len(),
                weights.len()
            ),
        );
    }
    let total_weight: f64 = weights.iter().sum();
    if total_weight == 0.0 {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Total weight is zero");
    }
    let result: f64 = values.iter().zip(&weights).map(|(v, w)| v * w).sum::<f64>() / total_weight;
    JsonRpcResponse::success(id, serde_json::json!({ "result": result }))
}

/// `noise.perlin2d` — CPU Perlin noise at (x, y).
pub(super) fn noise_perlin2d(params: &Value, id: Value) -> JsonRpcResponse {
    let Some(x) = extract_f64(params, "x") else {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Missing required param: x");
    };
    let Some(y) = extract_f64(params, "y") else {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Missing required param: y");
    };
    let result = barracuda::ops::procedural::perlin_noise::perlin_2d_cpu(x, y);
    JsonRpcResponse::success(id, serde_json::json!({ "result": result }))
}

/// `noise.perlin3d` — CPU 3D Perlin noise at (x, y, z).
///
/// Classic Perlin (2002) with 3D gradient vectors and trilinear interpolation.
/// Guarantees zero at all integer lattice points.
pub(super) fn noise_perlin3d(params: &Value, id: Value) -> JsonRpcResponse {
    let Some(x) = extract_f64(params, "x") else {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Missing required param: x");
    };
    let Some(y) = extract_f64(params, "y") else {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Missing required param: y");
    };
    let Some(z) = extract_f64(params, "z") else {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Missing required param: z");
    };
    let result = barracuda::ops::procedural::perlin_noise::perlin_3d_cpu(x, y, z);
    JsonRpcResponse::success(id, serde_json::json!({ "result": result }))
}

/// `rng.uniform` — generate n uniform random f64 values in [min, max).
pub(super) fn rng_uniform(params: &Value, id: Value) -> JsonRpcResponse {
    #[expect(
        clippy::cast_possible_truncation,
        reason = "n is a user-provided count; truncation on 32-bit is acceptable (capped at u32::MAX items)"
    )]
    let n = params.get("n").and_then(|v| v.as_u64()).unwrap_or(1) as usize;
    let min = extract_f64(params, "min").unwrap_or(0.0);
    let max = extract_f64(params, "max").unwrap_or(1.0);
    let seed = params.get("seed").and_then(|v| v.as_u64()).unwrap_or(0);

    if max <= min {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "max must be > min");
    }

    let result: Vec<f64> = barracuda::rng::uniform_f64_sequence(seed, n)
        .into_iter()
        .map(|v| v.mul_add(max - min, min))
        .collect();
    JsonRpcResponse::success(id, serde_json::json!({ "result": result }))
}

/// `activation.fitts` — Fitts' law movement time prediction.
///
/// Supports two formulations via the `variant` parameter:
/// - `"shannon"` (default): ID = log₂(D/W + 1) — MacKenzie 1992, ISO 9241-411
/// - `"fitts"`: ID = log₂(2D/W) — Fitts' original 1954 formulation
pub(super) fn activation_fitts(params: &Value, id: Value) -> JsonRpcResponse {
    let Some(distance) = extract_f64(params, "distance") else {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Missing required param: distance");
    };
    let Some(width) = extract_f64(params, "width") else {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Missing required param: width");
    };
    if width <= 0.0 {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "width must be > 0");
    }
    let a = extract_f64(params, "a").unwrap_or(0.0);
    let b = extract_f64(params, "b").unwrap_or(0.155);
    let variant = params
        .get("variant")
        .and_then(|v| v.as_str())
        .unwrap_or("shannon");
    let id_bits = match variant {
        "shannon" => (distance / width + 1.0).log2().max(0.0),
        "fitts" => (2.0 * distance / width).log2().max(0.0),
        other => {
            return JsonRpcResponse::error(
                id,
                INVALID_PARAMS,
                format!("Unknown variant: {other}. Expected \"shannon\" or \"fitts\""),
            );
        }
    };
    let mt = b.mul_add(id_bits, a);
    JsonRpcResponse::success(
        id,
        serde_json::json!({ "result": mt, "movement_time": mt, "index_of_difficulty": id_bits, "variant": variant }),
    )
}

/// `activation.hick` — Hick-Hyman law reaction time prediction.
///
/// - Default (`include_no_choice: false`): H = log2(n) — standard information-theoretic form
/// - With `include_no_choice: true`: H = log2(n + 1) — includes the no-go/no-choice stimulus
pub(super) fn activation_hick(params: &Value, id: Value) -> JsonRpcResponse {
    let Some(n_choices) = params.get("n_choices").and_then(|v| v.as_u64()) else {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Missing required param: n_choices");
    };
    if n_choices == 0 {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "n_choices must be > 0");
    }
    let a = extract_f64(params, "a").unwrap_or(0.0);
    let b = extract_f64(params, "b").unwrap_or(0.155);
    let include_no_choice = params
        .get("include_no_choice")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    #[expect(
        clippy::cast_precision_loss,
        reason = "n_choices is user-provided, fits f64 mantissa"
    )]
    let hick_bits = if include_no_choice {
        ((n_choices + 1) as f64).log2()
    } else {
        (n_choices as f64).log2()
    };
    let rt = b.mul_add(hick_bits, a);
    JsonRpcResponse::success(
        id,
        serde_json::json!({ "result": rt, "reaction_time": rt, "information_bits": hick_bits, "include_no_choice": include_no_choice }),
    )
}

// ── Linear algebra (CPU inline-data path) ──────────────────────────────

fn extract_matrix(params: &Value, key: &str) -> Option<Vec<Vec<f64>>> {
    params.get(key).and_then(|v| v.as_array()).map(|rows| {
        rows.iter()
            .filter_map(|row| {
                row.as_array()
                    .map(|cols| cols.iter().filter_map(|c| c.as_f64()).collect())
            })
            .collect()
    })
}

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
                a[row][col + k] -= factor * pivot_val;
            }
            let x_col = x[col];
            x[row] = (-factor).mul_add(x_col, x[row]);
        }
    }
    for col in (0..n).rev() {
        for row in 0..col {
            let factor = a[row][col] / a[col][col];
            let x_col = x[col];
            x[row] -= factor * x_col;
        }
        x[col] /= a[col][col];
    }
    JsonRpcResponse::success(id, serde_json::json!({ "result": x }))
}

/// `linalg.eigenvalues` — eigenvalues of a symmetric matrix via Jacobi iteration.
///
/// Inline-data CPU path for composition graphs (small N). GPU path
/// available via `tensor.create` + GPU `Eigh` for large matrices.
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
    }
    let eigenvalues: Vec<f64> = (0..n).map(|i| a[i * n + i]).collect();
    JsonRpcResponse::success(id, serde_json::json!({ "result": eigenvalues }))
}

// ── Spectral (CPU inline-data path) ────────────────────────────────────

/// `spectral.fft` — 1D complex FFT (Cooley-Tukey radix-2, zero-padded).
///
/// Input: real signal as `data` array. Output: complex spectrum as
/// interleaved `[re0, im0, re1, im1, ...]` plus separate `real`/`imag` arrays.
pub(super) fn spectral_fft(params: &Value, id: Value) -> JsonRpcResponse {
    let Some(data) = extract_f64_array(params, "data") else {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Missing required param: data (array)");
    };
    if data.is_empty() {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "data must be non-empty");
    }
    let n = data.len().next_power_of_two();
    let mut re: Vec<f64> = data;
    re.resize(n, 0.0);
    let mut im = vec![0.0; n];
    fft_in_place(&mut re, &mut im);
    JsonRpcResponse::success(
        id,
        serde_json::json!({ "result": re, "real": re, "imag": im, "n": n }),
    )
}

/// `spectral.power_spectrum` — power spectral density |X(k)|²/N.
pub(super) fn spectral_power_spectrum(params: &Value, id: Value) -> JsonRpcResponse {
    let Some(data) = extract_f64_array(params, "data") else {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Missing required param: data (array)");
    };
    if data.is_empty() {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "data must be non-empty");
    }
    let n = data.len().next_power_of_two();
    let mut re: Vec<f64> = data;
    re.resize(n, 0.0);
    let mut im = vec![0.0; n];
    fft_in_place(&mut re, &mut im);
    let inv_n = 1.0 / n as f64;
    let psd: Vec<f64> = re
        .iter()
        .zip(&im)
        .map(|(&r, &i)| r.mul_add(r, i * i) * inv_n)
        .collect();
    JsonRpcResponse::success(id, serde_json::json!({ "result": psd, "n": n }))
}

/// Cooley-Tukey radix-2 DIT FFT (in-place, power-of-2 length).
fn fft_in_place(re: &mut [f64], im: &mut [f64]) {
    let n = re.len();
    debug_assert!(n.is_power_of_two());
    let mut j = 0;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= n {
        let half = len / 2;
        let angle = -2.0 * std::f64::consts::PI / len as f64;
        let (wn_im, wn_re) = angle.sin_cos();
        let mut i = 0;
        while i < n {
            let mut wr = 1.0;
            let mut wi = 0.0;
            for k in 0..half {
                let tr = wr * re[i + k + half] - wi * im[i + k + half];
                let ti = wr.mul_add(im[i + k + half], wi * re[i + k + half]);
                re[i + k + half] = re[i + k] - tr;
                im[i + k + half] = im[i + k] - ti;
                re[i + k] += tr;
                im[i + k] += ti;
                let new_wr = wr.mul_add(wn_re, -(wi * wn_im));
                wi = wr.mul_add(wn_im, wi * wn_re);
                wr = new_wr;
            }
            i += len;
        }
        len <<= 1;
    }
}

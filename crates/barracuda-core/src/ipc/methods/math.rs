// SPDX-License-Identifier: AGPL-3.0-or-later
//! Math, activation, noise, and RNG handlers for JSON-RPC IPC.
//!
//! Wires barraCuda's CPU-side math primitives to the IPC surface per
//! `SEMANTIC_METHOD_NAMING_STANDARD.md`. These are lightweight CPU ops
//! suitable for composition graph nodes — GPU tensor ops live in `tensor.rs`.

use super::super::jsonrpc::{INTERNAL_ERROR, INVALID_PARAMS, JsonRpcResponse};
use super::params::{extract_f64, extract_f64_array, extract_matrix};
use serde_json::Value;

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

/// `stats.chi_squared` — Pearson's chi-squared goodness-of-fit test.
///
/// Params: `observed` (array), `expected` (array). Returns chi², p-value, and df.
pub(super) fn stats_chi_squared(params: &Value, id: Value) -> JsonRpcResponse {
    let Some(observed) = extract_f64_array(params, "observed") else {
        return JsonRpcResponse::error(
            id,
            INVALID_PARAMS,
            "Missing required param: observed (array)",
        );
    };
    let Some(expected) = extract_f64_array(params, "expected") else {
        return JsonRpcResponse::error(
            id,
            INVALID_PARAMS,
            "Missing required param: expected (array)",
        );
    };
    match barracuda::special::chi_squared::chi_squared_test(&observed, &expected) {
        Ok((chi2, p_value, df)) => JsonRpcResponse::success(
            id,
            serde_json::json!({ "result": chi2, "chi_squared": chi2, "p_value": p_value, "df": df }),
        ),
        Err(e) => JsonRpcResponse::error(id, INTERNAL_ERROR, format!("chi_squared failed: {e}")),
    }
}

/// `stats.anova_oneway` — one-way ANOVA F-test.
///
/// Params: `groups` (array of arrays). Returns F-statistic, p-value, df_between, df_within.
pub(super) fn stats_anova_oneway(params: &Value, id: Value) -> JsonRpcResponse {
    let Some(groups_val) = params.get("groups").and_then(|v| v.as_array()) else {
        return JsonRpcResponse::error(
            id,
            INVALID_PARAMS,
            "Missing required param: groups (array of arrays)",
        );
    };
    let groups: Vec<Vec<f64>> = groups_val
        .iter()
        .filter_map(|g| {
            g.as_array()
                .map(|arr| arr.iter().filter_map(|v| v.as_f64()).collect())
        })
        .collect();
    if groups.len() < 2 {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Need at least 2 groups");
    }
    let n_total: usize = groups.iter().map(Vec::len).sum();
    let k = groups.len();
    if n_total <= k {
        return JsonRpcResponse::error(
            id,
            INVALID_PARAMS,
            "Total observations must exceed number of groups",
        );
    }
    if groups.iter().any(|g| g.is_empty()) {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "All groups must be non-empty");
    }

    let grand_mean: f64 = groups.iter().flatten().sum::<f64>() / n_total as f64;

    let ss_between: f64 = groups
        .iter()
        .map(|g| {
            let g_mean: f64 = g.iter().sum::<f64>() / g.len() as f64;
            g.len() as f64 * (g_mean - grand_mean).powi(2)
        })
        .sum();

    let ss_within: f64 = groups
        .iter()
        .map(|g| {
            let g_mean: f64 = g.iter().sum::<f64>() / g.len() as f64;
            g.iter().map(|x| (x - g_mean).powi(2)).sum::<f64>()
        })
        .sum();

    let df_between = k - 1;
    let df_within = n_total - k;
    let ms_between = ss_between / df_between as f64;
    let ms_within = ss_within / df_within as f64;

    let f_stat = if ms_within.abs() < 1e-15 {
        if ms_between.abs() < 1e-15 {
            0.0
        } else {
            f64::INFINITY
        }
    } else {
        ms_between / ms_within
    };

    let p_value = f_distribution_sf(f_stat, df_between as f64, df_within as f64);

    JsonRpcResponse::success(
        id,
        serde_json::json!({
            "result": f_stat,
            "f_statistic": f_stat,
            "p_value": p_value,
            "df_between": df_between,
            "df_within": df_within,
        }),
    )
}

/// Survival function (1 - CDF) of the F-distribution via the regularized
/// incomplete beta function: P(F > x | d1, d2) = I(d2/(d2+d1*x), d2/2, d1/2).
fn f_distribution_sf(x: f64, d1: f64, d2: f64) -> f64 {
    if x <= 0.0 {
        return 1.0;
    }
    let z = d2 / d1.mul_add(x, d2);
    regularized_incomplete_beta(z, d2 / 2.0, d1 / 2.0)
}

/// Regularized incomplete beta function I_x(a,b) via continued fraction (Lentz's method).
fn regularized_incomplete_beta(x: f64, a: f64, b: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }
    // Use symmetry relation when x > (a+1)/(a+b+2) for better convergence
    if x > (a + 1.0) / (a + b + 2.0) {
        return 1.0 - regularized_incomplete_beta(1.0 - x, b, a);
    }
    let ln_prefix = a.mul_add(x.ln(), b * (1.0 - x).ln())
        - a.ln()
        - barracuda::special::ln_beta(a, b).unwrap_or(0.0);
    let prefix = ln_prefix.exp();
    betacf(x, a, b) * prefix
}

/// Continued fraction for incomplete beta (modified Lentz's algorithm).
fn betacf(x: f64, a: f64, b: f64) -> f64 {
    const MAX_ITER: usize = 200;
    const EPS: f64 = 3e-14;
    const TINY: f64 = 1e-30;

    let mut c = 1.0;
    let mut d = (1.0 - (a + b) * x / (a + 1.0)).recip().max(TINY);
    let mut h = d;

    for m in 1..=MAX_ITER {
        let m_f64 = m as f64;

        // Even step
        let am2 = 2.0f64.mul_add(m_f64, a);
        let num_even = m_f64 * (b - m_f64) * x / ((am2 - 1.0) * am2);
        d = (num_even.mul_add(d, 1.0)).recip().max(TINY);
        c = (num_even / c + 1.0).max(TINY);
        h *= d * c;

        // Odd step
        let num_odd = -((a + m_f64) * (a + b + m_f64)) * x / (am2 * (am2 + 1.0));
        d = (num_odd.mul_add(d, 1.0)).recip().max(TINY);
        c = (num_odd / c + 1.0).max(TINY);
        let delta = d * c;
        h *= delta;

        if (delta - 1.0).abs() < EPS {
            break;
        }
    }
    h
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

/// `activation.softmax` — element-wise softmax on f64 array (exp-normalize, CPU).
pub(super) fn activation_softmax(params: &Value, id: Value) -> JsonRpcResponse {
    let Some(data) = extract_f64_array(params, "data") else {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Missing required param: data (array)");
    };
    if data.is_empty() {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "data must be non-empty");
    }
    let max_val = data.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = data.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f64 = exps.iter().sum();
    let result: Vec<f64> = exps.iter().map(|e| e / sum).collect();
    JsonRpcResponse::success(id, serde_json::json!({ "result": result }))
}

/// `activation.gelu` — element-wise GELU on f64 array.
pub(super) fn activation_gelu(params: &Value, id: Value) -> JsonRpcResponse {
    let Some(data) = extract_f64_array(params, "data") else {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Missing required param: data (array)");
    };
    let result = barracuda::activations::gelu_batch(&data);
    JsonRpcResponse::success(id, serde_json::json!({ "result": result }))
}

// ── New statistics handlers (Sprint 49) ─────────────────────────────────

/// `stats.shannon` — Shannon entropy H' = −Σ pᵢ ln(pᵢ).
///
/// Accepts either raw `counts` (normalized internally) or pre-normalized
/// `frequencies`. Returns natural-log entropy (nats).
pub(super) fn stats_shannon(params: &Value, id: Value) -> JsonRpcResponse {
    if let Some(freqs) = extract_f64_array(params, "frequencies") {
        let h = barracuda::stats::shannon_from_frequencies(&freqs);
        return JsonRpcResponse::success(id, serde_json::json!({ "result": h, "unit": "nats" }));
    }
    let Some(counts) = extract_f64_array(params, "counts") else {
        return JsonRpcResponse::error(
            id,
            INVALID_PARAMS,
            "Missing required param: counts or frequencies (array)",
        );
    };
    let h = barracuda::stats::shannon(&counts);
    JsonRpcResponse::success(id, serde_json::json!({ "result": h, "unit": "nats" }))
}

/// `stats.covariance` — sample covariance of two vectors.
pub(super) fn stats_covariance(params: &Value, id: Value) -> JsonRpcResponse {
    let Some(x) = extract_f64_array(params, "x") else {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Missing required param: x (array)");
    };
    let Some(y) = extract_f64_array(params, "y") else {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Missing required param: y (array)");
    };
    match barracuda::stats::covariance(&x, &y) {
        Ok(cov) => JsonRpcResponse::success(
            id,
            serde_json::json!({ "result": cov, "convention": "sample", "denominator": "N-1" }),
        ),
        Err(e) => JsonRpcResponse::error(id, INTERNAL_ERROR, format!("covariance failed: {e}")),
    }
}

/// `stats.spearman` — Spearman rank correlation coefficient.
pub(super) fn stats_spearman(params: &Value, id: Value) -> JsonRpcResponse {
    let Some(x) = extract_f64_array(params, "x") else {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Missing required param: x (array)");
    };
    let Some(y) = extract_f64_array(params, "y") else {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Missing required param: y (array)");
    };
    match barracuda::stats::spearman_correlation(&x, &y) {
        Ok(rho) => JsonRpcResponse::success(id, serde_json::json!({ "result": rho })),
        Err(e) => JsonRpcResponse::error(id, INTERNAL_ERROR, format!("spearman failed: {e}")),
    }
}

/// `stats.fit_linear` — simple linear regression y = a·x + b.
///
/// Returns slope, intercept, R², and RMSE.
pub(super) fn stats_fit_linear(params: &Value, id: Value) -> JsonRpcResponse {
    let Some(x) = extract_f64_array(params, "x") else {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Missing required param: x (array)");
    };
    let Some(y) = extract_f64_array(params, "y") else {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Missing required param: y (array)");
    };
    match barracuda::stats::fit_linear(&x, &y) {
        Some(fit) => JsonRpcResponse::success(
            id,
            serde_json::json!({
                "slope": fit.params[0],
                "intercept": fit.params[1],
                "r_squared": fit.r_squared,
                "rmse": fit.rmse,
                "result": { "slope": fit.params[0], "intercept": fit.params[1] },
            }),
        ),
        None => JsonRpcResponse::error(
            id,
            INVALID_PARAMS,
            "Linear fit failed (need ≥2 points with non-degenerate x)",
        ),
    }
}

/// `ode.step` — stateless RK4 integration of a linear ODE system (Path A).
///
/// Accepts the current state, linear system matrix A and forcing vector b
/// (dy/dt = A*y + b), time step, and number of steps. Returns the final state.
/// The client manages snapshots for multi-call workflows.
///
/// Params: `state` (array, n_vars), `a` (matrix n_vars×n_vars),
///         `b` (array n_vars, optional, default zeros),
///         `dt` (number), `n_steps` (integer, default 1),
///         `t0` (number, default 0.0).
pub(super) fn ode_step(params: &Value, id: Value) -> JsonRpcResponse {
    let Some(state) = extract_f64_array(params, "state") else {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Missing required param: state (array)");
    };
    let Some(a_matrix) = extract_matrix(params, "a") else {
        return JsonRpcResponse::error(
            id,
            INVALID_PARAMS,
            "Missing required param: a (n_vars × n_vars matrix)",
        );
    };

    let n_vars = state.len();
    if n_vars == 0 {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "state must be non-empty");
    }
    if a_matrix.len() != n_vars {
        return JsonRpcResponse::error(
            id,
            INVALID_PARAMS,
            format!(
                "a: row count ({}) must equal state length ({n_vars})",
                a_matrix.len()
            ),
        );
    }
    for (i, row) in a_matrix.iter().enumerate() {
        if row.len() != n_vars {
            return JsonRpcResponse::error(
                id,
                INVALID_PARAMS,
                format!(
                    "a[{i}]: column count ({}) must equal state length ({n_vars})",
                    row.len()
                ),
            );
        }
    }

    let b = match extract_f64_array(params, "b") {
        Some(bv) if bv.len() == n_vars => bv,
        Some(bv) => {
            return JsonRpcResponse::error(
                id,
                INVALID_PARAMS,
                format!(
                    "b: length ({}) must equal state length ({n_vars})",
                    bv.len()
                ),
            );
        }
        None => vec![0.0; n_vars],
    };

    let dt = match params.get("dt").and_then(|v| v.as_f64()) {
        Some(d) if d > 0.0 => d,
        Some(_) => {
            return JsonRpcResponse::error(id, INVALID_PARAMS, "dt must be > 0");
        }
        None => {
            return JsonRpcResponse::error(
                id,
                INVALID_PARAMS,
                "Missing required param: dt (positive number)",
            );
        }
    };
    #[expect(clippy::cast_possible_truncation, reason = "n_steps is a step count")]
    let n_steps = params.get("n_steps").and_then(|v| v.as_u64()).unwrap_or(1) as usize;
    let t0 = params.get("t0").and_then(|v| v.as_f64()).unwrap_or(0.0);

    let deriv = |_t: f64, y: &[f64]| -> Vec<f64> {
        let mut dy = b.clone();
        for (i, row) in a_matrix.iter().enumerate() {
            for (j, &a_ij) in row.iter().enumerate() {
                dy[i] = a_ij.mul_add(y[j], dy[i]);
            }
        }
        dy
    };

    let mut y = state;
    let mut t = t0;

    for _ in 0..n_steps {
        let k1 = deriv(t, &y);
        let y2: Vec<f64> = y
            .iter()
            .zip(&k1)
            .map(|(&yi, &k)| (0.5 * dt).mul_add(k, yi))
            .collect();
        let t_mid = 0.5f64.mul_add(dt, t);
        let k2 = deriv(t_mid, &y2);
        let y3: Vec<f64> = y
            .iter()
            .zip(&k2)
            .map(|(&yi, &k)| (0.5 * dt).mul_add(k, yi))
            .collect();
        let k3 = deriv(t_mid, &y3);
        let y4: Vec<f64> = y
            .iter()
            .zip(&k3)
            .map(|(&yi, &k)| dt.mul_add(k, yi))
            .collect();
        let k4 = deriv(t + dt, &y4);

        let sixth = 1.0 / 6.0;
        for i in 0..n_vars {
            y[i] = (dt * sixth).mul_add(
                2.0f64.mul_add(k3[i], 2.0f64.mul_add(k2[i], k1[i])) + k4[i],
                y[i],
            );
        }
        t += dt;
    }

    JsonRpcResponse::success(
        id,
        serde_json::json!({ "state": y, "t_final": t, "n_steps": n_steps }),
    )
}

/// `stats.empirical_spectral_density` — eigenvalue histogram (normalized).
///
/// Params: `eigenvalues` (array), `n_bins` (integer, default 50).
pub(super) fn stats_empirical_spectral_density(params: &Value, id: Value) -> JsonRpcResponse {
    let Some(eigenvalues) = extract_f64_array(params, "eigenvalues") else {
        return JsonRpcResponse::error(
            id,
            INVALID_PARAMS,
            "Missing required param: eigenvalues (array)",
        );
    };
    #[expect(clippy::cast_possible_truncation, reason = "n_bins is a bin count")]
    let n_bins = params.get("n_bins").and_then(|v| v.as_u64()).unwrap_or(50) as usize;
    let (centers, density) = barracuda::stats::empirical_spectral_density(&eigenvalues, n_bins);
    JsonRpcResponse::success(
        id,
        serde_json::json!({ "result": &density, "bin_centers": centers, "density": density, "n_bins": n_bins }),
    )
}

// ── Diversity / ecology statistics (Sprint 72 — spring absorption) ───────

/// `stats.simpson` — Simpson diversity index D = 1 − Σ(pᵢ²).
///
/// Accepts `counts` (raw abundances). Returns Simpson diversity [0, 1).
pub(super) fn stats_simpson(params: &Value, id: Value) -> JsonRpcResponse {
    let Some(counts) = extract_f64_array(params, "counts") else {
        return JsonRpcResponse::error(
            id,
            INVALID_PARAMS,
            "Missing required param: counts (array)",
        );
    };
    let d = barracuda::stats::simpson(&counts);
    JsonRpcResponse::success(id, serde_json::json!({ "result": d, "index": "simpson" }))
}

/// `stats.bray_curtis` — Bray-Curtis dissimilarity between two sample vectors.
///
/// Accepts `a` and `b` (equal-length abundance vectors). Returns distance [0, 1].
pub(super) fn stats_bray_curtis(params: &Value, id: Value) -> JsonRpcResponse {
    let Some(a) = extract_f64_array(params, "a") else {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Missing required param: a (array)");
    };
    let Some(b) = extract_f64_array(params, "b") else {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Missing required param: b (array)");
    };
    if a.len() != b.len() {
        return JsonRpcResponse::error(
            id,
            INVALID_PARAMS,
            format!("length mismatch: a={}, b={}", a.len(), b.len()),
        );
    }
    let d = barracuda::stats::bray_curtis(&a, &b);
    JsonRpcResponse::success(id, serde_json::json!({ "result": d, "metric": "bray_curtis" }))
}

/// `stats.hill` — Hill function (sigmoidal dose-response): x^n / (k^n + x^n).
///
/// Accepts `x` (concentration/input), `k` (half-max), `n` (Hill coefficient).
pub(super) fn stats_hill(params: &Value, id: Value) -> JsonRpcResponse {
    let Some(x) = params.get("x").and_then(|v| v.as_f64()) else {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Missing required param: x (f64)");
    };
    let Some(k) = params.get("k").and_then(|v| v.as_f64()) else {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Missing required param: k (f64)");
    };
    let Some(n) = params.get("n").and_then(|v| v.as_f64()) else {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Missing required param: n (f64)");
    };
    let result = barracuda::stats::hill(x, k, n);
    JsonRpcResponse::success(id, serde_json::json!({ "result": result, "function": "hill" }))
}

// ── Regression models (Sprint 73 — spring absorption) ────────────────────

/// `stats.fit_quadratic` — fit y = a·x² + b·x + c via normal equations.
pub(super) fn stats_fit_quadratic(params: &Value, id: Value) -> JsonRpcResponse {
    let Some(x) = extract_f64_array(params, "x") else {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Missing required param: x (array)");
    };
    let Some(y) = extract_f64_array(params, "y") else {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Missing required param: y (array)");
    };
    match barracuda::stats::fit_quadratic(&x, &y) {
        Some(fit) => JsonRpcResponse::success(
            id,
            serde_json::json!({
                "model": "quadratic",
                "params": fit.params,
                "r_squared": fit.r_squared,
                "rmse": fit.rmse,
            }),
        ),
        None => JsonRpcResponse::error(
            id,
            INVALID_PARAMS,
            "Quadratic fit failed (need ≥3 points)",
        ),
    }
}

/// `stats.fit_exponential` — fit y = a·exp(b·x) via log-linearization.
pub(super) fn stats_fit_exponential(params: &Value, id: Value) -> JsonRpcResponse {
    let Some(x) = extract_f64_array(params, "x") else {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Missing required param: x (array)");
    };
    let Some(y) = extract_f64_array(params, "y") else {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Missing required param: y (array)");
    };
    match barracuda::stats::fit_exponential(&x, &y) {
        Some(fit) => JsonRpcResponse::success(
            id,
            serde_json::json!({
                "model": "exponential",
                "params": fit.params,
                "r_squared": fit.r_squared,
                "rmse": fit.rmse,
            }),
        ),
        None => JsonRpcResponse::error(
            id,
            INVALID_PARAMS,
            "Exponential fit failed (need ≥2 positive y values)",
        ),
    }
}

/// `stats.fit_logarithmic` — fit y = a·ln(x) + b via linearization.
pub(super) fn stats_fit_logarithmic(params: &Value, id: Value) -> JsonRpcResponse {
    let Some(x) = extract_f64_array(params, "x") else {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Missing required param: x (array)");
    };
    let Some(y) = extract_f64_array(params, "y") else {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Missing required param: y (array)");
    };
    match barracuda::stats::fit_logarithmic(&x, &y) {
        Some(fit) => JsonRpcResponse::success(
            id,
            serde_json::json!({
                "model": "logarithmic",
                "params": fit.params,
                "r_squared": fit.r_squared,
                "rmse": fit.rmse,
            }),
        ),
        None => JsonRpcResponse::error(
            id,
            INVALID_PARAMS,
            "Logarithmic fit failed (need ≥2 positive x values)",
        ),
    }
}

// ── Ecology / rarefaction (Sprint 73) ────────────────────────────────────

/// `stats.rarefaction_curve` — expected species richness E[S] at subsampled depths.
///
/// Accepts `counts` (species abundances) and `depths` (subsampling depths).
pub(super) fn stats_rarefaction_curve(params: &Value, id: Value) -> JsonRpcResponse {
    let Some(counts) = extract_f64_array(params, "counts") else {
        return JsonRpcResponse::error(
            id,
            INVALID_PARAMS,
            "Missing required param: counts (array)",
        );
    };
    let Some(depths) = extract_f64_array(params, "depths") else {
        return JsonRpcResponse::error(
            id,
            INVALID_PARAMS,
            "Missing required param: depths (array)",
        );
    };
    let curve = barracuda::stats::rarefaction_curve(&counts, &depths);
    JsonRpcResponse::success(
        id,
        serde_json::json!({ "result": curve, "depths": depths }),
    )
}

// ── Special functions (Sprint 73) ────────────────────────────────────────

/// `stats.gamma_cdf` — Gamma CDF P(X ≤ x) for X ~ Gamma(alpha, beta).
///
/// Uses the regularized incomplete gamma function.
pub(super) fn stats_gamma_cdf(params: &Value, id: Value) -> JsonRpcResponse {
    let Some(x) = params.get("x").and_then(|v| v.as_f64()) else {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Missing required param: x (f64)");
    };
    let Some(alpha) = params.get("alpha").and_then(|v| v.as_f64()) else {
        return JsonRpcResponse::error(
            id,
            INVALID_PARAMS,
            "Missing required param: alpha (f64)",
        );
    };
    let beta = params.get("beta").and_then(|v| v.as_f64()).unwrap_or(1.0);
    if x <= 0.0 {
        return JsonRpcResponse::success(id, serde_json::json!({ "result": 0.0 }));
    }
    match barracuda::special::regularized_gamma_p(alpha, x / beta) {
        Ok(p) => JsonRpcResponse::success(id, serde_json::json!({ "result": p })),
        Err(e) => {
            JsonRpcResponse::error(id, INTERNAL_ERROR, format!("gamma_cdf failed: {e}"))
        }
    }
}

/// `stats.gamma_fit` — fit Gamma(α,β) parameters via Thom (1958) MLE approximation.
///
/// Accepts `data` (positive observations). Returns alpha and beta.
pub(super) fn stats_gamma_fit(params: &Value, id: Value) -> JsonRpcResponse {
    let Some(data) = extract_f64_array(params, "data") else {
        return JsonRpcResponse::error(
            id,
            INVALID_PARAMS,
            "Missing required param: data (array)",
        );
    };
    let positive: Vec<f64> = data.iter().copied().filter(|&x| x > 0.0).collect();
    let n = positive.len();
    if n < 3 {
        return JsonRpcResponse::error(
            id,
            INVALID_PARAMS,
            "Need ≥3 positive observations for gamma MLE",
        );
    }
    let nf = n as f64;
    let mean_val: f64 = positive.iter().sum::<f64>() / nf;
    let log_mean: f64 = positive.iter().map(|x| x.ln()).sum::<f64>() / nf;
    let a_param = mean_val.ln() - log_mean;
    if a_param <= 0.0 {
        return JsonRpcResponse::error(
            id,
            INVALID_PARAMS,
            "Gamma MLE: A ≤ 0 (data may be degenerate)",
        );
    }
    let alpha = (1.0 / (4.0 * a_param)) * (1.0 + (a_param.mul_add(4.0 / 3.0, 1.0)).sqrt());
    let beta = mean_val / alpha;
    JsonRpcResponse::success(
        id,
        serde_json::json!({ "alpha": alpha, "beta": beta, "n_positive": n }),
    )
}

// ── Signal processing (Sprint 73 — healthSpring absorption) ──────────────

/// `signal.detect_peaks` — find local maxima with minimum distance and height.
///
/// Accepts `signal` (f64 array), `distance` (min samples between peaks),
/// optional `min_height`, optional `min_prominence`.
pub(super) fn signal_detect_peaks(params: &Value, id: Value) -> JsonRpcResponse {
    let Some(signal) = extract_f64_array(params, "signal") else {
        return JsonRpcResponse::error(
            id,
            INVALID_PARAMS,
            "Missing required param: signal (array)",
        );
    };
    let distance = params
        .get("distance")
        .and_then(|v| v.as_u64())
        .unwrap_or(1) as usize;
    let min_height = params.get("min_height").and_then(|v| v.as_f64());
    let min_prominence = params.get("min_prominence").and_then(|v| v.as_f64());
    let peaks =
        barracuda::ops::peak_detect_f64::find_peaks_cpu(&signal, distance, min_height, min_prominence);
    let indices: Vec<usize> = peaks.iter().map(|p| p.index).collect();
    let heights: Vec<f64> = peaks.iter().map(|p| p.height).collect();
    JsonRpcResponse::success(
        id,
        serde_json::json!({
            "indices": indices,
            "heights": heights,
            "count": peaks.len(),
        }),
    )
}

/// `signal.bandpass` — frequency-domain bandpass filter (zeros outside [low, high] Hz).
///
/// Accepts `signal`, `sample_rate` (Hz), `low_hz`, `high_hz`.
pub(super) fn signal_bandpass(params: &Value, id: Value) -> JsonRpcResponse {
    let Some(signal) = extract_f64_array(params, "signal") else {
        return JsonRpcResponse::error(
            id,
            INVALID_PARAMS,
            "Missing required param: signal (array)",
        );
    };
    let Some(sample_rate) = params.get("sample_rate").and_then(|v| v.as_f64()) else {
        return JsonRpcResponse::error(
            id,
            INVALID_PARAMS,
            "Missing required param: sample_rate (f64)",
        );
    };
    let Some(low_hz) = params.get("low_hz").and_then(|v| v.as_f64()) else {
        return JsonRpcResponse::error(
            id,
            INVALID_PARAMS,
            "Missing required param: low_hz (f64)",
        );
    };
    let Some(high_hz) = params.get("high_hz").and_then(|v| v.as_f64()) else {
        return JsonRpcResponse::error(
            id,
            INVALID_PARAMS,
            "Missing required param: high_hz (f64)",
        );
    };
    if sample_rate <= 0.0 || low_hz < 0.0 || high_hz <= low_hz {
        return JsonRpcResponse::error(
            id,
            INVALID_PARAMS,
            "Invalid filter params: need sample_rate > 0, 0 ≤ low_hz < high_hz",
        );
    }
    let filtered = bandpass_filter_cpu(&signal, sample_rate, low_hz, high_hz);
    JsonRpcResponse::success(id, serde_json::json!({ "result": filtered }))
}

/// CPU frequency-domain bandpass: FFT → zero outside band → IFFT.
fn bandpass_filter_cpu(signal: &[f64], fs: f64, low_hz: f64, high_hz: f64) -> Vec<f64> {
    let n = signal.len();
    if n == 0 {
        return vec![];
    }
    let n_freq = n / 2 + 1;
    let mut re = vec![0.0; n_freq];
    let mut im = vec![0.0; n_freq];

    for k in 0..n_freq {
        let mut sum_re = 0.0;
        let mut sum_im = 0.0;
        let angle_base = -2.0 * std::f64::consts::PI * (k as f64) / (n as f64);
        for (j, &s) in signal.iter().enumerate() {
            let angle = angle_base * (j as f64);
            sum_re += s * angle.cos();
            sum_im += s * angle.sin();
        }
        let freq = (k as f64) * fs / (n as f64);
        if freq >= low_hz && freq <= high_hz {
            re[k] = sum_re;
            im[k] = sum_im;
        }
    }

    let mut output = vec![0.0; n];
    let norm = 1.0 / (n as f64);
    for (j, out) in output.iter_mut().enumerate() {
        let mut sum = 0.0;
        for k in 0..n_freq {
            let angle = 2.0 * std::f64::consts::PI * (k as f64) * (j as f64) / (n as f64);
            let contrib = re[k].mul_add(angle.cos(), -(im[k] * angle.sin()));
            sum += contrib;
            if k > 0 && k < n_freq - 1 {
                sum += contrib;
            }
        }
        *out = sum * norm;
    }
    output
}

/// `signal.derivative` — 5-point derivative filter (Pan-Tompkins).
///
/// d[i] = (-x[i-2] - 2*x[i-1] + 2*x[i+1] + x[i+2]) / 8
pub(super) fn signal_derivative(params: &Value, id: Value) -> JsonRpcResponse {
    let Some(signal) = extract_f64_array(params, "signal") else {
        return JsonRpcResponse::error(
            id,
            INVALID_PARAMS,
            "Missing required param: signal (array)",
        );
    };
    let n = signal.len();
    let mut d = vec![0.0; n];
    for i in 2..n.saturating_sub(2) {
        d[i] = (2.0f64.mul_add(
            signal[i + 1],
            2.0f64.mul_add(-signal[i - 1], -signal[i - 2]),
        ) + signal[i + 2])
            / 8.0;
    }
    JsonRpcResponse::success(id, serde_json::json!({ "result": d }))
}

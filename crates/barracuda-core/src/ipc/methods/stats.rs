// SPDX-License-Identifier: AGPL-3.0-or-later
//! Statistics IPC handlers (`stats.*` namespace).
//!
//! Covers descriptive statistics, hypothesis tests, regression models, diversity
//! indices, rarefaction, and special functions. All handlers validate inputs
//! before computation and return structured JSON responses.

use super::super::jsonrpc::{INTERNAL_ERROR, INVALID_PARAMS, JsonRpcResponse};
use super::params::extract_f64_array;
use serde_json::Value;

// ── Descriptive statistics ────────────────────────────────────────────────

/// `stats.mean` — arithmetic mean of f64 array.
pub(super) fn stats_mean(params: &Value, id: Value) -> JsonRpcResponse {
    let Some(data) = extract_f64_array(params, "data") else {
        return JsonRpcResponse::error(id, INVALID_PARAMS, "Missing required param: data (array)");
    };
    let result = barracuda::stats::metrics::mean(&data);
    JsonRpcResponse::success(id, serde_json::json!({ "result": result }))
}

/// `stats.std_dev` — sample standard deviation (Bessel's correction, N-1 denominator).
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

// ── Hypothesis tests ──────────────────────────────────────────────────────

/// `stats.chi_squared` — Pearson's chi-squared goodness-of-fit test.
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

// ── Information theory ────────────────────────────────────────────────────

/// `stats.shannon` — Shannon entropy H' = −Σ pᵢ ln(pᵢ).
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

/// `stats.empirical_spectral_density` — eigenvalue histogram (normalized).
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

// ── Regression models ─────────────────────────────────────────────────────

/// `stats.fit_linear` — simple linear regression y = a·x + b.
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

// ── Diversity / ecology ───────────────────────────────────────────────────

/// `stats.simpson` — Simpson diversity index D = 1 − Σ(pᵢ²).
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

/// `stats.rarefaction_curve` — expected species richness E[S] at subsampled depths.
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

// ── Special functions ─────────────────────────────────────────────────────

/// `stats.gamma_cdf` — Gamma CDF P(X ≤ x) for X ~ Gamma(alpha, beta).
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

// ── Internal helpers ──────────────────────────────────────────────────────

/// Survival function (1 - CDF) of the F-distribution via regularized incomplete beta.
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

        let am2 = 2.0f64.mul_add(m_f64, a);
        let num_even = m_f64 * (b - m_f64) * x / ((am2 - 1.0) * am2);
        d = (num_even.mul_add(d, 1.0)).recip().max(TINY);
        c = (num_even / c + 1.0).max(TINY);
        h *= d * c;

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


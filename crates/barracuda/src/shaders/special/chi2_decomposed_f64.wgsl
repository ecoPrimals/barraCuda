// SPDX-License-Identifier: AGPL-3.0-only
// Chi-squared decomposed: residuals and per-bin chi-squared terms
//
// For each bin i with observed Oᵢ and expected Eᵢ:
//   Residual:    (Oᵢ - Eᵢ) / √Eᵢ   [pull, standardized residual]
//   Chi² term:   (Oᵢ - Eᵢ)² / Eᵢ   [contribution to Pearson χ²]
//
// Sum of chi² terms = Pearson chi-squared statistic.
// Residuals indicate which bins contribute most to the discrepancy.
//
// Input @binding(0): observed values (as vec2<u32> for f64)
// Input @binding(1): expected values (as vec2<u32> for f64)
// Output @binding(2): residuals
// Output @binding(3): chi² terms
//
// Params: size (number of bins)
//
// Applications: Goodness-of-fit analysis, identifying which cells drive χ²,
// residual plots, model diagnostics for categorical data.
// Reference: Pearson's chi-squared test decomposition
//
// Note: When Eᵢ = 0, output is NaN. Requires GPU f64 support including sqrt.

@group(0) @binding(0) var<storage, read> observed: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read> expected: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read_write> residuals: array<vec2<u32>>;
@group(0) @binding(3) var<storage, read_write> chi2_terms: array<vec2<u32>>;
@group(0) @binding(4) var<uniform> params: Params;

struct Params {
    size: u32,
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.size) {
        return;
    }

    let obs = bitcast<f64>(observed[idx]);
    let exp_val = bitcast<f64>(expected[idx]);

    let diff = obs - exp_val;

    let sqrt_exp = sqrt(exp_val);
    let residual = select(f64(0.0) / f64(0.0), diff / sqrt_exp, sqrt_exp > f64(0.0));

    let chi2_term = select(f64(0.0) / f64(0.0), diff * diff / exp_val, exp_val > f64(0.0));

    residuals[idx] = bitcast<vec2<u32>>(residual);
    chi2_terms[idx] = bitcast<vec2<u32>>(chi2_term);
}

// SPDX-License-Identifier: AGPL-3.0-or-later
//
// chi_squared_f64.wgsl — Chi-squared CDF and quantile (f64)
//
// Computes the chi-squared cumulative distribution function (CDF) and
// its inverse (quantile / percent-point function) for statistical testing.
//
// χ²-CDF(x, k) = P(k/2, x/2) where P is the regularized lower
// incomplete gamma function.
//
// Uses the series expansion for the regularized gamma:
//   P(a, x) = e^{-x} · x^a / Γ(a) · Σ_{n=0}^{∞} x^n / (a·(a+1)···(a+n))
//
// Provenance: groundSpring V74 request for chi_squared_cdf / chi_squared_quantile.
//
// Operations (via params.op):
//   0 = CDF:     input [x, k] → P(χ² ≤ x | k df)
//   1 = Quantile: input [p, k] → x such that P(χ² ≤ x | k df) = p
//
// Dispatch: (ceil(N / 256), 1, 1) — one thread per element

struct Params {
    n: u32,
    op: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f64>;
@group(0) @binding(2) var<storage, read_write> output: array<f64>;

fn lgamma_approx(x: f64) -> f64 {
    // Stirling's approximation with Lanczos-like correction
    // ln Γ(x) ≈ (x-0.5)·ln(x+4.5) - (x+4.5) + 0.5·ln(2π) + correction
    let z = x - f64(1.0);
    let g = f64(7.0);
    let c0 = f64(0.99999999999980993);
    let c1 = f64(676.5203681218851);
    let c2 = f64(-1259.1392167224028);
    let c3 = f64(771.32342877765313);
    let c4 = f64(-176.61502916214059);
    let c5 = f64(12.507343278686905);
    let c6 = f64(-0.13857109526572012);
    let c7 = f64(9.9843695780195716e-6);
    let c8 = f64(1.5056327351493116e-7);

    var sum = c0;
    sum += c1 / (z + f64(1.0));
    sum += c2 / (z + f64(2.0));
    sum += c3 / (z + f64(3.0));
    sum += c4 / (z + f64(4.0));
    sum += c5 / (z + f64(5.0));
    sum += c6 / (z + f64(6.0));
    sum += c7 / (z + f64(7.0));
    sum += c8 / (z + f64(8.0));

    let t = z + g + f64(0.5);
    return f64(0.9189385332046727) + (z + f64(0.5)) * log(t) - t + log(sum);
}

fn gamma_inc_lower_series(a: f64, x: f64) -> f64 {
    // Regularized lower incomplete gamma via series expansion
    // P(a,x) = e^{-x} · x^a / Γ(a) · Σ x^n / Π(a+k)
    if (x <= f64(0.0)) { return f64(0.0); }
    if (x > f64(200.0) * a) { return f64(1.0); }

    let ln_prefix = a * log(x) - x - lgamma_approx(a);
    let prefix = exp(ln_prefix);

    var sum = f64(1.0);
    var term = f64(1.0);
    for (var n = 1u; n < 200u; n++) {
        term *= x / (a + f64(n));
        sum += term;
        if (abs(term) < f64(1e-15) * abs(sum)) { break; }
    }

    return clamp(prefix * sum / a, f64(0.0), f64(1.0));
}

fn chi2_cdf(x: f64, k: f64) -> f64 {
    if (x <= f64(0.0)) { return f64(0.0); }
    return gamma_inc_lower_series(k / f64(2.0), x / f64(2.0));
}

fn chi2_quantile(p: f64, k: f64) -> f64 {
    // Newton-Raphson on CDF(x) = p
    // Initial guess: Wilson-Hilferty approximation
    if (p <= f64(0.0)) { return f64(0.0); }
    if (p >= f64(1.0)) { return f64(1e10); }

    // Wilson-Hilferty: x ≈ k·(1 - 2/(9k) + z·√(2/(9k)))³, z = Φ⁻¹(p)
    // Simplified: use k as starting point, iterate
    var x = max(k, f64(1.0));

    for (var iter = 0u; iter < 50u; iter++) {
        let cdf_val = chi2_cdf(x, k);
        let diff = cdf_val - p;
        if (abs(diff) < f64(1e-12)) { break; }

        // PDF of chi-squared for Newton step
        let half_k = k / f64(2.0);
        let ln_pdf = (half_k - f64(1.0)) * log(x) - x / f64(2.0)
                   - half_k * log(f64(2.0)) - lgamma_approx(half_k);
        let pdf = exp(ln_pdf);
        if (pdf < f64(1e-300)) { break; }

        x = max(x - diff / pdf, f64(1e-10));
    }

    return x;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.n) { return; }

    let base = idx * 2u;
    let val = input[base + 0u];
    let k = input[base + 1u];

    switch (params.op) {
        case 0u: {
            output[idx] = chi2_cdf(val, k);
        }
        case 1u: {
            output[idx] = chi2_quantile(val, k);
        }
        default: {
            output[idx] = f64(0.0);
        }
    }
}

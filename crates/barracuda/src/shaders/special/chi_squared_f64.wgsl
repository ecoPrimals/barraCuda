// Chi-squared distribution: PDF and CDF
//
// Chi-squared with k degrees of freedom:
//   PDF: f(x; k) = x^(k/2-1) * exp(-x/2) / (2^(k/2) * Γ(k/2))   for x > 0
//   CDF: F(x; k) = P(k/2, x/2)   where P is the lower regularized gamma
//
// Uses lgamma_f64, gamma_series_f64, gamma_cf_f64 (same as regularized_gamma_f64).
//
// Input: x values [x₀, x₁, ...] (as vec2<u32> for f64)
// Output @binding(1): PDF values
// Output @binding(3): CDF values
//
// Params: size (number of elements), df (degrees of freedom k)
//
// Applications: Pearson's chi-squared test, likelihood ratio tests, variance
// estimation, goodness-of-fit, categorical data analysis.
// Reference: Abramowitz & Stegun §26.4; Press et al. "Numerical Recipes"
//
// Note: Requires GPU f64 support including log/exp operations.

@group(0) @binding(0) var<storage, read> input: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec2<u32>>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read_write> output_cdf: array<vec2<u32>>;

struct Params {
    size: u32,
    df: u32,
}

const MAX_ITER: u32 = 100u;
const EPS: f64 = 1e-14;
const FPMIN: f64 = 1e-30;

const PI_F64: f64 = 3.14159265358979323846264338327950288;
const SQRT_2PI_F64: f64 = 2.5066282746310005024157652848110452;

fn lgamma_lanczos_f64(x: f64) -> f64 {
    let g = f64(7.0);
    let x_shifted = x - f64(1.0);
    var sum: f64 = f64(0.99999999999980993);
    sum = sum + f64(676.5203681218851) / (x_shifted + f64(1.0));
    sum = sum + f64(-1259.1392167224028) / (x_shifted + f64(2.0));
    sum = sum + f64(771.32342877765313) / (x_shifted + f64(3.0));
    sum = sum + f64(-176.61502916214059) / (x_shifted + f64(4.0));
    sum = sum + f64(12.507343278686905) / (x_shifted + f64(5.0));
    sum = sum + f64(-0.13857109526572012) / (x_shifted + f64(6.0));
    sum = sum + f64(9.9843695780195716e-6) / (x_shifted + f64(7.0));
    sum = sum + f64(1.5056327351493116e-7) / (x_shifted + f64(8.0));
    let t = x_shifted + g + f64(0.5);
    return log(SQRT_2PI_F64) + log(sum) + (x_shifted + f64(0.5)) * log(t) - t;
}

fn lgamma_f64(x: f64) -> f64 {
    if (x <= f64(0.0)) {
        let z = x - x;
        return z / z;
    }
    if (x < f64(0.5)) {
        return log(PI_F64 / sin(PI_F64 * x)) - lgamma_lanczos_f64(f64(1.0) - x);
    }
    return lgamma_lanczos_f64(x);
}

fn gamma_series_f64(a: f64, x: f64, gln: f64) -> f64 {
    var sum: f64 = f64(1.0) / a;
    var term: f64 = sum;
    for (var n: u32 = 1u; n < MAX_ITER; n = n + 1u) {
        term = term * x / (a + f64(n));
        sum = sum + term;
        if (abs(term) < abs(sum) * EPS) {
            break;
        }
    }
    return sum * exp(-x + a * log(x) - gln);
}

fn gamma_cf_f64(a: f64, x: f64, gln: f64) -> f64 {
    var b: f64 = x + f64(1.0) - a;
    var c: f64 = f64(1.0) / FPMIN;
    var d: f64 = f64(1.0) / b;
    var h: f64 = d;
    for (var n: u32 = 1u; n < MAX_ITER; n = n + 1u) {
        let an = -f64(n) * (f64(n) - a);
        b = b + f64(2.0);
        d = an * d + b;
        if (abs(d) < FPMIN) {
            d = FPMIN;
        }
        c = b + an / c;
        if (abs(c) < FPMIN) {
            c = FPMIN;
        }
        d = f64(1.0) / d;
        let delta = d * c;
        h = h * delta;
        if (abs(delta - f64(1.0)) < EPS) {
            break;
        }
    }
    return exp(-x + a * log(x) - gln) * h;
}

fn regularized_gamma_p_f64(a: f64, x: f64) -> f64 {
    if (a <= f64(0.0) || x < f64(0.0)) {
        let z = a - a;
        return z / z;
    }
    if (x == f64(0.0)) {
        return f64(0.0);
    }
    let gln = lgamma_f64(a);
    if (x < a + f64(1.0)) {
        return gamma_series_f64(a, x, gln);
    }
    return gamma_cf_f64(a, x, gln);
}

fn chi2_pdf_f64(x: f64, k: f64) -> f64 {
    if (x <= f64(0.0)) {
        return f64(0.0);
    }
    let half_k = k * f64(0.5);
    let gln = lgamma_f64(half_k);
    let log_pdf = (half_k - f64(1.0)) * log(x) - x * f64(0.5) - half_k * log(f64(2.0)) - gln;
    return exp(log_pdf);
}

fn chi2_cdf_f64(x: f64, k: f64) -> f64 {
    if (x <= f64(0.0)) {
        return f64(0.0);
    }
    let a = k * f64(0.5);
    let x_scaled = x * f64(0.5);
    return regularized_gamma_p_f64(a, x_scaled);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.size) {
        return;
    }

    let x = bitcast<f64>(input[idx]);
    let k = f64(params.df);

    let pdf = chi2_pdf_f64(x, k);
    let cdf = chi2_cdf_f64(x, k);

    output[idx] = bitcast<vec2<u32>>(pdf);
    output_cdf[idx] = bitcast<vec2<u32>>(cdf);
}

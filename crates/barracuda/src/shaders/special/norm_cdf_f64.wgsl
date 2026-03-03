// Normal distribution CDF Φ(x) = P(X ≤ x) where X ~ N(μ, σ²) (f64 canonical)
// Standard normal: Φ(x) = 0.5 * (1 + erf(x / √2))
// General normal: Φ(x; μ, σ) = 0.5 * (1 + erf((x - μ) / (σ√2)))
//
// Also provides PDF φ(x) = exp(-x²/2) / √(2π)
//
// Applications: Black-Scholes, probit models, hypothesis testing, Bayesian inference
// Reference: Abramowitz & Stegun §26.2

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;
@group(0) @binding(2) var<uniform> params: Params;

struct Params {
    size: u32,
    mu: f64,      // Mean (0.0 for standard normal)
    sigma: f64,   // Standard deviation (1.0 for standard normal)
    mode: u32,    // 0 = CDF, 1 = PDF
}

const PI: f64 = 3.14159265358979323846;
const SQRT_2: f64 = 1.41421356237309504880;
const INV_SQRT_2PI: f64 = 0.3989422804014327;  // 1/√(2π)

// Abramowitz and Stegun approximation for erf (embedded for module independence)
fn erf_f64(x: f64) -> f64 {
    let a1 =  0.254829592;
    let a2 = -0.284496736;
    let a3 =  1.421413741;
    let a4 = -1.453152027;
    let a5 =  1.061405429;
    let p  =  0.3275911;

    let sign = select(-1.0, 1.0, x >= 0.0);
    let abs_x = abs(x);

    let t = 1.0 / (1.0 + p * abs_x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp_f64(-abs_x * abs_x);

    return sign * y;
}

// Standard normal CDF: Φ(z) = 0.5 * (1 + erf(z / √2))
fn std_norm_cdf(z: f64) -> f64 {
    return 0.5 * (1.0 + erf_f64(z / SQRT_2));
}

// Standard normal PDF: φ(z) = exp(-z²/2) / √(2π)
fn std_norm_pdf(z: f64) -> f64 {
    return INV_SQRT_2PI * exp_f64(-0.5 * z * z);
}

// General normal CDF: Φ(x; μ, σ)
fn norm_cdf(x: f64, mu: f64, sigma: f64) -> f64 {
    let z = (x - mu) / sigma;
    return std_norm_cdf(z);
}

// General normal PDF: φ(x; μ, σ)
fn norm_pdf(x: f64, mu: f64, sigma: f64) -> f64 {
    let z = (x - mu) / sigma;
    return std_norm_pdf(z) / sigma;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.size) {
        return;
    }

    let x = input[idx];

    if (params.mode == 0u) {
        // CDF mode
        output[idx] = norm_cdf(x, params.mu, params.sigma);
    } else {
        // PDF mode
        output[idx] = norm_pdf(x, params.mu, params.sigma);
    }
}

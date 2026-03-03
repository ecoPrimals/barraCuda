// Incomplete gamma functions: lower γ(a,x) and upper Γ(a,x)
//
// Lower: γ(a,x) = ∫₀ˣ t^(a-1) e^(-t) dt = P(a,x) · Γ(a)
// Upper: Γ(a,x) = ∫ₓ^∞ t^(a-1) e^(-t) dt = Γ(a) - γ(a,x)
//
// Uses series expansion when x < a+1, Lentz continued fraction when x >= a+1,
// same as the regularized gamma P(a,x) = γ(a,x)/Γ(a).
//
// Input: pairs [a₀, x₀, a₁, x₁, ...] interleaved (as vec2<u32> for f64)
// Output: [γ(a₀,x₀), ...] at binding(1), [Γ(a₀,x₀), ...] at binding(3)
//
// Applications: Chi-squared PDF/CDF, exponential integrals, Poisson processes,
// gamma distribution, survival analysis, reliability engineering.
// Reference: Abramowitz & Stegun §6.5; DLMF 8.2
//
// Note: Requires GPU f64 support including log/exp operations.

@group(0) @binding(0) var<storage, read> input: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec2<u32>>;      // lower γ(a,x)
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read_write> output_upper: array<vec2<u32>>; // upper Γ(a,x)

struct Params {
    size: u32,   // Number of output elements (pairs)
}

const MAX_ITER: u32 = 100u;
const EPS: f64 = 1e-14;
const FPMIN: f64 = 1e-30;

// Constants for f64
const PI_F64: f64 = 3.14159265358979323846264338327950288;
const SQRT_2PI_F64: f64 = 2.5066282746310005024157652848110452;

// Lanczos approximation for lgamma (g=7)
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
        return z / z;  // NaN
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
        return z / z;  // NaN
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

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.size) {
        return;
    }

    let a = bitcast<f64>(input[idx * 2u]);
    let x = bitcast<f64>(input[idx * 2u + 1u]);

    let gln = lgamma_f64(a);
    let gamma_a = exp(gln);
    let p = regularized_gamma_p_f64(a, x);

    let lower = p * gamma_a;   // γ(a,x)
    let upper = gamma_a - lower;  // Γ(a,x)

    output[idx] = bitcast<vec2<u32>>(lower);
    output_upper[idx] = bitcast<vec2<u32>>(upper);
}

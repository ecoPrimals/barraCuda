// Log gamma function (lgamma) operation (f64 canonical)
// lgamma(x) = ln(Γ(x)) where Γ is the gamma function
// Uses Lanczos approximation for positive x

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;
@group(0) @binding(2) var<uniform> metadata: Metadata;

struct Metadata {
    size: u32,
}

// Lanczos approximation for gamma function
// Accurate for x > 0. WGSL does not allow recursion, so we use iterative reflection.
// Note: WGSL requires constant array indexing, so we unroll the loop.
fn lgamma_lanczos(x: f64) -> f64 {
    let g = 7.0;
    let x_shifted = x - 1.0;
    var sum = 0.99999999999980993;
    sum += 676.5203681218851 / (x_shifted + 1.0);
    sum += -1259.1392167224028 / (x_shifted + 2.0);
    sum += 771.32342877765313 / (x_shifted + 3.0);
    sum += -176.61502916214059 / (x_shifted + 4.0);
    sum += 12.507343278686905 / (x_shifted + 5.0);
    sum += -0.13857109526572012 / (x_shifted + 6.0);
    sum += 9.9843695780195716e-6 / (x_shifted + 7.0);
    sum += 1.5056327351493116e-7 / (x_shifted + 8.0);
    let t = x_shifted + g + 0.5;
    let sqrt_2pi = 2.5066282746310002;
    return log_f64(sqrt_2pi) + log_f64(sum) + (x_shifted + 0.5) * log_f64(t) - t;
}

fn lgamma_approx(x: f64) -> f64 {
    if (x <= 0.0) {
        // WGSL rejects constant 0/0; use runtime NaN
        let z = x - x;
        return z / z;
    }
    // For x < 0.5: use reflection formula (no recursion - single application)
    if (x < 0.5) {
        let pi = 3.14159265358979323846;
        return log_f64(pi / sin_f64(pi * x)) - lgamma_lanczos(1.0 - x);
    }
    return lgamma_lanczos(x);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= metadata.size) {
        return;
    }

    let x = input[idx];
    output[idx] = lgamma_approx(x);
}

// Derivative of the complementary error function: d/dx erfc(x) = -2/√π · exp(-x²)
//
// The complementary error function is erfc(x) = 1 - erf(x) = 2/√π ∫ₓ^∞ exp(-t²) dt.
// By the fundamental theorem of calculus: erfc'(x) = -2/√π · exp(-x²).
//
// Input: [x₀, x₁, x₂, ...] (as vec2<u32> for f64, one value per element)
// Output: [erfc'(x₀), erfc'(x₁), ...] (as vec2<u32> for f64)
//
// Applications: Diffusion equations, heat transfer, Brownian motion, option pricing
// (Greeks), Gaussian smoothing gradients, inverse CDF derivatives for normal dist.
// Reference: Abramowitz & Stegun §7.1; erfc(x) = 1 - erf(x)
//
// Note: Requires GPU f64 support including exp.

@group(0) @binding(0) var<storage, read> input: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec2<u32>>;
@group(0) @binding(2) var<uniform> params: Params;

struct Params {
    size: u32,   // Number of output elements
}

const INV_SQRT_PI: f64 = 0.56418958354775628694807945156077259;  // 1/√π

fn erfc_deriv_f64(x: f64) -> f64 {
    return -f64(2.0) * INV_SQRT_PI * exp(-x * x);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.size) {
        return;
    }

    let x = bitcast<f64>(input[idx]);
    let result = erfc_deriv_f64(x);
    output[idx] = bitcast<vec2<u32>>(result);
}

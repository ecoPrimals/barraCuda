// Physicist's Hermite polynomials Hₙ(x)
// Uses three-term recurrence relation:
//   H₀(x) = 1
//   H₁(x) = 2x
//   Hₙ₊₁(x) = 2x·Hₙ(x) - 2n·Hₙ₋₁(x)
//
// Applications: quantum harmonic oscillator wavefunctions
// Reference: Abramowitz & Stegun §22.3

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

struct Params {
    size: u32,
    n: u32,  // Polynomial order (0, 1, 2, ...)
}

// Hermite polynomial Hₙ(x) via recurrence
fn hermite(n: u32, x: f32) -> f32 {
    if (n == 0u) {
        return 1.0;
    }
    if (n == 1u) {
        return 2.0 * x;
    }

    var h_prev: f32 = 1.0;      // H₀
    var h_curr: f32 = 2.0 * x;  // H₁

    for (var k = 1u; k < n; k = k + 1u) {
        let h_next = 2.0 * x * h_curr - 2.0 * f32(k) * h_prev;
        h_prev = h_curr;
        h_curr = h_next;
    }

    return h_curr;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.size) {
        return;
    }
    output[idx] = hermite(params.n, input[idx]);
}

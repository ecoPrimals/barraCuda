// SPDX-License-Identifier: AGPL-3.0-or-later
// Generalized Laguerre polynomials L_n^(α)(x)
// Uses three-term recurrence relation:
//   L₀^(α)(x) = 1
//   L₁^(α)(x) = 1 + α - x
//   L_n^(α)(x) = ((2n-1+α-x)·L_{n-1} - (n-1+α)·L_{n-2}) / n
//
// Applications: hydrogen atom wavefunctions, harmonic oscillator basis, gamma distributions
// Reference: Abramowitz & Stegun §22.7, NIST DLMF Chapter 18

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

struct Params {
    size: u32,
    n: u32,       // Polynomial degree
    alpha: f32,   // Generalization parameter (0.0 for simple Laguerre)
}

// Generalized Laguerre polynomial L_n^(α)(x)
fn laguerre(n: u32, alpha: f32, x: f32) -> f32 {
    if (n == 0u) {
        return 1.0;
    }
    if (n == 1u) {
        return 1.0 + alpha - x;
    }

    var l_prev2: f32 = 1.0;              // L₀
    var l_prev1: f32 = 1.0 + alpha - x;  // L₁

    for (var k = 2u; k <= n; k = k + 1u) {
        let k_f = f32(k);
        // Three-term recurrence
        let l_curr = ((2.0 * k_f - 1.0 + alpha - x) * l_prev1 - (k_f - 1.0 + alpha) * l_prev2) / k_f;
        l_prev2 = l_prev1;
        l_prev1 = l_curr;
    }

    return l_prev1;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.size) {
        return;
    }
    output[idx] = laguerre(params.n, params.alpha, input[idx]);
}

// SPDX-License-Identifier: AGPL-3.0-or-later
// pow_simple_f64.wgsl — Power operation: output = input^exponent (f64 canonical)
//
// Supports arbitrary floating-point exponents via WGSL pow_f64().
// Special cases:
//   exponent = 0 → 1.0
//   exponent = 1 → identity
//   exponent = 2 → x * x (fast path)
//   negative exponent → 1/x^|exp| via pow_f64()
//
// Cross-domain: elementwise power for activation functions, polynomial
// features, physics (r^-n potentials), image gamma correction.

struct Params {
    total: u32,
    exponent: f64,
    _padding: vec2<u32>,
}

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if (idx >= params.total) {
        return;
    }

    let x = input[idx];
    let p = params.exponent;

    // Fast paths for common exponents
    if (p == 0.0) {
        output[idx] = 1.0;
    } else if (p == 1.0) {
        output[idx] = x;
    } else if (p == 2.0) {
        output[idx] = x * x;
    } else if (p == 3.0) {
        output[idx] = x * x * x;
    } else if (p == -1.0) {
        output[idx] = 1.0 / x;
    } else {
        // General case: use pow_f64 for positive base
        if (x > 0.0) {
            output[idx] = pow_f64(x, p);
        } else if (x == 0.0) {
            output[idx] = select(0.0, 0.0, p > 0.0);
        } else {
            let abs_result = pow_f64(abs(x), p);
            let is_odd = (f64(i32(p)) == p) && (i32(p) % 2 != 0);
            output[idx] = select(abs_result, -abs_result, is_odd);
        }
    }
}

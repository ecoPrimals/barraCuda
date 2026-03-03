// laguerre_generalized_f64.wgsl — Generalized Laguerre polynomials L_n^α(x)
//
// **Math**: Generalized Laguerre (Sonine) polynomials L_n^α(x), n ≥ 0, α > -1.
//
// Recurrence relation:
//   L_0^α(x) = 1
//   L_1^α(x) = 1 + α - x
//   L_{k+1}^α(x) = ((2k + 1 + α - x)·L_k^α - (k + α)·L_{k-1}^α) / (k + 1)
//
// Applications: Radial part of 3D hydrogen wavefunctions, orthogonal polynomials
// on [0,∞) with weight x^α·e^{-x}, quantum oscillator, Laguerre-Gauss quadrature.
// Reference: Abramowitz & Stegun §22.3, DLMF §18.9
//
// **Precision**: f64 via bitcast<f64>(vec2<u32>)
// **Workgroup**: @compute @workgroup_size(256)
//
// Bindings:
//   0: x         array<vec2<u32>>  read       — evaluation points
//   1: params_n  array<vec2<u32>>  read       — [n, alpha] per element, 2 f64s each
//   2: output    array<vec2<u32>>  read_write — L_n^α(x) values
//   3: params    uniform
//
// Params: { size: u32 }
//
// Layout: params_n[2*i] = n (as f64), params_n[2*i+1] = α

@group(0) @binding(0) var<storage, read> x_vals: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read> params_n: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read_write> output: array<vec2<u32>>;
@group(0) @binding(3) var<uniform> params: Params;

struct Params {
    size: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

fn laguerre_generalized(n: u32, alpha: f64, x: f64) -> f64 {
    if (n == 0u) {
        return f64(1.0);
    }
    if (n == 1u) {
        return f64(1.0) + alpha - x;
    }

    var l_prev: f64 = f64(1.0);
    var l_curr: f64 = f64(1.0) + alpha - x;

    for (var k: u32 = 1u; k < n; k = k + 1u) {
        let k_f64 = f64(k);
        let l_next = ((f64(2.0) * k_f64 + f64(1.0) + alpha - x) * l_curr - (k_f64 + alpha) * l_prev) / (k_f64 + f64(1.0));
        l_prev = l_curr;
        l_curr = l_next;
    }

    return l_curr;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= params.size) {
        return;
    }

    let x = bitcast<f64>(x_vals[i]);
    let n_val = bitcast<f64>(params_n[i * 2u]);
    let alpha = bitcast<f64>(params_n[i * 2u + 1u]);

    let n_ord = u32(n_val);

    let result = laguerre_generalized(n_ord, alpha, x);
    output[i] = bitcast<vec2<u32>>(result);
}

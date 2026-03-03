// hill_f64.wgsl — Element-wise Hill activation (f64 precision)
//
// Hill(x_i, K, n) = x_i^n / (K^n + x_i^n)
//
// Used in kinetic models:
//   - Quorum sensing / c-di-GMP regulatory cascades (wetSpring Exp019)
//   - Enzymatic rate laws (Michaelis-Menten is Hill with n=1)
//   - Cooperativity in ligand binding
//
// All inputs clamped to ≥ 0 to avoid NaN from negative pow().
// Output is in [0, 1].
//
// Bindings:
//   0: input  [N] f64 — concentration / stimulus values
//   1: output [N] f64 — Hill activation values in [0,1]
//   2: params uniform  { n_elements: u32, K: f64, n: f64 }

// f64 is enabled by compile_shader_f64() preamble injection — do not use `enable f64;`

struct HillParams {
    n_elements: u32,
    _pad:       u32,
    K:          f64,   // Half-saturation constant (EC50)
    n:          f64,   // Hill coefficient (cooperativity exponent)
}

@group(0) @binding(0) var<storage, read>       input:  array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;
@group(0) @binding(2) var<uniform>             params: HillParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.n_elements) { return; }

    let x  = max(input[idx], f64(0.0));
    let Kn = pow(max(params.K, f64(1e-30)), params.n);
    let xn = pow(x, params.n);
    output[idx] = xn / (Kn + xn);
}

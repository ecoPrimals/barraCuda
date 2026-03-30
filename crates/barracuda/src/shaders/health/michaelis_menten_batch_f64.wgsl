// SPDX-License-Identifier: AGPL-3.0-or-later
//
// michaelis_menten_batch_f64.wgsl — Batch Michaelis-Menten PK simulation
//
// Each thread simulates one patient: Euler integration of
//   dC/dt = -Vmax*C / (Km + C) / Vd
// for n_steps iterations, then computes trapezoidal AUC.
//
// Output: AUC per patient (f64).
// Patient variation: per-patient Vmax drawn from lognormal-like distribution
// via Wang hash + xorshift32 PRNG.
//
// Absorbed from healthSpring V19 (Exp083, Exp085).
// Dispatch: (ceil(n_patients / 256), 1, 1)
//
// f64 enabled by compile_shader_f64() preamble — do not use `enable f64;`.
//
// Requires prng_wang_f64.wgsl prepended for wang_hash, xorshift32,
// u32_to_uniform_f64.

struct Params {
    vmax: f64,
    km: f64,
    vd: f64,
    dt: f64,
    n_steps: u32,
    n_patients: u32,
    base_seed: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read_write> output: array<f64>;
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.n_patients { return; }

    var rng_state = wang_hash(params.base_seed + idx);
    if rng_state == 0u { rng_state = 1u; }
    let bits = xorshift32(&rng_state);
    let u = u32_to_uniform_f64(bits);
    // Lognormal-like Vmax variation: range [0.7, 1.3] x Vmax (CV ~20%).
    // Reflects inter-patient variability in hepatic CYP2C9 expression
    // for phenytoin (Gerber et al., Clin Pharmacol Ther 1985).
    let vmax_factor = 0.7 + u * 0.6;
    let patient_vmax = params.vmax * vmax_factor;

    // Reference: C0 = 300 mg / 50 L = 6 mg/L (standard 300 mg loading dose,
    // Vd ~50 L; Winter, Basic Clinical PK, 5th ed).
    let dose_mg = params.vd * 6.0;
    var c = dose_mg / params.vd;
    var auc = 0.0;

    for (var step = 0u; step < params.n_steps; step = step + 1u) {
        let c_prev = c;
        let elim = patient_vmax * c / (params.km + c);
        c = max(0.0, c - (elim / params.vd) * params.dt);
        auc = auc + (c_prev + c) * 0.5 * params.dt;
    }

    output[idx] = auc;
}

// SPDX-License-Identifier: AGPL-3.0-or-later
//
// population_pk_f64.wgsl — Population PK Monte Carlo on GPU
//
// Absorbed from healthSpring V44 (March 2026).
//
// Each thread simulates one virtual patient:
//   1. Wang hash + xorshift32 PRNG for well-distributed clearance variation
//   2. AUC = F * Dose / CL where CL varies per patient
//
// Uses u32-only PRNG (no SHADER_INT64 needed).
//
// Dispatch: (ceil(n_patients / 256), 1, 1)
//
// Requires prng_wang_f64.wgsl prepended for wang_hash, xorshift32,
// u32_to_uniform_f64.

struct Params {
    n_patients: u32,
    base_seed: u32,
    dose_mg: f64,
    f_bioavail: f64,
}

@group(0) @binding(0) var<storage, read_write> output: array<f64>;
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.n_patients {
        return;
    }

    var rng_state: u32 = wang_hash(params.base_seed + idx);
    if rng_state == 0u {
        rng_state = 1u;
    }

    let bits = xorshift32(&rng_state);
    let u = u32_to_uniform_f64(bits);

    // CL varies [0.5, 1.5] x base_cl — ~50% coefficient of variation
    // in hepatic clearance (Rowland & Tozer, Clinical PK, 4th ed).
    // base_cl = 10.0 L/hr (typical oral small-molecule clearance).
    let cl_factor = 0.5 + u;
    let cl = 10.0 * cl_factor;

    // Single-compartment AUC = F * Dose / CL
    let auc = params.f_bioavail * params.dose_mg / cl;

    output[idx] = auc;
}

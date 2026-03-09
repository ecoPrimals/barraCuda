// SPDX-License-Identifier: AGPL-3.0-or-later
//
// population_pk_f64.wgsl — Population PK Monte Carlo (f64, GPU-vectorized)
//
// Each thread simulates one virtual patient:
//   1. Wang hash + xorshift32 PRNG for well-distributed per-patient randomness
//   2. AUC = F × Dose / CL where CL = base_cl × (cl_low + u × (cl_high - cl_low))
//
// Evolved from healthSpring: fully parameterized (no hardcoded CL range or base).
// Uses u32-only PRNG (no SHADER_INT64 needed).
//
// Dispatch: (ceil(n_patients / WORKGROUP_SIZE), 1, 1)

// f64 is enabled by compile_shader_f64() preamble injection — do not use `enable f64;`

struct Params {
    n_patients: u32,
    base_seed:  u32,
    dose_mg:    f64,
    f_bioavail: f64,
    base_cl:    f64,
    cl_low:     f64,
    cl_high:    f64,
}

@group(0) @binding(0) var<storage, read_write> output: array<f64>;
@group(0) @binding(1) var<uniform> params: Params;

fn wang_hash(input: u32) -> u32 {
    var x = input;
    x = (x ^ 61u) ^ (x >> 16u);
    x = x * 9u;
    x = x ^ (x >> 4u);
    x = x * 0x27d4eb2du;
    x = x ^ (x >> 15u);
    return x;
}

fn xorshift32(state: ptr<function, u32>) -> u32 {
    var x = *state;
    x ^= x << 13u;
    x ^= x >> 17u;
    x ^= x << 5u;
    *state = x;
    return x;
}

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
    let u = f64(bits) / 4294967295.0;

    let cl_factor = params.cl_low + u * (params.cl_high - params.cl_low);
    let cl = params.base_cl * cl_factor;

    output[idx] = params.f_bioavail * params.dose_mg / cl;
}

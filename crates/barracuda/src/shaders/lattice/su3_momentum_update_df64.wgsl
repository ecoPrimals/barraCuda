// SPDX-License-Identifier: AGPL-3.0-or-later
// Momentum update: P[i] += dt * F[i] for SU(3) algebra elements (DF64 path).
//
// Silicon saturation: runs on FP32 cores via DF64 arithmetic, freeing
// FP64 units for precision-critical reductions (plaquette, KE, deltaH).
// Buffers remain f64 — conversion at load/store boundary only.
// Validated: |deltaP| = 1.44e-7 vs native at 16^4 (RTX 3090).
//
// Prepend: su3_df64_preamble (provides Df64, df64_from_f64, df64_to_f64, etc.)
//
// Binding layout matches hmc_leapfrog_f64.wgsl for drop-in dispatch.

struct LeapfrogParams {
    volume:  u32,
    n_links: u32,
    _pad0:   u32,
    _pad1:   u32,
    dt:      f64,
    _padf:   f64,
}

@group(0) @binding(0) var<uniform>             params:    LeapfrogParams;
@group(0) @binding(1) var<storage, read_write> links:     array<f64>;
@group(0) @binding(2) var<storage, read_write> momenta:   array<f64>;
@group(0) @binding(3) var<storage, read>       force:     array<f64>;
@group(0) @binding(4) var<storage, read_write> rng_state: array<u32>;

@compute @workgroup_size(64)
fn momentum_update_df64(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) num_wgs: vec3<u32>,
) {
    let idx = gid.y * (num_wgs.x * 64u) + gid.x;
    if idx >= params.n_links { return; }

    let base = idx * 18u;
    let dt = df64_from_f64(params.dt);

    for (var i = 0u; i < 18u; i++) {
        let p = df64_from_f64(momenta[base + i]);
        let f = df64_from_f64(force[base + i]);
        let result = df64_add(p, df64_mul(dt, f));
        momenta[base + i] = df64_to_f64(result);
    }
}

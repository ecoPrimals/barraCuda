// SPDX-License-Identifier: AGPL-3.0-only
// kinetic_energy_df64.wgsl — Hybrid per-link kinetic energy (DF64 core streaming)
//
// Prepend: complex_f64.wgsl + su3.wgsl + df64_core.wgsl + su3_df64.wgsl
//
// HYBRID PRECISION:
//   DF64 (FP32 cores): su3_mul_df64(P, P) — 27 complex FMAs = 108 f32 muls
//   f64  (FP64 units): -0.5 * Re Tr(P²) scalar output
//
// Buffer layout: UNCHANGED from kinetic_energy_f64.wgsl.

struct KineticParams {
    n_links: u32,
    _pad0:   u32,
    _pad1:   u32,
    _pad2:   u32,
}

@group(0) @binding(0) var<uniform>             params:  KineticParams;
@group(0) @binding(1) var<storage, read>       momenta: array<f64>;
@group(0) @binding(2) var<storage, read_write> energy:  array<f64>;

fn load_su3_df64(base: u32) -> array<Cdf64, 9> {
    var m: array<Cdf64, 9>;
    for (var i = 0u; i < 9u; i = i + 1u) {
        m[i] = cdf64_from_f64(momenta[base + i * 2u], momenta[base + i * 2u + 1u]);
    }
    return m;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let link_id = gid.x + gid.y * nwg.x * 64u;
    if (link_id >= params.n_links) { return; }

    let base = link_id * 18u;

    // DF64 zone: P² on FP32 cores
    let p = load_su3_df64(base);
    let p2 = su3_mul_df64(p, p);
    let re_tr = su3_re_trace_df64(p2);

    // f64 zone: scalar output
    energy[link_id] = -0.5 * df64_to_f64(re_tr);
}

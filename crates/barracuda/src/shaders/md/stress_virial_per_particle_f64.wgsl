// SPDX-License-Identifier: AGPL-3.0-or-later
//
// stress_virial_per_particle_f64.wgsl — Per-particle σ_xy for Green-Kubo (transport_gpu)
//
// Each thread computes one particle's contribution to σ_xy:
//   σ_xy,i = m·vx·vy + r_x·F_y  (kinetic + centroid virial)
//
// transport_gpu passes: pos, vel, out, params (4 bindings).
// Params struct: n_atoms, mass (caller packs as needed).
// Note: Full virial needs forces; if not passed, kinetic-only is used.
// This variant uses params for mass; virial term requires force buffer (not in transport_gpu).

@group(0) @binding(0) var<storage, read> pos: array<f64>;
@group(0) @binding(1) var<storage, read> vel: array<f64>;
@group(0) @binding(2) var<storage, read_write> out_xy: array<f64>;
@group(0) @binding(3) var<storage, read> params: array<f64>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let n = u32(params[0]);
    if i >= n { return; }

    let mass = params[1];
    let vx = vel[i * 3u];
    let vy = vel[i * 3u + 1u];

    // Kinetic contribution only (transport_gpu does not pass forces)
    // σ_xy,i = m·vx·vy
    out_xy[i] = mass * vx * vy;
}

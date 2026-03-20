// SPDX-License-Identifier: AGPL-3.0-or-later
// Per-Particle Kinetic Energy — f64 precision
//
// **Physics**: KE = 0.5 * m * v² per particle
// **Use Case**: Temperature calculation: T = 2*KE_total / (3*N*k_B)
//
// **f64 precision**: Full f64 (no math_f64 dependencies — pure arithmetic)
//
// Bindings:
//   0: velocities [N*3] f64, read
//   1: masses     [N]   f64, read  — per-particle masses
//   2: ke_buf     [N]   f64, write — per-particle KE
//   3: params     uniform

@group(0) @binding(0) var<storage, read> velocities: array<f64>;
@group(0) @binding(1) var<storage, read> masses: array<f64>;
@group(0) @binding(2) var<storage, read_write> ke_buf: array<f64>;
@group(0) @binding(3) var<uniform> params: Params;

struct Params {
    n_particles: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n_particles) { 
        return; 
    }

    let mass = masses[i];
    let vx = velocities[i * 3u];
    let vy = velocities[i * 3u + 1u];
    let vz = velocities[i * 3u + 2u];

    ke_buf[i] = 0.5 * mass * (vx * vx + vy * vy + vz * vz);
}

// Total kinetic energy reduction (single workgroup for small N)
var<workgroup> shared_ke: array<f64, 256>;

@compute @workgroup_size(256)
fn reduce_total(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let i = gid.x;
    let local_idx = lid.x;

    if (i < params.n_particles) {
        shared_ke[local_idx] = ke_buf[i];
    } else {
        shared_ke[local_idx] = ke_buf[0] - ke_buf[0];
    }

    workgroupBarrier();

    for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
        if (local_idx < stride) {
            shared_ke[local_idx] = shared_ke[local_idx] + shared_ke[local_idx + stride];
        }
        workgroupBarrier();
    }

    if (local_idx == 0u) {
        ke_buf[gid.x / 256u] = shared_ke[0];
    }
}

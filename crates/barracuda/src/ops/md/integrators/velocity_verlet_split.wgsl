// SPDX-License-Identifier: AGPL-3.0-only
// Velocity-Verlet Split Integrator (f64) — Kick-Drift-Kick Pattern
//
// **Algorithm**: Split VV for flexible thermostating and force kernel swap
// **Step 1**: Half-kick + drift + PBC wrap (this shader)
// **Step 2**: Force recomputation (separate shader)
// **Step 3**: Second half-kick (vv_half_kick_f64.wgsl)
//
// **Precision**: Full f64 via math_f64.wgsl preamble
// **Reference**: LAMMPS, GROMACS standard integrator pattern
//
// Requires: math_f64.wgsl preamble (floor_f64)
//
// Bindings:
//   0: positions    [N*3] f64, read-write  — updated in-place
//   1: velocities   [N*3] f64, read-write  — updated in-place
//   2: forces       [N*3] f64, read        — current forces
//   3: params       [8]   f64, read        — [n, dt, mass, _, box_x, box_y, box_z, _]

@group(0) @binding(0) var<storage, read_write> positions: array<f64>;
@group(0) @binding(1) var<storage, read_write> velocities: array<f64>;
@group(0) @binding(2) var<storage, read> forces: array<f64>;
@group(0) @binding(3) var<storage, read> params: array<f64>;

// params: [n_particles, dt, mass, _, box_x, box_y, box_z, _]

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let n = u32(params[0]);
    if (i >= n) { return; }

    let dt    = params[1];
    let mass  = params[2];
    let box_x = params[4];
    let box_y = params[5];
    let box_z = params[6];

    let inv_m = 1.0 / mass;
    let half_dt = 0.5 * dt;

    // Load velocities
    var vx = velocities[i * 3u];
    var vy = velocities[i * 3u + 1u];
    var vz = velocities[i * 3u + 2u];

    // Accelerations from current forces
    let ax = forces[i * 3u]      * inv_m;
    let ay = forces[i * 3u + 1u] * inv_m;
    let az = forces[i * 3u + 2u] * inv_m;

    // Half-kick: v += 0.5 * dt * a
    vx = vx + half_dt * ax;
    vy = vy + half_dt * ay;
    vz = vz + half_dt * az;

    // Drift: x += dt * v
    var px = positions[i * 3u]      + dt * vx;
    var py = positions[i * 3u + 1u] + dt * vy;
    var pz = positions[i * 3u + 2u] + dt * vz;

    // PBC wrap: keep x in [0, box)
    px = px - box_x * floor_f64(px / box_x);
    py = py - box_y * floor_f64(py / box_y);
    pz = pz - box_z * floor_f64(pz / box_z);

    // Store updated state
    positions[i * 3u]      = px;
    positions[i * 3u + 1u] = py;
    positions[i * 3u + 2u] = pz;
    velocities[i * 3u]      = vx;
    velocities[i * 3u + 1u] = vy;
    velocities[i * 3u + 2u] = vz;
}

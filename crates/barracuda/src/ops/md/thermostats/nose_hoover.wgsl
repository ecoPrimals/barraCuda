// SPDX-License-Identifier: AGPL-3.0-only
// Nosé-Hoover Thermostat (f64)
//
// **Physics**: Extended Lagrangian for canonical (NVT) ensemble
// **Properties**: Time-reversible, samples canonical distribution correctly
// **Use Case**: Production NVT runs (after Berendsen equilibration)
//
// **Algorithm** (Velocity-Verlet-like integration):
//   1. Half-step thermostat: ξ += (dt/2) * (KE/Q - T_target) / T_target
//   2. Half-step velocities: v += (dt/2) * (F/m - ξ*v)
//   3. Full-step positions: x += dt * v (done separately)
//   4. Recompute forces (done separately)
//   5. Half-step velocities: v += (dt/2) * (F/m - ξ*v)
//   6. Half-step thermostat: ξ += (dt/2) * (KE/Q - T_target) / T_target
//
// This shader implements steps 2 (pre-drift) and 5 (post-drift) in one dispatch.
// The thermostat variable ξ update is done on CPU between dispatches.
//
// **Precision**: Full f64 (no math_f64 dependencies — pure arithmetic)
//
// Bindings:
//   0: velocities [N*3] f64, read-write  — updated in-place
//   1: forces     [N*3] f64, read        — current forces
//   2: params     [8]   f64, read        — [n, dt, mass, xi, _, _, _, _]

@group(0) @binding(0) var<storage, read_write> velocities: array<f64>;
@group(0) @binding(1) var<storage, read> forces: array<f64>;
@group(0) @binding(2) var<storage, read> params: array<f64>;

// params layout:
//   [0] = n_particles
//   [1] = dt (timestep)
//   [2] = mass (particle mass, 3.0 in OCP units)
//   [3] = xi (thermostat variable, computed on CPU)

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let n = u32(params[0]);
    if (i >= n) { return; }

    let dt   = params[1];
    let mass = params[2];
    let xi   = params[3];

    let inv_m   = 1.0 / mass;
    let half_dt = 0.5 * dt;

    // Load current velocity
    var vx = velocities[i * 3u];
    var vy = velocities[i * 3u + 1u];
    var vz = velocities[i * 3u + 2u];

    // Acceleration from forces
    let ax = forces[i * 3u]      * inv_m;
    let ay = forces[i * 3u + 1u] * inv_m;
    let az = forces[i * 3u + 2u] * inv_m;

    // Half-step velocity with friction: v += (dt/2) * (a - ξ*v)
    // Rearranged for stability: v_new = (v + (dt/2)*a) / (1 + (dt/2)*ξ)
    // This avoids instability when ξ is large
    let denom = 1.0 / (1.0 + half_dt * xi);
    vx = (vx + half_dt * ax) * denom;
    vy = (vy + half_dt * ay) * denom;
    vz = (vz + half_dt * az) * denom;

    // Store updated velocity
    velocities[i * 3u]      = vx;
    velocities[i * 3u + 1u] = vy;
    velocities[i * 3u + 2u] = vz;
}

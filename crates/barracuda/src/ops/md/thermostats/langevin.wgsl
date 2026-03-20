// SPDX-License-Identifier: AGPL-3.0-or-later
// Langevin Thermostat (f64)
//
// **Physics**: Stochastic dynamics with friction and random noise
// **Properties**: Samples canonical ensemble, ergodic, good for non-equilibrium
// **Use Case**: Brownian dynamics, implicit solvent, driven systems
//
// **Algorithm** (BAOAB splitting for stability):
//   v += (dt/2) * a                    // B: half-kick from forces
//   v *= exp(-gamma * dt)              // A: friction decay
//   v += sigma * sqrt(1-exp(-2*gamma*dt)) * noise  // O: random kick
//   x += dt * v                        // A: drift (done separately)
//   v += (dt/2) * a                    // B: half-kick from forces
//
// This shader implements the friction + noise step (A + O combined).
// The random numbers are passed in from CPU (generated via rand crate).
//
// **Precision**: Full f64
//
// Note: exp/sqrt factors are pre-computed on CPU (params[4,5]) for efficiency
//
// Bindings:
//   0: velocities [N*3] f64, read-write — updated in-place
//   1: noise      [N*3] f64, read       — Gaussian random numbers (mean=0, std=1)
//   2: params     [8]   f64, read       — [n, gamma, sigma, dt, exp_factor, noise_factor, _, _]

@group(0) @binding(0) var<storage, read_write> velocities: array<f64>;
@group(0) @binding(1) var<storage, read> noise: array<f64>;
@group(0) @binding(2) var<storage, read> params: array<f64>;

// params layout:
//   [0] = n_particles
//   [1] = gamma (friction coefficient, units of 1/time)
//   [2] = sigma = sqrt(2 * gamma * k_B * T / m) (noise amplitude)
//   [3] = dt (timestep)
//   [4] = exp_factor = exp(-gamma * dt) (pre-computed on CPU)
//   [5] = noise_factor = sigma * sqrt(1 - exp(-2*gamma*dt)) (pre-computed on CPU)

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let n = u32(params[0]);
    if (i >= n) { return; }

    let exp_factor = params[4];
    let noise_factor = params[5];

    // Apply friction: v *= exp(-gamma * dt)
    // Apply noise: v += sigma * sqrt(1 - exp(-2*gamma*dt)) * noise
    // Combined: v_new = v * exp_factor + noise_factor * noise

    velocities[i * 3u]      = velocities[i * 3u]      * exp_factor + noise_factor * noise[i * 3u];
    velocities[i * 3u + 1u] = velocities[i * 3u + 1u] * exp_factor + noise_factor * noise[i * 3u + 1u];
    velocities[i * 3u + 2u] = velocities[i * 3u + 2u] * exp_factor + noise_factor * noise[i * 3u + 2u];
}

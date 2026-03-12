// SPDX-License-Identifier: AGPL-3.0-only
//! Velocity-Verlet Time Integration
//!
//! **Physics**: Symplectic integrator for classical mechanics (Hamilton's equations)
//! **Algorithm**: Second-order accurate, energy-conserving
//! **Use Case**: Molecular dynamics, N-body simulations
//!
//! **Advantages over Euler**:
//! - Preserves phase-space volume (Liouville's theorem)
//! - Long-term energy stability
//! - Time-reversible
//!
//! **Algorithm**:
//! 1. x(t+Δt) = x(t) + v(t)Δt + ½a(t)Δt²
//! 2. v(t+Δt) = v(t) + ½[a(t) + a(t+Δt)]Δt
//!
//! **Deep Debt Compliance**:
//! - ✅ Pure WGSL shader
//! - ✅ Zero unsafe code
//! - ✅ Capability-based (agnostic to system)

@group(0) @binding(0) var<storage, read> positions: array<f32>;      // [N, 3]
@group(0) @binding(1) var<storage, read> velocities: array<f32>;     // [N, 3]
@group(0) @binding(2) var<storage, read> forces_old: array<f32>;     // [N, 3] at time t
@group(0) @binding(3) var<storage, read> forces_new: array<f32>;     // [N, 3] at time t+Δt
@group(0) @binding(4) var<storage, read> masses: array<f32>;         // [N]
@group(0) @binding(5) var<storage, read_write> positions_new: array<f32>; // [N, 3]
@group(0) @binding(6) var<storage, read_write> velocities_new: array<f32>; // [N, 3]
@group(0) @binding(7) var<uniform> params: Params;

struct Params {
    n_particles: u32,
    dt: f32,           // Time step
    pad1: f32,
    pad2: f32,
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    
    if (i >= params.n_particles) {
        return;
    }
    
    // Load current state
    let pos = vec3<f32>(
        positions[i * 3u],
        positions[i * 3u + 1u],
        positions[i * 3u + 2u]
    );
    let vel = vec3<f32>(
        velocities[i * 3u],
        velocities[i * 3u + 1u],
        velocities[i * 3u + 2u]
    );
    let f_old = vec3<f32>(
        forces_old[i * 3u],
        forces_old[i * 3u + 1u],
        forces_old[i * 3u + 2u]
    );
    let f_new = vec3<f32>(
        forces_new[i * 3u],
        forces_new[i * 3u + 1u],
        forces_new[i * 3u + 2u]
    );
    let m = masses[i];
    
    // Compute accelerations: a = F/m
    let a_old = f_old / m;
    let a_new = f_new / m;
    
    // Velocity-Verlet update
    // Position: x(t+Δt) = x(t) + v(t)Δt + ½a(t)Δt²
    let pos_new = pos + vel * params.dt + 0.5 * a_old * params.dt * params.dt;
    
    // Velocity: v(t+Δt) = v(t) + ½[a(t) + a(t+Δt)]Δt
    let vel_new = vel + 0.5 * (a_old + a_new) * params.dt;
    
    // Write new state
    positions_new[i * 3u] = pos_new.x;
    positions_new[i * 3u + 1u] = pos_new.y;
    positions_new[i * 3u + 2u] = pos_new.z;
    
    velocities_new[i * 3u] = vel_new.x;
    velocities_new[i * 3u + 1u] = vel_new.y;
    velocities_new[i * 3u + 2u] = vel_new.z;
}

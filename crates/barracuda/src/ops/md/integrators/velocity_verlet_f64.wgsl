// SPDX-License-Identifier: AGPL-3.0-or-later
//! Velocity-Verlet Time Integration — f64 precision
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
//! **f64 precision**: Essential for long-time energy conservation

@group(0) @binding(0) var<storage, read> positions: array<f64>;      // [N, 3]
@group(0) @binding(1) var<storage, read> velocities: array<f64>;     // [N, 3]
@group(0) @binding(2) var<storage, read> forces_old: array<f64>;     // [N, 3] at time t
@group(0) @binding(3) var<storage, read> forces_new: array<f64>;     // [N, 3] at time t+Δt
@group(0) @binding(4) var<storage, read> masses: array<f64>;         // [N]
@group(0) @binding(5) var<storage, read_write> positions_new: array<f64>; // [N, 3]
@group(0) @binding(6) var<storage, read_write> velocities_new: array<f64>; // [N, 3]
@group(0) @binding(7) var<uniform> params: Params;

struct Params {
    n_particles: u32,
    _pad0: u32,
    dt: f64,           // Time step
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    
    if (i >= params.n_particles) {
        return;
    }
    
    // Load current state
    let pos_x = positions[i * 3u];
    let pos_y = positions[i * 3u + 1u];
    let pos_z = positions[i * 3u + 2u];
    
    let vel_x = velocities[i * 3u];
    let vel_y = velocities[i * 3u + 1u];
    let vel_z = velocities[i * 3u + 2u];
    
    let f_old_x = forces_old[i * 3u];
    let f_old_y = forces_old[i * 3u + 1u];
    let f_old_z = forces_old[i * 3u + 2u];
    
    let f_new_x = forces_new[i * 3u];
    let f_new_y = forces_new[i * 3u + 1u];
    let f_new_z = forces_new[i * 3u + 2u];
    
    let m = masses[i];
    let inv_m = 1.0 / m;
    
    // Compute accelerations: a = F/m
    let a_old_x = f_old_x * inv_m;
    let a_old_y = f_old_y * inv_m;
    let a_old_z = f_old_z * inv_m;
    
    let a_new_x = f_new_x * inv_m;
    let a_new_y = f_new_y * inv_m;
    let a_new_z = f_new_z * inv_m;
    
    let dt = params.dt;
    let dt_sq = dt * dt;
    
    // Position: x(t+Δt) = x(t) + v(t)Δt + ½a(t)Δt²
    let pos_new_x = pos_x + vel_x * dt + 0.5 * a_old_x * dt_sq;
    let pos_new_y = pos_y + vel_y * dt + 0.5 * a_old_y * dt_sq;
    let pos_new_z = pos_z + vel_z * dt + 0.5 * a_old_z * dt_sq;
    
    // Velocity: v(t+Δt) = v(t) + ½[a(t) + a(t+Δt)]Δt
    let vel_new_x = vel_x + 0.5 * (a_old_x + a_new_x) * dt;
    let vel_new_y = vel_y + 0.5 * (a_old_y + a_new_y) * dt;
    let vel_new_z = vel_z + 0.5 * (a_old_z + a_new_z) * dt;
    
    // Write new state
    positions_new[i * 3u] = pos_new_x;
    positions_new[i * 3u + 1u] = pos_new_y;
    positions_new[i * 3u + 2u] = pos_new_z;
    
    velocities_new[i * 3u] = vel_new_x;
    velocities_new[i * 3u + 1u] = vel_new_y;
    velocities_new[i * 3u + 2u] = vel_new_z;
}

// Half-step velocity update (first half of leapfrog)
@compute @workgroup_size(256)
fn velocity_half_step(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    
    if (i >= params.n_particles) {
        return;
    }
    
    let vel_x = velocities[i * 3u];
    let vel_y = velocities[i * 3u + 1u];
    let vel_z = velocities[i * 3u + 2u];
    
    let f_x = forces_old[i * 3u];
    let f_y = forces_old[i * 3u + 1u];
    let f_z = forces_old[i * 3u + 2u];
    
    let m = masses[i];
    let inv_m = 1.0 / m;
    let half_dt = 0.5 * params.dt;
    
    velocities_new[i * 3u] = vel_x + f_x * inv_m * half_dt;
    velocities_new[i * 3u + 1u] = vel_y + f_y * inv_m * half_dt;
    velocities_new[i * 3u + 2u] = vel_z + f_z * inv_m * half_dt;
}

// Position update using half-step velocities
@compute @workgroup_size(256)
fn position_update(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    
    if (i >= params.n_particles) {
        return;
    }
    
    let pos_x = positions[i * 3u];
    let pos_y = positions[i * 3u + 1u];
    let pos_z = positions[i * 3u + 2u];
    
    // Use velocities buffer as half-step velocities
    let vel_half_x = velocities[i * 3u];
    let vel_half_y = velocities[i * 3u + 1u];
    let vel_half_z = velocities[i * 3u + 2u];
    
    let dt = params.dt;
    
    positions_new[i * 3u] = pos_x + vel_half_x * dt;
    positions_new[i * 3u + 1u] = pos_y + vel_half_y * dt;
    positions_new[i * 3u + 2u] = pos_z + vel_half_z * dt;
}

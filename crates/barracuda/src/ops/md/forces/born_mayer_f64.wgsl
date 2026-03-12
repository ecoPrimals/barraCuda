// SPDX-License-Identifier: AGPL-3.0-only
// Born-Mayer Force Kernel — f64 precision
//
// **Purpose**: Repulsive core interactions (excluded volume)
// **Use Case**: Ionic crystals, hard-core repulsion, steric effects
// **Range**: Very short-range (exponential repulsion)
//
// **Potential**: U(r) = A * exp(-r/ρ)
// **Force**: F = (A/ρ) * exp(-r/ρ) * r_hat
//   where A = repulsion strength
//         ρ = range parameter (softness)
//
// **f64 precision**: Required for accurate energy conservation in MD

@group(0) @binding(0) var<storage, read> positions: array<f64>;
@group(0) @binding(1) var<storage, read> A_params: array<f64>;    // [N] per-particle A
@group(0) @binding(2) var<storage, read> rho_params: array<f64>;  // [N] per-particle ρ
@group(0) @binding(3) var<storage, read_write> forces: array<f64>;
@group(0) @binding(4) var<uniform> params: Params;

struct Params {
    n_particles: u32,
    _pad0: u32,
    cutoff_radius: f64,
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    
    if (i >= params.n_particles) {
        return;
    }
    
    let pos_i_x = positions[i * 3u];
    let pos_i_y = positions[i * 3u + 1u];
    let pos_i_z = positions[i * 3u + 2u];
    let A_i = A_params[i];
    let rho_i = rho_params[i];
    
    var force_x = 0.0;
    var force_y = 0.0;
    var force_z = 0.0;
    
    for (var j = 0u; j < params.n_particles; j = j + 1u) {
        if (i == j) {
            continue;
        }
        
        let pos_j_x = positions[j * 3u];
        let pos_j_y = positions[j * 3u + 1u];
        let pos_j_z = positions[j * 3u + 2u];
        let A_j = A_params[j];
        let rho_j = rho_params[j];
        
        let r_x = pos_j_x - pos_i_x;
        let r_y = pos_j_y - pos_i_y;
        let r_z = pos_j_z - pos_i_z;
        let r_sq = r_x * r_x + r_y * r_y + r_z * r_z;
        let r = sqrt(r_sq);
        
        if (r > params.cutoff_radius || r < 1e-10) {
            continue;
        }
        
        // Geometric mixing rules
        let A = sqrt(A_i * A_j);
        let rho = (rho_i + rho_j) * 0.5;
        
        // Born-Mayer force: F = (A/ρ) * exp(-r/ρ) * r_hat
        let exp_term = exp(-r / rho);
        let force_magnitude = (A / rho) * exp_term;
        
        let inv_r = 1.0 / r;
        force_x = force_x + force_magnitude * r_x * inv_r;
        force_y = force_y + force_magnitude * r_y * inv_r;
        force_z = force_z + force_magnitude * r_z * inv_r;
    }
    
    forces[i * 3u] = force_x;
    forces[i * 3u + 1u] = force_y;
    forces[i * 3u + 2u] = force_z;
}

// Kernel with energy output
@compute @workgroup_size(256)
fn born_mayer_with_energy(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    
    if (i >= params.n_particles) {
        return;
    }
    
    let pos_i_x = positions[i * 3u];
    let pos_i_y = positions[i * 3u + 1u];
    let pos_i_z = positions[i * 3u + 2u];
    let A_i = A_params[i];
    let rho_i = rho_params[i];
    
    var force_x = 0.0;
    var force_y = 0.0;
    var force_z = 0.0;
    
    for (var j = 0u; j < params.n_particles; j = j + 1u) {
        if (i == j) {
            continue;
        }
        
        let pos_j_x = positions[j * 3u];
        let pos_j_y = positions[j * 3u + 1u];
        let pos_j_z = positions[j * 3u + 2u];
        let A_j = A_params[j];
        let rho_j = rho_params[j];
        
        let r_x = pos_j_x - pos_i_x;
        let r_y = pos_j_y - pos_i_y;
        let r_z = pos_j_z - pos_i_z;
        let r_sq = r_x * r_x + r_y * r_y + r_z * r_z;
        let r = sqrt(r_sq);
        
        if (r > params.cutoff_radius || r < 1e-10) {
            continue;
        }
        
        let A = sqrt(A_i * A_j);
        let rho = (rho_i + rho_j) * 0.5;
        
        let exp_term = exp(-r / rho);
        let force_magnitude = (A / rho) * exp_term;
        
        let inv_r = 1.0 / r;
        force_x = force_x + force_magnitude * r_x * inv_r;
        force_y = force_y + force_magnitude * r_y * inv_r;
        force_z = force_z + force_magnitude * r_z * inv_r;
    }
    
    forces[i * 3u] = force_x;
    forces[i * 3u + 1u] = force_y;
    forces[i * 3u + 2u] = force_z;
}

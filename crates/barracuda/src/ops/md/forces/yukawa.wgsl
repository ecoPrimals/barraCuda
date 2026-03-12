// SPDX-License-Identifier: AGPL-3.0-only
// Yukawa Force Kernel
//
// **Purpose**: Screened Coulomb interactions (e.g., Debye screening in plasmas)
// **Use Case**: Dusty plasmas, colloidal systems, screened electrostatics
// **Range**: Short-range (exponential decay)
//
// **Formula**: F = k * q_i * q_j * exp(-κ*r) / r^2 * r_hat
//   where κ = screening parameter (inverse Debye length)
//   Reduces to Coulomb when κ → 0

@group(0) @binding(0) var<storage, read> positions: array<f32>;
@group(0) @binding(1) var<storage, read> charges: array<f32>;
@group(0) @binding(2) var<storage, read_write> forces: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

struct Params {
    n_particles: u32,
    yukawa_constant: f32,
    kappa: f32,           // Screening parameter
    cutoff_radius: f32,
    epsilon: f32,
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    
    if (i >= params.n_particles) {
        return;
    }
    
    let pos_i = vec3<f32>(
        positions[i * 3u],
        positions[i * 3u + 1u],
        positions[i * 3u + 2u]
    );
    let q_i = charges[i];
    
    var force = vec3<f32>(0.0, 0.0, 0.0);
    
    for (var j = 0u; j < params.n_particles; j = j + 1u) {
        if (i == j) {
            continue;
        }
        
        let pos_j = vec3<f32>(
            positions[j * 3u],
            positions[j * 3u + 1u],
            positions[j * 3u + 2u]
        );
        let q_j = charges[j];
        
        let r_vec = pos_j - pos_i;
        let r2 = dot(r_vec, r_vec) + params.epsilon;
        let r = sqrt(r2);
        
        if (r > params.cutoff_radius) {
            continue;
        }
        
        // Yukawa force: F = k * q_i * q_j * exp(-κ*r) / r^2 * r_hat
        let screening = exp(-params.kappa * r);
        let force_magnitude = params.yukawa_constant * q_i * q_j * screening / r2;
        let r_hat = r_vec / r;
        
        force = force + force_magnitude * r_hat;
    }
    
    forces[i * 3u] = force.x;
    forces[i * 3u + 1u] = force.y;
    forces[i * 3u + 2u] = force.z;
}

// SPDX-License-Identifier: AGPL-3.0-or-later
// Born-Mayer Force Kernel
//
// **Purpose**: Repulsive core interactions (excluded volume)
// **Use Case**: Ionic crystals, hard-core repulsion, steric effects
// **Range**: Very short-range (exponential repulsion)
//
// **Potential**: U(r) = A * exp(-r/ρ)
// **Force**: F = (A/ρ) * exp(-r/ρ) * r_hat
//   where A = repulsion strength
//         ρ = range parameter (softness)

@group(0) @binding(0) var<storage, read> positions: array<f32>;
@group(0) @binding(1) var<storage, read> A_params: array<f32>;    // [N] per-particle A
@group(0) @binding(2) var<storage, read> rho_params: array<f32>;  // [N] per-particle ρ
@group(0) @binding(3) var<storage, read_write> forces: array<f32>;
@group(0) @binding(4) var<uniform> params: Params;

struct Params {
    n_particles: u32,
    cutoff_radius: f32,
    pad1: f32,
    pad2: f32,
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
    let A_i = A_params[i];
    let rho_i = rho_params[i];
    
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
        let A_j = A_params[j];
        let rho_j = rho_params[j];
        
        let r_vec = pos_j - pos_i;
        let r = length(r_vec);
        
        if (r > params.cutoff_radius || r < 1e-6) {
            continue;
        }
        
        // Geometric mixing rules
        let A = sqrt(A_i * A_j);
        let rho = (rho_i + rho_j) * 0.5;
        
        // Born-Mayer force: F = (A/ρ) * exp(-r/ρ) * r_hat
        let exp_term = exp(-r / rho);
        let force_magnitude = (A / rho) * exp_term;
        
        let r_hat = r_vec / r;
        force = force + force_magnitude * r_hat;
    }
    
    forces[i * 3u] = force.x;
    forces[i * 3u + 1u] = force.y;
    forces[i * 3u + 2u] = force.z;
}

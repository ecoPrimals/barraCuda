// SPDX-License-Identifier: AGPL-3.0-only
// Coulomb Force Kernel
//
// **Purpose**: Electrostatic interactions (q1*q2/r)
// **Use Case**: Charged particles, ions, proteins
// **Range**: Long-range (1/r decay)
//
// **Deep Debt Compliance**:
// - Pure WGSL (universal GPU compute)
// - Vectorized calculation (all pairs)
// - Capability-based dispatch
//
// **Formula**: F = k * q_i * q_j / r^2 * r_hat
//   where k = Coulomb constant (can be absorbed into charges)
//   r_hat = unit vector from i to j

@group(0) @binding(0) var<storage, read> positions: array<f32>;  // [N, 3] flattened
@group(0) @binding(1) var<storage, read> charges: array<f32>;    // [N]
@group(0) @binding(2) var<storage, read_write> forces: array<f32>; // [N, 3]
@group(0) @binding(3) var<uniform> params: Params;

struct Params {
    n_particles: u32,
    coulomb_constant: f32,
    cutoff_radius: f32,
    epsilon: f32,  // Softening parameter (avoid singularity)
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    
    if (i >= params.n_particles) {
        return;
    }
    
    // Load position of particle i
    let pos_i = vec3<f32>(
        positions[i * 3u],
        positions[i * 3u + 1u],
        positions[i * 3u + 2u]
    );
    let q_i = charges[i];
    
    // Accumulate forces from all other particles
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
        
        // Compute distance vector (from i to j)
        let r_vec = pos_j - pos_i;
        let r2 = dot(r_vec, r_vec) + params.epsilon;  // Softened
        let r = sqrt(r2);
        
        // Apply cutoff
        if (r > params.cutoff_radius) {
            continue;
        }
        
        // Coulomb force on i from j
        // Standard formula: F = k * q_i * q_j / r² * (r_i - r_j) / |r_i - r_j|
        // Since r_vec = r_j - r_i, the unit vector should be -r_vec/r
        // Like charges (q_i * q_j > 0) → force away from j (along -r_hat)
        // Opposite charges (q_i * q_j < 0) → force toward j (along +r_hat)
        let force_magnitude = params.coulomb_constant * q_i * q_j / r2;
        let r_hat = r_vec / r;
        
        // Apply negative sign for correct direction
        force = force - force_magnitude * r_hat;
    }
    
    // Write accumulated force
    forces[i * 3u] = force.x;
    forces[i * 3u + 1u] = force.y;
    forces[i * 3u + 2u] = force.z;
}

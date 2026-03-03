// Lennard-Jones Force Kernel
//
// **Purpose**: Van der Waals interactions (noble gases, simple liquids)
// **Use Case**: Argon, simple molecular dynamics, coarse-grained models
// **Range**: Short-range (r^-6 attractive, r^-12 repulsive)
//
// **Potential**: U(r) = 4ε[(σ/r)^12 - (σ/r)^6]
// **Force**: F = 24ε/r * [2(σ/r)^12 - (σ/r)^6] * r_hat
//   where ε = depth of potential well
//         σ = distance at which U=0

@group(0) @binding(0) var<storage, read> positions: array<f32>;
@group(0) @binding(1) var<storage, read> sigmas: array<f32>;     // [N] per-particle σ
@group(0) @binding(2) var<storage, read> epsilons: array<f32>;   // [N] per-particle ε
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
    let sigma_i = sigmas[i];
    let epsilon_i = epsilons[i];
    
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
        let sigma_j = sigmas[j];
        let epsilon_j = epsilons[j];
        
        let r_vec = pos_j - pos_i;
        let r2 = dot(r_vec, r_vec);
        let r = sqrt(r2);
        
        if (r > params.cutoff_radius || r < 1e-6) {
            continue;
        }
        
        // Lorentz-Berthelot mixing rules
        let sigma = (sigma_i + sigma_j) * 0.5;
        let epsilon = sqrt(epsilon_i * epsilon_j);
        
        // LJ force: F = 24ε/r * [2(σ/r)^12 - (σ/r)^6] * r_hat
        let sigma_r = sigma / r;
        let sigma_r6 = sigma_r * sigma_r * sigma_r * sigma_r * sigma_r * sigma_r;
        let sigma_r12 = sigma_r6 * sigma_r6;
        
        let force_magnitude = 24.0 * epsilon / r * (2.0 * sigma_r12 - sigma_r6);
        let r_hat = r_vec / r;
        
        force = force + force_magnitude * r_hat;
    }
    
    forces[i * 3u] = force.x;
    forces[i * 3u + 1u] = force.y;
    forces[i * 3u + 2u] = force.z;
}

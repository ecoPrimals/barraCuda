// SPDX-License-Identifier: AGPL-3.0-or-later
// Morse Force Kernel
//
// **Purpose**: Bonded interactions (chemical bonds)
// **Use Case**: Molecular mechanics, reactive MD, bond stretching
// **Range**: Short-range (anharmonic bonded potential)
//
// **Potential**: U(r) = D[1 - exp(-a(r-r0))]^2
// **Force**: F = 2Da[1 - exp(-a(r-r0))] * exp(-a(r-r0)) * r_hat
//   where D = bond dissociation energy
//         a = width parameter
//         r0 = equilibrium bond distance

@group(0) @binding(0) var<storage, read> positions: array<f32>;
@group(0) @binding(1) var<storage, read> bond_pairs: array<f32>;  // [N_bonds, 2]
@group(0) @binding(2) var<storage, read> dissociation_energy: array<f32>;  // [N_bonds]
@group(0) @binding(3) var<storage, read> width_param: array<f32>;  // [N_bonds]
@group(0) @binding(4) var<storage, read> equilibrium_dist: array<f32>;  // [N_bonds]
@group(0) @binding(5) var<storage, read_write> forces: array<atomic<i32>>;  // Atomic for concurrent updates
@group(0) @binding(6) var<uniform> params: Params;

struct Params {
    n_particles: u32,
    n_bonds: u32,
    pad1: f32,
    pad2: f32,
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let bond_idx = global_id.x;
    
    if (bond_idx >= params.n_bonds) {
        return;
    }
    
    // Load bond pair (stored as f32, convert to u32)
    let i = u32(bond_pairs[bond_idx * 2u]);
    let j = u32(bond_pairs[bond_idx * 2u + 1u]);
    
    // Load positions
    let pos_i = vec3<f32>(
        positions[i * 3u],
        positions[i * 3u + 1u],
        positions[i * 3u + 2u]
    );
    let pos_j = vec3<f32>(
        positions[j * 3u],
        positions[j * 3u + 1u],
        positions[j * 3u + 2u]
    );
    
    // Load bond parameters
    let D = dissociation_energy[bond_idx];
    let a = width_param[bond_idx];
    let r0 = equilibrium_dist[bond_idx];
    
    // Compute distance
    let r_vec = pos_j - pos_i;
    let r = length(r_vec);
    
    if (r < 1e-6) {
        return;
    }
    
    // Morse force: F = 2Da[1 - exp(-a(r-r0))] * exp(-a(r-r0)) * r_hat
    let delta_r = r - r0;
    let exp_term = exp(-a * delta_r);
    let force_magnitude = 2.0 * D * a * (1.0 - exp_term) * exp_term;
    
    let r_hat = r_vec / r;
    let force_vec = force_magnitude * r_hat;
    
    // Convert to fixed-point for atomic operations (scale by 1000)
    let force_scaled = vec3<i32>(
        i32(force_vec.x * 1000.0),
        i32(force_vec.y * 1000.0),
        i32(force_vec.z * 1000.0)
    );
    
    // Apply forces (Newton's third law) using atomics
    atomicAdd(&forces[i * 3u], force_scaled.x);
    atomicAdd(&forces[i * 3u + 1u], force_scaled.y);
    atomicAdd(&forces[i * 3u + 2u], force_scaled.z);
    
    atomicAdd(&forces[j * 3u], -force_scaled.x);
    atomicAdd(&forces[j * 3u + 1u], -force_scaled.y);
    atomicAdd(&forces[j * 3u + 2u], -force_scaled.z);
}

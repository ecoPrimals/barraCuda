// Morse Force Kernel (f64) — Science-Grade Bonded Interactions
//
// **Deep Debt Evolution (Feb 16, 2026)**:
// - ✅ Pure WGSL f64 implementation
// - ✅ Native exp(f64) via Vulkan
// - ✅ Hardware-agnostic (NVIDIA/AMD/Intel via WebGPU)
// - ✅ WGSL as unified math language
//
// **Purpose**: Anharmonic bonded interactions (chemical bonds)
// **Use Case**: Molecular mechanics, reactive MD, bond stretching/breaking
// **Range**: Short-range (bonded potential)
//
// **Potential**: U(r) = D·[1 - exp(-a(r-r₀))]²
// **Force**: F = 2Da·[1 - exp(-a(r-r₀))]·exp(-a(r-r₀))·r̂
//   where D = bond dissociation energy
//         a = width parameter (controls potential shape)
//         r₀ = equilibrium bond distance
//
// **Why f64**:
// - Bond energy calculations need precision for thermodynamics
// - Reactive MD requires accurate energy conservation
// - Small force errors accumulate in long simulations

struct Params {
    n_bonds: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> positions: array<f64>;          // [N_particles*3]
@group(0) @binding(1) var<storage, read> bond_pairs: array<u32>;         // [N_bonds*2] (i,j) pairs
@group(0) @binding(2) var<storage, read> dissociation_energy: array<f64>; // [N_bonds] D values
@group(0) @binding(3) var<storage, read> width_param: array<f64>;         // [N_bonds] a values
@group(0) @binding(4) var<storage, read> equilibrium_dist: array<f64>;    // [N_bonds] r0 values
@group(0) @binding(5) var<storage, read_write> bond_forces: array<f64>;   // [N_bonds*6] forces per bond
@group(0) @binding(6) var<uniform> params: Params;

// Helper for f64 constants
fn f64_const(x: f64, c: f32) -> f64 {
    return x - x + f64(c);
}

// Compute Morse force for each bond
// Output: 6 values per bond (fx_i, fy_i, fz_i, fx_j, fy_j, fz_j)
@compute @workgroup_size(256)
fn morse_bonds_f64(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let bond_idx = global_id.x;
    
    if (bond_idx >= params.n_bonds) {
        return;
    }
    
    // Load bond pair
    let i = bond_pairs[bond_idx * 2u];
    let j = bond_pairs[bond_idx * 2u + 1u];
    
    // Load positions
    let xi = positions[i * 3u];
    let yi = positions[i * 3u + 1u];
    let zi = positions[i * 3u + 2u];
    
    let xj = positions[j * 3u];
    let yj = positions[j * 3u + 1u];
    let zj = positions[j * 3u + 2u];
    
    // Load bond parameters
    let D = dissociation_energy[bond_idx];
    let a = width_param[bond_idx];
    let r0 = equilibrium_dist[bond_idx];
    
    // Constants
    let zero = f64_const(D, 0.0);
    let two = f64_const(D, 2.0);
    let min_r_sq = f64_const(D, 1e-20);
    
    // Displacement vector
    let dx = xj - xi;
    let dy = yj - yi;
    let dz = zj - zi;
    
    // Distance
    let r_sq = dx * dx + dy * dy + dz * dz;
    
    if (r_sq < min_r_sq) {
        // Overlapping atoms - zero force
        let out_base = bond_idx * 6u;
        bond_forces[out_base] = zero;
        bond_forces[out_base + 1u] = zero;
        bond_forces[out_base + 2u] = zero;
        bond_forces[out_base + 3u] = zero;
        bond_forces[out_base + 4u] = zero;
        bond_forces[out_base + 5u] = zero;
        return;
    }
    
    let r = sqrt(r_sq);
    
    // Morse force: F = 2Da·[1 - exp(-a(r-r₀))]·exp(-a(r-r₀))·r̂
    let delta_r = r - r0;
    let exp_term = exp(-a * delta_r);
    let one_minus_exp = f64_const(D, 1.0) - exp_term;
    
    // Force magnitude (scalar)
    let force_magnitude = two * D * a * one_minus_exp * exp_term;
    
    // Force on particle i (points toward j if bond stretched)
    let force_over_r = force_magnitude / r;
    let fx_i = force_over_r * dx;
    let fy_i = force_over_r * dy;
    let fz_i = force_over_r * dz;
    
    // Newton's third law: force on j is opposite
    let out_base = bond_idx * 6u;
    bond_forces[out_base] = fx_i;
    bond_forces[out_base + 1u] = fy_i;
    bond_forces[out_base + 2u] = fz_i;
    bond_forces[out_base + 3u] = -fx_i;
    bond_forces[out_base + 4u] = -fy_i;
    bond_forces[out_base + 5u] = -fz_i;
}

// Also compute potential energy per bond
@group(0) @binding(7) var<storage, read_write> bond_energy: array<f64>;  // [N_bonds]

@compute @workgroup_size(256)
fn morse_with_energy_f64(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let bond_idx = global_id.x;
    
    if (bond_idx >= params.n_bonds) {
        return;
    }
    
    let i = bond_pairs[bond_idx * 2u];
    let j = bond_pairs[bond_idx * 2u + 1u];
    
    let xi = positions[i * 3u];
    let yi = positions[i * 3u + 1u];
    let zi = positions[i * 3u + 2u];
    
    let xj = positions[j * 3u];
    let yj = positions[j * 3u + 1u];
    let zj = positions[j * 3u + 2u];
    
    let D = dissociation_energy[bond_idx];
    let a = width_param[bond_idx];
    let r0 = equilibrium_dist[bond_idx];
    
    let zero = f64_const(D, 0.0);
    let one = f64_const(D, 1.0);
    let two = f64_const(D, 2.0);
    let min_r_sq = f64_const(D, 1e-20);
    
    let dx = xj - xi;
    let dy = yj - yi;
    let dz = zj - zi;
    
    let r_sq = dx * dx + dy * dy + dz * dz;
    
    if (r_sq < min_r_sq) {
        let out_base = bond_idx * 6u;
        bond_forces[out_base] = zero;
        bond_forces[out_base + 1u] = zero;
        bond_forces[out_base + 2u] = zero;
        bond_forces[out_base + 3u] = zero;
        bond_forces[out_base + 4u] = zero;
        bond_forces[out_base + 5u] = zero;
        bond_energy[bond_idx] = zero;
        return;
    }
    
    let r = sqrt(r_sq);
    let delta_r = r - r0;
    let exp_term = exp(-a * delta_r);
    let one_minus_exp = one - exp_term;
    
    // Force
    let force_magnitude = two * D * a * one_minus_exp * exp_term;
    let force_over_r = force_magnitude / r;
    let fx_i = force_over_r * dx;
    let fy_i = force_over_r * dy;
    let fz_i = force_over_r * dz;
    
    let out_base = bond_idx * 6u;
    bond_forces[out_base] = fx_i;
    bond_forces[out_base + 1u] = fy_i;
    bond_forces[out_base + 2u] = fz_i;
    bond_forces[out_base + 3u] = -fx_i;
    bond_forces[out_base + 4u] = -fy_i;
    bond_forces[out_base + 5u] = -fz_i;
    
    // Potential energy: U = D·[1 - exp(-a(r-r₀))]²
    bond_energy[bond_idx] = D * one_minus_exp * one_minus_exp;
}

// Reduce bond forces to per-particle forces
// This is a separate kernel to allow bond computation to be parallel
struct ReduceParams {
    n_particles: u32,
    n_bonds: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> reduce_params: ReduceParams;
@group(0) @binding(1) var<storage, read> bond_forces_in: array<f64>;   // [N_bonds*6]
@group(0) @binding(2) var<storage, read> bond_pairs_in: array<u32>;    // [N_bonds*2]
@group(0) @binding(3) var<storage, read_write> particle_forces: array<f64>; // [N_particles*3]

@compute @workgroup_size(256)
fn reduce_bond_forces_f64(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let particle_idx = global_id.x;
    
    if (particle_idx >= reduce_params.n_particles) {
        return;
    }
    
    // Get reference f64 for constants
    var fx = particle_forces[particle_idx * 3u];
    var fy = particle_forces[particle_idx * 3u + 1u];
    var fz = particle_forces[particle_idx * 3u + 2u];
    
    // Sum forces from all bonds involving this particle
    for (var b = 0u; b < reduce_params.n_bonds; b = b + 1u) {
        let i = bond_pairs_in[b * 2u];
        let j = bond_pairs_in[b * 2u + 1u];
        
        if (i == particle_idx) {
            // This particle is the 'i' in the bond
            fx = fx + bond_forces_in[b * 6u];
            fy = fy + bond_forces_in[b * 6u + 1u];
            fz = fz + bond_forces_in[b * 6u + 2u];
        } else if (j == particle_idx) {
            // This particle is the 'j' in the bond
            fx = fx + bond_forces_in[b * 6u + 3u];
            fy = fy + bond_forces_in[b * 6u + 4u];
            fz = fz + bond_forces_in[b * 6u + 5u];
        }
    }
    
    particle_forces[particle_idx * 3u] = fx;
    particle_forces[particle_idx * 3u + 1u] = fy;
    particle_forces[particle_idx * 3u + 2u] = fz;
}

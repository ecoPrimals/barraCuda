// SPDX-License-Identifier: AGPL-3.0-or-later
// Lennard-Jones Force Kernel (f64) — Science-Grade MD
//
// **Deep Debt Evolution (Feb 16, 2026)**:
// - ✅ Pure WGSL f64 implementation
// - ✅ Native sqrt(f64) via Vulkan
// - ✅ Hardware-agnostic (NVIDIA/AMD/Intel via WebGPU)
// - ✅ WGSL as unified math language (same code, any GPU)
//
// **Purpose**: Van der Waals interactions (noble gases, simple liquids)
// **Use Case**: Argon, simple molecular dynamics, coarse-grained models
// **Range**: Short-range (r^-6 attractive, r^-12 repulsive)
//
// **Potential**: U(r) = 4ε[(σ/r)^12 - (σ/r)^6]
// **Force**: F = 24ε/r * [2(σ/r)^12 - (σ/r)^6] * r̂
//   where ε = depth of potential well
//         σ = distance at which U=0
//
// Note: WGSL doesn't support vec3<f64>, so we use scalar operations.

struct Params {
    n_particles: u32,
    _pad0: u32,
    cutoff_radius: f64,
    cutoff_radius_sq: f64,  // Pre-computed for efficiency
}

@group(0) @binding(0) var<storage, read> positions: array<f64>;   // [N*3] x,y,z interleaved
@group(0) @binding(1) var<storage, read> sigmas: array<f64>;      // [N] per-particle σ
@group(0) @binding(2) var<storage, read> epsilons: array<f64>;    // [N] per-particle ε
@group(0) @binding(3) var<storage, read_write> forces: array<f64>; // [N*3] output forces
@group(0) @binding(4) var<uniform> params: Params;

// Helper for f64 constants
fn f64_const(x: f64, c: f32) -> f64 {
    return x - x + f64(c);
}

@compute @workgroup_size(256)
fn lennard_jones_f64(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    
    if (i >= params.n_particles) {
        return;
    }
    
    // Load particle i position
    let xi = positions[i * 3u];
    let yi = positions[i * 3u + 1u];
    let zi = positions[i * 3u + 2u];
    let sigma_i = sigmas[i];
    let epsilon_i = epsilons[i];
    
    // Constants
    let zero = f64_const(xi, 0.0);
    let half = f64_const(xi, 0.5);
    let one = f64_const(xi, 1.0);
    let two = f64_const(xi, 2.0);
    let twenty_four = f64_const(xi, 24.0);
    let min_r_sq = f64_const(xi, 1e-12);  // f64 precision cutoff
    
    // Accumulate forces
    var fx = zero;
    var fy = zero;
    var fz = zero;
    
    let cutoff_sq = params.cutoff_radius_sq;
    
    for (var j = 0u; j < params.n_particles; j = j + 1u) {
        if (i == j) {
            continue;
        }
        
        // Load particle j position
        let xj = positions[j * 3u];
        let yj = positions[j * 3u + 1u];
        let zj = positions[j * 3u + 2u];
        
        // Displacement vector r_ij = r_j - r_i
        let dx = xj - xi;
        let dy = yj - yi;
        let dz = zj - zi;
        
        // Distance squared
        let r_sq = dx * dx + dy * dy + dz * dz;
        
        // Skip if outside cutoff or too close
        if (r_sq > cutoff_sq || r_sq < min_r_sq) {
            continue;
        }
        
        let r = sqrt(r_sq);
        
        // Lorentz-Berthelot mixing rules
        let sigma_j = sigmas[j];
        let epsilon_j = epsilons[j];
        let sigma = (sigma_i + sigma_j) * half;
        let epsilon = sqrt(epsilon_i * epsilon_j);
        
        // LJ force calculation
        // F = 24ε/r * [2(σ/r)^12 - (σ/r)^6] * r̂
        let sigma_r = sigma / r;
        let sigma_r_sq = sigma_r * sigma_r;
        let sigma_r6 = sigma_r_sq * sigma_r_sq * sigma_r_sq;
        let sigma_r12 = sigma_r6 * sigma_r6;
        
        // Force magnitude (scalar): positive = repulsive
        let force_over_r = twenty_four * epsilon / r_sq * (two * sigma_r12 - sigma_r6);
        
        // Force vector components: F_i = -F * r_hat_ij (repulsion pushes AWAY from j)
        // r_hat_ij points from i to j (along dx,dy,dz), so negate for repulsive force
        fx = fx - force_over_r * dx;
        fy = fy - force_over_r * dy;
        fz = fz - force_over_r * dz;
    }
    
    // Write forces
    forces[i * 3u] = fx;
    forces[i * 3u + 1u] = fy;
    forces[i * 3u + 2u] = fz;
}

// Shifted potential variant: smoother cutoff behavior
// U_shifted(r) = U_LJ(r) - U_LJ(r_c) for r < r_c
@compute @workgroup_size(256)
fn lennard_jones_shifted_f64(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    
    if (i >= params.n_particles) {
        return;
    }
    
    let xi = positions[i * 3u];
    let yi = positions[i * 3u + 1u];
    let zi = positions[i * 3u + 2u];
    let sigma_i = sigmas[i];
    let epsilon_i = epsilons[i];
    
    let zero = f64_const(xi, 0.0);
    let half = f64_const(xi, 0.5);
    let two = f64_const(xi, 2.0);
    let twenty_four = f64_const(xi, 24.0);
    let min_r_sq = f64_const(xi, 1e-12);
    
    var fx = zero;
    var fy = zero;
    var fz = zero;
    
    let cutoff_sq = params.cutoff_radius_sq;
    let r_c = params.cutoff_radius;
    
    for (var j = 0u; j < params.n_particles; j = j + 1u) {
        if (i == j) {
            continue;
        }
        
        let xj = positions[j * 3u];
        let yj = positions[j * 3u + 1u];
        let zj = positions[j * 3u + 2u];
        
        let dx = xj - xi;
        let dy = yj - yi;
        let dz = zj - zi;
        
        let r_sq = dx * dx + dy * dy + dz * dz;
        
        if (r_sq > cutoff_sq || r_sq < min_r_sq) {
            continue;
        }
        
        let r = sqrt(r_sq);
        
        let sigma_j = sigmas[j];
        let epsilon_j = epsilons[j];
        let sigma = (sigma_i + sigma_j) * half;
        let epsilon = sqrt(epsilon_i * epsilon_j);
        
        let sigma_r = sigma / r;
        let sigma_r_sq = sigma_r * sigma_r;
        let sigma_r6 = sigma_r_sq * sigma_r_sq * sigma_r_sq;
        let sigma_r12 = sigma_r6 * sigma_r6;
        
        // Force with shifted potential (same formula, just well-defined at cutoff)
        let force_over_r = twenty_four * epsilon / r_sq * (two * sigma_r12 - sigma_r6);
        
        // Repulsion pushes AWAY from j
        fx = fx - force_over_r * dx;
        fy = fy - force_over_r * dy;
        fz = fz - force_over_r * dz;
    }
    
    forces[i * 3u] = fx;
    forces[i * 3u + 1u] = fy;
    forces[i * 3u + 2u] = fz;
}

// Also compute potential energy (useful for thermodynamic properties)
@group(0) @binding(5) var<storage, read_write> potential_energy: array<f64>;  // [N] per-particle

@compute @workgroup_size(256)
fn lennard_jones_with_energy_f64(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    
    if (i >= params.n_particles) {
        return;
    }
    
    let xi = positions[i * 3u];
    let yi = positions[i * 3u + 1u];
    let zi = positions[i * 3u + 2u];
    let sigma_i = sigmas[i];
    let epsilon_i = epsilons[i];
    
    let zero = f64_const(xi, 0.0);
    let half = f64_const(xi, 0.5);
    let two = f64_const(xi, 2.0);
    let four = f64_const(xi, 4.0);
    let twenty_four = f64_const(xi, 24.0);
    let min_r_sq = f64_const(xi, 1e-12);
    
    var fx = zero;
    var fy = zero;
    var fz = zero;
    var energy = zero;
    
    let cutoff_sq = params.cutoff_radius_sq;
    
    for (var j = 0u; j < params.n_particles; j = j + 1u) {
        if (i == j) {
            continue;
        }
        
        let xj = positions[j * 3u];
        let yj = positions[j * 3u + 1u];
        let zj = positions[j * 3u + 2u];
        
        let dx = xj - xi;
        let dy = yj - yi;
        let dz = zj - zi;
        
        let r_sq = dx * dx + dy * dy + dz * dz;
        
        if (r_sq > cutoff_sq || r_sq < min_r_sq) {
            continue;
        }
        
        let r = sqrt(r_sq);
        
        let sigma_j = sigmas[j];
        let epsilon_j = epsilons[j];
        let sigma = (sigma_i + sigma_j) * half;
        let epsilon = sqrt(epsilon_i * epsilon_j);
        
        let sigma_r = sigma / r;
        let sigma_r_sq = sigma_r * sigma_r;
        let sigma_r6 = sigma_r_sq * sigma_r_sq * sigma_r_sq;
        let sigma_r12 = sigma_r6 * sigma_r6;
        
        // Force: repulsion pushes AWAY from j
        let force_over_r = twenty_four * epsilon / r_sq * (two * sigma_r12 - sigma_r6);
        fx = fx - force_over_r * dx;
        fy = fy - force_over_r * dy;
        fz = fz - force_over_r * dz;
        
        // Potential energy: U = 4ε[(σ/r)^12 - (σ/r)^6]
        // Divide by 2 to avoid double counting
        energy = energy + half * four * epsilon * (sigma_r12 - sigma_r6);
    }
    
    forces[i * 3u] = fx;
    forces[i * 3u + 1u] = fy;
    forces[i * 3u + 2u] = fz;
    potential_energy[i] = energy;
}

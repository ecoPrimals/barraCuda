// SPDX-License-Identifier: AGPL-3.0-or-later
// Short-range erfc-damped Coulomb forces for PPPM (f64)
//
// **Physics**: Real-space Ewald sum with erfc damping
// **Formula**: F_ij = q_i*q_j * [erfc(αr)/r² + 2α/√π * exp(-α²r²)/r] * r̂
// **Precision**: Full f64 using native builtins + math_f64.wgsl for erf
// **Use Case**: PPPM real-space short-range contribution
//
// **Performance (Feb 15 2026 hotSpring finding)**:
// Native sqrt(f64): 1.5× faster than math_f64 software sqrt_f64
// Native exp(f64): 2.2× faster than math_f64 software exp_f64
//
// Requires: math_f64.wgsl preamble (erf_f64 for erfc, round_f64 for PBC)
//
// Bindings:
//   0: positions   [N*3] f64, read     — particle positions
//   1: charges     [N] f64, read       — particle charges
//   2: forces      [N*3] f64, r/w      — output forces (accumulated)
//   3: pe_buf      [N] f64, write      — per-particle potential energy
//   4: params      [16] f64, read      — simulation parameters

@group(0) @binding(0) var<storage, read> positions: array<f64>;
@group(0) @binding(1) var<storage, read> charges: array<f64>;
@group(0) @binding(2) var<storage, read_write> forces: array<f64>;
@group(0) @binding(3) var<storage, read_write> pe_buf: array<f64>;
@group(0) @binding(4) var<storage, read> params: array<f64>;

// params layout:
//   [0] = n_particles
//   [1] = alpha (Ewald parameter)
//   [2] = cutoff_sq (rc² in reduced units)
//   [3] = box_x, [4] = box_y, [5] = box_z
//   [6] = prefactor (typically 1.0 for reduced units, or ke for SI)

// Complementary error function: erfc(x) = 1 - erf(x)
fn erfc_f64(x: f64) -> f64 {
    return f64_const(x, 1.0) - erf_f64(x);
}

// PBC minimum image
fn pbc_delta(delta: f64, box_size: f64) -> f64 {
    return delta - box_size * round_f64(delta / box_size);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let n = u32(params[0]);
    if (i >= n) { return; }
    
    let alpha = params[1];
    let cutoff_sq = params[2];
    let box_x = params[3];
    let box_y = params[4];
    let box_z = params[5];
    let prefactor = params[6];
    
    // Constants
    let zero = positions[0] - positions[0];
    let one = zero + 1.0;
    let two = zero + 2.0;
    let pi = zero + 3.14159265358979323846;
    let sqrt_pi_inv = one / sqrt(pi);  // 1/√π (native f64 builtin)
    let two_alpha_sqrt_pi = two * alpha * sqrt_pi_inv;
    
    // Particle i data
    let xi = positions[i * 3u];
    let yi = positions[i * 3u + 1u];
    let zi = positions[i * 3u + 2u];
    let qi = charges[i];
    
    // Accumulate force and PE
    var fx = zero;
    var fy = zero;
    var fz = zero;
    var pe = zero;
    
    // All-pairs with cutoff
    for (var j = 0u; j < n; j = j + 1u) {
        if (i == j) { continue; }
        
        let xj = positions[j * 3u];
        let yj = positions[j * 3u + 1u];
        let zj = positions[j * 3u + 2u];
        let qj = charges[j];
        
        // PBC minimum image
        let dx = pbc_delta(xj - xi, box_x);
        let dy = pbc_delta(yj - yi, box_y);
        let dz = pbc_delta(zj - zi, box_z);
        
        let r_sq = dx * dx + dy * dy + dz * dz;
        
        if (r_sq > cutoff_sq) { continue; }
        
        // Native f64 builtins: 1.5-2.2× faster than math_f64 software
        let r = sqrt(r_sq);
        let inv_r = one / r;
        let inv_r2 = inv_r * inv_r;
        
        let alpha_r = alpha * r;
        
        // erfc(αr)/r (erfc uses erf_f64 from math_f64.wgsl)
        let erfc_term = erfc_f64(alpha_r) * inv_r;
        
        // exp(-α²r²) - native f64 builtin
        let exp_term = exp(-alpha_r * alpha_r);
        
        // Force magnitude: q_i*q_j * prefactor * [erfc(αr)/r² + 2α/√π * exp(-α²r²)/r]
        // = q_i*q_j * prefactor * inv_r * [erfc(αr)/r + 2α/√π * exp(-α²r²)]
        let force_scalar = qi * qj * prefactor * inv_r * (
            erfc_term * inv_r + two_alpha_sqrt_pi * exp_term
        );
        
        // Force on i due to j (Coulomb: like charges repel)
        // r̂_ij = (dx, dy, dz)/r points from i to j
        // For repulsion, force on i is in direction of -r̂_ij
        fx = fx - force_scalar * dx * inv_r;
        fy = fy - force_scalar * dy * inv_r;
        fz = fz - force_scalar * dz * inv_r;
        
        // Potential energy: U = q_i*q_j * prefactor * erfc(αr)/r (half-counted)
        pe = pe + (zero + 0.5) * qi * qj * prefactor * erfc_term;
    }
    
    // Accumulate to output (add to any existing k-space forces)
    forces[i * 3u]      = forces[i * 3u] + fx;
    forces[i * 3u + 1u] = forces[i * 3u + 1u] + fy;
    forces[i * 3u + 2u] = forces[i * 3u + 2u] + fz;
    pe_buf[i] = pe;
}

// Self-energy correction kernel
// E_self = -α/√π * Σ_i q_i²
@compute @workgroup_size(256)
fn self_energy(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let n = u32(params[0]);
    if (i >= n) { return; }
    
    let alpha = params[1];
    let prefactor = params[6];
    let qi = charges[i];
    
    let zero = charges[0] - charges[0];
    let pi = zero + 3.14159265358979323846;
    let sqrt_pi_inv = (zero + 1.0) / sqrt(pi);  // native f64 builtin
    
    // Self-energy contribution for particle i
    // E_self_i = -α/√π * prefactor * q_i²
    pe_buf[i] = pe_buf[i] - alpha * sqrt_pi_inv * prefactor * qi * qi;
}

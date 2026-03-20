// SPDX-License-Identifier: AGPL-3.0-or-later
// Green's function application for PPPM k-space (f64)
//
// **Physics**: Applies Green's function in k-space: φ(k) = G(k) * ρ(k)
// **Formula**: G(k) = 4π/k² × exp(-k²/(4α²)) × influence(k)
// **Precision**: Full f64 using native builtins where available
// **Use Case**: PPPM k-space step after FFT
//
// **Performance (Feb 15 2026 hotSpring finding)**:
// Native exp(f64): 2.2× faster than math_f64 software exp_f64
//
// Requires: math_f64.wgsl preamble (for complex ops if needed)
//
// Complex multiplication: (a + bi)(c + di) = (ac - bd) + (ad + bc)i
//
// Bindings:
//   0: rho_k_re    [Kx*Ky*Kz] f64, read  — charge density FFT real part
//   1: rho_k_im    [Kx*Ky*Kz] f64, read  — charge density FFT imag part
//   2: phi_k_re    [Kx*Ky*Kz] f64, write — potential FFT real part
//   3: phi_k_im    [Kx*Ky*Kz] f64, write — potential FFT imag part
//   4: greens      [Kx*Ky*Kz] f64, read  — precomputed G(k) values (real)
//   5: params      [16] f64, read        — simulation parameters

@group(0) @binding(0) var<storage, read> rho_k_re: array<f64>;
@group(0) @binding(1) var<storage, read> rho_k_im: array<f64>;
@group(0) @binding(2) var<storage, read_write> phi_k_re: array<f64>;
@group(0) @binding(3) var<storage, read_write> phi_k_im: array<f64>;
@group(0) @binding(4) var<storage, read> greens: array<f64>;
@group(0) @binding(5) var<storage, read> params: array<f64>;

// params layout:
//   [0] = total mesh size (Kx * Ky * Kz)

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = u32(params[0]);
    if (idx >= total) { return; }
    
    // G(k) is purely real for symmetric systems
    let g = greens[idx];
    let rho_re = rho_k_re[idx];
    let rho_im = rho_k_im[idx];
    
    // φ(k) = G(k) * ρ(k)
    // Since G is real: φ_re = G * ρ_re, φ_im = G * ρ_im
    phi_k_re[idx] = g * rho_re;
    phi_k_im[idx] = g * rho_im;
}

// Alternative entry point: compute G(k) on-the-fly instead of precomputed
// This is slower but uses less memory
@compute @workgroup_size(64)
fn apply_greens_inline(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let total = u32(params[0]);
    if (idx >= total) { return; }
    
    // Additional params for inline G computation:
    //   [1] = kx, [2] = ky, [3] = kz (mesh dims)
    //   [4] = box_x, [5] = box_y, [6] = box_z
    //   [7] = alpha (Ewald parameter)
    //   [8] = order (B-spline order)
    let kx = u32(params[1]);
    let ky = u32(params[2]);
    let kz = u32(params[3]);
    let box_x = params[4];
    let box_y = params[5];
    let box_z = params[6];
    let alpha = params[7];
    let order = u32(params[8]);
    
    // Convert linear index to 3D
    let iz = idx % kz;
    let iy = (idx / kz) % ky;
    let ix = idx / (ky * kz);
    
    let zero = rho_k_re[0] - rho_k_re[0];
    let one = zero + 1.0;
    let pi = zero + 3.14159265358979323846;
    let two_pi = pi * (zero + 2.0);
    
    // Compute k-vector components with Nyquist handling
    var kx_f = f64(ix);
    var ky_f = f64(iy);
    var kz_f = f64(iz);
    if (ix > kx / 2u) { kx_f = kx_f - f64(kx); }
    if (iy > ky / 2u) { ky_f = ky_f - f64(ky); }
    if (iz > kz / 2u) { kz_f = kz_f - f64(kz); }
    
    // k = 2π * n / L
    let kvec_x = two_pi * kx_f / box_x;
    let kvec_y = two_pi * ky_f / box_y;
    let kvec_z = two_pi * kz_f / box_z;
    let k_sq = kvec_x * kvec_x + kvec_y * kvec_y + kvec_z * kvec_z;
    
    // Handle k = 0 (DC component)
    var g = zero;
    if (k_sq > zero + 1e-30) {
        // Coulomb: 4π/k²
        let coulomb = (zero + 4.0) * pi / k_sq;
        
        // Ewald damping: exp(-k²/(4α²)) - native f64 builtin
        let ewald = exp(-k_sq / ((zero + 4.0) * alpha * alpha));
        
        // Simplified G (without influence function for this kernel)
        g = coulomb * ewald;
    }
    
    let rho_re = rho_k_re[idx];
    let rho_im = rho_k_im[idx];
    
    phi_k_re[idx] = g * rho_re;
    phi_k_im[idx] = g * rho_im;
}

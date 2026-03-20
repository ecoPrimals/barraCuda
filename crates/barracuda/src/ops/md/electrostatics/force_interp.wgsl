// SPDX-License-Identifier: AGPL-3.0-or-later
// Force interpolation kernel for PPPM (f64)
//
// **Physics**: Interpolates potential gradient from mesh to particle positions
// **Formula**: F_i = -q_i * ∇φ(r_i) where gradient uses B-spline derivatives
// **Precision**: Full f64 via math_f64.wgsl preamble
// **Use Case**: PPPM final step - mesh to particle forces
//
// Requires: math_f64.wgsl preamble, B-spline coefficients and derivatives
//
// Bindings:
//   0: charges     [N] f64, read        — particle charges
//   1: coeffs      [N*order*3] f64, read — B-spline coefficients
//   2: derivs      [N*order*3] f64, read — B-spline derivatives
//   3: base_idx    [N*3] i32, read      — base mesh indices
//   4: potential   [Kx*Ky*Kz] f64, read — potential mesh φ(n)
//   5: forces      [N*3] f64, write     — output forces (fx, fy, fz)
//   6: params      [16] f64, read       — simulation parameters

@group(0) @binding(0) var<storage, read> charges: array<f64>;
@group(0) @binding(1) var<storage, read> coeffs: array<f64>;
@group(0) @binding(2) var<storage, read> derivs: array<f64>;
@group(0) @binding(3) var<storage, read> base_idx: array<i32>;
@group(0) @binding(4) var<storage, read> potential: array<f64>;
@group(0) @binding(5) var<storage, read_write> forces: array<f64>;
@group(0) @binding(6) var<storage, read> params: array<f64>;

// params layout:
//   [0] = n_particles
//   [1] = order
//   [2] = mesh_kx, [3] = mesh_ky, [4] = mesh_kz
//   [5] = box_x, [6] = box_y, [7] = box_z

// Wrap index to [0, K) with periodic boundary
fn wrap_idx(idx: i32, k: u32) -> u32 {
    var wrapped = idx % i32(k);
    if (wrapped < 0) { wrapped = wrapped + i32(k); }
    return u32(wrapped);
}

// 3D mesh index to linear index
fn mesh_index(ix: u32, iy: u32, iz: u32, kx: u32, ky: u32, kz: u32) -> u32 {
    return ix * ky * kz + iy * kz + iz;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let n = u32(params[0]);
    if (i >= n) { return; }
    
    let order = u32(params[1]);
    let kx = u32(params[2]);
    let ky = u32(params[3]);
    let kz = u32(params[4]);
    let box_x = params[5];
    let box_y = params[6];
    let box_z = params[7];
    
    let q = charges[i];
    let stride = order * 3u;
    
    // Mesh spacing (for gradient scaling)
    let hx = box_x / f64(kx);
    let hy = box_y / f64(ky);
    let hz = box_z / f64(kz);
    
    // Base indices
    let bx = base_idx[i * 3u];
    let by = base_idx[i * 3u + 1u];
    let bz = base_idx[i * 3u + 2u];
    
    // Accumulate gradient components
    let zero = charges[0] - charges[0];  // f64 zero
    var grad_x = zero;
    var grad_y = zero;
    var grad_z = zero;
    
    // Triple loop over B-spline support
    for (var jx = 0u; jx < order; jx = jx + 1u) {
        let wx = coeffs[i * stride + jx];
        let dwx = derivs[i * stride + jx];
        let ix = wrap_idx(bx + i32(jx), kx);
        
        for (var jy = 0u; jy < order; jy = jy + 1u) {
            let wy = coeffs[i * stride + order + jy];
            let dwy = derivs[i * stride + order + jy];
            let iy = wrap_idx(by + i32(jy), ky);
            
            for (var jz = 0u; jz < order; jz = jz + 1u) {
                let wz = coeffs[i * stride + 2u * order + jz];
                let dwz = derivs[i * stride + 2u * order + jz];
                let iz = wrap_idx(bz + i32(jz), kz);
                
                // Get potential at this mesh point
                let midx = mesh_index(ix, iy, iz, kx, ky, kz);
                let phi = potential[midx];
                
                // Gradient components: ∂φ/∂x ≈ Σ φ(n) * dW_x/dx * W_y * W_z
                // Scale: dW/dx = dW/du * du/dx = dW/du * K/L
                grad_x = grad_x + phi * dwx * wy * wz * f64(kx) / box_x;
                grad_y = grad_y + phi * wx * dwy * wz * f64(ky) / box_y;
                grad_z = grad_z + phi * wx * wy * dwz * f64(kz) / box_z;
            }
        }
    }
    
    // Force = -q * ∇φ
    forces[i * 3u]      = -q * grad_x;
    forces[i * 3u + 1u] = -q * grad_y;
    forces[i * 3u + 2u] = -q * grad_z;
}

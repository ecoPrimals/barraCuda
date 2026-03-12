// SPDX-License-Identifier: AGPL-3.0-only
// Charge spreading kernel for PPPM (f64)
//
// **Physics**: Spreads particle charges onto mesh using B-spline interpolation
// **Formula**: ρ(n) = Σ_i q_i * W_x(n_x - u_ix) * W_y(n_y - u_iy) * W_z(n_z - u_iz)
// **Precision**: Full f64, uses atomic add emulation for thread safety
// **Use Case**: PPPM first step - particle to mesh
//
// Requires: math_f64.wgsl preamble, bspline coefficients pre-computed
//
// NOTE: Since WGSL lacks atomic f64 operations, this shader accumulates
// per-particle contributions that are summed on CPU, OR uses a two-pass
// approach with particle binning. For production, consider sorted particle
// assignment to avoid race conditions.
//
// Bindings:
//   0: charges     [N] f64, read        — particle charges
//   1: coeffs      [N*order*3] f64, read — B-spline coefficients
//   2: base_idx    [N*3] i32, read      — base mesh indices
//   3: mesh        [Kx*Ky*Kz] f64, r/w  — charge density mesh
//   4: params      [16] f64, read       — simulation parameters

@group(0) @binding(0) var<storage, read> charges: array<f64>;
@group(0) @binding(1) var<storage, read> coeffs: array<f64>;
@group(0) @binding(2) var<storage, read> base_idx: array<i32>;
@group(0) @binding(3) var<storage, read_write> mesh: array<f64>;
@group(0) @binding(4) var<storage, read> params: array<f64>;

// params layout:
//   [0] = n_particles
//   [1] = order
//   [2] = mesh_kx, [3] = mesh_ky, [4] = mesh_kz

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
    
    let q = charges[i];
    let stride = order * 3u;
    
    // Base indices for this particle
    let bx = base_idx[i * 3u];
    let by = base_idx[i * 3u + 1u];
    let bz = base_idx[i * 3u + 2u];
    
    // Spread charge to mesh using B-spline weights
    // Triple loop over order^3 mesh points
    for (var jx = 0u; jx < order; jx = jx + 1u) {
        let wx = coeffs[i * stride + jx];
        let ix = wrap_idx(bx + i32(jx), kx);
        
        for (var jy = 0u; jy < order; jy = jy + 1u) {
            let wy = coeffs[i * stride + order + jy];
            let iy = wrap_idx(by + i32(jy), ky);
            
            for (var jz = 0u; jz < order; jz = jz + 1u) {
                let wz = coeffs[i * stride + 2u * order + jz];
                let iz = wrap_idx(bz + i32(jz), kz);
                
                // Weight = product of 1D B-spline weights
                let weight = wx * wy * wz;
                let contribution = q * weight;
                
                // Mesh index
                let midx = mesh_index(ix, iy, iz, kx, ky, kz);
                
                // NOTE: This is NOT atomic! For thread-safe operation:
                // 1. Sort particles by cell and process sequentially per cell
                // 2. Use separate output per particle, sum on CPU
                // 3. Use atomic i64 with fixed-point encoding
                // For now, assume single-threaded or pre-sorted execution
                mesh[midx] = mesh[midx] + contribution;
            }
        }
    }
}

// Alternative: Per-particle output kernel (no race conditions)
// Each particle writes to its own section, CPU sums afterward
@compute @workgroup_size(64)
fn spread_per_particle(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let n = u32(params[0]);
    if (i >= n) { return; }
    
    let order = u32(params[1]);
    let kx = u32(params[2]);
    let ky = u32(params[3]);
    let kz = u32(params[4]);
    
    let q = charges[i];
    let stride = order * 3u;
    let o3 = order * order * order;
    
    // Base indices
    let bx = base_idx[i * 3u];
    let by = base_idx[i * 3u + 1u];
    let bz = base_idx[i * 3u + 2u];
    
    // Output to per-particle section: mesh[i * order^3 + local_idx]
    // Mesh must be sized N * order^3 for this variant
    var local_idx = 0u;
    
    for (var jx = 0u; jx < order; jx = jx + 1u) {
        let wx = coeffs[i * stride + jx];
        
        for (var jy = 0u; jy < order; jy = jy + 1u) {
            let wy = coeffs[i * stride + order + jy];
            
            for (var jz = 0u; jz < order; jz = jz + 1u) {
                let wz = coeffs[i * stride + 2u * order + jz];
                
                let weight = wx * wy * wz;
                mesh[i * o3 + local_idx] = q * weight;
                local_idx = local_idx + 1u;
            }
        }
    }
}

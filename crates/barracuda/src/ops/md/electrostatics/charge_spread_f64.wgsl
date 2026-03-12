// SPDX-License-Identifier: AGPL-3.0-only
// charge_spread_f64.wgsl — Charge spreading to mesh for PPPM
//
// **Physics**: Spread particle charges onto 3D mesh using B-spline weights for PPPM.
// ρ(n) = Σ_i q_i * W(u_i - n) where W is product of 1D B-splines.
//
// **Algorithm**: Each thread spreads one particle's charge onto the order³ nearest
// grid points. Weight = M_order(u_x - n_x) * M_order(u_y - n_y) * M_order(u_z - n_z).
//
// **Note**: WGSL has no atomic f64. Uses non-atomic add; for production use
// two-pass reduction or sorted particle assignment to avoid race conditions.
// Alternatively, mesh can use atomic<u32> with fixed-point encoding.
//
// **Precision**: f64 via bitcast<f64>(vec2<u32>)
// **Workgroup**: @compute @workgroup_size(256)
//
// Bindings:
//   0: charges    array<vec2<u32>>  read       — particle charges (f64)
//   1: frac_coords array<vec2<u32>>  read       — [u,v,w] per particle, 3 f64s each
//   2: mesh       array<vec2<u32>>  read_write — grid (grid_x * grid_y * grid_z f64s)
//
// Params: { n_particles, grid_x, grid_y, grid_z, order }
//
// Reference: Hockney & Eastwood; PPPM method

@group(0) @binding(0) var<storage, read> charges: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read> frac_coords: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read_write> mesh: array<vec2<u32>>;
@group(0) @binding(3) var<uniform> params: Params;

struct Params {
    n_particles: u32,
    grid_x: u32,
    grid_y: u32,
    grid_z: u32,
    order: u32,
}

fn wrap_idx(idx: i32, k: u32) -> u32 {
    var w = idx % i32(k);
    if (w < 0) { w = w + i32(k); }
    return u32(w);
}

fn mesh_index(ix: u32, iy: u32, iz: u32, gx: u32, gy: u32, gz: u32) -> u32 {
    return ix * gy * gz + iy * gz + iz;
}

// M_4 cardinal B-spline
fn M4(u: f64) -> f64 {
    let z = 0.0;
    if (u <= z || u >= 4.0) { return z; }
    let u2 = u * u;
    let u3 = u2 * u;
    if (u < 1.0) { return u3 / 6.0; }
    if (u < 2.0) { return (-3.0*u3 + 12.0*u2 - 12.0*u + 4.0) / 6.0; }
    if (u < 3.0) { return (3.0*u3 - 24.0*u2 + 60.0*u - 44.0) / 6.0; }
    return (4.0 - u) * (4.0 - u) * (4.0 - u) / 6.0;
}

fn bspline_w(order: u32, u: f64) -> f64 {
    if (order == 4u) { return M4(u); }
    // Simplified: order 4 only for now; extend M5, M6 if needed
    return M4(u);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    let n = params.n_particles;
    let gx = params.grid_x;
    let gy = params.grid_y;
    let gz = params.grid_z;
    let order = params.order;
    if (i >= n) {
        return;
    }

    let q = bitcast<f64>(charges[i]);
    let base = i * 3u;
    let ux = bitcast<f64>(frac_coords[base]);
    let uy = bitcast<f64>(frac_coords[base + 1u]);
    let uz = bitcast<f64>(frac_coords[base + 2u]);

    // Base grid index: floor(u) - (order - 1) for support
    let base_x = i32(floor(ux)) - i32(order) + 1;
    let base_y = i32(floor(uy)) - i32(order) + 1;
    let base_z = i32(floor(uz)) - i32(order) + 1;

    // Spread to order³ grid points
    for (var jx = 0u; jx < order; jx = jx + 1u) {
        let arg_x = ux - f64(base_x + i32(jx));
        let wx = bspline_w(order, arg_x);
        let ix = wrap_idx(base_x + i32(jx), gx);

        for (var jy = 0u; jy < order; jy = jy + 1u) {
            let arg_y = uy - f64(base_y + i32(jy));
            let wy = bspline_w(order, arg_y);
            let iy = wrap_idx(base_y + i32(jy), gy);

            for (var jz = 0u; jz < order; jz = jz + 1u) {
                let arg_z = uz - f64(base_z + i32(jz));
                let wz = bspline_w(order, arg_z);
                let iz = wrap_idx(base_z + i32(jz), gz);

                let weight = wx * wy * wz;
                let contribution = q * weight;

                let midx = mesh_index(ix, iy, iz, gx, gy, gz);
                let old_val = bitcast<f64>(mesh[midx]);
                mesh[midx] = bitcast<vec2<u32>>(old_val + contribution);
            }
        }
    }
}

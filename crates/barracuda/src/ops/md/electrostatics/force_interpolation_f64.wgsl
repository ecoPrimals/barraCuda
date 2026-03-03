// force_interpolation_f64.wgsl — Interpolate forces from mesh to particles (PPPM)
//
// **Physics**: F_i = -q_i * E(r_i). E = ∇φ is stored on mesh (Ex, Ey, Ez).
// Interpolate E at particle position: E(r) = Σ_m E(m) * W(r - m) using B-spline W.
// Force = -q * E_interp.
//
// **Algorithm**: Each thread handles one particle. Sample Ex, Ey, Ez at order³
// mesh points, weight by B-spline M_4, trilinear-style sum. Force = -q * (Ex, Ey, Ez).
//
// **Precision**: f64 via bitcast<f64>(vec2<u32>)
// **Workgroup**: @compute @workgroup_size(256)
//
// Bindings:
//   0: mesh_grad  array<vec2<u32>>  read — potential gradient [Ex_grid | Ey_grid | Ez_grid]
//      Layout: Ex[gx*gy*gz], Ey[gx*gy*gz], Ez[gx*gy*gz] — 3 * gx*gy*gz f64s
//   1: frac_coords array<vec2<u32>>  read — [u,v,w] per particle
//   2: charges    array<vec2<u32>>  read — particle charges
//   3: forces     array<vec2<u32>>  read_write — [fx,fy,fz] per particle
//
// Params: { n_particles, grid_x, grid_y, grid_z, order }
//
// Reference: Hockney & Eastwood; PPPM force interpolation

@group(0) @binding(0) var<storage, read> mesh_grad: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read> frac_coords: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read> charges: array<vec2<u32>>;
@group(0) @binding(3) var<storage, read_write> forces: array<vec2<u32>>;
@group(0) @binding(4) var<uniform> params: Params;

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

// M_4 cardinal B-spline for interpolation
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

    let base_x = i32(floor(ux)) - i32(order) + 1;
    let base_y = i32(floor(uy)) - i32(order) + 1;
    let base_z = i32(floor(uz)) - i32(order) + 1;

    let grid_size = gx * gy * gz;
    let zero = q - q;

    var grad_x = zero;
    var grad_y = zero;
    var grad_z = zero;

    // Interpolate E using B-spline weights
    for (var jx = 0u; jx < order; jx = jx + 1u) {
        let arg_x = ux - f64(base_x + i32(jx));
        let wx = M4(arg_x);
        let ix = wrap_idx(base_x + i32(jx), gx);

        for (var jy = 0u; jy < order; jy = jy + 1u) {
            let arg_y = uy - f64(base_y + i32(jy));
            let wy = M4(arg_y);
            let iy = wrap_idx(base_y + i32(jy), gy);

            for (var jz = 0u; jz < order; jz = jz + 1u) {
                let arg_z = uz - f64(base_z + i32(jz));
                let wz = M4(arg_z);
                let iz = wrap_idx(base_z + i32(jz), gz);

                let midx = mesh_index(ix, iy, iz, gx, gy, gz);
                let ex = bitcast<f64>(mesh_grad[midx]);
                let ey = bitcast<f64>(mesh_grad[grid_size + midx]);
                let ez = bitcast<f64>(mesh_grad[2u * grid_size + midx]);

                let w = wx * wy * wz;
                grad_x = grad_x + ex * w;
                grad_y = grad_y + ey * w;
                grad_z = grad_z + ez * w;
            }
        }
    }

    // Force = -q * ∇φ (box-normalized gradient)
    forces[base] = bitcast<vec2<u32>>(-q * grad_x);
    forces[base + 1u] = bitcast<vec2<u32>>(-q * grad_y);
    forces[base + 2u] = bitcast<vec2<u32>>(-q * grad_z);
}

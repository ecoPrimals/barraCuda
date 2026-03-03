// B-spline evaluation for PPPM charge assignment (f64)
//
// **Physics**: Cardinal B-splines M_p(x) for smooth particle-mesh interpolation
// **Formula**: M_p(x) = 1/(p-1)! * Σ_{k=0}^{p} (-1)^k * C(p,k) * max(0, x-k)^{p-1}
// **Precision**: Full f64 via math_f64.wgsl preamble
// **Use Case**: PPPM charge spreading and force interpolation
//
// Requires: math_f64.wgsl preamble
//
// Bindings:
//   0: positions  [N*3] f64, read     — particle positions (x,y,z)
//   1: coeffs     [N*order*3] f64, write — B-spline coefficients per particle/dim
//   2: derivs     [N*order*3] f64, write — B-spline derivatives per particle/dim
//   3: base_idx   [N*3] i32, write    — base mesh indices per particle
//   4: params     [16] f64, read      — simulation parameters

@group(0) @binding(0) var<storage, read> positions: array<f64>;
@group(0) @binding(1) var<storage, read_write> coeffs: array<f64>;
@group(0) @binding(2) var<storage, read_write> derivs: array<f64>;
@group(0) @binding(3) var<storage, read_write> base_idx: array<i32>;
@group(0) @binding(4) var<storage, read> params: array<f64>;

// params layout:
//   [0] = n_particles
//   [1] = order (B-spline order, typically 4-7)
//   [2] = mesh_kx, [3] = mesh_ky, [4] = mesh_kz
//   [5] = box_x, [6] = box_y, [7] = box_z

// Binomial coefficient C(n, k)
fn binomial(n: u32, k: u32) -> f64 {
    if (k > n) { return f64_const(positions[0], 0.0); }
    if (k == 0u || k == n) { return f64_const(positions[0], 1.0); }
    
    var result = f64_const(positions[0], 1.0);
    var ki = k;
    if (ki > n - ki) { ki = n - ki; }
    
    for (var i = 0u; i < ki; i = i + 1u) {
        result = result * f64(n - i) / f64(i + 1u);
    }
    return result;
}

// Factorial
fn factorial(n: u32) -> f64 {
    var result = f64_const(positions[0], 1.0);
    for (var i = 2u; i <= n; i = i + 1u) {
        result = result * f64(i);
    }
    return result;
}

// Cardinal B-spline M_p(x) using explicit formula
// M_p(x) = 1/(p-1)! * Σ_{k=0}^{p} (-1)^k * C(p,k) * max(0, x-k)^{p-1}
fn bspline_value(order: u32, x: f64) -> f64 {
    let zero = f64_const(x, 0.0);
    let one = f64_const(x, 1.0);
    
    if (order == 0u) { return zero; }
    if (order == 1u) {
        if (x >= zero && x < one) { return one; }
        return zero;
    }
    
    let order_f = f64(order);
    if (x <= zero || x >= order_f) { return zero; }
    
    let pm1 = order - 1u;
    let inv_fact = one / factorial(pm1);
    
    var sum = zero;
    for (var k = 0u; k <= order; k = k + 1u) {
        let xmk = x - f64(k);
        if (xmk > zero) {
            var sign = one;
            if ((k & 1u) == 1u) { sign = -one; }
            let binom = binomial(order, k);
            sum = sum + sign * binom * pow_f64(xmk, f64(pm1));
        }
    }
    
    return sum * inv_fact;
}

// B-spline derivative dM_p(x)/dx = M_{p-1}(x) - M_{p-1}(x-1)
fn bspline_deriv_value(order: u32, x: f64) -> f64 {
    if (order <= 1u) {
        return f64_const(x, 0.0);
    }
    return bspline_value(order - 1u, x) - bspline_value(order - 1u, x - f64_const(x, 1.0));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let n = u32(params[0]);
    if (i >= n) { return; }
    
    let order = u32(params[1]);
    let mesh_kx = u32(params[2]);
    let mesh_ky = u32(params[3]);
    let mesh_kz = u32(params[4]);
    let box_x = params[5];
    let box_y = params[6];
    let box_z = params[7];
    
    // Particle position
    let px = positions[i * 3u];
    let py = positions[i * 3u + 1u];
    let pz = positions[i * 3u + 2u];
    
    // Convert to mesh coordinates
    let ux = px / box_x * f64(mesh_kx);
    let uy = py / box_y * f64(mesh_ky);
    let uz = pz / box_z * f64(mesh_kz);
    
    // Base indices: floor(u) - (order - 1)
    let base_x = i32(floor_f64(ux)) - i32(order) + 1;
    let base_y = i32(floor_f64(uy)) - i32(order) + 1;
    let base_z = i32(floor_f64(uz)) - i32(order) + 1;
    
    // Store base indices
    base_idx[i * 3u] = base_x;
    base_idx[i * 3u + 1u] = base_y;
    base_idx[i * 3u + 2u] = base_z;
    
    // Compute coefficients for each dimension
    // Output layout: coeffs[i * order * 3 + dim * order + k]
    let stride = order * 3u;
    
    for (var k = 0u; k < order; k = k + 1u) {
        // X dimension
        let arg_x = ux - f64(base_x + i32(k));
        coeffs[i * stride + k] = bspline_value(order, arg_x);
        derivs[i * stride + k] = bspline_deriv_value(order, arg_x);
        
        // Y dimension
        let arg_y = uy - f64(base_y + i32(k));
        coeffs[i * stride + order + k] = bspline_value(order, arg_y);
        derivs[i * stride + order + k] = bspline_deriv_value(order, arg_y);
        
        // Z dimension
        let arg_z = uz - f64(base_z + i32(k));
        coeffs[i * stride + 2u * order + k] = bspline_value(order, arg_z);
        derivs[i * stride + 2u * order + k] = bspline_deriv_value(order, arg_z);
    }
}

// SPDX-License-Identifier: AGPL-3.0-or-later
// richards_picard_f64.wgsl — GPU kernels for Richards PDE Picard iteration
//
// Three entry points for the iterative solver pipeline:
//   1. compute_hydraulics: K(h), C(h), θ(h) for each node
//   2. assemble_tridiag: Build tridiagonal system coefficients
//   3. thomas_forward / thomas_backward: Thomas algorithm (sequential per-problem)
//
// Van Genuchten-Mualem soil hydraulic model throughout.
// REQUIRES: SHADER_F64

struct RichardsParams {
    n_nodes:   u32,
    bc_top:    u32,     // 0 = pressure head, 1 = flux
    bc_bottom: u32,     // 0 = pressure head, 1 = flux
    _pad:      u32,
    dz:        f64,
    dt:        f64,
    theta_s:   f64,
    theta_r:   f64,
    alpha:     f64,
    vg_n:      f64,
    k_sat:     f64,
    bc_top_val:    f64,
    bc_bottom_val: f64,
    _padf:     f64,
}

@group(0) @binding(0) var<uniform>             params:    RichardsParams;
@group(0) @binding(1) var<storage, read>       h:         array<f64>;  // current pressure head
@group(0) @binding(2) var<storage, read>       h_old:     array<f64>;  // previous time step
@group(0) @binding(3) var<storage, read_write> k_buf:     array<f64>;  // K(h) per node
@group(0) @binding(4) var<storage, read_write> c_buf:     array<f64>;  // C(h) per node
@group(0) @binding(5) var<storage, read_write> theta_buf: array<f64>;  // θ(h_old) per node
@group(0) @binding(6) var<storage, read_write> k_half:    array<f64>;  // harmonic mean K at interfaces
@group(0) @binding(7) var<storage, read_write> a_tri:     array<f64>;  // sub-diagonal
@group(0) @binding(8) var<storage, read_write> b_tri:     array<f64>;  // main diagonal
@group(0) @binding(9) var<storage, read_write> c_tri:     array<f64>;  // super-diagonal
@group(0) @binding(10) var<storage, read_write> d_vec:    array<f64>;  // RHS vector
@group(0) @binding(11) var<storage, read_write> h_new:    array<f64>;  // output

const HARMONIC_GUARD: f64 = f64(1e-30);
const MIN_CAP: f64 = f64(1e-10);

fn vg_m() -> f64 {
    let one = params.dz - params.dz + f64(1.0);
    return one - one / params.vg_n;
}

fn effective_saturation(pressure: f64) -> f64 {
    let one = params.dz - params.dz + f64(1.0);
    if (pressure >= params.dz - params.dz) { return one; }
    let ah = params.alpha * abs(pressure);
    let ah_n = pow_f64(max(ah, params.dz - params.dz), params.vg_n);
    return pow_f64(one + ah_n, -vg_m());
}

fn vg_theta(pressure: f64) -> f64 {
    return params.theta_r + (params.theta_s - params.theta_r) * effective_saturation(pressure);
}

fn vg_capacity(pressure: f64) -> f64 {
    let one = params.dz - params.dz + f64(1.0);
    let zero = params.dz - params.dz;
    if (pressure >= zero) { return zero; }
    let m = vg_m();
    let ah = params.alpha * abs(pressure);
    let ah_n = pow_f64(ah, params.vg_n);
    let denom = one + ah_n;
    return (params.theta_s - params.theta_r) * params.alpha * params.vg_n * m
           * pow_f64(ah, params.vg_n - one) / pow_f64(denom, m + one);
}

fn vg_conductivity(pressure: f64) -> f64 {
    let one = params.dz - params.dz + f64(1.0);
    let se = effective_saturation(pressure);
    let m = vg_m();
    let inner = one - pow_f64(one - pow_f64(se, one / m), m);
    return params.k_sat * sqrt_f64(se) * inner * inner;
}

// Entry 1: Compute K, C, θ per node (fully parallel)
@compute @workgroup_size(64)
fn compute_hydraulics(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n_nodes) { return; }

    k_buf[i] = vg_conductivity(h[i]);
    c_buf[i] = vg_capacity(h[i]);
    theta_buf[i] = vg_theta(h_old[i]);

    if (i < params.n_nodes - 1u) {
        let two = params.dz - params.dz + f64(2.0);
        k_half[i] = two * k_buf[i] * vg_conductivity(h[i + 1u])
                     / (k_buf[i] + vg_conductivity(h[i + 1u]) + HARMONIC_GUARD);
    }
}

// Entry 2: Assemble tridiagonal system (fully parallel)
@compute @workgroup_size(64)
fn assemble_tridiag(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.n_nodes) { return; }

    let one = params.dz - params.dz + f64(1.0);
    let half = params.dz - params.dz + f64(0.5);
    let zero = params.dz - params.dz;
    let dz2 = params.dz * params.dz;

    if (i == 0u) {
        if (params.bc_top == 0u) {
            b_tri[0u] = one;
            a_tri[0u] = zero;
            c_tri[0u] = zero;
            d_vec[0u] = params.bc_top_val;
        } else {
            let coeff_r = k_half[0u] / dz2;
            let ci = max(c_buf[0u], MIN_CAP);
            b_tri[0u] = ci + half * params.dt * coeff_r;
            a_tri[0u] = zero;
            c_tri[0u] = -half * params.dt * coeff_r;
            d_vec[0u] = ci * h_old[0u] - half * params.dt * coeff_r * h_old[0u]
                       + half * params.dt * coeff_r * h_old[1u]
                       + params.dt * params.bc_top_val / params.dz;
        }
    } else if (i == params.n_nodes - 1u) {
        if (params.bc_bottom == 0u) {
            b_tri[i] = one;
            a_tri[i] = zero;
            c_tri[i] = zero;
            d_vec[i] = params.bc_bottom_val;
        } else {
            let coeff_l = k_half[i - 1u] / dz2;
            let ci = max(c_buf[i], MIN_CAP);
            a_tri[i] = -half * params.dt * coeff_l;
            b_tri[i] = ci + half * params.dt * coeff_l;
            c_tri[i] = zero;
            d_vec[i] = ci * h_old[i] + half * params.dt * coeff_l * h_old[i - 1u]
                       - half * params.dt * coeff_l * h_old[i]
                       + params.dt * params.bc_bottom_val / params.dz;
        }
    } else {
        let ci = max(c_buf[i], MIN_CAP);
        let coeff_l = k_half[i - 1u] / dz2;
        let coeff_r = k_half[i] / dz2;

        a_tri[i] = -half * params.dt * coeff_l;
        c_tri[i] = -half * params.dt * coeff_r;
        b_tri[i] = ci + half * params.dt * (coeff_l + coeff_r);

        d_vec[i] = ci * h_old[i]
                 + half * params.dt * coeff_l * h_old[i - 1u]
                 - half * params.dt * (coeff_l + coeff_r) * h_old[i]
                 + half * params.dt * coeff_r * h_old[i + 1u]
                 + params.dt * (k_half[i] - k_half[i - 1u]) / params.dz
                 + theta_buf[i] - vg_theta(h[i]);
    }
}

// Entry 3: Thomas algorithm (sequential — single workgroup)
@compute @workgroup_size(1)
fn thomas_solve(@builtin(workgroup_id) wg: vec3<u32>) {
    let n = params.n_nodes;

    // Forward sweep (in-place on c_tri, d_vec)
    // c_tri[0] = c_tri[0] / b_tri[0], d_vec[0] = d_vec[0] / b_tri[0]
    c_tri[0u] = c_tri[0u] / b_tri[0u];
    d_vec[0u] = d_vec[0u] / b_tri[0u];

    for (var i = 1u; i < n; i = i + 1u) {
        let m_val = b_tri[i] - a_tri[i] * c_tri[i - 1u];
        c_tri[i] = c_tri[i] / m_val;
        d_vec[i] = (d_vec[i] - a_tri[i] * d_vec[i - 1u]) / m_val;
    }

    // Back substitution
    h_new[n - 1u] = d_vec[n - 1u];
    for (var i = i32(n) - 2; i >= 0; i = i - 1) {
        h_new[u32(i)] = d_vec[u32(i)] - c_tri[u32(i)] * h_new[u32(i) + 1u];
    }
}

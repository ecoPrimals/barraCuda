// SPDX-License-Identifier: AGPL-3.0-only
//
// rk45_adaptive_f64.wgsl — Adaptive Dormand-Prince RK45 (f64)
//
// Single adaptive step of the DP 5(4) embedded pair.
// Each thread handles one independent ODE system with Hill kinetics.
//
// 5th order solution + 4th order error estimate for step-size control:
//   h_new = h * min(5, max(0.2, 0.9*(tol/err)^0.2))
//
// Evolved from f32 → f64 for universal math library portability.

struct Params {
    n_systems: u32,
    dim: u32,
    n_coeffs: u32,
    _pad: u32,
    dt: f64,
    _pad2: f64,
}

@group(0) @binding(0) var<storage, read> state: array<f64>;
@group(0) @binding(1) var<storage, read> coeffs: array<f64>;
@group(0) @binding(2) var<storage, read_write> new_state: array<f64>;
@group(0) @binding(3) var<storage, read_write> error: array<f64>;
@group(0) @binding(4) var<uniform> params: Params;
@group(0) @binding(5) var<storage, read_write> scratch: array<f64>;

fn hill(x: f64, k: f64, n: f64) -> f64 {
    let xn = pow(x, n);
    return xn / (pow(k, n) + xn);
}

fn rhs(sys: u32, d: u32, y_base: u32) -> f64 {
    let c_base = sys * params.n_coeffs + d * 3u;
    let prod = coeffs[c_base];
    let deg = coeffs[c_base + 1u];
    let act_idx = u32(coeffs[c_base + 2u]);
    let activator = scratch[y_base + act_idx];
    return prod * hill(activator, f64(0.5), f64(2.0)) - deg * scratch[y_base + d];
}

fn write_k(sys: u32, stage: u32, d: u32, val: f64) {
    scratch[sys * params.dim * 8u + stage * params.dim + d] = val;
}

fn read_k(sys: u32, stage: u32, d: u32) -> f64 {
    return scratch[sys * params.dim * 8u + stage * params.dim + d];
}

fn write_tmp(sys: u32, d: u32, val: f64) {
    scratch[sys * params.dim * 8u + 7u * params.dim + d] = val;
}

@compute @workgroup_size(64)
fn rk45_step(@builtin(global_invocation_id) gid: vec3<u32>) {
    let sys = gid.x;
    if sys >= params.n_systems { return; }

    let dim = params.dim;
    let h = params.dt;
    let y_base = sys * dim * 8u + 7u * dim;

    for (var d: u32 = 0u; d < dim; d = d + 1u) {
        write_tmp(sys, d, state[sys * dim + d]);
    }

    // Stage 1
    for (var d: u32 = 0u; d < dim; d = d + 1u) {
        write_k(sys, 0u, d, rhs(sys, d, y_base));
    }

    // Stage 2
    for (var d: u32 = 0u; d < dim; d = d + 1u) {
        let y0 = state[sys * dim + d];
        write_tmp(sys, d, y0 + h * (f64(1.0) / f64(5.0)) * read_k(sys, 0u, d));
    }
    for (var d: u32 = 0u; d < dim; d = d + 1u) {
        write_k(sys, 1u, d, rhs(sys, d, y_base));
    }

    // Stage 3
    for (var d: u32 = 0u; d < dim; d = d + 1u) {
        let y0 = state[sys * dim + d];
        let inc = (f64(3.0) / f64(40.0)) * read_k(sys, 0u, d)
                + (f64(9.0) / f64(40.0)) * read_k(sys, 1u, d);
        write_tmp(sys, d, y0 + h * inc);
    }
    for (var d: u32 = 0u; d < dim; d = d + 1u) {
        write_k(sys, 2u, d, rhs(sys, d, y_base));
    }

    // Stage 4
    for (var d: u32 = 0u; d < dim; d = d + 1u) {
        let y0 = state[sys * dim + d];
        let inc = (f64(44.0) / f64(45.0)) * read_k(sys, 0u, d)
                - (f64(56.0) / f64(15.0)) * read_k(sys, 1u, d)
                + (f64(32.0) / f64(9.0))  * read_k(sys, 2u, d);
        write_tmp(sys, d, y0 + h * inc);
    }
    for (var d: u32 = 0u; d < dim; d = d + 1u) {
        write_k(sys, 3u, d, rhs(sys, d, y_base));
    }

    // Stage 5
    for (var d: u32 = 0u; d < dim; d = d + 1u) {
        let y0 = state[sys * dim + d];
        let inc = (f64(19372.0) / f64(6561.0))  * read_k(sys, 0u, d)
                - (f64(25360.0) / f64(2187.0))  * read_k(sys, 1u, d)
                + (f64(64448.0) / f64(6561.0))  * read_k(sys, 2u, d)
                - (f64(212.0) / f64(729.0))     * read_k(sys, 3u, d);
        write_tmp(sys, d, y0 + h * inc);
    }
    for (var d: u32 = 0u; d < dim; d = d + 1u) {
        write_k(sys, 4u, d, rhs(sys, d, y_base));
    }

    // Stage 6
    for (var d: u32 = 0u; d < dim; d = d + 1u) {
        let y0 = state[sys * dim + d];
        let inc = (f64(9017.0) / f64(3168.0))   * read_k(sys, 0u, d)
                - (f64(355.0) / f64(33.0))      * read_k(sys, 1u, d)
                + (f64(46732.0) / f64(5247.0))  * read_k(sys, 2u, d)
                + (f64(49.0) / f64(176.0))      * read_k(sys, 3u, d)
                - (f64(5103.0) / f64(18656.0))  * read_k(sys, 4u, d);
        write_tmp(sys, d, y0 + h * inc);
    }
    for (var d: u32 = 0u; d < dim; d = d + 1u) {
        write_k(sys, 5u, d, rhs(sys, d, y_base));
    }

    // 5th order solution + error estimate
    for (var d: u32 = 0u; d < dim; d = d + 1u) {
        let y0 = state[sys * dim + d];
        let k1 = read_k(sys, 0u, d);
        let k3 = read_k(sys, 2u, d);
        let k4 = read_k(sys, 3u, d);
        let k5 = read_k(sys, 4u, d);
        let k6 = read_k(sys, 5u, d);

        let y5 = y0 + h * (
            (f64(35.0) / f64(384.0))    * k1
          + (f64(500.0) / f64(1113.0))  * k3
          + (f64(125.0) / f64(192.0))   * k4
          - (f64(2187.0) / f64(6784.0)) * k5
          + (f64(11.0) / f64(84.0))     * k6
        );

        let e = h * (
            (f64(71.0) / f64(57600.0))     * k1
          - (f64(71.0) / f64(16695.0))     * k3
          + (f64(71.0) / f64(1920.0))      * k4
          - (f64(17253.0) / f64(339200.0)) * k5
          + (f64(22.0) / f64(525.0))       * k6
        );

        new_state[sys * dim + d] = y5;
        error[sys * dim + d] = abs(e);
    }
}

// SPDX-License-Identifier: AGPL-3.0-or-later
// L-BFGS Two-Loop Recursion — f64 canonical
//
// Implements the Nocedal (1980) two-loop recursion on GPU:
//   d = -H_k · g_k
//
// where H_k is the limited-memory BFGS inverse Hessian approximation
// stored implicitly via m correction pairs (s_i, y_i).
//
// Operates on batched problems: `batch_size` independent L-BFGS instances
// each with `n` dimensions and `m_used` stored pairs.

struct Params {
    n: u32,
    m_capacity: u32,
    m_used: u32,
    batch_size: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
// gradient: [batch_size * n]
@group(0) @binding(1) var<storage, read> gradient: array<f64>;
// s_history: [m_capacity * batch_size * n] — position differences
@group(0) @binding(2) var<storage, read> s_history: array<f64>;
// y_history: [m_capacity * batch_size * n] — gradient differences
@group(0) @binding(3) var<storage, read> y_history: array<f64>;
// rho: [m_capacity * batch_size]
@group(0) @binding(4) var<storage, read> rho: array<f64>;
// alpha_buf: [m_capacity * batch_size] — scratch for backward loop
@group(0) @binding(5) var<storage, read_write> alpha_buf: array<f64>;
// direction: [batch_size * n] — output descent direction
@group(0) @binding(6) var<storage, read_write> direction: array<f64>;

@compute @workgroup_size(64, 1, 1)
fn two_loop_recursion(@builtin(global_invocation_id) gid: vec3<u32>) {
    let b = gid.x;
    if (b >= params.batch_size) {
        return;
    }
    let n = params.n;
    let m = params.m_used;
    let mc = params.m_capacity;
    let b_off = b * n;

    // q = gradient[b]
    for (var j = 0u; j < n; j = j + 1u) {
        direction[b_off + j] = gradient[b_off + j];
    }

    // Backward loop: i = m-1 .. 0
    for (var ii = 0u; ii < m; ii = ii + 1u) {
        let i = m - 1u - ii;
        let s_off = (i * params.batch_size + b) * n;
        let rho_idx = i * params.batch_size + b;
        var dot: f64 = 0.0;
        for (var j = 0u; j < n; j = j + 1u) {
            dot = dot + s_history[s_off + j] * direction[b_off + j];
        }
        let alpha_i = rho[rho_idx] * dot;
        alpha_buf[i * params.batch_size + b] = alpha_i;
        let y_off = (i * params.batch_size + b) * n;
        for (var j = 0u; j < n; j = j + 1u) {
            direction[b_off + j] = direction[b_off + j] - alpha_i * y_history[y_off + j];
        }
    }

    // Scale by initial Hessian: H_0 = gamma * I
    if (m > 0u) {
        let last = m - 1u;
        let s_off = (last * params.batch_size + b) * n;
        let y_off = (last * params.batch_size + b) * n;
        var sy: f64 = 0.0;
        var yy: f64 = 0.0;
        for (var j = 0u; j < n; j = j + 1u) {
            sy = sy + s_history[s_off + j] * y_history[y_off + j];
            yy = yy + y_history[y_off + j] * y_history[y_off + j];
        }
        var gamma: f64 = 1.0;
        if (yy > 1e-30) {
            gamma = sy / yy;
        }
        for (var j = 0u; j < n; j = j + 1u) {
            direction[b_off + j] = direction[b_off + j] * gamma;
        }
    }

    // Forward loop: i = 0 .. m-1
    for (var i = 0u; i < m; i = i + 1u) {
        let y_off = (i * params.batch_size + b) * n;
        let rho_idx = i * params.batch_size + b;
        var dot: f64 = 0.0;
        for (var j = 0u; j < n; j = j + 1u) {
            dot = dot + y_history[y_off + j] * direction[b_off + j];
        }
        let beta = rho[rho_idx] * dot;
        let alpha_i = alpha_buf[i * params.batch_size + b];
        let s_off = (i * params.batch_size + b) * n;
        for (var j = 0u; j < n; j = j + 1u) {
            direction[b_off + j] = direction[b_off + j] + (alpha_i - beta) * s_history[s_off + j];
        }
    }

    // Negate for descent: d = -H * g
    for (var j = 0u; j < n; j = j + 1u) {
        direction[b_off + j] = -direction[b_off + j];
    }
}

// Line search step update: x_new = x + alpha * d
struct StepParams {
    n: u32,
    batch_size: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> step_params: StepParams;
@group(0) @binding(1) var<storage, read> x_current: array<f64>;
@group(0) @binding(2) var<storage, read> dir: array<f64>;
@group(0) @binding(3) var<storage, read> alphas: array<f64>;
@group(0) @binding(4) var<storage, read_write> x_next: array<f64>;

@compute @workgroup_size(256, 1, 1)
fn apply_step(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let n = step_params.n;
    let total = n * step_params.batch_size;
    if (idx >= total) {
        return;
    }
    let b = idx / n;
    x_next[idx] = x_current[idx] + alphas[b] * dir[idx];
}

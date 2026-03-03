// rk45_f64.wgsl — Adaptive Runge-Kutta-Fehlberg (RK45) single step, Dormand-Prince
//
// **Math**: Dormand-Prince 5(4) embedded pair. Seven stages, FSAL property.
//   y_new = y + h * Σ b_i * k_i     (5th order)
//   y*    = y + h * Σ b*_i * k_i    (4th order, for error estimate)
//   err   = y_new - y* = h * Σ (b_i - b*_i) * k_i
//
// **State layout**: [y0, y1, ..., y_{n_vars-1}, t, h] — (n_vars + 2) elements per system
// **RHS placeholder**: fn rhs(t: f64, y: f64) -> f64 { return -y; }
//   → Gets replaced by concatenation at compile time (math_f64 preamble pattern).
//   Host injects the actual ODE RHS before compilation.
//
// **Precision**: f64 via bitcast<f64>(vec2<u32>)
// **Workgroup**: @compute @workgroup_size(256)
//
// Bindings:
//   0: state   array<vec2<u32>>  read  — [y0, y1, ..., t, h] per system
//   1: new_state array<vec2<u32>> read_write — updated state
//   2: error   array<vec2<u32>>  read_write — error estimate per element
//
// Params: { n_systems: u32, n_vars: u32 }
//
// Reference: Dormand & Prince (1980); MATLAB ode45

// PLACEHOLDER RHS — replace via ShaderTemplate concatenation at compile time
// fn rhs(t: f64, y: f64) -> f64 { return -y; }
// For multi-variable systems, RHS is typically vector-valued; this placeholder
// represents one component. The injected RHS should match (t, y_vec) -> dy_vec.

fn rhs(t: f64, y: f64) -> f64 {
    return -y;
}

// Dormand-Prince c_i (time nodes)
fn get_c(stage: u32) -> f64 {
    let z = 0.0;
    if (stage == 1u) { return z + 1.0/5.0; }
    if (stage == 2u) { return z + 3.0/10.0; }
    if (stage == 3u) { return z + 4.0/5.0; }
    if (stage == 4u) { return z + 8.0/9.0; }
    if (stage == 5u) { return z + 1.0; }
    if (stage == 6u) { return z + 1.0; }
    return z;
}

// a_ij — lower triangular
fn get_a(stage: u32, j: u32) -> f64 {
    let z = 0.0;
    if (stage == 1u && j == 0u) { return z + 1.0/5.0; }
    if (stage == 2u) {
        if (j == 0u) { return z + 3.0/40.0; }
        if (j == 1u) { return z + 9.0/40.0; }
    }
    if (stage == 3u) {
        if (j == 0u) { return z + 44.0/45.0; }
        if (j == 1u) { return z - 56.0/15.0; }
        if (j == 2u) { return z + 32.0/9.0; }
    }
    if (stage == 4u) {
        if (j == 0u) { return z + 19372.0/6561.0; }
        if (j == 1u) { return z - 25360.0/2187.0; }
        if (j == 2u) { return z + 64448.0/6561.0; }
        if (j == 3u) { return z - 212.0/729.0; }
    }
    if (stage == 5u) {
        if (j == 0u) { return z + 9017.0/3168.0; }
        if (j == 1u) { return z - 355.0/33.0; }
        if (j == 2u) { return z + 46732.0/5247.0; }
        if (j == 3u) { return z + 49.0/176.0; }
        if (j == 4u) { return z - 5103.0/18656.0; }
    }
    if (stage == 6u) {
        if (j == 0u) { return z + 35.0/384.0; }
        if (j == 1u) { return z; }
        if (j == 2u) { return z + 500.0/1113.0; }
        if (j == 3u) { return z + 125.0/192.0; }
        if (j == 4u) { return z - 2187.0/6784.0; }
        if (j == 5u) { return z + 11.0/84.0; }
    }
    return z;
}

// 5th order weights b_i
fn get_b5(j: u32) -> f64 {
    let z = 0.0;
    if (j == 0u) { return z + 35.0/384.0; }
    if (j == 1u) { return z; }
    if (j == 2u) { return z + 500.0/1113.0; }
    if (j == 3u) { return z + 125.0/192.0; }
    if (j == 4u) { return z - 2187.0/6784.0; }
    if (j == 5u) { return z + 11.0/84.0; }
    if (j == 6u) { return z; }
    return z;
}

// 4th order weights b*_i (for error)
fn get_b4(j: u32) -> f64 {
    let z = 0.0;
    if (j == 0u) { return z + 5179.0/57600.0; }
    if (j == 1u) { return z; }
    if (j == 2u) { return z + 7571.0/16695.0; }
    if (j == 3u) { return z + 393.0/640.0; }
    if (j == 4u) { return z - 92097.0/339200.0; }
    if (j == 5u) { return z + 187.0/2100.0; }
    if (j == 6u) { return z + 1.0/40.0; }
    return z;
}

@group(0) @binding(0) var<storage, read> state: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read_write> new_state: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read_write> error: array<vec2<u32>>;
@group(0) @binding(3) var<uniform> params: Params;

struct Params {
    n_systems: u32,
    n_vars: u32,
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let sys = global_id.x;
    let n_systems = params.n_systems;
    let n_vars = params.n_vars;
    if (sys >= n_systems) {
        return;
    }

    let stride = n_vars + 2u;  // +t, +h
    let base = sys * stride;

    // Load t and h
    let t = bitcast<f64>(state[base + n_vars]);
    let h = bitcast<f64>(state[base + n_vars + 1u]);

    // For each variable, compute RK45 step
    for (var v = 0u; v < n_vars; v = v + 1u) {
        let idx = base + v;

        // Load current y
        var y_stage = bitcast<f64>(state[idx]);

        // k1..k6 (simplified: scalar RHS placeholder; vector RHS injected at compile time)
        // Stage 1: k1 = f(t, y)
        var k1 = rhs(t, y_stage);

        // Stage 2: k2 = f(t + c2*h, y + h*a21*k1)
        var y2 = y_stage + h * get_a(1u, 0u) * k1;
        var k2 = rhs(t + get_c(1u) * h, y2);

        // Stage 3
        var y3 = y_stage + h * (get_a(2u, 0u) * k1 + get_a(2u, 1u) * k2);
        var k3 = rhs(t + get_c(2u) * h, y3);

        // Stage 4
        var y4 = y_stage + h * (get_a(3u, 0u) * k1 + get_a(3u, 1u) * k2 + get_a(3u, 2u) * k3);
        var k4 = rhs(t + get_c(3u) * h, y4);

        // Stage 5
        var y5 = y_stage + h * (get_a(4u, 0u) * k1 + get_a(4u, 1u) * k2 + get_a(4u, 2u) * k3 + get_a(4u, 3u) * k4);
        var k5 = rhs(t + get_c(4u) * h, y5);

        // Stage 6
        var y6 = y_stage + h * (get_a(5u, 0u) * k1 + get_a(5u, 1u) * k2 + get_a(5u, 2u) * k3 + get_a(5u, 3u) * k4 + get_a(5u, 4u) * k5);
        var k6 = rhs(t + h, y6);

        // 5th order: y_new = y + h * Σ b5_i * k_i (b5_6 = 0, so no k7 here)
        var y_new = y_stage + h * (get_b5(0u) * k1 + get_b5(1u) * k2 + get_b5(2u) * k3 + get_b5(3u) * k4 + get_b5(4u) * k5 + get_b5(5u) * k6);

        // k7 = f(t+h, y_new) — FSAL (First Same As Last)
        var k7 = rhs(t + h, y_new);

        // 4th order: y4 = y + h * Σ b4_i * k_i (includes k7 with b4_6 = 1/40)
        var y4_order = y_stage + h * (get_b4(0u) * k1 + get_b4(1u) * k2 + get_b4(2u) * k3 + get_b4(3u) * k4 + get_b4(4u) * k5 + get_b4(5u) * k6 + get_b4(6u) * k7);

        // Error estimate: err = y_new - y4
        var err_est = y_new - y4_order;

        new_state[idx] = bitcast<vec2<u32>>(y_new);
        error[idx] = bitcast<vec2<u32>>(err_est);
    }

    // Copy t and h to new_state
    new_state[base + n_vars] = state[base + n_vars];
    new_state[base + n_vars + 1u] = state[base + n_vars + 1u];

    // Error for t,h slots — zero
    let zero = bitcast<vec2<u32>>(0.0);
    error[base + n_vars] = zero;
    error[base + n_vars + 1u] = zero;
}

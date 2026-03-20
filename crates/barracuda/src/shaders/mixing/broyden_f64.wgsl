// SPDX-License-Identifier: AGPL-3.0-or-later
// Broyden Density Mixing for Self-Consistent Field Solvers (f64)
//
// Generic vector mixing operations for SCF convergence:
//   1. Linear mixing: x_new = (1-α)·x_old + α·x_computed
//   2. Broyden mixing: x_new = x + α·F - Σ_m γ_m·(Δx_m + α·ΔF_m)
//
// Applications: DFT, HFB nuclear structure, Poisson-Boltzmann,
//   coupled-cluster, nonlinear equation solvers, fixed-point iterations
// Validated by: hotSpring nuclear EOS study (169/169 acceptance checks)
//
// Deep Debt: pure WGSL, f64, self-contained, physics-agnostic

// ═══════════════════════════════════════════════════════════════════
// Kernel 1: Linear mixing
//
// x_new[i] = (1 - α)·x_old[i] + α·x_computed[i]
//
// Simple damped iteration used for warmup or when Broyden diverges.
// Dispatch: (ceil(vec_dim / 256), 1, 1)
// ═══════════════════════════════════════════════════════════════════

struct LinearParams {
    vec_dim: u32,       // Vector dimension
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    alpha: f64,         // Mixing parameter (typically 0.3-0.7)
    clamp_min: f64,     // Optional clamp minimum (set to -1e308 to disable)
    clamp_max: f64,     // Optional clamp maximum (set to 1e308 to disable)
}

@group(0) @binding(0) var<uniform> linear_params: LinearParams;
@group(0) @binding(1) var<storage, read> old_vec: array<f64>;      // x_old
@group(0) @binding(2) var<storage, read> computed_vec: array<f64>; // x_computed
@group(0) @binding(3) var<storage, read_write> output_vec: array<f64>;

@compute @workgroup_size(256)
fn mix_linear(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= linear_params.vec_dim) { return; }

    let alpha = linear_params.alpha;
    let mixed = (f64(1.0) - alpha) * old_vec[idx] + alpha * computed_vec[idx];
    
    // Optional clamping (e.g., for non-negative densities)
    output_vec[idx] = clamp(mixed, linear_params.clamp_min, linear_params.clamp_max);
}

// ═══════════════════════════════════════════════════════════════════
// Kernel 2: Broyden vector update (Modified Broyden II)
//
// Computes: x_new[i] = x[i] + α·F[i] - Σ_m γ_m·(Δx_m[i] + α·ΔF_m[i])
//
// Where:
//   x     = current input vector
//   F     = residual: F(x) = x_out - x
//   γ_m   = Broyden coefficients (computed on CPU from history)
//   Δx_m  = difference in x between iteration m and m-1
//   ΔF_m  = difference in F between iteration m and m-1
//
// The Broyden history management and γ computation (small linear algebra)
// are handled by the CPU. This kernel applies the O(n_grid) vector update.
//
// Dispatch: (ceil(vec_dim / 256), 1, 1)
// ═══════════════════════════════════════════════════════════════════

struct BroydenParams {
    vec_dim: u32,       // Vector dimension (e.g., 2*n_grid for proton+neutron)
    n_history: u32,     // Number of Broyden history vectors to apply
    _pad0: u32,
    _pad1: u32,
    alpha_mix: f64,     // Mixing parameter α (typically 0.4)
    clamp_min: f64,     // Optional clamp minimum
    clamp_max: f64,     // Optional clamp maximum
}

@group(1) @binding(0) var<uniform> broy_params: BroydenParams;
@group(1) @binding(1) var<storage, read> input_vec: array<f64>;     // Current x
@group(1) @binding(2) var<storage, read> residual: array<f64>;      // F(x) = x_out - x
@group(1) @binding(3) var<storage, read> gammas: array<f64>;        // [n_history] γ coefficients
@group(1) @binding(4) var<storage, read> dx_history: array<f64>;    // [n_history × vec_dim] Δx_m
@group(1) @binding(5) var<storage, read> df_history: array<f64>;    // [n_history × vec_dim] ΔF_m
@group(1) @binding(6) var<storage, read_write> mixed_vec: array<f64>;

@compute @workgroup_size(256)
fn broyden_update(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= broy_params.vec_dim) { return; }

    // Start with x + α·F
    var result = input_vec[idx] + broy_params.alpha_mix * residual[idx];

    // Subtract Σ_m γ_m·(Δx_m + α·ΔF_m)
    for (var m = 0u; m < broy_params.n_history; m++) {
        let gamma_m = gammas[m];
        let dx_m = dx_history[m * broy_params.vec_dim + idx];
        let df_m = df_history[m * broy_params.vec_dim + idx];
        result -= gamma_m * (dx_m + broy_params.alpha_mix * df_m);
    }

    // Optional clamping
    mixed_vec[idx] = clamp(result, broy_params.clamp_min, broy_params.clamp_max);
}

// ═══════════════════════════════════════════════════════════════════
// Kernel 3: Compute residual  F(x) = x_out - x_in
//
// Simple vector subtraction, provided for convenience.
// Dispatch: (ceil(vec_dim / 256), 1, 1)
// ═══════════════════════════════════════════════════════════════════

@group(2) @binding(0) var<uniform> res_dim: u32;
@group(2) @binding(1) var<storage, read> x_in: array<f64>;
@group(2) @binding(2) var<storage, read> x_out: array<f64>;
@group(2) @binding(3) var<storage, read_write> residual_out: array<f64>;

@compute @workgroup_size(256)
fn compute_residual(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= res_dim) { return; }
    residual_out[idx] = x_out[idx] - x_in[idx];
}

// ═══════════════════════════════════════════════════════════════════
// Kernel 4: Vector difference for history  Δv = v_new - v_old
//
// Dispatch: (ceil(vec_dim / 256), 1, 1)
// ═══════════════════════════════════════════════════════════════════

struct DiffParams {
    vec_dim: u32,
    history_idx: u32,   // Which history slot to write to
    _pad0: u32,
    _pad1: u32,
}

@group(3) @binding(0) var<uniform> diff_params: DiffParams;
@group(3) @binding(1) var<storage, read> v_new: array<f64>;
@group(3) @binding(2) var<storage, read> v_old: array<f64>;
@group(3) @binding(3) var<storage, read_write> history: array<f64>;  // [n_history × vec_dim]

@compute @workgroup_size(256)
fn compute_diff_to_history(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= diff_params.vec_dim) { return; }
    let hist_offset = diff_params.history_idx * diff_params.vec_dim;
    history[hist_offset + idx] = v_new[idx] - v_old[idx];
}

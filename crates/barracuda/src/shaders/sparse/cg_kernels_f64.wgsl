// GPU-Resident Conjugate Gradient kernels - f64 Precision
// These kernels keep scalar values on GPU to eliminate per-iteration CPU sync.
// All bindings use read_write for consistency within this module.

struct CGParams {
    n: u32,
}

// ============================================================================
// CG update step 1: x = x + alpha * p, r = r - alpha * Ap
// ============================================================================
@group(0) @binding(0) var<storage, read_write> cg_x: array<f64>;
@group(0) @binding(1) var<storage, read_write> cg_r: array<f64>;
@group(0) @binding(2) var<storage, read_write> cg_p: array<f64>;
@group(0) @binding(3) var<storage, read_write> cg_Ap: array<f64>;
@group(0) @binding(4) var<storage, read_write> cg_alpha: array<f64>;
@group(0) @binding(5) var<uniform> cg_params1: CGParams;

@compute @workgroup_size(256)
fn cg_update_xr(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= cg_params1.n) {
        return;
    }

    let alpha = cg_alpha[0];
    cg_x[idx] = cg_x[idx] + alpha * cg_p[idx];
    cg_r[idx] = cg_r[idx] - alpha * cg_Ap[idx];
}

// ============================================================================
// CG update step 2: p = r + beta * p
// ============================================================================
@group(0) @binding(0) var<storage, read_write> cg_r2: array<f64>;
@group(0) @binding(1) var<storage, read_write> cg_p2: array<f64>;
@group(0) @binding(2) var<storage, read_write> cg_beta: array<f64>;
@group(0) @binding(3) var<uniform> cg_params2: CGParams;

@compute @workgroup_size(256)
fn cg_update_p(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= cg_params2.n) {
        return;
    }

    let beta = cg_beta[0];
    cg_p2[idx] = cg_r2[idx] + beta * cg_p2[idx];
}

// ============================================================================
// Compute alpha = rz / pAp from two scalar buffers
// ============================================================================
@group(0) @binding(0) var<storage, read_write> rz_in: array<f64>;
@group(0) @binding(1) var<storage, read_write> pap_in: array<f64>;
@group(0) @binding(2) var<storage, read_write> alpha_out: array<f64>;

@compute @workgroup_size(1)
fn compute_alpha(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let rz = rz_in[0];
    let pap = pap_in[0];

    if (abs(pap) > f64(1e-30)) {
        alpha_out[0] = rz / pap;
    } else {
        alpha_out[0] = f64(0.0);
    }
}

// ============================================================================
// Compute beta = rz_new / rz_old, also copies rz_new to rz_old
// ============================================================================
@group(0) @binding(0) var<storage, read_write> rz_new_in: array<f64>;
@group(0) @binding(1) var<storage, read_write> rz_old_inout: array<f64>;
@group(0) @binding(2) var<storage, read_write> beta_out: array<f64>;

@compute @workgroup_size(1)
fn compute_beta(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let rz_new = rz_new_in[0];
    let rz_old = rz_old_inout[0];

    if (abs(rz_old) > f64(1e-30)) {
        beta_out[0] = rz_new / rz_old;
    } else {
        beta_out[0] = f64(0.0);
    }

    // Update rz_old for next iteration
    rz_old_inout[0] = rz_new;
}

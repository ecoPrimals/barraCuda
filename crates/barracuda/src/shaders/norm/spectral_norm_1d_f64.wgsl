// SpectralNorm1D - Spectral normalization for 1D convolutions (f64 canonical)
//
// Normalizes weight matrix W by its largest singular value σ₁:
//   W_normalized = W / σ₁
//
// Uses power iteration to estimate σ₁:
//   1. v ← W^T u / ‖W^T u‖
//   2. u ← W v / ‖W v‖
//   3. σ₁ = u^T W v
//
// Used for stabilizing GAN training, Lipschitz-constrained networks,
// audio generation, and robust normalization.
//
// Host dispatches: power_iter_v → power_iter_u → (repeat N times) → compute_sigma → normalize_weights

struct Params {
    rows: u32,        // out_channels
    cols: u32,        // in_channels * kernel_size
    n_power_iter: u32,
    _padding: u32,
}

@group(0) @binding(0) var<storage, read> weights: array<f64>;      // [rows, cols]
@group(0) @binding(1) var<storage, read_write> u: array<f64>;     // [rows]
@group(0) @binding(2) var<storage, read_write> v: array<f64>;     // [cols]
@group(0) @binding(3) var<storage, read_write> output: array<f64>; // [rows, cols] or [1] for sigma
@group(0) @binding(4) var<uniform> params: Params;

// Power iteration step 1: v_new = W^T @ u, then normalize v
@compute @workgroup_size(256)
fn power_iter_v(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let c = global_id.x;

    if (c >= params.cols) {
        return;
    }

    // Compute (W^T @ u)[c] = Σ_r W[r,c] * u[r]
    var sum: f64 = 0.0;
    for (var r: u32 = 0u; r < params.rows; r = r + 1u) {
        sum = sum + weights[r * params.cols + c] * u[r];
    }

    v[c] = sum;
}

// Normalize v in-place: v ← v / ‖v‖
// Single-thread reduction (called with 1 workgroup of 1 thread)
@compute @workgroup_size(1)
fn normalize_v() {
    var norm_sq: f64 = 0.0;
    for (var c: u32 = 0u; c < params.cols; c = c + 1u) {
        norm_sq = norm_sq + v[c] * v[c];
    }
    let norm = sqrt_f64(max(norm_sq, 1e-12));
    for (var c: u32 = 0u; c < params.cols; c = c + 1u) {
        v[c] = v[c] / norm;
    }
}

// Power iteration step 2: u_new = W @ v, then normalize u
@compute @workgroup_size(256)
fn power_iter_u(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let r = global_id.x;

    if (r >= params.rows) {
        return;
    }

    // Compute (W @ v)[r] = Σ_c W[r,c] * v[c]
    var sum: f64 = 0.0;
    for (var c: u32 = 0u; c < params.cols; c = c + 1u) {
        sum = sum + weights[r * params.cols + c] * v[c];
    }

    u[r] = sum;
}

// Normalize u in-place: u ← u / ‖u‖
@compute @workgroup_size(1)
fn normalize_u() {
    var norm_sq: f64 = 0.0;
    for (var r: u32 = 0u; r < params.rows; r = r + 1u) {
        norm_sq = norm_sq + u[r] * u[r];
    }
    let norm = sqrt_f64(max(norm_sq, 1e-12));
    for (var r: u32 = 0u; r < params.rows; r = r + 1u) {
        u[r] = u[r] / norm;
    }
}

// Compute σ₁ = u^T W v and store in output[0]
// Single-thread reduction
@compute @workgroup_size(1)
fn compute_sigma() {
    var sigma: f64 = 0.0;
    for (var r: u32 = 0u; r < params.rows; r = r + 1u) {
        var wv_r: f64 = 0.0;
        for (var c: u32 = 0u; c < params.cols; c = c + 1u) {
            wv_r = wv_r + weights[r * params.cols + c] * v[c];
        }
        sigma = sigma + u[r] * wv_r;
    }
    // Store sigma in output[0] (first element acts as sigma storage)
    output[0] = max(sigma, 1e-12); // Clamp to avoid division by zero
}

// Normalize weights: output[idx] = weights[idx] / σ₁
// σ₁ is read from output[0] (computed by compute_sigma)
@compute @workgroup_size(256)
fn normalize_weights(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total = params.rows * params.cols;

    if (idx >= total) {
        return;
    }

    let sigma = output[0];
    output[idx] = weights[idx] / sigma;
}

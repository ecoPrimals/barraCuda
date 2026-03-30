// SPDX-License-Identifier: AGPL-3.0-or-later
//
// lanczos_iteration_f64.wgsl — One Lanczos iteration (tridiagonalization step)
//
// Computes steps 2–5 of the Lanczos algorithm for symmetric eigenvalue problems:
//   2. α_k = v_k^T * w
//   3. w = w - α_k * v_k - β_{k-1} * v_{k-1}
//   4. β_k = ||w||
//   5. v_{k+1} = w / β_k
//
// Step 1 (w = A * v_k) is done separately by the caller.
//
// Uses a single workgroup (256 threads); each thread strides through the vector
// for arbitrary n. Parallel reductions for dot product and L2 norm.
//
// Provenance: ToadStool spectral eigensolver

struct LanczosParams {
    n: u32,
    _pad: u32,
    beta_prev: f64,
}

@group(0) @binding(0) var<storage, read> w: array<f64>;           // A*v_k result
@group(0) @binding(1) var<storage, read> v_k: array<f64>;         // current Lanczos vector
@group(0) @binding(2) var<storage, read> v_prev: array<f64>;      // previous Lanczos vector
@group(0) @binding(3) var<storage, read_write> v_next: array<f64>; // output: next vector
@group(0) @binding(4) var<storage, read_write> tridiag: array<f64>; // [alpha_k, beta_k]
@group(0) @binding(5) var<uniform> params: LanczosParams;

var<workgroup> shared_dot: array<f64, 256>;
var<workgroup> shared_sq: array<f64, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let tid = lid.x;
    let n = params.n;
    let beta_prev = params.beta_prev;

    // ── Step 2: α = dot(v_k, w) via parallel reduction ─────────────────────
    var dot_sum: f64 = 0.0;
    var i = tid;
    while (i < n) {
        dot_sum = dot_sum + v_k[i] * w[i];
        i = i + 256u;
    }
    shared_dot[tid] = dot_sum;
    workgroupBarrier();

    for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
        if (tid < stride) {
            shared_dot[tid] = shared_dot[tid] + shared_dot[tid + stride];
        }
        workgroupBarrier();
    }
    let alpha = shared_dot[0];

    // ── Step 3: w_modified[i] = w[i] - α*v_k[i] - β_prev*v_prev[i] ─────────
    // Store in v_next (used as scratch)
    i = tid;
    while (i < n) {
        v_next[i] = w[i] - alpha * v_k[i] - beta_prev * v_prev[i];
        i = i + 256u;
    }
    workgroupBarrier();

    // ── Step 4: β = ||w_modified|| via parallel reduction ─────────────────
    var sq_sum: f64 = 0.0;
    i = tid;
    while (i < n) {
        let val = v_next[i];
        sq_sum = sq_sum + val * val;
        i = i + 256u;
    }
    shared_sq[tid] = sq_sum;
    workgroupBarrier();

    for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
        if (tid < stride) {
            shared_sq[tid] = shared_sq[tid] + shared_sq[tid + stride];
        }
        workgroupBarrier();
    }
    let beta = sqrt(shared_sq[0]);

    // ── Step 5: v_next[i] = w_modified[i] / β ──────────────────────────────
    i = tid;
    while (i < n) {
        v_next[i] = select(v_next[i] / beta, v_next[i], beta == 0.0); // guard div-by-zero
        i = i + 256u;
    }

    // Store alpha and beta in tridiag (only thread 0)
    if (tid == 0u) {
        tridiag[0] = alpha;
        tridiag[1] = beta;
    }
}

// SPDX-License-Identifier: AGPL-3.0-or-later
// Weighted Inner Product with Workgroup Reduction (f64)
//
// Computes: result = Σ_k w[k] · a[k] · b[k]   (weighted dot product)
//
// With high-performance workgroup tree reduction over k using shared memory.
// For large vectors (>10k elements), this is significantly faster than
// sequential accumulation.
//
// Applications:
//   - Galerkin methods: <φ_i|W|φ_j> = Σ φ_i(k)·W(k)·φ_j(k)
//   - FEM assembly: element stiffness/mass matrices
//   - Spectral methods: expansion coefficients
//   - Correlation computation: Σ w_i·x_i·y_i
//   - Nuclear physics: potential matrix elements (validated by hotSpring)
//
// Validated by: hotSpring nuclear EOS study (169/169 acceptance checks)
// Deep Debt: pure WGSL, f64, self-contained

// ═══════════════════════════════════════════════════════════════════
// Kernel 1: Simple weighted dot product (no reduction)
//
// For small vectors where sequential is fine.
// Dispatch: (1, 1, 1) — single thread computes the sum
// ═══════════════════════════════════════════════════════════════════

struct DotParams {
    n: u32,         // Vector length
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> weights: array<f64>;
@group(0) @binding(1) var<storage, read> vec_a: array<f64>;
@group(0) @binding(2) var<storage, read> vec_b: array<f64>;
@group(0) @binding(3) var<storage, read_write> result: array<f64>;  // [1] for simple, [n_wg] for parallel
@group(0) @binding(4) var<uniform> params: DotParams;

@compute @workgroup_size(1)
fn weighted_dot_simple(@builtin(global_invocation_id) gid: vec3<u32>) {
    var sum = f64(0.0);
    for (var i = 0u; i < params.n; i++) {
        sum += weights[i] * vec_a[i] * vec_b[i];
    }
    result[0] = sum;
}

// ═══════════════════════════════════════════════════════════════════
// Kernel 2: Parallel weighted dot with workgroup reduction
//
// Each workgroup computes a partial sum using shared memory tree reduction.
// Final reduction of partial sums done by a second pass or on CPU.
//
// Dispatch: (ceil(n / 256), 1, 1)
// Output: partial_sums[n_workgroups]
// ═══════════════════════════════════════════════════════════════════

var<workgroup> shared_sum: array<f64, 256>;

@compute @workgroup_size(256)
fn weighted_dot_parallel(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    let idx = gid.x;

    // Each thread computes its contribution
    var local_sum = f64(0.0);
    if (idx < params.n) {
        local_sum = weights[idx] * vec_a[idx] * vec_b[idx];
    }

    // Load into shared memory
    shared_sum[lid.x] = local_sum;
    workgroupBarrier();

    // Tree reduction within workgroup
    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if (lid.x < stride) {
            shared_sum[lid.x] += shared_sum[lid.x + stride];
        }
        workgroupBarrier();
    }

    // Thread 0 writes the workgroup's partial sum
    if (lid.x == 0u) {
        result[wg_id.x] = shared_sum[0];
    }
}

// ═══════════════════════════════════════════════════════════════════
// Kernel 3: Final reduction of partial sums
//
// Sums the partial results from weighted_dot_parallel.
// Dispatch: (1, 1, 1) if n_partials <= 256, else cascade
// ═══════════════════════════════════════════════════════════════════

struct FinalParams {
    n_partials: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(1) @binding(0) var<uniform> final_params: FinalParams;
@group(1) @binding(1) var<storage, read> partial_sums: array<f64>;
@group(1) @binding(2) var<storage, read_write> final_result: array<f64>;

var<workgroup> shared_final: array<f64, 256>;

@compute @workgroup_size(256)
fn final_reduce(
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    // Load partial sums (each thread may load one)
    var local_val = f64(0.0);
    if (lid.x < final_params.n_partials) {
        local_val = partial_sums[lid.x];
    }

    shared_final[lid.x] = local_val;
    workgroupBarrier();

    // Tree reduction
    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if (lid.x < stride) {
            shared_final[lid.x] += shared_final[lid.x + stride];
        }
        workgroupBarrier();
    }

    if (lid.x == 0u) {
        final_result[0] = shared_final[0];
    }
}

// ═══════════════════════════════════════════════════════════════════
// Kernel 4: Batched weighted dot products
//
// Computes M weighted dot products in parallel:
//   result[m] = Σ_k w[m,k] · a[m,k] · b[m,k]
//
// Useful for: batch Galerkin assembly, multiple matrix elements at once
// Dispatch: (ceil(n / 256), m, 1)
// ═══════════════════════════════════════════════════════════════════

struct BatchParams {
    n: u32,         // Vector length per dot product
    m: u32,         // Number of dot products
    _pad0: u32,
    _pad1: u32,
}

@group(2) @binding(0) var<uniform> batch_params: BatchParams;
@group(2) @binding(1) var<storage, read> batch_weights: array<f64>;  // [m × n]
@group(2) @binding(2) var<storage, read> batch_a: array<f64>;        // [m × n]
@group(2) @binding(3) var<storage, read> batch_b: array<f64>;        // [m × n]
@group(2) @binding(4) var<storage, read_write> batch_result: array<f64>;  // [m × n_wg]

var<workgroup> shared_batch: array<f64, 256>;

@compute @workgroup_size(256)
fn weighted_dot_batched(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    let vec_idx = gid.x;     // Index within vector
    let batch_idx = gid.y;   // Which dot product

    if (batch_idx >= batch_params.m) { return; }

    let n = batch_params.n;
    let base = batch_idx * n;

    // Each thread computes its contribution
    var local_sum = f64(0.0);
    if (vec_idx < n) {
        local_sum = batch_weights[base + vec_idx]
                  * batch_a[base + vec_idx]
                  * batch_b[base + vec_idx];
    }

    shared_batch[lid.x] = local_sum;
    workgroupBarrier();

    // Tree reduction
    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if (lid.x < stride) {
            shared_batch[lid.x] += shared_batch[lid.x + stride];
        }
        workgroupBarrier();
    }

    if (lid.x == 0u) {
        let n_wg = (n + 255u) / 256u;
        batch_result[batch_idx * n_wg + wg_id.x] = shared_batch[0];
    }
}

// ═══════════════════════════════════════════════════════════════════
// Kernel 5: Unweighted dot product (w = 1)
//
// Simplified version when no weights are needed.
// result = Σ_k a[k] · b[k]
// ═══════════════════════════════════════════════════════════════════

@compute @workgroup_size(256)
fn dot_parallel(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    let idx = gid.x;

    var local_sum = f64(0.0);
    if (idx < params.n) {
        local_sum = vec_a[idx] * vec_b[idx];
    }

    shared_sum[lid.x] = local_sum;
    workgroupBarrier();

    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if (lid.x < stride) {
            shared_sum[lid.x] += shared_sum[lid.x + stride];
        }
        workgroupBarrier();
    }

    if (lid.x == 0u) {
        result[wg_id.x] = shared_sum[0];
    }
}

// ═══════════════════════════════════════════════════════════════════
// Kernel 6: Vector norm squared  ||v||² = Σ v[k]²
//
// Special case for self-dot-product.
// ═══════════════════════════════════════════════════════════════════

@compute @workgroup_size(256)
fn norm_squared_parallel(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    let idx = gid.x;

    var local_sum = f64(0.0);
    if (idx < params.n) {
        let v = vec_a[idx];
        local_sum = v * v;
    }

    shared_sum[lid.x] = local_sum;
    workgroupBarrier();

    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if (lid.x < stride) {
            shared_sum[lid.x] += shared_sum[lid.x + stride];
        }
        workgroupBarrier();
    }

    if (lid.x == 0u) {
        result[wg_id.x] = shared_sum[0];
    }
}

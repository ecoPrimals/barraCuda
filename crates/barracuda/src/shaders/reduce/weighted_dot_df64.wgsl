// SPDX-License-Identifier: AGPL-3.0-or-later
//
// weighted_dot_df64.wgsl — Weighted inner product with DF64 workgroup memory
//
// Identical algorithm to weighted_dot_f64.wgsl, but workgroup shared memory
// uses f32-pair (DF64) accumulators instead of native f64. This avoids the
// shared-memory f64 reliability issue on Hybrid-precision devices (NVK,
// Titan V, consumer GPUs where f64 shared memory returns zeros).
//
// Buffer layout: UNCHANGED — input/output are array<f64>.
//
// Prepend: df64_core.wgsl (auto-injected by Rust wrapper)

struct DotParams {
    n: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> weights: array<f64>;
@group(0) @binding(1) var<storage, read> vec_a: array<f64>;
@group(0) @binding(2) var<storage, read> vec_b: array<f64>;
@group(0) @binding(3) var<storage, read_write> result: array<f64>;
@group(0) @binding(4) var<uniform> params: DotParams;

// DF64 workgroup accumulators (hi/lo pairs replace array<f64, 256>)
var<workgroup> s_sum_hi: array<f32, 256>;
var<workgroup> s_sum_lo: array<f32, 256>;

// ═══════════════════════════════════════════════════════════════════
// Kernel 1: Simple weighted dot product (no reduction, no shared mem)
// Dispatch: (1, 1, 1)
// ═══════════════════════════════════════════════════════════════════

@compute @workgroup_size(1)
fn weighted_dot_simple(@builtin(global_invocation_id) gid: vec3<u32>) {
    var acc = df64_zero();
    for (var i = 0u; i < params.n; i++) {
        let w = df64_from_f64(weights[i]);
        let a = df64_from_f64(vec_a[i]);
        let b = df64_from_f64(vec_b[i]);
        acc = df64_add(acc, df64_mul(w, df64_mul(a, b)));
    }
    result[0] = df64_to_f64(acc);
}

// ═══════════════════════════════════════════════════════════════════
// Kernel 2: Parallel weighted dot with DF64 workgroup reduction
// Dispatch: (ceil(n / 256), 1, 1)
// Output: partial_sums[n_workgroups]
// ═══════════════════════════════════════════════════════════════════

@compute @workgroup_size(256)
fn weighted_dot_parallel(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    let idx = gid.x;

    var local_sum = df64_zero();
    if (idx < params.n) {
        // @ilp_region begin — three independent memory loads
        let w = df64_from_f64(weights[idx]);
        let a = df64_from_f64(vec_a[idx]);
        let b = df64_from_f64(vec_b[idx]);
        local_sum = df64_mul(w, df64_mul(a, b));
        // @ilp_region end
    }

    s_sum_hi[lid.x] = local_sum.hi;
    s_sum_lo[lid.x] = local_sum.lo;
    workgroupBarrier();

    for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
        if (lid.x < stride) {
            // @ilp_region begin — two independent shared memory loads
            let a_val = Df64(s_sum_hi[lid.x], s_sum_lo[lid.x]);
            let b_val = Df64(s_sum_hi[lid.x + stride], s_sum_lo[lid.x + stride]);
            let sum = df64_add(a_val, b_val);
            // @ilp_region end
            s_sum_hi[lid.x] = sum.hi;
            s_sum_lo[lid.x] = sum.lo;
        }
        workgroupBarrier();
    }

    if (lid.x == 0u) {
        result[wg_id.x] = df64_to_f64(Df64(s_sum_hi[0], s_sum_lo[0]));
    }
}

// ═══════════════════════════════════════════════════════════════════
// Kernel 3: Final reduction of partial sums
// Dispatch: (1, 1, 1) if n_partials <= 256
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

var<workgroup> s_final_hi: array<f32, 256>;
var<workgroup> s_final_lo: array<f32, 256>;

@compute @workgroup_size(256)
fn final_reduce(
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    var local_val = df64_zero();
    if (lid.x < final_params.n_partials) {
        local_val = df64_from_f64(partial_sums[lid.x]);
    }

    s_final_hi[lid.x] = local_val.hi;
    s_final_lo[lid.x] = local_val.lo;
    workgroupBarrier();

    for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
        if (lid.x < stride) {
            let a_val = Df64(s_final_hi[lid.x], s_final_lo[lid.x]);
            let b_val = Df64(s_final_hi[lid.x + stride], s_final_lo[lid.x + stride]);
            let sum = df64_add(a_val, b_val);
            s_final_hi[lid.x] = sum.hi;
            s_final_lo[lid.x] = sum.lo;
        }
        workgroupBarrier();
    }

    if (lid.x == 0u) {
        final_result[0] = df64_to_f64(Df64(s_final_hi[0], s_final_lo[0]));
    }
}

// ═══════════════════════════════════════════════════════════════════
// Kernel 4: Batched weighted dot products with DF64 reduction
// result[m] = Σ_k w[m,k] · a[m,k] · b[m,k]
// Dispatch: (ceil(n / 256), m, 1)
// ═══════════════════════════════════════════════════════════════════

struct BatchParams {
    n: u32,
    m: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(2) @binding(0) var<uniform> batch_params: BatchParams;
@group(2) @binding(1) var<storage, read> batch_weights: array<f64>;
@group(2) @binding(2) var<storage, read> batch_a: array<f64>;
@group(2) @binding(3) var<storage, read> batch_b: array<f64>;
@group(2) @binding(4) var<storage, read_write> batch_result: array<f64>;

var<workgroup> s_batch_hi: array<f32, 256>;
var<workgroup> s_batch_lo: array<f32, 256>;

@compute @workgroup_size(256)
fn weighted_dot_batched(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    let vec_idx = gid.x;
    let batch_idx = gid.y;

    if (batch_idx >= batch_params.m) { return; }

    let n = batch_params.n;
    let base = batch_idx * n;

    var local_sum = df64_zero();
    if (vec_idx < n) {
        // @ilp_region begin — three independent memory loads
        let w = df64_from_f64(batch_weights[base + vec_idx]);
        let a = df64_from_f64(batch_a[base + vec_idx]);
        let b = df64_from_f64(batch_b[base + vec_idx]);
        local_sum = df64_mul(w, df64_mul(a, b));
        // @ilp_region end
    }

    s_batch_hi[lid.x] = local_sum.hi;
    s_batch_lo[lid.x] = local_sum.lo;
    workgroupBarrier();

    for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
        if (lid.x < stride) {
            // @ilp_region begin — two independent shared memory loads
            let a_val = Df64(s_batch_hi[lid.x], s_batch_lo[lid.x]);
            let b_val = Df64(s_batch_hi[lid.x + stride], s_batch_lo[lid.x + stride]);
            let sum = df64_add(a_val, b_val);
            // @ilp_region end
            s_batch_hi[lid.x] = sum.hi;
            s_batch_lo[lid.x] = sum.lo;
        }
        workgroupBarrier();
    }

    if (lid.x == 0u) {
        let n_wg = (n + 255u) / 256u;
        batch_result[batch_idx * n_wg + wg_id.x] = df64_to_f64(Df64(s_batch_hi[0], s_batch_lo[0]));
    }
}

// ═══════════════════════════════════════════════════════════════════
// Kernel 5: Unweighted dot product with DF64 reduction
// result = Σ_k a[k] · b[k]
// ═══════════════════════════════════════════════════════════════════

@compute @workgroup_size(256)
fn dot_parallel(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    let idx = gid.x;

    var local_sum = df64_zero();
    if (idx < params.n) {
        let a = df64_from_f64(vec_a[idx]);
        let b = df64_from_f64(vec_b[idx]);
        local_sum = df64_mul(a, b);
    }

    s_sum_hi[lid.x] = local_sum.hi;
    s_sum_lo[lid.x] = local_sum.lo;
    workgroupBarrier();

    for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
        if (lid.x < stride) {
            let a_val = Df64(s_sum_hi[lid.x], s_sum_lo[lid.x]);
            let b_val = Df64(s_sum_hi[lid.x + stride], s_sum_lo[lid.x + stride]);
            let sum = df64_add(a_val, b_val);
            s_sum_hi[lid.x] = sum.hi;
            s_sum_lo[lid.x] = sum.lo;
        }
        workgroupBarrier();
    }

    if (lid.x == 0u) {
        result[wg_id.x] = df64_to_f64(Df64(s_sum_hi[0], s_sum_lo[0]));
    }
}

// ═══════════════════════════════════════════════════════════════════
// Kernel 6: Vector norm squared with DF64 reduction
// ||v||² = Σ v[k]²
// ═══════════════════════════════════════════════════════════════════

@compute @workgroup_size(256)
fn norm_squared_parallel(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wg_id: vec3<u32>,
) {
    let idx = gid.x;

    var local_sum = df64_zero();
    if (idx < params.n) {
        let v = df64_from_f64(vec_a[idx]);
        local_sum = df64_mul(v, v);
    }

    s_sum_hi[lid.x] = local_sum.hi;
    s_sum_lo[lid.x] = local_sum.lo;
    workgroupBarrier();

    for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
        if (lid.x < stride) {
            let a_val = Df64(s_sum_hi[lid.x], s_sum_lo[lid.x]);
            let b_val = Df64(s_sum_hi[lid.x + stride], s_sum_lo[lid.x + stride]);
            let sum = df64_add(a_val, b_val);
            s_sum_hi[lid.x] = sum.hi;
            s_sum_lo[lid.x] = sum.lo;
        }
        workgroupBarrier();
    }

    if (lid.x == 0u) {
        result[wg_id.x] = df64_to_f64(Df64(s_sum_hi[0], s_sum_lo[0]));
    }
}

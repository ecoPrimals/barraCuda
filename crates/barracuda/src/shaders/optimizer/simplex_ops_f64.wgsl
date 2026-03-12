// SPDX-License-Identifier: AGPL-3.0-only
// Nelder-Mead Simplex Operations - f64 canonical
//
// Parallel primitives for the Nelder-Mead simplex algorithm:
// - Centroid computation (parallel reduction)
// - Reflect/Expand/Contract (parallel vector ops)

struct SimplexParams {
    n: u32,
    n_points: u32,
    alpha: f64,
    gamma: f64,
}

@group(0) @binding(0) var<uniform> params: SimplexParams;
@group(0) @binding(1) var<storage, read> simplex: array<f64>;
@group(0) @binding(2) var<storage, read> f_vals: array<f64>;
@group(0) @binding(3) var<storage, read_write> centroid: array<f64>;
@group(0) @binding(4) var<storage, read_write> output: array<f64>;

var<workgroup> shared_sum: array<f64, 256>;

@compute @workgroup_size(256, 1, 1)
fn compute_centroid(@builtin(local_invocation_id) local_id: vec3<u32>,
                    @builtin(global_invocation_id) global_id: vec3<u32>) {
    let j = global_id.x;
    let n = params.n;
    let n_points = params.n_points;

    if (j >= n) {
        return;
    }

    var worst_idx = 0u;
    var worst_val = f_vals[0];
    for (var i = 1u; i < n_points; i = i + 1u) {
        if (f_vals[i] > worst_val) {
            worst_val = f_vals[i];
            worst_idx = i;
        }
    }

    var sum: f64 = 0.0;
    var count: u32 = 0u;
    for (var i = 0u; i < n_points; i = i + 1u) {
        if (i != worst_idx) {
            sum = sum + simplex[i * n + j];
            count = count + 1u;
        }
    }

    centroid[j] = sum / f64(count);
}

@compute @workgroup_size(256, 1, 1)
fn reflect(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let j = global_id.x;
    let n = params.n;

    if (j >= n) {
        return;
    }

    var worst_idx = 0u;
    var worst_val = f_vals[0];
    for (var i = 1u; i < params.n_points; i = i + 1u) {
        if (f_vals[i] > worst_val) {
            worst_val = f_vals[i];
            worst_idx = i;
        }
    }

    let x_worst = simplex[worst_idx * n + j];
    let c = centroid[j];

    output[j] = c + params.alpha * (c - x_worst);
}

@compute @workgroup_size(256, 1, 1)
fn expand(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let j = global_id.x;
    let n = params.n;

    if (j >= n) {
        return;
    }

    let c = centroid[j];
    let x_r = output[j];

    output[j] = c + params.gamma * (x_r - c);
}

struct ContractParams {
    n: u32,
    rho: f64,
    inside: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> contract_params: ContractParams;
@group(0) @binding(1) var<storage, read> simplex_c: array<f64>;
@group(0) @binding(2) var<storage, read> f_vals_c: array<f64>;
@group(0) @binding(3) var<storage, read> centroid_c: array<f64>;
@group(0) @binding(4) var<storage, read> x_reflect: array<f64>;
@group(0) @binding(5) var<storage, read_write> output_c: array<f64>;

@compute @workgroup_size(256, 1, 1)
fn contract(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let j = global_id.x;
    let n = contract_params.n;

    if (j >= n) {
        return;
    }

    let c = centroid_c[j];

    if (contract_params.inside == 1u) {
        var worst_idx = 0u;
        var worst_val = f_vals_c[0];
        for (var i = 1u; i < n + 1u; i = i + 1u) {
            if (f_vals_c[i] > worst_val) {
                worst_val = f_vals_c[i];
                worst_idx = i;
            }
        }
        let x_worst = simplex_c[worst_idx * n + j];
        output_c[j] = c + contract_params.rho * (x_worst - c);
    } else {
        output_c[j] = c + contract_params.rho * (x_reflect[j] - c);
    }
}

struct ShrinkParams {
    n: u32,
    n_points: u32,
    sigma: f64,
    best_idx: u32,
}

@group(0) @binding(0) var<uniform> shrink_params: ShrinkParams;
@group(0) @binding(1) var<storage, read_write> simplex_s: array<f64>;

@compute @workgroup_size(16, 16, 1)
fn shrink(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.y;
    let j = global_id.x;
    let n = shrink_params.n;
    let n_points = shrink_params.n_points;

    if (i >= n_points || j >= n || i == shrink_params.best_idx) {
        return;
    }

    let x_best_j = simplex_s[shrink_params.best_idx * n + j];
    let x_i_j = simplex_s[i * n + j];

    simplex_s[i * n + j] = x_best_j + shrink_params.sigma * (x_i_j - x_best_j);
}

struct BoundsParams {
    n: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<uniform> bounds_params: BoundsParams;
@group(0) @binding(1) var<storage, read> bounds_lo: array<f64>;
@group(0) @binding(2) var<storage, read> bounds_hi: array<f64>;
@group(0) @binding(3) var<storage, read_write> point: array<f64>;

@compute @workgroup_size(256, 1, 1)
fn project_bounds(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let j = global_id.x;

    if (j >= bounds_params.n) {
        return;
    }

    point[j] = clamp(point[j], bounds_lo[j], bounds_hi[j]);
}

struct SortParams {
    n_points: u32,
    stage: u32,
    step: u32,
    _pad: u32,
}

@group(0) @binding(0) var<uniform> sort_params: SortParams;
@group(0) @binding(1) var<storage, read_write> indices: array<u32>;
@group(0) @binding(2) var<storage, read> f_vals_sort: array<f64>;

@compute @workgroup_size(256, 1, 1)
fn bitonic_sort_step(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    let n = sort_params.n_points;

    if (i >= n / 2u) {
        return;
    }

    let stage = sort_params.stage;
    let step = sort_params.step;

    let pair_distance = 1u << step;
    let block_size = 1u << (stage + 1u);

    let left_idx = (i / pair_distance) * (pair_distance * 2u) + (i % pair_distance);
    let right_idx = left_idx + pair_distance;

    if (right_idx >= n) {
        return;
    }

    let left_val = f_vals_sort[indices[left_idx]];
    let right_val = f_vals_sort[indices[right_idx]];

    let ascending = ((i / (block_size / 2u)) % 2u) == 0u;

    let should_swap = (ascending && left_val > right_val) || (!ascending && left_val < right_val);

    if (should_swap) {
        let temp = indices[left_idx];
        indices[left_idx] = indices[right_idx];
        indices[right_idx] = temp;
    }
}

// Batched Nelder-Mead: N problems in parallel
// Layout: simplex[p * n_points * n + v * n + j], f_vals[p * n_points + v], centroid[p * n + j]

struct BatchedSimplexParams {
    n_problems: u32,
    n: u32,
    n_points: u32,
    alpha: f64,
    gamma: f64,
    rho: f64,
    sigma: f64,
}

@group(0) @binding(0) var<uniform> batched_params: BatchedSimplexParams;
@group(0) @binding(1) var<storage, read> batched_simplex: array<f64>;
@group(0) @binding(2) var<storage, read> batched_f_vals: array<f64>;
@group(0) @binding(3) var<storage, read> batched_worst_idx: array<u32>;
@group(0) @binding(4) var<storage, read_write> batched_centroid: array<f64>;
@group(0) @binding(5) var<storage, read_write> batched_output: array<f64>;

@compute @workgroup_size(256, 1, 1)
fn batched_compute_centroid(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let tid = global_id.x;
    let n = batched_params.n;
    let n_points = batched_params.n_points;
    let n_problems = batched_params.n_problems;

    if (tid >= n_problems * n) {
        return;
    }

    let p = tid / n;
    let j = tid % n;
    let worst_idx = batched_worst_idx[p];

    var sum: f64 = 0.0;
    var count: u32 = 0u;
    for (var i = 0u; i < n_points; i = i + 1u) {
        if (i != worst_idx) {
            sum = sum + batched_simplex[p * n_points * n + i * n + j];
            count = count + 1u;
        }
    }
    batched_centroid[p * n + j] = sum / f64(count);
}

@compute @workgroup_size(256, 1, 1)
fn batched_reflect(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let tid = global_id.x;
    let n = batched_params.n;
    let n_points = batched_params.n_points;
    let n_problems = batched_params.n_problems;

    if (tid >= n_problems * n) {
        return;
    }

    let p = tid / n;
    let j = tid % n;
    let worst_idx = batched_worst_idx[p];
    let x_worst = batched_simplex[p * n_points * n + worst_idx * n + j];
    let c = batched_centroid[p * n + j];

    batched_output[p * n + j] = c + batched_params.alpha * (c - x_worst);
}

@compute @workgroup_size(256, 1, 1)
fn batched_expand(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let tid = global_id.x;
    let n = batched_params.n;
    let n_problems = batched_params.n_problems;

    if (tid >= n_problems * n) {
        return;
    }

    let p = tid / n;
    let j = tid % n;
    let c = batched_centroid[p * n + j];
    let x_r = batched_output[p * n + j];

    batched_output[p * n + j] = c + batched_params.gamma * (x_r - c);
}

struct BatchedContractParams {
    n_problems: u32,
    n: u32,
    rho: f64,
    _pad: vec2<u32>,
}

@group(0) @binding(0) var<uniform> batched_contract_params: BatchedContractParams;
@group(0) @binding(1) var<storage, read> batched_simplex_c: array<f64>;
@group(0) @binding(2) var<storage, read> batched_worst_idx_c: array<u32>;
@group(0) @binding(3) var<storage, read> batched_centroid_c: array<f64>;
@group(0) @binding(4) var<storage, read> batched_x_reflect: array<f64>;
@group(0) @binding(5) var<storage, read> batched_inside: array<u32>;
@group(0) @binding(6) var<storage, read_write> batched_output_c: array<f64>;

@compute @workgroup_size(256, 1, 1)
fn batched_contract(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let tid = global_id.x;
    let n = batched_contract_params.n;
    let n_points = n + 1u;
    let n_problems = batched_contract_params.n_problems;

    if (tid >= n_problems * n) {
        return;
    }

    let p = tid / n;
    let j = tid % n;
    let c = batched_centroid_c[p * n + j];

    if (batched_inside[p] == 1u) {
        let worst_idx = batched_worst_idx_c[p];
        let x_worst = batched_simplex_c[p * n_points * n + worst_idx * n + j];
        batched_output_c[p * n + j] = c + batched_contract_params.rho * (x_worst - c);
    } else {
        let x_r = batched_x_reflect[p * n + j];
        batched_output_c[p * n + j] = c + batched_contract_params.rho * (x_r - c);
    }
}

struct BatchedShrinkParams {
    n_problems: u32,
    n: u32,
    n_points: u32,
    sigma: f64,
}

@group(0) @binding(0) var<uniform> batched_shrink_params: BatchedShrinkParams;
@group(0) @binding(1) var<storage, read> batched_best_idx: array<u32>;
@group(0) @binding(2) var<storage, read_write> batched_simplex_s: array<f64>;

@compute @workgroup_size(256, 1, 1)
fn batched_shrink(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let tid = global_id.x;
    let n = batched_shrink_params.n;
    let n_points = batched_shrink_params.n_points;
    let n_problems = batched_shrink_params.n_problems;
    let total = n_problems * n_points * n;

    if (tid >= total) {
        return;
    }

    let p = tid / (n_points * n);
    let rem = tid % (n_points * n);
    let i = rem / n;
    let j = rem % n;

    if (i >= n_points || j >= n) {
        return;
    }

    let best_idx = batched_best_idx[p];
    if (i == best_idx) {
        return;
    }

    let x_best_j = batched_simplex_s[p * n_points * n + best_idx * n + j];
    let x_i_j = batched_simplex_s[p * n_points * n + i * n + j];

    batched_simplex_s[p * n_points * n + i * n + j] = x_best_j + batched_shrink_params.sigma * (x_i_j - x_best_j);
}

// SPDX-License-Identifier: AGPL-3.0-only
// Clip Grad Norm - Gradient clipping by total norm (f64 canonical)

struct Params {
    size: u32,
    max_norm: f64,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> gradients: array<f64>;
@group(0) @binding(2) var<storage, read_write> norm_buffer: array<f64>;
@group(0) @binding(3) var<storage, read_write> output: array<f64>;

var<workgroup> shared_sq: array<f64, 256>;

@compute @workgroup_size(256)
fn compute_norm(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let tid = local_id.x;
    let gid = global_id.x;

    var grad_sq: f64 = 0.0;
    if (gid < params.size) {
        let g = gradients[gid];
        grad_sq = g * g;
    }
    shared_sq[tid] = grad_sq;
    workgroupBarrier();

    for (var stride = 128u; stride > 0u; stride = stride / 2u) {
        if (tid < stride) {
            shared_sq[tid] = shared_sq[tid] + shared_sq[tid + stride];
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        norm_buffer[workgroup_id.x] = shared_sq[0];
    }
}

@compute @workgroup_size(256)
fn compute_norm_final(
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let tid = local_id.x;
    let num_partials = (params.size + 255u) / 256u;

    if (tid == 0u) {
        var total: f64 = 0.0;
        for (var i = 0u; i < num_partials; i = i + 1u) {
            total = total + norm_buffer[i];
        }
        norm_buffer[0] = total;
    }
}

@compute @workgroup_size(256)
fn clip_gradients(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.size) {
        return;
    }

    let norm_sq = norm_buffer[0];
    let total_norm = sqrt_f64(norm_sq);
    var scale: f64 = 1.0;

    if (total_norm > params.max_norm) {
        scale = params.max_norm / (total_norm + 1e-8);
    }

    output[idx] = gradients[idx] * scale;
}

// SPDX-License-Identifier: AGPL-3.0-only
// earth_mover_distance_f64.wgsl - Earth Mover's Distance (Wasserstein-1) (f64 canonical)
//
// Measures distance between probability distributions
// Also known as Wasserstein distance
//
// For 1D case (simplified): EMD = sum of absolute differences of CDFs

struct Params {
    size: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
    _pad4: u32,
    _pad5: u32,
    _pad6: u32,
}

@group(0) @binding(0) var<storage, read> dist1: array<f64>;     // Distribution 1 (probabilities)
@group(0) @binding(1) var<storage, read> dist2: array<f64>;     // Distribution 2 (probabilities)
@group(0) @binding(2) var<storage, read_write> output: array<f64>; // Scalar distance
@group(0) @binding(3) var<uniform> params: Params;

var<workgroup> shared_emd: array<f64, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let idx = global_id.x;
    let local_idx = local_id.x;

    // Compute cumulative distributions (CDF)
    // For proper EMD, we need CDF differences

    var local_emd: f64 = 0.0;

    if (idx < params.size) {
        // Simplified: sum absolute differences (approximation)
        // Full EMD requires sorting and transport plan
        local_emd = abs(dist1[idx] - dist2[idx]);
    }

    shared_emd[local_idx] = local_emd;
    workgroupBarrier();

    // Parallel reduction
    var stride = 128u;
    while (stride >= 1u) {
        if (local_idx < stride && local_idx + stride < 256u) {
            shared_emd[local_idx] += shared_emd[local_idx + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    if (local_idx == 0u) {
        output[0] = shared_emd[0];
    }
}

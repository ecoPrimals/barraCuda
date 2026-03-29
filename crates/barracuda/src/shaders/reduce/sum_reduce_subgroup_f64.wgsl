// SPDX-License-Identifier: AGPL-3.0-only
//
// Subgroup-accelerated tree reduction: uses subgroupAdd for warp/wavefront-level
// reduction, then shared memory for cross-subgroup accumulation.
//
// Absorbed from hotSpring V0632 (March 2026).
//
// RTX 3090: subgroup_size=32 -> 8 subgroups per WG, 3 barrier steps
// RX 6950 XT: subgroup_size=64 -> 4 subgroups per WG, 2 barrier steps
// vs 8 barrier steps in the shared-memory-only version.
//
// NOTE: Do NOT add `enable subgroups;` — naga 28 generates broken SPIR-V
// when the enable directive is present, causing all subgroup ops to return 0.
// wgpu's SUBGROUP device feature is sufficient; the directive is redundant.

struct ReduceParams {
    size: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;
@group(0) @binding(2) var<uniform> params: ReduceParams;

var<workgroup> wg_partial: array<f64, 8>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
    @builtin(subgroup_invocation_id) sg_id: u32,
    @builtin(subgroup_size) sg_size: u32,
) {
    let tid = local_id.x;
    let gid = global_id.x + global_id.y * nwg.x * 256u;

    var val: f64 = f64(0.0);
    if gid < params.size {
        val = input[gid];
    }

    let sg_sum = subgroupAdd(val);

    let sg_idx = tid / sg_size;
    if sg_id == 0u {
        wg_partial[sg_idx] = sg_sum;
    }
    workgroupBarrier();

    let n_subgroups = 256u / sg_size;
    if tid < n_subgroups {
        val = wg_partial[tid];
    } else {
        val = f64(0.0);
    }

    if tid < sg_size {
        let final_sum = subgroupAdd(val);
        if tid == 0u {
            let wg_linear = workgroup_id.x + workgroup_id.y * nwg.x;
            output[wg_linear] = final_sum;
        }
    }
}

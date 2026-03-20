// SPDX-License-Identifier: AGPL-3.0-or-later
// ArgSort_f64.wgsl — Return indices that sort the tensor (f64 canonical)
// Returns indices that would sort the input tensor
//
// Algorithm: Parallel bitonic sort with index tracking
// Same as sort.wgsl but tracks original indices

struct Params {
    size: u32,
    descending: u32,  // 0 = ascending, 1 = descending
    stage: u32,       // Current stage in bitonic sort
    step: u32,        // Current step within stage
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> values: array<f64>;   // Values being sorted
@group(0) @binding(2) var<storage, read_write> indices: array<u32>;  // Indices being sorted

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.size / 2u) {
        return;
    }

    // Bitonic sort: compute partner index
    let step_size = 1u << params.step;
    let stage_size = 1u << params.stage;

    let block_idx = idx / step_size;
    let within_block = idx % step_size;
    let partner_idx = (block_idx * 2u * step_size) + (2u * step_size - 1u - within_block);
    let my_idx = (block_idx * 2u * step_size) + within_block;

    if (partner_idx < params.size && my_idx < params.size) {
        let val1 = values[my_idx];
        let val2 = values[partner_idx];

        // Determine sort direction
        let ascending = ((my_idx / stage_size) % 2u) == 0u;
        var should_swap = false;

        if (params.descending == 0u) {
            should_swap = (ascending && val1 > val2) || (!ascending && val1 < val2);
        } else {
            should_swap = (ascending && val1 < val2) || (!ascending && val1 > val2);
        }

        if (should_swap) {
            // Swap values
            values[my_idx] = val2;
            values[partner_idx] = val1;

            // Swap indices
            let temp_idx = indices[my_idx];
            indices[my_idx] = indices[partner_idx];
            indices[partner_idx] = temp_idx;
        }
    }
}

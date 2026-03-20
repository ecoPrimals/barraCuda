// SPDX-License-Identifier: AGPL-3.0-or-later
// Sort - Parallel sorting (bitonic sort for GPU) (f64 canonical)
// Sorts tensor elements in ascending or descending order
//
// Algorithm: Bitonic sort (GPU-friendly comparison network)
// - Works in log²(n) passes
// - Each pass performs parallel comparisons and swaps
//
// Note: Most efficient for power-of-2 sizes

struct Params {
    size: u32,
    descending: u32,  // 0 = ascending, 1 = descending
    stage: u32,       // Current stage in bitonic sort
    step: u32,        // Current step within stage
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> data: array<f64>;

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
        let val1 = data[my_idx];
        let val2 = data[partner_idx];
        
        // Determine sort direction for this pair
        let ascending = ((my_idx / stage_size) % 2u) == 0u;
        var should_swap = false;
        
        if (params.descending == 0u) {
            should_swap = (ascending && val1 > val2) || (!ascending && val1 < val2);
        } else {
            should_swap = (ascending && val1 < val2) || (!ascending && val1 > val2);
        }
        
        if (should_swap) {
            data[my_idx] = val2;
            data[partner_idx] = val1;
        }
    }
}

// Masked Select - Extract elements where mask is true (f64 canonical)
// Selects elements from input where corresponding mask is non-zero
//
// Algorithm:
// 1. Parallel scan to compute output positions (prefix sum of mask)
// 2. Copy selected elements to compact output

struct Params {
    input_size: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f64>;
@group(0) @binding(2) var<storage, read> mask: array<u32>;    // 0 or 1 (boolean mask)
@group(0) @binding(3) var<storage, read> prefix_sum: array<u32>; // Computed externally
@group(0) @binding(4) var<storage, read_write> output: array<f64>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.input_size) {
        return;
    }

    // If mask is true, copy to output at position given by prefix sum
    if (mask[idx] != 0u) {
        let out_idx = prefix_sum[idx]; // exclusive scan gives 0-based output index
        output[out_idx] = input[idx];
    }
}

// SPDX-License-Identifier: AGPL-3.0-or-later
// TopK - Find top-k largest elements (pure WGSL) (f64 canonical)
// Returns indices of the k largest elements, sorted descending by value
//
// Algorithm: Rank-based selection
// For each output position i, find the element whose rank is i
// Rank = number of elements strictly greater (with index tiebreaking)
//
// Complexity: O(n * k) per workgroup - suitable for moderate n and small k

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<uniform> params: vec4<u32>;  // params.x = k

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let target_rank = global_id.x;
    let k = params.x;

    if (target_rank >= k) {
        return;
    }

    let n = arrayLength(&input);

    // Find the element whose descending rank equals target_rank
    for (var i = 0u; i < n; i++) {
        let val = input[i];
        var rank = 0u;

        for (var j = 0u; j < n; j++) {
            // Count elements that rank higher (larger value, or same value with lower index)
            if (input[j] > val || (input[j] == val && j < i)) {
                rank += 1u;
            }
        }

        if (rank == target_rank) {
            output[target_rank] = i;
            return;
        }
    }
}

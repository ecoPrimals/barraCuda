// SearchSorted - Binary search insertion indices (GPU parallel) (f64 canonical)
// Finds insertion points for values in a sorted array
//
// Example: searchsorted([1, 3, 5, 7], [2, 4, 6]) → [1, 2, 3]
//
// Algorithm:
// Each thread performs binary search for its value in the sorted array

struct Params {
    sorted_size: u32,
    values_size: u32,
    side_right: u32,  // 0 = left (default), 1 = right
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> sorted_array: array<f64>;  // Sorted input
@group(0) @binding(2) var<storage, read> values: array<f64>;        // Values to search for
@group(0) @binding(3) var<storage, read_write> output: array<u32>;  // Insertion indices

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.values_size) {
        return;
    }

    let value = values[idx];
    
    // Binary search
    var left = 0u;
    var right = params.sorted_size;
    
    while (left < right) {
        let mid = (left + right) / 2u;
        let mid_val = sorted_array[mid];
        
        if (params.side_right == 0u) {
            // Left side: find first position where value can be inserted
            if (mid_val < value) {
                left = mid + 1u;
            } else {
                right = mid;
            }
        } else {
            // Right side: find last position where value can be inserted
            if (mid_val <= value) {
                left = mid + 1u;
            } else {
                right = mid;
            }
        }
    }
    
    output[idx] = left;
}

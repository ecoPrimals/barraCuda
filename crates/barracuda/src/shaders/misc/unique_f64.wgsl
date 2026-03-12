// SPDX-License-Identifier: AGPL-3.0-only
// Unique - Find unique elements (GPU hash-based approach) (f64 canonical)
// Returns sorted unique values from input tensor
//
// Algorithm:
// 1. Hash each input value to a bucket
// 2. Mark unique values using atomic operations
// 3. Compact unique values to output
//
// Note: For production use, this is a simplified version.
// Full implementation would use parallel radix sort + unique filter.

struct Params {
    input_size: u32,
    num_buckets: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f64>;
@group(0) @binding(2) var<storage, read_write> hash_table: array<atomic<u32>>; // Hash table for unique detection
@group(0) @binding(3) var<storage, read_write> unique_flags: array<u32>;       // 1 if unique
@group(0) @binding(4) var<storage, read> prefix_sum: array<u32>;      // Prefix sum of unique_flags
@group(0) @binding(5) var<storage, read_write> output: array<f64>;

// Hash function for f64 - mix bits for better distribution of similar floats
fn hash_f64(val: f64) -> u32 {
    let bits = bitcast<vec2<u32>>(val);
    var h = bits.x ^ bits.y;
    h = h ^ (h >> 16u);
    h = h * 0x85ebca6bu;
    h = h ^ (h >> 13u);
    h = h * 0xc2b2ae35u;
    return (h ^ (h >> 16u)) % params.num_buckets;
}

// Step 1: Mark unique values in hash table
@compute @workgroup_size(256)
fn mark_unique(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.input_size) {
        return;
    }

    let val = input[idx];
    let hash = hash_f64(val);
    
    // Atomic compare-exchange to detect first occurrence
    let old = atomicCompareExchangeWeak(&hash_table[hash], 0u, 1u).old_value;
    if (old == 0u) {
        unique_flags[idx] = 1u; // First occurrence
    } else {
        unique_flags[idx] = 0u; // Duplicate
    }
}

// Step 2: Compact unique values to output
@compute @workgroup_size(256)
fn compact_unique(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.input_size) {
        return;
    }

    if (unique_flags[idx] == 1u) {
        // Use prefix sum to determine output position (exclusive scan gives 0-based index)
        let out_pos = prefix_sum[idx];
        output[out_pos] = input[idx];
    }
}

// SPDX-License-Identifier: AGPL-3.0-or-later
// Unique - Find unique elements (GPU hash-based approach) (f32 variant)
// Returns sorted unique values from input tensor
//
// f32 uses u32 bitcast (4 bytes) vs f64's vec2<u32> bitcast (8 bytes),
// so the hash function is structurally different.

struct Params {
    input_size: u32,
    num_buckets: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> hash_table: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> unique_flags: array<u32>;
@group(0) @binding(4) var<storage, read> prefix_sum: array<u32>;
@group(0) @binding(5) var<storage, read_write> output: array<f32>;

fn hash_f32(val: f32) -> u32 {
    var h = bitcast<u32>(val);
    h = h ^ (h >> 16u);
    h = h * 0x85ebca6bu;
    h = h ^ (h >> 13u);
    h = h * 0xc2b2ae35u;
    return (h ^ (h >> 16u)) % params.num_buckets;
}

@compute @workgroup_size(256)
fn mark_unique(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.input_size) {
        return;
    }

    let val = input[idx];
    let hash = hash_f32(val);

    let old = atomicCompareExchangeWeak(&hash_table[hash], 0u, 1u).old_value;
    if (old == 0u) {
        unique_flags[idx] = 1u;
    } else {
        unique_flags[idx] = 0u;
    }
}

@compute @workgroup_size(256)
fn compact_unique(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.input_size) {
        return;
    }

    if (unique_flags[idx] == 1u) {
        let out_pos = prefix_sum[idx];
        output[out_pos] = input[idx];
    }
}

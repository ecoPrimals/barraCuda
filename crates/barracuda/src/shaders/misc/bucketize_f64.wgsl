// Bucketize_f64.wgsl — Assign values to bins (f64 canonical)
// Maps each value to a bucket index based on boundaries

struct Params {
    size: u32,
    num_boundaries: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read> boundaries: array<f64>;
@group(0) @binding(2) var<storage, read_write> output: array<u32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if (idx >= params.size) {
        return;
    }

    let value = input[idx];
    var bucket: u32 = 0u;

    // Find which bucket this value belongs to
    for (var i = 0u; i < params.num_boundaries; i = i + 1u) {
        if (value >= boundaries[i]) {
            bucket = i + 1u;
        }
    }

    output[idx] = bucket;
}

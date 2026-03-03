// acos_f64.wgsl - Inverse cosine (arccosine) operation (f64 canonical)
// acos(x) returns the angle whose cosine is x, in range [0, π]
// Input must be in range [-1, 1]

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;
@group(0) @binding(2) var<uniform> metadata: Metadata;

struct Metadata {
    size: u32,
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= metadata.size) {
        return;
    }

    let x = input[idx];

    // acos is only defined for |x| <= 1
    // WGSL acos() returns NaN for out-of-range values
    output[idx] = acos_f64(x);
}

// Reshape - Pure metadata operation (zero-copy) (f64 canonical)
// Changes tensor shape without copying data
//
// Algorithm:
// This is a metadata-only operation. No actual computation needed.
// The WGSL shader is a simple copy for buffer compatibility.

struct Params {
    size: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> input: array<f64>;
@group(0) @binding(2) var<storage, read_write> output: array<f64>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.size) {
        return;
    }

    // Simple copy - reshape is handled at Tensor metadata level
    output[idx] = input[idx];
}

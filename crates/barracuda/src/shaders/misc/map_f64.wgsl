// SPDX-License-Identifier: AGPL-3.0-or-later
// Map: Generic element-wise transform (f64 canonical)
// CUDA equivalent: thrust::transform
// Use cases: Element-wise transforms, activation functions, normalization

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;

struct Params {
    size: u32,
    operation: u32,  // 0=Square, 1=Sqrt, 2=Abs, 3=Negate, 4=Reciprocal
}
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gid = global_id.x;
    
    if (gid >= params.size) {
        return;
    }
    
    let value = input[gid];
    var result: f64;
    
    switch (params.operation) {
        case 0u: { // Square
            result = value * value;
        }
        case 1u: { // Sqrt
            result = sqrt_f64(abs(value));
        }
        case 2u: { // Abs
            result = abs(value);
        }
        case 3u: { // Negate
            result = -value;
        }
        case 4u: { // Reciprocal
            result = 1.0 / value;
        }
        default: {
            result = value;
        }
    }
    
    output[gid] = result;
}

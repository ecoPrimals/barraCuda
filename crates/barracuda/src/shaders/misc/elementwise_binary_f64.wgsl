// SPDX-License-Identifier: AGPL-3.0-only
// ElementwiseBinary_f64.wgsl — C = A op B (f64 canonical)
// Supports: Add(0), Sub(1), Mul(2), Div(3), Pow(4), Max(5), Min(6)

@group(0) @binding(0) var<storage, read> a: array<f64>;
@group(0) @binding(1) var<storage, read> b: array<f64>;
@group(0) @binding(2) var<storage, read_write> output: array<f64>;

struct Params {
    size: u32,
    operation: u32,
}
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if (idx >= params.size) {
        return;
    }

    let val_a = a[idx];
    let val_b = b[idx];

    var result: f64;
    switch (params.operation) {
        case 0u: { result = val_a + val_b; }
        case 1u: { result = val_a - val_b; }
        case 2u: { result = val_a * val_b; }
        case 3u: { result = val_a / val_b; }
        case 4u: { result = pow_f64(val_a, val_b); }
        case 5u: { result = max(val_a, val_b); }
        case 6u: { result = min(val_a, val_b); }
        default: { result = 0.0; }
    }

    output[idx] = result;
}

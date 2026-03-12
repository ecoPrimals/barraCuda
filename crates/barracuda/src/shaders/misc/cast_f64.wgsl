// SPDX-License-Identifier: AGPL-3.0-only
// Cast_f64.wgsl — Type conversion with multiple modes (f64 canonical)
//
// Supports f64↔i32 and f64↔u32 conversions via a mode parameter.
//
// Modes:
//   0: f64 → f64 (identity / clamp)
//   1: f64 → i32 (truncate to integer, stored as f64 bit pattern)
//   2: f64 → u32 (clamp to non-negative, truncate)
//   3: i32 → f64 (interpret input bits as i32, convert to f64)
//   4: u32 → f64 (interpret input bits as u32, convert to f64)
//   5: f64 → f64 with clamp to [min_val, max_val]
//   6: f64 → bool (0.0 if input == 0, 1.0 otherwise)

struct Params {
    total: u32,
    mode: u32,       // Cast mode (see above)
    min_val: f64,    // For clamp mode
    max_val: f64,    // For clamp mode
}

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if (idx >= params.total) {
        return;
    }

    let x = input[idx];

    switch (params.mode) {
        case 0u: {
            output[idx] = x;
        }
        case 1u: {
            let i = i32(x);
            output[idx] = f64(i);
        }
        case 2u: {
            let u = u32(max(x, 0.0));
            output[idx] = f64(u);
        }
        case 3u: {
            let bits = bitcast<i32>(x);
            output[idx] = f64(bits);
        }
        case 4u: {
            let bits = bitcast<u32>(x);
            output[idx] = f64(bits);
        }
        case 5u: {
            output[idx] = clamp(x, params.min_val, params.max_val);
        }
        case 6u: {
            output[idx] = select(1.0, 0.0, x == 0.0);
        }
        default: {
            output[idx] = x;
        }
    }
}

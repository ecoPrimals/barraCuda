// SPDX-License-Identifier: AGPL-3.0-or-later
// Hardswish - MobileNetV3 activation function (f64 canonical)
// hardswish(x) = x * ReLU6(x + 3) / 6
// where ReLU6(x) = min(max(x, 0), 6)

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;
@group(0) @binding(2) var<uniform> size: u32;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if (idx >= size) {
        return;
    }

    let x = input[idx];
    let relu6_result = clamp(x + 3.0, 0.0, 6.0);
    let result = x * relu6_result / 6.0;

    output[idx] = result;
}

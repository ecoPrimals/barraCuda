// Conv2D - 2D convolution (f64 canonical)
// Simplified version: single channel, no padding, stride 1

struct Conv2DParams {
    input_width: u32,
    input_height: u32,
    kernel_width: u32,
    kernel_height: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read> kernel: array<f64>;
@group(0) @binding(2) var<storage, read_write> output: array<f64>;
@group(0) @binding(3) var<uniform> params: Conv2DParams;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let out_x = global_id.x;
    let out_y = global_id.y;
    
    let output_width = params.input_width - params.kernel_width + 1u;
    let output_height = params.input_height - params.kernel_height + 1u;
    
    if (out_x >= output_width || out_y >= output_height) {
        return;
    }
    
    var sum: f64 = 0.0;
    
    for (var ky = 0u; ky < params.kernel_height; ky = ky + 1u) {
        for (var kx = 0u; kx < params.kernel_width; kx = kx + 1u) {
            let in_x = out_x + kx;
            let in_y = out_y + ky;
            let in_idx = in_y * params.input_width + in_x;
            let k_idx = ky * params.kernel_width + kx;
            sum = sum + input[in_idx] * kernel[k_idx];
        }
    }
    
    let out_idx = out_y * output_width + out_x;
    output[out_idx] = sum;
}

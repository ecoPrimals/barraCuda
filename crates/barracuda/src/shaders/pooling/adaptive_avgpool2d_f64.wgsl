// Adaptive Average Pooling 2D
// Pools input to a specific output size regardless of input dimensions
//
// Unlike regular pooling with fixed kernel/stride, adaptive pooling
// automatically computes kernel and stride to produce desired output size.
//
// Input shape: [batch, channels, in_height, in_width]
// Output shape: [batch, channels, out_height, out_width]
//
// Used in: Classification networks (global features), SPPNet, PSPNet
// Benefits: Network can handle variable input sizes

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;
@group(0) @binding(2) var<uniform> params: Params;

struct Params {
    batch: u32,
    channels: u32,
    in_height: u32,
    in_width: u32,
    out_height: u32,
    out_width: u32,
}

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let out_x = global_id.x;
    let out_y = global_id.y;
    let batch_channel = global_id.z;
    
    if out_x >= params.out_width || out_y >= params.out_height {
        return;
    }
    
    if batch_channel >= params.batch * params.channels {
        return;
    }
    
    let b = batch_channel / params.channels;
    let c = batch_channel % params.channels;
    
    // Compute adaptive pooling window
    let start_h = (out_y * params.in_height) / params.out_height;
    let end_h = ((out_y + 1u) * params.in_height) / params.out_height;
    let start_w = (out_x * params.in_width) / params.out_width;
    let end_w = ((out_x + 1u) * params.in_width) / params.out_width;
    
    // Average over the adaptive window
    var sum: f64 = 0.0;
    var count: u32 = 0u;
    
    for (var h = start_h; h < end_h; h = h + 1u) {
        for (var w = start_w; w < end_w; w = w + 1u) {
            let input_idx = b * (params.channels * params.in_height * params.in_width) +
                           c * (params.in_height * params.in_width) +
                           h * params.in_width +
                           w;
            sum = sum + input[input_idx];
            count = count + 1u;
        }
    }
    
    let output_idx = b * (params.channels * params.out_height * params.out_width) +
                    c * (params.out_height * params.out_width) +
                    out_y * params.out_width +
                    out_x;
    
    output[output_idx] = sum / f64(count);
}

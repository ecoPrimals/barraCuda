// Conv1D (1D Convolution) - f64 canonical
// Convolution operation for sequences (time-series, NLP, audio)
//
// Input shape: [batch, in_channels, length]
// Weight shape: [out_channels, in_channels, kernel_size]
// Output shape: [batch, out_channels, out_length]

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read> weight: array<f64>;
@group(0) @binding(2) var<storage, read> bias: array<f64>;
@group(0) @binding(3) var<storage, read_write> output: array<f64>;
@group(0) @binding(4) var<uniform> params: Params;

struct Params {
    batch: u32,
    in_channels: u32,
    out_channels: u32,
    in_length: u32,
    kernel_size: u32,
    stride: u32,
    padding: u32,
    dilation: u32,
    out_length: u32,
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total_outputs = params.batch * params.out_channels * params.out_length;
    
    if idx >= total_outputs {
        return;
    }
    
    let b = idx / (params.out_channels * params.out_length);
    let oc = (idx / params.out_length) % params.out_channels;
    let ol = idx % params.out_length;
    
    var sum: f64 = 0.0;
    
    for (var ic: u32 = 0u; ic < params.in_channels; ic = ic + 1u) {
        for (var k: u32 = 0u; k < params.kernel_size; k = k + 1u) {
            let in_pos_raw = i32(ol * params.stride) + i32(k * params.dilation) - i32(params.padding);
            
            if in_pos_raw >= 0 && in_pos_raw < i32(params.in_length) {
                let in_pos = u32(in_pos_raw);
                
                let input_idx = b * (params.in_channels * params.in_length) +
                               ic * params.in_length +
                               in_pos;
                
                let weight_idx = oc * (params.in_channels * params.kernel_size) +
                                ic * params.kernel_size +
                                k;
                
                sum = sum + input[input_idx] * weight[weight_idx];
            }
        }
    }
    
    sum = sum + bias[oc];
    
    output[idx] = sum;
}

// gated_conv2d.wgsl - Gated Convolution 2D (f64 canonical)
//
// Output = tanh(W_f * x) ⊙ sigmoid(W_g * x)

struct Params {
    batch_size: u32,
    in_channels: u32,
    out_channels: u32,
    in_height: u32,
    in_width: u32,
    out_height: u32,
    out_width: u32,
    kernel_size: u32,
    stride: u32,
    padding: u32,
    _padding: vec2<u32>,
}

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read> weight_feature: array<f64>;
@group(0) @binding(2) var<storage, read> weight_gate: array<f64>;
@group(0) @binding(3) var<storage, read> bias_feature: array<f64>;
@group(0) @binding(4) var<storage, read> bias_gate: array<f64>;
@group(0) @binding(5) var<storage, read_write> output: array<f64>;
@group(0) @binding(6) var<uniform> params: Params;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let b = global_id.z % params.batch_size;
    let c_out = global_id.z / params.batch_size;
    let oh = global_id.y;
    let ow = global_id.x;
    
    if (c_out >= params.out_channels || oh >= params.out_height || ow >= params.out_width) {
        return;
    }
    
    var feature_sum: f64 = 0.0;
    var gate_sum: f64 = 0.0;
    
    for (var c_in: u32 = 0u; c_in < params.in_channels; c_in = c_in + 1u) {
        for (var kh: u32 = 0u; kh < params.kernel_size; kh = kh + 1u) {
            for (var kw: u32 = 0u; kw < params.kernel_size; kw = kw + 1u) {
                let ih_raw = i32(oh * params.stride) - i32(params.padding) + i32(kh);
                let iw_raw = i32(ow * params.stride) - i32(params.padding) + i32(kw);
                
                if (ih_raw >= 0 && ih_raw < i32(params.in_height) &&
                    iw_raw >= 0 && iw_raw < i32(params.in_width)) {
                    
                    let ih = u32(ih_raw);
                    let iw = u32(iw_raw);
                    
                    let in_idx = b * params.in_channels * params.in_height * params.in_width +
                                c_in * params.in_height * params.in_width +
                                ih * params.in_width +
                                iw;
                    
                    let w_idx = c_out * params.in_channels * params.kernel_size * params.kernel_size +
                               c_in * params.kernel_size * params.kernel_size +
                               kh * params.kernel_size +
                               kw;
                    
                    let val = input[in_idx];
                    feature_sum = feature_sum + val * weight_feature[w_idx];
                    gate_sum = gate_sum + val * weight_gate[w_idx];
                }
            }
        }
    }
    
    feature_sum = feature_sum + bias_feature[c_out];
    gate_sum = gate_sum + bias_gate[c_out];
    
    // Apply gating: tanh(feature) ⊙ sigmoid(gate)
    let feature = tanh_f64(feature_sum);
    let gate = 1.0 / (1.0 + exp_f64(-gate_sum)); // sigmoid
    
    let out_idx = b * params.out_channels * params.out_height * params.out_width +
                  c_out * params.out_height * params.out_width +
                  oh * params.out_width +
                  ow;
    
    output[out_idx] = feature * gate;
}

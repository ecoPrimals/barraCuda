// Channel Shuffle - Rearrange channels for efficient group convolutions (f64 canonical)
// Used in ShuffleNet and other efficient CNN architectures

struct Params {
    batch_size: u32,
    channels: u32,
    height: u32,
    width: u32,
    groups: u32,
    channels_per_group: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total_size = params.batch_size * params.channels * params.height * params.width;

    if (idx >= total_size) {
        return;
    }

    // Decode index: [batch, channel, height, width]
    let chw = params.channels * params.height * params.width;
    let hw = params.height * params.width;

    let batch = idx / chw;
    let remainder = idx % chw;
    let c = remainder / hw;
    let spatial = remainder % hw;

    // Channel shuffle: c = g * cpg + i → c' = i * g + g
    // where g = group index, i = index within group
    let g = c / params.channels_per_group;
    let i = c % params.channels_per_group;
    let c_shuffled = i * params.groups + g;

    // Output index with shuffled channel
    let output_idx = batch * chw + c_shuffled * hw + spatial;

    output[output_idx] = input[idx];
}

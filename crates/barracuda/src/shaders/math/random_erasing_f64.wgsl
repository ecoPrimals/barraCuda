// random_erasing_f64.wgsl - Random erasing data augmentation (f64 canonical)
//
// Randomly erases rectangular regions in images
// Reference: "Random Erasing Data Augmentation" by Zhong et al. (2017)

struct Params {
    batch_size: u32,
    channels: u32,
    height: u32,
    width: u32,
    erase_value: f64,  // Value to fill erased region (typically mean or 0)
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
    _pad4: u32,
    _pad5: u32,
    _pad6: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f64>;           // [B, C, H, W]
@group(0) @binding(1) var<storage, read> erase_boxes: array<u32>;     // [B, 4] - (top, left, height, width)
@group(0) @binding(2) var<storage, read_write> output: array<f64>;    // [B, C, H, W]
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let b = global_id.z % params.batch_size;
    let c = global_id.z / params.batch_size;
    let h = global_id.y;
    let w = global_id.x;

    if (c >= params.channels || h >= params.height || w >= params.width) {
        return;
    }

    let idx = b * params.channels * params.height * params.width +
              c * params.height * params.width +
              h * params.width +
              w;

    // Get erase box for this batch item
    let erase_top = erase_boxes[b * 4u + 0u];
    let erase_left = erase_boxes[b * 4u + 1u];
    let erase_h = erase_boxes[b * 4u + 2u];
    let erase_w = erase_boxes[b * 4u + 3u];

    // Check if pixel is in erase region
    let in_erase = (h >= erase_top && h < erase_top + erase_h &&
                    w >= erase_left && w < erase_left + erase_w);

    output[idx] = select(input[idx], params.erase_value, in_erase);
}

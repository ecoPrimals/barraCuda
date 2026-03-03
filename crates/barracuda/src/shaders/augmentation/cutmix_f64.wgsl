// cutmix.wgsl - CutMix data augmentation
//
// CutMix: Cuts and pastes patches between training images
// Improves model robustness and generalization
//
// Reference: "CutMix: Regularization Strategy to Train Strong Classifiers"
// by Yun et al. (2019)

struct Params {
    batch_size: u32,
    channels: u32,
    height: u32,
    width: u32,
    cut_x: u32,      // Top-left corner of cut region
    cut_y: u32,
    cut_w: u32,      // Cut region size
    cut_h: u32,
    mix_idx: u32,    // Index of image to mix with
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
    _pad4: u32,
    _pad5: u32,
    _pad6: u32,
    _pad7: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f64>;     // Original batch
@group(0) @binding(1) var<storage, read_write> output: array<f64>; // Mixed batch
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let b = global_id.z;
    let h = global_id.y;
    let w = global_id.x;
    
    if (b >= params.batch_size || h >= params.height || w >= params.width) {
        return;
    }
    
    for (var c: u32 = 0u; c < params.channels; c = c + 1u) {
        let idx = b * params.channels * params.height * params.width +
                 c * params.height * params.width +
                 h * params.width +
                 w;
        
        // Check if pixel is in cut region
        let in_cut = (w >= params.cut_x && w < params.cut_x + params.cut_w &&
                     h >= params.cut_y && h < params.cut_y + params.cut_h);
        
        if (in_cut) {
            // Replace with pixel from mix image
            let mix_idx_val = params.mix_idx * params.channels * params.height * params.width +
                             c * params.height * params.width +
                             h * params.width +
                             w;
            output[idx] = input[mix_idx_val];
        } else {
            // Keep original pixel
            output[idx] = input[idx];
        }
    }
}

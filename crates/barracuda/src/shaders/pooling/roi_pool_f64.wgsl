// ROI Pool - Region of Interest Pooling (f64 canonical)
// Extracts fixed-size feature maps from regions using max pooling
// Simpler than ROI Align but has quantization artifacts
//
// Algorithm:
// 1. For each ROI: (x1, y1, x2, y2)
// 2. Quantize ROI bounds to integer coordinates
// 3. Divide ROI into pooled_height x pooled_width bins
// 4. Max pool within each bin

struct Params {
    num_rois: u32,
    channels: u32,
    height: u32,
    width: u32,
    pooled_height: u32,
    pooled_width: u32,
    spatial_scale: f64,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> features: array<f64>;
@group(0) @binding(2) var<storage, read> rois: array<f64>;
@group(0) @binding(3) var<storage, read_write> output: array<f64>;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let roi_idx = global_id.z;
    let c = global_id.y;
    let bin_idx = global_id.x;

    if (roi_idx >= params.num_rois || c >= params.channels ||
        bin_idx >= params.pooled_height * params.pooled_width) {
        return;
    }

    let ph = bin_idx / params.pooled_width;
    let pw = bin_idx % params.pooled_width;

    let roi_offset = roi_idx * 4u;
    let x1 = u32(floor(rois[roi_offset] * params.spatial_scale));
    let y1 = u32(floor(rois[roi_offset + 1u] * params.spatial_scale));
    let x2 = u32(ceil(rois[roi_offset + 2u] * params.spatial_scale));
    let y2 = u32(ceil(rois[roi_offset + 3u] * params.spatial_scale));

    let roi_width = max(x2 - x1, 1u);
    let roi_height = max(y2 - y1, 1u);

    let bin_h_start = y1 + ph * roi_height / params.pooled_height;
    let bin_h_end = y1 + (ph + 1u) * roi_height / params.pooled_height;
    let bin_w_start = x1 + pw * roi_width / params.pooled_width;
    let bin_w_end = x1 + (pw + 1u) * roi_width / params.pooled_width;

    var max_val: f64 = -1e308;

    for (var h = bin_h_start; h < bin_h_end; h = h + 1u) {
        for (var w = bin_w_start; w < bin_w_end; w = w + 1u) {
            if (h < params.height && w < params.width) {
                let feat_idx = (c * params.height + h) * params.width + w;
                max_val = max(max_val, features[feat_idx]);
            }
        }
    }

    if (max_val == -1e308) {
        max_val = 0.0;
    }

    let out_idx = ((roi_idx * params.channels + c) * params.pooled_height + ph) * params.pooled_width + pw;
    output[out_idx] = max_val;
}

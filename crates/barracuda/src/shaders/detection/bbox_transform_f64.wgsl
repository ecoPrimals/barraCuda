// BBox Transform - Transform bounding boxes (f64 canonical)

struct Params {
    num_boxes: u32,
}

@group(0) @binding(0) var<storage, read> anchors: array<f64>;
@group(0) @binding(1) var<storage, read> deltas: array<f64>;
@group(0) @binding(2) var<storage, read_write> transformed: array<f64>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    
    if (i >= params.num_boxes) {
        return;
    }
    
    let idx = i * 4u;
    
    let anchor_w = anchors[idx + 2u] - anchors[idx];
    let anchor_h = anchors[idx + 3u] - anchors[idx + 1u];
    let anchor_cx = anchors[idx] + anchor_w * 0.5;
    let anchor_cy = anchors[idx + 1u] + anchor_h * 0.5;
    
    let pred_cx = deltas[idx] * anchor_w + anchor_cx;
    let pred_cy = deltas[idx + 1u] * anchor_h + anchor_cy;
    let pred_w = exp_f64(deltas[idx + 2u]) * anchor_w;
    let pred_h = exp_f64(deltas[idx + 3u]) * anchor_h;
    
    transformed[idx] = pred_cx - pred_w * 0.5;
    transformed[idx + 1u] = pred_cy - pred_h * 0.5;
    transformed[idx + 2u] = pred_cx + pred_w * 0.5;
    transformed[idx + 3u] = pred_cy + pred_h * 0.5;
}

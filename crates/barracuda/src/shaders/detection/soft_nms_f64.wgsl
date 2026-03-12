// SPDX-License-Identifier: AGPL-3.0-only
// Soft NMS - Soft Non-Maximum Suppression (f64 canonical)

struct Params {
    num_boxes: u32,
    iou_threshold: f64,
    sigma: f64,
}

@group(0) @binding(0) var<storage, read> boxes: array<f64>;
@group(0) @binding(1) var<storage, read_write> scores: array<f64>;
@group(0) @binding(2) var<storage, read_write> iou_matrix: array<f64>;
@group(0) @binding(3) var<uniform> params: Params;

fn compute_iou(box1_idx: u32, box2_idx: u32) -> f64 {
    let idx1 = box1_idx * 4u;
    let idx2 = box2_idx * 4u;
    
    let x1_1 = boxes[idx1];
    let y1_1 = boxes[idx1 + 1u];
    let x2_1 = boxes[idx1 + 2u];
    let y2_1 = boxes[idx1 + 3u];
    
    let x1_2 = boxes[idx2];
    let y1_2 = boxes[idx2 + 1u];
    let x2_2 = boxes[idx2 + 2u];
    let y2_2 = boxes[idx2 + 3u];
    
    let inter_x1 = max(x1_1, x1_2);
    let inter_y1 = max(y1_1, y1_2);
    let inter_x2 = min(x2_1, x2_2);
    let inter_y2 = min(y2_1, y2_2);
    
    let inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1);
    let box1_area = (x2_1 - x1_1) * (y2_1 - y1_1);
    let box2_area = (x2_2 - x1_2) * (y2_2 - y1_2);
    let union_area = box1_area + box2_area - inter_area;
    
    if (union_area > 0.0) {
        return inter_area / union_area;
    }
    return 0.0;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    
    if (i >= params.num_boxes) {
        return;
    }
    
    for (var j = 0u; j < params.num_boxes; j = j + 1u) {
        if (i != j) {
            let iou = compute_iou(i, j);
            iou_matrix[i * params.num_boxes + j] = iou;
            
            if (iou > params.iou_threshold) {
                let decay = exp_f64(-(iou * iou) / params.sigma);
                scores[j] = scores[j] * decay;
            }
        } else {
            iou_matrix[i * params.num_boxes + j] = 1.0;
        }
    }
}

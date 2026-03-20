// SPDX-License-Identifier: AGPL-3.0-or-later
// giou_loss_f64.wgsl - Generalized IoU Loss for object detection (f64 canonical)
//
// Improves upon IoU by considering the smallest enclosing box
// Reference: "Generalized Intersection over Union" by Rezatofighi et al. (2019)
//
// GIoU = IoU - (Area_C - Area_union) / Area_C
// GIoU Loss = 1 - GIoU

struct Params {
    num_boxes: u32,
    box_format: u32,  // 0 = xyxy, 1 = xywh, 2 = cxcywh
    _padding: vec2<u32>,
}

@group(0) @binding(0) var<storage, read> pred_boxes: array<f64>;    // [num_boxes, 4]
@group(0) @binding(1) var<storage, read> target_boxes: array<f64>;  // [num_boxes, 4]
@group(0) @binding(2) var<storage, read_write> output: array<f64>;  // [num_boxes]
@group(0) @binding(3) var<uniform> params: Params;

fn box_to_xyxy(box: vec4<f64>, format: u32) -> vec4<f64> {
    if (format == 0u) {
        return box;
    } else if (format == 1u) {
        return vec4<f64>(box.x, box.y, box.x + box.z, box.y + box.w);
    } else {
        let half_w = box.z / 2.0;
        let half_h = box.w / 2.0;
        return vec4<f64>(box.x - half_w, box.y - half_h, box.x + half_w, box.y + half_h);
    }
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;

    if (idx >= params.num_boxes) {
        return;
    }

    // Load boxes
    let pred = vec4<f64>(
        pred_boxes[idx * 4u + 0u],
        pred_boxes[idx * 4u + 1u],
        pred_boxes[idx * 4u + 2u],
        pred_boxes[idx * 4u + 3u]
    );

    let target_data = vec4<f64>(
        target_boxes[idx * 4u + 0u],
        target_boxes[idx * 4u + 1u],
        target_boxes[idx * 4u + 2u],
        target_boxes[idx * 4u + 3u]
    );

    // Convert to xyxy
    let p = box_to_xyxy(pred, params.box_format);
    let t = box_to_xyxy(target_data, params.box_format);

    // Intersection
    let x1 = max(p.x, t.x);
    let y1 = max(p.y, t.y);
    let x2 = min(p.z, t.z);
    let y2 = min(p.w, t.w);

    let inter_w = max(0.0, x2 - x1);
    let inter_h = max(0.0, y2 - y1);
    let inter_area = inter_w * inter_h;

    // Areas
    let pred_area = (p.z - p.x) * (p.w - p.y);
    let target_area = (t.z - t.x) * (t.w - t.y);
    let union_area = pred_area + target_area - inter_area;

    // IoU
    let iou = inter_area / (union_area + 1e-7);

    // Smallest enclosing box
    let c_x1 = min(p.x, t.x);
    let c_y1 = min(p.y, t.y);
    let c_x2 = max(p.z, t.z);
    let c_y2 = max(p.w, t.w);
    let c_area = (c_x2 - c_x1) * (c_y2 - c_y1);

    // GIoU
    let giou = iou - (c_area - union_area) / (c_area + 1e-7);

    // GIoU Loss = 1 - GIoU
    output[idx] = 1.0 - giou;
}

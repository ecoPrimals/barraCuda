// SPDX-License-Identifier: AGPL-3.0-or-later
// iou_f64.wgsl — Intersection over Union for bounding boxes (f64)
//
// **Math**: For axis-aligned boxes A and B:
//   Intersection area = max(0, min(A.x2,B.x2) - max(A.x1,B.x1)) * max(0, min(A.y2,B.y2) - max(A.y1,B.y1))
//   Union area = area_A + area_B - intersection
//   IoU = intersection / union  (0 if union = 0)
//
// **Algorithm**: Each thread i computes IoU between boxes_a[i] and boxes_b[i].
// Box layout: [x1, y1, x2, y2] — 4 f64 values per box (x1,y1 = min corner, x2,y2 = max corner).
//
// **Precision**: f64 via bitcast<f64>(vec2<u32>)
// **Workgroup**: @compute @workgroup_size(256)
//
// Bindings:
//   0: boxes_a   array<vec2<u32>>  read       — boxes [x1,y1,x2,y2] per box, 4 f64s each
//   1: boxes_b   array<vec2<u32>>  read       — same layout
//   2: output    array<vec2<u32>>  read_write — IoU values (one per pair)
//   3: params    uniform
//
// Params: { n: u32 }
//
// Applications: Object detection metrics, NMS, segmentation overlap, spatial similarity.

@group(0) @binding(0) var<storage, read> boxes_a: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read> boxes_b: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read_write> output: array<vec2<u32>>;
@group(0) @binding(3) var<uniform> params: Params;

struct Params {
    n: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= params.n) {
        return;
    }

    // Load box A: indices 4*i, 4*i+1, 4*i+2, 4*i+3
    let base_a = i * 4u;
    let a_x1 = bitcast<f64>(boxes_a[base_a]);
    let a_y1 = bitcast<f64>(boxes_a[base_a + 1u]);
    let a_x2 = bitcast<f64>(boxes_a[base_a + 2u]);
    let a_y2 = bitcast<f64>(boxes_a[base_a + 3u]);

    // Load box B
    let base_b = i * 4u;
    let b_x1 = bitcast<f64>(boxes_b[base_b]);
    let b_y1 = bitcast<f64>(boxes_b[base_b + 1u]);
    let b_x2 = bitcast<f64>(boxes_b[base_b + 2u]);
    let b_y2 = bitcast<f64>(boxes_b[base_b + 3u]);

    // Intersection: width = max(0, min(a_x2,b_x2) - max(a_x1,b_x1))
    let ix1 = max(a_x1, b_x1);
    let ix2 = min(a_x2, b_x2);
    let iy1 = max(a_y1, b_y1);
    let iy2 = min(a_y2, b_y2);

    let w = max(f64(0.0), ix2 - ix1);
    let h = max(f64(0.0), iy2 - iy1);
    let intersection = w * h;

    // Areas
    let area_a = (a_x2 - a_x1) * (a_y2 - a_y1);
    let area_b = (b_x2 - b_x1) * (b_y2 - b_y1);
    let union_area = area_a + area_b - intersection;

    // IoU (0 if union = 0)
    var iou = f64(0.0);
    if (union_area > f64(0.0)) {
        iou = intersection / union_area;
    }

    output[i] = bitcast<vec2<u32>>(iou);
}

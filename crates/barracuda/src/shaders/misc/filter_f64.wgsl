// SPDX-License-Identifier: AGPL-3.0-or-later
// filter_f64.wgsl — Stream compaction (predicate evaluation + scatter) (f64 canonical)
//
// Pass 1 — evaluate_predicate:
//   flags[i] = 1 if input[i] satisfies the predicate, else 0.
//
// Pass 3 — scatter:
//   output[scan[i]] = input[i] if flags[i] == 1
//
// Operations:
//   0 = GreaterThan, 1 = LessThan, 2 = Equal, 3 = NotEqual,
//   4 = GreaterOrEqual, 5 = LessOrEqual

struct FilterParams {
    size:      u32,
    operation: u32,
    n_groups:  u32,
    _pad:      u32,
    threshold: f64,
    epsilon:   f64,
    _pad2:     f64,
    _pad3:     f64,
}

@group(0) @binding(0) var<storage, read>       input:   array<f64>;
@group(0) @binding(1) var<storage, read_write> flags:   array<u32>;
@group(0) @binding(2) var<storage, read_write> scan:    array<u32>;
@group(0) @binding(3) var<storage, read_write> output:  array<f64>;
@group(0) @binding(4) var<storage, read_write> total:   array<u32>;
@group(0) @binding(5) var<uniform>             params:  FilterParams;

@compute @workgroup_size(256)
fn evaluate_predicate(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gid = global_id.x;
    if (gid >= params.size) { return; }

    let value = input[gid];
    let eps   = params.epsilon;
    let thr   = params.threshold;

    var keep: bool;
    switch (params.operation) {
        case 0u: { keep = value > thr; }
        case 1u: { keep = value < thr; }
        case 2u: { keep = abs(value - thr) < eps; }
        case 3u: { keep = abs(value - thr) >= eps; }
        case 4u: { keep = value >= thr; }
        case 5u: { keep = value <= thr; }
        default: { keep = false; }
    }

    flags[gid] = select(0u, 1u, keep);
}

@compute @workgroup_size(256)
fn scatter(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let gid = global_id.x;
    if (gid >= params.size) { return; }

    if (flags[gid] == 1u) {
        output[scan[gid]] = input[gid];
    }

    let last = params.size - 1u;
    if (gid == last) {
        total[0] = scan[last] + flags[last];
    }
}

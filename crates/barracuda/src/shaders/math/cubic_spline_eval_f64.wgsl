// SPDX-License-Identifier: AGPL-3.0-only
//
// cubic_spline_eval_f64.wgsl — Evaluate cubic spline at query points (native f64)
//
// Math: s(x) = a + b·dx + c·dx² + d·dx³ where dx = x - knots[segment]
// Algorithm: Binary search for segment, then Horner evaluation.
//
// Bindings:
//   0: query_x    array<f64> read — query x values [n_query]
//   1: knots      array<f64> read — knot x values [n_segments + 1]
//   2: coefs      array<f64> read — [a,b,c,d] per segment [4 * n_segments]
//   3: result     array<f64> read_write — interpolated y [n_query]
//   4: params     uniform

struct Params {
    n_query: u32,
    n_segments: u32,
}

@group(0) @binding(0) var<storage, read> query_x: array<f64>;
@group(0) @binding(1) var<storage, read> knots: array<f64>;
@group(0) @binding(2) var<storage, read> coefs: array<f64>;
@group(0) @binding(3) var<storage, read_write> result: array<f64>;
@group(0) @binding(4) var<uniform> params: Params;

fn find_segment(x: f64, n_segments: u32) -> u32 {
    if (n_segments == 0u) { return 0u; }
    var lo = 0u;
    var hi = n_segments;
    while (lo + 1u < hi) {
        let mid = (lo + hi) / 2u;
        if (x >= knots[mid]) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    if (x < knots[lo] && lo > 0u) {
        return lo - 1u;
    }
    return lo;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= params.n_query) { return; }

    let x = query_x[i];
    let seg = min(find_segment(x, params.n_segments), params.n_segments - 1u);
    let dx = x - knots[seg];

    let base = seg * 4u;
    let a = coefs[base];
    let b = coefs[base + 1u];
    let c = coefs[base + 2u];
    let d = coefs[base + 3u];

    // Horner: y = a + dx*(b + dx*(c + dx*d))
    result[i] = a + dx * (b + dx * (c + dx * d));
}

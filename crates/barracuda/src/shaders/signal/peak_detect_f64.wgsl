// SPDX-License-Identifier: AGPL-3.0-or-later
// Peak Detection (f64) — parallel local-maxima + prominence
//
// One thread per sample. Each thread:
//   1. Checks if its sample is a local maximum within a `distance` window
//   2. For detected peaks, computes prominence by scanning left/right for
//      the minimum valley before reaching a higher peak
//
// Width computation is deferred to the CPU orchestrator (simple linear scan
// over the small number of detected peaks).

struct PeakDetectParams {
    n: u32,
    distance: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read>       signal:     array<f64>;
@group(0) @binding(1) var<storage, read_write> is_peak:    array<u32>;
@group(0) @binding(2) var<storage, read_write> prominence: array<f64>;
@group(0) @binding(3) var<uniform>             params:     PeakDetectParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.n) {
        return;
    }

    let val = signal[idx];
    let dist = params.distance;

    // Check if local maximum within distance window (strictly greater than
    // all neighbours; equal neighbours disqualify to avoid plateau ambiguity)
    var is_max = 1u;
    let start = select(0u, idx - dist, idx >= dist);
    let end = min(idx + dist + 1u, params.n);

    for (var i = start; i < end; i = i + 1u) {
        if (i != idx && signal[i] >= val) {
            is_max = 0u;
            break;
        }
    }

    is_peak[idx] = is_max;

    if (is_max == 0u) {
        prominence[idx] = f64(0.0);
        return;
    }

    // Prominence: scan left, tracking minimum until a strictly higher peak
    var left_min: f64 = val;
    if (idx > 0u) {
        for (var i = idx - 1u; ; i = i - 1u) {
            let v = signal[i];
            left_min = min(left_min, v);
            if (v > val || i == 0u) {
                break;
            }
        }
    }

    // Scan right
    var right_min: f64 = val;
    for (var i = idx + 1u; i < params.n; i = i + 1u) {
        let v = signal[i];
        right_min = min(right_min, v);
        if (v > val) {
            break;
        }
    }

    // Prominence = peak height − higher of the two reference levels
    let reference = max(left_min, right_min);
    prominence[idx] = val - reference;
}

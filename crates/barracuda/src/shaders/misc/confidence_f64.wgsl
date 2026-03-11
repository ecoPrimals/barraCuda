// SPDX-License-Identifier: AGPL-3.0-only
//
// confidence_f64.wgsl — Per-residue confidence from logits (AlphaFold2)
//
// conf[i] = sum(softmax(logits[i,:]) * bin_centers)
// Expected value of LDDT bin under softmax distribution.
//
// Bindings: @0 logits[N*B], @1 bin_centers[B], @2 out[N], @3 uniform{n_residues, n_bins}

enable f64;

struct ConfidenceParams {
    n_residues: u32,
    n_bins: u32,
    _pad: [u32; 2],
}

@group(0) @binding(0) var<storage, read>       logits: array<f64>;     // [N*B]
@group(0) @binding(1) var<storage, read>       bin_centers: array<f64>; // [B]
@group(0) @binding(2) var<storage, read_write> out: array<f64>;       // [N]
@group(0) @binding(3) var<uniform>             params: ConfidenceParams;

fn softmax_max(row_base: u32, b: u32) -> f64 {
    var m = logits[row_base];
    for (var k = 1u; k < b; k = k + 1u) {
        m = max(m, logits[row_base + k]);
    }
    return m;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = params.n_residues;
    let B = params.n_bins;

    let i = gid.x;
    if i >= N { return; }

    let row_base = i * B;

    // Softmax with max subtraction for numerical stability
    let m = softmax_max(row_base, B);
    var sum_exp = f64(0.0);
    for (var k = 0u; k < B; k = k + 1u) {
        sum_exp += exp_f64(logits[row_base + k] - m);
    }

    var conf = f64(0.0);
    for (var k = 0u; k < B; k = k + 1u) {
        let p = exp_f64(logits[row_base + k] - m) / sum_exp;
        conf += p * bin_centers[k];
    }

    out[i] = conf;
}

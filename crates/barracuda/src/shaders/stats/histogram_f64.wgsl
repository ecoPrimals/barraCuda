// SPDX-License-Identifier: AGPL-3.0-or-later
//
// histogram.wgsl — Parallel histogram via atomic binning (f64 canonical)
//
// Each thread bins one input value. Uses atomicAdd for concurrent bin updates.
// After dispatch, normalize on CPU by dividing counts by total N.
//
// Provenance: neuralSpring baseCamp V18 handoff (Feb 2026)
// Use case: empirical spectral density, feature distributions, EDA

struct Params {
    n_values: u32,
    n_bins: u32,
    min_val: f64,
    inv_bin_width: f64,
}

@group(0) @binding(0) var<storage, read> values: array<f64>;
@group(0) @binding(1) var<storage, read_write> bins: array<atomic<u32>>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn histogram(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.n_values) {
        return;
    }

    let v = values[idx];
    var bin_idx = u32((v - params.min_val) * params.inv_bin_width);
    if (bin_idx >= params.n_bins) {
        bin_idx = params.n_bins - 1u;
    }

    atomicAdd(&bins[bin_idx], 1u);
}

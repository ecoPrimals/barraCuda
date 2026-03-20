// SPDX-License-Identifier: AGPL-3.0-or-later
// Kimura (1962) fixation probability — batch computation over parameter sets.
// Each thread computes P_fix for one (pop_size, selection, initial_freq) triplet.
// P_fix = (1 - exp(-4Ns·p0)) / (1 - exp(-4Ns))  [neutral case: p0]

struct Params {
    n_elements: u32,
}

@group(0) @binding(0) var<storage, read> pop_sizes: array<f64>;
@group(0) @binding(1) var<storage, read> selections: array<f64>;
@group(0) @binding(2) var<storage, read> freqs: array<f64>;
@group(0) @binding(3) var<storage, read_write> results: array<f64>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.n_elements) {
        return;
    }

    let n = pop_sizes[idx];
    let s = selections[idx];
    let p0 = freqs[idx];
    let four_ns = 4.0 * n * s;

    if (abs(four_ns) < 1e-10) {
        results[idx] = p0;
        return;
    }

    let num = 1.0 - exp(-four_ns * p0);
    let den = 1.0 - exp(-four_ns);

    if (abs(den) < 1e-15) {
        results[idx] = p0;
        return;
    }

    results[idx] = num / den;
}

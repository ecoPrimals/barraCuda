// SPDX-License-Identifier: AGPL-3.0-only
//
// locus_variance_f64.wgsl — Per-Locus Allele Frequency Variance (f64)
//
// Computes variance of allele frequencies across populations for each locus.
// Core building block for Weir-Cockerham FST estimation.
//
// Two-pass: mean then variance (numerically stable).
// Evolved from f32 → f64 for universal math library portability.

@group(0) @binding(0) var<storage, read> allele_freqs: array<f64>;
@group(0) @binding(1) var<storage, read_write> per_locus_var: array<f64>;

struct VarianceParams {
    n_pops: u32,
    n_loci: u32,
}
@group(0) @binding(2) var<uniform> params: VarianceParams;

@compute @workgroup_size(256)
fn locus_variance(@builtin(global_invocation_id) gid: vec3<u32>) {
    let locus = gid.x;
    if locus >= params.n_loci {
        return;
    }

    var sum: f64 = f64(0.0);
    for (var p: u32 = 0u; p < params.n_pops; p = p + 1u) {
        sum = sum + allele_freqs[p * params.n_loci + locus];
    }
    let mean = sum / f64(params.n_pops);

    var var_sum: f64 = f64(0.0);
    for (var p: u32 = 0u; p < params.n_pops; p = p + 1u) {
        let diff = allele_freqs[p * params.n_loci + locus] - mean;
        var_sum = var_sum + diff * diff;
    }

    per_locus_var[locus] = var_sum / f64(params.n_pops);
}

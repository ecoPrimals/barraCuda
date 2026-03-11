// SPDX-License-Identifier: AGPL-3.0-only
//
// multi_obj_fitness_f64.wgsl — Multi-objective fitness evaluation (f64)
//
// For each individual, splits genome into n_obj segments and computes:
//   fitness[i * n_obj + o] = mean(segment_o) + 0.1 * std(segment_o)
//
// Evolved from f32 → f64 for universal math library portability.

struct FitnessParams {
    pop:        u32,
    genome_len: u32,
    n_obj:      u32,
    _pad:       u32,
}

@group(0) @binding(0) var<storage, read>       genotypes: array<f64>;
@group(0) @binding(1) var<storage, read_write> fitness:   array<f64>;
@group(0) @binding(2) var<uniform>             params:    FitnessParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total = params.pop * params.n_obj;
    if (idx >= total) { return; }

    let ind = idx / params.n_obj;
    let obj = idx % params.n_obj;
    let seg_len = params.genome_len / params.n_obj;

    let gene_base = ind * params.genome_len + obj * seg_len;

    var sum: f64 = f64(0.0);
    for (var g = 0u; g < seg_len; g = g + 1u) {
        sum = sum + genotypes[gene_base + g];
    }
    let mean_val = sum / f64(seg_len);

    var var_sum: f64 = f64(0.0);
    for (var g = 0u; g < seg_len; g = g + 1u) {
        let d = genotypes[gene_base + g] - mean_val;
        var_sum = var_sum + d * d;
    }
    let std_val = sqrt(var_sum / f64(max(seg_len - 1u, 1u)));

    fitness[idx] = mean_val + f64(0.1) * std_val;
}

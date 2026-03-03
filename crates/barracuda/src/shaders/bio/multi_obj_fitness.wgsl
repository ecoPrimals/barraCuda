// multi_obj_fitness.wgsl — Multi-objective fitness evaluation (f32)
//
// neuralSpring absorption: evolutionary optimization with multiple objectives.
//
// For each individual in a population, splits the genome into n_obj equal
// segments and computes a fitness score per objective:
//   fitness[i * n_obj + o] = mean(segment_o) + 0.1 * std(segment_o)
//
// This rewards both high mean performance and consistency (low variance
// penalty via the positive std contribution).
//
// Bindings:
//   0: genotypes [pop × genome_len] f32 — flat population genomes
//   1: fitness   [pop × n_obj] f32 — per-individual, per-objective scores
//   2: params    uniform

struct FitnessParams {
    pop:        u32,
    genome_len: u32,
    n_obj:      u32,
    _pad:       u32,
}

@group(0) @binding(0) var<storage, read>       genotypes: array<f32>;
@group(0) @binding(1) var<storage, read_write> fitness:   array<f32>;
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

    // Online mean
    var sum: f32 = 0.0;
    for (var g = 0u; g < seg_len; g = g + 1u) {
        sum = sum + genotypes[gene_base + g];
    }
    let mean_val = sum / f32(seg_len);

    // Online variance (two-pass for numerical stability on GPU)
    var var_sum: f32 = 0.0;
    for (var g = 0u; g < seg_len; g = g + 1u) {
        let d = genotypes[gene_base + g] - mean_val;
        var_sum = var_sum + d * d;
    }
    let std_val = sqrt(var_sum / f32(max(seg_len - 1u, 1u)));

    fitness[idx] = mean_val + 0.1 * std_val;
}

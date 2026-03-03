// batch_fitness_eval.wgsl — Batch Fitness Evaluation (f32)
//
// Evaluates fitness for an entire EA population in a single GPU dispatch.
// Each individual's genotype is a row in the population matrix; fitness
// is computed as a dot product with a trait-weight vector (linear fitness).
//
// GPU dispatch: ceil(pop_size / 256) workgroups, 256 threads each.
//
// Provenance: neuralSpring metalForge (Feb 21, 2026) → ToadStool absorption
// Reference: Dolson et al. (2020) Nature Physics, (2022) eLife

@group(0) @binding(0) var<storage, read> population: array<f32>;  // [pop_size × genome_len]
@group(0) @binding(1) var<storage, read> weights: array<f32>;     // [genome_len]
@group(0) @binding(2) var<storage, read_write> fitness: array<f32>; // [pop_size]

struct FitnessParams {
    pop_size: u32,
    genome_len: u32,
}
@group(0) @binding(3) var<uniform> params: FitnessParams;

@compute @workgroup_size(256)
fn batch_fitness_linear(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.pop_size {
        return;
    }

    let base = i * params.genome_len;
    var acc: f32 = 0.0;
    for (var g: u32 = 0u; g < params.genome_len; g = g + 1u) {
        acc = fma(population[base + g], weights[g], acc);
    }
    fitness[i] = acc;
}

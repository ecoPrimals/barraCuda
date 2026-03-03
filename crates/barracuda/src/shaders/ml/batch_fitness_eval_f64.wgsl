// SPDX-License-Identifier: AGPL-3.0-or-later
//
// batch_fitness_eval_f64.wgsl — Batch Fitness Evaluation (f64)
//
// Evaluates fitness for an entire EA population: dot product of genotype
// with trait-weight vector (linear fitness).
//
// Evolved from f32 → f64 for universal math library portability.

@group(0) @binding(0) var<storage, read> population: array<f64>;
@group(0) @binding(1) var<storage, read> weights: array<f64>;
@group(0) @binding(2) var<storage, read_write> fitness: array<f64>;

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
    var acc: f64 = f64(0.0);
    for (var g: u32 = 0u; g < params.genome_len; g = g + 1u) {
        acc = acc + population[base + g] * weights[g];
    }
    fitness[i] = acc;
}

// mean_reduce.wgsl — Single-workgroup Mean Reduction (f32)
//
// Computes the arithmetic mean of an f32 array in a single-workgroup pass.
// For large arrays, use BarraCuda's ReduceScalarPipeline with multi-workgroup
// tree reduction.
//
// Provenance: neuralSpring metalForge (Feb 21, 2026) → ToadStool absorption

@group(0) @binding(0) var<storage, read> values: array<f32>;
@group(0) @binding(1) var<storage, read_write> result: array<f32>;

struct ReduceParams {
    n: u32,
}
@group(0) @binding(2) var<uniform> params: ReduceParams;

@compute @workgroup_size(1)
fn mean_reduce(@builtin(global_invocation_id) gid: vec3<u32>) {
    var sum: f32 = 0.0;
    for (var i: u32 = 0u; i < params.n; i = i + 1u) {
        sum = sum + values[i];
    }
    result[0] = sum / f32(params.n);
}

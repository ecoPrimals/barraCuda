// Fused KL Divergence (f64) — neuralSpring V24
//
// D_KL(P||Q) = Σ p_i * log(p_i / q_i)
// Computed entirely on GPU with numerical stability: clamp to epsilon to avoid log(0).
//
// Input: P and Q probability distributions (same length)
// Output: partial sums per workgroup, then reduced to scalar
//
// f64 enabled by compile_shader_f64() preamble injection

struct Params {
    n: u32,
    _pad1: u32,
    _pad2: u32,
    _pad3: u32,
}

const EPS: f64 = 1e-15;

@group(0) @binding(0) var<storage, read> p: array<f64>;
@group(0) @binding(1) var<storage, read> q: array<f64>;
@group(0) @binding(2) var<storage, read_write> output: array<f64>;
@group(0) @binding(3) var<uniform> params: Params;

var<workgroup> shared_data: array<f64, 256>;

// KL partial: output[wg_id] = Σ p_i * log(p_i/q_i) for this workgroup's range
@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
) {
    let tid = local_id.x;
    let gid = global_id.x;

    if (gid < params.n) {
        let pi = max(p[gid], EPS);
        let qi = max(q[gid], EPS);
        let term = pi * log(pi / qi);
        shared_data[tid] = term;
    } else {
        shared_data[tid] = f64(0.0);
    }
    workgroupBarrier();

    for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
        if (tid < stride) {
            shared_data[tid] = shared_data[tid] + shared_data[tid + stride];
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        output[workgroup_id.x] = shared_data[0];
    }
}

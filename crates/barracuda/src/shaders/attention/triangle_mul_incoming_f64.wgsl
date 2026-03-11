// SPDX-License-Identifier: AGPL-3.0-only
//
// triangle_mul_incoming_f64.wgsl — AlphaFold2 Evoformer triangle multiplicative update (incoming edges)
//
// Computes: z_ij = sigmoid(gate_ij) * sum_k(a_ki * b_kj)
// Incoming edges: for each (i,j), aggregate over k where a is indexed by (k,i) and b by (k,j).
//
// Bindings: @0 a[N*N], @1 b[N*N], @2 gate[N*N], @3 out[N*N], @4 uniform{n: u32}

enable f64;

struct TriangleMulParams {
    n: u32,
}

fn sigmoid_f64(x: f64) -> f64 {
    return 1.0 / (1.0 + exp_f64(-x));
}

@group(0) @binding(0) var<storage, read>       a: array<f64>;     // [N*N]
@group(0) @binding(1) var<storage, read>       b: array<f64>;     // [N*N]
@group(0) @binding(2) var<storage, read>       gate: array<f64>;  // [N*N]
@group(0) @binding(3) var<storage, read_write> out: array<f64>;   // [N*N]
@group(0) @binding(4) var<uniform>             params: TriangleMulParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = params.n;
    let idx = gid.x;
    if idx >= N * N { return; }

    let i = idx / N;
    let j = idx % N;

    var sum_val = f64(0.0);
    for (var k = 0u; k < N; k = k + 1u) {
        sum_val += a[k * N + i] * b[k * N + j];
    }

    let g = sigmoid_f64(gate[idx]);
    out[idx] = g * sum_val;
}

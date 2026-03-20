// SPDX-License-Identifier: AGPL-3.0-or-later
//
// pair_transition_f64.wgsl — AlphaFold2 pair representation transition layer
//
// 2-layer MLP on pair features: out[i,j,c] = ReLU(pair[i,j,:] * W1 + b1) * W2 + b2
// Pair representation update in the Evoformer stack.
//
// Bindings: @0 pair[N*N*C_in], @1 weights1[C_in*C_hidden], @2 weights2[C_hidden*C_out],
//          @3 bias1[C_hidden], @4 bias2[C_out], @5 out[N*N*C_out], @6 uniform{n, c_in, c_hidden, c_out}

enable f64;

struct PairTransitionParams {
    n: u32,
    c_in: u32,
    c_hidden: u32,
    c_out: u32,
}

fn relu_f64(x: f64) -> f64 {
    return max(x, 0.0);
}

@group(0) @binding(0) var<storage, read>       pair: array<f64>;      // [N*N*C_in]
@group(0) @binding(1) var<storage, read>       weights1: array<f64>;  // [C_in*C_hidden]
@group(0) @binding(2) var<storage, read>       weights2: array<f64>;  // [C_hidden*C_out]
@group(0) @binding(3) var<storage, read>       bias1: array<f64>;     // [C_hidden]
@group(0) @binding(4) var<storage, read>       bias2: array<f64>;     // [C_out]
@group(0) @binding(5) var<storage, read_write>  out: array<f64>;       // [N*N*C_out]
@group(0) @binding(6) var<uniform>              params: PairTransitionParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = params.n;
    let C_in = params.c_in;
    let C_hidden = params.c_hidden;
    let C_out = params.c_out;

    let idx = gid.x;
    if idx >= N * N * C_out { return; }

    let c_out = idx % C_out;
    let ij = idx / C_out;
    let j = ij % N;
    let i = ij / N;

    let pair_base = (i * N + j) * C_in;

    // Fused: out = b2 + sum_h ReLU(b1[h] + sum_c pair*W1[c,h]) * W2[h,c_out]
    var sum_val = bias2[c_out];
    for (var h = 0u; h < C_hidden; h = h + 1u) {
        var hidden_val = bias1[h];
        for (var c = 0u; c < C_in; c = c + 1u) {
            hidden_val += pair[pair_base + c] * weights1[c * C_hidden + h];
        }
        sum_val += relu_f64(hidden_val) * weights2[h * C_out + c_out];
    }
    out[idx] = sum_val;
}

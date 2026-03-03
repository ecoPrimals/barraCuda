// Bootstrap resampling for mean estimation
//
// Each thread computes one bootstrap sample: draw n values with replacement from
// the data array using xorshift32 PRNG, then output the sample mean.
// Embarrassingly parallel — each of n_bootstrap samples is independent.
//
// Input: data array of n f64 values (as vec2<u32>)
// Output: array of n_bootstrap bootstrap means (one per sample)
//
// Params: n (data size), n_bootstrap (number of bootstrap samples), seed (PRNG seed)
//
// Applications: Confidence intervals for the mean, standard error estimation,
// hypothesis testing, non-parametric inference when distribution is unknown.
// Reference: Efron & Tibshirani "An Introduction to the Bootstrap"
//
// Note: Requires GPU f64 support. Uses xorshift32 for reproducibility.

@group(0) @binding(0) var<storage, read> input: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read_write> output: array<vec2<u32>>;
@group(0) @binding(2) var<uniform> params: Params;

struct Params {
    n: u32,
    n_bootstrap: u32,
    seed: u32,
}

fn xorshift32(state: u32) -> u32 {
    var s = state;
    s = s ^ (s << 13u);
    s = s ^ (s >> 17u);
    s = s ^ (s << 5u);
    return s;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.n_bootstrap) {
        return;
    }

    let n = params.n;
    if (n == 0u) {
        output[idx] = bitcast<vec2<u32>>(f64(0.0) / f64(0.0));
        return;
    }

    var state = params.seed + idx;
    if (state == 0u) {
        state = 1u;
    }

    var sum: f64 = f64(0.0);
    for (var i: u32 = 0u; i < n; i = i + 1u) {
        state = xorshift32(state);
        let j = state % n;
        let val = bitcast<f64>(input[j]);
        sum = sum + val;
    }

    let mean = sum / f64(n);
    output[idx] = bitcast<vec2<u32>>(mean);
}

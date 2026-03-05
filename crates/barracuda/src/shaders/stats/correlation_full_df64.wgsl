// SPDX-License-Identifier: AGPL-3.0-or-later
//
// correlation_full_df64.wgsl — Fused 5-accumulator Pearson correlation, DF64
//
// Same algorithm as correlation_full_f64.wgsl, but all accumulation runs
// on the FP32 core array via DF64 (f32-pair, ~48-bit mantissa).
// f64 is used only for buffer I/O; the O(N) accumulation and O(log N)
// tree reduction run entirely in DF64.
//
// Output layout: [mean_x, mean_y, var_x, var_y, pearson_r] (5 f64 values)
// Buffer layout: UNCHANGED from correlation_full_f64.wgsl (array<f64>).
// Prepend: df64_core.wgsl (auto-injected or manual)

struct Params {
    n: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> x: array<f64>;
@group(0) @binding(1) var<storage, read> y: array<f64>;
@group(0) @binding(2) var<storage, read_write> output: array<f64>;
@group(0) @binding(3) var<uniform> params: Params;

// 5 shared-memory arrays — each split into hi/lo for DF64
var<workgroup> s_sx_hi: array<f32, 256>;
var<workgroup> s_sx_lo: array<f32, 256>;
var<workgroup> s_sy_hi: array<f32, 256>;
var<workgroup> s_sy_lo: array<f32, 256>;
var<workgroup> s_sxx_hi: array<f32, 256>;
var<workgroup> s_sxx_lo: array<f32, 256>;
var<workgroup> s_syy_hi: array<f32, 256>;
var<workgroup> s_syy_lo: array<f32, 256>;
var<workgroup> s_sxy_hi: array<f32, 256>;
var<workgroup> s_sxy_lo: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(num_workgroups) nwg: vec3<u32>,
) {
    let tid = lid.x;
    let n = params.n;
    let total_threads = nwg.x * 256u;

    // Grid-stride accumulation in DF64
    var sx = df64_zero();
    var sy = df64_zero();
    var sxx = df64_zero();
    var syy = df64_zero();
    var sxy = df64_zero();

    var idx = gid.x;
    while idx < n {
        let vx = df64_from_f64(x[idx]);
        let vy = df64_from_f64(y[idx]);
        sx = df64_add(sx, vx);
        sy = df64_add(sy, vy);
        sxx = df64_add(sxx, df64_mul(vx, vx));
        syy = df64_add(syy, df64_mul(vy, vy));
        sxy = df64_add(sxy, df64_mul(vx, vy));
        idx += total_threads;
    }

    s_sx_hi[tid] = sx.hi;   s_sx_lo[tid] = sx.lo;
    s_sy_hi[tid] = sy.hi;   s_sy_lo[tid] = sy.lo;
    s_sxx_hi[tid] = sxx.hi; s_sxx_lo[tid] = sxx.lo;
    s_syy_hi[tid] = syy.hi; s_syy_lo[tid] = syy.lo;
    s_sxy_hi[tid] = sxy.hi; s_sxy_lo[tid] = sxy.lo;
    workgroupBarrier();

    // Tree reduction in DF64
    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if tid < stride {
            let a_sx = Df64(s_sx_hi[tid], s_sx_lo[tid]);
            let b_sx = Df64(s_sx_hi[tid + stride], s_sx_lo[tid + stride]);
            let r_sx = df64_add(a_sx, b_sx);
            s_sx_hi[tid] = r_sx.hi; s_sx_lo[tid] = r_sx.lo;

            let a_sy = Df64(s_sy_hi[tid], s_sy_lo[tid]);
            let b_sy = Df64(s_sy_hi[tid + stride], s_sy_lo[tid + stride]);
            let r_sy = df64_add(a_sy, b_sy);
            s_sy_hi[tid] = r_sy.hi; s_sy_lo[tid] = r_sy.lo;

            let a_sxx = Df64(s_sxx_hi[tid], s_sxx_lo[tid]);
            let b_sxx = Df64(s_sxx_hi[tid + stride], s_sxx_lo[tid + stride]);
            let r_sxx = df64_add(a_sxx, b_sxx);
            s_sxx_hi[tid] = r_sxx.hi; s_sxx_lo[tid] = r_sxx.lo;

            let a_syy = Df64(s_syy_hi[tid], s_syy_lo[tid]);
            let b_syy = Df64(s_syy_hi[tid + stride], s_syy_lo[tid + stride]);
            let r_syy = df64_add(a_syy, b_syy);
            s_syy_hi[tid] = r_syy.hi; s_syy_lo[tid] = r_syy.lo;

            let a_sxy = Df64(s_sxy_hi[tid], s_sxy_lo[tid]);
            let b_sxy = Df64(s_sxy_hi[tid + stride], s_sxy_lo[tid + stride]);
            let r_sxy = df64_add(a_sxy, b_sxy);
            s_sxy_hi[tid] = r_sxy.hi; s_sxy_lo[tid] = r_sxy.lo;
        }
        workgroupBarrier();
    }

    if tid == 0u {
        if nwg.x == 1u {
            let nf = df64_from_f32(f32(n));
            let total_sx = Df64(s_sx_hi[0], s_sx_lo[0]);
            let total_sy = Df64(s_sy_hi[0], s_sy_lo[0]);
            let total_sxx = Df64(s_sxx_hi[0], s_sxx_lo[0]);
            let total_syy = Df64(s_syy_hi[0], s_syy_lo[0]);
            let total_sxy = Df64(s_sxy_hi[0], s_sxy_lo[0]);

            let mean_x = df64_div(total_sx, nf);
            let mean_y = df64_div(total_sy, nf);
            let var_x = df64_sub(df64_div(total_sxx, nf), df64_mul(mean_x, mean_x));
            let var_y = df64_sub(df64_div(total_syy, nf), df64_mul(mean_y, mean_y));
            let cov_xy = df64_sub(df64_div(total_sxy, nf), df64_mul(mean_x, mean_y));

            // sqrt via f64 for the denominator (precision-critical)
            let denom_f64 = sqrt(df64_to_f64(var_x) * df64_to_f64(var_y));
            var r_val: f64 = 0.0;
            if denom_f64 >= 1.0e-15 {
                r_val = df64_to_f64(cov_xy) / denom_f64;
            }

            output[0] = df64_to_f64(mean_x);
            output[1] = df64_to_f64(mean_y);
            output[2] = df64_to_f64(var_x);
            output[3] = df64_to_f64(var_y);
            output[4] = r_val;
        } else {
            let base = gid.x / 256u * 5u;
            output[base] = df64_to_f64(Df64(s_sx_hi[0], s_sx_lo[0]));
            output[base + 1u] = df64_to_f64(Df64(s_sy_hi[0], s_sy_lo[0]));
            output[base + 2u] = df64_to_f64(Df64(s_sxx_hi[0], s_sxx_lo[0]));
            output[base + 3u] = df64_to_f64(Df64(s_syy_hi[0], s_syy_lo[0]));
            output[base + 4u] = df64_to_f64(Df64(s_sxy_hi[0], s_sxy_lo[0]));
        }
    }
}

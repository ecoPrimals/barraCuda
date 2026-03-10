// SPDX-License-Identifier: AGPL-3.0-only
//
// beat_classify_batch_f64.wgsl — Batch beat template-matching classification
//
// Each thread classifies one beat window against N templates using
// normalized cross-correlation. Output: best template index (as f64)
// and correlation value per beat.
//
// Absorbed from healthSpring V19 (Exp082, Exp085).
// Dispatch: (ceil(n_beats / 256), 1, 1)

enable f64;

struct Params {
    n_beats: u32,
    n_templates: u32,
    window_size: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> beats: array<f64>;
@group(0) @binding(1) var<storage, read> templates: array<f64>;
@group(0) @binding(2) var<storage, read_write> output: array<f64>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let beat_idx = gid.x;
    if beat_idx >= params.n_beats { return; }

    let ws = params.window_size;
    let beat_offset = beat_idx * ws;

    var beat_sum = 0.0;
    for (var i = 0u; i < ws; i = i + 1u) {
        beat_sum = beat_sum + beats[beat_offset + i];
    }
    let beat_mean = beat_sum / f64(ws);

    var beat_var = 0.0;
    for (var i = 0u; i < ws; i = i + 1u) {
        let d = beats[beat_offset + i] - beat_mean;
        beat_var = beat_var + d * d;
    }

    var best_corr = -1.0;
    var best_tmpl = 0.0;

    for (var t = 0u; t < params.n_templates; t = t + 1u) {
        let tmpl_offset = t * ws;

        var tmpl_sum = 0.0;
        for (var i = 0u; i < ws; i = i + 1u) {
            tmpl_sum = tmpl_sum + templates[tmpl_offset + i];
        }
        let tmpl_mean = tmpl_sum / f64(ws);

        var cov = 0.0;
        var tmpl_var = 0.0;
        for (var i = 0u; i < ws; i = i + 1u) {
            let db = beats[beat_offset + i] - beat_mean;
            let dt_val = templates[tmpl_offset + i] - tmpl_mean;
            cov = cov + db * dt_val;
            tmpl_var = tmpl_var + dt_val * dt_val;
        }

        let denom_sq = beat_var * tmpl_var;
        if denom_sq > 1e-28 {
            let denom = f64(sqrt(f32(denom_sq)));
            let corr = cov / denom;
            if corr > best_corr {
                best_corr = corr;
                best_tmpl = f64(t);
            }
        }
    }

    let out_base = beat_idx * 2u;
    output[out_base]      = best_tmpl;
    output[out_base + 1u] = best_corr;
}

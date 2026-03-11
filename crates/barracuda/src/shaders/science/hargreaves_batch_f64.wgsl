// SPDX-License-Identifier: AGPL-3.0-only
//
// Hargreaves & Samani (1985) batch ET0 — GPU parallel over days.
// ET0 = 0.0023 * Ra * (t_mean + 17.8) * sqrt(t_max - t_min)
// Each thread processes one day.

enable f64;

struct Params {
    n_days: u32,
}

@group(0) @binding(0) var<storage, read> ra: array<f64>;
@group(0) @binding(1) var<storage, read> t_max: array<f64>;
@group(0) @binding(2) var<storage, read> t_min: array<f64>;
@group(0) @binding(3) var<storage, read_write> et0: array<f64>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.n_days) {
        return;
    }

    let r = ra[idx];
    let tx = t_max[idx];
    let tn = t_min[idx];
    let delta = tx - tn;

    if (delta < 0.0 || r < 0.0) {
        // Signal invalid with NaN
        et0[idx] = f64(0.0) / f64(0.0);
        return;
    }

    let t_mean = (tx + tn) * 0.5;
    et0[idx] = 0.0023 * r * (t_mean + 17.8) * sqrt(delta);
}

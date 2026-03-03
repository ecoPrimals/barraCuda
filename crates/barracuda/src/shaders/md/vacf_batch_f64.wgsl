// SPDX-License-Identifier: AGPL-3.0-or-later
//
// vacf_batch_f64.wgsl — Velocity Autocorrelation Function (f64)
//
// Computes C(τ) = (1/n_origins) Σₜ v(t)·v(t+τ) averaged over time origins.
// For each lag τ, sums over all valid origins t (t+τ < n_frames) the
// dot product of system velocity vectors, then normalizes.
//
// Layout: vel[frame_idx * stride + particle_idx * 3 + dim]
//        where stride = n_atoms * 3
//
// Each thread handles one lag value τ. Output C(τ) for τ = 0..n_lags-1.
//
// Reference: Allen & Tildesley (1987), Frenkel & Smit (2002)

struct VacfParams {
    n_atoms: u32,
    n_frames: u32,
    n_lags: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> vel: array<f64>;
@group(0) @binding(1) var<storage, read_write> c_out: array<f64>;
@group(0) @binding(2) var<uniform> params: VacfParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let lag = gid.x;
    if lag >= params.n_lags { return; }

    let n = params.n_atoms;
    let n_frames = params.n_frames;
    let stride = n * 3u;

    let n_origins = n_frames - lag;
    if n_origins == 0u {
        c_out[lag] = f64(0.0);
        return;
    }

    var sum: f64 = f64(0.0);
    for (var t: u32 = 0u; t < n_origins; t = t + 1u) {
        let t1 = t + lag;
        var dot: f64 = f64(0.0);
        for (var i: u32 = 0u; i < n; i = i + 1u) {
            let v0_x = vel[t * stride + i * 3u];
            let v0_y = vel[t * stride + i * 3u + 1u];
            let v0_z = vel[t * stride + i * 3u + 2u];
            let v1_x = vel[t1 * stride + i * 3u];
            let v1_y = vel[t1 * stride + i * 3u + 1u];
            let v1_z = vel[t1 * stride + i * 3u + 2u];
            dot = dot + v0_x * v1_x + v0_y * v1_y + v0_z * v1_z;
        }
        sum = sum + dot;
    }

    c_out[lag] = sum / f64(n_origins);
}

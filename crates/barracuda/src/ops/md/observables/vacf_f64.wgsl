// SPDX-License-Identifier: AGPL-3.0-or-later
// vacf_f64.wgsl — Velocity Autocorrelation Function
//
// Two entry points:
//
// 1. vacf_pair — (t0, lag) grid: c_raw[t0*max_lag+lag] = (1/N) Σ_i v(t0,i)·v(t0+lag,i)
//    Used by vacf_gpu.rs. Bindings: params[4], velocities[T×N×3] f64, c_raw[T×L] f64.
//    Dispatch: ceil(max_lag/16) × ceil(n_frames/16).
//
// 2. main — Per-particle dot products: output[i] = v0[3i]*vt[3i] + v0[3i+1]*vt[3i+1] + v0[3i+2]*vt[3i+2]
//    Uses vec2<u32> bitcast for f64. Bindings: velocities_t0, velocities_t, output, params{n}.
//    Workgroup_size(256). Host averages over time origins.
//
// Reference: Allen & Tildesley "Computer Simulation of Liquids"

// ─── vacf_pair: (t0, lag) grid, array<f64> (for vacf_gpu integration) ─────────
@group(0) @binding(0) var<storage, read> params: array<u32>;
@group(0) @binding(1) var<storage, read> velocities: array<f64>;
@group(0) @binding(2) var<storage, read_write> c_raw: array<f64>;

@compute @workgroup_size(16, 16)
fn vacf_pair(@builtin(global_invocation_id) gid: vec3<u32>) {
    let lag = gid.x;
    let t0 = gid.y;

    let n_particles = params[0];
    let n_frames = params[1];
    let max_lag = params[2];

    if (lag >= max_lag || t0 >= n_frames) { return; }
    let t1 = t0 + lag;
    if (t1 >= n_frames) { return; }

    var dot: f64 = 0.0;
    let base0 = t0 * n_particles * 3u;
    let base1 = t1 * n_particles * 3u;

    for (var i = 0u; i < n_particles; i = i + 1u) {
        let o0 = base0 + i * 3u;
        let o1 = base1 + i * 3u;
        dot = dot
            + velocities[o0] * velocities[o1]
            + velocities[o0 + 1u] * velocities[o1 + 1u]
            + velocities[o0 + 2u] * velocities[o1 + 2u];
    }

    c_raw[t0 * max_lag + lag] = dot / f64(n_particles);
}

// ─── main: Per-particle dot products (vec2<u32> bitcast pattern) ─────────────
@group(0) @binding(0) var<storage, read> velocities_t0: array<vec2<u32>>;
@group(0) @binding(1) var<storage, read> velocities_t: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read_write> output: array<vec2<u32>>;
@group(0) @binding(3) var<uniform> main_params: MainParams;

struct MainParams {
    n: u32,
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    let n = main_params.n;
    if (i >= n) { return; }

    let base = i * 3u;
    let v0x = bitcast<f64>(velocities_t0[base]);
    let v0y = bitcast<f64>(velocities_t0[base + 1u]);
    let v0z = bitcast<f64>(velocities_t0[base + 2u]);
    let vtx = bitcast<f64>(velocities_t[base]);
    let vty = bitcast<f64>(velocities_t[base + 1u]);
    let vtz = bitcast<f64>(velocities_t[base + 2u]);

    let dot_val = v0x * vtx + v0y * vty + v0z * vtz;
    output[i] = bitcast<vec2<u32>>(dot_val);
}

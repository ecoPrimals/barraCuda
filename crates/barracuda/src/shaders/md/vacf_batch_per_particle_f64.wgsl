// SPDX-License-Identifier: AGPL-3.0-or-later
//
// vacf_batch_per_particle_f64.wgsl — Per-particle VACF for one lag (transport_gpu)
//
// Each thread computes one particle's contribution to C(lag):
//   c_i(lag) = (1/n_origins) Σₜ v_i(t)·v_i(t+τ)
//
// Used by VacfBatchGpu for Green-Kubo diffusion (output reduced via ReduceScalarPipeline).

struct VacfBatchParams {
    n: u32,
    n_frames: u32,
    lag: u32,
    stride: u32,
}

@group(0) @binding(0) var<storage, read> vel: array<f64>;
@group(0) @binding(1) var<storage, read_write> c_out: array<f64>;
@group(0) @binding(2) var<uniform> params: VacfBatchParams;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.n { return; }

    let lag = params.lag;
    let n_frames = params.n_frames;
    let stride = params.stride;

    let n_origins = n_frames - lag;
    if n_origins == 0u {
        c_out[i] = f64(0.0);
        return;
    }

    var sum: f64 = f64(0.0);
    for (var t: u32 = 0u; t < n_origins; t = t + 1u) {
        let t1 = t + lag;
        let v0_x = vel[t * stride + i * 3u];
        let v0_y = vel[t * stride + i * 3u + 1u];
        let v0_z = vel[t * stride + i * 3u + 2u];
        let v1_x = vel[t1 * stride + i * 3u];
        let v1_y = vel[t1 * stride + i * 3u + 1u];
        let v1_z = vel[t1 * stride + i * 3u + 2u];
        sum = sum + v0_x * v1_x + v0_y * v1_y + v0_z * v1_z;
    }

    c_out[i] = sum / f64(n_origins);
}

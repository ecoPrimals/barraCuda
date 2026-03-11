// SPDX-License-Identifier: AGPL-3.0-only
//
// msd_f64.wgsl — Mean-Squared Displacement kernel (f64)
//
// For a given lag τ, each thread handles one (t0, particle_i) pair:
//   displacement² = |r(t0+τ) - r(t0)|²
//
// Output is a partial-sum buffer; host reduces to MSD(τ) = sum / count.
//
// Positions must be PBC-unwrapped before calling this kernel.
// Flattened layout: positions[frame * n * 3 + particle * 3 + xyz]
//
// Reference: Allen & Tildesley "Computer Simulation of Liquids"

struct Params {
    n_particles: u32,
    n_frames: u32,
    lag: u32,
    _pad0: u32,
}

@group(0) @binding(0) var<storage, read> positions: array<f64>;
@group(0) @binding(1) var<storage, read_write> output: array<f64>;
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = params.n_particles;
    let n_origins = params.n_frames - params.lag;
    let total = n_origins * n;
    let tid = gid.x;

    if (tid >= total) { return; }

    let t0 = tid / n;
    let i = tid % n;
    let t1 = t0 + params.lag;

    let stride = n * 3u;
    let base0 = t0 * stride + i * 3u;
    let base1 = t1 * stride + i * 3u;

    let dx = positions[base1]      - positions[base0];
    let dy = positions[base1 + 1u] - positions[base0 + 1u];
    let dz = positions[base1 + 2u] - positions[base0 + 2u];

    output[tid] = dx * dx + dy * dy + dz * dz;
}

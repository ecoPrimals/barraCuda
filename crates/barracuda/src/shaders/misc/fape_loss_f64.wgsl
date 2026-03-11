// SPDX-License-Identifier: AGPL-3.0-only
//
// fape_loss_f64.wgsl — Frame Aligned Point Error loss (AlphaFold2)
//
// FAPE = (1/N) Σ_i ||T_i^{-1}(x_pred) - T_i^{-1}(x_true)||
// Per-residue FAPE: compare predicted vs true atom positions in local frames.
// Frames stored as 4x4 row-major: [R|t; 0 0 0 1]. T^{-1}(x) = R^T * (x - t).
//
// Bindings: @0 pred_pos[N*M*3], @1 true_pos[N*M*3], @2 pred_frames[N*12], @3 true_frames[N*12],
//          @4 out[N], @5 uniform{n_residues, n_atoms, d_clamp}

enable f64;

struct FapeParams {
    n_residues: u32,
    n_atoms: u32,
    _pad: [u32; 2],
    d_clamp: f64,
}

@group(0) @binding(0) var<storage, read>       pred_pos: array<f64>;   // [N*M*3]
@group(0) @binding(1) var<storage, read>       true_pos: array<f64>;  // [N*M*3]
@group(0) @binding(2) var<storage, read>       pred_frames: array<f64>; // [N*12] 3x4 R|t
@group(0) @binding(3) var<storage, read>       true_frames: array<f64>; // [N*12]
@group(0) @binding(4) var<storage, read_write> out: array<f64>;       // [N] per-residue FAPE
@group(0) @binding(5) var<uniform>             params: FapeParams;

fn transform_inv_pred(i: u32, x: vec3<f64>) -> vec3<f64> {
    let base = i * 12u;
    let r00 = pred_frames[base + 0u];
    let r10 = pred_frames[base + 4u];
    let r20 = pred_frames[base + 8u];
    let tx = pred_frames[base + 3u];
    let ty = pred_frames[base + 7u];
    let tz = pred_frames[base + 11u];
    let dx = x.x - tx;
    let dy = x.y - ty;
    let dz = x.z - tz;
    return vec3<f64>(r00 * dx + r10 * dy + r20 * dz,
        pred_frames[base + 1u] * dx + pred_frames[base + 5u] * dy + pred_frames[base + 9u] * dz,
        pred_frames[base + 2u] * dx + pred_frames[base + 6u] * dy + pred_frames[base + 10u] * dz);
}

fn transform_inv_true(i: u32, x: vec3<f64>) -> vec3<f64> {
    let base = i * 12u;
    let r00 = true_frames[base + 0u];
    let r10 = true_frames[base + 4u];
    let r20 = true_frames[base + 8u];
    let tx = true_frames[base + 3u];
    let ty = true_frames[base + 7u];
    let tz = true_frames[base + 11u];
    let dx = x.x - tx;
    let dy = x.y - ty;
    let dz = x.z - tz;
    return vec3<f64>(r00 * dx + r10 * dy + r20 * dz,
        true_frames[base + 1u] * dx + true_frames[base + 5u] * dy + true_frames[base + 9u] * dz,
        true_frames[base + 2u] * dx + true_frames[base + 6u] * dy + true_frames[base + 10u] * dz);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = params.n_residues;
    let M = params.n_atoms;
    let d_clamp = params.d_clamp;

    let i = gid.x;
    if i >= N { return; }

    var sum_dist = f64(0.0);
    for (var a = 0u; a < M; a = a + 1u) {
        let pos_base = (i * M + a) * 3u;
        let pred_p = vec3<f64>(pred_pos[pos_base], pred_pos[pos_base + 1u], pred_pos[pos_base + 2u]);
        let true_p = vec3<f64>(true_pos[pos_base], true_pos[pos_base + 1u], true_pos[pos_base + 2u]);

        let pred_local = transform_inv_pred(i, pred_p);
        let true_local = transform_inv_true(i, true_p);

        let dx = pred_local.x - true_local.x;
        let dy = pred_local.y - true_local.y;
        let dz = pred_local.z - true_local.z;
        let d = sqrt_f64(dx * dx + dy * dy + dz * dz);
        sum_dist += min(d, d_clamp);
    }

    out[i] = sum_dist / f64(M);
}

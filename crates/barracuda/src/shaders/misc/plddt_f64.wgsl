// SPDX-License-Identifier: AGPL-3.0-or-later
//
// plddt_f64.wgsl — Predicted LDDT confidence score per residue (AlphaFold2)
//
// pLDDT[i] = (1/4) * Σ_t (fraction of neighbors within threshold_t after superposition)
// Simplified: single cutoff, fraction of neighbors within distance threshold.
// Uses pred_pos for structure; neighbors = residues within sequence window.
//
// Bindings: @0 pred_pos[N*3], @1 true_pos[N*3], @2 out[N], @3 uniform{n_residues, cutoff}

enable f64;

struct PlddtParams {
    n_residues: u32,
    _pad: [u32; 3],
    cutoff: f64,
}

@group(0) @binding(0) var<storage, read>       pred_pos: array<f64>;  // [N*3]
@group(0) @binding(1) var<storage, read>       true_pos: array<f64>; // [N*3]
@group(0) @binding(2) var<storage, read_write> out: array<f64>;       // [N]
@group(0) @binding(3) var<uniform>             params: PlddtParams;

fn dist_pred(i: u32, j: u32) -> f64 {
    let pi = i * 3u;
    let pj = j * 3u;
    let dx = pred_pos[pi] - pred_pos[pj];
    let dy = pred_pos[pi + 1u] - pred_pos[pj + 1u];
    let dz = pred_pos[pi + 2u] - pred_pos[pj + 2u];
    return sqrt_f64(dx * dx + dy * dy + dz * dz);
}

fn dist_true(i: u32, j: u32) -> f64 {
    let pi = i * 3u;
    let pj = j * 3u;
    let dx = true_pos[pi] - true_pos[pj];
    let dy = true_pos[pi + 1u] - true_pos[pj + 1u];
    let dz = true_pos[pi + 2u] - true_pos[pj + 2u];
    return sqrt_f64(dx * dx + dy * dy + dz * dz);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let N = params.n_residues;
    let cutoff = params.cutoff;

    let i = gid.x;
    if i >= N { return; }

    // Count neighbors j where both pred and true have dist < cutoff
    var count = 0u;
    var total = 0u;
    for (var j = 0u; j < N; j = j + 1u) {
        if i == j { continue; }
        total += 1u;
        let d_pred = dist_pred(i, j);
        let d_true = dist_true(i, j);
        if d_pred < cutoff && d_true < cutoff {
            count += 1u;
        }
    }

    let frac = select(f64(0.0), f64(count) / f64(total), total > 0u);
    out[i] = frac;
}

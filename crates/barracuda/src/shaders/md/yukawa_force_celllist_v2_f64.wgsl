// SPDX-License-Identifier: AGPL-3.0-or-later
//
// DEPRECATED — Use barracuda::ops::md::forces::YukawaCelllistF64
// Retained for fossil record. This was a flat-loop experiment to debug
// Naga/SPIR-V nested-loop issues. barracuda canonical version is stable.
//
// Yukawa Cell-List Force v2 (f64) — flat loop, same bindings as v1
//
// Identical physics to yukawa_force_celllist_f64.wgsl but uses a flat
// single loop over 27 neighbor offsets (precomputed) instead of
// 3 nested for-loops. This tests whether the Naga/SPIR-V compilation
// of deeply nested i32 loops was causing the force computation bug.
//
// Bindings: same as yukawa_force_celllist_f64.wgsl

@group(0) @binding(0) var<storage, read> positions: array<f64>;
@group(0) @binding(1) var<storage, read_write> forces: array<f64>;
@group(0) @binding(2) var<storage, read_write> pe_buf: array<f64>;
@group(0) @binding(3) var<storage, read> params: array<f64>;
@group(0) @binding(4) var<storage, read> cell_start: array<u32>;
@group(0) @binding(5) var<storage, read> cell_count: array<u32>;

fn pbc_wrap(delta: f64, box_size: f64) -> f64 {
    var d = delta;
    let half = box_size / 2.0;
    if (d > half)  { d = d - box_size; }
    if (d < -half) { d = d + box_size; }
    return d;
}

fn wrap_cell(c: i32, n: i32) -> i32 {
    var w = c;
    if (w < 0)  { w = w + n; }
    if (w >= n) { w = w - n; }
    return w;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let n_particles = u32(params[0]);
    if (idx >= n_particles) { return; }

    let kappa      = params[1];
    let prefactor  = params[2];
    let cutoff_sq  = params[3];
    let box_x      = params[4];
    let box_y      = params[5];
    let box_z      = params[6];
    let softening  = params[7];
    let ncx        = i32(params[8]);
    let ncy        = i32(params[9]);
    let ncz        = i32(params[10]);
    let csx        = params[11];
    let csy        = params[12];
    let csz        = params[13];

    let px = positions[idx * 3u];
    let py = positions[idx * 3u + 1u];
    let pz = positions[idx * 3u + 2u];

    // Determine this particle's cell
    let my_cx = i32(px / csx);
    let my_cy = i32(py / csy);
    let my_cz = i32(pz / csz);

    // Accumulate force and PE as f64 zeros
    var acc_fx = px - px;
    var acc_fy = px - px;
    var acc_fz = px - px;
    var acc_pe = px - px;

    // Flat loop over 27 neighbor offsets
    for (var neigh = 0u; neigh < 27u; neigh = neigh + 1u) {
        // Decode neighbor offset from flat index
        let off_x = i32(neigh % 3u) - 1;
        let off_y = i32((neigh / 3u) % 3u) - 1;
        let off_z = i32(neigh / 9u) - 1;

        let nb_cx = wrap_cell(my_cx + off_x, ncx);
        let nb_cy = wrap_cell(my_cy + off_y, ncy);
        let nb_cz = wrap_cell(my_cz + off_z, ncz);
        let cell_linear = u32(nb_cx + nb_cy * ncx + nb_cz * ncx * ncy);

        let start = cell_start[cell_linear];
        let cnt   = cell_count[cell_linear];

        for (var k = 0u; k < cnt; k = k + 1u) {
            let j = start + k;
            if (idx == j) { continue; }

            let qx = positions[j * 3u];
            let qy = positions[j * 3u + 1u];
            let qz = positions[j * 3u + 2u];

            let rx = pbc_wrap(qx - px, box_x);
            let ry = pbc_wrap(qy - py, box_y);
            let rz = pbc_wrap(qz - pz, box_z);

            let r_sq = rx * rx + ry * ry + rz * rz;
            if (r_sq > cutoff_sq) { continue; }

            let r = sqrt(r_sq + softening);
            let scr = exp(-kappa * r);
            let fmag = prefactor * scr * (1.0 + kappa * r) / r_sq;
            let ir = 1.0 / r;

            acc_fx = acc_fx - fmag * rx * ir;
            acc_fy = acc_fy - fmag * ry * ir;
            acc_fz = acc_fz - fmag * rz * ir;
            acc_pe = acc_pe + 0.5 * prefactor * scr * ir;
        }
    }

    forces[idx * 3u]      = acc_fx;
    forces[idx * 3u + 1u] = acc_fy;
    forces[idx * 3u + 2u] = acc_fz;
    pe_buf[idx] = acc_pe;
}

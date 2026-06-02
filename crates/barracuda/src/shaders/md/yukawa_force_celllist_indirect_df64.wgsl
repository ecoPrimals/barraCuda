// SPDX-License-Identifier: AGPL-3.0-or-later
//
// Yukawa Cell-List Force with Indirect Indexing (DF64)
//
// Prepend: df64_core.wgsl, df64_transcendentals.wgsl
// Also needs: round_f64 from math_f64 preamble (injected by compile_shader_f64)
//
// DF64 precision strategy (Fp64Strategy::Hybrid):
//   DF64 (FP32 cores): force magnitude, sqrt, exp, accumulation, screening
//   f64 (FP64 units): PBC rounding (precision-critical), cutoff compare, storage I/O
//
// Same bindings as yukawa_force_celllist_indirect_f64.wgsl — drop-in replacement.

@group(0) @binding(0) var<storage, read> positions: array<f64>;
@group(0) @binding(1) var<storage, read_write> forces: array<f64>;
@group(0) @binding(2) var<storage, read_write> pe_buf: array<f64>;
@group(0) @binding(3) var<storage, read> params: array<f64>;
@group(0) @binding(4) var<storage, read> cell_start: array<u32>;
@group(0) @binding(5) var<storage, read> cell_count: array<u32>;
@group(0) @binding(6) var<storage, read> sorted_indices: array<u32>;

fn pbc_delta_cl(delta: f64, box_size: f64) -> f64 {
    return delta - box_size * round_f64(delta / box_size);
}

fn cell_idx(cx: i32, cy: i32, cz: i32, nx: i32, ny: i32, nz: i32) -> u32 {
    var wx = cx;
    if (wx < 0)  { wx = wx + nx; }
    if (wx >= nx) { wx = wx - nx; }
    var wy = cy;
    if (wy < 0)  { wy = wy + ny; }
    if (wy >= ny) { wy = wy - ny; }
    var wz = cz;
    if (wz < 0)  { wz = wz + nz; }
    if (wz >= nz) { wz = wz - nz; }
    return u32(wx + wy * nx + wz * nx * ny);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let n = u32(params[0]);
    if (i >= n) { return; }

    let kappa     = params[1];
    let prefactor = params[2];
    let cutoff_sq = params[3];
    let box_x     = params[4];
    let box_y     = params[5];
    let box_z     = params[6];
    let eps       = params[7];
    let nx        = i32(params[8]);
    let ny        = i32(params[9]);
    let nz        = i32(params[10]);
    let cell_sx   = params[11];
    let cell_sy   = params[12];
    let cell_sz   = params[13];

    let xi = positions[i * 3u];
    let yi = positions[i * 3u + 1u];
    let zi = positions[i * 3u + 2u];

    let ci_x = i32(xi / cell_sx);
    let ci_y = i32(yi / cell_sy);
    let ci_z = i32(zi / cell_sz);

    // DF64 accumulators — force and PE accumulate on FP32 cores
    var fx = df64_zero();
    var fy = df64_zero();
    var fz = df64_zero();
    var pe = df64_zero();

    let kappa_df = df64_from_f64(kappa);
    let prefactor_df = df64_from_f64(prefactor);
    let half = df64_from_f32(0.5);
    let one = df64_from_f32(1.0);

    for (var dz = -1; dz <= 1; dz = dz + 1) {
        for (var dy = -1; dy <= 1; dy = dy + 1) {
            for (var dx = -1; dx <= 1; dx = dx + 1) {
                let c_idx = cell_idx(ci_x + dx, ci_y + dy, ci_z + dz, nx, ny, nz);
                let start = cell_start[c_idx];
                let count = cell_count[c_idx];

                for (var jj = 0u; jj < count; jj = jj + 1u) {
                    let j = sorted_indices[start + jj];
                    if (i == j) { continue; }

                    let xj = positions[j * 3u];
                    let yj = positions[j * 3u + 1u];
                    let zj = positions[j * 3u + 2u];

                    // PBC in f64 (precision-critical rounding)
                    let ddx_f64 = pbc_delta_cl(xj - xi, box_x);
                    let ddy_f64 = pbc_delta_cl(yj - yi, box_y);
                    let ddz_f64 = pbc_delta_cl(zj - zi, box_z);

                    let r_sq_f64 = ddx_f64 * ddx_f64 + ddy_f64 * ddy_f64 + ddz_f64 * ddz_f64;
                    if (r_sq_f64 > cutoff_sq) { continue; }

                    // Promote to DF64 for force math
                    let ddx = df64_from_f64(ddx_f64);
                    let ddy = df64_from_f64(ddy_f64);
                    let ddz = df64_from_f64(ddz_f64);
                    let r_sq = df64_from_f64(r_sq_f64);
                    let r = sqrt_df64(df64_add(r_sq, df64_from_f64(eps)));

                    let screening = exp_df64(df64_neg(df64_mul(kappa_df, r)));
                    let kappa_r = df64_mul(kappa_df, r);
                    let force_mag = df64_div(
                        df64_mul(prefactor_df, df64_mul(screening, df64_add(one, kappa_r))),
                        r_sq
                    );

                    let inv_r = df64_div(one, r);
                    fx = df64_sub(fx, df64_mul(force_mag, df64_mul(ddx, inv_r)));
                    fy = df64_sub(fy, df64_mul(force_mag, df64_mul(ddy, inv_r)));
                    fz = df64_sub(fz, df64_mul(force_mag, df64_mul(ddz, inv_r)));

                    pe = df64_add(pe, df64_mul(half, df64_mul(prefactor_df, df64_mul(screening, inv_r))));
                }
            }
        }
    }

    forces[i * 3u]      = df64_to_f64(fx);
    forces[i * 3u + 1u] = df64_to_f64(fy);
    forces[i * 3u + 2u] = df64_to_f64(fz);
    pe_buf[i] = df64_to_f64(pe);
}

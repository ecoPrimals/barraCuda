// Yukawa All-Pairs Force (DF64) — Full FP32 Core Streaming with PBC
//
// Prepend: df64_core.wgsl, df64_transcendentals.wgsl
// Also needs: round_f64 from math_f64 preamble (injected by compile_shader_f64)
//
// ALL-DF64 PRECISION:
//   DF64 (FP32 cores): force magnitude, sqrt, exp, accumulation, screening power
//   f64 (FP64 units): PBC rounding (precision-critical), cutoff compare, storage I/O

@group(0) @binding(0) var<storage, read> positions: array<f64>;
@group(0) @binding(1) var<storage, read_write> forces: array<f64>;
@group(0) @binding(2) var<storage, read_write> pe_buf: array<f64>;
@group(0) @binding(3) var<storage, read> params: array<f64>;

// PBC minimum image stays in f64 — rounding must be exact
fn pbc_delta(delta: f64, box_size: f64) -> f64 {
    return delta - box_size * round_f64(delta / box_size);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let n = u32(params[0]);
    if (i >= n) { return; }

    let xi = positions[i * 3u];
    let yi = positions[i * 3u + 1u];
    let zi = positions[i * 3u + 2u];

    let kappa     = params[1];
    let prefactor = params[2];
    let cutoff_sq = params[3];
    let box_x     = params[4];
    let box_y     = params[5];
    let box_z     = params[6];
    let eps       = params[7];

    // DF64 accumulators
    var fx = df64_zero();
    var fy = df64_zero();
    var fz = df64_zero();
    var pe = df64_zero();

    let kappa_df = df64_from_f64(kappa);
    let prefactor_df = df64_from_f64(prefactor);
    let half = df64_from_f32(0.5);
    let one = df64_from_f32(1.0);

    for (var j = 0u; j < n; j = j + 1u) {
        if (i == j) { continue; }

        let xj = positions[j * 3u];
        let yj = positions[j * 3u + 1u];
        let zj = positions[j * 3u + 2u];

        // PBC in f64 (precision-critical rounding)
        let dx_f64 = pbc_delta(xj - xi, box_x);
        let dy_f64 = pbc_delta(yj - yi, box_y);
        let dz_f64 = pbc_delta(zj - zi, box_z);

        let r_sq_f64 = dx_f64 * dx_f64 + dy_f64 * dy_f64 + dz_f64 * dz_f64;
        if (r_sq_f64 > cutoff_sq) { continue; }

        let dx = df64_from_f64(dx_f64);
        let dy = df64_from_f64(dy_f64);
        let dz = df64_from_f64(dz_f64);
        let r_sq = df64_from_f64(r_sq_f64);
        let r = sqrt_df64(df64_add(r_sq, df64_from_f64(eps)));

        let screening = exp_df64(df64_neg(df64_mul(kappa_df, r)));
        let kappa_r = df64_mul(kappa_df, r);
        let force_mag = df64_div(
            df64_mul(prefactor_df, df64_mul(screening, df64_add(one, kappa_r))),
            r_sq
        );

        let inv_r = df64_div(one, r);
        fx = df64_sub(fx, df64_mul(force_mag, df64_mul(dx, inv_r)));
        fy = df64_sub(fy, df64_mul(force_mag, df64_mul(dy, inv_r)));
        fz = df64_sub(fz, df64_mul(force_mag, df64_mul(dz, inv_r)));

        // PE: U = prefactor * exp(-kappa*r) / r
        pe = df64_add(pe, df64_mul(half, df64_mul(prefactor_df, df64_mul(screening, inv_r))));
    }

    forces[i * 3u]      = df64_to_f64(fx);
    forces[i * 3u + 1u] = df64_to_f64(fy);
    forces[i * 3u + 2u] = df64_to_f64(fz);
    pe_buf[i] = df64_to_f64(pe);
}

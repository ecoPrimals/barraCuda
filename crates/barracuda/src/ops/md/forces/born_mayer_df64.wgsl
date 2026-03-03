// Born-Mayer Force Kernel (DF64) — Full FP32 Core Streaming
//
// Prepend: df64_core.wgsl, df64_transcendentals.wgsl
//
// ALL-DF64 PRECISION:
//   DF64 (FP32 cores): distance, sqrt, exp, mixing rules, force magnitude, accumulation
//   f64 (FP64 units): only storage I/O and cutoff compare

@group(0) @binding(0) var<storage, read> positions: array<f64>;
@group(0) @binding(1) var<storage, read> A_params: array<f64>;
@group(0) @binding(2) var<storage, read> rho_params: array<f64>;
@group(0) @binding(3) var<storage, read_write> forces: array<f64>;
@group(0) @binding(4) var<uniform> params: Params;

struct Params {
    n_particles: u32,
    _pad0: u32,
    cutoff_radius: f64,
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= params.n_particles) { return; }

    let xi = df64_from_f64(positions[i * 3u]);
    let yi = df64_from_f64(positions[i * 3u + 1u]);
    let zi = df64_from_f64(positions[i * 3u + 2u]);
    let A_i = df64_from_f64(A_params[i]);
    let rho_i = df64_from_f64(rho_params[i]);

    let half = df64_from_f32(0.5);

    var fx = df64_zero();
    var fy = df64_zero();
    var fz = df64_zero();

    let cutoff = params.cutoff_radius;

    for (var j = 0u; j < params.n_particles; j = j + 1u) {
        if (i == j) { continue; }

        let xj = df64_from_f64(positions[j * 3u]);
        let yj = df64_from_f64(positions[j * 3u + 1u]);
        let zj = df64_from_f64(positions[j * 3u + 2u]);

        let rx = df64_sub(xj, xi);
        let ry = df64_sub(yj, yi);
        let rz = df64_sub(zj, zi);

        let r_sq = df64_add(df64_add(df64_mul(rx, rx), df64_mul(ry, ry)), df64_mul(rz, rz));

        let r = sqrt_df64(r_sq);
        let r_f64 = df64_to_f64(r);
        if (r_f64 > cutoff || r_f64 < 1e-10) { continue; }

        let A_j = df64_from_f64(A_params[j]);
        let rho_j = df64_from_f64(rho_params[j]);
        let A = sqrt_df64(df64_mul(A_i, A_j));
        let rho = df64_mul(df64_add(rho_i, rho_j), half);

        // Born-Mayer force: F = (A/ρ) * exp(-r/ρ) * r_hat
        let exp_term = exp_df64(df64_div(df64_neg(r), rho));
        let force_magnitude = df64_mul(df64_div(A, rho), exp_term);

        let inv_r = df64_div(df64_from_f32(1.0), r);
        fx = df64_add(fx, df64_mul(force_magnitude, df64_mul(rx, inv_r)));
        fy = df64_add(fy, df64_mul(force_magnitude, df64_mul(ry, inv_r)));
        fz = df64_add(fz, df64_mul(force_magnitude, df64_mul(rz, inv_r)));
    }

    forces[i * 3u] = df64_to_f64(fx);
    forces[i * 3u + 1u] = df64_to_f64(fy);
    forces[i * 3u + 2u] = df64_to_f64(fz);
}

@compute @workgroup_size(256)
fn born_mayer_with_energy(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= params.n_particles) { return; }

    let xi = df64_from_f64(positions[i * 3u]);
    let yi = df64_from_f64(positions[i * 3u + 1u]);
    let zi = df64_from_f64(positions[i * 3u + 2u]);
    let A_i = df64_from_f64(A_params[i]);
    let rho_i = df64_from_f64(rho_params[i]);

    let half = df64_from_f32(0.5);

    var fx = df64_zero();
    var fy = df64_zero();
    var fz = df64_zero();

    let cutoff = params.cutoff_radius;

    for (var j = 0u; j < params.n_particles; j = j + 1u) {
        if (i == j) { continue; }

        let xj = df64_from_f64(positions[j * 3u]);
        let yj = df64_from_f64(positions[j * 3u + 1u]);
        let zj = df64_from_f64(positions[j * 3u + 2u]);

        let rx = df64_sub(xj, xi);
        let ry = df64_sub(yj, yi);
        let rz = df64_sub(zj, zi);

        let r_sq = df64_add(df64_add(df64_mul(rx, rx), df64_mul(ry, ry)), df64_mul(rz, rz));

        let r = sqrt_df64(r_sq);
        let r_f64 = df64_to_f64(r);
        if (r_f64 > cutoff || r_f64 < 1e-10) { continue; }

        let A_j = df64_from_f64(A_params[j]);
        let rho_j = df64_from_f64(rho_params[j]);
        let A = sqrt_df64(df64_mul(A_i, A_j));
        let rho = df64_mul(df64_add(rho_i, rho_j), half);

        let exp_term = exp_df64(df64_div(df64_neg(r), rho));
        let force_magnitude = df64_mul(df64_div(A, rho), exp_term);

        let inv_r = df64_div(df64_from_f32(1.0), r);
        fx = df64_add(fx, df64_mul(force_magnitude, df64_mul(rx, inv_r)));
        fy = df64_add(fy, df64_mul(force_magnitude, df64_mul(ry, inv_r)));
        fz = df64_add(fz, df64_mul(force_magnitude, df64_mul(rz, inv_r)));
    }

    forces[i * 3u] = df64_to_f64(fx);
    forces[i * 3u + 1u] = df64_to_f64(fy);
    forces[i * 3u + 2u] = df64_to_f64(fz);
}

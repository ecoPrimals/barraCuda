// SPDX-License-Identifier: AGPL-3.0-only
// Lennard-Jones Force Kernel (DF64) — Full FP32 Core Streaming
//
// Prepend: df64_core.wgsl, df64_transcendentals.wgsl
//
// ALL-DF64 PRECISION:
//   DF64 (FP32 cores): distance, sqrt, σ/r powers, force accumulation
//   f64 (FP64 units): only cutoff comparison and storage I/O
//
// The O(N²) pairwise loop has ~13 DF64 multiplies per pair vs 2 f64 ops.
// Net: ~10× throughput on consumer GPUs for the force computation.
//
// Buffer layout: UNCHANGED from lennard_jones_f64.wgsl.

struct Params {
    n_particles: u32,
    _pad0: u32,
    cutoff_radius: f64,
    cutoff_radius_sq: f64,
}

@group(0) @binding(0) var<storage, read> positions: array<f64>;
@group(0) @binding(1) var<storage, read> sigmas: array<f64>;
@group(0) @binding(2) var<storage, read> epsilons: array<f64>;
@group(0) @binding(3) var<storage, read_write> forces: array<f64>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(256)
fn lennard_jones_f64(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= params.n_particles) { return; }

    let xi = df64_from_f64(positions[i * 3u]);
    let yi = df64_from_f64(positions[i * 3u + 1u]);
    let zi = df64_from_f64(positions[i * 3u + 2u]);
    let sigma_i = df64_from_f64(sigmas[i]);
    let epsilon_i = df64_from_f64(epsilons[i]);

    let half = df64_from_f32(0.5);
    let two = df64_from_f32(2.0);
    let twenty_four = df64_from_f32(24.0);

    var fx = df64_zero();
    var fy = df64_zero();
    var fz = df64_zero();

    let cutoff_sq = params.cutoff_radius_sq;

    for (var j = 0u; j < params.n_particles; j = j + 1u) {
        if (i == j) { continue; }

        let xj = df64_from_f64(positions[j * 3u]);
        let yj = df64_from_f64(positions[j * 3u + 1u]);
        let zj = df64_from_f64(positions[j * 3u + 2u]);

        let dx = df64_sub(xj, xi);
        let dy = df64_sub(yj, yi);
        let dz = df64_sub(zj, zi);

        let r_sq = df64_add(df64_add(df64_mul(dx, dx), df64_mul(dy, dy)), df64_mul(dz, dz));

        let r_sq_f64 = df64_to_f64(r_sq);
        if (r_sq_f64 > cutoff_sq || r_sq_f64 < 1e-12) { continue; }

        let r = sqrt_df64(r_sq);

        let sigma_j = df64_from_f64(sigmas[j]);
        let epsilon_j = df64_from_f64(epsilons[j]);
        let sigma = df64_mul(df64_add(sigma_i, sigma_j), half);
        let epsilon = sqrt_df64(df64_mul(epsilon_i, epsilon_j));

        // σ/r powers (DF64 — the hot compute path)
        let sigma_r = df64_div(sigma, r);
        let sigma_r_sq = df64_mul(sigma_r, sigma_r);
        let sigma_r6 = df64_mul(df64_mul(sigma_r_sq, sigma_r_sq), sigma_r_sq);
        let sigma_r12 = df64_mul(sigma_r6, sigma_r6);

        // Force magnitude: 24ε/r² × (2σ¹² - σ⁶)
        let bracket = df64_sub(df64_mul(two, sigma_r12), sigma_r6);
        let force_over_r = df64_mul(df64_div(df64_mul(twenty_four, epsilon), r_sq), bracket);

        // Accumulate force
        fx = df64_sub(fx, df64_mul(force_over_r, dx));
        fy = df64_sub(fy, df64_mul(force_over_r, dy));
        fz = df64_sub(fz, df64_mul(force_over_r, dz));
    }

    // Boundary: DF64 → f64 for store
    forces[i * 3u] = df64_to_f64(fx);
    forces[i * 3u + 1u] = df64_to_f64(fy);
    forces[i * 3u + 2u] = df64_to_f64(fz);
}

@compute @workgroup_size(256)
fn lennard_jones_shifted_f64(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= params.n_particles) { return; }

    let xi = df64_from_f64(positions[i * 3u]);
    let yi = df64_from_f64(positions[i * 3u + 1u]);
    let zi = df64_from_f64(positions[i * 3u + 2u]);
    let sigma_i = df64_from_f64(sigmas[i]);
    let epsilon_i = df64_from_f64(epsilons[i]);

    let half = df64_from_f32(0.5);
    let two = df64_from_f32(2.0);
    let twenty_four = df64_from_f32(24.0);

    var fx = df64_zero();
    var fy = df64_zero();
    var fz = df64_zero();

    let cutoff_sq = params.cutoff_radius_sq;

    for (var j = 0u; j < params.n_particles; j = j + 1u) {
        if (i == j) { continue; }

        let xj = df64_from_f64(positions[j * 3u]);
        let yj = df64_from_f64(positions[j * 3u + 1u]);
        let zj = df64_from_f64(positions[j * 3u + 2u]);

        let dx = df64_sub(xj, xi);
        let dy = df64_sub(yj, yi);
        let dz = df64_sub(zj, zi);

        let r_sq = df64_add(df64_add(df64_mul(dx, dx), df64_mul(dy, dy)), df64_mul(dz, dz));

        let r_sq_f64 = df64_to_f64(r_sq);
        if (r_sq_f64 > cutoff_sq || r_sq_f64 < 1e-12) { continue; }

        let r = sqrt_df64(r_sq);

        let sigma_j = df64_from_f64(sigmas[j]);
        let epsilon_j = df64_from_f64(epsilons[j]);
        let sigma = df64_mul(df64_add(sigma_i, sigma_j), half);
        let epsilon = sqrt_df64(df64_mul(epsilon_i, epsilon_j));

        let sigma_r = df64_div(sigma, r);
        let sigma_r_sq = df64_mul(sigma_r, sigma_r);
        let sigma_r6 = df64_mul(df64_mul(sigma_r_sq, sigma_r_sq), sigma_r_sq);
        let sigma_r12 = df64_mul(sigma_r6, sigma_r6);

        let bracket = df64_sub(df64_mul(two, sigma_r12), sigma_r6);
        let force_over_r = df64_mul(df64_div(df64_mul(twenty_four, epsilon), r_sq), bracket);

        fx = df64_sub(fx, df64_mul(force_over_r, dx));
        fy = df64_sub(fy, df64_mul(force_over_r, dy));
        fz = df64_sub(fz, df64_mul(force_over_r, dz));
    }

    forces[i * 3u] = df64_to_f64(fx);
    forces[i * 3u + 1u] = df64_to_f64(fy);
    forces[i * 3u + 2u] = df64_to_f64(fz);
}

@group(0) @binding(5) var<storage, read_write> potential_energy: array<f64>;

@compute @workgroup_size(256)
fn lennard_jones_with_energy_f64(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if (i >= params.n_particles) { return; }

    let xi = df64_from_f64(positions[i * 3u]);
    let yi = df64_from_f64(positions[i * 3u + 1u]);
    let zi = df64_from_f64(positions[i * 3u + 2u]);
    let sigma_i = df64_from_f64(sigmas[i]);
    let epsilon_i = df64_from_f64(epsilons[i]);

    let half = df64_from_f32(0.5);
    let two = df64_from_f32(2.0);
    let four = df64_from_f32(4.0);
    let twenty_four = df64_from_f32(24.0);

    var fx = df64_zero();
    var fy = df64_zero();
    var fz = df64_zero();
    var pe = df64_zero();

    let cutoff_sq = params.cutoff_radius_sq;

    for (var j = 0u; j < params.n_particles; j = j + 1u) {
        if (i == j) { continue; }

        let xj = df64_from_f64(positions[j * 3u]);
        let yj = df64_from_f64(positions[j * 3u + 1u]);
        let zj = df64_from_f64(positions[j * 3u + 2u]);

        let dx = df64_sub(xj, xi);
        let dy = df64_sub(yj, yi);
        let dz = df64_sub(zj, zi);

        let r_sq = df64_add(df64_add(df64_mul(dx, dx), df64_mul(dy, dy)), df64_mul(dz, dz));

        let r_sq_f64 = df64_to_f64(r_sq);
        if (r_sq_f64 > cutoff_sq || r_sq_f64 < 1e-12) { continue; }

        let r = sqrt_df64(r_sq);

        let sigma_j = df64_from_f64(sigmas[j]);
        let epsilon_j = df64_from_f64(epsilons[j]);
        let sigma = df64_mul(df64_add(sigma_i, sigma_j), half);
        let epsilon = sqrt_df64(df64_mul(epsilon_i, epsilon_j));

        let sigma_r = df64_div(sigma, r);
        let sigma_r_sq = df64_mul(sigma_r, sigma_r);
        let sigma_r6 = df64_mul(df64_mul(sigma_r_sq, sigma_r_sq), sigma_r_sq);
        let sigma_r12 = df64_mul(sigma_r6, sigma_r6);

        let bracket = df64_sub(df64_mul(two, sigma_r12), sigma_r6);
        let force_over_r = df64_mul(df64_div(df64_mul(twenty_four, epsilon), r_sq), bracket);

        fx = df64_sub(fx, df64_mul(force_over_r, dx));
        fy = df64_sub(fy, df64_mul(force_over_r, dy));
        fz = df64_sub(fz, df64_mul(force_over_r, dz));

        // U = 4ε(σ¹² - σ⁶), halved to avoid double counting
        pe = df64_add(pe, df64_mul(df64_mul(half, df64_mul(four, epsilon)), df64_sub(sigma_r12, sigma_r6)));
    }

    forces[i * 3u] = df64_to_f64(fx);
    forces[i * 3u + 1u] = df64_to_f64(fy);
    forces[i * 3u + 2u] = df64_to_f64(fz);
    potential_energy[i] = df64_to_f64(pe);
}

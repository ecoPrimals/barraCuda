// SPDX-License-Identifier: AGPL-3.0-only
// Morse Force Kernel (DF64) — Full FP32 Core Streaming
//
// Prepend: df64_core.wgsl, df64_transcendentals.wgsl
//
// ALL-DF64 PRECISION:
//   DF64 (FP32 cores): displacement, distance, sqrt, exp, force magnitude, direction
//   f64 (FP64 units): only storage I/O and degenerate-bond guard

struct Params {
    n_bonds: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> positions: array<f64>;
@group(0) @binding(1) var<storage, read> bond_pairs: array<u32>;
@group(0) @binding(2) var<storage, read> dissociation_energy: array<f64>;
@group(0) @binding(3) var<storage, read> width_param: array<f64>;
@group(0) @binding(4) var<storage, read> equilibrium_dist: array<f64>;
@group(0) @binding(5) var<storage, read_write> bond_forces: array<f64>;
@group(0) @binding(6) var<uniform> params: Params;

@compute @workgroup_size(256)
fn morse_bonds_f64(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let bond_idx = global_id.x;
    if (bond_idx >= params.n_bonds) { return; }

    let i = bond_pairs[bond_idx * 2u];
    let j = bond_pairs[bond_idx * 2u + 1u];

    // Load positions as DF64
    let xi = df64_from_f64(positions[i * 3u]);
    let yi = df64_from_f64(positions[i * 3u + 1u]);
    let zi = df64_from_f64(positions[i * 3u + 2u]);
    let xj = df64_from_f64(positions[j * 3u]);
    let yj = df64_from_f64(positions[j * 3u + 1u]);
    let zj = df64_from_f64(positions[j * 3u + 2u]);

    let D = df64_from_f64(dissociation_energy[bond_idx]);
    let a = df64_from_f64(width_param[bond_idx]);
    let r0 = df64_from_f64(equilibrium_dist[bond_idx]);

    // DF64 displacement
    let dx = df64_sub(xj, xi);
    let dy = df64_sub(yj, yi);
    let dz = df64_sub(zj, zi);

    let r_sq = df64_add(df64_add(df64_mul(dx, dx), df64_mul(dy, dy)), df64_mul(dz, dz));
    let r_sq_f64 = df64_to_f64(r_sq);

    let out_base = bond_idx * 6u;
    if (r_sq_f64 < 1e-20) {
        let z = df64_to_f64(df64_zero());
        bond_forces[out_base] = z;
        bond_forces[out_base + 1u] = z;
        bond_forces[out_base + 2u] = z;
        bond_forces[out_base + 3u] = z;
        bond_forces[out_base + 4u] = z;
        bond_forces[out_base + 5u] = z;
        return;
    }

    let r = sqrt_df64(r_sq);
    let delta_r = df64_sub(r, r0);
    let exp_term = exp_df64(df64_neg(df64_mul(a, delta_r)));

    // DF64 force computation
    let one = df64_from_f32(1.0);
    let two = df64_from_f32(2.0);
    let one_minus_exp = df64_sub(one, exp_term);
    let force_magnitude = df64_mul(df64_mul(df64_mul(two, D), a), df64_mul(one_minus_exp, exp_term));
    let force_over_r = df64_div(force_magnitude, r);

    let fx = df64_mul(force_over_r, dx);
    let fy = df64_mul(force_over_r, dy);
    let fz = df64_mul(force_over_r, dz);

    // Store as f64
    bond_forces[out_base] = df64_to_f64(fx);
    bond_forces[out_base + 1u] = df64_to_f64(fy);
    bond_forces[out_base + 2u] = df64_to_f64(fz);
    bond_forces[out_base + 3u] = -df64_to_f64(fx);
    bond_forces[out_base + 4u] = -df64_to_f64(fy);
    bond_forces[out_base + 5u] = -df64_to_f64(fz);
}

@group(0) @binding(7) var<storage, read_write> bond_energy: array<f64>;

@compute @workgroup_size(256)
fn morse_with_energy_f64(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let bond_idx = global_id.x;
    if (bond_idx >= params.n_bonds) { return; }

    let i = bond_pairs[bond_idx * 2u];
    let j = bond_pairs[bond_idx * 2u + 1u];

    let xi = df64_from_f64(positions[i * 3u]);
    let yi = df64_from_f64(positions[i * 3u + 1u]);
    let zi = df64_from_f64(positions[i * 3u + 2u]);
    let xj = df64_from_f64(positions[j * 3u]);
    let yj = df64_from_f64(positions[j * 3u + 1u]);
    let zj = df64_from_f64(positions[j * 3u + 2u]);

    let D = df64_from_f64(dissociation_energy[bond_idx]);
    let a = df64_from_f64(width_param[bond_idx]);
    let r0 = df64_from_f64(equilibrium_dist[bond_idx]);

    let dx = df64_sub(xj, xi);
    let dy = df64_sub(yj, yi);
    let dz = df64_sub(zj, zi);

    let r_sq = df64_add(df64_add(df64_mul(dx, dx), df64_mul(dy, dy)), df64_mul(dz, dz));
    let r_sq_f64 = df64_to_f64(r_sq);

    let out_base = bond_idx * 6u;
    if (r_sq_f64 < 1e-20) {
        let z = df64_to_f64(df64_zero());
        bond_forces[out_base] = z;
        bond_forces[out_base + 1u] = z;
        bond_forces[out_base + 2u] = z;
        bond_forces[out_base + 3u] = z;
        bond_forces[out_base + 4u] = z;
        bond_forces[out_base + 5u] = z;
        bond_energy[bond_idx] = z;
        return;
    }

    let r = sqrt_df64(r_sq);
    let delta_r = df64_sub(r, r0);
    let exp_term = exp_df64(df64_neg(df64_mul(a, delta_r)));

    let one = df64_from_f32(1.0);
    let two = df64_from_f32(2.0);
    let one_minus_exp = df64_sub(one, exp_term);
    let force_magnitude = df64_mul(df64_mul(df64_mul(two, D), a), df64_mul(one_minus_exp, exp_term));
    let force_over_r = df64_div(force_magnitude, r);

    let fx = df64_mul(force_over_r, dx);
    let fy = df64_mul(force_over_r, dy);
    let fz = df64_mul(force_over_r, dz);

    bond_forces[out_base] = df64_to_f64(fx);
    bond_forces[out_base + 1u] = df64_to_f64(fy);
    bond_forces[out_base + 2u] = df64_to_f64(fz);
    bond_forces[out_base + 3u] = -df64_to_f64(fx);
    bond_forces[out_base + 4u] = -df64_to_f64(fy);
    bond_forces[out_base + 5u] = -df64_to_f64(fz);

    // U = D·[1 - exp(-a(r-r₀))]²
    bond_energy[bond_idx] = df64_to_f64(df64_mul(D, df64_mul(one_minus_exp, one_minus_exp)));
}

// Reduce kernel stays in native f64 — it's just summation (memory-bound, not compute-bound)
struct ReduceParams {
    n_particles: u32,
    n_bonds: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> reduce_params: ReduceParams;
@group(0) @binding(1) var<storage, read> bond_forces_in: array<f64>;
@group(0) @binding(2) var<storage, read> bond_pairs_in: array<u32>;
@group(0) @binding(3) var<storage, read_write> particle_forces: array<f64>;

@compute @workgroup_size(256)
fn reduce_bond_forces_f64(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let particle_idx = global_id.x;
    if (particle_idx >= reduce_params.n_particles) { return; }

    var fx = particle_forces[particle_idx * 3u];
    var fy = particle_forces[particle_idx * 3u + 1u];
    var fz = particle_forces[particle_idx * 3u + 2u];

    for (var b = 0u; b < reduce_params.n_bonds; b = b + 1u) {
        let i = bond_pairs_in[b * 2u];
        let j = bond_pairs_in[b * 2u + 1u];

        if (i == particle_idx) {
            fx = fx + bond_forces_in[b * 6u];
            fy = fy + bond_forces_in[b * 6u + 1u];
            fz = fz + bond_forces_in[b * 6u + 2u];
        } else if (j == particle_idx) {
            fx = fx + bond_forces_in[b * 6u + 3u];
            fy = fy + bond_forces_in[b * 6u + 4u];
            fz = fz + bond_forces_in[b * 6u + 5u];
        }
    }

    particle_forces[particle_idx * 3u] = fx;
    particle_forces[particle_idx * 3u + 1u] = fy;
    particle_forces[particle_idx * 3u + 2u] = fz;
}

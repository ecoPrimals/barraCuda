// SPDX-License-Identifier: AGPL-3.0-or-later
// Harmonic Bond Force Kernel (f64) — CAZyme FEL Bonded Interactions
//
// **Potential**: U(r) = ½k(r - r₀)²
// **Force**: F = -k(r - r₀)·r̂
//   where k  = force constant (kJ/mol/nm² or kcal/mol/Å²)
//         r₀ = equilibrium bond length
//
// Standard harmonic bond used in GROMOS, GLYCAM, CHARMM, AMBER, OPLS.
// Two-pass design: per-bond forces → reduce to per-particle (reuses
// reduce_bond_forces_f64 from morse_f64.wgsl).

struct Params {
    n_bonds: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> positions: array<f64>;
@group(0) @binding(1) var<storage, read> bond_pairs: array<u32>;
@group(0) @binding(2) var<storage, read> force_constants: array<f64>;
@group(0) @binding(3) var<storage, read> eq_lengths: array<f64>;
@group(0) @binding(4) var<storage, read_write> bond_forces: array<f64>;
@group(0) @binding(5) var<uniform> params: Params;

fn f64_const(x: f64, c: f32) -> f64 {
    return x - x + f64(c);
}

@compute @workgroup_size(256)
fn harmonic_bond_f64(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let bond_idx = global_id.x;
    if (bond_idx >= params.n_bonds) { return; }

    let i = bond_pairs[bond_idx * 2u];
    let j = bond_pairs[bond_idx * 2u + 1u];

    let xi = positions[i * 3u];
    let yi = positions[i * 3u + 1u];
    let zi = positions[i * 3u + 2u];

    let xj = positions[j * 3u];
    let yj = positions[j * 3u + 1u];
    let zj = positions[j * 3u + 2u];

    let k = force_constants[bond_idx];
    let r0 = eq_lengths[bond_idx];

    let zero = f64_const(k, 0.0);
    let min_r_sq = f64_const(k, 1e-20);

    let dx = xj - xi;
    let dy = yj - yi;
    let dz = zj - zi;

    let r_sq = dx * dx + dy * dy + dz * dz;

    if (r_sq < min_r_sq) {
        let out_base = bond_idx * 6u;
        bond_forces[out_base] = zero;
        bond_forces[out_base + 1u] = zero;
        bond_forces[out_base + 2u] = zero;
        bond_forces[out_base + 3u] = zero;
        bond_forces[out_base + 4u] = zero;
        bond_forces[out_base + 5u] = zero;
        return;
    }

    let r = sqrt(r_sq);

    // F_i = k(r - r₀) · r̂_ij  (restoring force toward equilibrium)
    let force_magnitude = k * (r - r0);
    let force_over_r = force_magnitude / r;

    let fx_i = force_over_r * dx;
    let fy_i = force_over_r * dy;
    let fz_i = force_over_r * dz;

    let out_base = bond_idx * 6u;
    bond_forces[out_base] = fx_i;
    bond_forces[out_base + 1u] = fy_i;
    bond_forces[out_base + 2u] = fz_i;
    bond_forces[out_base + 3u] = -fx_i;
    bond_forces[out_base + 4u] = -fy_i;
    bond_forces[out_base + 5u] = -fz_i;
}

@group(0) @binding(6) var<storage, read_write> bond_energy: array<f64>;

@compute @workgroup_size(256)
fn harmonic_bond_with_energy_f64(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let bond_idx = global_id.x;
    if (bond_idx >= params.n_bonds) { return; }

    let i = bond_pairs[bond_idx * 2u];
    let j = bond_pairs[bond_idx * 2u + 1u];

    let xi = positions[i * 3u];
    let yi = positions[i * 3u + 1u];
    let zi = positions[i * 3u + 2u];

    let xj = positions[j * 3u];
    let yj = positions[j * 3u + 1u];
    let zj = positions[j * 3u + 2u];

    let k = force_constants[bond_idx];
    let r0 = eq_lengths[bond_idx];

    let zero = f64_const(k, 0.0);
    let half = f64_const(k, 0.5);
    let min_r_sq = f64_const(k, 1e-20);

    let dx = xj - xi;
    let dy = yj - yi;
    let dz = zj - zi;

    let r_sq = dx * dx + dy * dy + dz * dz;

    if (r_sq < min_r_sq) {
        let out_base = bond_idx * 6u;
        bond_forces[out_base] = zero;
        bond_forces[out_base + 1u] = zero;
        bond_forces[out_base + 2u] = zero;
        bond_forces[out_base + 3u] = zero;
        bond_forces[out_base + 4u] = zero;
        bond_forces[out_base + 5u] = zero;
        bond_energy[bond_idx] = zero;
        return;
    }

    let r = sqrt(r_sq);
    let delta_r = r - r0;

    let force_magnitude = k * delta_r;
    let force_over_r = force_magnitude / r;
    let fx_i = force_over_r * dx;
    let fy_i = force_over_r * dy;
    let fz_i = force_over_r * dz;

    let out_base = bond_idx * 6u;
    bond_forces[out_base] = fx_i;
    bond_forces[out_base + 1u] = fy_i;
    bond_forces[out_base + 2u] = fz_i;
    bond_forces[out_base + 3u] = -fx_i;
    bond_forces[out_base + 4u] = -fy_i;
    bond_forces[out_base + 5u] = -fz_i;

    // U = ½k(r - r₀)²
    bond_energy[bond_idx] = half * k * delta_r * delta_r;
}

// Reduce kernel: reuses the same layout as morse_f64.wgsl reduce_bond_forces_f64
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

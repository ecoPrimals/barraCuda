// SPDX-License-Identifier: AGPL-3.0-or-later
// Harmonic Angle Force Kernel (f64) — CAZyme FEL Bonded Interactions
//
// **Potential**: U(θ) = ½k_θ(θ - θ₀)²
// **Force**: Gradient of U w.r.t. Cartesian positions of atoms i, j, k
//   where k_θ = angular force constant (kJ/mol/rad²)
//         θ₀  = equilibrium angle (radians)
//         θ   = angle at vertex atom j in the i-j-k triple
//
// Three-body term: each angle contributes forces to 3 atoms.
// Intermediate buffer layout: [M*9] — (fx_i, fy_i, fz_i, fx_j, fy_j, fz_j, fx_k, fy_k, fz_k)

struct Params {
    n_angles: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> positions: array<f64>;
@group(0) @binding(1) var<storage, read> angle_triples: array<u32>;     // [N_angles*3] (i,j,k)
@group(0) @binding(2) var<storage, read> force_constants: array<f64>;   // [N_angles]
@group(0) @binding(3) var<storage, read> eq_angles: array<f64>;         // [N_angles] θ₀ in radians
@group(0) @binding(4) var<storage, read_write> angle_forces: array<f64>; // [N_angles*9]
@group(0) @binding(5) var<uniform> params: Params;

fn f64_const(x: f64, c: f32) -> f64 {
    return x - x + f64(c);
}

@compute @workgroup_size(256)
fn harmonic_angle_f64(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let angle_idx = global_id.x;
    if (angle_idx >= params.n_angles) { return; }

    let i = angle_triples[angle_idx * 3u];
    let j = angle_triples[angle_idx * 3u + 1u];
    let k = angle_triples[angle_idx * 3u + 2u];

    // Vectors from vertex j to endpoints
    let rji_x = positions[i * 3u] - positions[j * 3u];
    let rji_y = positions[i * 3u + 1u] - positions[j * 3u + 1u];
    let rji_z = positions[i * 3u + 2u] - positions[j * 3u + 2u];

    let rjk_x = positions[k * 3u] - positions[j * 3u];
    let rjk_y = positions[k * 3u + 1u] - positions[j * 3u + 1u];
    let rjk_z = positions[k * 3u + 2u] - positions[j * 3u + 2u];

    let rji_sq = rji_x * rji_x + rji_y * rji_y + rji_z * rji_z;
    let rjk_sq = rjk_x * rjk_x + rjk_y * rjk_y + rjk_z * rjk_z;

    let zero = f64_const(rji_sq, 0.0);
    let min_r_sq = f64_const(rji_sq, 1e-20);

    let out_base = angle_idx * 9u;
    if (rji_sq < min_r_sq || rjk_sq < min_r_sq) {
        for (var c = 0u; c < 9u; c = c + 1u) {
            angle_forces[out_base + c] = zero;
        }
        return;
    }

    let rji_inv = f64_const(rji_sq, 1.0) / sqrt(rji_sq);
    let rjk_inv = f64_const(rjk_sq, 1.0) / sqrt(rjk_sq);

    let cos_theta = (rji_x * rjk_x + rji_y * rjk_y + rji_z * rjk_z) * rji_inv * rjk_inv;

    // Clamp to [-1, 1] for numerical safety
    let one = f64_const(rji_sq, 1.0);
    let neg_one = f64_const(rji_sq, -1.0);
    var cos_clamped = cos_theta;
    if (cos_clamped > one) { cos_clamped = one; }
    if (cos_clamped < neg_one) { cos_clamped = neg_one; }

    let theta = acos(cos_clamped);

    let k_theta = force_constants[angle_idx];
    let theta_0 = eq_angles[angle_idx];

    // -dU/dθ = -k_θ(θ - θ₀)
    let neg_dU_dtheta = -k_theta * (theta - theta_0);

    // dθ/d(cos θ) = -1/sin θ
    let sin_theta_sq = one - cos_clamped * cos_clamped;
    let sin_min = f64_const(rji_sq, 1e-12);
    var sin_theta = sqrt(sin_theta_sq);
    if (sin_theta < sin_min) { sin_theta = sin_min; }

    // Prefactor: -dU/dθ · dθ/d(cos θ) = -dU/dθ · (-1/sin θ) = dU/dθ / sin θ
    let prefactor = neg_dU_dtheta / sin_theta;

    let rji_inv2 = rji_inv * rji_inv;
    let rjk_inv2 = rjk_inv * rjk_inv;

    // d(cos θ)/d(r_i) = (r_jk/(|r_ji||r_jk|)) - cos θ · (r_ji/|r_ji|²)
    // Force on i: prefactor * d(cos θ)/d(r_i)
    let fi_x = prefactor * (rjk_x * rji_inv * rjk_inv - cos_clamped * rji_x * rji_inv2);
    let fi_y = prefactor * (rjk_y * rji_inv * rjk_inv - cos_clamped * rji_y * rji_inv2);
    let fi_z = prefactor * (rjk_z * rji_inv * rjk_inv - cos_clamped * rji_z * rji_inv2);

    // d(cos θ)/d(r_k) = (r_ji/(|r_ji||r_jk|)) - cos θ · (r_jk/|r_jk|²)
    let fk_x = prefactor * (rji_x * rji_inv * rjk_inv - cos_clamped * rjk_x * rjk_inv2);
    let fk_y = prefactor * (rji_y * rji_inv * rjk_inv - cos_clamped * rjk_y * rjk_inv2);
    let fk_z = prefactor * (rji_z * rji_inv * rjk_inv - cos_clamped * rjk_z * rjk_inv2);

    // Force on j = -(F_i + F_k) by momentum conservation
    let fj_x = -(fi_x + fk_x);
    let fj_y = -(fi_y + fk_y);
    let fj_z = -(fi_z + fk_z);

    angle_forces[out_base] = fi_x;
    angle_forces[out_base + 1u] = fi_y;
    angle_forces[out_base + 2u] = fi_z;
    angle_forces[out_base + 3u] = fj_x;
    angle_forces[out_base + 4u] = fj_y;
    angle_forces[out_base + 5u] = fj_z;
    angle_forces[out_base + 6u] = fk_x;
    angle_forces[out_base + 7u] = fk_y;
    angle_forces[out_base + 8u] = fk_z;
}

@group(0) @binding(6) var<storage, read_write> angle_energy: array<f64>;

@compute @workgroup_size(256)
fn harmonic_angle_with_energy_f64(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let angle_idx = global_id.x;
    if (angle_idx >= params.n_angles) { return; }

    let i = angle_triples[angle_idx * 3u];
    let j = angle_triples[angle_idx * 3u + 1u];
    let k = angle_triples[angle_idx * 3u + 2u];

    let rji_x = positions[i * 3u] - positions[j * 3u];
    let rji_y = positions[i * 3u + 1u] - positions[j * 3u + 1u];
    let rji_z = positions[i * 3u + 2u] - positions[j * 3u + 2u];

    let rjk_x = positions[k * 3u] - positions[j * 3u];
    let rjk_y = positions[k * 3u + 1u] - positions[j * 3u + 1u];
    let rjk_z = positions[k * 3u + 2u] - positions[j * 3u + 2u];

    let rji_sq = rji_x * rji_x + rji_y * rji_y + rji_z * rji_z;
    let rjk_sq = rjk_x * rjk_x + rjk_y * rjk_y + rjk_z * rjk_z;

    let zero = f64_const(rji_sq, 0.0);
    let min_r_sq = f64_const(rji_sq, 1e-20);
    let out_base = angle_idx * 9u;

    if (rji_sq < min_r_sq || rjk_sq < min_r_sq) {
        for (var c = 0u; c < 9u; c = c + 1u) {
            angle_forces[out_base + c] = zero;
        }
        angle_energy[angle_idx] = zero;
        return;
    }

    let rji_inv = f64_const(rji_sq, 1.0) / sqrt(rji_sq);
    let rjk_inv = f64_const(rjk_sq, 1.0) / sqrt(rjk_sq);

    let cos_theta = (rji_x * rjk_x + rji_y * rjk_y + rji_z * rjk_z) * rji_inv * rjk_inv;

    let one = f64_const(rji_sq, 1.0);
    let neg_one = f64_const(rji_sq, -1.0);
    var cos_clamped = cos_theta;
    if (cos_clamped > one) { cos_clamped = one; }
    if (cos_clamped < neg_one) { cos_clamped = neg_one; }

    let theta = acos(cos_clamped);
    let k_theta = force_constants[angle_idx];
    let theta_0 = eq_angles[angle_idx];
    let delta_theta = theta - theta_0;

    let neg_dU_dtheta = -k_theta * delta_theta;

    let sin_theta_sq = one - cos_clamped * cos_clamped;
    let sin_min = f64_const(rji_sq, 1e-12);
    var sin_theta = sqrt(sin_theta_sq);
    if (sin_theta < sin_min) { sin_theta = sin_min; }

    let prefactor = neg_dU_dtheta / sin_theta;

    let rji_inv2 = rji_inv * rji_inv;
    let rjk_inv2 = rjk_inv * rjk_inv;

    let fi_x = prefactor * (rjk_x * rji_inv * rjk_inv - cos_clamped * rji_x * rji_inv2);
    let fi_y = prefactor * (rjk_y * rji_inv * rjk_inv - cos_clamped * rji_y * rji_inv2);
    let fi_z = prefactor * (rjk_z * rji_inv * rjk_inv - cos_clamped * rji_z * rji_inv2);

    let fk_x = prefactor * (rji_x * rji_inv * rjk_inv - cos_clamped * rjk_x * rjk_inv2);
    let fk_y = prefactor * (rji_y * rji_inv * rjk_inv - cos_clamped * rjk_y * rjk_inv2);
    let fk_z = prefactor * (rji_z * rji_inv * rjk_inv - cos_clamped * rjk_z * rjk_inv2);

    let fj_x = -(fi_x + fk_x);
    let fj_y = -(fi_y + fk_y);
    let fj_z = -(fi_z + fk_z);

    angle_forces[out_base] = fi_x;
    angle_forces[out_base + 1u] = fi_y;
    angle_forces[out_base + 2u] = fi_z;
    angle_forces[out_base + 3u] = fj_x;
    angle_forces[out_base + 4u] = fj_y;
    angle_forces[out_base + 5u] = fj_z;
    angle_forces[out_base + 6u] = fk_x;
    angle_forces[out_base + 7u] = fk_y;
    angle_forces[out_base + 8u] = fk_z;

    // U = ½k_θ(θ - θ₀)²
    let half = f64_const(rji_sq, 0.5);
    angle_energy[angle_idx] = half * k_theta * delta_theta * delta_theta;
}

// 3-body reduce: scatter angle forces to per-particle forces
struct ReduceParams {
    n_particles: u32,
    n_angles: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> reduce_params: ReduceParams;
@group(0) @binding(1) var<storage, read> angle_forces_in: array<f64>;  // [N_angles*9]
@group(0) @binding(2) var<storage, read> angle_triples_in: array<u32>; // [N_angles*3]
@group(0) @binding(3) var<storage, read_write> particle_forces: array<f64>;

@compute @workgroup_size(256)
fn reduce_angle_forces_f64(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let particle_idx = global_id.x;
    if (particle_idx >= reduce_params.n_particles) { return; }

    var fx = particle_forces[particle_idx * 3u];
    var fy = particle_forces[particle_idx * 3u + 1u];
    var fz = particle_forces[particle_idx * 3u + 2u];

    for (var a = 0u; a < reduce_params.n_angles; a = a + 1u) {
        let ai = angle_triples_in[a * 3u];
        let aj = angle_triples_in[a * 3u + 1u];
        let ak = angle_triples_in[a * 3u + 2u];

        if (ai == particle_idx) {
            fx = fx + angle_forces_in[a * 9u];
            fy = fy + angle_forces_in[a * 9u + 1u];
            fz = fz + angle_forces_in[a * 9u + 2u];
        }
        if (aj == particle_idx) {
            fx = fx + angle_forces_in[a * 9u + 3u];
            fy = fy + angle_forces_in[a * 9u + 4u];
            fz = fz + angle_forces_in[a * 9u + 5u];
        }
        if (ak == particle_idx) {
            fx = fx + angle_forces_in[a * 9u + 6u];
            fy = fy + angle_forces_in[a * 9u + 7u];
            fz = fz + angle_forces_in[a * 9u + 8u];
        }
    }

    particle_forces[particle_idx * 3u] = fx;
    particle_forces[particle_idx * 3u + 1u] = fy;
    particle_forces[particle_idx * 3u + 2u] = fz;
}

// SPDX-License-Identifier: AGPL-3.0-or-later
// Improper Dihedral Force Kernel (f64) — CAZyme FEL Bonded Interactions
//
// **Potential**: U(ψ) = ½k_ψ(ψ - ψ₀)²
// **Force**: Cartesian gradient over atoms i, j, k, l
//   where k_ψ = force constant (kJ/mol/rad²)
//         ψ₀  = equilibrium improper angle (radians)
//         ψ   = improper dihedral angle (same geometry as proper dihedral
//               but with harmonic instead of periodic potential)
//
// Used in GROMOS for planarity restraints (sp2 centers, aromatic rings).
// Same Blondel-Karplus decomposition as periodic dihedral.
// Four-body term. Intermediate buffer: [M*12].

struct Params {
    n_impropers: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> positions: array<f64>;
@group(0) @binding(1) var<storage, read> improper_quads: array<u32>;    // [N*4] (i,j,k,l)
@group(0) @binding(2) var<storage, read> force_constants: array<f64>;   // [N] k_ψ
@group(0) @binding(3) var<storage, read> eq_angles: array<f64>;         // [N] ψ₀
@group(0) @binding(4) var<storage, read_write> improper_forces: array<f64>; // [N*12]
@group(0) @binding(5) var<uniform> params: Params;

fn f64_const(x: f64, c: f32) -> f64 {
    return x - x + f64(c);
}

fn cross_x(ax: f64, ay: f64, az: f64, bx: f64, by: f64, bz: f64) -> f64 {
    return ay * bz - az * by;
}
fn cross_y(ax: f64, ay: f64, az: f64, bx: f64, by: f64, bz: f64) -> f64 {
    return az * bx - ax * bz;
}
fn cross_z(ax: f64, ay: f64, az: f64, bx: f64, by: f64, bz: f64) -> f64 {
    return ax * by - ay * bx;
}

@compute @workgroup_size(256)
fn improper_dihedral_f64(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.n_impropers) { return; }

    let ai = improper_quads[idx * 4u];
    let aj = improper_quads[idx * 4u + 1u];
    let ak = improper_quads[idx * 4u + 2u];
    let al = improper_quads[idx * 4u + 3u];

    let r_ij_x = positions[aj * 3u] - positions[ai * 3u];
    let r_ij_y = positions[aj * 3u + 1u] - positions[ai * 3u + 1u];
    let r_ij_z = positions[aj * 3u + 2u] - positions[ai * 3u + 2u];

    let r_jk_x = positions[ak * 3u] - positions[aj * 3u];
    let r_jk_y = positions[ak * 3u + 1u] - positions[aj * 3u + 1u];
    let r_jk_z = positions[ak * 3u + 2u] - positions[aj * 3u + 2u];

    let r_kl_x = positions[al * 3u] - positions[ak * 3u];
    let r_kl_y = positions[al * 3u + 1u] - positions[ak * 3u + 1u];
    let r_kl_z = positions[al * 3u + 2u] - positions[ak * 3u + 2u];

    let mx = cross_x(r_ij_x, r_ij_y, r_ij_z, r_jk_x, r_jk_y, r_jk_z);
    let my = cross_y(r_ij_x, r_ij_y, r_ij_z, r_jk_x, r_jk_y, r_jk_z);
    let mz = cross_z(r_ij_x, r_ij_y, r_ij_z, r_jk_x, r_jk_y, r_jk_z);

    let nx = cross_x(r_jk_x, r_jk_y, r_jk_z, r_kl_x, r_kl_y, r_kl_z);
    let ny = cross_y(r_jk_x, r_jk_y, r_jk_z, r_kl_x, r_kl_y, r_kl_z);
    let nz = cross_z(r_jk_x, r_jk_y, r_jk_z, r_kl_x, r_kl_y, r_kl_z);

    let m_sq = mx * mx + my * my + mz * mz;
    let n_sq = nx * nx + ny * ny + nz * nz;

    let zero = f64_const(m_sq, 0.0);
    let min_sq = f64_const(m_sq, 1e-20);
    let out_base = idx * 12u;

    if (m_sq < min_sq || n_sq < min_sq) {
        for (var c = 0u; c < 12u; c = c + 1u) {
            improper_forces[out_base + c] = zero;
        }
        return;
    }

    let m_inv = f64_const(m_sq, 1.0) / sqrt(m_sq);
    let n_inv = f64_const(n_sq, 1.0) / sqrt(n_sq);

    let cos_psi = (mx * nx + my * ny + mz * nz) * m_inv * n_inv;

    let r_jk_sq = r_jk_x * r_jk_x + r_jk_y * r_jk_y + r_jk_z * r_jk_z;
    let r_jk_inv = f64_const(r_jk_sq, 1.0) / sqrt(r_jk_sq);

    let mxn_x = cross_x(mx, my, mz, nx, ny, nz);
    let mxn_y = cross_y(mx, my, mz, nx, ny, nz);
    let mxn_z = cross_z(mx, my, mz, nx, ny, nz);
    let sin_psi = (mxn_x * r_jk_x + mxn_y * r_jk_y + mxn_z * r_jk_z) * r_jk_inv * m_inv * n_inv;

    let psi = atan2(sin_psi, cos_psi);

    let k_psi = force_constants[idx];
    let psi_0 = eq_angles[idx];

    // -dU/dψ = -k_ψ(ψ - ψ₀)
    let neg_dU_dpsi = -k_psi * (psi - psi_0);

    // Blondel-Karplus decomposition (same as periodic dihedral)
    let r_jk_len = sqrt(r_jk_sq);
    let m_sq_inv = f64_const(m_sq, 1.0) / m_sq;
    let n_sq_inv = f64_const(n_sq, 1.0) / n_sq;

    let dpsi_di_scale = r_jk_len * m_sq_inv;
    let dpsi_di_x = dpsi_di_scale * mx;
    let dpsi_di_y = dpsi_di_scale * my;
    let dpsi_di_z = dpsi_di_scale * mz;

    let dpsi_dl_scale = -r_jk_len * n_sq_inv;
    let dpsi_dl_x = dpsi_dl_scale * nx;
    let dpsi_dl_y = dpsi_dl_scale * ny;
    let dpsi_dl_z = dpsi_dl_scale * nz;

    let r_jk_sq_inv = f64_const(r_jk_sq, 1.0) / r_jk_sq;
    let dot_ij_jk = r_ij_x * r_jk_x + r_ij_y * r_jk_y + r_ij_z * r_jk_z;
    let dot_kl_jk = r_kl_x * r_jk_x + r_kl_y * r_jk_y + r_kl_z * r_jk_z;

    let coeff_j_i = dot_ij_jk * r_jk_sq_inv - f64_const(m_sq, 1.0);
    let coeff_j_l = -dot_kl_jk * r_jk_sq_inv;

    let dpsi_dj_x = coeff_j_i * dpsi_di_x + coeff_j_l * dpsi_dl_x;
    let dpsi_dj_y = coeff_j_i * dpsi_di_y + coeff_j_l * dpsi_dl_y;
    let dpsi_dj_z = coeff_j_i * dpsi_di_z + coeff_j_l * dpsi_dl_z;

    let coeff_k_l = dot_kl_jk * r_jk_sq_inv - f64_const(n_sq, 1.0);
    let coeff_k_i = -dot_ij_jk * r_jk_sq_inv;

    let dpsi_dk_x = coeff_k_l * dpsi_dl_x + coeff_k_i * dpsi_di_x;
    let dpsi_dk_y = coeff_k_l * dpsi_dl_y + coeff_k_i * dpsi_di_y;
    let dpsi_dk_z = coeff_k_l * dpsi_dl_z + coeff_k_i * dpsi_di_z;

    improper_forces[out_base] = neg_dU_dpsi * dpsi_di_x;
    improper_forces[out_base + 1u] = neg_dU_dpsi * dpsi_di_y;
    improper_forces[out_base + 2u] = neg_dU_dpsi * dpsi_di_z;
    improper_forces[out_base + 3u] = neg_dU_dpsi * dpsi_dj_x;
    improper_forces[out_base + 4u] = neg_dU_dpsi * dpsi_dj_y;
    improper_forces[out_base + 5u] = neg_dU_dpsi * dpsi_dj_z;
    improper_forces[out_base + 6u] = neg_dU_dpsi * dpsi_dk_x;
    improper_forces[out_base + 7u] = neg_dU_dpsi * dpsi_dk_y;
    improper_forces[out_base + 8u] = neg_dU_dpsi * dpsi_dk_z;
    improper_forces[out_base + 9u] = neg_dU_dpsi * dpsi_dl_x;
    improper_forces[out_base + 10u] = neg_dU_dpsi * dpsi_dl_y;
    improper_forces[out_base + 11u] = neg_dU_dpsi * dpsi_dl_z;
}

@group(0) @binding(6) var<storage, read_write> improper_energy: array<f64>;

@compute @workgroup_size(256)
fn improper_with_energy_f64(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.n_impropers) { return; }

    let ai = improper_quads[idx * 4u];
    let aj = improper_quads[idx * 4u + 1u];
    let ak = improper_quads[idx * 4u + 2u];
    let al = improper_quads[idx * 4u + 3u];

    let r_ij_x = positions[aj * 3u] - positions[ai * 3u];
    let r_ij_y = positions[aj * 3u + 1u] - positions[ai * 3u + 1u];
    let r_ij_z = positions[aj * 3u + 2u] - positions[ai * 3u + 2u];

    let r_jk_x = positions[ak * 3u] - positions[aj * 3u];
    let r_jk_y = positions[ak * 3u + 1u] - positions[aj * 3u + 1u];
    let r_jk_z = positions[ak * 3u + 2u] - positions[aj * 3u + 2u];

    let r_kl_x = positions[al * 3u] - positions[ak * 3u];
    let r_kl_y = positions[al * 3u + 1u] - positions[ak * 3u + 1u];
    let r_kl_z = positions[al * 3u + 2u] - positions[ak * 3u + 2u];

    let mx = cross_x(r_ij_x, r_ij_y, r_ij_z, r_jk_x, r_jk_y, r_jk_z);
    let my = cross_y(r_ij_x, r_ij_y, r_ij_z, r_jk_x, r_jk_y, r_jk_z);
    let mz = cross_z(r_ij_x, r_ij_y, r_ij_z, r_jk_x, r_jk_y, r_jk_z);

    let nx = cross_x(r_jk_x, r_jk_y, r_jk_z, r_kl_x, r_kl_y, r_kl_z);
    let ny = cross_y(r_jk_x, r_jk_y, r_jk_z, r_kl_x, r_kl_y, r_kl_z);
    let nz = cross_z(r_jk_x, r_jk_y, r_jk_z, r_kl_x, r_kl_y, r_kl_z);

    let m_sq = mx * mx + my * my + mz * mz;
    let n_sq = nx * nx + ny * ny + nz * nz;

    let zero = f64_const(m_sq, 0.0);
    let min_sq = f64_const(m_sq, 1e-20);
    let out_base = idx * 12u;

    if (m_sq < min_sq || n_sq < min_sq) {
        for (var c = 0u; c < 12u; c = c + 1u) {
            improper_forces[out_base + c] = zero;
        }
        improper_energy[idx] = zero;
        return;
    }

    let m_inv = f64_const(m_sq, 1.0) / sqrt(m_sq);
    let n_inv = f64_const(n_sq, 1.0) / sqrt(n_sq);

    let cos_psi = (mx * nx + my * ny + mz * nz) * m_inv * n_inv;

    let r_jk_sq = r_jk_x * r_jk_x + r_jk_y * r_jk_y + r_jk_z * r_jk_z;
    let r_jk_inv = f64_const(r_jk_sq, 1.0) / sqrt(r_jk_sq);

    let mxn_x = cross_x(mx, my, mz, nx, ny, nz);
    let mxn_y = cross_y(mx, my, mz, nx, ny, nz);
    let mxn_z = cross_z(mx, my, mz, nx, ny, nz);
    let sin_psi = (mxn_x * r_jk_x + mxn_y * r_jk_y + mxn_z * r_jk_z) * r_jk_inv * m_inv * n_inv;

    let psi = atan2(sin_psi, cos_psi);

    let k_psi = force_constants[idx];
    let psi_0 = eq_angles[idx];
    let delta_psi = psi - psi_0;

    let neg_dU_dpsi = -k_psi * delta_psi;

    let r_jk_len = sqrt(r_jk_sq);
    let m_sq_inv = f64_const(m_sq, 1.0) / m_sq;
    let n_sq_inv = f64_const(n_sq, 1.0) / n_sq;

    let dpsi_di_scale = r_jk_len * m_sq_inv;
    let dpsi_di_x = dpsi_di_scale * mx;
    let dpsi_di_y = dpsi_di_scale * my;
    let dpsi_di_z = dpsi_di_scale * mz;

    let dpsi_dl_scale = -r_jk_len * n_sq_inv;
    let dpsi_dl_x = dpsi_dl_scale * nx;
    let dpsi_dl_y = dpsi_dl_scale * ny;
    let dpsi_dl_z = dpsi_dl_scale * nz;

    let r_jk_sq_inv = f64_const(r_jk_sq, 1.0) / r_jk_sq;
    let dot_ij_jk = r_ij_x * r_jk_x + r_ij_y * r_jk_y + r_ij_z * r_jk_z;
    let dot_kl_jk = r_kl_x * r_jk_x + r_kl_y * r_jk_y + r_kl_z * r_jk_z;

    let coeff_j_i = dot_ij_jk * r_jk_sq_inv - f64_const(m_sq, 1.0);
    let coeff_j_l = -dot_kl_jk * r_jk_sq_inv;

    let dpsi_dj_x = coeff_j_i * dpsi_di_x + coeff_j_l * dpsi_dl_x;
    let dpsi_dj_y = coeff_j_i * dpsi_di_y + coeff_j_l * dpsi_dl_y;
    let dpsi_dj_z = coeff_j_i * dpsi_di_z + coeff_j_l * dpsi_dl_z;

    let coeff_k_l = dot_kl_jk * r_jk_sq_inv - f64_const(n_sq, 1.0);
    let coeff_k_i = -dot_ij_jk * r_jk_sq_inv;

    let dpsi_dk_x = coeff_k_l * dpsi_dl_x + coeff_k_i * dpsi_di_x;
    let dpsi_dk_y = coeff_k_l * dpsi_dl_y + coeff_k_i * dpsi_di_y;
    let dpsi_dk_z = coeff_k_l * dpsi_dl_z + coeff_k_i * dpsi_di_z;

    improper_forces[out_base] = neg_dU_dpsi * dpsi_di_x;
    improper_forces[out_base + 1u] = neg_dU_dpsi * dpsi_di_y;
    improper_forces[out_base + 2u] = neg_dU_dpsi * dpsi_di_z;
    improper_forces[out_base + 3u] = neg_dU_dpsi * dpsi_dj_x;
    improper_forces[out_base + 4u] = neg_dU_dpsi * dpsi_dj_y;
    improper_forces[out_base + 5u] = neg_dU_dpsi * dpsi_dj_z;
    improper_forces[out_base + 6u] = neg_dU_dpsi * dpsi_dk_x;
    improper_forces[out_base + 7u] = neg_dU_dpsi * dpsi_dk_y;
    improper_forces[out_base + 8u] = neg_dU_dpsi * dpsi_dk_z;
    improper_forces[out_base + 9u] = neg_dU_dpsi * dpsi_dl_x;
    improper_forces[out_base + 10u] = neg_dU_dpsi * dpsi_dl_y;
    improper_forces[out_base + 11u] = neg_dU_dpsi * dpsi_dl_z;

    // U = ½k_ψ(ψ - ψ₀)²
    let half = f64_const(m_sq, 0.5);
    improper_energy[idx] = half * k_psi * delta_psi * delta_psi;
}

// 4-body reduce (same as dihedral)
struct ReduceParams {
    n_particles: u32,
    n_impropers: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> reduce_params: ReduceParams;
@group(0) @binding(1) var<storage, read> improper_forces_in: array<f64>;
@group(0) @binding(2) var<storage, read> improper_quads_in: array<u32>;
@group(0) @binding(3) var<storage, read_write> particle_forces: array<f64>;

@compute @workgroup_size(256)
fn reduce_improper_forces_f64(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let particle_idx = global_id.x;
    if (particle_idx >= reduce_params.n_particles) { return; }

    var fx = particle_forces[particle_idx * 3u];
    var fy = particle_forces[particle_idx * 3u + 1u];
    var fz = particle_forces[particle_idx * 3u + 2u];

    for (var d = 0u; d < reduce_params.n_impropers; d = d + 1u) {
        for (var slot = 0u; slot < 4u; slot = slot + 1u) {
            if (improper_quads_in[d * 4u + slot] == particle_idx) {
                let base = d * 12u + slot * 3u;
                fx = fx + improper_forces_in[base];
                fy = fy + improper_forces_in[base + 1u];
                fz = fz + improper_forces_in[base + 2u];
            }
        }
    }

    particle_forces[particle_idx * 3u] = fx;
    particle_forces[particle_idx * 3u + 1u] = fy;
    particle_forces[particle_idx * 3u + 2u] = fz;
}

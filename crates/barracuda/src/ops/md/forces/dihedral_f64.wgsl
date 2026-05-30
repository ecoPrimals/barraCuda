// SPDX-License-Identifier: AGPL-3.0-or-later
// Periodic Dihedral Torsion Force Kernel (f64) — CAZyme FEL Bonded Interactions
//
// **Potential**: U(φ) = k_φ [1 + cos(nφ - δ)]
// **Force**: Cartesian gradient over atoms i, j, k, l
//   where k_φ = barrier height (kJ/mol)
//         n   = periodicity (integer, stored as f64)
//         δ   = phase shift (radians)
//         φ   = dihedral angle defined by planes (i,j,k) and (j,k,l)
//
// Four-body term. Intermediate buffer: [M*12] — 4 atoms × 3 dims.
// Dihedral computed via cross-product normals m = r_ij × r_jk, n = r_jk × r_kl.

struct Params {
    n_dihedrals: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> positions: array<f64>;
@group(0) @binding(1) var<storage, read> dihedral_quads: array<u32>;    // [N*4] (i,j,k,l)
@group(0) @binding(2) var<storage, read> barrier_height: array<f64>;    // [N] k_φ
@group(0) @binding(3) var<storage, read> periodicity: array<f64>;       // [N] n (integer stored as f64)
@group(0) @binding(4) var<storage, read> phase_shift: array<f64>;       // [N] δ
@group(0) @binding(5) var<storage, read_write> dihedral_forces: array<f64>; // [N*12]
@group(0) @binding(6) var<uniform> params: Params;

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
fn dihedral_torsion_f64(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.n_dihedrals) { return; }

    let ai = dihedral_quads[idx * 4u];
    let aj = dihedral_quads[idx * 4u + 1u];
    let ak = dihedral_quads[idx * 4u + 2u];
    let al = dihedral_quads[idx * 4u + 3u];

    // Bond vectors
    let r_ij_x = positions[aj * 3u] - positions[ai * 3u];
    let r_ij_y = positions[aj * 3u + 1u] - positions[ai * 3u + 1u];
    let r_ij_z = positions[aj * 3u + 2u] - positions[ai * 3u + 2u];

    let r_jk_x = positions[ak * 3u] - positions[aj * 3u];
    let r_jk_y = positions[ak * 3u + 1u] - positions[aj * 3u + 1u];
    let r_jk_z = positions[ak * 3u + 2u] - positions[aj * 3u + 2u];

    let r_kl_x = positions[al * 3u] - positions[ak * 3u];
    let r_kl_y = positions[al * 3u + 1u] - positions[ak * 3u + 1u];
    let r_kl_z = positions[al * 3u + 2u] - positions[ak * 3u + 2u];

    // Normal vectors: m = r_ij × r_jk, n = r_jk × r_kl
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
            dihedral_forces[out_base + c] = zero;
        }
        return;
    }

    let m_inv = f64_const(m_sq, 1.0) / sqrt(m_sq);
    let n_inv = f64_const(n_sq, 1.0) / sqrt(n_sq);

    // cos φ = (m · n) / (|m| |n|)
    let cos_phi = (mx * nx + my * ny + mz * nz) * m_inv * n_inv;

    // sin φ sign via (m × n) · r_jk_hat
    let r_jk_sq = r_jk_x * r_jk_x + r_jk_y * r_jk_y + r_jk_z * r_jk_z;
    let r_jk_inv = f64_const(r_jk_sq, 1.0) / sqrt(r_jk_sq);

    let mxn_x = cross_x(mx, my, mz, nx, ny, nz);
    let mxn_y = cross_y(mx, my, mz, nx, ny, nz);
    let mxn_z = cross_z(mx, my, mz, nx, ny, nz);
    let sin_phi = (mxn_x * r_jk_x + mxn_y * r_jk_y + mxn_z * r_jk_z) * r_jk_inv * m_inv * n_inv;

    let phi = atan2(sin_phi, cos_phi);

    let kd = barrier_height[idx];
    let n_period = periodicity[idx];
    let delta = phase_shift[idx];

    // -dU/dφ = k_φ · n · sin(nφ - δ)
    let neg_dU_dphi = kd * n_period * sin(n_period * phi - delta);

    // Derivatives of φ w.r.t. Cartesian coords using the Blondel-Karplus formulation:
    //   dφ/dr_i =  (|r_jk| / m²) · m
    //   dφ/dr_l = -(|r_jk| / n²) · n
    //   dφ/dr_j = [(r_ij · r_jk)/|r_jk|² - 1] · dφ/dr_i - [(r_kl · r_jk)/|r_jk|²] · dφ/dr_l
    //   dφ/dr_k = [(r_kl · r_jk)/|r_jk|² - 1] · dφ/dr_l - [(r_ij · r_jk)/|r_jk|²] · dφ/dr_i

    let r_jk_len = sqrt(r_jk_sq);
    let m_sq_inv = f64_const(m_sq, 1.0) / m_sq;
    let n_sq_inv = f64_const(n_sq, 1.0) / n_sq;

    // dφ/dr_i = (|r_jk|/m²) · m
    let dphi_di_scale = r_jk_len * m_sq_inv;
    let dphi_di_x = dphi_di_scale * mx;
    let dphi_di_y = dphi_di_scale * my;
    let dphi_di_z = dphi_di_scale * mz;

    // dφ/dr_l = -(|r_jk|/n²) · n
    let dphi_dl_scale = -r_jk_len * n_sq_inv;
    let dphi_dl_x = dphi_dl_scale * nx;
    let dphi_dl_y = dphi_dl_scale * ny;
    let dphi_dl_z = dphi_dl_scale * nz;

    let r_jk_sq_inv = f64_const(r_jk_sq, 1.0) / r_jk_sq;
    let dot_ij_jk = r_ij_x * r_jk_x + r_ij_y * r_jk_y + r_ij_z * r_jk_z;
    let dot_kl_jk = r_kl_x * r_jk_x + r_kl_y * r_jk_y + r_kl_z * r_jk_z;

    let coeff_j_i = dot_ij_jk * r_jk_sq_inv - f64_const(m_sq, 1.0);
    let coeff_j_l = -dot_kl_jk * r_jk_sq_inv;

    let dphi_dj_x = coeff_j_i * dphi_di_x + coeff_j_l * dphi_dl_x;
    let dphi_dj_y = coeff_j_i * dphi_di_y + coeff_j_l * dphi_dl_y;
    let dphi_dj_z = coeff_j_i * dphi_di_z + coeff_j_l * dphi_dl_z;

    let coeff_k_l = dot_kl_jk * r_jk_sq_inv - f64_const(n_sq, 1.0);
    let coeff_k_i = -dot_ij_jk * r_jk_sq_inv;

    let dphi_dk_x = coeff_k_l * dphi_dl_x + coeff_k_i * dphi_di_x;
    let dphi_dk_y = coeff_k_l * dphi_dl_y + coeff_k_i * dphi_di_y;
    let dphi_dk_z = coeff_k_l * dphi_dl_z + coeff_k_i * dphi_di_z;

    // F_atom = -dU/dφ · dφ/dr_atom
    dihedral_forces[out_base] = neg_dU_dphi * dphi_di_x;
    dihedral_forces[out_base + 1u] = neg_dU_dphi * dphi_di_y;
    dihedral_forces[out_base + 2u] = neg_dU_dphi * dphi_di_z;
    dihedral_forces[out_base + 3u] = neg_dU_dphi * dphi_dj_x;
    dihedral_forces[out_base + 4u] = neg_dU_dphi * dphi_dj_y;
    dihedral_forces[out_base + 5u] = neg_dU_dphi * dphi_dj_z;
    dihedral_forces[out_base + 6u] = neg_dU_dphi * dphi_dk_x;
    dihedral_forces[out_base + 7u] = neg_dU_dphi * dphi_dk_y;
    dihedral_forces[out_base + 8u] = neg_dU_dphi * dphi_dk_z;
    dihedral_forces[out_base + 9u] = neg_dU_dphi * dphi_dl_x;
    dihedral_forces[out_base + 10u] = neg_dU_dphi * dphi_dl_y;
    dihedral_forces[out_base + 11u] = neg_dU_dphi * dphi_dl_z;
}

@group(0) @binding(7) var<storage, read_write> dihedral_energy: array<f64>;

@compute @workgroup_size(256)
fn dihedral_with_energy_f64(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.n_dihedrals) { return; }

    let ai = dihedral_quads[idx * 4u];
    let aj = dihedral_quads[idx * 4u + 1u];
    let ak = dihedral_quads[idx * 4u + 2u];
    let al = dihedral_quads[idx * 4u + 3u];

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
            dihedral_forces[out_base + c] = zero;
        }
        dihedral_energy[idx] = zero;
        return;
    }

    let m_inv = f64_const(m_sq, 1.0) / sqrt(m_sq);
    let n_inv = f64_const(n_sq, 1.0) / sqrt(n_sq);

    let cos_phi = (mx * nx + my * ny + mz * nz) * m_inv * n_inv;

    let r_jk_sq = r_jk_x * r_jk_x + r_jk_y * r_jk_y + r_jk_z * r_jk_z;
    let r_jk_inv = f64_const(r_jk_sq, 1.0) / sqrt(r_jk_sq);

    let mxn_x = cross_x(mx, my, mz, nx, ny, nz);
    let mxn_y = cross_y(mx, my, mz, nx, ny, nz);
    let mxn_z = cross_z(mx, my, mz, nx, ny, nz);
    let sin_phi = (mxn_x * r_jk_x + mxn_y * r_jk_y + mxn_z * r_jk_z) * r_jk_inv * m_inv * n_inv;

    let phi = atan2(sin_phi, cos_phi);

    let kd = barrier_height[idx];
    let n_period = periodicity[idx];
    let delta = phase_shift[idx];

    let neg_dU_dphi = kd * n_period * sin(n_period * phi - delta);

    let r_jk_len = sqrt(r_jk_sq);
    let m_sq_inv = f64_const(m_sq, 1.0) / m_sq;
    let n_sq_inv = f64_const(n_sq, 1.0) / n_sq;

    let dphi_di_scale = r_jk_len * m_sq_inv;
    let dphi_di_x = dphi_di_scale * mx;
    let dphi_di_y = dphi_di_scale * my;
    let dphi_di_z = dphi_di_scale * mz;

    let dphi_dl_scale = -r_jk_len * n_sq_inv;
    let dphi_dl_x = dphi_dl_scale * nx;
    let dphi_dl_y = dphi_dl_scale * ny;
    let dphi_dl_z = dphi_dl_scale * nz;

    let r_jk_sq_inv = f64_const(r_jk_sq, 1.0) / r_jk_sq;
    let dot_ij_jk = r_ij_x * r_jk_x + r_ij_y * r_jk_y + r_ij_z * r_jk_z;
    let dot_kl_jk = r_kl_x * r_jk_x + r_kl_y * r_jk_y + r_kl_z * r_jk_z;

    let coeff_j_i = dot_ij_jk * r_jk_sq_inv - f64_const(m_sq, 1.0);
    let coeff_j_l = -dot_kl_jk * r_jk_sq_inv;

    let dphi_dj_x = coeff_j_i * dphi_di_x + coeff_j_l * dphi_dl_x;
    let dphi_dj_y = coeff_j_i * dphi_di_y + coeff_j_l * dphi_dl_y;
    let dphi_dj_z = coeff_j_i * dphi_di_z + coeff_j_l * dphi_dl_z;

    let coeff_k_l = dot_kl_jk * r_jk_sq_inv - f64_const(n_sq, 1.0);
    let coeff_k_i = -dot_ij_jk * r_jk_sq_inv;

    let dphi_dk_x = coeff_k_l * dphi_dl_x + coeff_k_i * dphi_di_x;
    let dphi_dk_y = coeff_k_l * dphi_dl_y + coeff_k_i * dphi_di_y;
    let dphi_dk_z = coeff_k_l * dphi_dl_z + coeff_k_i * dphi_di_z;

    dihedral_forces[out_base] = neg_dU_dphi * dphi_di_x;
    dihedral_forces[out_base + 1u] = neg_dU_dphi * dphi_di_y;
    dihedral_forces[out_base + 2u] = neg_dU_dphi * dphi_di_z;
    dihedral_forces[out_base + 3u] = neg_dU_dphi * dphi_dj_x;
    dihedral_forces[out_base + 4u] = neg_dU_dphi * dphi_dj_y;
    dihedral_forces[out_base + 5u] = neg_dU_dphi * dphi_dj_z;
    dihedral_forces[out_base + 6u] = neg_dU_dphi * dphi_dk_x;
    dihedral_forces[out_base + 7u] = neg_dU_dphi * dphi_dk_y;
    dihedral_forces[out_base + 8u] = neg_dU_dphi * dphi_dk_z;
    dihedral_forces[out_base + 9u] = neg_dU_dphi * dphi_dl_x;
    dihedral_forces[out_base + 10u] = neg_dU_dphi * dphi_dl_y;
    dihedral_forces[out_base + 11u] = neg_dU_dphi * dphi_dl_z;

    // U = k_φ [1 + cos(nφ - δ)]
    dihedral_energy[idx] = kd * (f64_const(m_sq, 1.0) + cos(n_period * phi - delta));
}

// 4-body reduce: scatter dihedral forces to per-particle forces
struct ReduceParams {
    n_particles: u32,
    n_dihedrals: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<uniform> reduce_params: ReduceParams;
@group(0) @binding(1) var<storage, read> dihedral_forces_in: array<f64>;  // [N*12]
@group(0) @binding(2) var<storage, read> dihedral_quads_in: array<u32>;   // [N*4]
@group(0) @binding(3) var<storage, read_write> particle_forces: array<f64>;

@compute @workgroup_size(256)
fn reduce_dihedral_forces_f64(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let particle_idx = global_id.x;
    if (particle_idx >= reduce_params.n_particles) { return; }

    var fx = particle_forces[particle_idx * 3u];
    var fy = particle_forces[particle_idx * 3u + 1u];
    var fz = particle_forces[particle_idx * 3u + 2u];

    for (var d = 0u; d < reduce_params.n_dihedrals; d = d + 1u) {
        for (var slot = 0u; slot < 4u; slot = slot + 1u) {
            if (dihedral_quads_in[d * 4u + slot] == particle_idx) {
                let base = d * 12u + slot * 3u;
                fx = fx + dihedral_forces_in[base];
                fy = fy + dihedral_forces_in[base + 1u];
                fz = fz + dihedral_forces_in[base + 2u];
            }
        }
    }

    particle_forces[particle_idx * 3u] = fx;
    particle_forces[particle_idx * 3u + 1u] = fy;
    particle_forces[particle_idx * 3u + 2u] = fz;
}

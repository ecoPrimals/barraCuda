// SPDX-License-Identifier: AGPL-3.0-or-later
//
// heat_current_f64.wgsl — Per-particle heat current contribution (f64)
//
// Computes the microscopic heat current J_q for Green-Kubo thermal conductivity:
//
//   J_q = Σ_i e_i × v_i + (1/2) Σ_{i<j} (F_ij · v_i + F_ji · v_j) × r_ij
//
// where e_i = KE_i + PE_i (per-particle energy), F_ij is the Yukawa pair force,
// and r_ij is the minimum image displacement.
//
// Each thread computes one particle's contribution to J_q:
//   - Convective: e_i × v_i
//   - Virial: (1/2) Σ_{j≠i} (F_ij · v_i) × r_ij  (symmetrized)
//
// Output: per-particle [J_x, J_y, J_z] f64 vectors.
// Host sums over particles to get total J_q(t) per snapshot.
//
// Absorbed from hotSpring CPU compute_heat_current() → GPU.
// Reference: Allen & Tildesley (1987), Evans & Morriss (1990)

struct HeatParams {
    n: u32,
    _pad0: u32,
    box_side: f64,
    kappa: f64,
    mass: f64,
}

@group(0) @binding(0) var<uniform> params: HeatParams;
@group(0) @binding(1) var<storage, read> pos: array<f64>;
@group(0) @binding(2) var<storage, read> vel: array<f64>;
@group(0) @binding(3) var<storage, read_write> jq_out: array<f64>;

@compute @workgroup_size(64)
fn heat_current(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.n { return; }

    let L = params.box_side;
    let kappa = params.kappa;
    let mass = params.mass;
    let n = params.n;

    let xi = pos[i * 3u];
    let yi = pos[i * 3u + 1u];
    let zi = pos[i * 3u + 2u];
    let vix = vel[i * 3u];
    let viy = vel[i * 3u + 1u];
    let viz = vel[i * 3u + 2u];

    var pe_i: f64 = f64(0.0);
    var jq_x: f64 = f64(0.0);
    var jq_y: f64 = f64(0.0);
    var jq_z: f64 = f64(0.0);

    let guard: f64 = f64(1e-30);

    for (var j: u32 = 0u; j < n; j = j + 1u) {
        if j == i { continue; }

        var dx = pos[j * 3u] - xi;
        var dy = pos[j * 3u + 1u] - yi;
        var dz = pos[j * 3u + 2u] - zi;

        // Minimum image convention
        dx = dx - L * round(dx / L);
        dy = dy - L * round(dy / L);
        dz = dz - L * round(dz / L);

        let r2 = dx * dx + dy * dy + dz * dz;
        let r = sqrt(r2);
        if r < guard { continue; }

        // Yukawa potential: u(r) = exp(-κr) / r
        let exp_kr = exp(-kappa * r);
        let u_pair = exp_kr / r;
        pe_i = pe_i + f64(0.5) * u_pair;

        // Yukawa force magnitude: F = exp(-κr)(κr+1)/r²
        let f_mag = exp_kr * (kappa * r + f64(1.0)) / r2;
        let inv_r = f64(1.0) / r;
        let fx = f_mag * dx * inv_r;
        let fy = f_mag * dy * inv_r;
        let fz = f_mag * dz * inv_r;

        // Virial contribution: (1/2)(F_ij · v_i) × r_ij
        let f_dot_vi = fx * vix + fy * viy + fz * viz;
        jq_x = jq_x + f64(0.5) * f_dot_vi * dx;
        jq_y = jq_y + f64(0.5) * f_dot_vi * dy;
        jq_z = jq_z + f64(0.5) * f_dot_vi * dz;
    }

    // Convective part: e_i × v_i where e_i = KE_i + PE_i
    let ke_i = f64(0.5) * mass * (vix * vix + viy * viy + viz * viz);
    let e_i = ke_i + pe_i;

    jq_x = jq_x + e_i * vix;
    jq_y = jq_y + e_i * viy;
    jq_z = jq_z + e_i * viz;

    jq_out[i * 3u] = jq_x;
    jq_out[i * 3u + 1u] = jq_y;
    jq_out[i * 3u + 2u] = jq_z;
}

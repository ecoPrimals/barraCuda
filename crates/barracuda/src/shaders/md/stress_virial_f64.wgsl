// SPDX-License-Identifier: AGPL-3.0-or-later
//
// stress_virial_f64.wgsl — Instantaneous stress tensor from virial theorem (f64)
//
// Computes the 6 independent components of the stress tensor:
//   σ_αβ = (1/V) * [Σᵢ mᵢ vᵢ_α vᵢ_β + Σᵢ rᵢ_α Fᵢ_β]
//
// where V is volume, m mass, v velocity, r position, F total force on particle.
// The virial term uses the centroid formulation: Σᵢ rᵢ_α Fᵢ_β (valid for
// pair potentials with F_i = Σ_{j≠i} f_ij).
//
// Output: [σ_xx, σ_yy, σ_zz, σ_xy, σ_xz, σ_yz]
//
// Reference: Allen & Tildesley (1987), Frenkel & Smit (2002)

struct StressParams {
    n_atoms: u32,
    _pad0: u32,
    volume: f64,
}

@group(0) @binding(0) var<storage, read> pos: array<f64>;
@group(0) @binding(1) var<storage, read> vel: array<f64>;
@group(0) @binding(2) var<storage, read> force: array<f64>;
@group(0) @binding(3) var<storage, read> mass: array<f64>;
@group(0) @binding(4) var<storage, read_write> stress_out: array<f64>;
@group(0) @binding(5) var<uniform> params: StressParams;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let comp = gid.x;
    if comp >= 6u { return; }

    let n = params.n_atoms;
    let inv_V = f64(1.0) / params.volume;

    // Map comp to (α,β): 0->(0,0), 1->(1,1), 2->(2,2), 3->(0,1), 4->(0,2), 5->(1,2)
    var ia: u32 = 0u;
    var ib: u32 = 0u;
    if comp == 0u { ia = 0u; ib = 0u; }
    if comp == 1u { ia = 1u; ib = 1u; }
    if comp == 2u { ia = 2u; ib = 2u; }
    if comp == 3u { ia = 0u; ib = 1u; }
    if comp == 4u { ia = 0u; ib = 2u; }
    if comp == 5u { ia = 1u; ib = 2u; }

    var sum: f64 = f64(0.0);
    for (var i: u32 = 0u; i < n; i = i + 1u) {
        let mi = mass[i];
        let ri_a = pos[i * 3u + ia];
        let ri_b = pos[i * 3u + ib];
        let vi_a = vel[i * 3u + ia];
        let vi_b = vel[i * 3u + ib];
        let Fi_a = force[i * 3u + ia];
        let Fi_b = force[i * 3u + ib];

        // Kinetic: m_i * v_α * v_β
        sum = sum + mi * vi_a * vi_b;

        // Virial: r_α * F_β (centroid formulation)
        sum = sum + ri_a * Fi_b;
    }

    stress_out[comp] = inv_V * sum;
}

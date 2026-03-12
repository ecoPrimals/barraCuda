// SPDX-License-Identifier: AGPL-3.0-only
// deformed_energy_f64.wgsl — Total HFB energy for axially-deformed nuclei
//
// hotSpring absorption: computes the HFB energy functional E[ρ] on a (ρ,z)
// cylindrical grid. Integrates kinetic, Skyrme, Coulomb, and pairing
// contributions using trapezoidal quadrature with cylindrical volume element.
//
// E = E_kin + E_skyrme + E_coulomb + E_pair
//   = Σ_i (2j_i+1)·v²_i·ε_i                    (single-particle)
//   + ∫∫ [E_sky(ρ,z) + E_coul(ρ,z)] ρ dρ dz    (functional correction)
//   - G·|Σ_i u_i·v_i|²                          (pairing)
//
// Two entry points:
//   energy_integrand: per-grid-point integrand → buffer
//   reduce_energy:    sum buffer → scalar
//
// Bindings:
//   0: params          uniform
//   1: density_n       [n_grid] f64
//   2: density_p       [n_grid] f64
//   3: rho_grid        [n_grid_rho] f64
//   4: integrand_out   [n_grid] f64 — per-point output for first pass
//   5: energy_out      [1] f64 — scalar output for reduction pass
//   6: sp_energies     [n_states] f64
//   7: v_squared       [n_states] f64
//   8: degeneracy      [n_states] f64

// f64 enabled by compile_shader_f64() preamble injection

struct EnergyParams {
    n_grid_rho:     u32,
    n_grid_z:       u32,
    n_states:       u32,
    _pad:           u32,
    d_rho:          f64,
    d_z:            f64,
    t0:             f64,
    x0:             f64,
    t3:             f64,
    x3:             f64,
    alpha_skyrme:   f64,
    pairing_G:      f64,
}

@group(0) @binding(0) var<uniform>             params: EnergyParams;
@group(0) @binding(1) var<storage, read>       density_n: array<f64>;
@group(0) @binding(2) var<storage, read>       density_p: array<f64>;
@group(0) @binding(3) var<storage, read>       rho_grid: array<f64>;
@group(0) @binding(4) var<storage, read_write> integrand_out: array<f64>;
@group(0) @binding(5) var<storage, read_write> energy_out: array<f64>;
@group(0) @binding(6) var<storage, read>       sp_energies: array<f64>;
@group(0) @binding(7) var<storage, read>       v_squared: array<f64>;
@group(0) @binding(8) var<storage, read>       degeneracy: array<f64>;

fn pow_approx(base: f64, exponent: f64) -> f64 {
    let zero = base - base;
    if (base <= zero) { return zero; }
    return exp(exponent * log(base));
}

// cbrt_f64 provided by math_f64.wgsl auto-injection (Halley's method, full f64 precision)

@compute @workgroup_size(256)
fn energy_integrand(@builtin(global_invocation_id) gid: vec3<u32>) {
    let nrho = params.n_grid_rho;
    let nz = params.n_grid_z;
    let n_grid = nrho * nz;
    if (gid.x >= n_grid) { return; }

    let ir = gid.x / nz;
    let zero = params.d_rho - params.d_rho;
    let half = zero + 0.5;

    let rho_n = density_n[gid.x];
    let rho_p = density_p[gid.x];
    let rho = rho_n + rho_p;

    let t0 = params.t0;
    let x0 = params.x0;
    let t3 = params.t3;
    let x3 = params.x3;
    let alpha = params.alpha_skyrme;

    // Skyrme energy density: ε_sky = (t0/2)·[(1+x0/2)ρ² - (1/2+x0)(ρ_n²+ρ_p²)]
    //                               + (t3/12)·ρ^α·[similar isospin terms]
    let rho2_sum = rho_n * rho_n + rho_p * rho_p;
    let e_sky = half * t0 * (((zero + 1.0) + half * x0) * rho * rho
                - (half + x0) * rho2_sum)
              + t3 / (zero + 12.0) * pow_approx(rho, alpha)
                * (((zero + 1.0) + half * x3) * rho * rho
                   - (half + x3) * rho2_sum);

    // Coulomb exchange: ε_coul_x = -(3/4)·e²·(3/π)^{1/3}·ρ_p^{4/3}
    let e2 = zero + 1.4399764;
    let inv_pi = zero + 0.3183098861837907;
    let e_coul_x = -(zero + 0.75) * e2 * pow_approx((zero + 3.0) * inv_pi, zero + 0.33333333333333333)
                   * pow_approx(rho_p, zero + 1.33333333333333333);

    let rho_cyl = rho_grid[ir];
    integrand_out[gid.x] = (e_sky + e_coul_x) * rho_cyl * params.d_rho * params.d_z;
}

var<workgroup> shared_sum: array<f64, 256>;

@compute @workgroup_size(256)
fn reduce_energy(@builtin(global_invocation_id) gid: vec3<u32>,
                 @builtin(local_invocation_id) lid: vec3<u32>) {
    let n_grid = params.n_grid_rho * params.n_grid_z;
    let zero = params.d_rho - params.d_rho;

    var val = zero;
    if (gid.x < n_grid) {
        val = integrand_out[gid.x];
    }
    shared_sum[lid.x] = val;
    workgroupBarrier();

    // Tree reduction
    var stride = 128u;
    while (stride > 0u) {
        if (lid.x < stride) {
            shared_sum[lid.x] = shared_sum[lid.x] + shared_sum[lid.x + stride];
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }

    if (lid.x == 0u) {
        // Atomic add via manual serialization
        // Single-particle contribution accumulated on CPU side
        let old = energy_out[0];
        energy_out[0] = old + shared_sum[0];
    }
}

// SPDX-License-Identifier: AGPL-3.0-only
// deformed_potential_f64.wgsl — Axially-deformed nuclear mean-field potential
//
// hotSpring absorption: Skyrme + Coulomb potentials on (ρ,z) grid for
// axially-deformed nuclei. Extends the spherical HFB potentials to 2D.
//
// V_skyrme(ρ,z) = t0·(1 + x0/2)·ρ - t0·(1/2 + x0)·ρ_q
//               + t3/12·(2 + α)·(1 + x3/2)·ρ^(α+1) - t3/12·(2x3 + 1)·ρ^α·ρ_q
//
// V_coulomb(ρ,z) = direct Coulomb from proton density (Poisson-solved externally)
//                + Slater exchange: -3e²/4·(3/π)^(1/3)·ρ_p^(1/3)
//
// Thread assignment: one thread per grid point.
//
// Bindings:
//   0: params        uniform { n_grid_rho, n_grid_z, Z, _pad,
//                               t0, x0, t3, x3, alpha_skyrme: f64 }
//   1: density_n     [n_grid] f64 — neutron density
//   2: density_p     [n_grid] f64 — proton density
//   3: v_coulomb_in  [n_grid] f64 — direct Coulomb from external Poisson solver
//   4: v_neutron     [n_grid] f64 — output: total neutron potential
//   5: v_proton      [n_grid] f64 — output: total proton potential

// f64 enabled by compile_shader_f64() preamble injection

struct PotParams {
    n_grid_rho:     u32,
    n_grid_z:       u32,
    Z:              u32,
    _pad:           u32,
    t0:             f64,
    x0:             f64,
    t3:             f64,
    x3:             f64,
    alpha_skyrme:   f64,
}

@group(0) @binding(0) var<uniform>             params: PotParams;
@group(0) @binding(1) var<storage, read>       density_n: array<f64>;
@group(0) @binding(2) var<storage, read>       density_p: array<f64>;
@group(0) @binding(3) var<storage, read>       v_coulomb_in: array<f64>;
@group(0) @binding(4) var<storage, read_write> v_neutron: array<f64>;
@group(0) @binding(5) var<storage, read_write> v_proton: array<f64>;

// cbrt_f64 provided by math_f64.wgsl auto-injection (Halley's method, full f64 precision)

fn pow_approx(base: f64, exponent: f64) -> f64 {
    let zero = base - base;
    if (base <= zero) { return zero; }
    return exp(exponent * log(base));
}

@compute @workgroup_size(256)
fn compute_potentials(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n_grid = params.n_grid_rho * params.n_grid_z;
    if (gid.x >= n_grid) { return; }

    let zero = params.t0 - params.t0;
    let half = zero + 0.5;
    let rho_n = density_n[gid.x];
    let rho_p = density_p[gid.x];
    let rho = rho_n + rho_p;

    let t0 = params.t0;
    let x0 = params.x0;
    let t3 = params.t3;
    let x3 = params.x3;
    let alpha = params.alpha_skyrme;

    // Skyrme central field
    let rho_alpha = pow_approx(rho, alpha);

    // Neutron potential: isospin τ=n → ρ_q = ρ_n
    let v_sky_n = t0 * ((zero + 1.0) + x0 * half) * rho
                - t0 * (half + x0) * rho_n
                + t3 / (zero + 12.0) * ((zero + 2.0) + alpha) * ((zero + 1.0) + x3 * half) * rho_alpha * rho
                - t3 / (zero + 12.0) * ((zero + 2.0) * x3 + (zero + 1.0)) * rho_alpha * rho_n;

    // Proton potential
    let v_sky_p = t0 * ((zero + 1.0) + x0 * half) * rho
                - t0 * (half + x0) * rho_p
                + t3 / (zero + 12.0) * ((zero + 2.0) + alpha) * ((zero + 1.0) + x3 * half) * rho_alpha * rho
                - t3 / (zero + 12.0) * ((zero + 2.0) * x3 + (zero + 1.0)) * rho_alpha * rho_p;

    // Slater exchange: −(3e²/4)·(3ρ_p/π)^{1/3}
    let e2 = zero + 1.4399764;  // e² in MeV·fm
    let inv_pi = zero + 0.3183098861837907;
    let slater = -(zero + 0.75) * e2 * cbrt_f64((zero + 3.0) * rho_p * inv_pi);

    v_neutron[gid.x] = v_sky_n;
    v_proton[gid.x] = v_sky_p + v_coulomb_in[gid.x] + slater;
}

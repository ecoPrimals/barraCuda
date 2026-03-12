// SPDX-License-Identifier: AGPL-3.0-only
// batched_hfb_potentials_f64.wgsl — Skyrme + Coulomb potentials for spherical HFB
//
// hotSpring absorption: nuclear physics GPU-resident SCF.
//
// Computes the self-consistent single-particle potential for protons and
// neutrons on a radial grid, given the nuclear densities.
//
// Physics:
//   V_q(r) = Σ Skyrme central terms(ρ_p, ρ_n, α) + V_Coulomb(r) for protons
//
// Skyrme central: t0(1+x0/2)ρ - t0(1/2+x0)ρ_q + (t3/12)(2+α)(1+x3/2)ρ^(α+1) - ...
// Coulomb direct: forward/backward radial cumulative sum of charge density
// Coulomb exchange: -e²(3/π)^(1/3) ρ_p^(1/3) (Slater approximation)
//
// Bindings:
//   0: dims      uniform { n_grid: u32, n_batch: u32, _pad: u32[2] }
//   1: rho_p     [n_batch × n_grid] f64 — proton density
//   2: rho_n     [n_batch × n_grid] f64 — neutron density
//   3: r_grid    [n_grid] f64 — radial grid points
//   4: sky_params uniform { t0,t1,t2,t3,x0,x1,x2,x3,alpha,W0,e2: f64, _pad: f64 }
//   5: u_total   [n_batch × 2 × n_grid] f64 — output: q=0 proton, q=1 neutron
//   6: f_q       [n_batch × 2 × n_grid] f64 — effective mass form factor

// f64 enabled by compile_shader_f64() preamble injection

struct Dims {
    n_grid:  u32,
    n_batch: u32,
    _pad0:   u32,
    _pad1:   u32,
}

struct SkyParams {
    t0:    f64,
    t1:    f64,
    t2:    f64,
    t3:    f64,
    x0:    f64,
    x1:    f64,
    x2:    f64,
    x3:    f64,
    alpha: f64,
    W0:    f64,
    e2:    f64,
    _pad:  f64,
}

@group(0) @binding(0) var<uniform>             dims: Dims;
@group(0) @binding(1) var<storage, read>       rho_p: array<f64>;
@group(0) @binding(2) var<storage, read>       rho_n: array<f64>;
@group(0) @binding(3) var<storage, read>       r_grid: array<f64>;
@group(0) @binding(4) var<uniform>             sky: SkyParams;
@group(0) @binding(5) var<storage, read_write> u_total: array<f64>;
@group(0) @binding(6) var<storage, read_write> f_q: array<f64>;

// cbrt_f64 provided by math_f64.wgsl auto-injection (Halley's method, full f64 precision)

@compute @workgroup_size(256)
fn compute_potentials(
    @builtin(global_invocation_id) gid: vec3<u32>
) {
    let idx = gid.x;
    let ng = dims.n_grid;
    let total = dims.n_batch * ng;
    if (idx >= total) { return; }

    let batch = idx / ng;
    let ir = idx % ng;
    let zero = r_grid[0] - r_grid[0];
    let one = zero + 1.0;
    let half = zero + 0.5;

    let rp = rho_p[batch * ng + ir];
    let rn = rho_n[batch * ng + ir];
    let rho = rp + rn;

    // Skyrme central for isospin q=0 (proton), q=1 (neutron)
    let rho_alpha = pow(max(rho, zero + 1e-30), sky.alpha);

    // Common Skyrme terms
    let t0_term_iso = sky.t0 * (one + half * sky.x0) * rho;
    let t3_term_iso = sky.t3 / (zero + 12.0) * ((zero + 2.0) + sky.alpha) *
                      (one + half * sky.x3) * rho_alpha * rho;

    // Proton potential (q=0)
    let u_p = t0_term_iso - sky.t0 * (half + sky.x0) * rp +
              t3_term_iso - sky.t3 / (zero + 12.0) * ((zero + 2.0) + sky.alpha) *
              (half + sky.x3) * rho_alpha * rp;

    // Neutron potential (q=1)
    let u_n = t0_term_iso - sky.t0 * (half + sky.x0) * rn +
              t3_term_iso - sky.t3 / (zero + 12.0) * ((zero + 2.0) + sky.alpha) *
              (half + sky.x3) * rho_alpha * rn;

    // Effective mass form factor: f_q = 1 + (m/ℏ²)(t1(1+x1/2) + t2(1+x2/2)) ρ / 4
    let f_common = (sky.t1 * (one + half * sky.x1) +
                    sky.t2 * (one + half * sky.x2)) * rho / (zero + 4.0);
    let f_p_val = one + f_common - (sky.t1 * (half + sky.x1) -
                  sky.t2 * (half + sky.x2)) * rp / (zero + 4.0);
    let f_n_val = one + f_common - (sky.t1 * (half + sky.x1) -
                  sky.t2 * (half + sky.x2)) * rn / (zero + 4.0);

    let p_off = batch * 2u * ng;
    let n_off = p_off + ng;

    u_total[p_off + ir] = u_p;
    u_total[n_off + ir] = u_n;
    f_q[p_off + ir] = f_p_val;
    f_q[n_off + ir] = f_n_val;
}

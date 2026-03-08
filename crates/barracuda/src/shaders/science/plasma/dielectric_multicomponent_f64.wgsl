// SPDX-License-Identifier: AGPL-3.0-only
// Multi-component Mermin dielectric function (electron-ion plasma).
//
// Each thread computes ε(k,ω) for one frequency point, summing species
// susceptibilities. Supports both standard and completed (momentum-conserving)
// Mermin formulations.
//
// Species layout in species_buf: [mass, charge, density, temp, nu, v_th, k_debye] × N_species
// Prepend: math_f64.wgsl for exp_f64

struct MulticompParams {
    n_points: u32,
    n_species: u32,
    use_completed: u32,
    _pad: u32,
    k: f64,
}

@group(0) @binding(0) var<uniform> params: MulticompParams;
@group(0) @binding(1) var<storage, read> omegas: array<f64>;
@group(0) @binding(2) var<storage, read> species_buf: array<f64>;
@group(0) @binding(3) var<storage, read_write> loss_out: array<f64>;
@group(0) @binding(4) var<storage, read_write> dsf_out: array<f64>;

const SQRT_PI: f64 = 1.7724538509055159;
const SQRT_2: f64 = 1.4142135623730951;
const PI: f64 = 3.14159265358979323846;

fn cmul(a: vec2<f64>, b: vec2<f64>) -> vec2<f64> {
    return vec2<f64>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

fn cinv(z: vec2<f64>) -> vec2<f64> {
    let d = z.x * z.x + z.y * z.y;
    return vec2<f64>(z.x / d, -z.y / d);
}

fn cabs(z: vec2<f64>) -> f64 {
    return sqrt(z.x * z.x + z.y * z.y);
}

fn cscale(z: vec2<f64>, s: f64) -> vec2<f64> {
    return vec2<f64>(z.x * s, z.y * s);
}

fn plasma_z_asymptotic(z: vec2<f64>) -> vec2<f64> {
    let z2 = cmul(z, z);
    let z2_inv = cinv(z2);
    var sum = vec2<f64>(-1.0, 0.0);
    sum = sum + cscale(z2_inv, -0.5);
    let z4_inv = cmul(z2_inv, z2_inv);
    sum = sum + cscale(z4_inv, -0.75);
    return cmul(cinv(z), sum);
}

fn plasma_z_series(z: vec2<f64>) -> vec2<f64> {
    let z2 = cmul(z, z);
    let neg2z2 = cscale(z2, -2.0);
    var term = vec2<f64>(1.0, 0.0);
    var total = term;
    for (var n = 1u; n < 60u; n++) {
        term = cscale(cmul(term, neg2z2), 1.0 / f64(2u * n + 1u));
        total = total + term;
        if cabs(term) < 1e-15 { break; }
    }
    let e_val = exp_f64(-z2.x);
    let exp_re = e_val * cos(z2.y);
    let exp_im = -e_val * sin(z2.y);
    let zexp = vec2<f64>(-SQRT_PI * exp_im, SQRT_PI * exp_re);
    return zexp - cscale(cmul(z, total), 2.0);
}

fn plasma_z(z: vec2<f64>) -> vec2<f64> {
    if cabs(z) > 6.0 {
        return plasma_z_asymptotic(z);
    }
    return plasma_z_series(z);
}

fn species_chi0(k: f64, omega: vec2<f64>, sp_idx: u32) -> vec2<f64> {
    let base = sp_idx * 7u;
    let v_th = species_buf[base + 5u];
    let k_d = species_buf[base + 6u];

    if abs(k) < 1e-30 { return vec2<f64>(0.0, 0.0); }

    let z = cscale(omega, 1.0 / (k * v_th * SQRT_2));
    let z_func = plasma_z(z);
    let w_prime = cscale(vec2<f64>(1.0, 0.0) + cmul(z, z_func), -2.0);

    let ratio_sq = (k_d / k) * (k_d / k);
    return cscale(w_prime, -0.5 * ratio_sq);
}

@compute @workgroup_size(64)
fn compute_multicomp(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.n_points { return; }

    let omega = omegas[idx];
    let k = params.k;
    let zero = k - k;

    if abs(omega) < 1e-15 {
        loss_out[idx] = zero;
        dsf_out[idx] = zero;
        return;
    }

    let omega_c = vec2<f64>(omega, 0.0);
    var total_chi = vec2<f64>(0.0, 0.0);

    for (var s = 0u; s < params.n_species; s++) {
        let base = s * 7u;
        let nu_s = species_buf[base + 4u];
        let v_th = species_buf[base + 5u];

        let omega_shifted = vec2<f64>(omega, nu_s);

        let chi_shifted = species_chi0(k, omega_shifted, s);
        let chi_static = species_chi0(k, vec2<f64>(0.0, 0.0), s);

        let numer = cmul(cmul(omega_shifted, cinv(omega_c)), chi_shifted);

        var r = vec2<f64>(0.0, 0.0);
        if cabs(chi_static) > 1e-30 {
            r = cmul(chi_shifted, cinv(chi_static));
        }

        var denom = vec2<f64>(1.0, 0.0);
        if params.use_completed != 0u {
            let k2_vth2 = k * k * v_th * v_th;
            let omega_prod = cmul(omega_c, omega_shifted);
            let g_p = cscale(cmul(r, omega_prod), 1.0 / k2_vth2);
            let correction = cmul(r, vec2<f64>(1.0, 0.0) - g_p);
            denom = vec2<f64>(1.0, 0.0) + cmul(vec2<f64>(0.0, nu_s), cmul(cinv(omega_c), correction));
        } else {
            denom = vec2<f64>(1.0, 0.0) + cmul(vec2<f64>(0.0, nu_s), cmul(cinv(omega_c), r));
        }

        total_chi = total_chi + cmul(numer, cinv(denom));
    }

    let eps = vec2<f64>(1.0, 0.0) + total_chi;
    let eps_inv = cinv(eps);
    loss_out[idx] = -eps_inv.y;

    // DSF computation requires total T, n from species
    var total_nt = zero;
    var total_n = zero;
    for (var s = 0u; s < params.n_species; s++) {
        let base = s * 7u;
        let n_s = species_buf[base + 2u];
        let t_s = species_buf[base + 3u];
        total_nt = total_nt + n_s * t_s;
        total_n = total_n + n_s;
    }
    let t_avg = total_nt / max(total_n, 1e-30);

    let prefactor = t_avg * k * k / (PI * total_n);
    dsf_out[idx] = max(prefactor * loss_out[idx] / omega, zero);
}

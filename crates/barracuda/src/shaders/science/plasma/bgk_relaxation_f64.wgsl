// SPDX-License-Identifier: AGPL-3.0-or-later
// Batched BGK relaxation step for multi-species kinetic plasma.
//
// Each thread handles one velocity grid point for one species.
// Phase 1: Compute velocity-space moments (n, nu, nv²) via atomics.
// Phase 2: Build target Maxwellians and relax f → f + dt·ν·(f_M - f).
//
// Since WGSL lacks f64 atomics, we use a two-pass approach:
// Pass 1 (moments): GPU computes per-point contributions, CPU reduces.
// Pass 2 (relax): GPU updates f using the CPU-computed target params.
//
// Prepend: math_f64.wgsl for exp_f64

struct BgkParams {
    nv: u32,
    n_species: u32,
    dt: f64,
    v_min: f64,
    dv: f64,
}

@group(0) @binding(0) var<uniform> params: BgkParams;
@group(0) @binding(1) var<storage, read_write> f_data: array<f64>;
@group(0) @binding(2) var<storage, read> species_params: array<f64>;
@group(0) @binding(3) var<storage, read_write> moment_contrib: array<f64>;

fn velocity_at(j: u32) -> f64 {
    let zero = params.dv - params.dv;
    return params.v_min + (zero + f64(j)) * params.dv;
}

// Pass 1: compute per-point moment contributions (n_j, n_j·v_j, n_j·v_j²).
// CPU sums these across velocity points.
@compute @workgroup_size(64)
fn compute_moments(@builtin(global_invocation_id) gid: vec3<u32>) {
    let nv = params.nv;
    let sp = gid.x / nv;
    let j = gid.x % nv;
    if sp >= params.n_species { return; }

    let f_val = f_data[sp * nv + j];
    let v = velocity_at(j);

    let base = sp * nv * 3u + j * 3u;
    moment_contrib[base + 0u] = f_val * params.dv;
    moment_contrib[base + 1u] = f_val * v * params.dv;
    moment_contrib[base + 2u] = f_val * v * v * params.dv;
}

// Pass 2: apply BGK relaxation toward target Maxwellian.
// species_params layout per species: [mass, nu, n_star, u_star, t_star]
@compute @workgroup_size(64)
fn bgk_relax(@builtin(global_invocation_id) gid: vec3<u32>) {
    let nv = params.nv;
    let sp = gid.x / nv;
    let j = gid.x % nv;
    if sp >= params.n_species { return; }

    let zero = params.dv - params.dv;
    let pi = zero + 3.14159265358979323846;

    let base_sp = sp * 5u;
    let m = species_params[base_sp + 0u];
    let nu = species_params[base_sp + 1u];
    let n_star = species_params[base_sp + 2u];
    let u_star = species_params[base_sp + 3u];
    let t_star = max(species_params[base_sp + 4u], zero + 1e-12);

    let v = velocity_at(j);
    let dv_sq = (v - u_star) * (v - u_star);
    let exponent = -m * dv_sq / ((zero + 2.0) * t_star);
    let coeff = n_star * sqrt(m / ((zero + 2.0) * pi * t_star));
    let f_target = coeff * exp_f64(max(exponent, zero - 500.0));

    let idx = sp * nv + j;
    let f_old = f_data[idx];
    f_data[idx] = max(f_old + params.dt * nu * (f_target - f_old), zero);
}

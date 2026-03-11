// SPDX-License-Identifier: AGPL-3.0-only
// Density Accumulation, BCS, Energy Functional, and Observables
// for Deformed HFB on 2D Cylindrical Grid (f64)
//
// Kernels:
//   1. compute_bcs_occupations  — BCS v² from eigenvalues + chemical potential
//   2. compute_densities        — ρ_p(r), ρ_n(r) from wavefunctions and occupations
//   3. mix_density_linear       — linear density mixing (warmup phase)
//   4. mix_density_broyden      — Broyden vector ops (u + α·F corrections)
//   5. compute_energy_integrands — Skyrme + Coulomb energy density per grid point
//   6. compute_quadrupole       — Q20 = ∫ ρ(2z²-ρ²) dV  (reduction kernel)
//   7. compute_rms_radius       — <r²> = ∫ ρ·r² dV / ∫ ρ dV  (reduction kernel)
//
// Grid: row-major [n_rho × n_z], ρ starts at d_rho.
// Deep Debt: pure WGSL, f64, self-contained.

// ═══════════════════════════════════════════════════════════════════
// Parameters
// ═══════════════════════════════════════════════════════════════════

struct DensityParams {
    n_rho: u32,
    n_z: u32,
    n_states: u32,
    n_particles: u32,   // Z or N for this species
    d_rho: f64,
    d_z: f64,
    z_min: f64,
    delta_pair: f64,    // pairing gap Δ
    fermi_energy: f64,  // chemical potential μ (from BCS bisection)
    mix_alpha: f64,     // mixing parameter (0.5 for linear, 0.4 for Broyden)
}

@group(0) @binding(0) var<uniform> params: DensityParams;

// ═══════════════════════════════════════════════════════════════════
// Kernel 1: BCS occupation probabilities
//
// v²_i = 0.5 * (1 - ε_i / E_qp_i)
// where ε_i = e_i - μ, E_qp_i = √(ε²_i + Δ²)
//
// Input:  eigenvalues[n_states] — sorted single-particle energies
// Output: occupations[n_states] — BCS v²_i
// Dispatch: (ceil(n_states / 256), 1, 1)
// ═══════════════════════════════════════════════════════════════════

@group(0) @binding(1) var<storage, read> eigenvalues: array<f64>;
@group(0) @binding(2) var<storage, read_write> occupations: array<f64>;

@compute @workgroup_size(256)
fn compute_bcs_occupations(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.n_states) { return; }

    if (params.delta_pair > f64(1e-10)) {
        let eps = eigenvalues[idx] - params.fermi_energy;
        let e_qp = sqrt(eps * eps + params.delta_pair * params.delta_pair);
        let v2 = f64(0.5) * (f64(1.0) - eps / e_qp);
        occupations[idx] = clamp(v2, f64(0.0), f64(1.0));
    } else {
        // Sharp Fermi surface: filled below μ
        if (eigenvalues[idx] < params.fermi_energy) {
            occupations[idx] = f64(1.0);
        } else {
            occupations[idx] = f64(0.0);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════
// Kernel 2: Density accumulation
//
// ρ(r) = Σ_i  2 · n_i · |ψ_i(r)|²
// (factor 2 for time-reversal degeneracy)
//
// Each thread handles one grid point, loops over states.
// Input:  wavefunctions[n_states × n_grid], occupations[n_states]
// Output: density_out[n_grid]
// Dispatch: (ceil(n_grid / 256), 1, 1)
// ═══════════════════════════════════════════════════════════════════

@group(0) @binding(3) var<storage, read> wavefunctions: array<f64>;
@group(0) @binding(4) var<storage, read_write> density_out: array<f64>;

@compute @workgroup_size(256)
fn compute_densities(@builtin(global_invocation_id) gid: vec3<u32>) {
    let grid_idx = gid.x;
    let n_grid = params.n_rho * params.n_z;

    if (grid_idx >= n_grid) { return; }

    var rho_val = f64(0.0);

    for (var si = 0u; si < params.n_states; si++) {
        let occ = occupations[si] * f64(2.0); // time-reversal
        if (occ < f64(1e-15)) { continue; }

        let psi = wavefunctions[si * n_grid + grid_idx];
        rho_val += occ * psi * psi;
    }

    density_out[grid_idx] = rho_val;
}

// ═══════════════════════════════════════════════════════════════════
// Kernel 3: Linear density mixing
//
// ρ_new[i] = (1-α) · ρ_old[i] + α · ρ_computed[i]
//
// Input:  density_out (as ρ_computed), old_density (read), mix_alpha
// Output: mixed_density (write)
// We reuse bindings differently for this kernel:
// binding(3) = old_density (read), binding(4) = new_computed, binding(5) = output
// Dispatch: (ceil(n_grid / 256), 1, 1)
// ═══════════════════════════════════════════════════════════════════

@group(0) @binding(5) var<storage, read> old_density: array<f64>;
@group(0) @binding(6) var<storage, read_write> mixed_density: array<f64>;

@compute @workgroup_size(256)
fn mix_density_linear(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let n_grid = params.n_rho * params.n_z;
    if (idx >= n_grid) { return; }

    let alpha = params.mix_alpha;
    let mixed = (f64(1.0) - alpha) * old_density[idx] + alpha * density_out[idx];
    // Ensure non-negative
    mixed_density[idx] = max(mixed, f64(0.0));
}

// ═══════════════════════════════════════════════════════════════════
// Kernel 4: Broyden mixing vector operations
//
// Computes: u_new[i] = u[i] + α·F[i] - Σ_m γ_m · (du_m[i] + α·df_m[i])
//
// This is the inner loop of Modified Broyden mixing.
// Input:  input_vec, residual, broyden_gammas, broyden_du/df history
// Output: mixed_vec
//
// For now, the Broyden history management and γ computation remain
// on CPU (small linear algebra on history vectors). This kernel does
// the final vector update which is O(n_grid).
//
// Dispatch: (ceil(vec_dim / 256), 1, 1)
// ═══════════════════════════════════════════════════════════════════

struct BroydenParams {
    vec_dim: u32,       // 2 * n_grid (proton + neutron densities concatenated)
    n_history: u32,     // number of Broyden history vectors
    _pad0: u32,
    _pad1: u32,
    alpha_mix: f64,
}

@group(1) @binding(0) var<uniform> broy_params: BroydenParams;
@group(1) @binding(1) var<storage, read> input_vec: array<f64>;     // current x
@group(1) @binding(2) var<storage, read> residual: array<f64>;      // F(x) = x_out - x
@group(1) @binding(3) var<storage, read> gammas: array<f64>;        // [n_history] γ coefficients
@group(1) @binding(4) var<storage, read> du_history: array<f64>;    // [n_history × vec_dim]
@group(1) @binding(5) var<storage, read> df_history: array<f64>;    // [n_history × vec_dim]
@group(1) @binding(6) var<storage, read_write> mixed_vec: array<f64>; // output

@compute @workgroup_size(256)
fn broyden_update(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= broy_params.vec_dim) { return; }

    var result = input_vec[idx] + broy_params.alpha_mix * residual[idx];

    for (var m = 0u; m < broy_params.n_history; m++) {
        let gamma_m = gammas[m];
        let du_m = du_history[m * broy_params.vec_dim + idx];
        let df_m = df_history[m * broy_params.vec_dim + idx];
        result -= gamma_m * (du_m + broy_params.alpha_mix * df_m);
    }

    // Ensure non-negative (for density vectors)
    mixed_vec[idx] = max(result, f64(0.0));
}

// ═══════════════════════════════════════════════════════════════════
// Kernel 5: Energy functional integrands
//
// Computes per-grid-point energy contributions:
//   H_0 = t0/4 * [(2+x0)ρ² - (1+2x0)(ρ_p² + ρ_n²)]
//   H_3 = t3/24 * ρ^α * [(2+x3)ρ² - (1+2x3)(ρ_p² + ρ_n²)]
//   E_Coulomb_exchange = -3/4 · e² · (3/π)^(1/3) · ρ_p^(4/3)
//
// Output: energy_integrands[n_grid] (integrate with SumReduceF64 for total)
// Dispatch: (ceil(n_grid / 256), 1, 1)
// ═══════════════════════════════════════════════════════════════════

struct EnergyParams {
    n_rho: u32,
    n_z: u32,
    _pad0: u32,
    _pad1: u32,
    d_rho: f64,
    d_z: f64,
    t0: f64,
    t3: f64,
    x0: f64,
    x3: f64,
    alpha: f64,
    e2: f64,           // e² = 1.4399764 MeV·fm
}

@group(2) @binding(0) var<uniform> energy_params: EnergyParams;
@group(2) @binding(1) var<storage, read> e_rho_p: array<f64>;
@group(2) @binding(2) var<storage, read> e_rho_n: array<f64>;
@group(2) @binding(3) var<storage, read_write> energy_integrands: array<f64>;

const PI_E: f64 = 3.14159265358979323846;

@compute @workgroup_size(256)
fn compute_energy_integrands(@builtin(global_invocation_id) gid: vec3<u32>) {
    let grid_idx = gid.x;
    let n_grid = energy_params.n_rho * energy_params.n_z;
    if (grid_idx >= n_grid) { return; }

    let i_rho = grid_idx / energy_params.n_z;
    let rho_coord = f64(i_rho + 1u) * energy_params.d_rho;
    let dv = f64(2.0) * PI_E * rho_coord * energy_params.d_rho * energy_params.d_z;

    let rp = max(e_rho_p[grid_idx], f64(0.0));
    let rn = max(e_rho_n[grid_idx], f64(0.0));
    let rho = rp + rn;

    // Skyrme central EDF
    let h_0 = energy_params.t0 / f64(4.0)
        * ((f64(2.0) + energy_params.x0) * rho * rho
           - (f64(1.0) + f64(2.0) * energy_params.x0) * (rp * rp + rn * rn));

    let h_3 = energy_params.t3 / f64(24.0) * pow(rho, energy_params.alpha)
        * ((f64(2.0) + energy_params.x3) * rho * rho
           - (f64(1.0) + f64(2.0) * energy_params.x3) * (rp * rp + rn * rn));

    // Coulomb exchange
    let coul_exch = f64(-0.75) * energy_params.e2
        * pow(f64(3.0) / PI_E, f64(1.0) / f64(3.0))
        * pow(rp, f64(4.0) / f64(3.0));

    energy_integrands[grid_idx] = (h_0 + h_3 + coul_exch) * dv;
}

// ═══════════════════════════════════════════════════════════════════
// Kernel 6: Quadrupole moment integrand
//
// Q20_integrand[i] = ρ_total(i) · (2z² - ρ²) · dV(i)
// Sum with SumReduceF64 to get total Q20.
//
// Dispatch: (ceil(n_grid / 256), 1, 1)
// ═══════════════════════════════════════════════════════════════════

struct ObsParams {
    n_rho: u32,
    n_z: u32,
    _pad0: u32,
    _pad1: u32,
    d_rho: f64,
    d_z: f64,
    z_min: f64,
}

@group(3) @binding(0) var<uniform> obs_params: ObsParams;
@group(3) @binding(1) var<storage, read> rho_total: array<f64>;
@group(3) @binding(2) var<storage, read_write> q20_integrands: array<f64>;
@group(3) @binding(3) var<storage, read_write> rms_r2_integrands: array<f64>;
@group(3) @binding(4) var<storage, read_write> rms_rho_integrands: array<f64>;

@compute @workgroup_size(256)
fn compute_observables(@builtin(global_invocation_id) gid: vec3<u32>) {
    let grid_idx = gid.x;
    let n_grid = obs_params.n_rho * obs_params.n_z;
    if (grid_idx >= n_grid) { return; }

    let i_rho = grid_idx / obs_params.n_z;
    let i_z = grid_idx % obs_params.n_z;

    let rho_coord = f64(i_rho + 1u) * obs_params.d_rho;
    let z_coord = obs_params.z_min + (f64(i_z) + f64(0.5)) * obs_params.d_z;
    let dv = f64(2.0) * PI_E * rho_coord * obs_params.d_rho * obs_params.d_z;

    let rho_val = rho_total[grid_idx];
    let r2 = rho_coord * rho_coord + z_coord * z_coord;

    // Q20: quadrupole moment
    q20_integrands[grid_idx] = rho_val * (f64(2.0) * z_coord * z_coord - rho_coord * rho_coord) * dv;

    // RMS radius components: <r²> = Σ ρ·r²·dV / Σ ρ·dV
    rms_r2_integrands[grid_idx] = rho_val * r2 * dv;
    rms_rho_integrands[grid_idx] = rho_val * dv;
}

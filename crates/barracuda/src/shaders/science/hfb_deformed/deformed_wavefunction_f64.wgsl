// SPDX-License-Identifier: AGPL-3.0-only
// deformed_wavefunction_f64.wgsl — Nilsson harmonic oscillator basis on (ρ,z) grid
//
// hotSpring absorption: axially-deformed nuclear HFB.
//
// Evaluates HO basis wavefunctions ψ_{n_ρ, n_z, Λ}(ρ, z) on a cylindrical
// grid for each quantum state in the basis.
//
// ψ(ρ, z) = R_{n_ρ}^{|Λ|}(ρ) · Z_{n_z}(z)
//   R = ρ^|Λ| · L_{n_ρ}^{|Λ|}(ρ²/b²) · exp(-ρ²/(2b²))
//   Z = H_{n_z}(z/b) · exp(-z²/(2b²))
//
// where L is the associated Laguerre polynomial, H is the Hermite polynomial,
// and b = sqrt(ℏ/(mω)) is the oscillator length.
//
// Thread assignment: one thread per (grid_point, state) pair.
//
// Bindings:
//   0: params       uniform { n_grid_rho, n_grid_z, n_states, _pad, b_osc: f64 }
//   1: state_params [n_states × 3] u32 — (n_rho, n_z, Lambda) per state
//   2: rho_grid     [n_grid_rho] f64
//   3: z_grid       [n_grid_z] f64
//   4: wavefunctions [n_states × n_grid_rho × n_grid_z] f64 — output

// f64 enabled by compile_shader_f64() preamble injection

struct WfParams {
    n_grid_rho: u32,
    n_grid_z:   u32,
    n_states:   u32,
    _pad:       u32,
    b_osc:      f64,
}

@group(0) @binding(0) var<uniform>             params: WfParams;
@group(0) @binding(1) var<storage, read>       state_params: array<u32>;
@group(0) @binding(2) var<storage, read>       rho_grid: array<f64>;
@group(0) @binding(3) var<storage, read>       z_grid: array<f64>;
@group(0) @binding(4) var<storage, read_write> wavefunctions: array<f64>;

fn hermite(n: u32, x: f64) -> f64 {
    let zero = x - x;
    if (n == 0u) { return zero + 1.0; }
    if (n == 1u) { return (zero + 2.0) * x; }
    var h_prev = zero + 1.0;
    var h_curr = (zero + 2.0) * x;
    for (var k = 2u; k <= n; k = k + 1u) {
        let h_next = (zero + 2.0) * x * h_curr - (zero + 2.0) * f64(k - 1u) * h_prev;
        h_prev = h_curr;
        h_curr = h_next;
    }
    return h_curr;
}

fn laguerre_assoc(n: u32, alpha: u32, x: f64) -> f64 {
    let zero = x - x;
    let one = zero + 1.0;
    if (n == 0u) { return one; }
    let a = f64(alpha);
    var l_prev = one;
    var l_curr = one + a - x;
    for (var k = 2u; k <= n; k = k + 1u) {
        let kf = f64(k);
        let l_next = ((zero + 2.0) * kf - one + a - x) * l_curr - (kf - one + a) * l_prev;
        l_prev = l_curr;
        l_curr = l_next / kf;
    }
    return l_curr;
}

@compute @workgroup_size(256)
fn evaluate_basis(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let nrho = params.n_grid_rho;
    let nz = params.n_grid_z;
    let ns = params.n_states;
    let total = ns * nrho * nz;
    if (idx >= total) { return; }

    let state = idx / (nrho * nz);
    let grid_idx = idx % (nrho * nz);
    let ir = grid_idx / nz;
    let iz = grid_idx % nz;

    let zero = params.b_osc - params.b_osc;

    let n_rho = state_params[state * 3u];
    let n_z = state_params[state * 3u + 1u];
    let lambda = state_params[state * 3u + 2u];

    let b = params.b_osc;
    let rho = rho_grid[ir];
    let z = z_grid[iz];

    // Radial part: ρ^|Λ| · L_{n_ρ}^{|Λ|}(ρ²/b²) · exp(-ρ²/(2b²))
    let rho_b = rho / b;
    let rho_b2 = rho_b * rho_b;
    var radial = exp(-(zero + 0.5) * rho_b2) * laguerre_assoc(n_rho, lambda, rho_b2);
    for (var l = 0u; l < lambda; l = l + 1u) {
        radial = radial * rho_b;
    }

    // Axial part: H_{n_z}(z/b) · exp(-z²/(2b²))
    let z_b = z / b;
    let axial = exp(-(zero + 0.5) * z_b * z_b) * hermite(n_z, z_b);

    wavefunctions[idx] = radial * axial;
}

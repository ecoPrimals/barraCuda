// batched_hfb_energy_f64.wgsl — HFB total energy functional for spherical nuclei
//
// hotSpring absorption: nuclear physics GPU-resident SCF.
//
// Computes per-grid-point energy density integrands for the Skyrme HFB
// energy functional, then reduces to a scalar total energy per batch.
//
// Energy components:
//   E_t0    = t0/4 [(2+x0)ρ² - (2x0+1)(ρ_p²+ρ_n²)]
//   E_t3    = t3/24 ρ^α [(2+x3)ρ² - (2x3+1)(ρ_p²+ρ_n²)]
//   E_coul  = (e²/2) ∫ ρ_p(r) φ(r) r² dr        (direct)
//           - (3e²/4)(3/π)^(1/3) ∫ ρ_p^(4/3) r² dr (exchange, Slater)
//   E_pair  = -Δ²/G (pairing energy)
//   E_cm    = -<P²>/(2mA) (center of mass correction)
//
// Two entry points:
//   1. energy_integrands — Compute per-point energy density contributions
//   2. reduce_energy     — Shared-memory tree reduction to scalar
//
// Bindings:
//   0: params    uniform { n_grid, n_batch, _pad[2], alpha, t0..t3, x0..x3, e2, pair_G, pair_delta, A }
//   1: rho_p     [n_batch × n_grid] f64
//   2: rho_n     [n_batch × n_grid] f64
//   3: r_grid    [n_grid] f64
//   4: phi_coul  [n_batch × n_grid] f64 — Coulomb potential from charge integration
//   5: integrands [n_batch × n_grid] f64 — per-point energy density × r² × dr (output)
//   6: energies  [n_batch] f64 — total energy per nucleus (output from reduce)

// f64 enabled by compile_shader_f64() preamble injection

struct EnergyParams {
    n_grid:     u32,
    n_batch:    u32,
    _pad0:      u32,
    _pad1:      u32,
    alpha:      f64,
    t0:         f64,
    t3:         f64,
    x0:         f64,
    x3:         f64,
    e2:         f64,
    pair_G:     f64,
    pair_delta: f64,
    mass_A:     f64,
}

@group(0) @binding(0) var<uniform>             params: EnergyParams;
@group(0) @binding(1) var<storage, read>       rho_p: array<f64>;
@group(0) @binding(2) var<storage, read>       rho_n: array<f64>;
@group(0) @binding(3) var<storage, read>       r_grid: array<f64>;
@group(0) @binding(4) var<storage, read>       phi_coul: array<f64>;
@group(0) @binding(5) var<storage, read_write> integrands: array<f64>;
@group(0) @binding(6) var<storage, read_write> energies: array<f64>;

var<workgroup> shared_sum: array<f64, 256>;

@compute @workgroup_size(256)
fn energy_integrands(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let ng = params.n_grid;
    let total = params.n_batch * ng;
    if (idx >= total) { return; }

    let batch = idx / ng;
    let ir = idx % ng;

    let zero = r_grid[0] - r_grid[0];
    let one = zero + 1.0;
    let half = zero + 0.5;
    let pi = zero + 3.141592653589793;
    let four_pi = zero + 12.566370614359172;

    let rp = rho_p[batch * ng + ir];
    let rn = rho_n[batch * ng + ir];
    let rho = rp + rn;
    let r = r_grid[ir];
    let r2 = r * r;

    // dr for trapezoidal rule
    var dr = zero;
    if (ir == 0u) {
        dr = r_grid[1] - r_grid[0];
    } else if (ir == ng - 1u) {
        dr = r_grid[ng - 1u] - r_grid[ng - 2u];
    } else {
        dr = (r_grid[ir + 1u] - r_grid[ir - 1u]) * half;
    }

    let rho2 = rho * rho;
    let rho_q2 = rp * rp + rn * rn;

    // E_t0
    let e_t0 = params.t0 / (zero + 4.0) *
               (((zero + 2.0) + params.x0) * rho2 -
                ((zero + 2.0) * params.x0 + one) * rho_q2);

    // E_t3
    let rho_alpha = pow(max(rho, zero + 1e-30), params.alpha);
    let e_t3 = params.t3 / (zero + 24.0) * rho_alpha *
               (((zero + 2.0) + params.x3) * rho2 -
                ((zero + 2.0) * params.x3 + one) * rho_q2);

    // E_Coulomb direct
    let e_coul_dir = half * params.e2 * rp * phi_coul[batch * ng + ir];

    // E_Coulomb exchange (Slater approximation)
    let three_over_pi = (zero + 3.0) / pi;
    let rp_43 = pow(max(rp, zero + 1e-30), zero + 1.333333333333333);
    let e_coul_ex = -(zero + 0.75) * params.e2 * pow(three_over_pi, zero + 0.333333333333333) * rp_43;

    let e_density = (e_t0 + e_t3 + e_coul_dir + e_coul_ex) * four_pi * r2;

    integrands[idx] = e_density * dr;
}

@compute @workgroup_size(256)
fn reduce_energy(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>
) {
    let tid = lid.x;
    let batch = wgid.x;
    let ng = params.n_grid;

    if (batch >= params.n_batch) { return; }

    let base = batch * ng;
    let zero = r_grid[0] - r_grid[0];

    // Grid-stride accumulation
    var acc = zero;
    var i = tid;
    while (i < ng) {
        acc = acc + integrands[base + i];
        i = i + 256u;
    }

    shared_sum[tid] = acc;
    workgroupBarrier();

    // Tree reduction
    var stride = 128u;
    while (stride > 0u) {
        if (tid < stride) {
            shared_sum[tid] = shared_sum[tid] + shared_sum[tid + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    if (tid == 0u) {
        // Add pairing energy: E_pair = -Δ²/G
        let e_pair = -params.pair_delta * params.pair_delta / max(params.pair_G, zero + 1e-30);
        energies[batch] = shared_sum[0] + e_pair;
    }
}

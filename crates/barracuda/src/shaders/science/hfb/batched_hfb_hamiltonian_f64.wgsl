// batched_hfb_hamiltonian_f64.wgsl — Build single-particle Hamiltonian for spherical HFB
//
// hotSpring absorption: nuclear physics GPU-resident SCF.
//
// Constructs the HFB Hamiltonian matrix H[i,j] for each batch (nucleus).
// H = T_eff + V, where T_eff includes kinetic energy and effective mass,
// and V is the self-consistent Skyrme + Coulomb potential.
//
// The Hamiltonian is block-diagonal in (l, j) quantum numbers:
//   H_{ab} = ∫ φ_a(r) [-ℏ²/(2m*) d²/dr² + V_eff(r)] φ_b(r) r² dr
//
// Uses trapezoidal radial integration with the radial wavefunctions.
//
// Bindings:
//   0: dims      uniform { n_grid, n_basis, n_batch, _pad }
//   1: wf_batch  [n_batch × n_basis × n_grid] f64 — radial wavefunctions φ_a(r)
//   2: dwf_batch [n_batch × n_basis × n_grid] f64 — dφ_a/dr
//   3: u_total   [n_batch × n_grid] f64 — total potential V(r) for this (l,j) block
//   4: f_q       [n_batch × n_grid] f64 — effective mass form factor
//   5: r_grid    [n_grid] f64 — radial grid points
//   6: H_batch   [n_batch × n_basis × n_basis] f64 — output Hamiltonian matrix

// f64 enabled by compile_shader_f64() preamble injection

struct HamDims {
    n_grid:  u32,
    n_basis: u32,
    n_batch: u32,
    _pad:    u32,
}

@group(0) @binding(0) var<uniform>             dims: HamDims;
@group(0) @binding(1) var<storage, read>       wf_batch: array<f64>;
@group(0) @binding(2) var<storage, read>       dwf_batch: array<f64>;
@group(0) @binding(3) var<storage, read>       u_total: array<f64>;
@group(0) @binding(4) var<storage, read>       f_q: array<f64>;
@group(0) @binding(5) var<storage, read>       r_grid: array<f64>;
@group(0) @binding(6) var<storage, read_write> H_batch: array<f64>;

@compute @workgroup_size(16, 16, 1)
fn build_hamiltonian(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(workgroup_id) wgid: vec3<u32>
) {
    let a = gid.x;
    let b = gid.y;
    let batch = wgid.z;

    let nb = dims.n_basis;
    let ng = dims.n_grid;

    if (a >= nb || b >= nb || batch >= dims.n_batch) { return; }
    // Upper triangle only (symmetric matrix)
    if (b < a) { return; }

    let zero = r_grid[0] - r_grid[0];
    let half = zero + 0.5;

    let wf_off = batch * nb * ng;
    let pot_off = batch * ng;

    // Trapezoidal integration: H_ab = ∫ [f_q·dφ_a/dr·dφ_b/dr + V·φ_a·φ_b] r² dr
    var integral = zero;
    for (var ir = 0u; ir < ng; ir = ir + 1u) {
        let r = r_grid[ir];
        let r2 = r * r;
        let phi_a = wf_batch[wf_off + a * ng + ir];
        let phi_b = wf_batch[wf_off + b * ng + ir];
        let dphi_a = dwf_batch[wf_off + a * ng + ir];
        let dphi_b = dwf_batch[wf_off + b * ng + ir];
        let v = u_total[pot_off + ir];
        let fq = f_q[pot_off + ir];

        // Kinetic: (ℏ²/2m) · f_q · dφ_a/dr · dφ_b/dr · r²
        // Potential: V · φ_a · φ_b · r²
        let integrand = fq * dphi_a * dphi_b * r2 + v * phi_a * phi_b * r2;

        // Trapezoidal weight
        var dr = zero;
        if (ir == 0u) {
            dr = r_grid[1] - r_grid[0];
        } else if (ir == ng - 1u) {
            dr = r_grid[ng - 1u] - r_grid[ng - 2u];
        } else {
            dr = (r_grid[ir + 1u] - r_grid[ir - 1u]) * half;
        }
        integral = integral + integrand * dr;
    }

    let h_off = batch * nb * nb;
    H_batch[h_off + a * nb + b] = integral;
    if (a != b) {
        H_batch[h_off + b * nb + a] = integral;
    }
}

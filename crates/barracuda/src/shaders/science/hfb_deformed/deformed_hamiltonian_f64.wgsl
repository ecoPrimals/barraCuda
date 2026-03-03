// deformed_hamiltonian_f64.wgsl — Axially-deformed HFB Hamiltonian matrix
//
// hotSpring absorption: builds single-particle Hamiltonian matrix elements
// H_{ab} = ∫∫ ψ_a(ρ,z) [-ℏ²/(2m*) ∇² + V(ρ,z)] ψ_b(ρ,z) ρ dρ dz
//
// Uses pre-evaluated wavefunctions and potentials on a (ρ,z) grid.
// The ∇² (Laplacian in cylindrical coords with azimuthal m) is applied
// via 5-point finite differences on the 2D grid.
//
// Thread assignment: one thread per (row, col) matrix element.
//
// Bindings:
//   0: params         uniform { n_grid_rho, n_grid_z, n_states, _pad,
//                                d_rho, d_z, hbar2_2m: f64 }
//   1: wavefunctions  [n_states × n_grid] f64
//   2: potential      [n_grid] f64 — V(ρ,z) for this isospin
//   3: rho_grid       [n_grid_rho] f64
//   4: hamiltonian    [n_states × n_states] f64 — output

// f64 enabled by compile_shader_f64() preamble injection

struct HamiltonianParams {
    n_grid_rho: u32,
    n_grid_z:   u32,
    n_states:   u32,
    _pad:       u32,
    d_rho:      f64,
    d_z:        f64,
    hbar2_2m:   f64,
}

@group(0) @binding(0) var<uniform>             params: HamiltonianParams;
@group(0) @binding(1) var<storage, read>       wavefunctions: array<f64>;
@group(0) @binding(2) var<storage, read>       potential: array<f64>;
@group(0) @binding(3) var<storage, read>       rho_grid: array<f64>;
@group(0) @binding(4) var<storage, read_write> hamiltonian: array<f64>;

@compute @workgroup_size(16, 16)
fn build_hamiltonian(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ns = params.n_states;
    let a = gid.x;
    let b = gid.y;
    if (a >= ns || b >= ns) { return; }

    let nrho = params.n_grid_rho;
    let nz = params.n_grid_z;
    let n_grid = nrho * nz;
    let dr = params.d_rho;
    let dz = params.d_z;
    let h2m = params.hbar2_2m;
    let zero = dr - dr;
    let two = zero + 2.0;

    var sum = zero;
    let inv_dr2 = (zero + 1.0) / (dr * dr);
    let inv_dz2 = (zero + 1.0) / (dz * dz);

    for (var ir = 0u; ir < nrho; ir = ir + 1u) {
        let rho_val = rho_grid[ir];
        if (rho_val == zero) { continue; }
        let inv_rho = (zero + 1.0) / rho_val;

        for (var iz = 0u; iz < nz; iz = iz + 1u) {
            let gidx = ir * nz + iz;
            let psi_a = wavefunctions[a * n_grid + gidx];
            let psi_b = wavefunctions[b * n_grid + gidx];

            // 5-point Laplacian in cylindrical coordinates (axial symmetry)
            // ∇² ψ = d²ψ/dρ² + (1/ρ)dψ/dρ + d²ψ/dz²
            var lap_b = zero;

            // d²ψ/dρ²
            var psi_rm1 = zero;
            if (ir > 0u) { psi_rm1 = wavefunctions[b * n_grid + (ir - 1u) * nz + iz]; }
            var psi_rp1 = zero;
            if (ir < nrho - 1u) { psi_rp1 = wavefunctions[b * n_grid + (ir + 1u) * nz + iz]; }
            lap_b = lap_b + (psi_rp1 - two * psi_b + psi_rm1) * inv_dr2;

            // (1/ρ) dψ/dρ
            lap_b = lap_b + inv_rho * (psi_rp1 - psi_rm1) / (two * dr);

            // d²ψ/dz²
            var psi_zm1 = zero;
            if (iz > 0u) { psi_zm1 = wavefunctions[b * n_grid + ir * nz + iz - 1u]; }
            var psi_zp1 = zero;
            if (iz < nz - 1u) { psi_zp1 = wavefunctions[b * n_grid + ir * nz + iz + 1u]; }
            lap_b = lap_b + (psi_zp1 - two * psi_b + psi_zm1) * inv_dz2;

            // H_{ab} contribution: ψ_a · [-ℏ²/(2m*) ∇²ψ_b + V·ψ_b] · ρ·dρ·dz
            let integrand = psi_a * (-h2m * lap_b + potential[gidx] * psi_b);
            sum = sum + integrand * rho_val * dr * dz;
        }
    }

    hamiltonian[a * ns + b] = sum;
}

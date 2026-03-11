// SPDX-License-Identifier: AGPL-3.0-only
// 2D Finite-Difference Gradient Kernels for Cylindrical Grid (f64)
//
// Computes kinetic density τ, spin-current J, density gradient ∇ρ, and
// radial derivative dρ/dr on a (ρ, z) cylindrical mesh.
//
// Grid: row-major [n_rho × n_z], ρ starts at d_rho (not 0).
//   index(i_rho, i_z) = i_rho * n_z + i_z
//
// All kernels dispatch one thread per grid point.
// Dispatch: (ceil(n_grid / 256), 1, 1) unless noted.
//
// Deep Debt: pure WGSL, f64, self-contained, no recursion.

// ═══════════════════════════════════════════════════════════════════
// Parameters
// ═══════════════════════════════════════════════════════════════════

struct GridParams {
    n_rho: u32,
    n_z: u32,
    n_states: u32,
    _pad0: u32,
    d_rho: f64,
    d_z: f64,
    z_min: f64,        // z grid minimum (z[i] = z_min + (i+0.5)*d_z)
}

@group(0) @binding(0) var<uniform> params: GridParams;
// wavefunctions: [n_states × n_grid] f64 (row-major per state)
@group(0) @binding(1) var<storage, read> wavefunctions: array<f64>;
// occupations: [n_states] f64 — BCS occupation v²_i (NOT doubled yet)
@group(0) @binding(2) var<storage, read> occupations: array<f64>;

// Output buffers (one per kernel that writes):
// tau:        [n_grid] f64 — kinetic density
// j_density:  [n_grid] f64 — spin-current density
@group(0) @binding(3) var<storage, read_write> tau_out: array<f64>;
@group(0) @binding(4) var<storage, read_write> j_density_out: array<f64>;

// For spin-current: need lambda and sigma per state
// Packed as i32 pairs in a storage buffer
struct SpinParams {
    lambda: i32,      // orbital angular momentum projection
    sigma: i32,       // spin projection (+1 or -1)
}
@group(0) @binding(5) var<storage, read> spin_params: array<SpinParams>;

const PI: f64 = 3.14159265358979323846;

// ═══════════════════════════════════════════════════════════════════
// Kernel 1: Kinetic density  τ(r) = Σ_i  2·n_i · |∇ψ_i|²
//
// ∇ψ on (ρ, z) grid via central finite differences:
//   dψ/dρ = (ψ[i+1,j] - ψ[i-1,j]) / (2·dρ)     (central, forward/backward at edges)
//   dψ/dz = (ψ[i,j+1] - ψ[i,j-1]) / (2·dz)
//
// |∇ψ|² = (dψ/dρ)² + (dψ/dz)²
//
// Accumulates over all states: τ[grid] = Σ_state 2·occ[state] · |∇ψ[state]|²
//
// Strategy: loop over states inside kernel (state count is small ~220).
// Dispatch: (ceil(n_grid / 256), 1, 1)
// ═══════════════════════════════════════════════════════════════════

@compute @workgroup_size(256)
fn compute_tau(@builtin(global_invocation_id) gid: vec3<u32>) {
    let grid_idx = gid.x;
    let n_grid = params.n_rho * params.n_z;

    if (grid_idx >= n_grid) { return; }

    let i_rho = grid_idx / params.n_z;
    let i_z = grid_idx % params.n_z;

    var tau_val = f64(0.0);

    for (var si = 0u; si < params.n_states; si++) {
        let occ_i = occupations[si] * f64(2.0);  // time-reversal degeneracy
        if (occ_i < f64(1e-15)) { continue; }

        let base = si * n_grid;

        // ── dψ/dρ (finite difference) ──
        var dpsi_drho: f64;
        if (i_rho == 0u) {
            dpsi_drho = (wavefunctions[base + 1u * params.n_z + i_z]
                       - wavefunctions[base + grid_idx]) / params.d_rho;
        } else if (i_rho == params.n_rho - 1u) {
            dpsi_drho = (wavefunctions[base + grid_idx]
                       - wavefunctions[base + (i_rho - 1u) * params.n_z + i_z]) / params.d_rho;
        } else {
            dpsi_drho = (wavefunctions[base + (i_rho + 1u) * params.n_z + i_z]
                       - wavefunctions[base + (i_rho - 1u) * params.n_z + i_z])
                       / (f64(2.0) * params.d_rho);
        }

        // ── dψ/dz (finite difference) ──
        var dpsi_dz: f64;
        if (i_z == 0u) {
            dpsi_dz = (wavefunctions[base + i_rho * params.n_z + 1u]
                     - wavefunctions[base + grid_idx]) / params.d_z;
        } else if (i_z == params.n_z - 1u) {
            dpsi_dz = (wavefunctions[base + grid_idx]
                     - wavefunctions[base + i_rho * params.n_z + i_z - 1u]) / params.d_z;
        } else {
            dpsi_dz = (wavefunctions[base + i_rho * params.n_z + i_z + 1u]
                     - wavefunctions[base + i_rho * params.n_z + i_z - 1u])
                     / (f64(2.0) * params.d_z);
        }

        tau_val += occ_i * (dpsi_drho * dpsi_drho + dpsi_dz * dpsi_dz);
    }

    tau_out[grid_idx] = tau_val;
}

// ═══════════════════════════════════════════════════════════════════
// Kernel 2: Spin-current density  J(r) = Σ_i  2·n_i · Λ_i·σ_i/2 · |ψ_i|²
//
// For axial symmetry, only the z-component matters: <l·s> = Λ·σ/2
// Dispatch: (ceil(n_grid / 256), 1, 1)
// ═══════════════════════════════════════════════════════════════════

@compute @workgroup_size(256)
fn compute_spin_current(@builtin(global_invocation_id) gid: vec3<u32>) {
    let grid_idx = gid.x;
    let n_grid = params.n_rho * params.n_z;

    if (grid_idx >= n_grid) { return; }

    var j_val = f64(0.0);

    for (var si = 0u; si < params.n_states; si++) {
        let occ_i = occupations[si] * f64(2.0);
        if (occ_i < f64(1e-15)) { continue; }

        let sp = spin_params[si];
        let ls = f64(sp.lambda) * f64(sp.sigma) * f64(0.5);

        let psi = wavefunctions[si * n_grid + grid_idx];
        j_val += occ_i * ls * psi * psi;
    }

    j_density_out[grid_idx] = j_val;
}

// ═══════════════════════════════════════════════════════════════════
// Kernel 3: Density radial derivative  dρ/dr
//
// Projects the 2D gradient (dρ/dρ_coord, dρ/dz) onto radial direction:
//   dρ/dr = (dρ/dρ_coord · ρ_coord + dρ/dz · z) / r
// where r = sqrt(ρ² + z²)
//
// Input: density[n_grid] from binding 3 (reuse tau_out buffer)
// Output: deriv[n_grid] to binding 4 (reuse j_density_out buffer)
//
// NOTE: caller must set up bindings differently for this kernel.
//       binding(3) = density input, binding(4) = derivative output
// Dispatch: (ceil(n_grid / 256), 1, 1)
// ═══════════════════════════════════════════════════════════════════

@compute @workgroup_size(256)
fn density_radial_derivative(@builtin(global_invocation_id) gid: vec3<u32>) {
    let grid_idx = gid.x;
    let n_grid = params.n_rho * params.n_z;

    if (grid_idx >= n_grid) { return; }

    let i_rho = grid_idx / params.n_z;
    let i_z = grid_idx % params.n_z;

    // We're reading from tau_out as the density input for this kernel
    // (caller rebinds the buffers appropriately)

    // dρ/dρ_coord
    var d_drho: f64;
    if (i_rho == 0u) {
        d_drho = (tau_out[1u * params.n_z + i_z] - tau_out[grid_idx]) / params.d_rho;
    } else if (i_rho == params.n_rho - 1u) {
        d_drho = (tau_out[grid_idx] - tau_out[(i_rho - 1u) * params.n_z + i_z]) / params.d_rho;
    } else {
        d_drho = (tau_out[(i_rho + 1u) * params.n_z + i_z]
                - tau_out[(i_rho - 1u) * params.n_z + i_z])
                / (f64(2.0) * params.d_rho);
    }

    // dρ/dz
    var d_dz: f64;
    if (i_z == 0u) {
        d_dz = (tau_out[i_rho * params.n_z + 1u] - tau_out[grid_idx]) / params.d_z;
    } else if (i_z == params.n_z - 1u) {
        d_dz = (tau_out[grid_idx] - tau_out[i_rho * params.n_z + i_z - 1u]) / params.d_z;
    } else {
        d_dz = (tau_out[i_rho * params.n_z + i_z + 1u]
              - tau_out[i_rho * params.n_z + i_z - 1u])
              / (f64(2.0) * params.d_z);
    }

    // Project onto radial direction
    let rho_coord = f64(i_rho + 1u) * params.d_rho;
    let z_coord = params.z_min + (f64(i_z) + f64(0.5)) * params.d_z;
    let r = max(sqrt(rho_coord * rho_coord + z_coord * z_coord), f64(0.01));

    j_density_out[grid_idx] = (d_drho * rho_coord + d_dz * z_coord) / r;
}

// SPDX-License-Identifier: AGPL-3.0-or-later
// Spin-Orbit Coupling Kernel (f64) — Nuclear HFB Evolution
//
// Computes diagonal spin-orbit corrections to the HFB Hamiltonian.
// This is the missing piece that was computed on CPU in hotSpring.
//
// Evolution: Feb 16, 2026 — TIER 2.1 from hotSpring handoff
//
// Formula:
//   H_so[i,i] += w0 * ls_i * ∫ wf_i²(r) · (dρ/dr / r) · r² dr
//             = w0 * ls_i * ∫ wf_i²(r) · dρ/dr · r dr
//
// where:
//   ls_i = (j(j+1) - l(l+1) - 3/4) / 2  — spin-orbit factor for state i
//   wf_i(r) — radial wavefunction for state i
//   ρ(r) — total density at radius r
//   w0 — Skyrme spin-orbit coupling strength (MeV·fm⁵)
//
// The kernel is batched across:
// - nuclei (different density profiles)
// - states (different l, j quantum numbers)
//
// Inputs:
//   wf_squared: [batch × n_states × n_grid] — |ψ_i(r)|²
//   drho_dr: [batch × n_grid] — density gradient per nucleus
//   r_grid: [n_grid] — radial grid points
//   dr: grid spacing
//   ls_factors: [batch × n_states] — pre-computed ls_i values
//   w0: spin-orbit coupling constant
//
// Outputs:
//   h_so_diag: [batch × n_states] — diagonal H_so corrections

struct SpinOrbitParams {
    batch_size: u32,
    n_states: u32,
    n_grid: u32,
    _pad: u32,
    dr: f64,       // Grid spacing
    w0: f64,       // Spin-orbit coupling (MeV·fm⁵)
}

@group(0) @binding(0) var<uniform> params: SpinOrbitParams;
@group(0) @binding(1) var<storage, read> wf_squared: array<f64>;   // [batch × n_states × n_grid]
@group(0) @binding(2) var<storage, read> drho_dr: array<f64>;      // [batch × n_grid]
@group(0) @binding(3) var<storage, read> r_grid: array<f64>;       // [n_grid]
@group(0) @binding(4) var<storage, read> ls_factors: array<f64>;   // [batch × n_states]
@group(0) @binding(5) var<storage, read_write> h_so_diag: array<f64>; // [batch × n_states]

// Helper: index into wf_squared
fn wf_idx(batch: u32, state: u32, grid: u32, n_states: u32, n_grid: u32) -> u32 {
    return batch * n_states * n_grid + state * n_grid + grid;
}

// Main kernel: one thread per (batch, state) pair
// Dispatch: (batch_size × n_states / 64, 1, 1)
@compute @workgroup_size(64, 1, 1)
fn spin_orbit_diagonal(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let flat_idx = gid.x;
    let batch = flat_idx / params.n_states;
    let state = flat_idx % params.n_states;
    
    if (batch >= params.batch_size) {
        return;
    }
    
    let n_states = params.n_states;
    let n_grid = params.n_grid;
    let dr = params.dr;
    let w0 = params.w0;
    
    // Get spin-orbit factor ls_i for this state
    let ls_i = ls_factors[batch * n_states + state];
    
    // Integrate: ∫ wf²(r) · (dρ/dr) · r · dr
    // Using trapezoidal rule (matching CPU implementation)
    var integral = f64(0.0);
    let drho_base = batch * n_grid;
    
    for (var k = 0u; k < n_grid; k = k + 1u) {
        let r = r_grid[k];
        let wf2 = wf_squared[wf_idx(batch, state, k, n_states, n_grid)];
        let drho = drho_dr[drho_base + k];
        
        // Integrand: wf²(r) · (dρ/dr) · r
        let integrand = wf2 * drho * r;
        
        // Trapezoidal weighting
        var weight = dr;
        if (k == 0u || k == n_grid - 1u) {
            weight = f64(0.5) * dr;
        }
        
        integral = integral + integrand * weight;
    }
    
    // H_so[i,i] = w0 · ls_i · integral
    h_so_diag[batch * n_states + state] = w0 * ls_i * integral;
}

// Alternative: compute gradient and spin-orbit in one kernel
// This version takes density and computes gradient internally
// More efficient when density is already on GPU

@group(0) @binding(6) var<storage, read> density: array<f64>;  // [batch × n_grid] — optional

@compute @workgroup_size(64, 1, 1)
fn spin_orbit_with_gradient(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let flat_idx = gid.x;
    let batch = flat_idx / params.n_states;
    let state = flat_idx % params.n_states;
    
    if (batch >= params.batch_size) {
        return;
    }
    
    let n_states = params.n_states;
    let n_grid = params.n_grid;
    let dr = params.dr;
    let w0 = params.w0;
    
    let ls_i = ls_factors[batch * n_states + state];
    let rho_base = batch * n_grid;
    
    // Integrate with on-the-fly gradient computation
    var integral = f64(0.0);
    
    for (var k = 0u; k < n_grid; k = k + 1u) {
        let r = r_grid[k];
        let wf2 = wf_squared[wf_idx(batch, state, k, n_states, n_grid)];
        
        // Compute dρ/dr using central differences
        var drho: f64;
        if (k == 0u) {
            // Forward difference at boundary
            drho = (density[rho_base + 1u] - density[rho_base]) / dr;
        } else if (k == n_grid - 1u) {
            // Backward difference at boundary
            drho = (density[rho_base + k] - density[rho_base + k - 1u]) / dr;
        } else {
            // Central difference
            drho = (density[rho_base + k + 1u] - density[rho_base + k - 1u]) / (2.0 * dr);
        }
        
        let integrand = wf2 * drho * r;
        
        var weight = dr;
        if (k == 0u || k == n_grid - 1u) {
            weight = f64(0.5) * dr;
        }
        
        integral = integral + integrand * weight;
    }
    
    h_so_diag[batch * n_states + state] = w0 * ls_i * integral;
}
